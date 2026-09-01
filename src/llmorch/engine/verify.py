"""Automatic evaluation of generated artifacts.

Two tiers, cheapest first.

**Tier 0** is deterministic and costs nothing: does the Python parse, does the
SQL parse, did generation stop mid-sentence, is the file a placeholder rather
than an implementation. Truncation and syntax errors are the most common
free-model failures, and paying a model to notice that code does not parse
would be pure waste.

**Tier 1** is a cross-vendor LLM review, wired at Milestone 4 when two live
vendors exist. Its defining rule is enforced here rather than requested in a
prompt: *the reviewer must come from a different vendor than the author*.
Self-review is a well-known weak spot — a model that made a mistake tends to
re-approve it — and a peer from another vendor has decorrelated blind spots.
"""

from __future__ import annotations

import ast
import re
import sqlite3

from ..registry.manifest import Manifest
from ..types import Issue, OutputKind, VerifyResult, Verdict
from .salvage import extract_code, looks_truncated

# Markers that indicate a stub rather than an implementation.
#
# A comment prefix is required, so prose like "a todo list app" cannot trigger
# it — that phrase is a plausible thing for a notes-app spec to contain.
#
# Deliberate tradeoff: real hand-written code carries legitimate TODOs, so this
# would be too strict for a general linter. Here the artifact is generated fresh
# from a spec that said "produce this", and a TODO means the model described the
# work instead of doing it. A false positive costs one failover to another
# vendor; a false negative ships a stub as if it were an implementation.
_PLACEHOLDER = re.compile(
    r"(?:^|\n)[ \t]*(?:#|//|/\*|<!--|--)[ \t]*"
    r"(TODO|FIXME|XXX|IMPLEMENT ME|YOUR CODE HERE|PLACEHOLDER)\b",
    re.IGNORECASE,
)

# A bare `...` on its own line is Python's idiomatic "body omitted".
_ELLIPSIS_STUB = re.compile(r"(?:^|\n)[ \t]*\.\.\.[ \t]*(?=\n|$)")

_LANG_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
}


def _lang_for(output_path: str) -> str | None:
    for suffix, lang in _LANG_BY_SUFFIX.items():
        if output_path.lower().endswith(suffix):
            return lang
    return None


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def check_python(code: str) -> list[Issue]:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return [
            Issue(
                severity="error",
                what=f"Python does not parse: {exc.msg}",
                why="the file cannot run at all",
                line=exc.lineno,
            )
        ]
    return []


def check_sql(sql: str) -> list[Issue]:
    """Validate DDL by executing it against a throwaway in-memory database.

    Stricter than a parse check and still free — it catches unknown types and
    malformed constraints that a parser alone would accept.
    """
    statements = [s for s in sql.split(";") if s.strip()]
    if not statements:
        return [Issue(severity="error", what="SQL file is empty")]

    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(sql)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return [
            Issue(
                severity="error",
                what=f"SQL is invalid: {exc}",
                why="the schema cannot be created",
            )
        ]
    return []


def check_html(html: str) -> list[Issue]:
    """Lightweight structural check.

    Not a full parser: browsers recover from most malformed HTML, so only
    failures that indicate the model stopped early are worth flagging.
    """
    issues: list[Issue] = []
    lowered = html.lower()

    for tag in ("html", "body"):
        opens = lowered.count(f"<{tag}")
        closes = lowered.count(f"</{tag}>")
        if opens and opens != closes:
            issues.append(
                Issue(
                    severity="error",
                    what=f"<{tag}> opened {opens}x but closed {closes}x",
                    why="usually means generation stopped early",
                )
            )

    if "<script" in lowered and lowered.count("<script") != lowered.count("</script>"):
        issues.append(
            Issue(severity="error", what="unbalanced <script> tags")
        )
    return issues


def check_css(css: str) -> list[Issue]:
    if css.count("{") != css.count("}"):
        return [
            Issue(
                severity="error",
                what=f"unbalanced braces: {css.count('{')} open, {css.count('}')} close",
                why="usually means generation stopped early",
            )
        ]

    # A stylesheet with no rule block at all is not a stylesheet. Catches the
    # case where a model returns entirely the wrong kind of content — which
    # bracket-balance alone happily accepts, since prose and Python both have
    # zero braces and zero is balanced.
    body = "\n".join(
        line for line in css.splitlines() if not line.strip().startswith(("/*", "*", "//"))
    )
    if body.strip() and "{" not in body:
        return [
            Issue(
                severity="error",
                what="no CSS rule blocks found",
                why="the content does not look like a stylesheet",
            )
        ]
    return []


def check_javascript(js: str) -> list[Issue]:
    """Bracket balance only — a real JS parser is not worth the dependency, and
    imbalance catches the truncation case this is really guarding against."""
    issues: list[Issue] = []
    for opener, closer, label in (("{", "}", "braces"), ("(", ")", "parens")):
        if js.count(opener) != js.count(closer):
            issues.append(
                Issue(
                    severity="error",
                    what=f"unbalanced {label}",
                    why="usually means generation stopped early",
                )
            )
    return issues


def check_placeholder(text: str) -> list[Issue]:
    match = _PLACEHOLDER.search(text)
    if match:
        return [
            Issue(
                severity="error",
                what=f"contains a placeholder marker ({match.group(1).upper()})",
                why="the model described the work instead of doing it",
            )
        ]
    if _ELLIPSIS_STUB.search(text):
        return [
            Issue(
                severity="error",
                what="contains a bare `...` body",
                why="the implementation was elided",
            )
        ]
    return []


# --------------------------------------------------------------------------
# Tier 0
# --------------------------------------------------------------------------


def verify_tier0(
    artifact: str,
    *,
    output_path: str,
    output_kind: OutputKind,
    truncated_flag: bool = False,
) -> VerifyResult:
    """Deterministic checks. Zero requests, runs on every node."""
    issues: list[Issue] = []

    if not artifact or not artifact.strip():
        return VerifyResult(
            verdict=Verdict.REJECT,
            tier=0,
            issues=(Issue(severity="error", what="artifact is empty"),),
        )

    # Authoritative truncation signal: generation hit max_tokens.
    if truncated_flag:
        issues.append(
            Issue(
                severity="error",
                what="generation stopped at max_tokens",
                why="the artifact is incomplete",
            )
        )
    elif looks_truncated(artifact):
        issues.append(
            Issue(severity="error", what="artifact appears to end mid-thought")
        )

    lang = _lang_for(output_path)
    code = extract_code(artifact, prefer_lang=lang)

    match lang:
        case "python":
            issues += check_python(code)
        case "sql":
            issues += check_sql(code)
        case "html":
            issues += check_html(code)
        case "css":
            issues += check_css(code)
        case "javascript":
            issues += check_javascript(code)

    if output_kind in (OutputKind.CODE, OutputKind.SCHEMA):
        issues += check_placeholder(code)

    if any(i.severity == "error" for i in issues):
        return VerifyResult(verdict=Verdict.REJECT, tier=0, issues=tuple(issues))
    return VerifyResult(verdict=Verdict.PASS, tier=0, issues=tuple(issues))


# --------------------------------------------------------------------------
# Tier 1 reviewer selection
# --------------------------------------------------------------------------


def pick_reviewer(
    manifest: Manifest,
    *,
    author_model_id: str,
    candidates: list[str],
    prefer_provider: str | None = None,
) -> str | None:
    """Choose a reviewer from a different vendor than the author.

    Returns None when no cross-vendor reviewer exists, in which case review is
    skipped entirely. Falling back to same-vendor review would spend a request
    to obtain the one opinion least likely to find the fault — a model from the
    same family shares the author's blind spots, and self-review in particular
    tends to re-approve its own mistake.
    """
    author_vendor = manifest.vendor_of(author_model_id)
    cross = [
        m
        for m in candidates
        if m != author_model_id and manifest.vendor_of(m) != author_vendor
    ]
    if not cross:
        return None

    # Prefer the provider with abundant daily requests, so review never eats
    # the scarce budget the critical path depends on.
    preferred = prefer_provider or manifest.defaults.review_prefers_provider
    same_provider = [m for m in cross if manifest.vendor_of(m) == preferred]
    pool = same_provider or cross

    return max(pool, key=lambda m: manifest.model(m).quality_prior)


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        # Ordered first, and required, so the reviewer has to trace the file
        # before it is allowed to judge it. Measured on the one artifact known
        # to be broken: asked for a verdict alone the reviewer said "pass";
        # made to walk one path through the code first, it found the fault and
        # named the exact line. The trace is discarded — its value is that
        # producing it comes before the verdict, not after.
        "trace": {"type": "string"},
        "verdict": {"type": "string", "enum": ["pass", "revise", "reject"]},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["error", "warning", "info"]},
                    "what": {"type": "string"},
                    "why": {"type": "string"},
                    "line": {"type": ["integer", "null"]},
                },
                "required": ["severity", "what"],
            },
        },
    },
    "required": ["trace", "verdict", "issues"],
}


def parse_review(payload: dict, reviewer_model_id: str) -> VerifyResult:
    """Turn a reviewer's structured response into a verdict.

    Reviewer output is untrusted data. Only the recognised fields are read, and
    nothing here can influence routing, the manifest, or governor state.
    """
    raw = str(payload.get("verdict", "")).strip().lower()
    verdict = {
        "pass": Verdict.PASS,
        "revise": Verdict.REVISE,
        "reject": Verdict.REJECT,
    }.get(raw, Verdict.REVISE)

    issues: list[Issue] = []
    for item in payload.get("issues") or []:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "warning")).lower()
        if severity not in ("error", "warning", "info"):
            severity = "warning"
        line = item.get("line")
        issues.append(
            Issue(
                severity=severity,
                what=str(item.get("what", ""))[:500],
                why=str(item.get("why", ""))[:500],
                line=line if isinstance(line, int) else None,
            )
        )

    return VerifyResult(
        verdict=verdict,
        tier=1,
        issues=tuple(issues),
        reviewer_model_id=reviewer_model_id,
    )
