"""Recover usable content from imperfect model output.

Models wrap JSON in prose, fence code with language tags, and append commentary
after the payload — regardless of what the prompt asked for. Every one of those
is recoverable locally.

This matters more than it looks: a repair request costs quota, and against a
provider allowing 250 requests a day, salvage that works is worth more than any
amount of prompt engineering telling the model to stop doing it.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(
    r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)(?:\r?\n)?```",
    re.DOTALL,
)

# Language tags that indicate a fenced block holds code rather than data.
_CODE_LANGS = {
    "python", "py", "javascript", "js", "html", "css", "sql", "bash", "sh",
    "json", "yaml", "yml", "toml", "text", "plaintext", "",
}


# Reasoning some models emit *inline in the message content* rather than in a
# separate field. qwen3.6 on Groq does exactly this, and reports
# reasoning_tokens=0 while doing it, so the monologue is invisible to
# accounting and arrives as if it were the artifact.
_THINK = re.compile(
    r"<(think|thinking|reasoning)\b[^>]*>.*?</\1\s*>", re.DOTALL | re.IGNORECASE
)

# An unclosed opener means generation stopped mid-thought: everything from the
# tag onward is monologue, and there is no artifact after it to keep.
_THINK_UNCLOSED = re.compile(
    r"<(think|thinking|reasoning)\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE
)


def strip_reasoning(text: str) -> str:
    """Remove inline reasoning blocks.

    Runs before fence extraction and before verification, so a model that
    thinks out loud is judged on what it actually produced. Without this, every
    artifact such a model writes opens with its own deliberation — and a
    Tier 0 syntax check on `<think>` fails a file that was otherwise fine.
    """
    if not text or "<" not in text:
        return text
    cleaned = _THINK.sub("", text)
    cleaned = _THINK_UNCLOSED.sub("", cleaned)
    return cleaned.strip()


def strip_fences(text: str) -> str:
    """Return the contents of the first fenced block, or the text unchanged.

    Used before writing an artifact to disk: a file whose first line is
    ```python is not a working file.
    """
    if not text:
        return ""
    match = _FENCE.search(text)
    if match:
        return match.group(2).strip("\n")
    return text.strip()


def all_fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Every fenced block as (language, content)."""
    return [(m.group(1).lower(), m.group(2).strip("\n")) for m in _FENCE.finditer(text)]


def _balanced_span(text: str, opener: str, closer: str) -> tuple[int, int] | None:
    """Locate the first balanced bracket span, ignoring brackets inside strings.

    A naive search for the last closing brace breaks on trailing commentary and
    on nested structures; tracking string state and escapes handles both.
    """
    start = text.find(opener)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return start, i + 1
    return None


def extract_json(text: str) -> Any | None:
    """Best-effort JSON recovery. Returns None only when nothing parses.

    Tried in order of cost: the whole string, each fenced block, then the first
    balanced object or array found anywhere in the text.
    """
    if not text or not text.strip():
        return None

    candidates: list[str] = [text.strip()]
    candidates.extend(content for _, content in all_fenced_blocks(text))

    for opener, closer in (("{", "}"), ("[", "]")):
        span = _balanced_span(text, opener, closer)
        if span:
            candidates.append(text[span[0] : span[1]])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def extract_code(text: str, *, prefer_lang: str | None = None) -> str:
    """Pull code out of a response.

    When several fenced blocks are present, a language hint selects the right
    one — models often precede the real file with a short usage example.
    """
    blocks = all_fenced_blocks(text)
    if not blocks:
        return text.strip()

    if prefer_lang:
        wanted = prefer_lang.lower()
        for lang, content in blocks:
            if lang == wanted:
                return content

    code_blocks = [c for lang, c in blocks if lang in _CODE_LANGS]
    pool = code_blocks or [c for _, c in blocks]
    # The longest block is almost always the artifact rather than a snippet.
    return max(pool, key=len)


def looks_truncated(text: str) -> bool:
    """Heuristic check for output that stopped mid-thought.

    Complements the authoritative signal (generation stopping exactly at
    max_tokens) for cases where that is unavailable.
    """
    if not text:
        return True
    stripped = text.rstrip()
    if not stripped:
        return True

    # An unclosed code fence means the block never finished.
    if stripped.count("```") % 2 == 1:
        return True

    # Ends mid-token rather than on any sensible terminator.
    if stripped[-1] not in ".!?;:,)}]>\"'`\n" and not stripped[-1].isalnum():
        return True

    return False
