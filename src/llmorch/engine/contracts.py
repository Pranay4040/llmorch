"""Cross-artifact validation: do the pieces actually fit together?

Every check up to this point looks at one file at a time. Tier 0 asks whether
this file parses; Tier 1 asks whether this file does its job. Neither can see
the failure that a split build makes likely and a single author would never
make: the frontend calls `/api/note/{id}` and the backend serves
`/api/notes/{id}`, the page links a stylesheet nobody was asked to write, the
API returns a field the schema has no column for. Each file is impeccable. The
project is broken.

That gap is structural, not incidental. The whole design turns on models never
speaking to each other — coordination happens through the shared interface
contract instead — so the contract is the only thing that can be checked *for*
agreement, and something has to actually check it.

This costs nothing: no request, no model, no network. It is string and AST work
over artifacts already sitting in memory. Against a 250-request daily budget,
"free and deterministic" is the whole argument for doing as much as possible
here and as little as possible with a model.

What it deliberately does not do is decide the run's fate. By the time these
checks run the artifacts exist and were paid for; a mismatch is reported loudly
and the files stay on disk, because a half-matching project a person can fix in
two minutes beats an empty folder and a clean conscience.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..types import InterfaceContract

# href="..." / src="..." — single or double quoted.
_ASSET_REF = re.compile(r"""(?:href|src)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)

# fetch("/api/..."), axios.get('/api/...'), XHR .open('GET', '/api/...')
_CALL_REF = re.compile(r"""["'](/[A-Za-z0-9_\-./{}$+:]*)["']""")

# A path segment that is a placeholder rather than a literal: {id}, :id, ${id}.
_PLACEHOLDER = re.compile(r"^(\{.*\}|:.+|\$\{.*\}|<.+>)$")

_MARKUP = (".html", ".htm")
_SCRIPT = (".js", ".mjs", ".ts")
_SCHEMA = (".sql",)
_CODE = (".py", ".js", ".mjs", ".ts", ".rb", ".go")


@dataclass(frozen=True, slots=True)
class ContractIssue:
    severity: str  # "error" | "warning"
    what: str
    where: str = ""
    why: str = ""


@dataclass(slots=True)
class ContractReport:
    issues: list[ContractIssue] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    def add(self, severity: str, what: str, where: str = "", why: str = "") -> None:
        self.issues.append(ContractIssue(severity, what, where, why))

    @property
    def errors(self) -> list[ContractIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ContractIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def route_pattern(path: str) -> tuple[str, ...]:
    """Split a route into segments, with placeholders normalised.

    `/api/notes/{id}` and `/api/notes/42` both become `("api", "notes", "*")`,
    so a frontend that interpolates a real id still matches the declared route.
    """
    parts = [p for p in path.split("?")[0].split("/") if p]
    return tuple("*" if _PLACEHOLDER.match(p) else p for p in parts)


def _is_local_asset(ref: str) -> bool:
    """Whether a href/src points at a file this project is supposed to produce."""
    if not ref or ref.startswith(("#", "data:", "mailto:", "javascript:")):
        return False
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", ref) or ref.startswith("//"):
        return False
    return True


def _normalise_asset(ref: str) -> str:
    return ref.split("?")[0].split("#")[0].lstrip("./").lstrip("/")


def _declared_routes(interface: InterfaceContract) -> list[dict[str, Any]]:
    return [r for r in interface.routes if isinstance(r, dict) and r.get("path")]


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_pages_exist(
    interface: InterfaceContract, artifacts: dict[str, str], report: ContractReport
) -> None:
    """Every page the contract promises was actually written."""
    report.checks_run.append("pages exist")
    for page in interface.pages:
        if _normalise_asset(page) not in artifacts:
            report.add(
                "error",
                f"the contract lists page `{page}` but nothing produced it",
                why="a node was dropped, degraded, or wrote to a different path",
            )


def check_assets_resolve(artifacts: dict[str, str], report: ContractReport) -> None:
    """Every stylesheet and script a page links to exists in the output.

    The classic split-build failure: the page author confidently links
    `style.css` because the contract implies one, and the styling node degraded
    an hour ago.
    """
    report.checks_run.append("assets resolve")
    for path, text in artifacts.items():
        if not path.lower().endswith(_MARKUP):
            continue
        for ref in _ASSET_REF.findall(text):
            if not _is_local_asset(ref):
                continue
            target = _normalise_asset(ref)
            if target and target not in artifacts:
                report.add(
                    "error",
                    f"`{path}` references `{ref}`, which no node produced",
                    where=path,
                )


def check_frontend_calls_are_declared(
    interface: InterfaceContract, artifacts: dict[str, str], report: ContractReport
) -> None:
    """Every API path the frontend calls appears in the contract.

    This is the check that catches two models agreeing with the spec and not
    with each other — `/api/note/1` against `/api/notes/1`, singular versus
    plural, invisible in either file on its own.
    """
    report.checks_run.append("frontend calls are declared")
    declared = {route_pattern(r["path"]) for r in _declared_routes(interface)}
    if not declared:
        return

    # Only paths that look like API calls: a page linking `/about.html` is a
    # navigation target, not a contract violation.
    api_prefixes = {p[0] for p in declared if p}

    for path, text in artifacts.items():
        if not path.lower().endswith(_SCRIPT + _MARKUP):
            continue
        for ref in _CALL_REF.findall(text):
            pattern = route_pattern(ref)
            if not pattern or pattern[0] not in api_prefixes:
                continue
            if pattern not in declared:
                report.add(
                    "error",
                    f"`{path}` calls `{ref}`, which the contract does not declare",
                    where=path,
                    why="the backend was never asked to serve it",
                )


def check_routes_are_served(
    interface: InterfaceContract, artifacts: dict[str, str], report: ContractReport
) -> None:
    """Each declared route appears somewhere in the backend code.

    Deliberately shallow — it matches the literal path prefix rather than
    reasoning about the dispatch. A backend that mentions the route might still
    handle it wrongly; a backend that never mentions it certainly does not
    handle it at all, and that is worth knowing for free.
    """
    report.checks_run.append("routes are served")
    backend = {p: t for p, t in artifacts.items() if p.lower().endswith(".py")}
    if not backend:
        return
    blob = "\n".join(backend.values())
    where = ", ".join(backend)

    for route in _declared_routes(interface):
        segments = route_pattern(route["path"])
        method = route.get("method", "GET")
        literal = []
        for segment in segments:
            if segment == "*":
                break
            literal.append(segment)
        base = "/" + "/".join(literal)

        if base not in blob:
            report.add(
                "error",
                f"no backend file mentions `{route['path']}`",
                where=where,
                why=f"declared in the contract as {method}",
            )
            continue

        # A parameterised route shares its base with the collection route, so
        # the base alone proves nothing about it. Look for the base *plus* a
        # separator, which is how a path-prefix dispatch is written.
        if len(literal) < len(segments) and f"{base}/" not in blob:
            report.add(
                "warning",
                f"`{route['path']}` may not be handled: the backend mentions "
                f"`{base}` but never `{base}/`",
                where=where,
                why="a segment-splitting dispatch can be correct without the "
                "literal prefix, so this is a prompt to look rather than a verdict",
            )


def check_schema_covers_models(
    interface: InterfaceContract, artifacts: dict[str, str], report: ContractReport
) -> None:
    """Fields the API promises have somewhere to live in the schema.

    A warning rather than an error: a field can legitimately be computed rather
    than stored, and naming can differ between the wire and the table without
    anything being wrong.
    """
    report.checks_run.append("schema covers models")
    schema = "\n".join(
        t for p, t in artifacts.items() if p.lower().endswith(_SCHEMA)
    ).lower()
    if not schema:
        return

    for model in interface.data_models:
        if not isinstance(model, dict):
            continue
        for field_name in (model.get("fields") or {}):
            if not re.search(rf"\b{re.escape(str(field_name).lower())}\b", schema):
                report.add(
                    "warning",
                    f"`{model.get('name', '?')}.{field_name}` has no matching "
                    "column in the schema",
                    why="computed at read time, renamed, or genuinely missing",
                )


def check_contract(
    interface: InterfaceContract, artifacts: dict[str, str]
) -> ContractReport:
    """Run every cross-artifact check over a completed set of artifacts.

    `artifacts` maps output path to contents — the same shape materialize
    writes, so this can run before or after anything touches the filesystem.
    """
    report = ContractReport()
    check_pages_exist(interface, artifacts, report)
    check_assets_resolve(artifacts, report)
    check_frontend_calls_are_declared(interface, artifacts, report)
    check_routes_are_served(interface, artifacts, report)
    check_schema_covers_models(interface, artifacts, report)
    check_python_calls(artifacts, report)
    return report


def artifacts_from_results(nodes: dict, results: dict) -> dict[str, str]:
    """Build the path -> contents map from a run's outcome.

    Only nodes that actually produced something: a degraded node's stub would
    otherwise satisfy an asset reference that nothing really answers, turning a
    missing file into a silent pass.
    """
    from ..types import NodeState

    return {
        nodes[node_id].output_path: result.artifact
        for node_id, result in results.items()
        if node_id in nodes and result.state is NodeState.DONE and result.artifact
    }


# --------------------------------------------------------------------------
# Python call agreement
# --------------------------------------------------------------------------


def _module_functions(path: str, text: str) -> dict[str, dict[str, Any]]:
    """Module-level function signatures defined in one Python artifact."""
    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}

    found: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
        found[node.name] = {
            "where": path,
            "params": names,
            "positional": len(args.posonlyargs) + len(args.args),
            "has_varargs": args.vararg is not None,
            "has_kwargs": args.kwarg is not None,
        }
    return found


def check_python_calls(artifacts: dict[str, str], report: ContractReport) -> None:
    """Do the modules call each other the way they are actually written?

    This is the failure that made the case for the whole file. Two models split
    a Python package: one wrote `parse_csv(data, delimiter, strip_whitespace)`,
    the other called `parse_csv(text=..., has_header=...)`. Both files parse.
    Both are exactly what their own spec asked for. The tool raises TypeError on
    the first line of real work.

    Unlike the route checks, this needs nothing from the planner — it reads the
    code. Which matters, because the interface contract is shaped for web work
    and has no vocabulary for a module API.

    Deliberately conservative, since a false accusation about working code is
    worse than a missed fault: only names defined exactly once across the whole
    artifact set, never a function taking `**kwargs`, and only calls that are
    unambiguously to that function.
    """
    report.checks_run.append("python calls agree")

    defined: dict[str, dict[str, Any]] = {}
    ambiguous: set[str] = set()
    for path, text in artifacts.items():
        if not path.lower().endswith(".py"):
            continue
        for name, signature in _module_functions(path, text).items():
            if name in defined:
                # Two definitions of one name: any call could mean either.
                ambiguous.add(name)
            defined[name] = signature

    for name in ambiguous:
        defined.pop(name, None)
    if not defined:
        return

    import ast

    module_stems = {Path(p).stem: p for p in artifacts if p.lower().endswith(".py")}

    for path, text in artifacts.items():
        if not path.lower().endswith(".py"):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            func = node.func
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                # `parser.parse_csv(...)` counts only when `parser` is one of
                # this project's own modules; anything else may be a method.
                if func.value.id not in module_stems:
                    continue
                name = func.attr
            else:
                continue

            signature = defined.get(name)
            if signature is None or signature["has_kwargs"]:
                continue

            for keyword in node.keywords:
                if keyword.arg is None:  # **kwargs at the call site
                    break
                if keyword.arg not in signature["params"]:
                    report.add(
                        "error",
                        f"`{path}` calls `{name}({keyword.arg}=...)` but "
                        f"{signature['where']} defines "
                        f"`{name}({', '.join(signature['params'])})`",
                        where=path,
                        why="the two files were written by different models "
                        "against the same spec and disagree on the signature",
                    )

            if (
                not signature["has_varargs"]
                and len(node.args) > signature["positional"]
            ):
                report.add(
                    "error",
                    f"`{path}` calls `{name}` with {len(node.args)} positional "
                    f"argument(s); {signature['where']} accepts "
                    f"{signature['positional']}",
                    where=path,
                )
