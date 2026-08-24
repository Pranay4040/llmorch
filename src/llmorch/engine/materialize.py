"""Write artifacts to a real, runnable project folder.

Without this step the orchestrator produces text blobs in a database and feels
broken even when every component worked.

Security note: `output_path` originates in LLM output, which makes it untrusted
input, and this is the only place in the system where such input reaches the
filesystem. Treat it as hostile. The guard is a resolve-then-verify-containment
check rather than string inspection, because string checks miss symlinks,
Windows 8.3 short names, and alternate separators.
"""

from __future__ import annotations

import ntpath
import posixpath
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from ..errors import UnsafePath
from ..types import NodeResult, NodeState, OutputKind, TaskNode
from .salvage import extract_code

# Names Windows refuses to create regardless of extension.
_RESERVED_WINDOWS = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

_LANG_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".md": "markdown",
    ".json": "json",
}


def safe_join(root: Path, candidate: str) -> Path:
    """Resolve `candidate` under `root`, or raise `UnsafePath`.

    Rejects absolute paths, drive letters, UNC paths, parent traversal, reserved
    device names, and anything that resolves outside `root` — including via a
    symlink planted earlier in the same run.
    """
    if not candidate or not candidate.strip():
        raise UnsafePath("output_path is empty")

    raw = candidate.strip().replace("\\", "/")

    if "\x00" in raw:
        raise UnsafePath("output_path contains a null byte")

    # Absolute in either convention, or drive-relative like "C:file".
    if posixpath.isabs(raw) or ntpath.isabs(candidate):
        raise UnsafePath(f"output_path must be relative, got {candidate!r}")
    if PureWindowsPath(candidate).drive or raw.startswith("//"):
        raise UnsafePath(f"output_path must not name a drive or share: {candidate!r}")

    parts = [p for p in PurePosixPath(raw).parts if p not in (".", "")]
    if not parts:
        raise UnsafePath(f"output_path resolves to nothing: {candidate!r}")
    if any(p == ".." for p in parts):
        raise UnsafePath(f"output_path must not traverse upwards: {candidate!r}")

    for part in parts:
        stem = part.split(".")[0].lower()
        if stem in _RESERVED_WINDOWS:
            raise UnsafePath(f"output_path uses reserved name {part!r}")
        if part.endswith(" ") or part.endswith("."):
            # Windows silently strips these, so the written path would differ
            # from the checked one.
            raise UnsafePath(f"output_path component {part!r} has a trailing space/dot")

    root = root.resolve()
    target = (root / PurePosixPath(*parts)).resolve()

    # The containment check. Done after resolve() so symlinks are followed.
    if target != root and root not in target.parents:
        raise UnsafePath(
            f"output_path escapes the output directory: {candidate!r} -> {target}"
        )

    # An existing symlink anywhere along the chain could still redirect the write.
    probe = target
    while probe != root and probe != probe.parent:
        if probe.is_symlink():
            raise UnsafePath(f"output_path passes through a symlink: {probe}")
        probe = probe.parent

    return target


@dataclass(frozen=True, slots=True)
class MaterializeReport:
    written: tuple[str, ...] = ()
    stubbed: tuple[str, ...] = ()
    rejected: tuple[tuple[str, str], ...] = ()
    """(output_path, reason) for paths refused by the safety check."""

    @property
    def total(self) -> int:
        return len(self.written) + len(self.stubbed)


def _stub_for(node: TaskNode, result: NodeResult) -> str:
    """Placeholder for a node that could not be produced.

    Written rather than skipped so the folder structure stays complete and the
    gap is self-explanatory instead of merely absent.
    """
    comment = {
        OutputKind.CODE: ("//" if node.output_path.endswith((".js", ".css")) else "#"),
        OutputKind.SCHEMA: "--",
    }.get(node.output_kind, "#")
    if node.output_path.endswith(".html"):
        head, tail = "<!--", "-->"
    else:
        head, tail = comment, ""

    reason = result.error or "no healthy model could produce this artifact"
    lines = [
        f"{head} DEGRADED — not generated {tail}".rstrip(),
        f"{head} node: {node.id} ({node.role.value}) {tail}".rstrip(),
        f"{head} reason: {reason} {tail}".rstrip(),
        f"{head} tried: {', '.join(result.vendors_tried) or 'none'} {tail}".rstrip(),
        "",
        f"{head} Original spec: {tail}".rstrip(),
    ]
    lines += [f"{head} {line} {tail}".rstrip() for line in node.spec.splitlines()]
    return "\n".join(lines) + "\n"


def _readme(nodes: list[TaskNode], report_lines: list[str]) -> str:
    return "\n".join(
        [
            "# Generated project",
            "",
            "Built by `llmorch` — different slices written by models from",
            "different vendors, assembled against a shared interface contract.",
            "",
            "## Run it",
            "",
            "```bash",
            "python server.py",
            "```",
            "",
            "Then open <http://localhost:8000>.",
            "",
            "## Files",
            "",
            *report_lines,
            "",
        ]
    )


def materialize(
    output_dir: Path,
    nodes: dict[str, TaskNode],
    results: dict[str, NodeResult],
    *,
    write_readme: bool = True,
) -> MaterializeReport:
    """Write every node's artifact into `output_dir`.

    A rejected path degrades that single file rather than aborting the run —
    one bad path from one model should not discard the work of every other.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()

    written: list[str] = []
    stubbed: list[str] = []
    rejected: list[tuple[str, str]] = []
    manifest_lines: list[str] = []

    for node_id, node in nodes.items():
        result = results.get(node_id)
        if result is None:
            continue

        try:
            target = safe_join(root, node.output_path)
        except UnsafePath as exc:
            rejected.append((node.output_path, str(exc)))
            continue

        if result.state is NodeState.DONE and result.artifact.strip():
            lang = _LANG_BY_SUFFIX.get(Path(node.output_path).suffix.lower())
            content = extract_code(result.artifact, prefer_lang=lang)
            if not content.endswith("\n"):
                content += "\n"
            status = f"{result.model_id or 'unknown'}"
            written.append(node.output_path)
        else:
            content = _stub_for(node, result)
            status = "DEGRADED — not generated"
            stubbed.append(node.output_path)

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        manifest_lines.append(f"- `{node.output_path}` — {node.title} ({status})")

    if write_readme:
        (root / "README.md").write_text(
            _readme(list(nodes.values()), manifest_lines), encoding="utf-8", newline="\n"
        )

    return MaterializeReport(
        written=tuple(written),
        stubbed=tuple(stubbed),
        rejected=tuple(rejected),
    )
