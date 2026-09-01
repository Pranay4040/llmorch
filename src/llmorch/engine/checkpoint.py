"""Run state that survives a quota day.

A daily cap is not a crash — it is a scheduled event. Gemini allows 250 requests
a day, so a run of any size will meet the wall, and what happens next decides
whether the system is usable at all. Without a checkpoint the answer is: the
blocked nodes degrade to stubs, the run reports success, and tomorrow's re-run
spends quota re-requesting every node that already worked. The wall gets paid
for twice.

So each wave of the scheduler writes down what is finished, including the
artifact text itself. `llmorch resume` reloads that and runs only what is
missing. Completed work is never re-requested — which is the entire point, since
the scarce resource is requests rather than time or money.

Three rules hold this together:

* **Writes are atomic.** A checkpoint is written to a temporary file and then
  renamed over the old one. A crash halfway through a write must not destroy the
  last good checkpoint — that would turn a recoverable interruption into exactly
  the total loss this file exists to prevent.
* **The graph is fingerprinted.** A checkpoint only applies to the task it came
  from. If the node set changed, resuming would silently mix artifacts from two
  different plans, so the mismatch is refused rather than merged.
* **Only DONE nodes are restored.** A degraded node carries no usable artifact,
  and its stub must not be mistaken for finished work on the next pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import runs_dir
from ..errors import EngineError, UnsafePath
from ..types import (
    InterfaceContract,
    NodeResult,
    NodeState,
    OutputKind,
    Role,
    SplitHint,
    TaskNode,
    Usage,
)

CHECKPOINT_VERSION = 1
CHECKPOINT_NAME = "checkpoint.json"

# Run ids are generated from a timestamp, but `llmorch resume <id>` lets one
# arrive from the command line, and it is joined onto a path. Anything outside
# this alphabet is refused rather than sanitised: silently rewriting a path is
# how a traversal slips through.
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class CheckpointError(EngineError):
    """A checkpoint is missing, unreadable, or does not fit this task."""


@dataclass(frozen=True, slots=True)
class NodeSnapshot:
    """One node's outcome, complete enough to never need re-requesting."""

    node_id: str
    state: NodeState
    artifact: str = ""
    summary: str = ""
    model_id: str | None = None
    attempts: int = 0
    vendors_tried: tuple[str, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None

    @classmethod
    def of(cls, result: NodeResult) -> NodeSnapshot:
        return cls(
            node_id=result.node_id,
            state=result.state,
            # A degraded node has nothing worth keeping, and storing its stub
            # would risk it being read back as finished work.
            artifact=result.artifact if result.state is NodeState.DONE else "",
            summary=result.summary,
            model_id=result.model_id,
            attempts=result.attempts,
            vendors_tried=tuple(result.vendors_tried),
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            error=result.error,
        )

    def to_result(self) -> NodeResult:
        return NodeResult(
            node_id=self.node_id,
            state=self.state,
            artifact=self.artifact,
            summary=self.summary,
            model_id=self.model_id,
            attempts=self.attempts,
            vendors_tried=self.vendors_tried,
            usage=Usage(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
            ),
            error=self.error,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "artifact": self.artifact,
            "summary": self.summary,
            "model_id": self.model_id,
            "attempts": self.attempts,
            "vendors_tried": list(self.vendors_tried),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeSnapshot:
        return cls(
            node_id=str(data["node_id"]),
            state=NodeState(data.get("state", "pending")),
            artifact=data.get("artifact") or "",
            summary=data.get("summary") or "",
            model_id=data.get("model_id"),
            attempts=int(data.get("attempts") or 0),
            vendors_tried=tuple(data.get("vendors_tried") or ()),
            prompt_tokens=int(data.get("prompt_tokens") or 0),
            completion_tokens=int(data.get("completion_tokens") or 0),
            error=data.get("error"),
        )


@dataclass(slots=True)
class Checkpoint:
    """Everything needed to pick a run back up on the other side of a wall."""

    run_id: str
    task: str
    task_signature: str
    created_utc: str
    updated_utc: str
    plan: dict[str, Any] = field(default_factory=dict)
    """The graph and contract this run was made with.

    Stored rather than re-derived, because re-planning on resume is wrong twice
    over: it spends the scarcest request in the system to rediscover a known
    answer, and it can return a *different* graph — the plan signature includes
    the roster, so adding a provider between the run and the resume is enough
    to change it. A checkpoint that cannot reconstruct its own graph is not
    self-contained."""
    nodes: dict[str, NodeSnapshot] = field(default_factory=dict)
    blocked_until: dict[str, str] = field(default_factory=dict)
    """model id -> ISO-8601 UTC moment its daily quota next resets."""
    version: int = CHECKPOINT_VERSION

    # -- inspection -------------------------------------------------------

    @property
    def completed(self) -> list[str]:
        return sorted(
            n for n, s in self.nodes.items() if s.state is NodeState.DONE
        )

    @property
    def unfinished(self) -> list[str]:
        """Nodes a resume would have to run again — degraded ones included."""
        return sorted(
            n for n, s in self.nodes.items() if s.state is not NodeState.DONE
        )

    @property
    def is_complete(self) -> bool:
        return bool(self.nodes) and not self.unfinished

    def resumable_at(self) -> datetime | None:
        """When the first blocked model comes back, or None if nothing is.

        None means resume now: either no model was blocked on quota, or the
        blockage was something waiting cannot fix — in which case the resume
        will fail over rather than sit still.
        """
        moments = []
        for value in self.blocked_until.values():
            try:
                moments.append(datetime.fromisoformat(value))
            except ValueError:
                continue
        future = [m for m in moments if m.tzinfo is not None]
        return min(future) if future else None

    def seconds_until_resumable(self, now: datetime | None = None) -> float:
        moment = self.resumable_at()
        if moment is None:
            return 0.0
        reference = now or datetime.now(timezone.utc)
        return max(0.0, (moment - reference).total_seconds())

    def restore_results(self) -> dict[str, NodeResult]:
        """The finished work, ready to seed a fresh run.

        Only DONE nodes. Anything else is re-run, because a stub that looks
        finished is worse than no artifact at all.
        """
        return {
            node_id: snapshot.to_result()
            for node_id, snapshot in self.nodes.items()
            if snapshot.state is NodeState.DONE
        }

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_id": self.run_id,
            "task": self.task,
            "task_signature": self.task_signature,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "plan": self.plan,
            "blocked_until": dict(self.blocked_until),
            "nodes": [s.to_dict() for s in self.nodes.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        version = int(data.get("version") or 0)
        if version != CHECKPOINT_VERSION:
            raise CheckpointError(
                f"checkpoint is version {version}, this build writes "
                f"v{CHECKPOINT_VERSION}; start a fresh run rather than "
                "resuming across a format change"
            )
        nodes = {
            snapshot["node_id"]: NodeSnapshot.from_dict(snapshot)
            for snapshot in data.get("nodes") or []
        }
        return cls(
            run_id=str(data["run_id"]),
            task=str(data.get("task") or ""),
            task_signature=str(data.get("task_signature") or ""),
            created_utc=str(data.get("created_utc") or ""),
            updated_utc=str(data.get("updated_utc") or ""),
            nodes=nodes,
            plan=dict(data.get("plan") or {}),
            blocked_until=dict(data.get("blocked_until") or {}),
            version=version,
        )


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------


def plan_to_dict(
    nodes: dict[str, TaskNode], interface: InterfaceContract
) -> dict[str, Any]:
    """Freeze the graph and its contract into the checkpoint.

    Deliberately hand-rolled here rather than reusing the decomposer's
    serialiser: the engine must not depend on the negotiation package to read
    back its own state, or a checkpoint could only be loaded by the thing that
    happened to write it.
    """
    return {
        "interface": {
            "routes": [dict(r) for r in interface.routes],
            "data_models": [dict(m) for m in interface.data_models],
            "pages": list(interface.pages),
            "runtime": interface.runtime,
            "notes": interface.notes,
        },
        "nodes": [
            {
                "id": n.id,
                "title": n.title,
                "role": n.role.value,
                "spec": n.spec,
                "output_path": n.output_path,
                "output_kind": n.output_kind.value,
                "deps": list(n.deps),
                "needs": list(n.needs),
                "est_output_tokens": n.est_output_tokens,
                "split_hint": n.split_hint.value,
            }
            for n in nodes.values()
        ],
    }


def plan_from_dict(data: dict[str, Any]) -> tuple[list[TaskNode], InterfaceContract]:
    """Rebuild the graph a checkpoint was written against."""
    raw_interface = data.get("interface") or {}
    interface = InterfaceContract(
        routes=tuple(raw_interface.get("routes") or ()),
        data_models=tuple(raw_interface.get("data_models") or ()),
        pages=tuple(raw_interface.get("pages") or ()),
        runtime=str(raw_interface.get("runtime") or ""),
        notes=str(raw_interface.get("notes") or ""),
    )
    nodes = [
        TaskNode(
            id=str(raw["id"]),
            title=str(raw.get("title") or raw["id"]),
            role=Role(raw.get("role", "backend")),
            spec=str(raw.get("spec") or ""),
            output_path=str(raw.get("output_path") or ""),
            output_kind=OutputKind(raw.get("output_kind", "text")),
            deps=tuple(raw.get("deps") or ()),
            needs=tuple(raw.get("needs") or ()),
            est_output_tokens=int(raw.get("est_output_tokens") or 800),
            split_hint=SplitHint(raw.get("split_hint", "none")),
        )
        for raw in (data.get("nodes") or [])
    ]
    return nodes, interface


def signature_of(task: str, nodes: dict[str, TaskNode]) -> str:
    """Fingerprint the plan a checkpoint belongs to.

    Covers the identity of the work rather than its wording: node ids, roles,
    output paths, and dependency edges. Editing a spec is a reason to re-run
    that node, not to invalidate the whole checkpoint; adding or removing a node
    changes the shape of the graph, and resuming across that would silently
    stitch together two different plans.
    """
    canonical = json.dumps(
        {
            "task": task.strip(),
            "nodes": sorted(
                [n.id, n.role.value, n.output_path, sorted(n.deps)]
                for n in nodes.values()
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def new_checkpoint(
    *,
    run_id: str,
    task: str,
    nodes: dict[str, TaskNode],
    interface: InterfaceContract | None = None,
    now: datetime | None = None,
) -> Checkpoint:
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    return Checkpoint(
        run_id=run_id,
        task=task,
        task_signature=signature_of(task, nodes),
        created_utc=stamp,
        updated_utc=stamp,
        plan=plan_to_dict(nodes, interface or InterfaceContract()),
    )


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def validate_run_id(run_id: str) -> str:
    """Refuse a run id that could escape the runs directory."""
    if not _RUN_ID.match(run_id or ""):
        raise UnsafePath(
            f"run id {run_id!r} is not a plain identifier; refusing to use it "
            "as a path component"
        )
    return run_id


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / CHECKPOINT_NAME


def save(run_dir: Path, checkpoint: Checkpoint, *, now: datetime | None = None) -> Path:
    """Write the checkpoint atomically.

    Temp file then `os.replace`, which is atomic on both POSIX and Windows. A
    crash during the write leaves the previous checkpoint intact; a torn file
    here would turn a recoverable interruption into the total loss this whole
    module exists to prevent.
    """
    checkpoint.updated_utc = (now or datetime.now(timezone.utc)).isoformat()
    run_dir.mkdir(parents=True, exist_ok=True)

    target = checkpoint_path(run_dir)
    temp = target.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(checkpoint.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temp, target)
    return target


def load(run_dir: Path) -> Checkpoint:
    path = checkpoint_path(run_dir)
    if not path.is_file():
        raise CheckpointError(f"no checkpoint at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise CheckpointError(f"{path} is unreadable: {exc}") from exc
    return Checkpoint.from_dict(data)


def load_run(run_id: str, *, root: Path | None = None) -> Checkpoint:
    root = root or runs_dir()
    return load(root / validate_run_id(run_id))


def list_checkpoints(*, root: Path | None = None, limit: int = 20) -> list[Checkpoint]:
    """Every readable checkpoint, newest run first.

    Unreadable ones are skipped rather than raised: one corrupt directory must
    not hide every other resumable run from the listing.
    """
    root = root or runs_dir()
    if not root.is_dir():
        return []

    found: list[Checkpoint] = []
    for run_dir in sorted(root.iterdir(), reverse=True):
        if len(found) >= limit:
            break
        if not run_dir.is_dir() or not checkpoint_path(run_dir).is_file():
            continue
        try:
            found.append(load(run_dir))
        except CheckpointError:
            continue
    return found


def latest_resumable(*, root: Path | None = None) -> Checkpoint | None:
    """The newest run that still has work left in it."""
    return next(
        (cp for cp in list_checkpoints(root=root) if not cp.is_complete), None
    )


def check_applies(checkpoint: Checkpoint, task: str, nodes: dict[str, TaskNode]) -> None:
    """Refuse a checkpoint that belongs to a different plan.

    Resuming across a changed graph would mix artifacts built against two
    different sets of assumptions — the interface contract chief among them —
    and the result would look plausible while being incoherent.
    """
    actual = signature_of(task, nodes)
    if checkpoint.task_signature != actual:
        raise CheckpointError(
            f"checkpoint {checkpoint.run_id} was written for a different task "
            f"graph ({checkpoint.task_signature} != {actual}). Start a new run "
            "instead of resuming onto a changed plan."
        )
