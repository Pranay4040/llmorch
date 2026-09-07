"""What a run is doing right now, published while it is still doing it.

Everything else this system reports is retrospective: the spend table at the
end, `report.md` beside the folder, the ledger the next morning. During the
thirty seconds to several minutes a run actually takes, the only visible thing
was a plan and then silence — and the numbers a person most wants during a run
are exactly the ones that change during it.

**It goes through a file, because the watcher is a different process.**
`llmorch dashboard` is started separately and outlives any one run, so there is
no object to share. The checkpoint already established the shape: a run writes
its state to its own directory after every wave, and something else reads it.
This is that, at a finer grain and with no obligation to be resumable — which
is the difference worth keeping, since a progress file that failed to write must
never be able to fail a run.

**Every write is whole.** Written to a temporary name and renamed over, so a
reader either sees the previous state or the next one, never half of either. A
dashboard polling every two seconds will hit a write in progress eventually, and
a half-written JSON document is a crash in the reader rather than a stale
number.

**A reserved token is not a spent one.** The governor reserves on an estimate at
acquire time and reconciles to the truth on commit, so a node in flight has a
budget and not a bill. Reporting the estimate as spend would show a total that
walks backwards when the wave lands, which is the sort of number people stop
trusting.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..types import Assignment, NodeResult, NodeState, TaskNode

PROGRESS_NAME = "progress.json"
PROGRESS_VERSION = 1

STALE_AFTER_S = 90.0
"""How long after its last write a run is presumed dead rather than working.

Generous on purpose: a node can legitimately sit for a minute waiting out a
per-minute rate limit, and calling that "finished" would make the dashboard
lie about the most interesting moment there is.
"""


def progress_path(run_dir: Path) -> Path:
    return run_dir / PROGRESS_NAME


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(slots=True)
class NodeProgress:
    node_id: str
    title: str = ""
    role: str = ""
    model_id: str = ""
    state: str = "pending"
    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    started_utc: str = ""
    ended_utc: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            # Written by the planner, therefore untrusted. It reaches a browser,
            # so the page sets it as text and never as markup — the same rule the
            # rest of the dashboard follows.
            "title": self.title[:120],
            "role": self.role,
            "model_id": self.model_id,
            "state": self.state,
            "attempts": self.attempts,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "started_utc": self.started_utc,
            "ended_utc": self.ended_utc,
            "error": self.error[:300],
        }


@dataclass(slots=True)
class ProgressWriter:
    """One run's live state, on disk.

    Held by the scheduler, which calls it on every transition it already knows
    about. Nothing here may raise: a run that died because its progress file
    could not be written would be the monitoring costing more than it reports.
    """

    run_dir: Path
    run_id: str
    task: str = ""
    live: bool = False
    nodes: dict[str, NodeProgress] = field(default_factory=dict)
    started_utc: str = field(default_factory=_utc)
    finished: bool = False
    note: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    """The end-of-run facts the terminal used to print. See `summarise`."""
    _headroom: Any = None

    # ------------------------------------------------------------------
    # What the scheduler tells it
    # ------------------------------------------------------------------

    def begin(
        self,
        graph_nodes: dict[str, TaskNode],
        assignments: dict[str, Assignment],
        *,
        headroom=None,
    ) -> None:
        """The plan, before any of it has happened.

        Published immediately so a dashboard opened during the first request
        shows the whole shape of the run rather than only the parts that have
        already finished.
        """
        self._headroom = headroom
        for node_id, node in graph_nodes.items():
            assigned = assignments.get(node_id)
            self.nodes[node_id] = NodeProgress(
                node_id=node_id,
                title=node.title,
                role=node.role.value,
                model_id=assigned.model_id if assigned else "",
            )
        self.write()

    def node_started(self, node_id: str, model_id: str) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        node.state = "running"
        node.model_id = model_id
        node.started_utc = _utc()
        self.write()

    def node_finished(self, node_id: str, result: NodeResult) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        node.state = result.state.value
        node.attempts = result.attempts
        node.model_id = result.model_id or node.model_id
        node.ended_utc = _utc()
        node.error = result.error or ""
        if result.usage is not None:
            node.prompt_tokens = result.usage.prompt_tokens
            node.completion_tokens = result.usage.completion_tokens
        self.write()

    def restored(self, node_id: str, result: NodeResult) -> None:
        """A node a resume carried over: finished, but not by this process.

        Marked rather than shown as running, so a resumed run does not report
        tokens it did not spend today.
        """
        node = self.nodes.get(node_id)
        if node is None:
            return
        node.state = result.state.value
        node.model_id = result.model_id or ""
        node.error = "carried over from an earlier attempt"

    def summarise(self, **facts: Any) -> None:
        """Everything the terminal used to print at the end.

        The tables moved to the browser, so the browser has to carry what they
        carried: where the folder is, what the cross-artifact checks found,
        whether the smoke run started anything, and the warnings. Recomputed
        nowhere — these are the same objects the renderers were handed, which is
        the rule `report.md` already follows so the file and the screen cannot
        disagree about what happened.
        """
        self.summary.update(facts)
        self.write()

    def finish(self, note: str = "") -> None:
        self.finished = True
        self.note = note
        for node in self.nodes.values():
            if node.state in ("pending", "running"):
                # The run ended without this node settling. Saying so is better
                # than leaving it spinning in a page nobody will refresh again.
                node.state = NodeState.DEGRADED.value
                node.error = node.error or "the run ended before this node settled"
        self.write()

    # ------------------------------------------------------------------
    # The document
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        nodes = [n.to_dict() for n in self.nodes.values()]
        spent = [n for n in self.nodes.values() if n.prompt_tokens or n.completion_tokens]

        by_model: dict[str, dict[str, int]] = {}
        for node in self.nodes.values():
            if not node.model_id:
                continue
            entry = by_model.setdefault(
                node.model_id, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            )
            if node.prompt_tokens or node.completion_tokens:
                entry["calls"] += 1
                entry["prompt_tokens"] += node.prompt_tokens
                entry["completion_tokens"] += node.completion_tokens

        # Read live rather than accumulated here: the governor holds
        # reservations as well as commits, so this is what the *next* request
        # will actually be admitted against.
        headroom: list[dict[str, Any]] = []
        if self._headroom is not None:
            try:
                for model_id, head in self._headroom().items():
                    headroom.append(
                        {
                            "model_id": model_id,
                            "provider": head.provider,
                            "requests_used": head.requests_used,
                            "requests_limit": head.requests_limit,
                            "tokens_used_minute": head.tokens_used_minute,
                            "tokens_limit_minute": head.tokens_limit_minute,
                            "healthy": head.healthy,
                        }
                    )
            except Exception:  # never let a reading break the run
                headroom = []

        return {
            "version": PROGRESS_VERSION,
            "run_id": self.run_id,
            "task": self.task[:400],
            "live": self.live,
            "finished": self.finished,
            "note": self.note[:300],
            "started_utc": self.started_utc,
            "updated_utc": _utc(),
            "summary": self.summary,
            "nodes": nodes,
            "by_model": [{"model_id": k, **v} for k, v in sorted(by_model.items())],
            "headroom": headroom,
            "totals": {
                "nodes": len(self.nodes),
                "settled": sum(
                    1
                    for n in self.nodes.values()
                    if n.state not in ("pending", "running")
                ),
                "running": sum(1 for n in self.nodes.values() if n.state == "running"),
                "prompt_tokens": sum(n.prompt_tokens for n in spent),
                "completion_tokens": sum(n.completion_tokens for n in spent),
            },
        }

    def write(self) -> None:
        """Atomically, and never fatally."""
        target = progress_path(self.run_dir)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(f"{PROGRESS_NAME}.{os.getpid()}.tmp")
            temp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
            temp.replace(target)
        except OSError:
            # Monitoring that can fail a run costs more than it reports.
            return


def read_progress(run_dir: Path) -> dict[str, Any] | None:
    """One run's published state, or None if it has none or it is unreadable."""
    path = progress_path(run_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or int(raw.get("version", 0)) != PROGRESS_VERSION:
        return None
    return raw


def is_active(payload: dict[str, Any], *, now: datetime | None = None) -> bool:
    """Whether that state is worth showing as happening rather than as history.

    A finished run is never active. An unfinished one whose file has not moved
    in `STALE_AFTER_S` is a process that was killed — the file is the only thing
    it leaves behind, and nothing gets to write "finished" on its behalf.
    """
    if payload.get("finished"):
        return False
    stamp = str(payload.get("updated_utc") or "")
    try:
        updated = datetime.fromisoformat(stamp)
    except ValueError:
        return False
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    moment = now or datetime.now(timezone.utc)
    return (moment - updated).total_seconds() <= STALE_AFTER_S
