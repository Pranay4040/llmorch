"""A conversation with the orchestrator, instead of one shot at it.

`llmorch run "build a notes app"` plans the whole build from one sentence and
then forgets it. The second thing anyone wants to say is "now add tags", and the
only way to say it was to re-run the whole sentence and get a fresh folder.

So this keeps a session: each instruction is planned against what the previous
ones produced, and only the files that must change are rewritten.

**What "memory" means here is the design decision.** The tempting version — feed
the previous files back to the planner — is exactly what the rest of this system
refuses to do, because pasting artifacts into prompts is the fastest way to
exhaust a 6,000 tokens-per-minute budget, and it grows with the project rather
than with the request. So a conversation remembers what the blackboard already
remembers between nodes:

- the instructions, verbatim — they are one line each,
- the interface contract, which is the shared spec anyway,
- one summary per file, written by the model that wrote the file, in the same
  response as the file itself, so it cost nothing extra.

Never the file contents. A ten-file project costs the same to remember as a
three-file one, and the twentieth turn costs what the second did.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import runs_dir
from .types import InterfaceContract, LaunchSpec, NodeResult, NodeState, TaskNode

CONVERSATION_NAME = "conversation.json"
CONVERSATION_VERSION = 1


@dataclass(slots=True)
class FileNote:
    """One artifact, as the next turn will remember it."""

    path: str
    node_id: str
    role: str = ""
    summary: str = ""
    model_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "node_id": self.node_id,
            "role": self.role,
            "summary": self.summary,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FileNote:
        return cls(
            path=str(raw.get("path", "")),
            node_id=str(raw.get("node_id", "")),
            role=str(raw.get("role", "")),
            summary=str(raw.get("summary", "")),
            model_id=str(raw.get("model_id", "")),
        )


@dataclass(slots=True)
class Turn:
    instruction: str
    utc: str = ""
    planned: tuple[str, ...] = ()
    completed: tuple[str, ...] = ()
    degraded: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "utc": self.utc,
            "planned": list(self.planned),
            "completed": list(self.completed),
            "degraded": list(self.degraded),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Turn:
        return cls(
            instruction=str(raw.get("instruction", "")),
            utc=str(raw.get("utc", "")),
            planned=tuple(str(x) for x in raw.get("planned") or ()),
            completed=tuple(str(x) for x in raw.get("completed") or ()),
            degraded=tuple(str(x) for x in raw.get("degraded") or ()),
        )


@dataclass(slots=True)
class Conversation:
    """Everything a later turn needs to know about the earlier ones."""

    session_id: str
    turns: list[Turn] = field(default_factory=list)
    interface: InterfaceContract = field(default_factory=InterfaceContract)
    files: dict[str, FileNote] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def started(self) -> bool:
        """Whether anything has been built yet.

        The first instruction plans a whole project; every later one plans a
        change to it. That is the only branch in this module.
        """
        return bool(self.files)

    def record(
        self,
        instruction: str,
        nodes: dict[str, TaskNode],
        results: dict[str, NodeResult],
        interface: InterfaceContract,
        *,
        now: str | None = None,
    ) -> Turn:
        """Fold one turn's outcome into the memory.

        A rewritten file replaces its note rather than adding one: the next turn
        must see the project as it stands, not as a history of what it has been.
        A degraded node leaves the previous note alone, because the file on disk
        is still the previous one.
        """
        self.interface = interface
        completed, degraded = [], []

        for node_id, result in results.items():
            node = nodes.get(node_id)
            if node is None:
                continue
            if result.state is NodeState.DONE:
                completed.append(node_id)
                self.files[node.output_path] = FileNote(
                    path=node.output_path,
                    node_id=node_id,
                    role=node.role.value,
                    summary=result.summary or node.title,
                    model_id=result.model_id or "",
                )
            else:
                degraded.append(node_id)

        turn = Turn(
            instruction=instruction,
            utc=now or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            planned=tuple(sorted(nodes)),
            completed=tuple(sorted(completed)),
            degraded=tuple(sorted(degraded)),
        )
        self.turns.append(turn)
        return turn

    def note_for(self, node_id: str) -> FileNote | None:
        for note in self.files.values():
            if note.node_id == node_id:
                return note
        return None

    def seed_results(self) -> dict[str, NodeResult]:
        """Prior work, as the blackboard expects to receive it.

        Seeded so a new node can declare `needs: ["server.summary"]` against a
        file an earlier turn wrote. Summary only — the artifact field stays
        empty, because the file itself is on disk and belongs nowhere near a
        prompt.
        """
        return {
            note.node_id: NodeResult(
                node_id=note.node_id,
                state=NodeState.DONE,
                summary=note.summary,
                model_id=note.model_id,
            )
            for note in self.files.values()
            if note.node_id
        }

    # ------------------------------------------------------------------
    # What the planner is told
    # ------------------------------------------------------------------

    def render_memory(self) -> str:
        """The conversation so far, in the form the next plan is made against."""
        lines = ["## What you have been asked so far", ""]
        for index, turn in enumerate(self.turns, start=1):
            lines.append(f"{index}. {turn.instruction}")

        lines += ["", "## The project as it stands", ""]
        for path in sorted(self.files):
            note = self.files[path]
            role = f" ({note.role})" if note.role else ""
            summary = note.summary.strip().splitlines()
            first = summary[0] if summary else ""
            lines.append(f"- `{path}`{role} — {first}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return runs_dir() / self.session_id / CONVERSATION_NAME

    def save(self) -> Path:
        """Written after every turn, so a quota wall or a crash costs one turn
        rather than the conversation."""
        target = self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CONVERSATION_VERSION,
            "session_id": self.session_id,
            "turns": [t.to_dict() for t in self.turns],
            "files": [n.to_dict() for n in self.files.values()],
            "interface": {
                "routes": list(self.interface.routes),
                "data_models": list(self.interface.data_models),
                "pages": list(self.interface.pages),
                "runtime": self.interface.runtime,
                "launch": self.interface.launch.to_dict(),
                "notes": self.interface.notes,
            },
        }
        temp = target.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(target)
        return target

    @classmethod
    def load(cls, session_id: str) -> Conversation | None:
        path = runs_dir() / session_id / CONVERSATION_NAME
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if int(raw.get("version", 0)) != CONVERSATION_VERSION:
            return None

        interface_raw = raw.get("interface") or {}
        conversation = cls(
            session_id=str(raw.get("session_id") or session_id),
            turns=[Turn.from_dict(t) for t in raw.get("turns") or []],
            interface=InterfaceContract(
                routes=tuple(interface_raw.get("routes") or ()),
                data_models=tuple(interface_raw.get("data_models") or ()),
                pages=tuple(str(p) for p in interface_raw.get("pages") or ()),
                runtime=str(interface_raw.get("runtime") or ""),
                launch=LaunchSpec.from_payload(interface_raw.get("launch")),
                notes=str(interface_raw.get("notes") or ""),
            ),
        )
        for note_raw in raw.get("files") or []:
            note = FileNote.from_dict(note_raw)
            if note.path:
                conversation.files[note.path] = note
        return conversation


def latest_session() -> str | None:
    """The most recent conversation, for `llmorch chat --continue`."""
    root = runs_dir()
    if not root.is_dir():
        return None
    sessions = sorted(
        (p.parent.name for p in root.glob(f"*/{CONVERSATION_NAME}")), reverse=True
    )
    return sessions[0] if sessions else None


def merge_interfaces(
    current: InterfaceContract, update: InterfaceContract
) -> InterfaceContract:
    """Fold a revision's contract into the standing one.

    A union rather than a replacement, because the planner is being asked about
    a change and answers about the change: a revision that adds one route and
    does not restate the other three is describing an addition, not a deletion.
    Losing the other three would silently unserve them — and every check
    downstream measures the artifacts against this contract, so it would report
    the wrong thing with total confidence.
    """

    def _merge(existing: tuple, incoming: tuple, key) -> tuple:
        merged = {key(item): item for item in existing if isinstance(item, dict)}
        for item in incoming:
            if isinstance(item, dict):
                merged[key(item)] = item
        return tuple(merged.values())

    return InterfaceContract(
        routes=_merge(
            current.routes,
            update.routes,
            lambda r: (str(r.get("method", "GET")).upper(), str(r.get("path", ""))),
        ),
        data_models=_merge(
            current.data_models, update.data_models, lambda m: str(m.get("name", ""))
        ),
        pages=tuple(dict.fromkeys((*current.pages, *update.pages))),
        runtime=update.runtime or current.runtime,
        launch=update.launch if update.launch.declared else current.launch,
        notes=update.notes or current.notes,
    )
