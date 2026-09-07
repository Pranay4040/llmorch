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
import re
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

    kind: str = "build"
    """build | ask | remark — which lane the line was handled in.

    Recorded rather than inferred, because the same sentence can go either way
    once `/ask` and `/build` exist, and a later turn reading the history has to
    see what was actually done, not what the classifier would decide today."""

    answer: str = ""
    """The reply, for an `ask` turn. Kept so a follow-up question has the thread
    of the conversation, and so `/history` can show what was said back."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "utc": self.utc,
            "planned": list(self.planned),
            "completed": list(self.completed),
            "degraded": list(self.degraded),
            "kind": self.kind,
            "answer": self.answer,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Turn:
        # `kind` and `answer` are additive: a conversation written before they
        # existed loads as a session of builds, which is what it was. That is
        # why the file version does not move — bumping it would decline every
        # session already on disk rather than reading it.
        return cls(
            instruction=str(raw.get("instruction", "")),
            utc=str(raw.get("utc", "")),
            planned=tuple(str(x) for x in raw.get("planned") or ()),
            completed=tuple(str(x) for x in raw.get("completed") or ()),
            degraded=tuple(str(x) for x in raw.get("degraded") or ()),
            kind=str(raw.get("kind") or "build"),
            answer=str(raw.get("answer") or ""),
        )


@dataclass(slots=True)
class Conversation:
    """Everything a later turn needs to know about the earlier ones."""

    session_id: str
    turns: list[Turn] = field(default_factory=list)
    interface: InterfaceContract = field(default_factory=InterfaceContract)
    files: dict[str, FileNote] = field(default_factory=dict)
    mode: str = ""
    """chat | agent | crew — how this session was opened.

    Remembered so `--continue` resumes the session it was rather than asking
    again and possibly getting a different answer: a conversation whose files
    were written by a crew is not a conversation one agent has been having."""

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

    def name_for(self, instruction: str) -> str:
        """Give the session a name taken from the first thing said to it.

        Only ever before its first save. The id is a directory name, a
        checkpoint key and the `run_id` on every ledger row this session
        produces; renaming it once any of those exist would orphan all three, so
        the window for naming is the one moment when none of them do.
        """
        if self.turns or self.files or self.path.exists():
            return self.session_id
        self.session_id = name_session(
            self.session_id, instruction, taken=set(existing_sessions())
        )
        return self.session_id

    def record_said(
        self, line: str, *, kind: str, answer: str = "", now: str | None = None
    ) -> Turn:
        """Record a turn that built nothing.

        A question and an acknowledgement both leave the project exactly as it
        was, so neither touches `files` or `interface`. They are still turns:
        the session is a record of what was said, and a question that vanished
        from it would take its answer with it.
        """
        turn = Turn(
            instruction=line,
            utc=now or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            kind=kind,
            answer=answer,
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

    @property
    def instructions(self) -> list[Turn]:
        """The turns that asked for something to be built.

        Questions and acknowledgements are excluded deliberately. The planner is
        shown this list as "what you have been asked so far", and "what does the
        server do?" was never asked of it — leaving it in would invite a plan
        that answers it in a file.
        """
        return [t for t in self.turns if t.kind == "build"]

    def render_memory(self) -> str:
        """The conversation so far, in the form the next plan is made against."""
        lines = ["## What you have been asked so far", ""]
        for index, turn in enumerate(self.instructions, start=1):
            lines.append(f"{index}. {turn.instruction}")

        lines += ["", "## The project as it stands", ""]
        for path in sorted(self.files):
            note = self.files[path]
            role = f" ({note.role})" if note.role else ""
            summary = note.summary.strip().splitlines()
            first = summary[0] if summary else ""
            lines.append(f"- `{path}`{role} — {first}")

        return "\n".join(lines)

    def render_for_answer(self) -> str:
        """The same memory, written for someone answering a question about it.

        A separate document from `render_memory` rather than a flag on it: the
        planner is told what to change, and is not helped by knowing which model
        wrote which file. Someone answering "why is the page like that?" is.
        """
        lines = ["## What this project was asked to be", ""]
        for index, turn in enumerate(self.instructions, start=1):
            lines.append(f"{index}. {turn.instruction}")
        if not self.instructions:
            lines.append("(nothing has been built in this session yet)")

        lines += ["", "## The files, and who wrote each", ""]
        if not self.files:
            lines.append("(none yet)")
        for path in sorted(self.files):
            note = self.files[path]
            role = f" ({note.role})" if note.role else ""
            by = f" [written by {note.model_id}]" if note.model_id else ""
            summary = " ".join(note.summary.strip().split())
            lines.append(f"- `{path}`{role}{by} — {summary}")

        return "\n".join(lines)

    def render_exchanges(self, limit: int = 3, *, width: int = 400) -> str:
        """The last few questions and answers, for a follow-up to hang on.

        Bounded twice — how many, and how long each — because this is the one
        part of the memory that would otherwise grow without limit, and "why?"
        needs the previous answer, not the previous ten.
        """
        asked = [t for t in self.turns if t.kind == "ask" and t.answer][-limit:]
        if not asked:
            return ""
        lines = ["## Questions already answered in this session", ""]
        for turn in asked:
            reply = " ".join(turn.answer.split())
            if len(reply) > width:
                reply = reply[:width].rstrip() + "…"
            lines.append(f"- Q: {turn.instruction}")
            lines.append(f"  A: {reply}")
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
            "mode": self.mode,
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
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            return None
        if int(raw.get("version", 0)) != CONVERSATION_VERSION:
            return None

        interface_raw = raw.get("interface") or {}
        conversation = cls(
            session_id=str(raw.get("session_id") or session_id),
            mode=str(raw.get("mode") or ""),
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


# Words that carry no information about what a session is *about*. The leading
# verb goes because every instruction starts with one — a directory called
# `build-a-notes-app` and one called `notes-app` say the same thing, and only one
# of them is still readable at 32 characters.
_OPENERS = frozenset(
    {"build", "make", "create", "write", "generate", "produce", "give", "add",
     "implement", "do", "please", "now", "can", "you", "could", "i", "want",
     "need", "lets", "let's"}
)
_STOPWORDS = frozenset(
    {"a", "an", "the", "that", "this", "to", "for", "with", "and", "of", "in",
     "on", "at", "by", "from", "into", "as", "it", "its", "me", "my", "some",
     "using", "use", "which", "is", "are", "be"}
)

SLUG_WORDS = 4
SLUG_CHARS = 32

_WORDS = re.compile(r"[A-Za-z0-9]+")


def slugify(instruction: str, *, words: int = SLUG_WORDS, chars: int = SLUG_CHARS) -> str:
    """A short, filesystem-safe name for what was asked for.

    Empty when nothing survives, which is a real answer: "build a thing" reduces
    to nothing worth naming a directory after, and a session called
    `20260907-165836` is better than one called `20260907-165836-thing`.
    """
    tokens = [w.lower() for w in _WORDS.findall(instruction)]

    while tokens and tokens[0] in _OPENERS:
        tokens.pop(0)
    kept = [w for w in tokens if w not in _STOPWORDS][:words]

    slug = "-".join(kept)[:chars].strip("-")
    return slug


def name_session(stamp: str, instruction: str, *, taken=None) -> str:
    """`<stamp>-<slug>`, or the bare stamp when there is no usable slug.

    The timestamp stays in front and stays fixed-width, because it is what makes
    these names sort chronologically — `latest_session`, `resume --list` and the
    dashboard's run list all order by the directory name and nothing else.
    """
    slug = slugify(instruction)
    if not slug:
        return stamp

    candidate = f"{stamp}-{slug}"
    if taken is None:
        return candidate

    # Two sessions in the same second about the same thing. Vanishingly rare and
    # cheap to rule out, and the alternative is one session writing into
    # another's directory.
    suffix = 2
    unique = candidate
    while unique in taken:
        unique = f"{candidate}-{suffix}"
        suffix += 1
    return unique


def existing_sessions() -> list[str]:
    root = runs_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def resolve_session(name: str) -> str | None:
    """Find a session from whatever the person typed.

    An exact id wins. Otherwise the slug alone is enough — `llmorch resume
    notes-app` rather than `llmorch resume 20260907-165836-notes-app` — as long
    as it picks out one session. Ambiguity returns None rather than guessing:
    resuming the wrong conversation is a worse outcome than being asked again.
    """
    if not name:
        return None
    sessions = existing_sessions()
    if name in sessions:
        return name

    matches = [s for s in sessions if s.endswith(f"-{name}") or name in s]
    return matches[-1] if len(matches) == 1 else None


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
