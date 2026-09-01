"""Learned track record: how each model actually performs at each role.

The dispatcher already reads three inputs it can trust to some degree — a
hand-written capability sheet, live quota, and the models' own bids. This adds
the only one grounded in what actually happened: for each (model, role), how
often the artifact landed, and how often a reviewer from another vendor sent it
back.

Two properties make it worth persisting rather than recomputing:

* **It is expensive to learn.** Every data point cost a live request. Throwing
  the history away at process exit would mean paying for the same lesson every
  session, against a budget of 250 requests a day.
* **It converges slowly and should say so.** One success is not evidence. A
  record is shrunk toward neutral by its own sample count, so a model with two
  observations barely moves the score and one with twenty carries real weight.

The invariant this file exists to respect: **running out of quota is not a
performance failure.** A model at its daily cap was never asked, and recording
that as a defeat would teach the dispatcher to avoid the model that was working
perfectly well right up until midnight.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import profiles_path
from ..types import NodeResult, NodeState, Role, Verdict

# How fast a record forgets. 0.25 means roughly the last dozen outcomes carry
# most of the weight — recent behaviour matters more than history, because a
# provider swapping the model behind a name is a real and common event.
ALPHA = 0.25

# Samples needed before a record is trusted at full weight. Below this the score
# is shrunk toward NEUTRAL in proportion to how little is known.
CONFIDENT_AFTER = 8

NEUTRAL = 0.5
"""Score for a pairing never observed. Deliberately mid-range: an unproven model
stays eligible but unfavoured, exactly like an unlisted role affinity."""

SCHEMA_VERSION = 1


@dataclass(slots=True)
class Record:
    """One (model, role) pairing's history."""

    quality: float = NEUTRAL
    """EWMA of per-attempt outcomes in 0..1."""
    attempts: int = 0
    successes: int = 0
    rejections: int = 0
    """Times a cross-vendor reviewer sent the work back."""
    degradations: int = 0
    updated_utc: str = ""

    def observe(self, outcome: float, *, now: datetime | None = None) -> None:
        outcome = max(0.0, min(1.0, outcome))
        if self.attempts == 0:
            self.quality = outcome
        else:
            self.quality = (1 - ALPHA) * self.quality + ALPHA * outcome
        self.attempts += 1
        self.updated_utc = (now or datetime.now(timezone.utc)).isoformat()

    @property
    def score(self) -> float:
        """Quality, shrunk toward neutral by how little has been seen.

        Without this a single lucky first result would outrank a model with a
        long, slightly-imperfect history — and the dispatcher would chase noise
        it paid real requests to generate.
        """
        if self.attempts == 0:
            return NEUTRAL
        weight = min(1.0, self.attempts / CONFIDENT_AFTER)
        return NEUTRAL + weight * (self.quality - NEUTRAL)

    def to_dict(self) -> dict:
        return {
            "quality": round(self.quality, 4),
            "attempts": self.attempts,
            "successes": self.successes,
            "rejections": self.rejections,
            "degradations": self.degradations,
            "updated_utc": self.updated_utc,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Record:
        return cls(
            quality=float(data.get("quality", NEUTRAL)),
            attempts=int(data.get("attempts", 0)),
            successes=int(data.get("successes", 0)),
            rejections=int(data.get("rejections", 0)),
            degradations=int(data.get("degradations", 0)),
            updated_utc=str(data.get("updated_utc", "")),
        )


@dataclass(slots=True)
class Profiles:
    """The whole track record, keyed by (model_id, role)."""

    records: dict[tuple[str, Role], Record] = field(default_factory=dict)
    path: Path | None = None

    # -- reading ----------------------------------------------------------

    def record_for(self, model_id: str, role: Role) -> Record:
        return self.records.setdefault((model_id, role), Record())

    def score(self, model_id: str, role: Role) -> float:
        record = self.records.get((model_id, role))
        return record.score if record else NEUTRAL

    def as_track_record(self) -> dict[tuple[str, Role], float]:
        """The shape `ReconcileInput` wants."""
        return {key: record.score for key, record in self.records.items()}

    # -- writing ----------------------------------------------------------

    def observe_result(
        self, result: NodeResult, role: Role, *, now: datetime | None = None
    ) -> None:
        """Fold one node's outcome into the record of whoever produced it.

        Scoring, worst to best:

        * degraded — nobody produced it, so nothing is recorded against the
          model that was merely assigned it
        * rejected by review — 0.0, the strongest negative signal available,
          since a peer from another vendor read the work and refused it
        * revised after review — 0.5, it worked but needed a second pass
        * clean pass — 1.0, minus a little for each retry it took to get there
        """
        if result.state is NodeState.DEGRADED:
            # No model owns this: the node reached nobody who could serve it.
            # Recording a defeat here would punish a model for a quota wall.
            return
        if not result.model_id or result.state is not NodeState.DONE:
            return

        record = self.record_for(result.model_id, role)

        if result.review is not None and result.review.verdict is Verdict.REJECT:
            outcome = 0.0
            record.rejections += 1
        elif result.review is not None and result.review.verdict is Verdict.REVISE:
            outcome = 0.5
        else:
            # Retries are real cost even when the artifact is fine in the end.
            outcome = max(0.4, 1.0 - 0.2 * max(0, result.attempts - 1))
            record.successes += 1

        record.observe(outcome, now=now)

    def observe_degradation(self, model_id: str, role: Role) -> None:
        """A model that was asked and could not deliver, as distinct from one
        that was never asked because it had no quota left."""
        record = self.record_for(model_id, role)
        record.degradations += 1
        record.observe(0.0)

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": SCHEMA_VERSION,
            "records": {
                f"{model_id}|{role.value}": record.to_dict()
                for (model_id, role), record in sorted(
                    self.records.items(), key=lambda kv: (kv[0][0], kv[0][1].value)
                )
            },
        }

    @classmethod
    def from_dict(cls, data: dict, *, path: Path | None = None) -> Profiles:
        profiles = cls(path=path)
        if int(data.get("version", 0)) != SCHEMA_VERSION:
            # A format change means the history cannot be read safely. Starting
            # neutral is a small loss; misreading it would quietly bias every
            # assignment from here on.
            return profiles
        for key, value in (data.get("records") or {}).items():
            model_id, _, role_name = key.rpartition("|")
            try:
                role = Role(role_name)
            except ValueError:
                continue
            profiles.records[(model_id, role)] = Record.from_dict(value)
        return profiles

    @classmethod
    def load(cls, path: Path | None = None) -> Profiles:
        target = path or profiles_path()
        if not target.is_file():
            return cls(path=target)
        try:
            return cls.from_dict(
                json.loads(target.read_text(encoding="utf-8")), path=target
            )
        except (json.JSONDecodeError, OSError):
            # A corrupt history is not worth failing a run over.
            return cls(path=target)

    def save(self, path: Path | None = None) -> Path:
        target = path or self.path or profiles_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=False), encoding="utf-8"
        )
        temp.replace(target)
        return target

    # -- reporting --------------------------------------------------------

    def rows(self) -> list[tuple[str, Role, Record]]:
        return [
            (model_id, role, record)
            for (model_id, role), record in sorted(
                self.records.items(), key=lambda kv: (kv[0][0], kv[0][1].value)
            )
        ]
