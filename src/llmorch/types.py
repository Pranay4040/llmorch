"""Core data types.

This module deliberately imports nothing from the rest of the package, so it can
be imported from anywhere without circularity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any

# --------------------------------------------------------------------------
# Roles
# --------------------------------------------------------------------------


class Role(str, Enum):
    """Fixed task taxonomy.

    Deliberately closed: `profiles.json` accumulates a per-(model, role) track
    record across runs, and free-text roles would make that history
    unmatchable between one run and the next.
    """

    PLANNING = "planning"
    RESEARCH = "research"
    BACKEND = "backend"
    FRONTEND = "frontend"
    STYLING = "styling"
    CONTENT = "content"
    REVIEW = "review"
    INTEGRATION = "integration"


class OutputKind(str, Enum):
    """What a node produces. Drives verification: CODE gets stricter checks."""

    CODE = "code"
    SCHEMA = "schema"
    TEXT = "text"
    SPEC = "spec"


class SplitHint(str, Enum):
    """How to chunk a node deterministically when it exceeds a model's ceiling."""

    NONE = "none"
    PER_FILE = "per_file"
    PER_ROUTE = "per_route"
    PER_SECTION = "per_section"


# --------------------------------------------------------------------------
# Quota
# --------------------------------------------------------------------------


class LimitKind(str, Enum):
    RPM = "rpm"
    TPM = "tpm"
    RPD = "rpd"
    TPD = "tpd"
    COST_USD_RUN = "cost_usd_run"
    REQUESTS_PER_RUN = "requests_per_run"


class LimitScope(str, Enum):
    """Whether a limit is shared across an account or tracked per model.

    Getting this wrong is the classic bug in this domain: if a limit is
    account-scoped, falling back to a *different model on the same provider*
    buys no additional quota at all.
    """

    ACCOUNT = "account"
    MODEL = "model"


class Admission(str, Enum):
    """Verdict from the governor on whether a request may proceed."""

    GRANTED = "granted"
    WAIT = "wait"
    EXHAUSTED_TODAY = "exhausted_today"
    UNSERVABLE = "unservable"
    COST_BLOCKED = "cost_blocked"


class Priority(str, Enum):
    """NORMAL cannot dip into a limit's `reserve`; HIGH can.

    The reserve exists so a wide fan-out cannot consume the last few requests
    that a critical-path retry will need.
    """

    NORMAL = "normal"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens


@dataclass(frozen=True, slots=True)
class RateLimitSnapshot:
    """Rate-limit state parsed from response headers.

    The server is authoritative. Local counters are an estimate that these
    values overwrite whenever they are present.
    """

    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    limit_requests: int | None = None
    limit_tokens: int | None = None
    """The provider's own view of the bucket size. Preferred over the manifest
    value when present: a published limit can be stale, or — for providers that
    publish none at all — simply a guess."""
    reset_requests_s: float | None = None
    reset_tokens_s: float | None = None
    retry_after_s: float | None = None
    daily_limit_hit: bool = False


@dataclass(frozen=True, slots=True)
class Headroom:
    """Human-facing snapshot of what a model has left. Used by `llmorch quota`."""

    model_id: str
    provider: str
    requests_used: int
    requests_limit: int | None
    tokens_used_minute: int
    tokens_limit_minute: int | None
    seconds_to_reset: float | None
    reset_tz: str
    healthy: bool = True


@dataclass(frozen=True, slots=True)
class Ticket:
    """Proof of a granted reservation.

    Tokens are reserved on an *estimate* at acquire time and reconciled to the
    true count at commit time; `release` refunds the reservation when a request
    never reached the provider.
    """

    ticket_id: str
    model_id: str
    provider: str
    est_prompt_tokens: int
    est_completion_tokens: int
    reserved_tokens: int
    acquired_monotonic: float
    priority: Priority = Priority.NORMAL

    @property
    def est_total_tokens(self) -> int:
        return self.est_prompt_tokens + self.est_completion_tokens


@dataclass(frozen=True, slots=True)
class Denial:
    """Why a request was refused, and whether waiting could ever help."""

    verdict: Admission
    model_id: str
    reason: str
    retry_after_s: float | None = None

    @property
    def is_permanent_today(self) -> bool:
        """True when waiting cannot help within the current quota day.

        UNSERVABLE means the request is larger than the model's per-minute
        ceiling and will never fit, no matter how long the scheduler waits.
        """
        return self.verdict in (
            Admission.UNSERVABLE,
            Admission.EXHAUSTED_TODAY,
            Admission.COST_BLOCKED,
        )


# --------------------------------------------------------------------------
# Provider I/O
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """A single provider call.

    `max_tokens` is mandatory, not optional. It converts the completion side of
    the token estimate from a guess into a hard upper bound, which is what lets
    admission control be sound rather than hopeful.
    """

    model_id: str
    messages: tuple[Message, ...]
    max_tokens: int
    system: str | None = None
    temperature: float = 0.2
    json_schema: dict[str, Any] | None = None
    timeout_s: float = 120.0


@dataclass(frozen=True, slots=True)
class ChatResponse:
    text: str
    usage: Usage
    model_reported: str
    latency_ms: int
    raw_status: int = 200
    rate_limit: RateLimitSnapshot | None = None
    truncated: bool = False
    """True when generation stopped at max_tokens rather than finishing."""


# --------------------------------------------------------------------------
# Task graph
# --------------------------------------------------------------------------


class NodeState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    VERIFYING = "verifying"
    DONE = "done"
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADED = "degraded"
    """Could not be produced, but the run continues and a stub is written."""
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TaskNode:
    id: str
    title: str
    role: Role
    spec: str
    output_path: str
    """Destination inside runs/<id>/output/. Untrusted: validated by materialize."""
    output_kind: OutputKind = OutputKind.TEXT
    deps: tuple[str, ...] = ()
    needs: tuple[str, ...] = ()
    """Upstream references like "n1.summary" — summaries, never whole artifacts."""
    est_output_tokens: int = 1500
    split_hint: SplitHint = SplitHint.NONE


@dataclass(slots=True)
class NodeResult:
    node_id: str
    state: NodeState
    artifact: str = ""
    summary: str = ""
    model_id: str | None = None
    attempts: int = 0
    vendors_tried: tuple[str, ...] = ()
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    review: VerifyResult | None = None
    """Tier 1 outcome, when a cross-vendor reviewer had an opinion. None means
    unreviewed — no reviewer, no quota, or review switched off — never
    'rejected'."""


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    """How to start what the build produces, stated rather than guessed.

    `InterfaceContract.runtime` is prose, written for the models: it says which
    OS and interpreter the code must be correct under. This is the machine-
    readable half of the same fact, and it exists because the smoke run
    previously had to infer it — hunting for `server.py` and reading a port
    literal out of the source. That inference is right for the pinned stack and
    wrong for everything else, so a Node or Go build could never be run at all.

    Empty is the honest default: a plan that declares nothing gets the old
    inference, and only a plan that states its launch gets to be started on its
    own terms. Nothing here is trusted — `engine.smoke.plan_launch` re-validates
    every field at the point of execution, because a contract can arrive from a
    model, a checkpoint, or a plan-cache file somebody edited by hand.
    """

    command: tuple[str, ...] = ()
    """argv, already split. `("python", "server.py")`, `("node", "server.js")`."""

    port: int | None = None
    """The port it listens on. None means read it out of the source, as before."""

    ready_path: str = "/"
    """A path that should answer once the server is up. A 404 here is fine — an
    API with no root route is a normal shape — but a 5xx or silence is not."""

    @property
    def declared(self) -> bool:
        return bool(self.command)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "port": self.port,
            "ready_path": self.ready_path,
        }

    @classmethod
    def from_payload(cls, raw: Any) -> LaunchSpec:
        """Read one out of untrusted JSON — a model's plan, a checkpoint, a
        hand-edited cache file.

        Shape only. Whether the result may actually be executed is decided at
        the point of execution by `engine.smoke.plan_launch`, because two of
        those three sources never pass through here.
        """
        if not isinstance(raw, dict):
            return cls()

        command = raw.get("command")
        if isinstance(command, str):
            # A model that wrote a command line where a list was asked for.
            # Splitting on whitespace is right for `python server.py` and wrong
            # for a path containing a space — the second is then refused by
            # name, which beats guessing at what was meant.
            command = command.split()
        if not isinstance(command, list):
            return cls()
        parts = tuple(
            str(a).strip()
            for a in command[:64]
            if isinstance(a, (str, int, float)) and str(a).strip()
        )

        port = raw.get("port")
        if isinstance(port, bool):
            port = None
        elif isinstance(port, str) and port.strip().isdigit():
            port = int(port)
        elif not isinstance(port, int):
            port = None

        ready = raw.get("ready_path")
        ready = str(ready)[:200] if isinstance(ready, str) and ready.strip() else "/"

        return cls(command=parts, port=port, ready_path=ready)


@dataclass(frozen=True, slots=True)
class InterfaceContract:
    """The shared spec every node receives verbatim.

    This is what allows a frontend written by one vendor's model to work against
    a backend written by another's, without the two models ever exchanging a
    message.
    """

    routes: tuple[dict[str, Any], ...] = ()
    data_models: tuple[dict[str, Any], ...] = ()
    pages: tuple[str, ...] = ()
    runtime: str = ""
    """Where the artifacts will actually run: OS, interpreter, working
    directory, how they are launched.

    Without it a model writes for the environment it imagines. The first live
    run produced path handling that is correct on POSIX and resolves every page
    to the drive root on Windows — and a reviewer told only the stack passed it,
    because on the platform it assumed, the code was right."""
    launch: LaunchSpec = LaunchSpec()
    """The same fact in a form a program can act on. See `LaunchSpec`."""
    notes: str = ""


# --------------------------------------------------------------------------
# Negotiation
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bid:
    """One model's self-assessment for one node.

    `confidence` is raw and untrustworthy on its own — free models tend to claim
    high competence uniformly. It is z-normalized within each bidder before it
    influences anything.
    """

    model_id: str
    node_id: str
    confidence: float
    est_output_tokens: int = 0
    why: str = ""


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Per-component scoring detail, surfaced by `llmorch plan --explain`."""

    z_confidence: float = 0.0
    role_affinity: float = 0.0
    track_record: float = 0.0
    quality_prior: float = 0.0
    quota_pressure: float = 0.0

    @property
    def total(self) -> float:
        return (
            0.35 * self.z_confidence
            + 0.25 * self.role_affinity
            + 0.15 * self.track_record
            + 0.15 * self.quality_prior
            - 0.10 * self.quota_pressure
        )


@dataclass(frozen=True, slots=True)
class Assignment:
    node_id: str
    model_id: str
    score: float
    breakdown: ScoreBreakdown
    rationale: str = ""


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


class Verdict(str, Enum):
    PASS = "pass"
    REVISE = "revise"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class Issue:
    severity: str  # "error" | "warning" | "info"
    what: str
    why: str = ""
    line: int | None = None


@dataclass(frozen=True, slots=True)
class VerifyResult:
    verdict: Verdict
    tier: int  # 0 = deterministic/free, 1 = cross-vendor LLM review
    issues: tuple[Issue, ...] = ()
    reviewer_model_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.PASS


# --------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """One row in the ledger — the single source of truth for all accounting.

    Quota counters are derived from these events rather than stored separately,
    so a crash cannot leave a counter drifting out of sync with reality.
    """

    run_id: str
    node_id: str | None
    purpose: str  # decompose | bid | execute | review | repair | retry
    provider: str
    model_id: str
    ts_utc: str
    day_key: str
    """Provider-local YYYY-MM-DD, since reset timezones differ per provider."""
    est_prompt_tokens: int
    est_completion_tokens: int
    usage: Usage
    cost_usd: Decimal = Decimal("0")
    ok: bool = True
    http_status: int = 200
    latency_ms: int = 0
    error: str | None = None
