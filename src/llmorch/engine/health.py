"""Failover: what happens when a model cannot do the job.

Two scales, handled differently.

**Node-level.** One task fails, so try the next rung of its role chain —
*preferring a different vendor*. This preference is the whole point. Failure
modes correlate within a vendor: if a model returned malformed JSON twice, a
third attempt at the same vendor tends to fail identically, whereas a different
vendor has decorrelated blind spots.

**Model-level.** A model fails repeatedly, so stop using it for the rest of the
run and hand its remaining work to someone else in bulk, rather than letting
every one of its nodes discover the same breakage separately.

One distinction runs through both: **running out of quota is not a health
failure.** A model at its daily cap is unavailable, not broken, and must not
accumulate a track record penalty for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..errors import LLMOrchError, QuotaExhausted, RateLimited, Unservable
from ..registry.manifest import Manifest
from ..types import Role


RATE_LIMIT_EXTRA_ATTEMPTS = 2
"""Extra same-model attempts allowed for rate limits, over the normal retry
budget. Waiting is the correct response to a full window; failing over is not."""


class ModelHealth(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    """Tripped the circuit breaker — a genuine fault."""
    EXHAUSTED = "exhausted"
    """Out of quota for today. Not a fault; excluded without penalty."""


@dataclass(slots=True)
class HealthTracker:
    """Per-model health for the duration of one run."""

    threshold: int = 2
    _consecutive: dict[str, int] = field(default_factory=dict)
    _status: dict[str, ModelHealth] = field(default_factory=dict)
    _reassigned: set[str] = field(default_factory=set)
    """Models whose pending work has already been redistributed once."""
    events: list[str] = field(default_factory=list)

    # -- recording --------------------------------------------------------

    def note(self, message: str) -> None:
        """Record something worth reporting that is nobody's fault.

        The events list doubles as the run's notes, and a budget being raised
        belongs there — it explains a retry that would otherwise look like a
        model misbehaving.
        """
        self.events.append(message)

    def record_success(self, model_id: str) -> None:
        """Reset the streak. The breaker counts *consecutive* failures, so an
        intermittent fault should not accumulate toward tripping it."""
        self._consecutive[model_id] = 0

    def record_failure(self, model_id: str, error: Exception) -> ModelHealth:
        """Record a failure and return the model's resulting health."""
        if isinstance(error, QuotaExhausted):
            # Unavailable, not broken. No streak increment, no penalty.
            self._status[model_id] = ModelHealth.EXHAUSTED
            self.events.append(f"{model_id}: out of quota for today")
            return ModelHealth.EXHAUSTED

        if isinstance(error, RateLimited) and error.daily:
            # The server itself says this is the daily wall, not a burst.
            self._status[model_id] = ModelHealth.EXHAUSTED
            self.events.append(f"{model_id}: out of quota for today")
            return ModelHealth.EXHAUSTED

        if isinstance(error, RateLimited) and not error.daily:
            # Busy is not broken. A per-minute window clears in seconds, and a
            # model that is merely at its burst limit must not accumulate a
            # streak toward the circuit breaker — that is the same mistake as
            # counting a daily cap as a fault, one rung down.
            self.events.append(f"{model_id}: rate limited, waiting it out")
            return self._status.get(model_id, ModelHealth.HEALTHY)

        if isinstance(error, Unservable):
            # A sizing mismatch, not a defect. The node was routed to a model
            # that could never serve it; the feasibility filter owns that bug.
            self.events.append(f"{model_id}: request unservable at this size")
            return self._status.get(model_id, ModelHealth.HEALTHY)

        streak = self._consecutive.get(model_id, 0) + 1
        self._consecutive[model_id] = streak

        if streak >= self.threshold:
            self._status[model_id] = ModelHealth.UNHEALTHY
            self.events.append(
                f"{model_id}: circuit breaker tripped after {streak} "
                f"consecutive failures ({type(error).__name__})"
            )
            return ModelHealth.UNHEALTHY

        self.events.append(
            f"{model_id}: failure {streak}/{self.threshold} ({type(error).__name__})"
        )
        return ModelHealth.HEALTHY

    def mark_exhausted(self, model_id: str) -> None:
        self._status[model_id] = ModelHealth.EXHAUSTED

    # -- querying ---------------------------------------------------------

    def status(self, model_id: str) -> ModelHealth:
        return self._status.get(model_id, ModelHealth.HEALTHY)

    def is_available(self, model_id: str) -> bool:
        return self.status(model_id) is ModelHealth.HEALTHY

    def available(self, candidates: list[str]) -> list[str]:
        return [m for m in candidates if self.is_available(m)]

    @property
    def unhealthy_models(self) -> list[str]:
        """Genuinely broken, as distinct from merely exhausted."""
        return sorted(
            m for m, s in self._status.items() if s is ModelHealth.UNHEALTHY
        )

    # -- bulk reassignment ------------------------------------------------

    def needs_reassignment(self, model_id: str) -> bool:
        """Whether this model's pending work should be redistributed now.

        True at most once per model. Repeating it would let a flapping model
        trigger reassignment after reassignment and thrash the schedule.
        """
        if self.status(model_id) is ModelHealth.HEALTHY:
            return False
        if model_id in self._reassigned:
            return False
        self._reassigned.add(model_id)
        return True


# --------------------------------------------------------------------------
# Chain ordering
# --------------------------------------------------------------------------


def failover_chain(
    manifest: Manifest,
    role: Role,
    *,
    exclude: set[str] | None = None,
    tried_vendors: set[str] | None = None,
    health: HealthTracker | None = None,
) -> list[str]:
    """Ordered candidates for a role, vendor-diverse first.

    Models from vendors not yet attempted are promoted ahead of untried models
    at a vendor that has already failed. Manifest preference order is preserved
    within each group.
    """
    exclude = exclude or set()
    tried_vendors = tried_vendors or set()

    chain = [
        m
        for m in manifest.chain(role)
        if m not in exclude and (health is None or health.is_available(m))
    ]

    fresh = [m for m in chain if manifest.vendor_of(m) not in tried_vendors]
    repeat = [m for m in chain if manifest.vendor_of(m) in tried_vendors]
    return fresh + repeat


def next_model(
    manifest: Manifest,
    role: Role,
    *,
    exclude: set[str],
    tried_vendors: set[str],
    health: HealthTracker,
) -> str | None:
    """The next model to try, or None when the chain is spent."""
    chain = failover_chain(
        manifest, role, exclude=exclude, tried_vendors=tried_vendors, health=health
    )
    return chain[0] if chain else None


# --------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------


def should_retry_same_model(error: Exception, attempts: int, max_retries: int) -> bool:
    """Whether to retry the same model, rather than failing over.

    Only transport-shaped failures earn a same-model retry: those are transient
    and unrelated to the model's competence. A malformed response or a truncated
    output reflects how this model handles this prompt, and asking it again
    usually produces the same thing — which is why those cases fail over to a
    different vendor instead.
    """
    if not isinstance(error, LLMOrchError):
        return False
    if isinstance(error, (QuotaExhausted, Unservable)):
        return False
    if isinstance(error, RateLimited) and not error.daily:
        # Rate limits get their own, longer allowance. Unlike a malformed
        # response, waiting genuinely fixes them, and failing over early just
        # moves the same load onto a model with the same window.
        return attempts < max_retries + RATE_LIMIT_EXTRA_ATTEMPTS
    if attempts >= max_retries:
        return False
    return error.is_retryable and type(error).__name__ in (
        "TransportError",
        "RateLimited",
    )


def backoff_seconds(attempt: int, *, base: float = 1.0, cap: float = 30.0) -> float:
    """Exponential backoff with deterministic jitter.

    Jitter is derived from the attempt number rather than a random source, so a
    replayed run behaves identically.
    """
    delay = min(cap, base * (2 ** max(0, attempt - 1)))
    jitter = 1.0 + ((attempt * 37) % 20) / 100.0  # 1.00 .. 1.19
    return round(delay * jitter, 3)
