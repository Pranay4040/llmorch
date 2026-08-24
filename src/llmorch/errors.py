"""Exception hierarchy.

Split along one axis that matters everywhere else in the codebase: whether
retrying could ever succeed. `is_retryable` is what the failover ladder in
`engine/health.py` branches on.
"""

from __future__ import annotations


class LLMOrchError(Exception):
    """Base for every error this package raises."""

    is_retryable: bool = False


# --------------------------------------------------------------------------
# Configuration / startup — never retryable
# --------------------------------------------------------------------------


class ConfigError(LLMOrchError):
    """Bad or missing configuration."""


class ManifestError(ConfigError):
    """models.yaml is malformed or internally inconsistent."""


class MissingKeyError(ConfigError):
    """A provider was requested but its API key is absent.

    Never include the key name's *value* in the message — only its variable name.
    """


# --------------------------------------------------------------------------
# Quota — the distinction here is load-bearing
# --------------------------------------------------------------------------


class QuotaError(LLMOrchError):
    """Base for admission-control refusals."""


class QuotaExhausted(QuotaError):
    """The model's daily allowance is spent.

    Not retryable within this quota day, but the work is not lost: it is
    checkpointed and `llmorch resume` picks it up after the reset. Crucially,
    this is *not* a health failure — running out of quota is not the model
    being broken, so it must never count toward the circuit breaker.
    """

    is_retryable = False


class QuotaBusy(QuotaError):
    """Rate limited right now, but the window will clear on its own.

    Distinct from `QuotaExhausted` deliberately, and for the same reason the
    governor separates WAIT from EXHAUSTED_TODAY: a model that is merely busy
    is healthy. It keeps its place in every later fallback chain and takes no
    track-record penalty. Collapsing the two benches a working model for the
    rest of the run over a pause of a few seconds — and against an
    account-scoped 30 RPM limit, that happens seconds into the first fan-out.
    """

    is_retryable = True


class Unservable(QuotaError):
    """The request is larger than the model's per-minute token ceiling.

    Permanently impossible for this model at this size — waiting cannot help.
    Conflating this with a wait condition hangs the scheduler forever, which is
    why it is a distinct type. Against a 6,000 TPM provider this fires often.
    """

    is_retryable = False


class CostBlocked(QuotaError):
    """A paid provider was reached without sufficient budget or --allow-paid."""

    is_retryable = False


# --------------------------------------------------------------------------
# Provider — mostly retryable
# --------------------------------------------------------------------------


class ProviderError(LLMOrchError):
    """A provider call failed."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class TransportError(ProviderError):
    """Network failure, timeout, or 5xx. Worth retrying on the same model."""

    is_retryable = True


class RateLimited(ProviderError):
    """HTTP 429. Retryable once the counters are resynced from the headers."""

    is_retryable = True

    def __init__(
        self, message: str, *, retry_after_s: float | None = None, daily: bool = False
    ) -> None:
        super().__init__(message, status=429)
        self.retry_after_s = retry_after_s
        self.daily = daily
        """True when the 429 signals a *daily* cap rather than a per-minute one,
        in which case retrying today walks straight back into the wall."""


class SchemaInvalid(ProviderError):
    """Response did not match the expected schema.

    Retryable, but salvage is attempted first — recovering a fenced JSON block
    costs nothing, whereas a repair request costs quota.
    """

    is_retryable = True


class Truncated(ProviderError):
    """Generation stopped at max_tokens. Among the most common free-model
    failures, and detectable for free."""

    is_retryable = True


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


class EngineError(LLMOrchError):
    """Base for orchestration failures."""


class GraphError(EngineError):
    """The task DAG is malformed — cycle, dangling dependency, unknown role."""


class UnsafePath(EngineError):
    """A node's output_path tried to escape the run's output directory.

    Raised on absolute paths, drive letters, `..` traversal, and symlink escapes.
    Model output is untrusted input wherever it reaches the filesystem.
    """


class NoHealthyModel(EngineError):
    """Every candidate for a node is unhealthy, exhausted, or infeasible.

    The node is degraded rather than failing the run.
    """
