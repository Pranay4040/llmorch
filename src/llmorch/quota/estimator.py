"""Token estimation with self-calibration.

No tokenizer dependency. Every provider here uses a different tokenizer, so
shipping one (tiktoken, say) would be precise for a model none of them run and
misleading for the rest.

Instead: a cheap character-based estimate, corrected by a per-provider ratio
learned from actual usage. Every commit compares the estimate against the true
count and folds the error into an EWMA. After ~20 calls the estimate lands
within a few percent, at zero dependency and zero extra requests.

The estimate only ever needs to be good enough to reserve capacity safely; the
governor reconciles to the true count as soon as the response arrives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Characters per token before correction. English prose sits near 4.0; code and
# JSON tokenize denser, so this leans low deliberately — for admission control,
# over-estimating tokens is the safe direction.
BASE_CHARS_PER_TOKEN = 3.6

# EWMA responsiveness. 0.2 converges in roughly 20 samples while staying stable
# against a single outlier response.
ALPHA = 0.2

# Bounds on the learned correction. A ratio outside this range means something
# is structurally wrong (wrong model, mangled response) rather than a tokenizer
# difference, and clamping stops one bad sample from poisoning the estimator.
MIN_RATIO = 0.4
MAX_RATIO = 3.0


@dataclass(slots=True)
class Calibration:
    ratio: float = 1.0
    samples: int = 0

    def observe(self, actual: int, estimated: int) -> None:
        if estimated <= 0 or actual <= 0:
            return
        observed = max(MIN_RATIO, min(MAX_RATIO, actual / estimated))
        if self.samples == 0:
            self.ratio = observed
        else:
            self.ratio = (1 - ALPHA) * self.ratio + ALPHA * observed
        self.samples += 1

    @property
    def is_warmed_up(self) -> bool:
        """Whether enough samples exist for the ratio to be trustworthy."""
        return self.samples >= 5


@dataclass(slots=True)
class TokenEstimator:
    """Per-provider token estimation."""

    _calibrations: dict[str, Calibration] = field(default_factory=dict)

    def calibration(self, provider: str) -> Calibration:
        return self._calibrations.setdefault(provider, Calibration())

    def estimate_text(self, text: str, provider: str = "") -> int:
        """Estimate tokens in a string."""
        if not text:
            return 0
        raw = math.ceil(len(text) / BASE_CHARS_PER_TOKEN)
        if provider:
            raw = math.ceil(raw * self.calibration(provider).ratio)
        return max(1, raw)

    def estimate_prompt(
        self,
        *,
        system: str | None,
        messages: list[str],
        provider: str = "",
    ) -> int:
        """Estimate a full prompt, including per-message framing overhead.

        Chat formats wrap each message in role markers and separators that are
        invisible in the text but real on the wire; ~4 tokens per message is the
        usual allowance.
        """
        total = self.estimate_text(system or "", provider)
        for m in messages:
            total += self.estimate_text(m, provider) + 4
        return total + 3  # priming for the assistant turn

    def observe(self, provider: str, actual_prompt: int, estimated_prompt: int) -> None:
        """Fold a real measurement into the provider's correction ratio.

        Only prompt tokens are used: completion length is bounded by max_tokens
        and reflects what the model chose to say, so it carries no information
        about the tokenizer.
        """
        self.calibration(provider).observe(actual_prompt, estimated_prompt)

    def error_rate(self, provider: str) -> float:
        """How far the current ratio sits from 1.0 — reported by `llmorch report`
        as estimator calibration drift."""
        return abs(self.calibration(provider).ratio - 1.0)

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            provider: {"ratio": c.ratio, "samples": c.samples}
            for provider, c in self._calibrations.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, float]]) -> TokenEstimator:
        est = cls()
        for provider, values in (data or {}).items():
            est._calibrations[provider] = Calibration(
                ratio=float(values.get("ratio", 1.0)),
                samples=int(values.get("samples", 0)),
            )
        return est
