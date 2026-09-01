"""Parse rate-limit information out of provider response headers.

The governor counts locally because it must decide *before* a request is sent.
But local counting is inference, and these headers are fact — the provider's own
view of what is left, delivered free with every response. Wherever the two
disagree, the headers win.

Two provider dialects matter here:

* **Groq** returns the full OpenAI-style set (`x-ratelimit-remaining-requests`,
  `-tokens`, and matching `-reset-*` durations written like `2m59.56s`). Its
  request bucket is the *daily* one, which is exactly what the governor's RPD
  counter tracks.
* **Gemini**'s OpenAI-compatible endpoint returns almost nothing. When a limit
  is hit, the fact lives in the 429 body rather than a header, so the body text
  is scanned as a fallback.

Nothing here raises. A missing or unparseable header is simply absent from the
snapshot, leaving the local estimate in place.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from ..types import RateLimitSnapshot

# Durations arrive as compound strings: "1s", "7.66s", "2m59.56s", "1h2m3s",
# "500ms", or a bare number of seconds.
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)?", re.IGNORECASE)

_UNIT_SECONDS = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "": 1.0,
}

# A per-minute bucket always resets within a minute. A "remaining: 0" carrying a
# reset far beyond that is therefore a day-scale bucket, and waiting it out is
# not something the scheduler should attempt.
_DAY_SCALE_RESET_S = 300.0

# Some providers state the wait in prose rather than a header — Gemini answers a
# 429 with "Please retry in 12.398389909s." and no rate-limit headers at all.
# That sentence is the most useful thing in the response: it is the server
# saying exactly when to come back.
_RETRY_IN_PROSE = re.compile(
    r"retry\s+(?:in|after)\s+(\d+(?:\.\d+)?)\s*(ms|s|m|h)?", re.IGNORECASE
)


def parse_duration(value: str | None) -> float | None:
    """Seconds from a provider duration string, or None if unreadable."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    total = 0.0
    matched = False
    for amount, unit in _DURATION_PART.findall(text):
        try:
            number = float(amount)
        except ValueError:  # pragma: no cover - regex guarantees a number
            continue
        total += number * _UNIT_SECONDS[unit.lower()]
        matched = True
    return total if matched else None


def _int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        # Some providers report token counts with thousands separators.
        return int(float(str(value).replace(",", "").strip()))
    except ValueError:
        return None


class _CaseInsensitive:
    """Header lookup that does not care about casing.

    `http.client` and `urllib` hand back differently-cased keys depending on the
    path taken, and a dict built from a test fixture is different again.
    """

    __slots__ = ("_items",)

    def __init__(self, headers: Mapping[str, str] | None) -> None:
        self._items = {str(k).lower(): v for k, v in (headers or {}).items()}

    def get(self, *names: str) -> str | None:
        for name in names:
            value = self._items.get(name.lower())
            if value not in (None, ""):
                return str(value)
        return None


def retry_after_from_body(text: str | None) -> float | None:
    """The wait a provider states in prose, when it states none in a header."""
    if not text:
        return None
    match = _RETRY_IN_PROSE.search(text)
    if not match:
        return None
    return float(match.group(1)) * _UNIT_SECONDS[(match.group(2) or "s").lower()]


def looks_like_daily_limit(text: str | None) -> bool:
    """Whether an error message describes a *daily* cap rather than a burst one.

    The distinction decides whether the scheduler may wait: a per-minute 429
    clears in seconds, whereas a daily one means this model is done until the
    provider's next local midnight. Gemini phrases it as a quota id such as
    `GenerateRequestsPerDayPerProjectPerModel`, Groq as "requests per day", so
    the comparison is made against a letters-only reduction of the message.
    """
    if not text:
        return False
    flat = re.sub(r"[^a-z]", "", text.lower())
    return any(
        marker in flat
        for marker in ("perday", "requestsperday", "dailylimit", "dailyquota", "rpd")
    )


def parse_rate_limit_headers(
    headers: Mapping[str, str] | None, *, body: str | None = None
) -> RateLimitSnapshot:
    """Build a snapshot from response headers, with the body as a fallback."""
    h = _CaseInsensitive(headers)

    remaining_requests = _int(
        h.get("x-ratelimit-remaining-requests", "x-ratelimit-remaining")
    )
    remaining_tokens = _int(h.get("x-ratelimit-remaining-tokens"))
    limit_requests = _int(h.get("x-ratelimit-limit-requests"))
    limit_tokens = _int(h.get("x-ratelimit-limit-tokens"))

    reset_requests_s = parse_duration(
        h.get("x-ratelimit-reset-requests", "x-ratelimit-reset")
    )
    reset_tokens_s = parse_duration(h.get("x-ratelimit-reset-tokens"))
    retry_after_s = parse_duration(h.get("retry-after", "x-ratelimit-retry-after"))

    if retry_after_s is None:
        retry_after_s = retry_after_from_body(body)

    daily = looks_like_daily_limit(body)

    # A stated wait outranks any keyword. Google lists several quota ids in one
    # 429 — some of them per-day — while the metric that actually tripped is
    # per-minute, so scanning the body for "per day" reads the wrong line. If
    # the server says come back in twelve seconds, waiting works, and by
    # definition this is not the daily wall.
    if daily and retry_after_s is not None and retry_after_s < _DAY_SCALE_RESET_S:
        daily = False

    if (
        not daily
        and remaining_requests == 0
        and reset_requests_s is not None
        and reset_requests_s > _DAY_SCALE_RESET_S
    ):
        # Remaining is zero on a bucket that will not reset for hours: this is
        # the daily wall wearing a per-request header.
        daily = True

    return RateLimitSnapshot(
        remaining_requests=remaining_requests,
        remaining_tokens=remaining_tokens,
        limit_requests=limit_requests,
        limit_tokens=limit_tokens,
        reset_requests_s=reset_requests_s,
        reset_tokens_s=reset_tokens_s,
        retry_after_s=retry_after_s,
        daily_limit_hit=daily,
    )
