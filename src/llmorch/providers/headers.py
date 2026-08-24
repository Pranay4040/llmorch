"""Parse rate-limit facts out of provider responses.

Local counting is inference; these headers are fact, and they cost nothing to
read. Every response — success or failure — is worth parsing, because the
cheapest moment to learn a provider's real limits is one it has already
answered.

Three things make this less trivial than it looks:

* **Durations are not seconds.** Groq answers `7.66s`, `2m59.56s`, `521ms`;
  `Retry-After` is either a plain integer or an HTTP date. All of them have to
  land in the same float.
* **A 429 does not say which wall was hit.** Per-minute and per-day exhaustion
  arrive with identical status codes and near-identical shapes, but the correct
  responses are opposites — wait a moment, versus stop until midnight. See
  `looks_daily`.
* **Absent is not zero.** Gemini's OpenAI-compatible endpoint sends no
  rate-limit headers at all. Every field is therefore `None` unless the server
  actually said something, so local counters stay in charge where they must.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from ..types import RateLimitSnapshot

# Anything resetting further out than this is a daily window, not a per-minute
# one. Per-minute resets are bounded by 60s, so an hour sits in the wide empty
# gap between the two and separates them unambiguously.
DAILY_RESET_THRESHOLD_S = 3600.0

_DURATION_PART = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s|m|h|d)")

# The whole string must be duration tokens and nothing else. Without the
# anchoring, an HTTP-date `Retry-After` ("Fri, 01 May 2026 11:00:00 GMT") parses
# as a duration — `01 May` reads as one minute — and a caller waits 60 seconds
# for a wall that is an hour away.
_DURATION_FULL = re.compile(r"(?:\d+(?:\.\d+)?\s*(?:ms|s|m|h|d)\s*)+")

_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}

# Phrases providers use when the wall is the *daily* one. Matched against the
# error body, which for several of them is the only place it is stated.
_DAILY_MARKERS = (
    "per day",
    "per-day",
    "daily",
    "rpd",
    "tpd",
    "quota exceeded",
    "quota_exceeded",
    "free_tier",
    "generate_content_free_tier",
)


def parse_duration(text: str | None) -> float | None:
    """Seconds from a provider duration string.

    Handles the compound form Groq uses (`2m59.56s`), single units (`7.66s`,
    `521ms`), and a bare number, which every provider means as seconds.
    Returns None for anything unrecognisable rather than guessing — a wrong
    duration is worse than no duration, because it would be trusted.
    """
    if text is None:
        return None
    raw = str(text).strip().lower()
    if not raw:
        return None

    try:
        return max(0.0, float(raw))
    except ValueError:
        pass

    if not _DURATION_FULL.fullmatch(raw):
        return None

    total = 0.0
    for part in _DURATION_PART.finditer(raw):
        total += float(part["value"]) * _UNIT_SECONDS[part["unit"]]
    return total


def parse_retry_after(text: str | None, *, now: datetime | None = None) -> float | None:
    """`Retry-After`, which RFC 9110 allows to be either seconds or a date."""
    seconds = parse_duration(text)
    if seconds is not None:
        return seconds
    if not text:
        return None
    try:
        when = parsedate_to_datetime(str(text))
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(timezone.utc)
    return max(0.0, (when - reference).total_seconds())


def _get(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive lookup; HTTP header casing is not guaranteed."""
    if not headers:
        return None
    direct = headers.get(name)
    if direct is not None:
        return direct
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def _int(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        return int(float(str(text).strip()))
    except (TypeError, ValueError):
        return None


def looks_daily(
    *,
    body: str = "",
    remaining_requests: int | None = None,
    reset_requests_s: float | None = None,
    reset_tokens_s: float | None = None,
    retry_after_s: float | None = None,
) -> bool:
    """Whether a refusal means "not until tomorrow" rather than "not this minute".

    Getting this backwards is expensive in both directions. Treating a daily cap
    as a per-minute one burns the rest of the run retrying into a wall that will
    not move for hours; treating a per-minute cap as daily benches a healthy
    model for the whole day over a two-second pause.

    Two independent signals, either sufficient:

    1. The body says so. Groq names the window in its error text, and Gemini's
       quota errors name the free-tier daily metric.
    2. The reset horizon is implausible for a minute window. Nothing per-minute
       resets an hour out, so a long reset with nothing remaining is a daily
       wall whatever the text says.
    """
    haystack = (body or "").lower()
    if any(marker in haystack for marker in _DAILY_MARKERS):
        return True

    horizons = [
        h for h in (reset_requests_s, reset_tokens_s, retry_after_s) if h is not None
    ]
    if not horizons:
        return False
    # An unstated remaining count is treated as exhausted: the server refused,
    # so something is at zero whether or not it said which.
    exhausted = remaining_requests is None or remaining_requests == 0
    return exhausted and max(horizons) > DAILY_RESET_THRESHOLD_S


def parse_rate_limit(
    headers: Mapping[str, str] | None,
    *,
    status: int = 200,
    body: str = "",
    now: datetime | None = None,
) -> RateLimitSnapshot:
    """Build a snapshot from one response's headers.

    Safe to call on every response. When a provider sends no rate-limit headers
    the result is empty, and `Governor.sync_from_headers` then changes nothing —
    which is the correct outcome, not a silent failure.
    """
    headers = headers or {}

    remaining_requests = _int(_get(headers, "x-ratelimit-remaining-requests"))
    remaining_tokens = _int(_get(headers, "x-ratelimit-remaining-tokens"))
    reset_requests_s = parse_duration(_get(headers, "x-ratelimit-reset-requests"))
    reset_tokens_s = parse_duration(_get(headers, "x-ratelimit-reset-tokens"))
    retry_after_s = parse_retry_after(_get(headers, "retry-after"), now=now)

    daily = False
    if status == 429:
        daily = looks_daily(
            body=body,
            remaining_requests=remaining_requests,
            reset_requests_s=reset_requests_s,
            reset_tokens_s=reset_tokens_s,
            retry_after_s=retry_after_s,
        )

    return RateLimitSnapshot(
        remaining_requests=remaining_requests,
        remaining_tokens=remaining_tokens,
        reset_requests_s=reset_requests_s,
        reset_tokens_s=reset_tokens_s,
        retry_after_s=retry_after_s,
        daily_limit_hit=daily,
    )


def observed_limits(headers: Mapping[str, str] | None) -> dict[str, int]:
    """Limits the provider states outright, for the ones that publish none.

    NVIDIA NIM and Mistral are marked `limits_are_estimated` in the manifest
    because their free tiers are undocumented. Where a response volunteers the
    real ceiling, that is worth more than any guess in the YAML.
    """
    headers = headers or {}
    out: dict[str, int] = {}
    for kind, header in (
        ("rpm", "x-ratelimit-limit-requests"),
        ("tpm", "x-ratelimit-limit-tokens"),
    ):
        value = _int(_get(headers, header))
        if value is not None and value > 0:
            out[kind] = value
    return out
