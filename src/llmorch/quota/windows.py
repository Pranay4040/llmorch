"""Time windows for rate limiting, and the clock they read.

The clock is injectable so tests can drive day rollovers and window expiry
deterministically instead of sleeping.

One rule runs through this module: **sliding windows use the monotonic clock,
day boundaries use the wall clock, and the two are never mixed.** Monotonic
time cannot jump backwards over an NTP correction or a DST change, which is
what a per-minute window needs. But it has no relationship to midnight, which
is what a daily quota resets on. Using wall time for sliding windows would let
a clock adjustment silently grant extra quota; using monotonic for day
boundaries would never reset at all.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol
from zoneinfo import ZoneInfo


class Clock(Protocol):
    """Time source. Real in production, controllable in tests.

    A clock owns the *passage* of time as well as its reading, which is why
    `sleep` lives here rather than being called directly. Anything that waits
    on a rate-limit window has to wait on the same clock it measures the window
    with: sleeping on the real clock while reading a fake one never advances
    toward the deadline, and the waiter spins forever.
    """

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin; never decreases."""
        ...

    def now_utc(self) -> datetime:
        """Timezone-aware wall clock in UTC."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Wait, advancing this clock by `seconds`."""
        ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


@dataclass(slots=True)
class FakeClock:
    """Test clock. Advancing it moves both time bases together."""

    _monotonic: float = 0.0
    _now: datetime = field(
        default_factory=lambda: datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    )

    def monotonic(self) -> float:
        return self._monotonic

    def now_utc(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._monotonic += seconds
        self._now += timedelta(seconds=seconds)

    async def sleep(self, seconds: float) -> None:
        """Advance instead of waiting.

        This is what makes a rate-limit wait testable: the governor's admission
        loop sleeps and re-checks, and here that loop converges immediately
        while still exercising every branch it would take in production.
        """
        self.advance(seconds)

    def set_utc(self, moment: datetime) -> None:
        """Jump wall time without touching monotonic time.

        Models an NTP correction or a laptop resuming from sleep — the case
        where the two clocks legitimately disagree.
        """
        if moment.tzinfo is None:
            raise ValueError("set_utc requires an aware datetime")
        self._now = moment.astimezone(timezone.utc)


# --------------------------------------------------------------------------
# Sliding windows (monotonic)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class SlidingWindow:
    """Rolling count of events within `span` seconds.

    Used for both requests-per-minute (each event weighs 1) and
    tokens-per-minute (each event weighs its token count). A running sum is
    maintained incrementally so admission is O(expired) rather than O(n).
    """

    span: float = 60.0
    _events: deque[tuple[float, int]] = field(default_factory=deque)
    _total: int = 0

    def _expire(self, now: float) -> None:
        cutoff = now - self.span
        while self._events and self._events[0][0] <= cutoff:
            self._total -= self._events.popleft()[1]

    def current(self, now: float) -> int:
        self._expire(now)
        return self._total

    def add(self, now: float, weight: int = 1) -> None:
        self._expire(now)
        self._events.append((now, weight))
        self._total += weight

    def adjust_last(self, delta: int) -> None:
        """Correct the most recent entry in place.

        Tokens are reserved on an estimate before the call and reconciled to the
        true count afterwards; this applies that correction without disturbing
        the entry's timestamp.
        """
        if not self._events:
            return
        ts, weight = self._events[-1]
        new_weight = max(0, weight + delta)
        self._events[-1] = (ts, new_weight)
        self._total += new_weight - weight

    def remove(self, weight: int) -> None:
        """Refund a reservation for a request that never reached the provider."""
        if not self._events:
            return
        for i in range(len(self._events) - 1, -1, -1):
            ts, w = self._events[i]
            if w == weight:
                del self._events[i]
                self._total -= w
                return
        # No exact match (already reconciled): subtract from the newest entry.
        self.adjust_last(-weight)

    def seconds_until_room(self, now: float, needed: int, limit: int) -> float | None:
        """How long until `needed` more units fit under `limit`.

        Returns 0.0 if there is room now, or None if `needed` exceeds `limit`
        outright — a request larger than the whole window can never fit, and
        waiting for it would hang the scheduler forever.
        """
        if needed > limit:
            return None
        self._expire(now)
        if self._total + needed <= limit:
            return 0.0

        # Expire entries oldest-first until enough capacity is freed.
        freed = 0
        target = self._total + needed - limit
        for ts, weight in self._events:
            freed += weight
            if freed >= target:
                return max(0.0, ts + self.span - now)
        return self.span

    def reset(self) -> None:
        self._events.clear()
        self._total = 0


# --------------------------------------------------------------------------
# Day counters (wall clock)
# --------------------------------------------------------------------------


def day_key(moment: datetime, tz_name: str) -> str:
    """Calendar date in the provider's own reset timezone.

    Providers do not agree on when a day ends: Groq rolls over at UTC midnight,
    Gemini at Pacific midnight. Bucketing by the provider's own timezone is the
    only way both stay correct in a single run.
    """
    return moment.astimezone(ZoneInfo(tz_name)).strftime("%Y-%m-%d")


def next_reset_utc(moment: datetime, tz_name: str) -> datetime:
    """The next midnight in `tz_name`, expressed in UTC.

    Computed by advancing the local date and re-localising, so DST transitions
    are handled by zoneinfo rather than by arithmetic on a fixed offset.
    """
    tz = ZoneInfo(tz_name)
    local = moment.astimezone(tz)
    tomorrow = (local + timedelta(days=1)).date()
    midnight = datetime(
        tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=tz
    )
    return midnight.astimezone(timezone.utc)


@dataclass(slots=True)
class DayCounter:
    """Usage within one provider-local calendar day.

    Rolls over lazily: the day key is recomputed on each access, so a counter
    that has been idle across midnight reports the new day correctly without
    needing a background task.
    """

    tz_name: str = "UTC"
    _day: str | None = None
    _count: int = 0

    def _roll(self, now: datetime) -> None:
        today = day_key(now, self.tz_name)
        if self._day != today:
            self._day = today
            self._count = 0

    def current(self, now: datetime) -> int:
        self._roll(now)
        return self._count

    def add(self, now: datetime, weight: int = 1) -> None:
        self._roll(now)
        self._count += weight

    def remove(self, weight: int = 1) -> None:
        self._count = max(0, self._count - weight)

    def seconds_until_reset(self, now: datetime) -> float:
        return max(0.0, (next_reset_utc(now, self.tz_name) - now).total_seconds())

    def set_count(self, now: datetime, value: int) -> None:
        """Overwrite from an authoritative source.

        Response headers are ground truth; local counting is only an estimate
        that fills the gaps between them.
        """
        self._roll(now)
        self._count = max(0, value)

    @property
    def day(self) -> str | None:
        return self._day
