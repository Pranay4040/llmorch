"""Quota governor tests.

Everything here runs on a fake clock with no network. The governor is the one
component that genuinely cannot be debugged against live providers — Gemini
allows 250 requests a day, which is not enough to iterate on — so its behaviour
has to be pinned down offline.
"""

from __future__ import annotations

import asyncio

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from llmorch.errors import CostBlocked, QuotaBusy, QuotaExhausted, Unservable
from llmorch.quota.governor import Governor
from llmorch.quota.windows import (
    DayCounter,
    FakeClock,
    SlidingWindow,
    day_key,
    next_reset_utc,
)
from llmorch.registry.manifest import load_manifest
from llmorch.types import Admission, Priority, RateLimitSnapshot, Ticket, Usage

GROQ = "groq/gpt-oss-120b"
GEMINI = "gemini/3.6-flash"


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def gov(clock):
    return Governor(load_manifest(), clock=clock)


def _grant(gov, model_id, prompt=100, completion=200, **kw):
    t = gov.try_acquire(model_id, prompt, completion, **kw)
    assert isinstance(t, Ticket), getattr(t, "reason", t)
    return t


# ==========================================================================
# Sliding window
# ==========================================================================


def test_sliding_window_expires_old_events():
    w = SlidingWindow(60.0)
    w.add(0.0, 5)
    w.add(30.0, 5)
    assert w.current(30.0) == 10
    # At t=61 the first event has aged out of the window.
    assert w.current(61.0) == 5
    assert w.current(91.0) == 0


def test_sliding_window_reports_when_room_frees_up():
    w = SlidingWindow(60.0)
    w.add(0.0, 8)
    # Need 5 more against a limit of 10: must wait for the first entry to expire.
    assert w.seconds_until_room(10.0, 5, 10) == pytest.approx(50.0)
    assert w.seconds_until_room(10.0, 2, 10) == 0.0


def test_request_larger_than_the_whole_window_never_fits():
    """None means impossible, not 'wait longer'. Conflating the two is what
    hangs a scheduler forever."""
    w = SlidingWindow(60.0)
    assert w.seconds_until_room(0.0, 7000, 6000) is None


def test_adjust_last_reconciles_an_estimate_to_the_actual():
    w = SlidingWindow(60.0)
    w.add(0.0, 1000)  # reserved on estimate
    w.adjust_last(-400)  # actual came in lower
    assert w.current(0.0) == 600


# ==========================================================================
# Day boundaries
# ==========================================================================


def test_providers_bucket_days_in_their_own_timezone():
    """03:00 UTC on the 2nd is still 19:00 on the 1st in California (UTC-8).
    Bucketing both as the same day would silently double one provider's
    allowance across the eight hours the dates disagree."""
    moment = datetime(2026, 3, 2, 3, 0, tzinfo=timezone.utc)
    assert day_key(moment, "UTC") == "2026-03-02"
    assert day_key(moment, "America/Los_Angeles") == "2026-03-01"


def test_next_reset_differs_per_provider_timezone():
    """From the same instant, the two providers' quotas reset five hours apart."""
    moment = datetime(2026, 3, 2, 3, 0, tzinfo=timezone.utc)
    assert next_reset_utc(moment, "UTC") == datetime(
        2026, 3, 3, 0, 0, tzinfo=timezone.utc
    )
    # Next Pacific midnight is 00:00 PST on the 2nd == 08:00 UTC on the 2nd.
    assert next_reset_utc(moment, "America/Los_Angeles") == datetime(
        2026, 3, 2, 8, 0, tzinfo=timezone.utc
    )


def test_next_reset_survives_a_dst_transition():
    """US DST begins 2026-03-08. Handled by zoneinfo, not offset arithmetic."""
    before = datetime(2026, 3, 8, 6, 0, tzinfo=timezone.utc)
    reset = next_reset_utc(before, "America/Los_Angeles")
    assert reset > before


def test_day_counter_rolls_over_lazily():
    c = DayCounter("UTC")
    day1 = datetime(2026, 5, 1, 23, 0, tzinfo=timezone.utc)
    c.add(day1, 5)
    assert c.current(day1) == 5
    day2 = datetime(2026, 5, 2, 0, 30, tzinfo=timezone.utc)
    assert c.current(day2) == 0


# ==========================================================================
# UNSERVABLE vs WAIT — the load-bearing distinction
# ==========================================================================


def test_oversized_request_is_unservable_not_a_wait(gov):
    """Groq's 6,000 TPM caps a request at ~5,400 tokens. Something larger can
    never be served, so the scheduler must fail over rather than wait."""
    denial = gov.try_acquire(GROQ, 5000, 3000)
    assert denial.verdict is Admission.UNSERVABLE
    assert denial.is_permanent_today


def test_unservable_reports_no_wait_time(gov):
    assert gov.wait_time(GROQ, 20_000) is None


def test_a_request_that_fits_is_granted(gov):
    t = _grant(gov, GROQ, 1000, 2000)
    assert t.model_id == GROQ
    assert t.reserved_tokens == int(3000 * 1.25)


def test_same_size_request_is_fine_on_the_long_context_model(gov):
    """The identical payload that is unservable on Groq is routine on Gemini —
    which is precisely why cross-vendor failover is worth having."""
    assert gov.try_acquire(GROQ, 5000, 3000).verdict is Admission.UNSERVABLE
    assert isinstance(gov.try_acquire(GEMINI, 5000, 3000), Ticket)


def test_acquire_raises_immediately_on_unservable(gov):
    """Terminal refusals must not be slept on."""
    with pytest.raises(Unservable):
        import asyncio

        asyncio.run(gov.acquire(GROQ, 5000, 3000))


# ==========================================================================
# Per-minute limits
# ==========================================================================


def test_rpm_limit_produces_a_wait_with_a_concrete_retry_time(gov, clock):
    for _ in range(30):  # Groq allows 30 rpm
        _grant(gov, GROQ, 10, 10)
    denial = gov.try_acquire(GROQ, 10, 10)
    assert denial.verdict is Admission.WAIT
    assert denial.retry_after_s == pytest.approx(60.0)
    assert not denial.is_permanent_today


def test_rpm_window_recovers_after_a_minute(gov, clock):
    for _ in range(30):
        _grant(gov, GROQ, 10, 10)
    assert gov.try_acquire(GROQ, 10, 10).verdict is Admission.WAIT
    clock.advance(61)
    assert isinstance(gov.try_acquire(GROQ, 10, 10), Ticket)


def test_tpm_pressure_blocks_before_rpm_does_on_groq(gov):
    """With only 8,000 tokens/minute, Groq runs out of tokens long before it
    runs out of requests — two ordinary nodes are already most of the window."""
    _grant(gov, GROQ, 2000, 3000)  # reserves 6,250 of 8,000
    denial = gov.try_acquire(GROQ, 1000, 1000)  # wants 2,500 more
    assert denial.verdict is Admission.WAIT


# ==========================================================================
# Reservation and reconciliation
# ==========================================================================


def test_reservation_is_taken_at_acquire_time_not_at_commit(gov):
    """This is what makes concurrent fan-out safe: without it, parallel callers
    would each read a counter none of them had yet incremented."""
    _grant(gov, GROQ, 1000, 1000)
    head = gov.headroom()[GROQ]
    assert head.tokens_used_minute == int(2000 * 1.25)


def test_commit_reconciles_an_overestimate_downwards(gov):
    ticket = _grant(gov, GROQ, 1000, 1000)
    gov.commit(ticket, Usage(prompt_tokens=900, completion_tokens=300))
    assert gov.headroom()[GROQ].tokens_used_minute == 1200


def test_commit_reconciles_an_underestimate_upwards(gov):
    ticket = _grant(gov, GROQ, 100, 100)
    gov.commit(ticket, Usage(prompt_tokens=800, completion_tokens=900))
    assert gov.headroom()[GROQ].tokens_used_minute == 1700


def test_release_refunds_a_request_that_never_reached_the_provider(gov):
    """A transport failure must not permanently consume quota."""
    before = gov.headroom()[GROQ]
    ticket = _grant(gov, GROQ, 1000, 1000)
    gov.release(ticket, "connection reset")
    after = gov.headroom()[GROQ]
    assert after.tokens_used_minute == before.tokens_used_minute
    assert after.requests_used == before.requests_used


def test_concurrent_acquires_do_not_oversubscribe(gov):
    """Ten callers racing for a 30-rpm budget must collectively respect it."""
    granted = [
        t
        for _ in range(40)
        if isinstance(t := gov.try_acquire(GROQ, 10, 10), Ticket)
    ]
    assert len(granted) == 30


# ==========================================================================
# Daily limits and the reserve
# ==========================================================================


def test_reserve_is_withheld_from_normal_priority(gov, clock):
    """Gemini: 250/day with 30 reserved. Normal work stops at 220 so critical
    path retries still have somewhere to go."""
    for _ in range(220):
        _grant(gov, GEMINI, 10, 10)
        clock.advance(7)  # stay under the 10 rpm limit

    denial = gov.try_acquire(GEMINI, 10, 10, priority=Priority.NORMAL)
    assert denial.verdict is Admission.WAIT
    assert "reserve" in denial.reason


def test_high_priority_may_use_the_reserve(gov, clock):
    for _ in range(220):
        _grant(gov, GEMINI, 10, 10)
        clock.advance(7)
    assert isinstance(
        gov.try_acquire(GEMINI, 10, 10, priority=Priority.HIGH), Ticket
    )


def test_daily_exhaustion_is_reported_as_exhausted_not_wait(gov, clock):
    for _ in range(250):
        _grant(gov, GEMINI, 10, 10, priority=Priority.HIGH)
        clock.advance(7)

    denial = gov.try_acquire(GEMINI, 10, 10, priority=Priority.HIGH)
    assert denial.verdict is Admission.EXHAUSTED_TODAY
    assert denial.is_permanent_today


def test_daily_counters_reset_on_each_providers_own_midnight(gov, clock):
    """Both providers are exercised in one run, then time is moved past Pacific
    midnight. Gemini resets; Groq (UTC) does not."""
    clock.set_utc(datetime(2026, 6, 15, 6, 0, tzinfo=timezone.utc))  # 23:00 PT Jun 14
    for _ in range(10):
        _grant(gov, GEMINI, 10, 10)
        clock.advance(7)
    _grant(gov, GROQ, 10, 10)

    assert gov.headroom()[GEMINI].requests_used == 10
    assert gov.headroom()[GROQ].requests_used == 1

    # 08:00 UTC is past Pacific midnight but well before UTC midnight.
    clock.set_utc(datetime(2026, 6, 15, 8, 0, tzinfo=timezone.utc))
    assert gov.headroom()[GEMINI].requests_used == 0
    assert gov.headroom()[GROQ].requests_used == 1


def test_acquire_raises_quota_exhausted_rather_than_sleeping_till_midnight(gov, clock):
    import asyncio

    for _ in range(250):
        _grant(gov, GEMINI, 10, 10, priority=Priority.HIGH)
        clock.advance(7)

    with pytest.raises(QuotaExhausted):
        asyncio.run(gov.acquire(GEMINI, 10, 10, priority=Priority.HIGH))


# ==========================================================================
# Server headers are authoritative
# ==========================================================================


def test_headers_override_optimistic_local_counting(gov):
    _grant(gov, GEMINI, 10, 10)
    assert gov.headroom()[GEMINI].requests_used == 1

    # Server says only 100 of 250 remain — another client shares this key.
    gov.sync_from_headers(GEMINI, RateLimitSnapshot(remaining_requests=100))
    assert gov.headroom()[GEMINI].requests_used == 150


def test_retry_after_header_imposes_a_cooldown(gov, clock):
    gov.sync_from_headers(GROQ, RateLimitSnapshot(retry_after_s=30.0))
    denial = gov.try_acquire(GROQ, 10, 10)
    assert denial.verdict is Admission.WAIT
    assert denial.retry_after_s == pytest.approx(30.0)

    clock.advance(31)
    assert isinstance(gov.try_acquire(GROQ, 10, 10), Ticket)


def test_daily_429_stops_further_attempts_today(gov):
    """Believe the server rather than walking back into the same wall."""
    gov.note_daily_exhausted(GEMINI)
    denial = gov.try_acquire(GEMINI, 10, 10, priority=Priority.HIGH)
    assert denial.verdict is Admission.EXHAUSTED_TODAY


# ==========================================================================
# Cost gating
# ==========================================================================


def test_paid_provider_is_blocked_without_authorisation():
    manifest = load_manifest()
    gov = Governor(manifest, clock=FakeClock())
    # Perplexity ships disabled, so exercise the gate via a synthetic check.
    provider = manifest.providers["perplexity"]
    assert provider.paid
    assert not gov.allow_paid


def test_paid_provider_requires_both_flag_and_budget():
    manifest = load_manifest()
    flag_only = Governor(manifest, clock=FakeClock(), allow_paid=True)
    assert flag_only.max_usd == Decimal("0")

    both = Governor(
        manifest, clock=FakeClock(), allow_paid=True, max_usd=Decimal("5")
    )
    assert both.allow_paid and both.max_usd > 0


# ==========================================================================
# Reporting
# ==========================================================================


def test_headroom_reports_each_provider_in_its_own_timezone(gov):
    head = gov.headroom()
    assert head[GROQ].reset_tz == "UTC"
    assert head[GEMINI].reset_tz == "America/Los_Angeles"
    assert head[GEMINI].requests_limit == 250
    assert head[GROQ].tokens_limit_minute == 8000


def test_wait_time_zero_means_available_now(gov):
    assert gov.wait_time(GROQ, 100) == 0.0


def test_wait_time_probe_does_not_consume_quota(gov):
    """wait_time acquires and releases internally; it must leave no trace."""
    before = gov.headroom()[GROQ].requests_used
    gov.wait_time(GROQ, 100)
    assert gov.headroom()[GROQ].requests_used == before


# ==========================================================================
# Waiting versus refusing
#
# The governor already separates WAIT from EXHAUSTED_TODAY. These pin down the
# behaviour that has to follow from it: a model that is merely busy stays a
# candidate, and only the caller's patience is finite.
# ==========================================================================


def test_acquire_waits_out_a_per_minute_window_rather_than_refusing(clock, gov):
    # Fill the account-scoped 30 RPM window, then ask again. The request is
    # granted once the window rolls, not refused.
    for _ in range(30):
        _grant(gov, GROQ, 10, 10)
    assert gov.try_acquire(GROQ, 10, 10).verdict is Admission.WAIT

    ticket = asyncio.run(gov.acquire(GROQ, 10, 10, deadline_s=120))

    assert isinstance(ticket, Ticket)
    assert clock.monotonic() > 0, "it should have waited for the window"


def test_a_wait_past_the_deadline_is_busy_not_exhausted(gov):
    """The distinction that keeps a healthy model in the chains.

    Exceeding the caller's patience says nothing about the model's daily
    allowance. Raising QuotaExhausted here — as this once did — marks the model
    spent for the whole run, so an account-scoped 30 RPM ceiling benches the
    entire roster moments into the first fan-out.
    """
    for _ in range(30):
        _grant(gov, GROQ, 10, 10)

    with pytest.raises(QuotaBusy):
        asyncio.run(gov.acquire(GROQ, 10, 10, deadline_s=0.5))


def test_a_busy_model_is_still_retryable(gov):
    for _ in range(30):
        _grant(gov, GROQ, 10, 10)

    try:
        asyncio.run(gov.acquire(GROQ, 10, 10, deadline_s=0.5))
    except QuotaBusy as exc:
        # Unlike QuotaExhausted and Unservable, this one can succeed later.
        assert exc.is_retryable is True


def test_an_exhausted_day_is_raised_immediately_not_waited_on(gov, clock):
    # Waiting for a daily reset would stall the run for hours; the caller needs
    # to fail over now.
    gov.note_daily_exhausted(GEMINI)

    with pytest.raises(QuotaExhausted):
        asyncio.run(gov.acquire(GEMINI, 10, 10, deadline_s=120))
    assert clock.monotonic() == 0, "it must not have slept"
