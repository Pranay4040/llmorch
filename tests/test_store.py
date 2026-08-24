"""The durable usage ledger, and the counters restored from it.

The property under test is the one a single run cannot demonstrate: that quota
spent by an earlier process is still spent when the next one starts. Every test
runs against a temporary database, never the real one at `state_db_path()` —
polluting the account's ledger from a test suite would make the governor
believe quota was spent that never was.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from llmorch.quota.governor import Governor
from llmorch.quota.store import LedgerStore, build_event, open_store
from llmorch.quota.windows import FakeClock
from llmorch.registry.manifest import load_manifest
from llmorch.types import LimitKind, LimitScope, Usage

GROQ = "groq/llama-3.3-70b"
QWEN = "groq/qwen3-32b"
GEMINI = "gemini/2.5-flash"

# Deliberately late in the UTC day and *still the previous day* in Pacific:
# 06:00 UTC on the 2nd is 23:00 on the 1st in Los Angeles. Any code that
# buckets both providers by one shared date gets this wrong.
STRADDLE = datetime(2026, 5, 2, 6, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    with open_store(tmp_path / "ledger.db") as ledger:
        yield ledger


@pytest.fixture
def manifest():
    return load_manifest()


def _event(model_id, provider, reset_tz, *, prompt=100, completion=200, **kw):
    return build_event(
        run_id=kw.pop("run_id", "run-1"),
        node_id=kw.pop("node_id", "n1"),
        purpose=kw.pop("purpose", "execute"),
        provider=provider,
        model_id=model_id,
        reset_tz=reset_tz,
        est_prompt_tokens=kw.pop("est_prompt", prompt),
        est_completion_tokens=kw.pop("est_completion", completion),
        usage=Usage(prompt_tokens=prompt, completion_tokens=completion),
        now=kw.pop("now", STRADDLE),
        **kw,
    )


# ==========================================================================
# Schema and round-tripping
# ==========================================================================


def test_store_creates_its_own_database_and_parents(tmp_path):
    path = tmp_path / "nested" / "deeper" / "state.db"
    with open_store(path):
        pass
    assert path.is_file()


def test_event_round_trips_intact(store):
    store.record(_event(GROQ, "groq", "UTC", prompt=11, completion=22))
    (event,) = store.events_for_run("run-1")

    assert event.model_id == GROQ
    assert event.usage.prompt_tokens == 11
    assert event.usage.completion_tokens == 22
    assert event.day_key == "2026-05-02"
    assert event.ok is True


def test_cost_survives_as_a_decimal_not_a_float(store):
    # Money through a float is how a ledger drifts. Stored as text, read back
    # as Decimal.
    store.record(_event(GROQ, "groq", "UTC", cost_usd=Decimal("0.009")))
    (event,) = store.events_for_run("run-1")
    assert event.cost_usd == Decimal("0.009")
    assert isinstance(event.cost_usd, Decimal)


def test_day_key_uses_each_providers_own_reset_timezone(store):
    store.record(_event(GROQ, "groq", "UTC"))
    store.record(_event(GEMINI, "gemini", "America/Los_Angeles"))

    groq_event, gemini_event = store.events_for_run("run-1")
    assert groq_event.day_key == "2026-05-02"
    # Same instant, still the 1st in Pacific — Gemini's quota day has not rolled.
    assert gemini_event.day_key == "2026-05-01"


# ==========================================================================
# Aggregation
# ==========================================================================


def test_day_usage_totals_per_model(store):
    for _ in range(3):
        store.record(_event(GROQ, "groq", "UTC", prompt=100, completion=200))
    store.record(_event(QWEN, "groq", "UTC", prompt=50, completion=50))

    by_model = {u.model_id: u for u in store.day_usage("groq", "2026-05-02")}

    assert by_model[GROQ].requests == 3
    assert by_model[GROQ].total_tokens == 900
    assert by_model[QWEN].requests == 1


def test_usage_from_another_day_is_not_counted(store):
    store.record(_event(GROQ, "groq", "UTC", now=datetime(2026, 5, 1, 6, 0, tzinfo=timezone.utc)))
    store.record(_event(GROQ, "groq", "UTC", now=STRADDLE))

    (today,) = store.day_usage("groq", "2026-05-02")
    assert today.requests == 1


def test_failed_requests_still_count_against_quota(store):
    # A 429 consumed a slot with the rate limiter even though it produced
    # nothing. Over-counting costs a little unused quota; under-counting means
    # hitting a wall with no warning.
    store.record(_event(GROQ, "groq", "UTC", ok=False, http_status=429, prompt=0, completion=0))
    (usage,) = store.day_usage("groq", "2026-05-02")
    assert usage.requests == 1


def test_usage_by_model_today_asks_each_provider_about_its_own_day(store):
    store.record(_event(GROQ, "groq", "UTC"))
    store.record(_event(GEMINI, "gemini", "America/Los_Angeles"))

    totals = store.usage_by_model_today(
        {"groq": "UTC", "gemini": "America/Los_Angeles"}, now=STRADDLE
    )

    assert totals[GROQ] == (1, 300)
    assert totals[GEMINI] == (1, 300)


def test_run_summaries_are_grouped_and_newest_first(store):
    store.record(_event(GROQ, "groq", "UTC", run_id="run-a"))
    store.record(_event(GROQ, "groq", "UTC", run_id="run-a"))
    store.record(
        _event(
            GROQ,
            "groq",
            "UTC",
            run_id="run-b",
            now=datetime(2026, 5, 3, 6, 0, tzinfo=timezone.utc),
            ok=False,
        )
    )

    runs = store.runs()
    assert [r.run_id for r in runs] == ["run-b", "run-a"]
    assert runs[1].requests == 2
    assert runs[0].failures == 1


def test_spent_usd_can_be_scoped_to_one_run(store):
    store.record(_event(GROQ, "groq", "UTC", run_id="run-a", cost_usd=Decimal("0.01")))
    store.record(_event(GROQ, "groq", "UTC", run_id="run-b", cost_usd=Decimal("0.02")))

    assert store.spent_usd() == pytest.approx(Decimal("0.03"))
    assert store.spent_usd(run_id="run-a") == pytest.approx(Decimal("0.01"))


def test_kv_round_trips_and_overwrites(store):
    assert store.get_kv("calibration") is None
    store.set_kv("calibration", '{"groq": 1.1}')
    store.set_kv("calibration", '{"groq": 1.2}')
    assert store.get_kv("calibration") == '{"groq": 1.2}'


def test_prune_keeps_recent_events(store):
    store.record(_event(GROQ, "groq", "UTC", now=datetime(2020, 1, 1, tzinfo=timezone.utc)))
    store.record(_event(GROQ, "groq", "UTC", now=datetime.now(timezone.utc)))

    assert store.prune(keep_days=90) == 1
    assert len(store.events_for_run("run-1")) == 1


def test_a_second_store_sees_the_first_ones_writes(tmp_path):
    # Two clones sharing one account must share one ledger, or both believe
    # they hold the full daily allowance.
    path = tmp_path / "shared.db"
    with open_store(path) as first:
        first.record(_event(GROQ, "groq", "UTC"))
    with open_store(path) as second:
        assert len(second.events_for_run("run-1")) == 1


# ==========================================================================
# Restoring the governor
# ==========================================================================


def test_a_fresh_governor_starts_empty_without_the_ledger(manifest):
    governor = Governor(manifest, clock=FakeClock())
    assert governor.headroom()[GEMINI].requests_used == 0


def test_yesterdays_spend_does_not_follow_into_today(store, manifest):
    clock = FakeClock(_now=STRADDLE)
    store.record(
        _event(GEMINI, "gemini", "America/Los_Angeles", now=datetime(2026, 4, 20, tzinfo=timezone.utc))
    )

    governor = Governor(manifest, clock=clock)
    governor.restore_daily(
        store.usage_by_model_today(governor.reset_timezones(), now=STRADDLE)
    )

    assert governor.headroom()[GEMINI].requests_used == 0


def test_todays_spend_is_restored_into_a_new_process(store, manifest):
    clock = FakeClock(_now=STRADDLE)
    for _ in range(240):
        store.record(_event(GEMINI, "gemini", "America/Los_Angeles", prompt=10, completion=10))

    governor = Governor(manifest, clock=clock)
    governor.restore_daily(
        store.usage_by_model_today(governor.reset_timezones(), now=STRADDLE)
    )

    assert governor.headroom()[GEMINI].requests_used == 240


def test_restored_counters_actually_refuse_admission(store, manifest):
    # The point of persisting at all: the second run must not rediscover
    # yesterday's wall by walking into it.
    clock = FakeClock(_now=STRADDLE)
    for _ in range(250):
        store.record(_event(GEMINI, "gemini", "America/Los_Angeles", prompt=10, completion=10))

    governor = Governor(manifest, clock=clock)
    governor.restore_daily(
        store.usage_by_model_today(governor.reset_timezones(), now=STRADDLE)
    )

    denial = governor.try_acquire(GEMINI, 100, 100)
    assert denial.verdict.value == "exhausted_today"
    assert denial.is_permanent_today


# ==========================================================================
# Limit scope
#
# Regression cover for a bug found while wiring the ledger: every scope bucket
# was given every counter, so a limit was enforced at whichever scope it was
# not declared at.
# ==========================================================================


def test_groq_rpd_is_per_model_not_shared_across_the_account(manifest):
    assert manifest.providers["groq"].limit(LimitKind.RPD).scope is LimitScope.MODEL

    governor = Governor(manifest, clock=FakeClock())
    for _ in range(5):
        governor.try_acquire(GROQ, 100, 200)

    headroom = governor.headroom()
    assert headroom[GROQ].requests_used == 5
    # A sibling model on the same provider has spent nothing of its own
    # 14,400/day allowance.
    assert headroom[QWEN].requests_used == 0


def test_groq_tpm_is_per_model_not_shared_across_the_account(manifest):
    # 6,000 TPM each, not 6,000 between them. Sharing one window would make
    # concurrent Groq nodes wrongly unservable.
    governor = Governor(manifest, clock=FakeClock())
    governor.try_acquire(GROQ, 1000, 1000)

    headroom = governor.headroom()
    assert headroom[GROQ].tokens_used_minute > 0
    assert headroom[QWEN].tokens_used_minute == 0


def test_groq_rpm_is_shared_across_the_account(manifest):
    # The other half of the same rule: RPM *is* account-scoped, so a sibling
    # model buys no extra requests per minute. 30 RPM, and the 31st waits
    # whichever model asks.
    governor = Governor(manifest, clock=FakeClock())
    for _ in range(30):
        governor.try_acquire(GROQ, 10, 10)

    denial = governor.try_acquire(QWEN, 10, 10)
    assert denial.verdict.value == "wait", getattr(denial, "reason", denial)


def test_restore_does_not_multiply_account_scoped_usage(store, manifest):
    # Applying per-model totals to a shared bucket once per model would count
    # the same requests three times and bench a barely-used provider.
    clock = FakeClock(_now=STRADDLE)
    for model in (GROQ, QWEN):
        for _ in range(10):
            store.record(_event(model, "groq", "UTC", prompt=10, completion=10))

    governor = Governor(manifest, clock=clock)
    governor.restore_daily(
        store.usage_by_model_today(governor.reset_timezones(), now=STRADDLE)
    )

    headroom = governor.headroom()
    assert headroom[GROQ].requests_used == 10
    assert headroom[QWEN].requests_used == 10


# ==========================================================================
# Cost
# ==========================================================================


def test_free_providers_cost_exactly_zero(manifest):
    assert manifest.providers["groq"].cost_for(10_000, 10_000) == Decimal("0")


def test_per_request_fee_dominates_a_short_paid_call(manifest):
    # Perplexity's $0.009/request is the larger half of the bill for calls this
    # size; a token-only cost model would under-report it severalfold.
    perplexity = manifest.providers["perplexity"]
    cost = perplexity.cost_for(1000, 1000)

    assert cost == Decimal("0.011")
    assert perplexity.cost.per_request / cost > Decimal("0.8")
