"""Ledger tests: persistence, day boundaries, and replay into the governor.

The ledger exists because quota belongs to the *account*, not to the process.
Everything here is about the seam where those two disagree — a second run an
hour later, a day that ends at Pacific midnight rather than UTC, a call that
failed after the provider had already counted it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from llmorch.doctor import FAIL, OK, WARN, offline_checks, probe_models
from llmorch.providers.openai_compat import HttpResponse
from llmorch.quota.governor import Governor
from llmorch.quota.store import (
    LedgerStore,
    cost_of,
    make_event,
    restore_governor,
)
from llmorch.quota.windows import FakeClock
from llmorch.registry.manifest import load_manifest
from llmorch.types import Admission, Ticket, Usage

GROQ = "groq/gpt-oss-120b"
GROQ_2 = "groq/qwen3-27b"
GEMINI = "gemini/3.6-flash"

# 03:00 UTC on 2 January is still 1 January in Los Angeles — the moment that
# separates Groq's quota day from Gemini's.
SPLIT_MOMENT = datetime(2026, 1, 2, 3, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


@pytest.fixture
def store(tmp_path):
    with LedgerStore(tmp_path / "state.db") as s:
        yield s


def _write(store, manifest, model_id, *, n=1, now=None, tokens=100, ok=True,
           status=200, purpose="execute", run_id="r1"):
    for i in range(n):
        store.record(
            make_event(
                run_id=run_id,
                node_id=f"n{i}",
                purpose=purpose,
                manifest=manifest,
                model_id=model_id,
                usage=Usage(prompt_tokens=tokens, completion_tokens=tokens),
                est_prompt_tokens=tokens,
                now=now or SPLIT_MOMENT,
                ok=ok,
                http_status=status,
                error=None if ok else "boom",
            )
        )


# ==========================================================================
# Storage
# ==========================================================================


def test_a_recorded_call_reads_back_intact(store, manifest):
    _write(store, manifest, GROQ, tokens=250)
    row = store.recent(1)[0]

    assert row.model_id == GROQ
    assert row.provider == "groq"
    assert row.prompt_tokens == 250
    assert row.ok is True


def test_the_ledger_survives_being_closed_and_reopened(tmp_path, manifest):
    path = tmp_path / "state.db"
    with LedgerStore(path) as first:
        _write(first, manifest, GROQ, n=3)
    with LedgerStore(path) as second:
        assert second.total_events == 3


def test_day_keys_use_each_provider_own_timezone(store, manifest):
    """Groq rolls over at UTC midnight, Gemini at Pacific. One shared date
    column would be wrong for one of them every single day."""
    _write(store, manifest, GROQ, now=SPLIT_MOMENT)
    _write(store, manifest, GEMINI, now=SPLIT_MOMENT)

    days = {row.model_id: row.day_key for row in store.recent(5)}
    assert days[GROQ] == "2026-01-02"
    assert days[GEMINI] == "2026-01-01"


def test_account_totals_span_every_model_on_the_provider(store, manifest):
    """Groq's request limit is org-scoped: a second model buys no extra quota,
    so the account total is the number that matters."""
    _write(store, manifest, GROQ, n=10)
    _write(store, manifest, GROQ_2, n=5)

    assert store.model_day_usage(GROQ, "2026-01-02").requests == 10
    assert store.provider_day_usage("groq", "2026-01-02").requests == 15


def test_a_429_counts_as_a_request_but_a_dead_socket_does_not(store, manifest):
    """A refused request has usually already consumed its slot. A call that
    never left this machine has not."""
    _write(store, manifest, GROQ, ok=False, status=429)
    _write(store, manifest, GROQ, ok=False, status=0)

    usage = store.model_day_usage(GROQ, "2026-01-02")
    assert usage.requests == 1
    assert usage.failures == 2


def test_day_table_groups_by_model_and_day(store, manifest):
    _write(store, manifest, GROQ, n=2)
    _write(store, manifest, GROQ, n=1, now=SPLIT_MOMENT.replace(day=3))

    table = store.day_table(days=5)
    assert {(r.day_key, r.requests) for r in table} == {
        ("2026-01-03", 1),
        ("2026-01-02", 2),
    }


def test_prune_keeps_the_recent_days_and_drops_the_rest(store, manifest):
    for day in (2, 3, 4, 5):
        _write(store, manifest, GROQ, now=SPLIT_MOMENT.replace(day=day))

    assert store.prune(keep_days=2) == 2
    assert {r.day_key for r in store.day_table(days=10)} == {"2026-01-05", "2026-01-04"}


def test_calibration_survives_the_process(tmp_path):
    """~20 live calls to converge, and every one of them costs a request. Far
    too expensive to relearn on each start."""
    with LedgerStore(tmp_path / "state.db") as store:
        store.save_calibration({"groq": {"ratio": 1.18, "samples": 24}})
    with LedgerStore(tmp_path / "state.db") as store:
        assert store.load_calibration()["groq"]["ratio"] == 1.18


def test_an_in_memory_store_never_touches_the_real_ledger():
    with LedgerStore(":memory:") as store:
        assert store.total_events == 0


# ==========================================================================
# Cost
# ==========================================================================


def test_free_providers_cost_nothing(manifest):
    assert cost_of(manifest.providers["groq"], Usage(1000, 1000)) == Decimal("0")


def test_the_per_request_fee_dominates_a_short_paid_call(manifest):
    """Perplexity charges ~$0.009 per *request* on top of per-token pricing.
    For the short calls this system makes, the fee is the whole cost."""
    cost = cost_of(manifest.providers["perplexity"], Usage(500, 500))
    assert cost > Decimal("0.009")
    assert cost < Decimal("0.011")


# ==========================================================================
# Replay into the governor
# ==========================================================================


def _clock_at(moment: datetime) -> FakeClock:
    clock = FakeClock()
    clock.set_utc(moment)
    return clock


def test_a_fresh_process_inherits_what_earlier_runs_spent(store, manifest):
    """Without this the second run of the day believes it holds the entire
    allowance — and finds out otherwise by spending a live request."""
    _write(store, manifest, GEMINI, n=40, now=SPLIT_MOMENT)

    governor = Governor(manifest, clock=_clock_at(SPLIT_MOMENT))
    restored = restore_governor(governor, store, manifest)

    assert restored[GEMINI].requests == 40
    assert governor.headroom()[GEMINI].requests_used == 40


def test_replaying_a_spent_day_makes_the_model_exhausted_not_merely_slow(
    store, manifest
):
    """EXHAUSTED_TODAY, not WAIT: the difference between failing over to another
    vendor and hanging until Pacific midnight."""
    _write(store, manifest, GEMINI, n=250, now=SPLIT_MOMENT)

    governor = Governor(manifest, clock=_clock_at(SPLIT_MOMENT))
    restore_governor(governor, store, manifest)

    denial = governor.try_acquire(GEMINI, 100, 100)
    assert not isinstance(denial, Ticket)
    assert denial.verdict is Admission.EXHAUSTED_TODAY
    assert denial.is_permanent_today


def test_yesterday_does_not_count_against_today(store, manifest):
    """The counter is keyed by the provider's own day, so a run that ended
    minutes before midnight leaves the new day clean."""
    _write(store, manifest, GEMINI, n=200, now=SPLIT_MOMENT)

    later = SPLIT_MOMENT.replace(day=3)
    governor = Governor(manifest, clock=_clock_at(later))
    assert restore_governor(governor, store, manifest) == {}
    assert isinstance(governor.try_acquire(GEMINI, 100, 100), Ticket)


def test_gemini_keeps_its_own_day_while_groq_has_already_rolled(store, manifest):
    """At 03:00 UTC the two providers are on different calendar days. Both
    counters must be right simultaneously."""
    _write(store, manifest, GROQ, n=5, now=SPLIT_MOMENT)
    _write(store, manifest, GEMINI, n=7, now=SPLIT_MOMENT)

    governor = Governor(manifest, clock=_clock_at(SPLIT_MOMENT))
    restored = restore_governor(governor, store, manifest)

    assert restored[GROQ].requests == 5
    assert restored[GEMINI].requests == 7


def test_restored_account_usage_is_shared_by_every_model_on_the_provider(
    store, manifest
):
    """Groq's request cap is org-scoped, so quota spent on one Groq model is
    quota gone for all of them. Restoring per-model only would hand back
    allowance that does not exist."""
    _write(store, manifest, GROQ, n=10)
    _write(store, manifest, GROQ_2, n=5)

    governor = Governor(manifest, clock=_clock_at(SPLIT_MOMENT))
    restore_governor(governor, store, manifest)

    assert governor.headroom()[GROQ].requests_used == 15
    assert governor.headroom()[GROQ_2].requests_used == 15


def test_restore_leaves_the_per_minute_windows_alone(store, manifest):
    """Sliding windows run on the monotonic clock, which has no meaning across
    processes — and they drain in under a minute anyway."""
    _write(store, manifest, GROQ, n=3, tokens=2000)

    governor = Governor(manifest, clock=_clock_at(SPLIT_MOMENT))
    restore_governor(governor, store, manifest)

    assert governor.headroom()[GROQ].tokens_used_minute == 0


# ==========================================================================
# Doctor
# ==========================================================================


def _no_keys(monkeypatch):
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(var, "")


def test_offline_checks_pass_on_a_correctly_set_up_machine(store, manifest, monkeypatch):
    _no_keys(monkeypatch)
    checks = offline_checks(manifest, store)

    failures = [c for c in checks if c.status == FAIL]
    assert not failures, [c.detail for c in failures]


def test_a_missing_key_warns_rather_than_fails(store, manifest, monkeypatch):
    """A keyless provider is dormant, not broken: the roster is deliberately
    partial while Milestone 2 runs Groq alone."""
    _no_keys(monkeypatch)
    checks = {c.name: c for c in offline_checks(manifest, store)}

    assert checks["key groq"].status == WARN
    assert "GROQ_API_KEY" in checks["key groq"].detail


def test_no_check_ever_prints_a_key_value(store, manifest, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-secret-value-123")
    text = " ".join(c.detail for c in offline_checks(manifest, store))
    assert "sk-secret-value-123" not in text


def test_the_pacific_timezone_resolves(store, manifest, monkeypatch):
    """Windows ships no timezone database. Without tzdata, Gemini's daily
    counter would never roll over at all."""
    _no_keys(monkeypatch)
    checks = {c.name: c for c in offline_checks(manifest, store)}
    assert checks["reset tz gemini"].status == OK
    assert "America/Los_Angeles" in checks["reset tz gemini"].detail


def test_doctor_reports_quota_already_spent_today(store, manifest, monkeypatch):
    _no_keys(monkeypatch)
    _write(store, manifest, GEMINI, n=12, now=datetime.now(timezone.utc))

    checks = {c.name: c for c in offline_checks(manifest, store)}
    assert checks["spent today"].status == WARN
    assert "12 req" in checks["spent today"].detail


def test_groq_prompt_headroom_is_flagged_as_tight(store, manifest, monkeypatch):
    """4,096 output tokens out of a ~7,200 ceiling leaves ~3,100 for the
    prompt. That is the constraint the whole design bends around, so it should
    be visible, not folklore."""
    _no_keys(monkeypatch)
    checks = {c.name: c for c in offline_checks(manifest, store)}
    detail = checks[f"ceiling {GROQ}"].detail
    assert "7200" in detail or "7,200" in detail


# ==========================================================================
# Doctor probe
# ==========================================================================


class _ProbeTransport:
    def __init__(self, statuses: dict[str, int] | None = None) -> None:
        self.statuses = statuses or {}
        self.calls: list[dict] = []

    async def post(self, url, *, headers, body, timeout_s):
        payload = json.loads(body.decode("utf-8"))
        self.calls.append(payload)
        status = self.statuses.get(payload["model"], 200)
        if status != 200:
            return HttpResponse(
                status,
                {},
                json.dumps({"error": {"message": "model does not exist"}}),
            )
        return HttpResponse(
            200,
            {"x-ratelimit-remaining-requests": "14399"},
            json.dumps(
                {
                    "model": payload["model"],
                    "choices": [{"finish_reason": "stop",
                                 "message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 2},
                }
            ),
        )


async def test_probe_confirms_each_wire_name_with_one_small_call(
    store, manifest, monkeypatch
):
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    monkeypatch.setenv("GEMINI_API_KEY", "")

    transport = _ProbeTransport()
    checks = await probe_models(
        manifest, providers={"groq"}, store=store, transport=transport
    )

    assert all(c.status == OK for c in checks), [c.detail for c in checks]
    # One call per Groq model, and each names the wire string, not our id.
    assert {c["model"] for c in transport.calls} == {
        m.wire_name for m in manifest.enabled_models if m.provider == "groq"
    }
    assert all(c["max_tokens"] <= 16 for c in transport.calls)


async def test_probe_names_the_wire_string_that_failed(store, manifest, monkeypatch):
    """The whole point: find the 404 now, not halfway through a run that has
    already spent its planning request."""
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    transport = _ProbeTransport(statuses={"qwen/qwen3.8-27b": 404})

    checks = await probe_models(
        manifest, providers={"groq"}, store=store, transport=transport
    )
    failed = [c for c in checks if c.status == FAIL]
    assert len(failed) == 1
    assert "qwen3.8-27b" in failed[0].detail


async def test_probe_calls_are_recorded_against_quota_like_any_other(
    store, manifest, monkeypatch
):
    """A diagnostic that bypasses the ledger would under-count the day it just
    spent requests in."""
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    before = store.total_events

    await probe_models(
        manifest, providers={"groq"}, store=store, transport=_ProbeTransport()
    )

    assert store.total_events > before
    assert all(r.purpose == "doctor" for r in store.recent(5))


async def test_probe_skips_a_provider_with_no_key(store, manifest, monkeypatch):
    _no_keys(monkeypatch)
    checks = await probe_models(manifest, store=store, transport=_ProbeTransport())
    assert [c.status for c in checks] == ["skip"]
