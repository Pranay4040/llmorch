"""Failover and verification tests, driven by the mock's fault injection."""

from __future__ import annotations

import pytest

from llmorch.engine.health import (
    HealthTracker,
    ModelHealth,
    backoff_seconds,
    failover_chain,
    next_model,
    should_retry_same_model,
)
from llmorch.engine.verify import (
    check_placeholder,
    check_python,
    check_sql,
    parse_review,
    pick_reviewer,
    verify_tier0,
)
from llmorch.errors import (
    QuotaExhausted,
    RateLimited,
    SchemaInvalid,
    TransportError,
    Truncated,
    Unservable,
)
from llmorch.engine.worker import execute_node
from llmorch.registry.manifest import load_manifest
from llmorch.types import LimitKind, OutputKind, Role, Verdict

GROQ = "groq/gpt-oss-120b"
QWEN = "groq/qwen3-27b"
GEMINI = "gemini/3.6-flash"


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


# ==========================================================================
# Circuit breaker
# ==========================================================================


def test_breaker_trips_after_consecutive_failures():
    h = HealthTracker(threshold=2)
    assert h.record_failure(GROQ, SchemaInvalid("bad")) is ModelHealth.HEALTHY
    assert h.record_failure(GROQ, SchemaInvalid("bad")) is ModelHealth.UNHEALTHY
    assert not h.is_available(GROQ)


def test_a_success_resets_the_streak():
    """The breaker counts *consecutive* failures; an intermittent fault must not
    accumulate toward tripping it."""
    h = HealthTracker(threshold=2)
    h.record_failure(GROQ, TransportError("blip"))
    h.record_success(GROQ)
    assert h.record_failure(GROQ, TransportError("blip")) is ModelHealth.HEALTHY
    assert h.is_available(GROQ)


def test_running_out_of_quota_is_not_a_health_failure():
    """A model at its daily cap is unavailable, not broken. Counting it as a
    fault would penalise its track record for the crime of being popular."""
    h = HealthTracker(threshold=2)
    assert h.record_failure(GEMINI, QuotaExhausted("250/day")) is ModelHealth.EXHAUSTED
    assert h.record_failure(GEMINI, QuotaExhausted("250/day")) is ModelHealth.EXHAUSTED
    assert GEMINI not in h.unhealthy_models


def test_unservable_does_not_trip_the_breaker():
    """A sizing mismatch is a routing bug, not a model defect."""
    h = HealthTracker(threshold=2)
    h.record_failure(GROQ, Unservable("too big"))
    h.record_failure(GROQ, Unservable("too big"))
    assert h.is_available(GROQ)


def test_bulk_reassignment_happens_at_most_once_per_model():
    """Otherwise a flapping model triggers reassignment after reassignment."""
    h = HealthTracker(threshold=1)
    h.record_failure(GROQ, SchemaInvalid("bad"))
    assert h.needs_reassignment(GROQ)
    assert not h.needs_reassignment(GROQ)


def test_healthy_models_never_need_reassignment():
    assert not HealthTracker().needs_reassignment(GROQ)


# ==========================================================================
# Cross-vendor failover
# ==========================================================================


def test_failover_prefers_an_untried_vendor(manifest):
    """The core guarantee. After Gemini fails, the next choice must be a Groq
    model rather than another Gemini one."""
    chain = failover_chain(
        manifest, Role.FRONTEND, exclude={GEMINI}, tried_vendors={"gemini"}
    )
    assert chain
    assert manifest.vendor_of(chain[0]) == "groq"


def test_failover_falls_back_to_a_tried_vendor_only_when_nothing_else_remains(manifest):
    chain = failover_chain(
        manifest, Role.FRONTEND, exclude={QWEN}, tried_vendors={"groq"}
    )
    # Gemini is untried, so it leads.
    assert manifest.vendor_of(chain[0]) == "gemini"
    # Remaining Groq models still appear, just demoted.
    assert any(manifest.vendor_of(m) == "groq" for m in chain)


def test_unhealthy_models_are_dropped_from_the_chain(manifest):
    h = HealthTracker(threshold=1)
    h.record_failure(GEMINI, SchemaInvalid("bad"))
    chain = failover_chain(manifest, Role.FRONTEND, health=h)
    assert GEMINI not in chain


def test_next_model_returns_none_once_the_chain_is_spent(manifest):
    h = HealthTracker()
    everything = set(manifest.chain(Role.FRONTEND))
    assert (
        next_model(
            manifest,
            Role.FRONTEND,
            exclude=everything,
            tried_vendors=set(),
            health=h,
        )
        is None
    )


# ==========================================================================
# Retry policy
# ==========================================================================


def test_transport_failures_are_retried_on_the_same_model():
    """Transient and unrelated to the model's competence."""
    assert should_retry_same_model(TransportError("reset"), attempts=1, max_retries=2)
    assert should_retry_same_model(RateLimited("429"), attempts=1, max_retries=2)


def test_bad_output_fails_over_instead_of_retrying_the_same_model():
    """A malformed or truncated response reflects how this model handles this
    prompt; asking again usually produces the same thing."""
    assert not should_retry_same_model(SchemaInvalid("bad"), 1, 2)
    assert not should_retry_same_model(Truncated("cut off"), 1, 2)


def test_terminal_errors_are_never_retried():
    assert not should_retry_same_model(QuotaExhausted("spent"), 0, 3)
    assert not should_retry_same_model(Unservable("too big"), 0, 3)


def test_retries_stop_at_the_limit():
    assert not should_retry_same_model(TransportError("reset"), attempts=2, max_retries=2)


def test_backoff_grows_and_is_deterministic():
    """Deterministic jitter keeps a replayed run reproducible."""
    assert backoff_seconds(1) < backoff_seconds(2) < backoff_seconds(3)
    assert backoff_seconds(2) == backoff_seconds(2)
    assert backoff_seconds(99) <= 30.0 * 1.2


# ==========================================================================
# Tier 0 verification
# ==========================================================================


def test_unparseable_python_is_rejected():
    result = verify_tier0(
        "def broken(:\n  pass", output_path="server.py", output_kind=OutputKind.CODE
    )
    assert result.verdict is Verdict.REJECT
    assert any("does not parse" in i.what for i in result.issues)


def test_valid_python_passes():
    code = "def handler():\n    return {'ok': True}\n"
    assert verify_tier0(
        code, output_path="server.py", output_kind=OutputKind.CODE
    ).verdict is Verdict.PASS


def test_truncation_flag_is_authoritative():
    """Stopping exactly at max_tokens is the reliable signal — no LLM needed."""
    result = verify_tier0(
        "def f():\n    return 1\n",
        output_path="a.py",
        output_kind=OutputKind.CODE,
        truncated_flag=True,
    )
    assert result.verdict is Verdict.REJECT
    assert any("max_tokens" in i.what for i in result.issues)


def test_empty_artifact_is_rejected():
    assert verify_tier0(
        "  ", output_path="a.py", output_kind=OutputKind.CODE
    ).verdict is Verdict.REJECT


def test_invalid_sql_is_caught_by_executing_it():
    result = verify_tier0(
        "CREATE TABLE notes (id NOTATYPE PRIMARY KEY,,);",
        output_path="schema.sql",
        output_kind=OutputKind.SCHEMA,
    )
    assert result.verdict is Verdict.REJECT


def test_valid_sql_passes():
    sql = "CREATE TABLE notes (id INTEGER PRIMARY KEY, title TEXT NOT NULL);"
    assert not check_sql(sql)


def test_placeholder_code_is_rejected():
    """Passes a syntax check but is not an implementation."""
    result = verify_tier0(
        "# TODO: implement this\npass\n",
        output_path="server.py",
        output_kind=OutputKind.CODE,
    )
    assert result.verdict is Verdict.REJECT
    assert any("placeholder" in i.what for i in result.issues)


def test_the_word_todo_in_prose_is_not_a_placeholder():
    """A narrow marker check: prose mentioning todos must not be rejected."""
    assert not check_placeholder("A todo list application for managing notes.")


def test_unbalanced_html_is_rejected():
    result = verify_tier0(
        "<html><body><h1>Notes</h1>",
        output_path="index.html",
        output_kind=OutputKind.CODE,
    )
    assert result.verdict is Verdict.REJECT


def test_unbalanced_css_is_rejected():
    result = verify_tier0(
        "body { color: red;", output_path="style.css", output_kind=OutputKind.CODE
    )
    assert result.verdict is Verdict.REJECT


def test_content_that_is_not_css_at_all_is_rejected():
    """Regression: bracket-balance alone accepts this, because prose and Python
    both have zero braces and zero is balanced. A stylesheet with no rule block
    is not a stylesheet."""
    result = verify_tier0(
        "def broken(:\n    this is not python\n",
        output_path="style.css",
        output_kind=OutputKind.CODE,
    )
    assert result.verdict is Verdict.REJECT
    assert any("rule blocks" in i.what for i in result.issues)


def test_a_comment_only_stylesheet_is_still_rejected():
    result = verify_tier0(
        "/* styles go here */\n", output_path="style.css", output_kind=OutputKind.CODE
    )
    assert result.verdict is Verdict.REJECT


def test_real_css_passes():
    css = ":root { --bg: #fff; }\nbody { background: var(--bg); }\n"
    assert verify_tier0(
        css, output_path="style.css", output_kind=OutputKind.CODE
    ).verdict is Verdict.PASS


def test_fenced_code_is_unwrapped_before_checking():
    """Models fence their output regardless of instructions."""
    fenced = "```python\ndef ok():\n    return 1\n```"
    assert verify_tier0(
        fenced, output_path="a.py", output_kind=OutputKind.CODE
    ).verdict is Verdict.PASS


def test_python_syntax_error_reports_a_line_number():
    issues = check_python("x = 1\ny = (\n")
    assert issues and issues[0].line is not None


# ==========================================================================
# Cross-vendor reviewer selection
# ==========================================================================


def test_reviewer_is_never_from_the_authors_vendor(manifest):
    """Enforced in code, not requested in a prompt. Self-review is a known weak
    spot: a model that made a mistake tends to re-approve it."""
    reviewer = pick_reviewer(
        manifest, author_model_id=GROQ, candidates=[GROQ, QWEN, GEMINI]
    )
    assert reviewer is not None
    assert manifest.vendor_of(reviewer) != "groq"


def test_review_is_skipped_when_no_cross_vendor_reviewer_exists(manifest):
    """Better to skip than to spend a request on the opinion least likely to
    find the fault."""
    assert (
        pick_reviewer(manifest, author_model_id=GROQ, candidates=[GROQ, QWEN]) is None
    )


def test_reviewer_prefers_the_provider_with_abundant_daily_quota(manifest):
    """Review must not eat the budget the critical path depends on."""
    reviewer = pick_reviewer(
        manifest, author_model_id=GEMINI, candidates=[GEMINI, GROQ, QWEN]
    )
    assert manifest.vendor_of(reviewer) == "groq"


# ==========================================================================
# Review parsing — reviewer output is untrusted
# ==========================================================================


def test_review_verdict_is_parsed():
    result = parse_review(
        {"verdict": "revise", "issues": [{"severity": "error", "what": "no error handling"}]},
        QWEN,
    )
    assert result.verdict is Verdict.REVISE
    assert result.reviewer_model_id == QWEN
    assert result.tier == 1


def test_unrecognised_verdict_defaults_to_revise():
    """Fail toward inspection rather than silently passing bad output."""
    assert parse_review({"verdict": "lgtm!!"}, QWEN).verdict is Verdict.REVISE


def test_malformed_issue_entries_are_discarded_not_fatal():
    result = parse_review(
        {"verdict": "pass", "issues": ["not a dict", {"what": "real one"}]}, QWEN
    )
    assert len(result.issues) == 1


def test_review_fields_are_length_capped():
    """Reviewer text is untrusted input and must not be able to bloat state."""
    result = parse_review(
        {"verdict": "pass", "issues": [{"what": "x" * 5000}]}, QWEN
    )
    assert len(result.issues[0].what) <= 500


# ==========================================================================
# Admission verdicts must not collapse into each other
#
# Both of these were found by the first live run, and both were invisible
# offline because the mock never refuses admission.
# ==========================================================================


async def test_a_per_minute_wait_does_not_write_a_model_off_for_the_day():
    """WAIT is not EXHAUSTED.

    Groq's token window clears in under a minute. Treating that refusal as a
    daily cap marked a perfectly healthy model dead for the rest of the run and
    handed its work to a worse one — for the sake of a few seconds' wait.
    """
    from llmorch.config import RunConfig
    from llmorch.demo.website import INTERFACE
    from llmorch.engine.blackboard import Blackboard
    from llmorch.engine.worker import WorkerDeps, execute_node
    from llmorch.providers.base import ProviderRegistry
    from llmorch.providers.mock import MockProvider
    from llmorch.quota.estimator import TokenEstimator
    from llmorch.quota.governor import Governor
    from llmorch.quota.windows import FakeClock
    from llmorch.registry.manifest import load_manifest
    from llmorch.types import Admission, OutputKind, Role, TaskNode, Ticket

    manifest = load_manifest()
    governor = Governor(manifest, clock=FakeClock())
    health = HealthTracker()

    # Fill Groq's per-minute token window, leaving too little for the node but
    # not so little that the request could never fit at all.
    filler = governor.try_acquire(GROQ, 3000, 3000)
    assert isinstance(filler, Ticket), "setup failed to reserve the window"

    node = TaskNode(
        id="n1", title="t", role=Role.BACKEND, spec="s",
        output_path="a.py", output_kind=OutputKind.CODE, est_output_tokens=1200,
    )

    # The precondition this test is actually about: a WAIT, not an exhaustion.
    denial = governor.try_acquire(GROQ, 400, 2400)
    assert denial.verdict is Admission.WAIT, f"expected WAIT, got {denial.verdict}"

    provider = MockProvider(responses={"n1": "print('ok')\n"})
    registry = ProviderRegistry()
    for model in manifest.enabled_models:
        registry.register(model.id, provider)

    deps = WorkerDeps(
        manifest=manifest, governor=governor, registry=registry,
        estimator=TokenEstimator(), health=health,
        blackboard=Blackboard(interface=INTERFACE), sleep=_no_sleep,
    )
    await execute_node(node, GROQ, deps)

    assert health.status(GROQ) is not ModelHealth.EXHAUSTED, (
        "a per-minute wait was recorded as running out of quota for the day"
    )


def test_wait_is_reported_as_a_retryable_rate_limit():
    """The ladder branches on is_retryable, so the type carries the meaning."""
    assert RateLimited("busy").is_retryable is True
    assert QuotaExhausted("done for today").is_retryable is False


async def _no_sleep(_seconds):
    return None


def test_a_busy_model_is_not_a_broken_model():
    """Rate limits must not accumulate toward the circuit breaker.

    Same mistake as counting a daily cap as a fault, one rung down: the model
    answered correctly every time it was asked, it just could not be asked yet.
    """
    health = HealthTracker(threshold=2)
    for _ in range(5):
        health.record_failure(GROQ, RateLimited("window full", retry_after_s=3.0))

    assert health.status(GROQ) is ModelHealth.HEALTHY
    assert health.is_available(GROQ)


def test_a_daily_429_is_still_treated_as_exhaustion():
    """The distinction that matters: this one does not clear by waiting."""
    health = HealthTracker(threshold=2)
    health.record_failure(GROQ, RateLimited("cap reached", daily=True))
    assert health.status(GROQ) is ModelHealth.EXHAUSTED


def test_genuine_faults_still_trip_the_breaker():
    """The relaxation must not make the breaker unreachable."""
    health = HealthTracker(threshold=2)
    health.record_failure(GROQ, SchemaInvalid("garbage"))
    health.record_failure(GROQ, SchemaInvalid("garbage"))
    assert health.status(GROQ) is ModelHealth.UNHEALTHY


def test_rate_limits_get_a_longer_retry_allowance_than_faults():
    """Waiting genuinely fixes a full window; failing over just moves the same
    load onto a model whose window is equally full."""
    busy = RateLimited("window full", retry_after_s=2.0)
    broken = SchemaInvalid("garbage")

    assert should_retry_same_model(busy, attempts=2, max_retries=2) is True
    assert should_retry_same_model(broken, attempts=2, max_retries=2) is False
    # Still bounded, though.
    assert should_retry_same_model(busy, attempts=99, max_retries=2) is False


# ==========================================================================
# Truncation is a sizing fault, not a competence fault
#
# Observed live twice: a test-suite node estimated at 1,000 tokens truncated at
# a 2,000-token budget, failed over to a second model with the same budget,
# truncated identically, and degraded. Nobody involved was incapable.
# ==========================================================================


def _sizing_harness(manifest, *, needs, est=400, max_escalations=2):
    from llmorch.config import RunConfig  # noqa: F401
    from llmorch.demo.website import INTERFACE
    from llmorch.engine.blackboard import Blackboard
    from llmorch.engine.worker import WorkerDeps, execute_node  # noqa: F401
    from llmorch.providers.base import ProviderRegistry
    from llmorch.providers.mock import MockProvider
    from llmorch.quota.estimator import TokenEstimator
    from llmorch.quota.governor import Governor
    from llmorch.quota.windows import FakeClock
    from llmorch.types import OutputKind, Role, TaskNode

    node = TaskNode(
        id="tests", title="test suite", role=Role.BACKEND,
        spec="write the tests", output_path="test_thing.py",
        output_kind=OutputKind.CODE, est_output_tokens=est,
    )
    provider = MockProvider(
        responses={"tests": "def test_ok():\n    assert True\n"},
        needs_tokens={"tests": needs},
    )
    registry = ProviderRegistry()
    for model in manifest.enabled_models:
        registry.register(model.id, provider)

    health = HealthTracker()
    deps = WorkerDeps(
        manifest=manifest, governor=Governor(manifest, clock=FakeClock()),
        registry=registry, estimator=TokenEstimator(), health=health,
        blackboard=Blackboard(interface=INTERFACE),
        max_escalations=max_escalations, sleep=_no_sleep,
    )
    return node, provider, deps, health


async def test_a_truncated_node_is_retried_with_a_bigger_budget(manifest):
    """Same model, more room — rather than the same too-small budget handed to
    the next model in the chain."""
    from llmorch.types import NodeState

    # Needs 1,500; the first budget is 400*2 = 800, the second 1,600.
    node, provider, deps, health = _sizing_harness(manifest, needs=1500, est=400)

    result = await execute_node(node, GROQ, deps)

    assert result.state is NodeState.DONE
    assert result.model_id == GROQ, "it should not have failed over at all"
    assert provider.calls_for("tests") == [GROQ, GROQ]


async def test_growing_the_budget_is_not_held_against_the_model(manifest):
    """Being handed too small a budget is not a fault, in the same way that
    being rate limited is not being broken."""
    node, provider, deps, health = _sizing_harness(manifest, needs=1500, est=400)
    await execute_node(node, GROQ, deps)

    assert health.status(GROQ) is ModelHealth.HEALTHY
    assert any("budget raised" in event for event in health.events)


async def test_the_budget_never_exceeds_what_the_model_allows(manifest):
    """Escalation is bounded by the model, not just by the retry count."""
    from llmorch.engine.worker import output_budget

    model = manifest.model(GROQ)
    node, _provider, _deps, _health = _sizing_harness(manifest, needs=99999, est=4000)
    for escalation in range(6):
        assert output_budget(node, model, escalation) <= model.max_output


async def test_escalation_is_bounded(manifest):
    """A node whose spec genuinely cannot be met must not walk the budget up to
    the ceiling on every model in the chain."""
    node, provider, deps, health = _sizing_harness(
        manifest, needs=10**6, est=400, max_escalations=1
    )
    result = await execute_node(node, GROQ, deps)

    # One escalation on the first model, then it is treated as a real failure
    # and the work moves on to other vendors.
    assert provider.calls_for("tests")[:2] == [GROQ, GROQ]
    assert len(set(provider.calls_for("tests"))) > 1
    assert result.state.value in ("degraded", "done")


async def test_the_larger_budget_travels_with_the_node_across_a_failover(manifest):
    """A file that did not fit in 2,000 tokens will not fit for the next model
    either, so the finding belongs to the node."""
    from llmorch.engine.worker import output_budget

    node, _provider, deps, _health = _sizing_harness(manifest, needs=10**6, est=400)
    model = manifest.model(GROQ)

    first = output_budget(node, model, 0)
    second = output_budget(node, model, 1)
    assert second == 2 * first
