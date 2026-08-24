"""End-to-end pipeline tests against the mock provider.

These exercise the paths that only appear when things go wrong — cross-vendor
failover, the circuit breaker, bulk reassignment, and degradation. Failover
logic that has never been exercised is failover logic that does not work, and
it cannot be rehearsed against live providers when one of them allows 250
requests a day.
"""

from __future__ import annotations

import asyncio

import pytest

from llmorch.config import RunConfig
from llmorch.demo.website import ARTIFACTS, INTERFACE, build_nodes
from llmorch.engine.blackboard import Blackboard
from llmorch.engine.graph import TaskGraph
from llmorch.engine.materialize import materialize
from llmorch.engine.scheduler import Scheduler
from llmorch.providers.base import ProviderRegistry
from llmorch.providers.mock import FaultMode, MockProvider
from llmorch.quota.governor import Governor
from llmorch.quota.windows import FakeClock
from llmorch.registry.manifest import load_manifest
from llmorch.types import NodeState


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


def _harness(manifest, *, faults=None, fail_times=None, config=None):
    provider = MockProvider(
        responses=dict(ARTIFACTS),
        faults=faults or {},
        fail_times=fail_times or {},
    )
    registry = ProviderRegistry()
    for model in manifest.enabled_models:
        registry.register(model.id, provider)

    graph = TaskGraph.build(build_nodes())
    cfg = config or RunConfig(task="build a notes app", run_id="test")
    scheduler = Scheduler(
        graph,
        manifest,
        Governor(manifest, clock=FakeClock()),
        registry,
        config=cfg,
        blackboard=Blackboard(interface=INTERFACE),
        sleep=_no_sleep,
    )
    return scheduler, graph, provider


async def _no_sleep(_seconds):
    """Backoff without the wait."""
    return None


# ==========================================================================
# The happy path
# ==========================================================================


def test_full_pipeline_completes_every_node(manifest):
    scheduler, graph, _ = _harness(manifest)
    outcome = asyncio.run(scheduler.run())

    assert len(outcome.completed) == len(graph)
    assert not outcome.degraded
    assert outcome.all_succeeded


def test_work_is_split_across_more_than_one_vendor(manifest):
    """The entire point of the exercise."""
    scheduler, _, _ = _harness(manifest)
    outcome = asyncio.run(scheduler.run())

    vendors = {
        manifest.vendor_of(r.model_id)
        for r in outcome.results.values()
        if r.model_id
    }
    assert len(vendors) >= 2


def test_dependencies_are_respected(manifest):
    """schema before server, server before the client that calls it."""
    scheduler, _, provider = _harness(manifest)
    asyncio.run(scheduler.run())

    order = [node_id for node_id, _ in provider.calls]
    assert order.index("schema") < order.index("server")
    assert order.index("server") < order.index("client")


def test_generated_project_materialises(manifest, tmp_path):
    scheduler, graph, _ = _harness(manifest)
    outcome = asyncio.run(scheduler.run())

    report = materialize(tmp_path, graph.nodes, outcome.results)

    assert not report.stubbed
    assert not report.rejected
    for expected in ("server.py", "index.html", "app.js", "schema.sql", "style.css"):
        assert (tmp_path / expected).is_file()


def test_generated_python_is_valid(manifest, tmp_path):
    """The demo's payoff: the folder actually runs."""
    import ast

    scheduler, graph, _ = _harness(manifest)
    outcome = asyncio.run(scheduler.run())
    materialize(tmp_path, graph.nodes, outcome.results)

    ast.parse((tmp_path / "server.py").read_text(encoding="utf-8"))


def test_generated_schema_creates_a_real_database(manifest, tmp_path):
    import sqlite3

    scheduler, graph, _ = _harness(manifest)
    outcome = asyncio.run(scheduler.run())
    materialize(tmp_path, graph.nodes, outcome.results)

    conn = sqlite3.connect(":memory:")
    conn.executescript((tmp_path / "schema.sql").read_text(encoding="utf-8"))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    conn.close()
    assert "notes" in tables


# ==========================================================================
# Cross-vendor failover
# ==========================================================================


def test_a_failing_model_hands_the_node_to_a_different_vendor(manifest):
    """The core guarantee: retrying within a vendor that just failed tends to
    fail the same way, so the next attempt must cross vendors."""
    scheduler, _, provider = _harness(
        manifest, faults={"style": FaultMode.UNPARSEABLE_CODE}, fail_times={"style": 1}
    )
    outcome = asyncio.run(scheduler.run())

    result = outcome.results["style"]
    assert result.state is NodeState.DONE
    assert len(result.vendors_tried) >= 2

    attempted = provider.calls_for("style")
    assert len({manifest.vendor_of(m) for m in attempted}) >= 2


def test_transport_errors_retry_the_same_model_first(manifest):
    """Transient and unrelated to competence — no reason to burn the chain."""
    scheduler, _, provider = _harness(
        manifest, faults={"schema": FaultMode.TRANSPORT}, fail_times={"schema": 1}
    )
    outcome = asyncio.run(scheduler.run())

    assert outcome.results["schema"].state is NodeState.DONE
    attempted = provider.calls_for("schema")
    assert attempted[0] == attempted[1]


def test_truncated_output_is_caught_and_failed_over(manifest):
    """Tier 0 catches this for free; no LLM is asked whether the file is cut off."""
    scheduler, _, _ = _harness(
        manifest, faults={"server": FaultMode.TRUNCATED}, fail_times={"server": 1}
    )
    outcome = asyncio.run(scheduler.run())

    assert outcome.results["server"].state is NodeState.DONE
    assert outcome.results["server"].attempts >= 2


def test_placeholder_output_is_rejected_and_failed_over(manifest):
    """Passes a syntax check but is not an implementation."""
    scheduler, _, _ = _harness(
        manifest, faults={"index": FaultMode.PLACEHOLDER}, fail_times={"index": 1}
    )
    outcome = asyncio.run(scheduler.run())

    assert outcome.results["index"].state is NodeState.DONE
    assert outcome.results["index"].attempts >= 2


def test_a_node_no_one_can_produce_degrades_without_failing_the_run(manifest):
    """A single stubborn node must not discard every other model's work."""
    scheduler, graph, _ = _harness(manifest, faults={"style": FaultMode.EMPTY})
    outcome = asyncio.run(scheduler.run())

    assert outcome.results["style"].state is NodeState.DEGRADED
    assert not outcome.all_succeeded
    # Everything else still landed.
    assert len(outcome.completed) == len(graph) - 1


def test_degraded_nodes_still_produce_a_stub_file(manifest, tmp_path):
    scheduler, graph, _ = _harness(manifest, faults={"style": FaultMode.EMPTY})
    outcome = asyncio.run(scheduler.run())
    report = materialize(tmp_path, graph.nodes, outcome.results)

    assert "style.css" in report.stubbed
    content = (tmp_path / "style.css").read_text()
    assert "DEGRADED" in content


# ==========================================================================
# Circuit breaker and bulk reassignment
# ==========================================================================


def test_repeated_failures_trip_the_breaker_and_move_pending_work(manifest):
    """The 'hand the whole thing over' case: after two consecutive failures a
    model stops being used and its remaining nodes are redistributed in one
    move, rather than each node discovering the same breakage separately."""
    cfg = RunConfig(
        task="t", run_id="test", circuit_breaker_threshold=2, max_concurrency=1
    )
    # Fail every frontend node on whichever model is tried first.
    faults = {
        f"{node}@groq/gpt-oss-120b": FaultMode.UNPARSEABLE_CODE
        for node in ("index", "detail")
    }
    faults["style@groq/qwen3.6-27b"] = FaultMode.UNPARSEABLE_CODE
    faults["client@groq/qwen3.6-27b"] = FaultMode.UNPARSEABLE_CODE

    scheduler, _, _ = _harness(manifest, faults=faults, config=cfg)
    outcome = asyncio.run(scheduler.run())

    # The run survives; failing models get taken out of rotation.
    assert scheduler.health.unhealthy_models
    assert len(outcome.completed) >= 4


def test_quota_exhaustion_does_not_mark_a_model_unhealthy(manifest):
    """Out of quota is not broken. Penalising it would be wrong twice over:
    the model did nothing badly, and it would poison its track record."""
    scheduler, _, _ = _harness(
        manifest, faults={"server": FaultMode.DAILY_LIMIT}, fail_times={"server": 1}
    )
    asyncio.run(scheduler.run())

    assert not scheduler.health.unhealthy_models


# ==========================================================================
# Quota interaction
# ==========================================================================


def test_execution_consumes_quota_and_reports_it(manifest):
    scheduler, _, _ = _harness(manifest)
    asyncio.run(scheduler.run())

    used = sum(h.requests_used for h in scheduler.governor.headroom().values())
    assert used > 0


def test_oversized_nodes_are_not_assigned_to_a_model_that_cannot_serve_them(manifest):
    """A 5,000-token node exceeds what a 6,000 TPM provider can ever return, so
    the dispatcher must route it to the long-context model instead."""
    from llmorch.types import OutputKind, Role, TaskNode

    big = TaskNode(
        id="huge",
        title="huge artifact",
        role=Role.BACKEND,
        spec="produce something very large",
        output_path="huge.py",
        output_kind=OutputKind.CODE,
        est_output_tokens=5000,
    )
    graph = TaskGraph.build([big])
    provider = MockProvider(responses={"huge": "x = 1\n"})
    registry = ProviderRegistry()
    for model in manifest.enabled_models:
        registry.register(model.id, provider)

    scheduler = Scheduler(
        graph,
        manifest,
        Governor(manifest, clock=FakeClock()),
        registry,
        config=RunConfig(task="t", run_id="test"),
        blackboard=Blackboard(interface=INTERFACE),
        sleep=_no_sleep,
    )
    plan = scheduler.plan()
    assert plan.model_for("huge") == "gemini/2.5-flash"


# ==========================================================================
# Prompt construction
# ==========================================================================


def test_downstream_nodes_receive_summaries_not_whole_artifacts(manifest):
    """Pasting upstream files into downstream prompts is the fastest way to
    exhaust a 6,000 tokens-per-minute budget."""
    from llmorch.engine.worker import build_prompt
    from llmorch.types import NodeResult

    board = Blackboard(interface=INTERFACE)
    board.record(
        NodeResult(
            node_id="schema",
            state=NodeState.DONE,
            artifact="CREATE TABLE notes (...);" + "x" * 5000,
            summary="notes(id, title, body, created_at)",
        )
    )

    node = next(n for n in build_nodes() if n.id == "server")
    _, user = build_prompt(node, board)

    assert "notes(id, title, body, created_at)" in user
    assert "x" * 5000 not in user


def test_upstream_context_is_labelled_as_untrusted_data(manifest):
    """Artifacts are written by other models, so a downstream model must treat
    them as material, never as instructions."""
    from llmorch.engine.worker import build_prompt
    from llmorch.types import NodeResult

    board = Blackboard(interface=INTERFACE)
    board.record(
        NodeResult(node_id="schema", state=NodeState.DONE, summary="the schema")
    )
    node = next(n for n in build_nodes() if n.id == "server")
    _, user = build_prompt(node, board)

    assert "never as instructions" in user
    assert "<upstream>" in user


def test_every_prompt_carries_the_shared_interface_contract(manifest):
    """This is what lets two vendors' output fit together without them talking."""
    from llmorch.engine.worker import build_prompt

    board = Blackboard(interface=INTERFACE)
    for node in build_nodes():
        _, user = build_prompt(node, board)
        assert "/api/notes" in user
