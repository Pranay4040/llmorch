"""Checkpoint and resume tests.

The claim under test is narrow and expensive to get wrong: **work that was
already paid for is never bought twice.** Every assertion about the mock's call
log is really an assertion about quota — a resumed node that shows up in
`provider.calls` is a request spent on an artifact the system already had.

The quota wall itself is simulated with fault injection rather than waited for.
Gemini's 250-a-day cap is not something a test suite can afford to reach
honestly.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from llmorch.config import RunConfig
from llmorch.demo.website import ARTIFACTS, INTERFACE, build_nodes
from llmorch.engine.blackboard import Blackboard
from llmorch.engine.checkpoint import (
    Checkpoint,
    CheckpointError,
    NodeSnapshot,
    check_applies,
    checkpoint_path,
    latest_resumable,
    list_checkpoints,
    load,
    load_run,
    new_checkpoint,
    save,
    signature_of,
)
from llmorch.engine.graph import TaskGraph
from llmorch.engine.scheduler import Scheduler
from llmorch.errors import UnsafePath
from llmorch.providers.base import ProviderRegistry
from llmorch.providers.mock import FaultMode, MockProvider
from llmorch.quota.governor import Governor
from llmorch.quota.windows import FakeClock
from llmorch.registry.manifest import load_manifest
from llmorch.types import NodeResult, NodeState, Usage

TASK = "build a notes app"


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """Point the runs directory at a temp folder for the whole test."""
    root = tmp_path / "runs"
    monkeypatch.setenv("LLMORCH_RUNS_DIR", str(root))
    return root


async def _no_sleep(_seconds):
    return None


def _harness(manifest, *, faults=None, fail_times=None, run_id="cp-test"):
    provider = MockProvider(
        responses=dict(ARTIFACTS), faults=faults or {}, fail_times=fail_times or {}
    )
    registry = ProviderRegistry()
    for model in manifest.enabled_models:
        registry.register(model.id, provider)

    graph = TaskGraph.build(build_nodes())
    scheduler = Scheduler(
        graph,
        manifest,
        Governor(manifest, clock=FakeClock()),
        registry,
        config=RunConfig(task=TASK, run_id=run_id),
        blackboard=Blackboard(interface=INTERFACE),
        checkpoints=True,
        sleep=_no_sleep,
    )
    return scheduler, graph, provider


def _result(node_id, state=NodeState.DONE, **kw):
    return NodeResult(
        node_id=node_id,
        state=state,
        artifact=kw.get("artifact", f"# {node_id}"),
        summary=kw.get("summary", f"{node_id} summary"),
        model_id=kw.get("model_id", "groq/gpt-oss-120b"),
        attempts=kw.get("attempts", 1),
        usage=Usage(prompt_tokens=100, completion_tokens=200),
        error=kw.get("error"),
    )


# ==========================================================================
# The file itself
# ==========================================================================


def test_a_checkpoint_round_trips(tmp_path):
    book = new_checkpoint(run_id="r1", task=TASK, nodes=TaskGraph.build(build_nodes()).nodes)
    book.nodes["n1"] = NodeSnapshot.of(_result("n1"))

    save(tmp_path, book)
    back = load(tmp_path)

    assert back.run_id == "r1"
    assert back.nodes["n1"].artifact == "# n1"
    assert back.task_signature == book.task_signature


def test_a_degraded_node_keeps_no_artifact(tmp_path):
    """Its stub must never read back as finished work on the next pass."""
    snapshot = NodeSnapshot.of(
        _result("n1", state=NodeState.DEGRADED, artifact="# TODO stub")
    )
    assert snapshot.artifact == ""


def test_only_done_nodes_are_restored():
    book = new_checkpoint(run_id="r1", task=TASK, nodes={})
    book.nodes["done"] = NodeSnapshot.of(_result("done"))
    book.nodes["degraded"] = NodeSnapshot.of(_result("degraded", state=NodeState.DEGRADED))

    restored = book.restore_results()
    assert set(restored) == {"done"}


def test_the_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    """A torn write would turn a recoverable interruption into the exact total
    loss this module exists to prevent."""
    book = new_checkpoint(run_id="r1", task=TASK, nodes={})
    save(tmp_path, book)
    save(tmp_path, book)

    assert checkpoint_path(tmp_path).is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_a_half_written_checkpoint_is_refused_not_half_read(tmp_path):
    checkpoint_path(tmp_path).write_text('{"version": 1, "run_id": "r1"', encoding="utf-8")
    with pytest.raises(CheckpointError):
        load(tmp_path)


def test_a_checkpoint_from_a_future_format_is_refused(tmp_path):
    checkpoint_path(tmp_path).write_text(
        json.dumps({"version": 99, "run_id": "r1", "nodes": []}), encoding="utf-8"
    )
    with pytest.raises(CheckpointError):
        load(tmp_path)


# ==========================================================================
# Fingerprinting
# ==========================================================================


def test_the_signature_ignores_spec_wording(manifest):
    """Editing a node's prompt is a reason to re-run that node, not to throw
    away every other artifact in the run."""
    nodes = TaskGraph.build(build_nodes()).nodes
    first = signature_of(TASK, nodes)

    node_id = next(iter(nodes))
    import dataclasses

    nodes[node_id] = dataclasses.replace(nodes[node_id], spec="totally different words")
    assert signature_of(TASK, nodes) == first


def test_the_signature_changes_when_the_graph_does(manifest):
    nodes = TaskGraph.build(build_nodes()).nodes
    first = signature_of(TASK, nodes)
    nodes.pop(next(iter(nodes)))
    assert signature_of(TASK, nodes) != first


def test_resuming_onto_a_changed_plan_is_refused():
    """Artifacts built against two different interface contracts would look
    plausible together and be incoherent."""
    nodes = TaskGraph.build(build_nodes()).nodes
    book = new_checkpoint(run_id="r1", task=TASK, nodes=nodes)

    nodes.pop(next(iter(nodes)))
    with pytest.raises(CheckpointError):
        check_applies(book, TASK, nodes)


def test_a_run_id_that_could_escape_the_runs_folder_is_refused(runs):
    for hostile in ("../../etc", "a/b", "..", "C:evil"):
        with pytest.raises(UnsafePath):
            load_run(hostile)


# ==========================================================================
# Blocked-until accounting
# ==========================================================================


def test_resumable_at_is_the_earliest_reset():
    now = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
    book = new_checkpoint(run_id="r1", task=TASK, nodes={})
    book.blocked_until = {
        "gemini/3.6-flash": (now + timedelta(hours=8)).isoformat(),
        "groq/gpt-oss-120b": (now + timedelta(hours=3)).isoformat(),
    }
    assert book.seconds_until_resumable(now) == pytest.approx(3 * 3600)


def test_nothing_blocked_means_resume_now():
    book = new_checkpoint(run_id="r1", task=TASK, nodes={})
    assert book.resumable_at() is None
    assert book.seconds_until_resumable() == 0.0


# ==========================================================================
# The scheduler writes one
# ==========================================================================


async def test_a_run_leaves_a_checkpoint_behind(manifest, runs):
    scheduler, graph, _ = _harness(manifest)
    outcome = await scheduler.run()

    book = load(scheduler.config.run_dir)
    assert set(book.nodes) == set(outcome.results)
    assert book.is_complete
    assert book.completed == sorted(graph.nodes)


async def test_the_checkpoint_carries_the_artifact_text(manifest, runs):
    """Not a pointer to it. The artifact is the thing that cost a request."""
    scheduler, _, _ = _harness(manifest)
    await scheduler.run()

    book = load(scheduler.config.run_dir)
    assert all(s.artifact.strip() for s in book.nodes.values())


async def test_a_quota_wall_is_recorded_as_unfinished_work(manifest, runs):
    """Every model refusing on daily quota: the node degrades, but the run
    leaves behind exactly what a resume needs."""
    faults = {node: FaultMode.DAILY_LIMIT for node in ("style",)}
    scheduler, _, _ = _harness(manifest, faults=faults)
    outcome = await scheduler.run()

    book = load(scheduler.config.run_dir)
    assert "style" in book.unfinished
    assert book.completed and not book.is_complete
    assert outcome.results["style"].state is NodeState.DEGRADED


# ==========================================================================
# Resume — the point of the whole exercise
# ==========================================================================


async def test_resume_does_not_re_request_finished_nodes(manifest, runs):
    """The headline claim. A carried-over node appearing in the call log is a
    request spent on an artifact the system already held."""
    scheduler, graph, _ = _harness(manifest, faults={"style": FaultMode.DAILY_LIMIT})
    await scheduler.run()
    book = load(scheduler.config.run_dir)

    # Second attempt, quota restored, same run folder.
    scheduler2, _, provider2 = _harness(manifest, run_id=scheduler.config.run_id)
    outcome = await scheduler2.run(resume=book)

    called = {node_id for node_id, _ in provider2.calls}
    assert called == {"style"}
    assert outcome.all_succeeded
    assert len(outcome.results) == len(graph.nodes)


async def test_resume_keeps_the_original_artifacts(manifest, runs):
    scheduler, _, _ = _harness(manifest, faults={"style": FaultMode.DAILY_LIMIT})
    await scheduler.run()
    book = load(scheduler.config.run_dir)
    carried = book.nodes["server"].artifact

    scheduler2, _, _ = _harness(manifest, run_id=scheduler.config.run_id)
    outcome = await scheduler2.run(resume=book)

    assert outcome.results["server"].artifact == carried


async def test_a_resumed_node_still_reaches_the_blackboard(manifest, runs):
    """Downstream nodes read upstream summaries off the blackboard. A restored
    node that never lands there silently strips its dependants of context."""
    scheduler, _, _ = _harness(manifest, faults={"style": FaultMode.DAILY_LIMIT})
    await scheduler.run()
    book = load(scheduler.config.run_dir)

    scheduler2, _, _ = _harness(manifest, run_id=scheduler.config.run_id)
    await scheduler2.run(resume=book)

    assert scheduler2.blackboard.summary_of("server")
    assert "no summary" not in scheduler2.blackboard.summary_of("server")


async def test_resuming_a_finished_run_calls_nobody(manifest, runs):
    scheduler, _, _ = _harness(manifest)
    await scheduler.run()
    book = load(scheduler.config.run_dir)

    scheduler2, _, provider2 = _harness(manifest, run_id=scheduler.config.run_id)
    outcome = await scheduler2.run(resume=book)

    assert provider2.calls == []
    assert outcome.all_succeeded


async def test_a_degraded_node_is_retried_rather_than_trusted(manifest, runs):
    """Its stub is not an artifact, so resume must ask for it again."""
    scheduler, _, _ = _harness(manifest, faults={"style": FaultMode.DAILY_LIMIT})
    await scheduler.run()
    book = load(scheduler.config.run_dir)

    assert book.nodes["style"].state is NodeState.DEGRADED
    assert "style" not in book.restore_results()


async def test_the_second_pass_updates_the_same_checkpoint(manifest, runs):
    scheduler, _, _ = _harness(manifest, faults={"style": FaultMode.DAILY_LIMIT})
    await scheduler.run()
    first = load(scheduler.config.run_dir)

    scheduler2, _, _ = _harness(manifest, run_id=scheduler.config.run_id)
    await scheduler2.run(resume=first)

    after = load(scheduler.config.run_dir)
    assert after.is_complete
    assert after.run_id == first.run_id
    assert after.updated_utc >= first.updated_utc


# ==========================================================================
# Listing
# ==========================================================================


async def test_listing_finds_the_unfinished_run(manifest, runs):
    scheduler, _, _ = _harness(manifest, faults={"style": FaultMode.DAILY_LIMIT},
                               run_id="20260102-000001")
    await scheduler.run()

    finished, _, _ = _harness(manifest, run_id="20260102-000002")
    await finished.run()

    listed = {cp.run_id for cp in list_checkpoints()}
    assert listed == {"20260102-000001", "20260102-000002"}

    resumable = latest_resumable()
    assert resumable is not None and resumable.run_id == "20260102-000001"


def test_listing_survives_a_corrupt_run_folder(runs):
    """One unreadable directory must not hide every other resumable run."""
    (runs / "broken").mkdir(parents=True)
    checkpoint_path(runs / "broken").write_text("{not json", encoding="utf-8")

    good = runs / "20260101-000000"
    good.mkdir(parents=True)
    save(good, new_checkpoint(run_id="20260101-000000", task=TASK, nodes={}))

    assert [cp.run_id for cp in list_checkpoints()] == ["20260101-000000"]


def test_nothing_to_resume_is_not_an_error(runs):
    assert list_checkpoints() == []
    assert latest_resumable() is None


# ==========================================================================
# Through the CLI
# ==========================================================================


def _cli(argv):
    from llmorch.__main__ import build_parser

    args = build_parser().parse_args(argv)
    return args.func(args)


async def _seed_blocked_run(manifest, run_id):
    scheduler, _, _ = _harness(manifest, faults={"style": FaultMode.DAILY_LIMIT},
                               run_id=run_id)
    # A real clock, so the recorded reset moment is genuinely in the future.
    scheduler.governor = Governor(manifest)
    await scheduler.run()
    return load(scheduler.config.run_dir)


def test_resume_refuses_while_the_blocked_model_is_still_out(manifest, runs, capsys):
    """Spending three other models on work that is waiting for a fourth is how
    a quota wall turns into two quota walls."""
    # Synchronous on purpose: cmd_resume owns its own event loop.
    asyncio.run(_seed_blocked_run(manifest, "20260101-120000"))

    assert _cli(["resume", "20260101-120000"]) == 2
    assert "still blocked" in capsys.readouterr().out


def test_force_resumes_anyway(manifest, runs):
    asyncio.run(_seed_blocked_run(manifest, "20260101-120000"))
    assert _cli(["resume", "20260101-120000", "--force"]) == 0


def test_resuming_a_complete_run_is_a_no_op(manifest, runs, capsys):
    scheduler, _, _ = _harness(manifest, run_id="20260101-130000")
    asyncio.run(scheduler.run())

    assert _cli(["resume", "20260101-130000"]) == 0
    assert "already complete" in capsys.readouterr().out


def test_resume_list_runs_with_no_checkpoints(runs, capsys):
    assert _cli(["resume", "--list"]) == 0
    assert "no checkpoints yet" in capsys.readouterr().out


# ==========================================================================
# A checkpoint reconstructs its own graph
#
# Found by resuming a real run after a provider had been added: the plan
# signature includes the roster, so the resume re-planned, and the re-plan
# needed the one model that was rate limited at that moment.
# ==========================================================================


def test_the_checkpoint_carries_the_graph_it_was_run_against(runs):
    from llmorch.engine.checkpoint import plan_from_dict
    from llmorch.demo.website import INTERFACE

    nodes = {n.id: n for n in build_nodes()}
    book = new_checkpoint(run_id="r1", task=TASK, nodes=nodes, interface=INTERFACE)

    restored, interface = plan_from_dict(book.plan)
    assert {n.id for n in restored} == set(nodes)
    assert interface.runtime == INTERFACE.runtime
    assert interface.pages == INTERFACE.pages


def test_a_restored_node_keeps_everything_the_worker_needs(runs):
    from llmorch.engine.checkpoint import plan_from_dict

    nodes = {n.id: n for n in build_nodes()}
    book = new_checkpoint(run_id="r1", task=TASK, nodes=nodes)
    restored = {n.id: n for n in plan_from_dict(book.plan)[0]}

    for node_id, original in nodes.items():
        back = restored[node_id]
        assert back.role is original.role
        assert back.output_path == original.output_path
        assert back.output_kind is original.output_kind
        assert back.deps == original.deps
        assert back.needs == original.needs
        assert back.est_output_tokens == original.est_output_tokens


def test_the_stored_graph_survives_the_file(tmp_path):
    from llmorch.engine.checkpoint import plan_from_dict

    nodes = {n.id: n for n in build_nodes()}
    save(tmp_path, new_checkpoint(run_id="r1", task=TASK, nodes=nodes))

    reloaded = load(tmp_path)
    assert {n.id for n in plan_from_dict(reloaded.plan)[0]} == set(nodes)


def test_the_restored_graph_matches_its_own_signature(runs):
    """Which is what lets `check_applies` pass on resume without re-planning."""
    from llmorch.engine.checkpoint import plan_from_dict

    nodes = {n.id: n for n in build_nodes()}
    book = new_checkpoint(run_id="r1", task=TASK, nodes=nodes)

    restored, _interface = plan_from_dict(book.plan)
    check_applies(book, TASK, {n.id: n for n in restored})
