"""What a run publishes while it is still running.

Everything else this system reports is retrospective. This is the one part that
has to be right *during* a run, so what the tests pin is the difference between
the two: a node that has not come back yet is visible, its tokens are not
guessed, and a process that was killed does not leave a page claiming it is
still working.

The last group is about a number rather than a mechanism. The panel's "today"
comes from the vendor's own headers and the quota table's comes from this
machine's ledger; they disagree on purpose, and the page has to say which is
which. Two numbers under one name is how a reader ends up trusting neither.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from llmorch.engine.progress import (
    PROGRESS_NAME,
    ProgressWriter,
    is_active,
    read_progress,
)
from llmorch.demo.website import build_nodes
from llmorch.types import (
    Assignment,
    NodeResult,
    NodeState,
    ScoreBreakdown,
    Usage,
)

NODES = {n.id: n for n in build_nodes()}
ASSIGNED = {
    node_id: Assignment(
        node_id=node_id,
        model_id="groq/gpt-oss-120b",
        score=0.5,
        breakdown=ScoreBreakdown(),
    )
    for node_id in NODES
}


def _writer(tmp_path, **kw) -> ProgressWriter:
    return ProgressWriter(run_dir=tmp_path, run_id="20260101-000000", **kw)


def _done(node_id: str, prompt: int = 100, completion: int = 200) -> NodeResult:
    return NodeResult(
        node_id=node_id,
        state=NodeState.DONE,
        attempts=1,
        model_id="groq/gpt-oss-120b",
        usage=Usage(prompt_tokens=prompt, completion_tokens=completion),
    )


# ==========================================================================
# The plan, before any of it has happened
# ==========================================================================


def test_the_whole_shape_is_published_before_the_first_request(tmp_path):
    """A dashboard opened during the first call should show the run, not only
    the parts of it that have already finished."""
    writer = _writer(tmp_path)
    writer.begin(NODES, ASSIGNED)

    payload = read_progress(tmp_path)
    assert payload is not None
    assert len(payload["nodes"]) == len(NODES)
    assert {n["state"] for n in payload["nodes"]} == {"pending"}
    assert payload["totals"]["nodes"] == len(NODES)
    assert payload["totals"]["prompt_tokens"] == 0


def test_a_node_in_flight_is_visible_before_it_comes_back(tmp_path):
    """The whole point. The ledger only learns about a call that answered."""
    writer = _writer(tmp_path)
    writer.begin(NODES, ASSIGNED)
    writer.node_started("server", "groq/qwen3-27b")

    node = next(n for n in read_progress(tmp_path)["nodes"] if n["node_id"] == "server")
    assert node["state"] == "running"
    assert node["model_id"] == "groq/qwen3-27b"
    assert node["started_utc"]


def test_a_node_in_flight_reports_no_tokens(tmp_path):
    """The governor reserves on an estimate and reconciles on commit, so a node
    in flight has a budget and not a bill. Publishing the estimate as spend
    would show a total that walks backwards when the wave lands."""
    writer = _writer(tmp_path)
    writer.begin(NODES, ASSIGNED)
    writer.node_started("server", "groq/gpt-oss-120b")

    assert read_progress(tmp_path)["totals"]["completion_tokens"] == 0


def test_tokens_land_as_each_node_finishes(tmp_path):
    writer = _writer(tmp_path)
    writer.begin(NODES, ASSIGNED)
    writer.node_finished("server", _done("server", 100, 200))
    writer.node_finished("index", _done("index", 50, 75))

    totals = read_progress(tmp_path)["totals"]
    assert totals["prompt_tokens"] == 150
    assert totals["completion_tokens"] == 275
    assert totals["settled"] == 2


def test_spend_is_grouped_by_model(tmp_path):
    writer = _writer(tmp_path)
    writer.begin(NODES, ASSIGNED)
    writer.node_finished("server", _done("server", 100, 200))
    writer.node_finished("index", _done("index", 50, 75))

    by_model = {r["model_id"]: r for r in read_progress(tmp_path)["by_model"]}
    assert by_model["groq/gpt-oss-120b"]["calls"] == 2
    assert by_model["groq/gpt-oss-120b"]["completion_tokens"] == 275


def test_a_degraded_node_carries_its_reason(tmp_path):
    writer = _writer(tmp_path)
    writer.begin(NODES, ASSIGNED)
    writer.node_finished(
        "server",
        NodeResult(
            node_id="server",
            state=NodeState.DEGRADED,
            attempts=3,
            error="every vendor in the chain refused",
        ),
    )

    node = next(n for n in read_progress(tmp_path)["nodes"] if n["node_id"] == "server")
    assert node["state"] == "degraded"
    assert node["attempts"] == 3
    assert "refused" in node["error"]


# ==========================================================================
# Ending, in both the ways a run can end
# ==========================================================================


def test_finishing_settles_whatever_was_still_in_flight(tmp_path):
    """Better than leaving a node spinning in a page nobody will refresh again."""
    writer = _writer(tmp_path)
    writer.begin(NODES, ASSIGNED)
    writer.node_started("server", "groq/gpt-oss-120b")
    writer.finish("stopped early")

    payload = read_progress(tmp_path)
    assert payload["finished"] is True
    assert not [n for n in payload["nodes"] if n["state"] in ("pending", "running")]
    assert is_active(payload) is False


def test_a_killed_run_stops_being_active_without_anything_writing_finished(tmp_path):
    """The file is all a killed process leaves behind, and nothing gets to write
    "finished" on its behalf — so staleness is the only honest signal."""
    writer = _writer(tmp_path)
    writer.begin(NODES, ASSIGNED)
    writer.node_started("server", "groq/gpt-oss-120b")

    payload = read_progress(tmp_path)
    assert is_active(payload) is True

    later = datetime.now(timezone.utc) + timedelta(minutes=10)
    assert is_active(payload, now=later) is False


def test_a_rate_limited_wait_is_not_mistaken_for_a_dead_run(tmp_path):
    """A node can legitimately sit for a minute waiting out a per-minute window,
    and calling that "stopped" would misreport the most interesting moment."""
    writer = _writer(tmp_path)
    writer.begin(NODES, ASSIGNED)

    payload = read_progress(tmp_path)
    soon = datetime.now(timezone.utc) + timedelta(seconds=45)
    assert is_active(payload, now=soon) is True


# ==========================================================================
# It must never be able to break the run it is describing
# ==========================================================================


def test_an_unwritable_directory_is_not_a_crash(tmp_path):
    """Monitoring that can fail a run costs more than it reports."""
    writer = ProgressWriter(run_dir=tmp_path / "file.txt" / "nested", run_id="x")
    writer.begin(NODES, ASSIGNED)  # must not raise
    writer.node_finished("server", _done("server"))
    writer.finish()


def test_a_headroom_reading_that_throws_does_not_take_the_run_with_it(tmp_path):
    def broken():
        raise RuntimeError("governor exploded")

    writer = _writer(tmp_path)
    writer.begin(NODES, ASSIGNED, headroom=broken)

    assert read_progress(tmp_path)["headroom"] == []


def test_a_half_written_file_is_never_read(tmp_path):
    """A dashboard polling every two seconds will hit a write eventually, and
    half a JSON document is a crash in the reader rather than a stale number."""
    (tmp_path / PROGRESS_NAME).write_text('{"version": 1, "nodes": [', encoding="utf-8")
    assert read_progress(tmp_path) is None


def test_a_document_from_another_version_is_declined(tmp_path):
    (tmp_path / PROGRESS_NAME).write_text(
        json.dumps({"version": 99, "nodes": []}), encoding="utf-8"
    )
    assert read_progress(tmp_path) is None


# ==========================================================================
# What the terminal used to print
# ==========================================================================


def test_the_end_of_run_findings_travel_to_the_page(tmp_path):
    """The tables moved to the browser, so the browser has to carry what they
    carried."""
    writer = _writer(tmp_path)
    writer.begin(NODES, ASSIGNED)
    writer.summarise(
        output_dir="/runs/x/output",
        contract={"ok": False, "checks_run": ["pages exist"], "issues": []},
        smoke=None,
        warnings=["groq/qwen3-27b: rate limited, waiting it out"],
    )

    summary = read_progress(tmp_path)["summary"]
    assert summary["output_dir"] == "/runs/x/output"
    assert summary["contract"]["ok"] is False
    assert summary["smoke"] is None
    assert "rate limited" in summary["warnings"][0]


def test_a_smoke_run_that_never_happened_is_not_a_pass(tmp_path):
    """`None` and "it passed" are different answers, and the absence of evidence
    is never a tick."""
    writer = _writer(tmp_path)
    writer.begin(NODES, ASSIGNED)
    writer.summarise(smoke=None)

    assert read_progress(tmp_path)["summary"]["smoke"] is None


# ==========================================================================
# Through the scheduler and the CLI
# ==========================================================================


@pytest.mark.asyncio
async def test_the_scheduler_publishes_every_transition(tmp_path, monkeypatch):
    from llmorch.config import RunConfig
    from llmorch.demo.website import ARTIFACTS, INTERFACE
    from llmorch.engine.blackboard import Blackboard
    from llmorch.engine.graph import TaskGraph
    from llmorch.engine.scheduler import Scheduler
    from llmorch.providers.base import ProviderRegistry
    from llmorch.providers.mock import MockProvider
    from llmorch.quota.governor import Governor
    from llmorch.registry.manifest import load_manifest

    manifest = load_manifest()
    provider = MockProvider(responses=dict(ARTIFACTS))
    registry = ProviderRegistry()
    for model in manifest.enabled_models:
        registry.register(model.id, provider)

    writer = _writer(tmp_path)
    scheduler = Scheduler(
        TaskGraph.build(build_nodes()),
        manifest,
        Governor(manifest),
        registry,
        config=RunConfig(task="build a notes app", run_id="20260101-000000"),
        blackboard=Blackboard(interface=INTERFACE),
        progress=writer,
    )

    await scheduler.run()

    payload = read_progress(tmp_path)
    assert payload["totals"]["settled"] == len(NODES)
    assert payload["totals"]["completion_tokens"] > 0
    assert all(n["model_id"] for n in payload["nodes"])


def test_the_cli_keeps_the_tables_out_of_the_terminal(tmp_path, monkeypatch, capsys):
    """The tables are a page of numbers that changes while it is printed, and a
    terminal can only show the state they were in when the run ended."""
    from llmorch import __main__ as cli

    monkeypatch.setenv("LLMORCH_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("LLMORCH_STATE_DB", str(tmp_path / "state.db"))

    assert cli.main(["run", "build a notes app"]) == 0

    out = capsys.readouterr().out
    assert "Assignment" not in out
    assert "Token share" not in out
    assert "Quota efficiency" not in out

    # What is left is the verdict and where to look.
    assert "file(s) written" in out
    assert "checks passed" in out
    assert "127.0.0.1" in out


def test_the_tables_come_back_when_asked_for(tmp_path, monkeypatch, capsys):
    from llmorch import __main__ as cli

    monkeypatch.setenv("LLMORCH_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("LLMORCH_STATE_DB", str(tmp_path / "state.db"))

    assert cli.main(["run", "--tables", "build a notes app"]) == 0

    out = capsys.readouterr().out
    assert "Assignment" in out
    assert "Token share" in out


def test_a_run_leaves_its_progress_behind_for_the_dashboard(tmp_path, monkeypatch):
    from llmorch import __main__ as cli
    from llmorch.dashboard.state import _current_run

    monkeypatch.setenv("LLMORCH_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("LLMORCH_STATE_DB", str(tmp_path / "state.db"))
    cli.main(["run", "build a notes app"])

    current = _current_run()
    assert current is not None
    assert current["finished"] is True
    assert current["active"] is False
    assert current["summary"]["contract"]["ok"] is True
    assert current["totals"]["settled"] == current["totals"]["nodes"]


def test_a_probe_that_answered_is_not_published_as_a_failure():
    """`Probe` records what came back and leaves the judgement to the report's
    issues. Reading an `ok` field it does not have made every 200 a failure, and
    the dashboard painted a passing smoke run entirely red."""
    from llmorch import __main__ as cli
    from llmorch.engine.smoke import Probe, SmokeReport

    report = SmokeReport(
        ran=True,
        entrypoint="server.py",
        port=8000,
        probes=[
            Probe(method="GET", path="/", status=200),
            Probe(method="POST", path="/api/notes", status=201),
            Probe(method="GET", path="/missing", status=404),
            Probe(method="GET", path="/dead", status=None, detail="connection refused"),
        ],
    )

    facts = cli._smoke_facts(report)
    verdicts = {p["path"]: p["ok"] for p in facts["probes"]}

    assert verdicts == {"/": True, "/api/notes": True, "/missing": False, "/dead": False}
