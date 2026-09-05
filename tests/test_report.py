"""The run, written down.

Every number in `report.md` comes from the objects the terminal renderers read,
so the two can never disagree about what happened. What these tests hold in
place is the ordering and the honesty of the verdict — the part somebody reads
weeks later, next to a folder of artifacts that look equally plausible whether
the run went well or not.
"""

from __future__ import annotations

from llmorch.config import RunConfig
from llmorch.demo.website import build_nodes
from llmorch.engine.contracts import ContractReport
from llmorch.engine.graph import TaskGraph
from llmorch.engine.materialize import MaterializeReport
from llmorch.engine.scheduler import RunOutcome
from llmorch.engine.smoke import Probe, SmokeReport
from llmorch.negotiate.reconcile import ReconcileResult
from llmorch.report.document import render_run_report
from llmorch.types import Assignment, NodeResult, NodeState, ScoreBreakdown, Usage

STAMP = "2026-01-01T00:00:00+00:00"


def _graph() -> TaskGraph:
    return TaskGraph.build(build_nodes())


def _outcome(graph: TaskGraph, *, degrade: set[str] = frozenset()) -> RunOutcome:
    outcome = RunOutcome()
    for index, node_id in enumerate(graph.nodes):
        degraded = node_id in degrade
        outcome.results[node_id] = NodeResult(
            node_id=node_id,
            state=NodeState.DEGRADED if degraded else NodeState.DONE,
            artifact="" if degraded else "x",
            model_id="groq/gpt-oss-120b" if index % 2 else "gemini/3.6-flash",
            attempts=1,
            vendors_tried=("groq",),
            usage=Usage(prompt_tokens=100, completion_tokens=200),
            error="no healthy model" if degraded else None,
        )
    return outcome


def _plan(graph: TaskGraph) -> ReconcileResult:
    plan = ReconcileResult()
    for index, node_id in enumerate(graph.nodes):
        plan.assignments[node_id] = Assignment(
            node_id=node_id,
            model_id="groq/gpt-oss-120b" if index % 2 else "gemini/3.6-flash",
            score=1.0,
            breakdown=ScoreBreakdown(),
        )
    return plan


def _render(**overrides) -> str:
    graph = _graph()
    outcome = overrides.pop("outcome", None) or _outcome(graph)
    kwargs = {
        "config": RunConfig(task="build a notes app", run_id="20260101-000000"),
        "graph": graph,
        "plan": _plan(graph),
        "outcome": outcome,
        "materialized": MaterializeReport(written=("server.py",)),
        "contract": ContractReport(checks_run=["pages exist"]),
        "smoke": None,
        "generated_utc": STAMP,
    }
    kwargs.update(overrides)
    return render_run_report(**kwargs)


def _verdict(text: str) -> str:
    return text.split("## Verdict", 1)[1].split("##", 1)[0]


# ==========================================================================
# The verdict
# ==========================================================================


def test_the_verdict_comes_before_the_evidence():
    """Someone reading this later wants the answer, then the support for it."""
    text = _render()
    assert text.startswith("# Run 20260101-000000")
    assert text.index("## Verdict") < text.index("## Nodes") < text.index("## Spend")


def test_a_clean_run_says_so_on_every_axis():
    verdict = _verdict(_render(smoke=SmokeReport(ran=True, probes=[Probe("GET", "/")])))
    assert "All 6 nodes produced their artifact" in verdict
    assert "1 cross-artifact checks passed" in verdict
    assert "The assembled project runs" in verdict


def test_degraded_nodes_are_named_in_the_verdict():
    graph = _graph()
    text = _render(outcome=_outcome(graph, degrade={"style"}))
    assert "**1 of 6 nodes degraded** — style" in _verdict(text)


def test_a_run_whose_nodes_all_succeeded_can_still_have_failed():
    """The case the verdict exists for.

    Every other section reports on a step of the pipeline. Only this one reports
    on the result, and the result can be broken while every step went perfectly.
    """
    smoke = SmokeReport(ran=True, probes=[Probe("GET", "/", 500)])
    smoke.add("error", "`/` returned 500")

    verdict = _verdict(_render(smoke=smoke))

    assert "All 6 nodes produced their artifact" in verdict
    assert "**The assembled project does not run**" in verdict


def test_a_contract_mismatch_is_not_buried():
    contract = ContractReport(checks_run=["pages exist"])
    contract.add(
        "error", "`app.js` calls `/api/note`, which the contract does not declare"
    )

    verdict = _verdict(_render(contract=contract))

    assert "**1 cross-artifact mismatch(es)**" in verdict


def test_a_smoke_run_that_never_happened_is_not_a_pass():
    assert "Not started — pass `--smoke`" in _verdict(_render())


def test_a_skipped_smoke_run_reports_why():
    text = _render(smoke=SmokeReport(skipped="port 8000 already in use"))
    assert "**Not started** — port 8000 already in use" in _verdict(text)


# ==========================================================================
# The body
# ==========================================================================


def test_every_node_appears_with_the_model_that_wrote_it():
    graph = _graph()
    outcome = _outcome(graph)
    text = _render(outcome=outcome)

    section = text.split("## Nodes", 1)[1].split("##", 1)[0]
    for node_id, node in graph.nodes.items():
        model = outcome.results[node_id].model_id
        row = f"| `{node_id}` | {node.role.value} | {model} | done | 1 | groq |"
        assert row in section


def test_a_degraded_node_is_marked_in_its_row():
    graph = _graph()
    text = _render(outcome=_outcome(graph, degrade={"style"}))
    assert "| `style` | styling |" in text
    assert "**degraded**" in text


def test_planned_and_written_shares_are_both_shown():
    """They come apart: the split is enforced over estimates when the plan is
    made, and failover moves work after the fact."""
    text = _render()
    section = text.split("## Fair share", 1)[1].split("##", 1)[0]
    assert "planned tokens" in section and "written tokens" in section
    assert "An even split across 2 model(s) is 50% each" in section


def test_the_smoke_probes_are_written_down():
    smoke = SmokeReport(
        ran=True,
        entrypoint="server.js",
        port=3000,
        installed="npm ci --ignore-scripts",
        probes=[Probe("GET", "/api/items", 200), Probe("POST", "/api/items", 201)],
    )
    text = _render(smoke=smoke)
    assert "Dependencies installed with `npm ci --ignore-scripts`." in text
    assert "Started `server.js` on port 3000." in text
    assert "| 200 | `GET /api/items` |" in text
    assert "| 201 | `POST /api/items` |" in text


def test_a_dead_project_keeps_its_stderr():
    smoke = SmokeReport(entrypoint="server.js", stderr_tail="Cannot find module 'x'")
    smoke.add("error", "server.js exited with code 1 instead of serving")

    text = _render(smoke=smoke)

    assert "Cannot find module 'x'" in text
    assert "```" in text


def test_resuming_is_explained_only_when_there_is_something_to_resume():
    graph = _graph()
    assert "## Resuming" not in _render()
    assert "## Resuming" in _render(outcome=_outcome(graph, degrade={"style"}))


def test_a_rejected_output_path_is_recorded():
    text = _render(
        materialized=MaterializeReport(
            written=("server.py",), rejected=(("../etc/x", "must be relative"),)
        )
    )
    assert "**rejected** `../etc/x`: must be relative" in text


def test_the_same_run_renders_the_same_document():
    """Two readings of one run must not differ; the timestamp is the only thing
    that moves, and it is passed in."""
    assert _render() == _render()
