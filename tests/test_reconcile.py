"""Dispatcher and graph tests — the assignment logic, offline and deterministic."""

from __future__ import annotations

import pytest

from llmorch.engine.graph import TaskGraph
from llmorch.errors import GraphError
from llmorch.negotiate.reconcile import (
    ReconcileInput,
    is_feasible,
    normalise_bids,
    reconcile,
)
from llmorch.negotiate.roles import needs_review, parse_role
from llmorch.registry.manifest import load_manifest
from llmorch.types import Bid, OutputKind, Role, TaskNode

GROQ = "groq/gpt-oss-120b"
QWEN = "groq/qwen3-27b"
GEMINI = "gemini/3.6-flash"


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


def _node(node_id, role, tokens=1500, deps=(), path=None):
    return TaskNode(
        id=node_id,
        title=f"{role.value} work",
        role=role,
        spec=f"do the {role.value} part",
        output_path=path or f"{node_id}.txt",
        deps=tuple(deps),
        est_output_tokens=tokens,
    )


# ==========================================================================
# Graph
# ==========================================================================


def test_levels_respect_dependency_order():
    g = TaskGraph.build(
        [
            _node("schema", Role.BACKEND),
            _node("api", Role.BACKEND, deps=["schema"]),
            _node("ui", Role.FRONTEND, deps=["api"]),
        ]
    )
    assert g.levels() == [["schema"], ["api"], ["ui"]]


def test_independent_nodes_share_a_level():
    g = TaskGraph.build([_node("a", Role.FRONTEND), _node("b", Role.BACKEND)])
    assert g.levels() == [["a", "b"]]


def test_cycles_are_repaired_rather_than_fatal():
    """A decomposing model will emit a cycle eventually. Discarding the whole
    plan would waste the request that produced it."""
    g = TaskGraph.build(
        [_node("a", Role.BACKEND, deps=["b"]), _node("b", Role.FRONTEND, deps=["a"])]
    )
    assert len(g.levels_flat()) == 2
    assert any("cycle" in w for w in g.warnings)


def test_cycles_can_be_made_fatal_when_requested():
    with pytest.raises(GraphError):
        TaskGraph.build(
            [
                _node("a", Role.BACKEND, deps=["b"]),
                _node("b", Role.FRONTEND, deps=["a"]),
            ],
            repair=False,
        )


def test_dangling_dependencies_are_dropped_with_a_warning():
    g = TaskGraph.build([_node("a", Role.BACKEND, deps=["ghost"])])
    assert g.nodes["a"].deps == ()
    assert any("ghost" in w for w in g.warnings)


def test_duplicate_node_ids_are_rejected():
    with pytest.raises(GraphError, match="duplicate"):
        TaskGraph.build([_node("a", Role.BACKEND), _node("a", Role.FRONTEND)])


def test_ready_nodes_unblock_as_dependencies_complete():
    g = TaskGraph.build(
        [_node("a", Role.BACKEND), _node("b", Role.FRONTEND, deps=["a"])]
    )
    assert g.ready_nodes(set()) == ["a"]
    assert g.ready_nodes({"a"}) == ["b"]


def test_pruning_merges_same_role_leaves_to_fit_the_budget():
    """Runs before bidding, so the bid prompt stays small."""
    g = TaskGraph.build([_node(f"n{i}", Role.CONTENT) for i in range(6)])
    absorbed = g.prune_to_budget(3)
    assert len(g) <= 3
    assert absorbed


def test_pruning_is_a_no_op_when_already_within_budget():
    g = TaskGraph.build([_node("a", Role.BACKEND)])
    assert g.prune_to_budget(10) == []
    assert len(g) == 1


# ==========================================================================
# Bid normalisation — the anti-bragging defence
# ==========================================================================


def test_a_bidder_claiming_high_confidence_on_everything_is_neutralised():
    """The central defence against overclaiming. A model answering 0.95 to every
    question contributes a flat zero rather than dominating."""
    bids = [Bid("braggart", f"n{i}", 0.95) for i in range(4)]
    normalised = normalise_bids(bids)
    assert all(v == 0.0 for v in normalised.values())


def test_relative_preference_within_a_bidder_is_preserved():
    """What matters is which nodes a model rates above its own average, not the
    absolute numbers it happens to use."""
    bids = [
        Bid("m", "backend", 0.9),
        Bid("m", "frontend", 0.5),
        Bid("m", "research", 0.2),
    ]
    n = normalise_bids(bids)
    assert n[("m", "backend")] > n[("m", "frontend")] > n[("m", "research")]


def test_two_bidders_using_different_scales_are_comparable_after_normalisation():
    """A cautious bidder (0.3-0.5) and a confident one (0.8-0.95) both express
    the same preference; normalisation must surface that."""
    bids = [
        Bid("cautious", "a", 0.5),
        Bid("cautious", "b", 0.3),
        Bid("confident", "a", 0.95),
        Bid("confident", "b", 0.80),
    ]
    n = normalise_bids(bids)
    assert n[("cautious", "a")] > 0 and n[("confident", "a")] > 0
    assert n[("cautious", "b")] < 0 and n[("confident", "b")] < 0


def test_a_single_bid_yields_no_signal():
    """One data point has no distribution to normalise against."""
    assert normalise_bids([Bid("m", "n1", 0.9)]) == {}


# ==========================================================================
# Feasibility
# ==========================================================================


def test_node_too_large_for_groq_is_infeasible(manifest):
    """Groq's 6,000 TPM caps output at 4,096 tokens; a 5,000-token node cannot
    be served there however good the score."""
    big = _node("big", Role.BACKEND, tokens=5000)
    assert not is_feasible(manifest, GROQ, big)
    assert is_feasible(manifest, GEMINI, big)


def test_normal_sized_node_is_feasible_everywhere(manifest):
    small = _node("small", Role.BACKEND, tokens=1500)
    assert is_feasible(manifest, GROQ, small)
    assert is_feasible(manifest, GEMINI, small)


# ==========================================================================
# Assignment
# ==========================================================================


def _inp(manifest, nodes, **kw):
    return ReconcileInput(
        graph=TaskGraph.build(nodes),
        manifest=manifest,
        candidates=kw.pop("candidates", [GROQ, QWEN, GEMINI]),
        **kw,
    )


def test_research_goes_to_gemini_and_backend_to_a_groq_model(manifest):
    """The user's own worked example. Also the quota-correct outcome: it keeps
    Gemini's scarce daily requests for work that needs its context."""
    res = reconcile(
        _inp(
            manifest,
            [
                _node("research", Role.RESEARCH, tokens=1200),
                _node("backend", Role.BACKEND, tokens=1200),
            ],
        )
    )
    assert res.model_for("research") == GEMINI
    assert res.model_for("backend").startswith("groq/")


def test_infeasible_nodes_are_reported_rather_than_misassigned(manifest):
    res = reconcile(
        _inp(manifest, [_node("huge", Role.BACKEND, tokens=5000)], candidates=[GROQ])
    )
    assert res.unassigned == ["huge"]
    assert not res.assignments
    assert any("ceiling" in n for n in res.notes)


def test_work_is_spread_rather_than_hoarded_by_the_best_model(manifest):
    """Evenness is enforced by a capacity bound, not requested in a prompt."""
    nodes = [_node(f"n{i}", Role.BACKEND, tokens=1000) for i in range(6)]
    res = reconcile(_inp(manifest, nodes))
    assert len(set(a.model_id for a in res.assignments.values())) > 1


def test_token_share_stays_within_the_imbalance_tolerance(manifest):
    nodes = [_node(f"n{i}", Role.FRONTEND, tokens=1000) for i in range(9)]
    inp = _inp(manifest, nodes, imbalance_tolerance=0.35)
    res = reconcile(inp)

    shares = res.token_share(inp.graph)
    ideal = sum(shares.values()) / len(inp.candidates)
    assert max(shares.values()) <= ideal * 1.35 + 1


def test_balance_is_measured_in_tokens_not_node_count(manifest):
    """One 3,000-token node and three 500-token nodes is not an even split by
    count, and the algorithm must not treat it as one."""
    nodes = [
        _node("big", Role.FRONTEND, tokens=3000),
        *[_node(f"small{i}", Role.FRONTEND, tokens=500) for i in range(3)],
    ]
    inp = _inp(manifest, nodes, candidates=[GEMINI, QWEN])
    res = reconcile(inp)

    shares = res.token_share(inp.graph)
    assert len(shares) == 2  # both models used
    assert max(shares.values()) <= 4500 * 0.5 * 1.35 + 3000


def test_quota_pressure_breaks_a_close_contest(manifest):
    """Backend is nearly a tie (both 0.70 affinity), so quota pressure decides
    it. No special-casing anywhere — the penalty term does this on its own."""
    nodes = [_node("be", Role.BACKEND, tokens=1000)]

    fresh = reconcile(_inp(manifest, nodes, candidates=[GEMINI, GROQ]))
    assert fresh.model_for("be") == GEMINI

    pressured = reconcile(
        _inp(manifest, nodes, candidates=[GEMINI, GROQ], quota_pressure={GEMINI: 1.0})
    )
    assert pressured.model_for("be") == GROQ


def test_quota_pressure_does_not_override_a_decisive_capability_gap(manifest):
    """Gemini is far better at research than any Groq model, so pressure alone
    should not move that work. The governor's hard EXHAUSTED_TODAY is what
    actually stops a spent model — this term only nudges."""
    nodes = [_node("research", Role.RESEARCH, tokens=1000)]
    pressured = reconcile(_inp(manifest, nodes, quota_pressure={GEMINI: 1.0}))
    assert pressured.model_for("research") == GEMINI


def test_track_record_influences_future_assignments(manifest):
    """A model that has repeatedly failed at a role stops winning it."""
    nodes = [_node("fe", Role.FRONTEND, tokens=1000)]

    baseline = reconcile(_inp(manifest, nodes)).model_for("fe")
    penalised = reconcile(
        _inp(manifest, nodes, track_record={(baseline, Role.FRONTEND): 0.0})
    ).model_for("fe")
    assert penalised != baseline


def test_bids_can_override_static_priors(manifest):
    """A model that rates itself unusually high on a node it would not normally
    win can still take it."""
    nodes = [_node("fe", Role.FRONTEND, tokens=1000), _node("be", Role.BACKEND, 1000)]

    bids = [
        Bid(QWEN, "fe", 0.95),
        Bid(QWEN, "be", 0.10),
        Bid(GEMINI, "fe", 0.20),
        Bid(GEMINI, "be", 0.90),
    ]
    res = reconcile(_inp(manifest, nodes, bids=bids))
    assert res.model_for("fe") == QWEN


def test_every_assignment_explains_itself(manifest):
    """`plan --explain` needs this to show why a model won a node."""
    res = reconcile(_inp(manifest, [_node("research", Role.RESEARCH)]))
    a = res.assignments["research"]
    assert "affinity" in a.rationale
    assert a.breakdown.total == pytest.approx(a.score)


def test_no_candidates_degrades_every_node(manifest):
    res = reconcile(_inp(manifest, [_node("a", Role.BACKEND)], candidates=[]))
    assert res.unassigned == ["a"]


# ==========================================================================
# Role parsing
# ==========================================================================


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("frontend", Role.FRONTEND),
        ("front-end", Role.FRONTEND),
        ("Front End", Role.FRONTEND),
        ("UI", Role.FRONTEND),
        ("api", Role.BACKEND),
        ("database", Role.BACKEND),
        ("CSS", Role.STYLING),
        ("research", Role.RESEARCH),
        ("copy", Role.CONTENT),
        ("QA", Role.REVIEW),
    ],
)
def test_role_aliases_are_normalised(raw, expected):
    """Decomposers return near-misses constantly; normalising costs nothing
    while a repair request costs quota."""
    assert parse_role(raw) is expected


def test_unrecognised_role_falls_back_without_raising():
    assert parse_role("wibble") is Role.INTEGRATION
    assert parse_role("") is Role.INTEGRATION


def test_review_policy_gates_which_nodes_get_reviewed():
    assert not needs_review(Role.BACKEND, OutputKind.CODE, "off")
    assert needs_review(Role.BACKEND, OutputKind.CODE, "code")
    assert not needs_review(Role.CONTENT, OutputKind.TEXT, "code")
    assert needs_review(Role.CONTENT, OutputKind.TEXT, "all")
