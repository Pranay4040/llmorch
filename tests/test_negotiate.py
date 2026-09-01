"""Negotiation: planning, bidding, and the learned track record.

Three inputs to one decision, and each is trusted differently. The plan is a
model deciding *structure*, so its output is validated hard — everything
downstream keys off node ids and writes to output paths. Bids are a model
describing *itself*, so they are normalised into a preference ordering and can
be discarded entirely. The track record is the only input grounded in what
actually happened, so it is persisted, and shrunk toward neutral until there is
enough of it to mean something.
"""

from __future__ import annotations

import json

import pytest

from llmorch.config import RunConfig
from llmorch.demo.website import INTERFACE
from llmorch.engine.blackboard import Blackboard
from llmorch.engine.health import HealthTracker
from llmorch.engine.worker import WorkerDeps
from llmorch.negotiate import plancache
from llmorch.negotiate.bidding import BID_SYSTEM, collect_bids, parse_bids, should_bid
from llmorch.negotiate.decompose import (
    DecomposeError,
    Decomposition,
    decompose,
    parse_interface,
    parse_nodes,
    pick_planner,
    plan_signature,
)
from llmorch.negotiate.profiles import CONFIDENT_AFTER, NEUTRAL, Profiles, Record
from llmorch.providers.base import ProviderRegistry
from llmorch.providers.mock import CANNED_PLAN, MockProvider
from llmorch.quota.estimator import TokenEstimator
from llmorch.quota.governor import Governor
from llmorch.quota.windows import FakeClock
from llmorch.registry.manifest import load_manifest
from llmorch.types import (
    LimitKind,
    NodeResult,
    NodeState,
    Role,
    Verdict,
    VerifyResult,
)

GROQ = "groq/gpt-oss-120b"
GEMINI = "gemini/3.6-flash"


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


def _deps(manifest, provider):
    registry = ProviderRegistry()
    for model in manifest.enabled_models:
        registry.register(model.id, provider)
    return WorkerDeps(
        manifest=manifest,
        governor=Governor(manifest, clock=FakeClock()),
        registry=registry,
        estimator=TokenEstimator(),
        health=HealthTracker(),
        blackboard=Blackboard(interface=INTERFACE),
    )


def _done(model_id=GROQ, attempts=1, review=None):
    return NodeResult(
        node_id="n1", state=NodeState.DONE, artifact="x", model_id=model_id,
        attempts=attempts, review=review,
    )


# ==========================================================================
# Track record
# ==========================================================================


def test_an_unseen_pairing_is_neutral_not_zero():
    """An unproven model stays eligible but unfavoured — the same reasoning as
    an unlisted role affinity. Zero would exclude it forever, and it would never
    get the chance to prove otherwise."""
    assert Profiles().score(GROQ, Role.BACKEND) == NEUTRAL


def test_one_good_result_barely_moves_the_score():
    """A single success is not evidence. Without shrinkage the dispatcher would
    chase noise it paid real requests to generate."""
    profiles = Profiles()
    profiles.observe_result(_done(), Role.BACKEND)
    thin = profiles.score(GROQ, Role.BACKEND)

    for _ in range(CONFIDENT_AFTER):
        profiles.observe_result(_done(), Role.BACKEND)
    thick = profiles.score(GROQ, Role.BACKEND)

    assert NEUTRAL < thin < thick
    assert thick > 0.9


def test_a_quota_wall_is_not_a_performance_failure():
    """The load-bearing invariant. A degraded node means nobody produced it —
    recording a defeat would teach the dispatcher to avoid the model that was
    working perfectly until midnight."""
    profiles = Profiles()
    profiles.observe_result(
        NodeResult(node_id="n1", state=NodeState.DEGRADED, model_id=GROQ),
        Role.BACKEND,
    )
    assert profiles.score(GROQ, Role.BACKEND) == NEUTRAL
    assert not profiles.records


def test_a_rejected_artifact_is_the_strongest_negative_signal():
    """A peer from another vendor read the work and refused it."""
    profiles = Profiles()
    rejected = VerifyResult(verdict=Verdict.REJECT, tier=1, reviewer_model_id=GEMINI)
    for _ in range(CONFIDENT_AFTER):
        profiles.observe_result(_done(review=rejected), Role.BACKEND)

    assert profiles.score(GROQ, Role.BACKEND) < 0.1
    assert profiles.record_for(GROQ, Role.BACKEND).rejections == CONFIDENT_AFTER


def test_a_revision_scores_between_success_and_rejection():
    profiles = Profiles()
    revised = VerifyResult(verdict=Verdict.REVISE, tier=1, reviewer_model_id=GEMINI)
    for _ in range(CONFIDENT_AFTER):
        profiles.observe_result(_done(review=revised), Role.BACKEND)
    assert 0.4 < profiles.score(GROQ, Role.BACKEND) < 0.6


def test_retries_cost_something_even_when_the_artifact_lands():
    """Two attempts to produce one file is real quota spent."""
    clean, retried = Profiles(), Profiles()
    for _ in range(CONFIDENT_AFTER):
        clean.observe_result(_done(attempts=1), Role.BACKEND)
        retried.observe_result(_done(attempts=3), Role.BACKEND)
    assert clean.score(GROQ, Role.BACKEND) > retried.score(GROQ, Role.BACKEND)


def test_the_record_is_kept_per_role_not_per_model():
    """A model strong at backends and weak at styling is the normal case; one
    number per model would average that away."""
    profiles = Profiles()
    for _ in range(CONFIDENT_AFTER):
        profiles.observe_result(_done(), Role.BACKEND)
    assert profiles.score(GROQ, Role.BACKEND) > NEUTRAL
    assert profiles.score(GROQ, Role.STYLING) == NEUTRAL


def test_the_record_survives_the_process(tmp_path):
    """Every data point cost a live request; relearning it each session would
    mean paying for the same lesson forever."""
    path = tmp_path / "profiles.json"
    profiles = Profiles(path=path)
    for _ in range(CONFIDENT_AFTER):
        profiles.observe_result(_done(), Role.BACKEND)
    profiles.save()

    reloaded = Profiles.load(path)
    assert reloaded.score(GROQ, Role.BACKEND) == pytest.approx(
        profiles.score(GROQ, Role.BACKEND)
    )


def test_a_corrupt_or_foreign_history_starts_neutral_rather_than_failing(tmp_path):
    """Misreading a history would bias every assignment from here on; losing it
    costs a few requests."""
    bad = tmp_path / "profiles.json"
    bad.write_text("{not json", encoding="utf-8")
    assert Profiles.load(bad).records == {}

    future = tmp_path / "future.json"
    future.write_text(json.dumps({"version": 99, "records": {}}), encoding="utf-8")
    assert Profiles.load(future).records == {}


def test_the_track_record_reaches_the_dispatcher_in_its_own_shape():
    profiles = Profiles()
    profiles.observe_result(_done(), Role.BACKEND)
    assert (GROQ, Role.BACKEND) in profiles.as_track_record()


# ==========================================================================
# Decomposition — untrusted structure
# ==========================================================================


def test_a_plan_becomes_nodes(manifest):
    payload = json.loads(CANNED_PLAN)
    nodes = parse_nodes(payload, max_nodes=10, max_node_tokens=4096)
    assert [n.id for n in nodes] == ["server", "page"]
    assert nodes[1].deps == ("server",)


def test_the_runtime_survives_into_the_contract():
    """The field that turned a passing review into a caught bug (§11)."""
    interface = parse_interface(json.loads(CANNED_PLAN))
    assert "Windows" in interface.runtime


def test_node_ids_are_normalised_not_trusted():
    """Ids key the checkpoint, the prompt marker and the results dict."""
    payload = {"nodes": [
        {"id": "My Node!!", "title": "t", "role": "backend", "spec": "s",
         "output_path": "a.py"},
        {"id": "My Node!!", "title": "t", "role": "backend", "spec": "s",
         "output_path": "b.py"},
    ]}
    nodes = parse_nodes(payload, max_nodes=10, max_node_tokens=4096)
    ids = [n.id for n in nodes]
    # Spaces become underscores, punctuation is dropped, collisions get a suffix.
    assert ids == ["my_node", "my_node2"], ids


def test_dependencies_that_point_nowhere_are_dropped():
    payload = {"nodes": [
        {"id": "a", "title": "t", "role": "backend", "spec": "s",
         "output_path": "a.py", "deps": ["ghost", "a"]},
    ]}
    assert parse_nodes(payload, max_nodes=10, max_node_tokens=4096)[0].deps == ()


def test_an_oversized_node_is_clamped_rather_than_rejected():
    """An over-ambitious estimate is the planner being wrong about size; the
    node itself may be perfectly good."""
    payload = {"nodes": [
        {"id": "a", "title": "t", "role": "backend", "spec": "s",
         "output_path": "a.py", "est_output_tokens": 999999},
    ]}
    node = parse_nodes(payload, max_nodes=10, max_node_tokens=4096)[0]
    assert node.est_output_tokens == 4096


def test_the_node_budget_is_enforced_on_the_plan():
    payload = {"nodes": [
        {"id": f"n{i}", "title": "t", "role": "backend", "spec": "s",
         "output_path": f"{i}.py"} for i in range(20)
    ]}
    assert len(parse_nodes(payload, max_nodes=4, max_node_tokens=4096)) == 4


def test_an_unusable_plan_is_refused_outright():
    """Half-running a plan nobody can execute wastes the whole run."""
    for payload in ({}, {"nodes": []}, {"nodes": [{"id": "a"}]}):
        with pytest.raises(DecomposeError):
            parse_nodes(payload, max_nodes=10, max_node_tokens=4096)


def test_an_unknown_role_falls_back_rather_than_failing():
    payload = {"nodes": [
        {"id": "a", "title": "t", "role": "interpretive dance", "spec": "s",
         "output_path": "a.py"},
    ]}
    assert isinstance(parse_nodes(payload, max_nodes=10, max_node_tokens=4096)[0].role, Role)


def test_the_plan_signature_covers_the_roster_not_just_the_task(manifest):
    """A graph split for a 4,096-token ceiling is the wrong graph once a
    65,536-token model joins."""
    # Manifest is a pydantic model, not a dataclass.
    smaller = manifest.model_copy(update={"models": manifest.models[:2]})
    assert plan_signature("build x", manifest) != plan_signature("build x", smaller)
    assert plan_signature("build x", manifest) == plan_signature("Build X ", manifest)


def test_the_planner_is_whoever_declares_the_best_planning_affinity(manifest):
    """Not a hardcoded name: adding a stronger planner to the manifest is
    enough to change who plans."""
    candidates = [m.id for m in manifest.enabled_models]
    chosen = pick_planner(manifest, candidates)
    best = max(candidates, key=lambda m: manifest.model(m).affinity(Role.PLANNING))
    assert chosen == best


async def test_decomposing_goes_through_admission_control(manifest):
    """Planning is a request like any other; routing it around the governor
    would let the one request the run depends on blow the daily cap."""
    provider = MockProvider()
    deps = _deps(manifest, provider)
    rpd = manifest.providers["gemini"].limit(LimitKind.RPD).value
    deps.governor.restore_day_usage(GEMINI, model_requests=rpd)

    with pytest.raises(DecomposeError):
        await decompose("build a thing", deps=deps, model_id=GEMINI, max_nodes=5)


async def test_a_live_decomposition_produces_a_runnable_graph(manifest):
    provider = MockProvider()
    deps = _deps(manifest, provider)
    plan = await decompose("build a thing", deps=deps, model_id=GEMINI, max_nodes=5)

    assert [n.id for n in plan.nodes] == ["server", "page"]
    assert plan.model_id == GEMINI
    assert "Windows" in plan.interface.runtime


async def test_a_planner_that_answers_with_prose_is_refused(manifest):
    provider = MockProvider(plan_response="Sure! Here is a great plan for you.")
    deps = _deps(manifest, provider)
    with pytest.raises(DecomposeError):
        await decompose("build a thing", deps=deps, model_id=GEMINI, max_nodes=5)


# ==========================================================================
# Plan cache
# ==========================================================================


def test_a_cached_plan_costs_no_request(tmp_path):
    plan = Decomposition(
        nodes=parse_nodes(json.loads(CANNED_PLAN), max_nodes=10, max_node_tokens=4096),
        interface=parse_interface(json.loads(CANNED_PLAN)),
        model_id=GEMINI,
    )
    plancache.save("sig1", plan, task="build a thing", root=tmp_path)

    back = plancache.load("sig1", root=tmp_path)
    assert back is not None
    assert [n.id for n in back.nodes] == [n.id for n in plan.nodes]
    assert back.cached is True


def test_a_cache_miss_is_not_an_error(tmp_path):
    assert plancache.load("nothing-here", root=tmp_path) is None


def test_an_unreadable_cache_entry_is_ignored(tmp_path):
    """The run simply pays for a fresh plan rather than failing."""
    (tmp_path / "sig2.json").write_text("{not json", encoding="utf-8")
    assert plancache.load("sig2", root=tmp_path) is None

    (tmp_path / "sig3.json").write_text(
        json.dumps({"version": 99, "plan": {}}), encoding="utf-8"
    )
    assert plancache.load("sig3", root=tmp_path) is None


def test_cached_plans_are_listable_and_carry_their_task(tmp_path):
    """A cache nobody can inspect gets blamed for every strange result."""
    plan = Decomposition(
        nodes=parse_nodes(json.loads(CANNED_PLAN), max_nodes=10, max_node_tokens=4096),
        interface=parse_interface(json.loads(CANNED_PLAN)),
    )
    plancache.save("sig4", plan, task="build a link shortener", root=tmp_path)
    entries = plancache.entries(root=tmp_path)

    assert entries and entries[0][1] == "build a link shortener"
    assert plancache.forget("sig4", root=tmp_path) is True


# ==========================================================================
# Bidding — the least trusted input
# ==========================================================================


@pytest.mark.parametrize(
    "policy,nodes,models,expected",
    [
        ("never", 6, 4, False),
        ("always", 1, 4, True),
        ("auto", 6, 4, True),
        ("auto", 2, 4, False),   # fewer nodes than models: nothing to decide
        ("auto", 6, 1, False),   # one model: nothing to decide
    ],
)
def test_bidding_only_runs_when_it_would_inform_something(
    policy, nodes, models, expected
):
    fake_nodes = [object()] * nodes
    assert should_bid(policy, fake_nodes, ["m"] * models) is expected


def test_bids_for_nodes_that_do_not_exist_are_dropped():
    """Bidder output is untrusted: a phantom node id must not create work."""
    payload = {"bids": [
        {"node_id": "real", "confidence": 0.7},
        {"node_id": "invented", "confidence": 0.9},
    ]}
    bids = parse_bids(payload, GROQ, known={"real"})
    assert [b.node_id for b in bids] == ["real"]


def test_confidence_is_clamped():
    """A stray 900 would dominate a z-score computed over the bidder's own
    distribution."""
    payload = {"bids": [
        {"node_id": "a", "confidence": 900},
        {"node_id": "b", "confidence": -5},
    ]}
    bids = parse_bids(payload, GROQ, known={"a", "b"})
    assert [b.confidence for b in bids] == [1.0, 0.0]


async def test_bidding_costs_one_request_per_model_not_per_node(manifest):
    """The cost has to stay bounded by the roster, not by the size of the graph."""
    provider = MockProvider()
    deps = _deps(manifest, provider)
    nodes = parse_nodes(json.loads(CANNED_PLAN), max_nodes=10, max_node_tokens=4096)
    candidates = [m.id for m in manifest.enabled_models]

    bids = await collect_bids(nodes, deps=deps, candidates=candidates)

    assert len(provider.calls) == len(candidates)
    assert {b.model_id for b in bids} == set(candidates)


async def test_a_model_that_cannot_bid_simply_does_not(manifest):
    """Bidding is optional input; its failure must not touch the run."""
    provider = MockProvider(bid_response="I am very confident about everything!")
    deps = _deps(manifest, provider)
    nodes = parse_nodes(json.loads(CANNED_PLAN), max_nodes=10, max_node_tokens=4096)

    bids = await collect_bids(
        nodes, deps=deps, candidates=[m.id for m in manifest.enabled_models]
    )
    assert bids == []


def test_the_bid_prompt_asks_for_spread_not_confidence():
    """Uniform bragging conveys nothing, so the instruction is comparative."""
    assert "Spread your numbers" in BID_SYSTEM
    assert "your own average" in BID_SYSTEM
