"""The dispatcher: decides which model gets which task.

Deterministic Python, not an LLM. That choice buys four things an LLM manager
cannot offer: it costs no requests, it runs instantly, it is testable offline,
and it cannot hallucinate an assignment that violates a rate limit.

Four inputs, in decreasing order of trustworthiness:

1. **Feasibility** (hard filter) — can this model physically serve this node?
2. **Role affinity** — hand-written priors from the manifest.
3. **Track record** — how the model has actually performed at this role.
4. **Self-reported bids** — what the model claims, z-normalised within each
   bidder so that uniform bragging conveys nothing.

Plus a quota-pressure penalty, so a model near its daily wall deprioritises
itself without needing a special case anywhere else.

"Split evenly" is enforced here as a capacity constraint measured in tokens.
Asking a model to divide work fairly does not produce fair division; an
assignment algorithm with a capacity bound does.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from ..engine.graph import TaskGraph
from ..registry.manifest import Manifest
from ..types import Assignment, Bid, Role, ScoreBreakdown, TaskNode

# A bidder whose confidences barely vary is telling us nothing. Below this
# spread its bids are discarded and the static priors carry the decision.
MIN_CONFIDENCE_SPREAD = 0.05


@dataclass(slots=True)
class ReconcileInput:
    graph: TaskGraph
    manifest: Manifest
    candidates: list[str]
    """Model ids eligible for this run, already filtered for health/enablement."""
    bids: list[Bid] = field(default_factory=list)
    track_record: dict[tuple[str, Role], float] = field(default_factory=dict)
    quota_pressure: dict[str, float] = field(default_factory=dict)
    """model_id -> 0..1, how close the model is to its daily wall."""
    imbalance_tolerance: float = 0.35


@dataclass(slots=True)
class ReconcileResult:
    assignments: dict[str, Assignment] = field(default_factory=dict)
    unassigned: list[str] = field(default_factory=list)
    """Nodes no candidate can physically serve — these degrade."""
    notes: list[str] = field(default_factory=list)

    def model_for(self, node_id: str) -> str | None:
        a = self.assignments.get(node_id)
        return a.model_id if a else None

    def token_share(self, graph: TaskGraph) -> dict[str, int]:
        share: dict[str, int] = defaultdict(int)
        for node_id, a in self.assignments.items():
            share[a.model_id] += graph.nodes[node_id].est_output_tokens
        return dict(share)


# --------------------------------------------------------------------------
# Feasibility
# --------------------------------------------------------------------------


def is_feasible(manifest: Manifest, model_id: str, node: TaskNode) -> bool:
    """Can this model serve this node at all?

    Applied as a hard filter before scoring rather than as a penalty. A model
    that physically cannot produce the output should be eliminated, not merely
    ranked lower — otherwise a strong enough score could still select it.
    """
    model = manifest.model(model_id)
    if node.est_output_tokens > model.max_output:
        return False

    # Prompt overhead: the spec, the interface contract, and upstream summaries.
    est_prompt = 400 + len(node.spec) // 4
    if est_prompt + node.est_output_tokens > manifest.max_request_tokens(model_id):
        return False

    return True


# --------------------------------------------------------------------------
# Bid normalisation
# --------------------------------------------------------------------------


def normalise_bids(bids: list[Bid]) -> dict[tuple[str, str], float]:
    """Z-score each bidder's confidences against its own distribution.

    This is the defence against overclaiming. Free models tend to report high
    confidence on everything; the useful signal is not the absolute number but
    which nodes a model rates *above its own average*. A model that answers 0.95
    to every question ends up contributing a flat zero across the board rather
    than dominating the assignment.
    """
    by_model: dict[str, list[Bid]] = defaultdict(list)
    for bid in bids:
        by_model[bid.model_id].append(bid)

    out: dict[tuple[str, str], float] = {}
    for model_id, model_bids in by_model.items():
        confidences = [b.confidence for b in model_bids]
        if len(confidences) < 2:
            continue

        spread = statistics.pstdev(confidences)
        if spread < MIN_CONFIDENCE_SPREAD:
            # Uninformative bidder: contributes nothing rather than noise.
            for bid in model_bids:
                out[(model_id, bid.node_id)] = 0.0
            continue

        mean = statistics.fmean(confidences)
        for bid in model_bids:
            z = (bid.confidence - mean) / spread
            out[(model_id, bid.node_id)] = max(-2.0, min(2.0, z)) / 2.0

    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score_pair(
    inp: ReconcileInput,
    normalised: dict[tuple[str, str], float],
    model_id: str,
    node: TaskNode,
) -> ScoreBreakdown:
    model = inp.manifest.model(model_id)
    return ScoreBreakdown(
        z_confidence=normalised.get((model_id, node.id), 0.0),
        role_affinity=model.affinity(node.role),
        track_record=inp.track_record.get((model_id, node.role), 0.5),
        quality_prior=model.quality_prior,
        quota_pressure=inp.quota_pressure.get(model_id, 0.0),
    )


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------


def reconcile(inp: ReconcileInput) -> ReconcileResult:
    """Assign every node to a model, balancing quality against fair share."""
    result = ReconcileResult()
    normalised = normalise_bids(inp.bids)

    if not inp.candidates:
        result.unassigned = sorted(inp.graph.nodes)
        result.notes.append("no candidate models available")
        return result

    # 1. Feasible (model, node) pairs, scored.
    scored: dict[str, list[tuple[float, str, ScoreBreakdown]]] = {}
    for node_id, node in inp.graph.nodes.items():
        options: list[tuple[float, str, ScoreBreakdown]] = []
        for model_id in inp.candidates:
            if not is_feasible(inp.manifest, model_id, node):
                continue
            breakdown = score_pair(inp, normalised, model_id, node)
            options.append((breakdown.total, model_id, breakdown))
        if not options:
            result.unassigned.append(node_id)
            result.notes.append(
                f"{node_id}: no candidate can serve ~{node.est_output_tokens} "
                "output tokens within its provider's per-request ceiling"
            )
            continue
        options.sort(key=lambda t: (-t[0], t[1]))
        scored[node_id] = options

    if not scored:
        return result

    # 2. Fair-share capacity, measured in tokens rather than node count.
    #    Two nodes of wildly different size are not an even split.
    #
    #    The floor matters: with few nodes, an even share can come out smaller
    #    than a single node, and a cap no node can satisfy is not a constraint —
    #    it just sends every node down the "best available" path and silently
    #    disables balancing altogether.
    eligible = {m for opts in scored.values() for _, m, _ in opts}
    total_tokens = sum(inp.graph.nodes[n].est_output_tokens for n in scored)
    largest_node = max(inp.graph.nodes[n].est_output_tokens for n in scored)
    per_model_cap = max(
        float(largest_node),
        total_tokens / max(1, len(eligible)) * (1 + inp.imbalance_tolerance),
    )

    # 3. Greedy by descending score, subject to capacity. Largest nodes first,
    #    since they are the hardest to place once capacity is committed.
    load: dict[str, float] = defaultdict(float)
    order = sorted(
        scored, key=lambda n: -inp.graph.nodes[n].est_output_tokens
    )

    for node_id in order:
        node = inp.graph.nodes[node_id]
        placed = False
        for score, model_id, breakdown in scored[node_id]:
            if load[model_id] + node.est_output_tokens <= per_model_cap:
                result.assignments[node_id] = Assignment(
                    node_id=node_id,
                    model_id=model_id,
                    score=score,
                    breakdown=breakdown,
                    rationale=_rationale(breakdown, model_id, node.role),
                )
                load[model_id] += node.est_output_tokens
                placed = True
                break

        if not placed:
            # Everyone is at capacity: take the best option regardless, since a
            # slightly uneven split beats refusing to do the work.
            score, model_id, breakdown = scored[node_id][0]
            result.assignments[node_id] = Assignment(
                node_id=node_id,
                model_id=model_id,
                score=score,
                breakdown=breakdown,
                rationale=_rationale(breakdown, model_id, node.role)
                + "; capacity exceeded, best available",
            )
            load[model_id] += node.est_output_tokens

    # 4. 2-opt: swap pairs where both nodes end up better off.
    _two_opt(inp, scored, result, load, per_model_cap)

    return result


def _two_opt(
    inp: ReconcileInput,
    scored: dict[str, list[tuple[float, str, ScoreBreakdown]]],
    result: ReconcileResult,
    load: dict[str, float],
    cap: float,
) -> None:
    """Improve total score by swapping assignments, never violating capacity.

    Greedy placement commits early and can strand a strong pairing; a swap pass
    recovers most of that at negligible cost.
    """
    node_ids = sorted(result.assignments)
    improved = True
    passes = 0

    while improved and passes < 3:
        improved = False
        passes += 1
        for i, a_id in enumerate(node_ids):
            for b_id in node_ids[i + 1 :]:
                a, b = result.assignments[a_id], result.assignments[b_id]
                if a.model_id == b.model_id:
                    continue

                a_alt = _option(scored, a_id, b.model_id)
                b_alt = _option(scored, b_id, a.model_id)
                if a_alt is None or b_alt is None:
                    continue

                if a_alt[0] + b_alt[0] <= a.score + b.score + 1e-9:
                    continue

                a_tokens = inp.graph.nodes[a_id].est_output_tokens
                b_tokens = inp.graph.nodes[b_id].est_output_tokens
                new_a_load = load[a.model_id] - a_tokens + b_tokens
                new_b_load = load[b.model_id] - b_tokens + a_tokens
                if new_a_load > cap or new_b_load > cap:
                    continue

                result.assignments[a_id] = Assignment(
                    a_id, b.model_id, a_alt[0], a_alt[1], a.rationale + "; swapped"
                )
                result.assignments[b_id] = Assignment(
                    b_id, a.model_id, b_alt[0], b_alt[1], b.rationale + "; swapped"
                )
                load[a.model_id], load[b.model_id] = new_a_load, new_b_load
                improved = True


def _option(
    scored: dict[str, list[tuple[float, str, ScoreBreakdown]]],
    node_id: str,
    model_id: str,
) -> tuple[float, ScoreBreakdown] | None:
    for score, mid, breakdown in scored.get(node_id, []):
        if mid == model_id:
            return score, breakdown
    return None


def _rationale(breakdown: ScoreBreakdown, model_id: str, role: Role) -> str:
    """Human-readable explanation, surfaced by `llmorch plan --explain`."""
    parts = [f"{role.value} affinity {breakdown.role_affinity:.2f}"]
    if breakdown.z_confidence > 0.1:
        parts.append(f"bid above its own average (+{breakdown.z_confidence:.2f})")
    elif breakdown.z_confidence < -0.1:
        parts.append(f"bid below its own average ({breakdown.z_confidence:.2f})")
    if breakdown.quota_pressure > 0.3:
        parts.append(f"quota pressure {breakdown.quota_pressure:.2f}")
    return ", ".join(parts)
