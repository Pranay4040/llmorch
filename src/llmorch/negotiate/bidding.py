"""The bidding round: each model rates itself on each node.

This is the input the design trusts least, and deliberately keeps anyway.

Asked to rate their own competence, free models claim high confidence at
everything — which is why the bid never decides anything on its own. It is
z-normalised within each bidder before it reaches the dispatcher, so what
survives is not "how good do you say you are" but **"which of these nodes do you
prefer, relative to your own baseline"**. A model that says 0.9 to everything
contributes nothing; a model that says 0.9 to the schema and 0.6 to the CSS has
told us something real, whatever its absolute numbers mean.

The cost is bounded and small: one request per model, ever, for the whole run —
not one per node. Bidding is also entirely optional. Every failure path here
returns fewer bids rather than raising, and the dispatcher falls back on the
hand-written capability sheet, which is the more trustworthy input anyway.
"""

from __future__ import annotations

from ..errors import LLMOrchError
from ..engine.salvage import extract_json
from ..types import Bid, ChatRequest, Message, Priority, TaskNode, Ticket

BID_MAX_TOKENS = 1200

BID_SCHEMA = {
    "type": "object",
    "properties": {
        "bids": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "confidence": {"type": "number"},
                    "est_output_tokens": {"type": "integer"},
                    "why": {"type": "string"},
                },
                "required": ["node_id", "confidence"],
            },
        }
    },
    "required": ["bids"],
}

BID_SYSTEM = """\
You are one of several models about to divide a build between you. For each \
task below, rate how well suited *you specifically* are to it, from 0 to 1.

Spread your numbers. These ratings are compared against your own average, not \
against the other models', so rating everything 0.9 tells us nothing and rating \
your strongest task above your weakest tells us everything. Rate honestly: you \
gain nothing by claiming work you would do badly, and the assignment is made by \
an algorithm that can overrule you.

Also estimate how many output tokens each task would actually take you.

Reply with ONLY a JSON object: \
{"bids": [{"node_id": "...", "confidence": 0.0, "est_output_tokens": 0, "why": "..."}]}
"""


def build_bid_prompt(nodes: list[TaskNode], interface_text: str) -> str:
    lines = ["[bid]", interface_text, "", "## The tasks"]
    for node in nodes:
        lines.append(
            f"- id: {node.id} | role: {node.role.value} | {node.title}"
            f" -> {node.output_path}"
        )
        lines.append(f"    {node.spec[:300]}")
    lines.append("")
    lines.append("Rate every task by id.")
    return "\n".join(lines)


def parse_bids(payload: dict, model_id: str, known: set[str]) -> list[Bid]:
    """Read a bidder's reply, keeping only what is usable.

    Bidder output is untrusted data: unknown node ids are dropped rather than
    creating phantom nodes, and confidences are clamped rather than allowed to
    dominate a z-score with a 900 someone typed by accident.
    """
    bids: list[Bid] = []
    for raw in payload.get("bids") or []:
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("node_id") or "").strip()
        if node_id not in known:
            continue
        try:
            confidence = float(raw.get("confidence"))
        except (TypeError, ValueError):
            continue
        try:
            tokens = int(raw.get("est_output_tokens") or 0)
        except (TypeError, ValueError):
            tokens = 0
        bids.append(
            Bid(
                model_id=model_id,
                node_id=node_id,
                confidence=max(0.0, min(1.0, confidence)),
                est_output_tokens=max(0, tokens),
                why=str(raw.get("why") or "")[:200],
            )
        )
    return bids


async def collect_bids(
    nodes: list[TaskNode],
    *,
    deps,
    candidates: list[str],
) -> list[Bid]:
    """One bid request per model, in sequence, all of them optional.

    Sequential rather than concurrent on purpose: bidding is the least valuable
    traffic in the run, and firing every model at once is the surest way to fill
    a per-minute window that the actual work is about to need.
    """
    if not nodes or not candidates:
        return []

    known = {n.id for n in nodes}
    user = build_bid_prompt(nodes, deps.blackboard.interface_text())
    collected: list[Bid] = []

    for model_id in candidates:
        if not deps.health.is_available(model_id) or model_id not in deps.registry:
            continue

        model = deps.manifest.model(model_id)
        provider_name = deps.manifest.vendor_of(model_id)
        est_prompt = deps.estimator.estimate_prompt(
            system=BID_SYSTEM, messages=[user], provider=provider_name
        )
        max_tokens = min(
            model.max_output, max(model.min_output_tokens, BID_MAX_TOKENS)
        )

        # NORMAL priority: a bid must never consume headroom the work itself
        # will need. A model too busy to bid simply does not bid.
        ticket = deps.governor.try_acquire(
            model_id, est_prompt, max_tokens, priority=Priority.NORMAL
        )
        if not isinstance(ticket, Ticket):
            continue

        try:
            response = await deps.registry.get(model_id).chat(
                ChatRequest(
                    model_id=model_id,
                    messages=(Message("user", user),),
                    system=BID_SYSTEM,
                    max_tokens=max_tokens,
                    json_schema=BID_SCHEMA if model.supports_json_schema else None,
                )
            )
        except LLMOrchError:
            deps.governor.release(ticket, "bid failed")
            # A model that cannot bid has simply declined to; its static
            # affinities still speak for it.
            continue

        deps.governor.commit(ticket, response.usage)
        deps.estimator.observe(provider_name, response.usage.prompt_tokens, est_prompt)
        if response.rate_limit:
            deps.governor.sync_from_headers(model_id, response.rate_limit)

        payload = extract_json(response.text)
        if isinstance(payload, dict):
            collected.extend(parse_bids(payload, model_id, known))

    return collected


def should_bid(policy: str, nodes: list[TaskNode], candidates: list[str]) -> bool:
    """Whether a bidding round is worth its requests.

    `auto` — the default — skips it when there is nothing to decide: one model,
    or fewer nodes than models. Bidding to allocate two nodes between four
    models spends four requests to inform a choice the capability sheet already
    makes well.
    """
    if policy == "never" or not nodes or len(candidates) < 2:
        return False
    if policy == "always":
        return True
    return len(nodes) >= len(candidates)
