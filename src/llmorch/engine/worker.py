"""Execute one node: build the prompt, call the model, verify, fail over.

The failover ladder lives here in executable form:

    transport error  -> retry the same model (transient, not a competence issue)
    429              -> resync counters, then re-select
    bad output       -> salvage locally before spending a repair request
    still failing    -> next chain rung, preferring a DIFFERENT vendor
    chain exhausted  -> DEGRADED; the run continues
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..errors import (
    LLMOrchError,
    QuotaExhausted,
    RateLimited,
    Truncated,
    Unservable,
)
from ..quota.estimator import TokenEstimator
from ..quota.governor import Governor
from ..registry.manifest import Manifest
from ..types import (
    ChatRequest,
    Message,
    NodeResult,
    NodeState,
    Priority,
    RateLimitSnapshot,
    TaskNode,
    Verdict,
)
from .blackboard import Blackboard
from .health import HealthTracker, backoff_seconds, next_model, should_retry_same_model
from .salvage import extract_code
from .verify import verify_tier0

SYSTEM_PROMPT = """\
You are one of several models collaborating on a single project. Other models \
are building the other pieces; you will never speak to them directly. The \
interface contract below is the only coordination point, so follow it exactly.

Produce ONLY the file contents for your task. No commentary, no explanation. \
If you use a code fence, use exactly one.
"""


@dataclass(slots=True)
class WorkerDeps:
    manifest: Manifest
    governor: Governor
    registry: object  # ProviderRegistry
    estimator: TokenEstimator
    health: HealthTracker
    blackboard: Blackboard
    max_retries: int = 2
    sleep: object = asyncio.sleep
    """Injectable so tests do not actually wait out a backoff."""


def build_prompt(node: TaskNode, blackboard: Blackboard) -> tuple[str, str]:
    """Return (system, user) for a node.

    The `[node:<id>]` marker lets the mock provider key its canned responses by
    node without threading extra state through the Provider protocol.
    """
    parts = [
        f"[node:{node.id}]",
        blackboard.interface_text(),
        "",
        f"## Your task: {node.title}",
        node.spec,
        "",
        f"Write the complete contents of `{node.output_path}`.",
    ]

    context = blackboard.context_for(node.needs)
    if context:
        parts.insert(2, context)

    return SYSTEM_PROMPT, "\n".join(parts)


async def execute_node(
    node: TaskNode,
    model_id: str,
    deps: WorkerDeps,
    *,
    priority: Priority = Priority.NORMAL,
) -> NodeResult:
    """Run one node to completion, failing over across vendors as needed."""
    result = NodeResult(node_id=node.id, state=NodeState.PENDING)
    tried_models: set[str] = set()
    tried_vendors: set[str] = set()
    current: str | None = model_id
    attempts_on_current = 0

    system, user = build_prompt(node, deps.blackboard)

    while current is not None:
        provider_name = deps.manifest.vendor_of(current)
        tried_models.add(current)
        tried_vendors.add(provider_name)
        result.attempts += 1
        result.state = NodeState.RUNNING

        try:
            response = await _call(node, current, system, user, deps, priority)
        except LLMOrchError as exc:
            health = deps.health.record_failure(current, exc)
            result.error = str(exc)

            if isinstance(exc, RateLimited) and exc.daily:
                deps.governor.note_daily_exhausted(current)
                deps.health.mark_exhausted(current)
            elif should_retry_same_model(exc, attempts_on_current, deps.max_retries):
                attempts_on_current += 1
                await deps.sleep(backoff_seconds(attempts_on_current))
                continue

            # Fail over — deliberately to a vendor that has not failed yet.
            current = next_model(
                deps.manifest,
                node.role,
                exclude=tried_models,
                tried_vendors=tried_vendors,
                health=deps.health,
            )
            attempts_on_current = 0
            if current is not None:
                result.state = NodeState.FALLBACK
            continue

        # Verify before accepting. Tier 0 is free and catches the common
        # free-model failures without an LLM ever being asked.
        result.state = NodeState.VERIFYING
        artifact = extract_code(response.text)
        verdict = verify_tier0(
            response.text,
            output_path=node.output_path,
            output_kind=node.output_kind,
            truncated_flag=response.truncated,
        )

        if verdict.verdict is Verdict.PASS:
            deps.health.record_success(current)
            result.state = NodeState.DONE
            result.artifact = artifact
            result.summary = _summarise(node, artifact)
            result.model_id = current
            result.usage = response.usage
            result.error = None
            result.vendors_tried = tuple(sorted(tried_vendors))
            return result

        # Failed verification: treat it as this model's failure and move on.
        failure = Truncated("output truncated") if response.truncated else LLMOrchError(
            "; ".join(i.what for i in verdict.issues) or "failed verification"
        )
        deps.health.record_failure(current, failure)
        result.error = "; ".join(i.what for i in verdict.issues)

        current = next_model(
            deps.manifest,
            node.role,
            exclude=tried_models,
            tried_vendors=tried_vendors,
            health=deps.health,
        )
        attempts_on_current = 0
        if current is not None:
            result.state = NodeState.FALLBACK

    # Chain exhausted. Degrade rather than failing the run: a quota wall or a
    # stubborn node must not discard every other model's completed work.
    result.state = NodeState.DEGRADED
    result.vendors_tried = tuple(sorted(tried_vendors))
    return result


async def _call(
    node: TaskNode,
    model_id: str,
    system: str,
    user: str,
    deps: WorkerDeps,
    priority: Priority,
):
    """One governed provider call: reserve, send, reconcile."""
    provider_name = deps.manifest.vendor_of(model_id)
    model = deps.manifest.model(model_id)

    est_prompt = deps.estimator.estimate_prompt(
        system=system, messages=[user], provider=provider_name
    )
    max_tokens = min(model.max_output, max(256, node.est_output_tokens * 2))

    ticket = deps.governor.try_acquire(
        model_id, est_prompt, max_tokens, priority=priority
    )
    if not hasattr(ticket, "ticket_id"):
        # A Denial. Terminal refusals are raised so the caller fails over
        # rather than sleeping on something that will never clear.
        if ticket.verdict.value == "unservable":
            raise Unservable(ticket.reason)
        raise QuotaExhausted(ticket.reason)

    provider = deps.registry.get(model_id)
    request = ChatRequest(
        model_id=model_id,
        messages=(Message("user", user),),
        system=system,
        max_tokens=max_tokens,
    )

    try:
        response = await provider.chat(request)
    except RateLimited as exc:
        deps.governor.release(ticket, "rate limited")
        deps.governor.sync_from_headers(
            model_id,
            RateLimitSnapshot(retry_after_s=exc.retry_after_s, daily_limit_hit=exc.daily),
        )
        raise
    except LLMOrchError:
        # Never reached the provider, or failed in transit: refund the
        # reservation so a transport blip does not permanently cost quota.
        deps.governor.release(ticket, "call failed")
        raise

    deps.governor.commit(ticket, response.usage)
    deps.estimator.observe(provider_name, response.usage.prompt_tokens, est_prompt)
    if response.rate_limit:
        deps.governor.sync_from_headers(model_id, response.rate_limit)
    return response


def _summarise(node: TaskNode, artifact: str) -> str:
    """Compact description passed to downstream nodes in place of the artifact.

    Real runs get the model's own summary from the same response; this is the
    fallback when none was supplied.
    """
    first_line = next(
        (l.strip() for l in artifact.splitlines() if l.strip()), ""
    )
    return f"{node.output_path}: {node.title}. Starts: {first_line[:80]}"
