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
    CostBlocked,
    LLMOrchError,
    QuotaExhausted,
    RateLimited,
    Truncated,
    Unservable,
)
from ..quota.estimator import TokenEstimator
from ..quota.governor import Governor
from ..quota.store import LedgerStore, make_event
from ..registry.manifest import Manifest
from ..types import (
    Admission,
    ChatRequest,
    Message,
    NodeResult,
    NodeState,
    Priority,
    RateLimitSnapshot,
    TaskNode,
    Ticket,
    Usage,
    Verdict,
)
from .blackboard import Blackboard
from .health import HealthTracker, backoff_seconds, next_model, should_retry_same_model
from .review import repair_artifact, review_artifact, should_review
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
    max_escalations: int = 2
    """Times a truncated node may be retried with a larger output budget before
    the model is blamed. Bounded: a node whose spec genuinely cannot be met
    would otherwise walk the budget up to the ceiling on every model in turn."""
    review: str = "off"
    """off | code | all — which nodes get a cross-vendor Tier 1 read."""
    max_repairs: int = 1
    """Repair attempts per node. Bounded, or a harsh reviewer and a stubborn
    author trade requests until the daily budget is gone."""
    ledger: LedgerStore | None = None
    """Persistent usage ledger. None on a dry run — mock calls consume no
    real quota, and recording them would corrupt tomorrow's admission maths."""
    run_id: str = ""
    """Ledger key for this run. Blank on a dry run, which never records."""


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
    # A property of the node, not of the model: a file that did not fit in
    # 2,000 tokens will not fit for the next model either, so the larger budget
    # travels with the work rather than resetting on failover.
    escalation = 0

    system, user = build_prompt(node, deps.blackboard)

    while current is not None:
        provider_name = deps.manifest.vendor_of(current)
        tried_models.add(current)
        tried_vendors.add(provider_name)
        result.attempts += 1
        result.state = NodeState.RUNNING

        try:
            response = await _call(
                node, current, system, user, deps, priority, escalation
            )
        except LLMOrchError as exc:
            health = deps.health.record_failure(current, exc)
            result.error = str(exc)

            if isinstance(exc, RateLimited) and exc.daily:
                deps.governor.note_daily_exhausted(current)
                deps.health.mark_exhausted(current)
            elif should_retry_same_model(exc, attempts_on_current, deps.max_retries):
                attempts_on_current += 1
                # A rate limiter says exactly how long it needs; guessing an
                # exponential backoff either wastes time or wakes up too early
                # and spends another request discovering the window is still
                # full.
                asked = getattr(exc, "retry_after_s", None) or 0.0
                await deps.sleep(max(asked, backoff_seconds(attempts_on_current)))
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
            # Tier 0 only proves the file parses. A second vendor decides
            # whether it does what it was asked to do.
            artifact, review = await _tier1(node, artifact, current, deps)

            if review is not None and review.verdict is Verdict.REJECT:
                # A different vendor says this is not the file that was asked
                # for. Treat it as this model's failure and try another.
                deps.health.record_failure(
                    current, LLMOrchError("rejected by cross-vendor review")
                )
                result.error = "; ".join(i.what for i in review.issues) or "rejected"
                result.review = review
                current = next_model(
                    deps.manifest, node.role, exclude=tried_models,
                    tried_vendors=tried_vendors, health=deps.health,
                )
                attempts_on_current = 0
                if current is not None:
                    result.state = NodeState.FALLBACK
                continue

            deps.health.record_success(current)
            result.state = NodeState.DONE
            result.artifact = artifact
            result.summary = _summarise(node, artifact)
            result.model_id = current
            result.usage = response.usage
            result.error = None
            result.review = review
            result.vendors_tried = tuple(sorted(tried_vendors))
            return result

        # Truncation is evidence about the budget, not about the model. The
        # estimate was made before any of the work existed, and it is
        # systematically low for exactly the files nobody can size in advance.
        # Failing over here sends the same too-small budget to the next model,
        # which truncates identically — which is how a node burns every vendor
        # in its chain and degrades over an arithmetic error.
        if (
            response.truncated
            and escalation < deps.max_escalations
            and can_grow(node, deps.manifest.model(current), escalation)
        ):
            escalation += 1
            grown = output_budget(node, deps.manifest.model(current), escalation)
            result.error = f"output truncated; retrying with {grown} tokens"
            deps.health.note(f"{node.id}: output budget raised to {grown}")
            continue

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


async def _tier1(node: TaskNode, artifact: str, model_id: str, deps: WorkerDeps):
    """Review the artifact, and repair it once if the reviewer found faults.

    Returns (artifact, review). The artifact only changes if a repair both
    happened and passed Tier 0 — a repair that breaks the file is worse than
    the fault it was fixing.
    """
    if not should_review(node, deps.review):
        return artifact, None

    review = await review_artifact(
        node, artifact, author_model_id=model_id, deps=deps
    )
    if review is None or review.verdict is not Verdict.REVISE:
        return artifact, review

    if deps.max_repairs > 0:
        repaired = await repair_artifact(
            node, artifact, review, author_model_id=model_id, deps=deps
        )
        if repaired:
            # Not re-reviewed: one round only, or the budget goes to arguing.
            return repaired, review
    return artifact, review


def output_budget(node: TaskNode, model, escalation: int = 0) -> int:
    """How many output tokens to ask for, and how that grows after a truncation.

    The base is the planner's estimate doubled, floored for models that charge
    invisible reasoning tokens against the same budget — ask only for the
    visible answer there and the whole reply comes back cut off with nothing in
    it.

    The estimate is a guess made before any of the work was done, and it is
    systematically low for the files that are hardest to size in advance: test
    suites, full pages, anything enumerating cases. So a truncation doubles the
    budget rather than condemning the model, up to whatever the model itself
    allows.
    """
    base = max(model.min_output_tokens, 256, node.est_output_tokens * 2)
    return min(model.max_output, base * (2 ** max(0, escalation)))


def can_grow(node: TaskNode, model, escalation: int) -> bool:
    """Whether asking again with a bigger budget could change the outcome."""
    return output_budget(node, model, escalation) < model.max_output


async def _call(
    node: TaskNode,
    model_id: str,
    system: str,
    user: str,
    deps: WorkerDeps,
    priority: Priority,
    escalation: int = 0,
):
    """One governed provider call: reserve, send, reconcile."""
    provider_name = deps.manifest.vendor_of(model_id)
    model = deps.manifest.model(model_id)

    est_prompt = deps.estimator.estimate_prompt(
        system=system, messages=[user], provider=provider_name
    )
    max_tokens = output_budget(node, model, escalation)

    ticket = deps.governor.try_acquire(
        model_id, est_prompt, max_tokens, priority=priority
    )
    if not isinstance(ticket, Ticket):
        # Each refusal means something different, and collapsing them is how a
        # model that is busy for nine seconds gets written off for the day.
        match ticket.verdict:
            case Admission.UNSERVABLE:
                # Never fits on this model, however long anyone waits.
                raise Unservable(ticket.reason)
            case Admission.EXHAUSTED_TODAY:
                # Gone until the provider's next local midnight.
                raise QuotaExhausted(ticket.reason)
            case Admission.COST_BLOCKED:
                raise CostBlocked(ticket.reason)
            case _:
                # WAIT: a per-minute window, clearing in seconds. Raised as a
                # retryable rate limit so the ladder backs off and retries,
                # rather than marking a perfectly healthy model exhausted.
                raise RateLimited(ticket.reason, retry_after_s=ticket.retry_after_s)

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
        # A 429 still reached the provider, so it is recorded: on most vendors
        # a refused request has already consumed its slot for the day.
        _record(deps, node, model_id, est_prompt, max_tokens, exc)
        raise
    except LLMOrchError as exc:
        # Never reached the provider, or failed in transit: refund the
        # reservation so a transport blip does not permanently cost quota.
        deps.governor.release(ticket, "call failed")
        _record(deps, node, model_id, est_prompt, max_tokens, exc)
        raise

    deps.governor.commit(ticket, response.usage)
    deps.estimator.observe(provider_name, response.usage.prompt_tokens, est_prompt)
    if response.rate_limit:
        deps.governor.sync_from_headers(model_id, response.rate_limit)
    _record(
        deps,
        node,
        model_id,
        est_prompt,
        max_tokens,
        None,
        usage=response.usage,
        latency_ms=response.latency_ms,
        status=response.raw_status,
    )
    return response


def _record(
    deps: WorkerDeps,
    node: TaskNode,
    model_id: str,
    est_prompt: int,
    est_completion: int,
    error: Exception | None,
    *,
    usage: Usage | None = None,
    latency_ms: int = 0,
    status: int = 200,
) -> None:
    """Append one call to the persistent ledger, if there is one.

    Failures are recorded as deliberately as successes. A run that burned forty
    requests on a model that was 404ing the whole time looks identical to an
    idle day unless the failures are on the record.
    """
    if deps.ledger is None:
        return

    if error is not None:
        # A transport failure never reached the server, so it cost no quota;
        # status 0 is what the ledger's request count keys off.
        status = getattr(error, "status", None) or 0
    deps.ledger.record(
        make_event(
            run_id=deps.run_id,
            node_id=node.id,
            purpose="execute",
            manifest=deps.manifest,
            model_id=model_id,
            usage=usage or Usage(),
            est_prompt_tokens=est_prompt,
            est_completion_tokens=est_completion,
            ok=error is None,
            http_status=status,
            latency_ms=latency_ms,
            error=str(error) if error is not None else None,
        )
    )


def _summarise(node: TaskNode, artifact: str) -> str:
    """Compact description passed to downstream nodes in place of the artifact.

    Real runs get the model's own summary from the same response; this is the
    fallback when none was supplied.
    """
    first_line = next(
        (l.strip() for l in artifact.splitlines() if l.strip()), ""
    )
    return f"{node.output_path}: {node.title}. Starts: {first_line[:80]}"
