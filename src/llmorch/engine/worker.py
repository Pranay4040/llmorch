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
from decimal import Decimal

from ..errors import (
    LLMOrchError,
    NoHealthyModel,
    RateLimited,
    Truncated,
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
    Usage,
    Verdict,
)
from .blackboard import Blackboard
from .health import HealthTracker, backoff_seconds, next_model, should_retry_same_model
from .salvage import extract_code, strip_reasoning
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
    store: object | None = None
    """LedgerStore, when the run is live. None on a dry run — the mock spends
    no real quota, and recording its traffic would have the governor refuse
    real requests tomorrow on the strength of imaginary ones."""
    run_id: str = ""
    purpose: str = "execute"
    admission_deadline_s: float = 45.0
    """How long a node may wait on a per-minute window before trying a
    different model. Sized just under the 60s window both providers use, so a
    wait that could clear does, and one that cannot gives way promptly."""


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
        # Strip inline reasoning before either step looks at the text: some
        # models put their deliberation in the message body, and both the
        # artifact and the syntax check must see the file, not the monologue.
        spoken = strip_reasoning(response.text)
        artifact = extract_code(spoken)
        verdict = verify_tier0(
            spoken,
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
    # The artifact budget, plus whatever this model spends thinking first.
    # Clamped to max_output, which is itself held under the provider's
    # per-request ceiling by the manifest.
    wanted = max(256, node.est_output_tokens * 2) + model.reasoning_headroom
    max_tokens = min(model.max_output, wanted)

    # Wait out a per-minute window rather than treating it as a refusal.
    # `acquire` raises the terminal verdicts straight through — UNSERVABLE,
    # EXHAUSTED_TODAY, COST_BLOCKED — so those still fail over immediately,
    # and only WAIT is actually slept on. Converting WAIT into a refusal (as
    # this once did) benches a healthy model for the whole run over a pause of
    # a few seconds, which against an account-scoped 30 RPM ceiling means the
    # entire roster is benched moments into the first fan-out.
    ticket = await deps.governor.acquire(
        model_id,
        est_prompt,
        max_tokens,
        priority=priority,
        deadline_s=deps.admission_deadline_s,
    )

    try:
        provider = deps.registry.get(model_id)
    except KeyError as exc:
        # No adapter for this model — a missing key, or a manifest that names a
        # provider the run never wired up. The registry raises a bare KeyError,
        # which would sail past every `except LLMOrchError` in the failover
        # ladder and take the whole run with it. Convert it, so the node fails
        # over like any other and the run survives.
        deps.governor.release(ticket, "no provider registered")
        raise NoHealthyModel(f"no provider is registered for {model_id}") from exc

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
        _record(deps, node, model_id, ticket, exc)
        raise
    except LLMOrchError as exc:
        # Never reached the provider, or failed in transit: refund the
        # reservation so a transport blip does not permanently cost quota.
        deps.governor.release(ticket, "call failed")
        _record(deps, node, model_id, ticket, exc)
        raise

    cost = deps.manifest.provider_of(model_id).cost_for(
        response.usage.prompt_tokens, response.usage.completion_tokens
    )
    deps.governor.commit(ticket, response.usage, cost)
    deps.estimator.observe(provider_name, response.usage.prompt_tokens, est_prompt)
    if response.rate_limit:
        deps.governor.sync_from_headers(model_id, response.rate_limit)
    _record(deps, node, model_id, ticket, None, response=response, cost=cost)
    return response


def _record(
    deps: WorkerDeps,
    node: TaskNode,
    model_id: str,
    ticket,
    error: Exception | None,
    *,
    response=None,
    cost=None,
) -> None:
    """Append this call to the durable ledger, when there is one.

    Failures are recorded too, but only the ones that actually reached the
    provider: a rejected request still consumed a slot with the rate limiter,
    whereas a connection that never opened consumed nothing and would inflate
    tomorrow's picture of today. `ProviderError.status` is what separates them
    — it is set only once a server answered.

    Ledger writes never propagate. A disk problem is a reporting failure, and
    losing the run over it would throw away every artifact already built.
    """
    if deps.store is None:
        return

    reached_provider = error is None or getattr(error, "status", None) is not None
    if not reached_provider:
        return

    from ..quota.store import build_event  # local: keeps the engine import-light

    provider = deps.manifest.provider_of(model_id)
    usage = response.usage if response is not None else Usage()

    try:
        deps.store.record(
            build_event(
                run_id=deps.run_id,
                node_id=node.id,
                purpose=deps.purpose,
                provider=provider.name,
                model_id=model_id,
                reset_tz=provider.reset_tz,
                est_prompt_tokens=ticket.est_prompt_tokens,
                est_completion_tokens=ticket.est_completion_tokens,
                usage=usage,
                cost_usd=cost if cost is not None else Decimal("0"),
                ok=error is None,
                http_status=(
                    response.raw_status
                    if response is not None
                    else int(getattr(error, "status", 0) or 0)
                ),
                latency_ms=response.latency_ms if response is not None else 0,
                error=None if error is None else str(error)[:500],
            )
        )
    except Exception:  # pragma: no cover - reporting must never fail a run
        pass


def _summarise(node: TaskNode, artifact: str) -> str:
    """Compact description passed to downstream nodes in place of the artifact.

    Real runs get the model's own summary from the same response; this is the
    fallback when none was supplied.
    """
    first_line = next(
        (l.strip() for l in artifact.splitlines() if l.strip()), ""
    )
    return f"{node.output_path}: {node.title}. Starts: {first_line[:80]}"
