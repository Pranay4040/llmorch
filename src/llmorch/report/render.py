"""Text rendering for plans, spend reports, and quota headroom."""

from __future__ import annotations

from ..engine.graph import TaskGraph
from ..engine.scheduler import RunOutcome
from ..negotiate.reconcile import ReconcileResult
from ..types import Headroom, NodeState


def _bar(used: int, limit: int | None, width: int = 20) -> str:
    if not limit:
        return "—"
    filled = min(width, round(width * used / limit))
    return "█" * filled + "░" * (width - filled)


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m"


def render_plan(
    plan: ReconcileResult, graph: TaskGraph, *, explain: bool = False
) -> str:
    """The assignment table — which model got which job, and why."""
    lines = ["", "Assignment", "=" * 78]

    if not plan.assignments:
        lines.append("  (nothing could be assigned)")
    else:
        lines.append(f"  {'node':<10} {'role':<11} {'model':<24} {'tokens':>7}  score")
        lines.append("  " + "-" * 74)
        for node_id in sorted(plan.assignments):
            a = plan.assignments[node_id]
            node = graph.nodes[node_id]
            lines.append(
                f"  {node_id:<10} {node.role.value:<11} {a.model_id:<24} "
                f"{node.est_output_tokens:>7}  {a.score:.3f}"
            )
            if explain:
                b = a.breakdown
                lines.append(
                    f"      {a.rationale}"
                )
                lines.append(
                    f"      z_conf {b.z_confidence:+.2f} · affinity {b.role_affinity:.2f}"
                    f" · record {b.track_record:.2f} · prior {b.quality_prior:.2f}"
                    f" · quota −{b.quota_pressure:.2f}"
                )

    share = plan.token_share(graph)
    if share:
        total = sum(share.values())
        lines += ["", "Token share (the 'even split' the assignment enforces)"]
        for model_id in sorted(share, key=lambda m: -share[m]):
            pct = 100 * share[model_id] / total
            lines.append(
                f"  {model_id:<24} {share[model_id]:>6} tokens  {pct:5.1f}%  "
                f"{_bar(share[model_id], max(share.values()), 16)}"
            )

    if plan.unassigned:
        lines += ["", f"Unassigned ({len(plan.unassigned)}):"]
        lines += [f"  {n}" for n in plan.unassigned]
    for note in plan.notes:
        lines.append(f"  ! {note}")

    return "\n".join(lines)


def render_outcome(outcome: RunOutcome, graph: TaskGraph) -> str:
    """Per-node execution result."""
    lines = ["", "Execution", "=" * 78]
    lines.append(f"  {'node':<10} {'state':<10} {'model':<24} {'try':>4}  vendors")
    lines.append("  " + "-" * 74)

    for node_id in sorted(outcome.results):
        r = outcome.results[node_id]
        mark = {
            NodeState.DONE: "ok",
            NodeState.DEGRADED: "DEGRADED",
        }.get(r.state, r.state.value)
        lines.append(
            f"  {node_id:<10} {mark:<10} {(r.model_id or '—'):<24} "
            f"{r.attempts:>4}  {', '.join(r.vendors_tried) or '—'}"
        )
        if r.state is NodeState.DEGRADED and r.error:
            lines.append(f"      reason: {r.error}")

    done, degraded = len(outcome.completed), len(outcome.degraded)
    lines += ["", f"  {done} completed, {degraded} degraded"]

    for note in outcome.reassignments:
        lines.append(f"  ↻ {note}")

    return "\n".join(lines)


def render_spend(outcome: RunOutcome) -> str:
    """Token accounting, including the headline efficiency metric."""
    by_model: dict[str, tuple[int, int, int]] = {}
    for r in outcome.results.values():
        if not r.model_id:
            continue
        p, c, n = by_model.get(r.model_id, (0, 0, 0))
        by_model[r.model_id] = (
            p + r.usage.prompt_tokens,
            c + r.usage.completion_tokens,
            n + 1,
        )

    lines = ["", "Spend", "=" * 78]
    if not by_model:
        lines.append("  (nothing spent)")
        return "\n".join(lines)

    lines.append(f"  {'model':<24} {'calls':>6} {'prompt':>9} {'output':>9}")
    lines.append("  " + "-" * 52)
    total_prompt = total_completion = 0
    for model_id in sorted(by_model):
        p, c, n = by_model[model_id]
        total_prompt += p
        total_completion += c
        lines.append(f"  {model_id:<24} {n:>6} {p:>9,} {c:>9,}")
    lines.append("  " + "-" * 52)
    lines.append(f"  {'total':<24} {'':>6} {total_prompt:>9,} {total_completion:>9,}")

    # Useful output = completion tokens from nodes that actually landed.
    useful = sum(
        r.usage.completion_tokens
        for r in outcome.results.values()
        if r.state is NodeState.DONE
    )
    spent = total_prompt + total_completion
    if spent:
        lines += [
            "",
            f"  Quota efficiency: {100 * useful / spent:.1f}% "
            f"({useful:,} useful of {spent:,} spent)",
            "  Retries, repairs and degraded nodes count as waste.",
        ]
    return "\n".join(lines)


def render_quota(headroom: dict[str, Headroom]) -> str:
    """Per-provider headroom, each in its own reset timezone."""
    lines = ["", "Quota", "=" * 78]
    lines.append(
        f"  {'model':<24} {'requests':>13}  {'per-minute tokens':>18}  resets in"
    )
    lines.append("  " + "-" * 74)

    for model_id in sorted(headroom):
        h = headroom[model_id]
        req = f"{h.requests_used}/{h.requests_limit}" if h.requests_limit else "—"
        tok = (
            f"{h.tokens_used_minute}/{h.tokens_limit_minute}"
            if h.tokens_limit_minute
            else "—"
        )
        lines.append(
            f"  {model_id:<24} {req:>13}  {tok:>18}  "
            f"{_duration(h.seconds_to_reset)} ({h.reset_tz})"
        )
    return "\n".join(lines)


def render_warnings(warnings: list[str]) -> str:
    if not warnings:
        return ""
    lines = ["", "Notes", "=" * 78]
    lines += [f"  · {w}" for w in warnings]
    return "\n".join(lines)
