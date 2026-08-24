"""Text rendering for the durable ledger.

`report/render.py` shows what one run did. This shows what the *account* has
done — across runs, across days, and against the walls each provider will
enforce tomorrow morning. The two answer different questions: a run can look
perfectly efficient while having spent the last of a daily allowance that four
queued runs are about to need.
"""

from __future__ import annotations

from decimal import Decimal

from ..quota.store import DayUsage, RunSummary
from ..types import LimitKind
from ..registry.manifest import Manifest


def _short_run(run_id: str, width: int = 17) -> str:
    return run_id if len(run_id) <= width else run_id[: width - 1] + "…"


def _percent_bar(used: int, limit: int | None, width: int = 14) -> str:
    if not limit:
        return "—"
    filled = min(width, round(width * used / limit))
    return "█" * filled + "░" * (width - filled)


def render_runs(runs: tuple[RunSummary, ...], *, limit: int = 20) -> str:
    """Run history, newest first."""
    lines = ["", "Run history", "=" * 78]

    if not runs:
        lines.append("  (the ledger is empty — no live run has been recorded yet)")
        return "\n".join(lines)

    lines.append(
        f"  {'run':<18} {'when (UTC)':<20} {'calls':>6} {'tokens':>10} "
        f"{'models':>7} {'fail':>5}"
    )
    lines.append("  " + "-" * 74)

    for run in runs[:limit]:
        tokens = run.prompt_tokens + run.completion_tokens
        lines.append(
            f"  {_short_run(run.run_id):<18} {run.started_utc[:19]:<20} "
            f"{run.requests:>6} {tokens:>10,} {run.models:>7} {run.failures:>5}"
        )

    total_calls = sum(r.requests for r in runs)
    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in runs)
    total_cost = sum((r.cost_usd for r in runs), Decimal("0"))

    lines.append("  " + "-" * 74)
    lines.append(
        f"  {'total':<18} {'':<20} {total_calls:>6} {total_tokens:>10,}"
    )
    if total_cost > 0:
        lines.append(f"  real spend: ${total_cost:.4f}")

    return "\n".join(lines)


def render_today(
    usage: dict[str, tuple[int, int]], manifest: Manifest
) -> str:
    """What today has already cost against each provider's daily wall.

    Each provider is measured against its own calendar day — Groq's rolls at
    UTC midnight, Gemini's at Pacific — so the two rows are not comparable as
    clock time, and are not meant to be.
    """
    lines = ["", "Today (each provider in its own reset timezone)", "=" * 78]

    if not usage:
        lines.append("  (nothing recorded today)")
        return "\n".join(lines)

    lines.append(
        f"  {'model':<24} {'requests':>13}  {'tokens':>10}  used"
    )
    lines.append("  " + "-" * 74)

    for model_id in sorted(usage):
        requests, tokens = usage[model_id]
        try:
            provider = manifest.provider_of(model_id)
        except Exception:
            # A model that has since been removed from the manifest still has
            # history worth showing; it just has no limit to show it against.
            lines.append(f"  {model_id:<24} {requests:>13}  {tokens:>10,}  —")
            continue

        rpd = provider.limit(LimitKind.RPD)
        cap = f"{requests}/{rpd.value}" if rpd else str(requests)
        lines.append(
            f"  {model_id:<24} {cap:>13}  {tokens:>10,}  "
            f"{_percent_bar(requests, rpd.value if rpd else None)}"
        )

    lines.append("")
    lines.append("  Requests are the scarce resource here, not tokens.")
    return "\n".join(lines)


def render_totals(totals: dict[str, DayUsage]) -> str:
    """Lifetime usage per model — the raw material for the M4 track record."""
    lines = ["", "Lifetime totals", "=" * 78]

    if not totals:
        lines.append("  (nothing recorded)")
        return "\n".join(lines)

    lines.append(
        f"  {'model':<24} {'calls':>7} {'prompt':>11} {'output':>11} {'cost':>9}"
    )
    lines.append("  " + "-" * 74)

    for model_id in sorted(totals, key=lambda m: -totals[m].requests):
        usage = totals[model_id]
        cost = f"${usage.cost_usd:.4f}" if usage.cost_usd > 0 else "free"
        lines.append(
            f"  {model_id:<24} {usage.requests:>7} {usage.prompt_tokens:>11,} "
            f"{usage.completion_tokens:>11,} {cost:>9}"
        )
    return "\n".join(lines)


def render_run_detail(events: tuple, *, run_id: str) -> str:
    """Every request one run made, in order.

    This is the view that makes a wasteful run legible: retries, failovers and
    repairs all appear as separate rows against the same node, which is exactly
    what the efficiency percentage in the spend report is measuring.
    """
    lines = ["", f"Run {run_id}", "=" * 78]

    if not events:
        lines.append("  (no events recorded for this run)")
        return "\n".join(lines)

    lines.append(
        f"  {'#':>3} {'node':<10} {'purpose':<10} {'model':<22} "
        f"{'tokens':>8} {'ms':>6}  status"
    )
    lines.append("  " + "-" * 74)

    for index, event in enumerate(events, start=1):
        status = "ok" if event.ok else f"FAIL {event.http_status}"
        lines.append(
            f"  {index:>3} {(event.node_id or '—'):<10} {event.purpose:<10} "
            f"{event.model_id:<22} {event.usage.total_tokens:>8,} "
            f"{event.latency_ms:>6}  {status}"
        )
        if event.error:
            lines.append(f"      {event.error[:70]}")

    failures = sum(1 for e in events if not e.ok)
    lines.append("")
    lines.append(f"  {len(events)} requests, {failures} failed")
    return "\n".join(lines)


def render_estimator_drift(drift: dict[str, tuple[float, int]]) -> str:
    """How far each provider's learned token ratio sits from the naive guess.

    A ratio far from 1.0 is not a fault — it is the estimator having learned
    that this provider's tokenizer disagrees with a character count, which is
    the entire reason it self-calibrates. Worth showing because a ratio still
    at exactly 1.0 after many calls means calibration is not being fed.
    """
    lines = ["", "Estimator calibration", "=" * 78]

    if not drift:
        lines.append("  (no samples yet — the character heuristic stands)")
        return "\n".join(lines)

    lines.append(f"  {'provider':<16} {'ratio':>8} {'samples':>9}  status")
    lines.append("  " + "-" * 52)
    for provider in sorted(drift):
        ratio, samples = drift[provider]
        status = "warmed up" if samples >= 5 else "warming up"
        lines.append(f"  {provider:<16} {ratio:>8.3f} {samples:>9}  {status}")
    return "\n".join(lines)
