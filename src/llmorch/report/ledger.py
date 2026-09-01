"""Rendering for the persistent ledger.

`report/render.py` shows what *this* run did. This module shows what the
account has done — across runs, across processes, across the day boundary that
each provider draws in its own timezone. That distinction is the whole point of
the ledger: the number that decides whether a request can be sent is the
account's, not this process's.
"""

from __future__ import annotations

from decimal import Decimal

from ..quota.store import DayUsage, LedgerRow, ModelDayUsage


def _money(amount: Decimal) -> str:
    return "—" if amount == 0 else f"${amount:.4f}"


def _short_ts(ts_utc: str) -> str:
    """`2026-08-24T18:05:12.482+00:00` -> `08-24 18:05:12`."""
    date, _, rest = ts_utc.partition("T")
    return f"{date[5:]} {rest[:8]}" if rest else ts_utc[:16]


def render_day_usage(
    rows: list[ModelDayUsage], *, title: str = "Ledger — usage by day"
) -> str:
    """Per-(model, day) totals, as recorded on disk."""
    lines = ["", title, "=" * 78]
    if not rows:
        lines.append("  (no calls recorded yet — every run so far has been a dry run)")
        return "\n".join(lines)

    lines.append(
        f"  {'day':<12} {'model':<24} {'calls':>6} {'failed':>7} {'tokens':>10}  cost"
    )
    lines.append("  " + "-" * 74)

    current_day = None
    for row in rows:
        day = "" if row.day_key == current_day else row.day_key
        current_day = row.day_key
        lines.append(
            f"  {day:<12} {row.model_id:<24} {row.requests:>6} {row.failures:>7} "
            f"{row.tokens:>10,}  {_money(row.cost_usd)}"
        )

    lines.append("  " + "-" * 74)
    lines.append(
        f"  {'total':<12} {'':<24} {sum(r.requests for r in rows):>6} "
        f"{sum(r.failures for r in rows):>7} {sum(r.tokens for r in rows):>10,}  "
        f"{_money(sum((r.cost_usd for r in rows), Decimal('0')))}"
    )
    # Days are labelled in each provider's own reset timezone, so two rows with
    # the same date can end at different moments.
    lines.append("  Days are counted in each provider's own reset timezone.")
    return "\n".join(lines)


def render_recent(rows: list[LedgerRow], *, limit: int | None = None) -> str:
    """The last N calls, newest first — the flight recorder view."""
    shown = rows[:limit] if limit else rows
    lines = ["", "Ledger — recent calls", "=" * 78]
    if not shown:
        lines.append("  (nothing recorded)")
        return "\n".join(lines)

    lines.append(
        f"  {'when':<16} {'model':<22} {'purpose':<10} {'node':<8} "
        f"{'tokens':>8} {'ms':>6}  status"
    )
    lines.append("  " + "-" * 74)
    for row in shown:
        status = "ok" if row.ok else f"FAIL {row.http_status}"
        lines.append(
            f"  {_short_ts(row.ts_utc):<16} {row.model_id:<22} {row.purpose:<10} "
            f"{(row.node_id or '—'):<8} "
            f"{row.prompt_tokens + row.completion_tokens:>8,} {row.latency_ms:>6}  "
            f"{status}"
        )
        if row.error:
            lines.append(f"      {row.error[:70]}")
    return "\n".join(lines)


def render_restored(restored: dict[str, DayUsage]) -> str:
    """What the governor inherited from earlier runs at startup.

    Worth printing every time. A run that quietly begins with 200 of Gemini's
    250 daily requests already spent behaves very differently from one that
    starts fresh, and the difference should never be a surprise.
    """
    if not restored:
        return ""
    lines = ["", "Carried over from earlier runs today", "=" * 78]
    for model_id in sorted(restored):
        usage = restored[model_id]
        lines.append(
            f"  {model_id:<24} {usage.requests:>4} requests, "
            f"{usage.tokens:>8,} tokens already spent"
        )
    return "\n".join(lines)
