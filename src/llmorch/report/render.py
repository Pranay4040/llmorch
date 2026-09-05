"""Text rendering for plans, spend reports, quota headroom, and diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..engine.graph import TaskGraph
from ..engine.scheduler import RunOutcome
from ..negotiate.reconcile import ReconcileResult
from ..types import Headroom, NodeState

if TYPE_CHECKING:  # pragma: no cover - doctor imports providers; keep it lazy
    from ..doctor import Check
    from ..engine.checkpoint import Checkpoint
    from ..discover import Discovery
    from ..engine.contracts import ContractReport
    from ..engine.smoke import SmokeReport


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


_MARKS = {"ok": "ok  ", "warn": "warn", "fail": "FAIL", "skip": "--  "}


def render_doctor(checks: "list[Check]") -> str:
    """Diagnostic sweep, worst news last.

    `warn` and `fail` are kept distinct because they mean different things here:
    a missing key is a warning (that provider is simply dormant), whereas an
    unresolvable timezone or an unwritable ledger is a failure — the run would
    proceed while quietly miscounting quota, which is worse than not running.
    """
    lines = ["", "Doctor", "=" * 78]
    for check in checks:
        mark = _MARKS.get(check.status, check.status)
        lines.append(f"  [{mark}] {check.name:<30} {check.detail}")

    failures = sum(1 for c in checks if c.status == "fail")
    warnings = sum(1 for c in checks if c.status == "warn")
    lines.append("  " + "-" * 74)
    lines.append(
        f"  {len(checks)} checks · {failures} failed · {warnings} warned"
        if failures or warnings
        else f"  {len(checks)} checks, all clear"
    )
    return "\n".join(lines)


def render_resume_list(checkpoints: "list[Checkpoint]") -> str:
    """Runs that can be picked back up, and what each is waiting on.

    The blocked column is the one that matters: a run waiting on a daily reset
    is not stuck, it is scheduled, and the time shown is when it stops being
    cheaper to wait than to spend another vendor's quota on the same work.
    """
    lines = ["", "Resumable runs", "=" * 78]
    if not checkpoints:
        lines.append("  (no checkpoints yet)")
        return "\n".join(lines)

    lines.append(f"  {'run':<24} {'done':>6} {'left':>6}  {'blocked until':<26} task")
    lines.append("  " + "-" * 74)

    for book in checkpoints:
        left = len(book.unfinished)
        moment = book.resumable_at()
        when = "—" if moment is None else f"{moment:%Y-%m-%d %H:%M} UTC"
        if book.is_complete:
            when = "complete"
        lines.append(
            f"  {book.run_id:<24} {len(book.completed):>6} {left:>6}  "
            f"{when:<26} {book.task[:24]}"
        )
    return "\n".join(lines)


def render_contracts(report: "ContractReport") -> str:
    """Whether the pieces fit each other, as opposed to fitting their own specs.

    Reported after the files are written, never as a gate on writing them: the
    artifacts are already paid for, and a half-matching project someone can fix
    in two minutes beats an empty folder.
    """
    lines = ["", "Contract", "=" * 78]
    if not report.issues:
        lines.append(
            f"  {len(report.checks_run)} cross-artifact checks passed: "
            "pages exist, assets resolve, calls match declared routes"
        )
        return "\n".join(lines)

    for issue in report.errors + report.warnings:
        mark = "FAIL" if issue.severity == "error" else "warn"
        lines.append(f"  [{mark}] {issue.what}")
        if issue.why:
            lines.append(f"         {issue.why}")

    lines.append("  " + "-" * 74)
    lines.append(
        f"  {len(report.errors)} mismatch(es), {len(report.warnings)} warning(s) "
        "across artifacts written by different models"
    )
    return "\n".join(lines)


def _stderr_block(text: str, limit: int = 12) -> list[str]:
    """The captured stderr, keeping both ends when it is too long for one.

    A tail alone is the wrong half for the most common case: Node puts
    `Cannot find module` on the first line and thirty frames of stack after it,
    so twelve lines from the bottom show the stack and lose the cause.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= limit:
        body = lines
    else:
        head = (limit + 1) // 2
        elision = f"... {len(lines) - limit} more lines ..."
        body = lines[:head] + [elision] + lines[head - limit :]
    return [f"    {line}" for line in body]


def render_smoke(report: "SmokeReport") -> str:
    """What happened when the generated project was actually run.

    A skip is printed as a skip, never as a pass. The whole value of this step is
    that it is the only evidence in the run that came from executing the code,
    so "we did not get to find out" must not read like "it works".
    """
    lines = ["", "Smoke run", "=" * 78]

    if not report.ran:
        # A skip and a crash both leave `ran` False and mean opposite things:
        # one is no evidence, the other is the worst evidence there is.
        if report.skipped:
            lines.append(f"  skipped — {report.skipped}")
        else:
            lines.append(
                f"  {report.entrypoint or 'the project'} never reached a serving state"
            )
        for issue in report.errors + report.warnings:
            mark = "FAIL" if issue.severity == "error" else "warn"
            lines.append(f"  [{mark}] {issue.what}")
            if issue.why:
                lines.append(f"         {issue.why}")
        if report.stderr_tail:
            lines.append("  " + "-" * 74)
            lines.append("  stderr:")
            lines += _stderr_block(report.stderr_tail)
        return "\n".join(lines)

    if report.installed:
        lines.append(f"  {report.installed}")
    lines.append(f"  {report.entrypoint} on http://127.0.0.1:{report.port}")
    for probe in report.probes:
        status = str(probe.status) if probe.status is not None else "---"
        note = f"  {probe.detail}" if probe.status is None and probe.detail else ""
        lines.append(f"    {status:>4}  {probe.label}{note}")

    if not report.issues:
        lines.append("  " + "-" * 74)
        lines.append(
            f"  {len(report.probes)} request(s), no server-side failures — "
            "the assembled project runs"
        )
        return "\n".join(lines)

    lines.append("  " + "-" * 74)
    for issue in report.errors + report.warnings:
        mark = "FAIL" if issue.severity == "error" else "warn"
        lines.append(f"  [{mark}] {issue.what}")
        if issue.why:
            lines.append(f"         {issue.why}")

    if report.stderr_tail:
        lines.append("  " + "-" * 74)
        lines.append("  stderr:")
        lines += _stderr_block(report.stderr_tail)

    lines.append("  " + "-" * 74)
    lines.append(
        f"  {len(report.errors)} failure(s), {len(report.warnings)} warning(s) "
        f"across {len(report.probes)} request(s)"
    )
    return "\n".join(lines)


def render_discovery(found: "list[Discovery]") -> str:
    """What each spare key turns out to be worth.

    The distinction the table exists to draw: a provider that lists models is
    not a provider that will run one. Both of the rejections below answered
    `GET /models` perfectly well.
    """
    marks = {"ok": "ok  ", "no key": "--  ", "auth": "AUTH",
             "unreachable": "DOWN", "unexpected": "????"}
    lines = ["", "Discovery", "=" * 78]
    lines.append(f"  {'provider':<16} {'':<4} {'models':>7}  detail")
    lines.append("  " + "-" * 74)

    for entry in found:
        lines.append(
            f"  {entry.provider:<16} [{marks.get(entry.status, entry.status)}] "
            f"{len(entry.models):>7}  {entry.detail[:44]}"
        )
        if entry.base_url and entry.status != "no key":
            lines.append(f"                          {entry.base_url}")

    usable = [e for e in found if e.usable]
    lines.append("  " + "-" * 74)
    lines.append(
        f"  {len(usable)} of {len(found)} keys reached a model list. "
        "Listing is not entitlement — probe before enabling."
    )
    return "\n".join(lines)
