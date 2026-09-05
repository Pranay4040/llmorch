"""The run, written down.

Everything this system learns about a run has so far existed only in terminal
scrollback: which model wrote which file, what the quota bought, whether the
artifacts agree with each other, whether the assembled project actually ran.
Close the window and the evidence is gone, while the artifacts it describes stay
on disk looking equally plausible either way.

So `report.md` sits beside the output folder it is about. It is the same
information the terminal shows, ordered for someone reading it later rather than
watching it happen: the verdict first, then the parts that support it.

Two constraints carried from the original plan. **No key is ever written here** —
this file is as publishable as the folder next to it, and the rule that keeps
secrets out of the ledger and out of errors applies to it too. And **nothing is
recomputed**: every number comes from the same objects the terminal renderers
read, so the file and the screen can never disagree about what happened.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..config import RunConfig
from ..engine.graph import TaskGraph
from ..engine.scheduler import RunOutcome
from ..negotiate.reconcile import ReconcileResult
from ..types import NodeState

if TYPE_CHECKING:  # pragma: no cover
    from ..engine.contracts import ContractReport
    from ..engine.materialize import MaterializeReport
    from ..engine.smoke import SmokeReport

REPORT_NAME = "report.md"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(cell) + " |" for cell in rows),
    ]


def _verdict(
    outcome: RunOutcome,
    contract: "ContractReport",
    smoke: "SmokeReport | None",
) -> list[str]:
    """The answer first, in the order that decides whether to trust the output.

    A run whose nodes all succeeded can still have produced a project that does
    not run, and that is exactly the case worth putting at the top: every other
    section reports on a step, and only this one reports on the result.
    """
    lines: list[str] = []

    if outcome.degraded:
        lines.append(
            f"- **{len(outcome.degraded)} of {len(outcome.results)} nodes "
            f"degraded** — {', '.join(outcome.degraded)}"
        )
    elif outcome.results:
        lines.append(f"- All {len(outcome.results)} nodes produced their artifact")

    if contract.errors:
        lines.append(
            f"- **{len(contract.errors)} cross-artifact mismatch(es)** — the files "
            "do not agree with each other"
        )
    elif contract.checks_run:
        lines.append(f"- {len(contract.checks_run)} cross-artifact checks passed")

    if smoke is None:
        lines.append("- Not started — pass `--smoke` to run what was written")
    elif not smoke.ran:
        lines.append(
            f"- **Not started** — {smoke.skipped or 'it never began serving'}"
        )
    elif smoke.errors:
        lines.append(
            f"- **The assembled project does not run** — {len(smoke.errors)} "
            f"failure(s) across {len(smoke.probes)} request(s)"
        )
    else:
        lines.append(
            f"- The assembled project runs — {len(smoke.probes)} request(s), "
            "no server-side failures"
        )

    return lines


def _nodes(graph: TaskGraph, outcome: RunOutcome) -> list[str]:
    rows = []
    for node_id in sorted(outcome.results):
        result = outcome.results[node_id]
        node = graph.nodes.get(node_id)
        rows.append(
            [
                f"`{node_id}`",
                node.role.value if node else "—",
                result.model_id or "—",
                "done" if result.state is NodeState.DONE else "**degraded**",
                str(result.attempts),
                ", ".join(result.vendors_tried) or "—",
            ]
        )
    return _table(["node", "role", "model", "state", "tries", "vendors"], rows)


def _spend(outcome: RunOutcome) -> list[str]:
    by_model: dict[str, tuple[int, int, int]] = {}
    for result in outcome.results.values():
        if not result.model_id:
            continue
        prompt, completion, calls = by_model.get(result.model_id, (0, 0, 0))
        by_model[result.model_id] = (
            prompt + result.usage.prompt_tokens,
            completion + result.usage.completion_tokens,
            calls + 1,
        )
    if not by_model:
        return ["Nothing was spent."]

    rows = []
    total_prompt = total_completion = 0
    for model_id in sorted(by_model):
        prompt, completion, calls = by_model[model_id]
        total_prompt += prompt
        total_completion += completion
        rows.append([model_id, str(calls), f"{prompt:,}", f"{completion:,}"])
    rows.append(["**total**", "", f"**{total_prompt:,}**", f"**{total_completion:,}**"])

    lines = _table(["model", "calls", "prompt", "output"], rows)

    useful = sum(
        r.usage.completion_tokens
        for r in outcome.results.values()
        if r.state is NodeState.DONE
    )
    spent = total_prompt + total_completion
    if spent:
        lines += [
            "",
            f"**Quota efficiency: {100 * useful / spent:.1f}%** "
            f"({useful:,} useful of {spent:,} spent). "
            "Retries, repairs and degraded nodes count as waste.",
        ]
    return lines


def _fair_share(
    plan: ReconcileResult, graph: TaskGraph, outcome: RunOutcome, config: RunConfig
) -> list[str]:
    """Planned share against realized, because they come apart.

    "Split evenly" is enforced when the plan is made, over estimated tokens. What
    a model actually wrote is a different number, and failover moves work after
    the fact — so a plan that was even can end lopsided, and the only way to see
    that is to put both columns next to each other.
    """
    planned = plan.token_share(graph) if plan.assignments else {}
    realized: dict[str, int] = {}
    for result in outcome.results.values():
        if result.model_id:
            realized[result.model_id] = (
                realized.get(result.model_id, 0) + result.usage.completion_tokens
            )
    if not planned and not realized:
        return []

    models = sorted(set(planned) | set(realized))
    total_planned = sum(planned.values())
    total_realized = sum(realized.values())
    even = 100 / len(models) if models else 0
    ceiling = 100 * (1 + config.imbalance_tolerance)

    rows = []
    for model_id in models:
        share_planned = (
            f"{100 * planned.get(model_id, 0) / total_planned:.0f}%"
            if total_planned
            else "—"
        )
        share_realized = (
            f"{100 * realized.get(model_id, 0) / total_realized:.0f}%"
            if total_realized
            else "—"
        )
        rows.append(
            [
                model_id,
                f"{planned.get(model_id, 0):,}",
                share_planned,
                f"{realized.get(model_id, 0):,}",
                share_realized,
            ]
        )

    return _table(
        ["model", "planned tokens", "planned share", "written tokens", "written share"],
        rows,
    ) + [
        "",
        f"An even split across {len(models)} model(s) is {even:.0f}% each; the "
        f"assignment allows a model up to {ceiling:.0f}% of an even share "
        "before it stops taking work.",
    ]


def _findings(issues: list, kind: str) -> list[str]:
    lines = []
    for issue in issues:
        mark = "**FAIL**" if issue.severity == "error" else "warn"
        where = f" (`{issue.where}`)" if getattr(issue, "where", "") else ""
        lines.append(f"- {mark} {issue.what}{where}")
        if issue.why:
            lines.append(f"  - {issue.why}")
    return lines or [f"No {kind}."]


def _smoke(smoke: "SmokeReport | None") -> list[str]:
    if smoke is None:
        return [
            "Not attempted. `llmorch run --smoke` starts the generated project "
            "and drives this contract against it."
        ]
    if not smoke.ran:
        lines = [f"Not started — {smoke.skipped or 'it never began serving'}."]
        lines += _findings(smoke.errors + smoke.warnings, "findings")
        if smoke.stderr_tail:
            lines += ["", "```", smoke.stderr_tail, "```"]
        return lines

    lines = []
    if smoke.installed:
        lines.append(f"Dependencies installed with `{smoke.installed}`.")
    lines.append(f"Started `{smoke.entrypoint}` on port {smoke.port}.")
    lines.append("")
    lines += _table(
        ["status", "request"],
        [
            [str(p.status) if p.status is not None else "no answer", f"`{p.label}`"]
            for p in smoke.probes
        ],
    )
    if smoke.issues:
        lines += ["", *_findings(smoke.errors + smoke.warnings, "findings")]
    if smoke.stderr_tail:
        lines += ["", "```", smoke.stderr_tail, "```"]
    return lines


def render_run_report(
    *,
    config: RunConfig,
    graph: TaskGraph,
    plan: ReconcileResult,
    outcome: RunOutcome,
    materialized: "MaterializeReport",
    contract: "ContractReport",
    smoke: "SmokeReport | None" = None,
    generated_utc: str | None = None,
) -> str:
    """The whole run as one Markdown document."""
    stamp = generated_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")
    mode = "mock provider, no network" if config.dry_run else "live"

    lines = [
        f"# Run {config.run_id}",
        "",
        f"**Task:** {config.task}",
        "",
        f"- Mode: {mode}",
        f"- Finished: {stamp}",
        f"- Output: `{config.output_dir}`",
        "",
        "## Verdict",
        "",
        *_verdict(outcome, contract, smoke),
        "",
        "## Nodes",
        "",
        *_nodes(graph, outcome),
    ]

    for note in outcome.reassignments:
        lines.append(f"- reassigned: {note}")

    lines += [
        "",
        "## Spend",
        "",
        *_spend(outcome),
        "",
        "## Fair share",
        "",
        *_fair_share(plan, graph, outcome, config),
        "",
        "## Files written",
        "",
        f"{len(materialized.written)} written, {len(materialized.stubbed)} stubbed.",
    ]
    for path, reason in materialized.rejected:
        lines.append(f"- **rejected** `{path}`: {reason}")

    lines += [
        "",
        "## Cross-artifact checks",
        "",
    ]
    if contract.issues:
        lines += _findings(contract.errors + contract.warnings, "mismatches")
    else:
        lines.append(
            f"{len(contract.checks_run)} checks passed: "
            + ", ".join(contract.checks_run)
        )

    lines += ["", "## Smoke run", "", *_smoke(smoke)]

    if outcome.warnings:
        lines += ["", "## Warnings", "", *(f"- {w}" for w in outcome.warnings)]

    if outcome.degraded:
        lines += [
            "",
            "## Resuming",
            "",
            f"The finished work is checkpointed. `llmorch resume {config.run_id}` "
            "picks up only what is missing.",
        ]

    return "\n".join(lines).rstrip() + "\n"
