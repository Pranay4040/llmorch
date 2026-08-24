"""Command line entry point.

    llmorch run --dry-run "build a notes app"
    llmorch plan --explain "build a notes app"
    llmorch quota
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from .config import RunConfig, load_dotenv, runs_dir
from .demo.website import ARTIFACTS, INTERFACE, SUMMARIES, TASK, build_nodes
from .engine.blackboard import Blackboard
from .engine.graph import TaskGraph
from .engine.materialize import materialize
from .engine.scheduler import Scheduler
from .errors import LLMOrchError
from .providers.base import ProviderRegistry
from .providers.mock import MockProvider
from .quota.governor import Governor
from .registry.manifest import load_manifest
from .report.render import (
    render_outcome,
    render_plan,
    render_quota,
    render_spend,
    render_warnings,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _mock_registry(manifest) -> tuple[ProviderRegistry, MockProvider]:
    """One mock instance shared by every model id, so a single call log covers
    the whole run and fault injection can target any model."""
    provider = MockProvider(responses=dict(ARTIFACTS))
    registry = ProviderRegistry()
    for model in manifest.enabled_models:
        registry.register(model.id, provider)
    return registry, provider


def _setup(args) -> tuple:
    manifest = load_manifest()
    config = RunConfig(
        task=args.task or TASK,
        run_id=_run_id(),
        dry_run=not getattr(args, "live", False),
        allow_paid=getattr(args, "allow_paid", False),
        max_usd=Decimal(str(getattr(args, "max_usd", 0) or 0)),
        review=getattr(args, "review", "code"),
        max_nodes=getattr(args, "max_nodes", 10),
        max_concurrency=getattr(args, "concurrency", 4),
    )

    graph = TaskGraph.build(build_nodes())
    graph.prune_to_budget(config.max_nodes)

    governor = Governor(
        manifest,
        max_usd=config.max_usd,
        allow_paid=config.allow_paid,
        safety_factor=config.token_safety_factor,
    )

    blackboard = Blackboard(interface=INTERFACE)
    registry, mock = _mock_registry(manifest)

    scheduler = Scheduler(
        graph,
        manifest,
        governor,
        registry,
        config=config,
        blackboard=blackboard,
    )
    return manifest, config, graph, governor, scheduler, mock


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_plan(args) -> int:
    _, _, graph, _, scheduler, _ = _setup(args)
    plan = scheduler.plan()
    print(render_plan(plan, graph, explain=args.explain))
    print(render_warnings(graph.warnings))
    return 0


def cmd_quota(args) -> int:
    manifest = load_manifest()
    print(render_quota(Governor(manifest).headroom()))
    print("\n  Milestone 1 runs entirely on the mock provider — no keys needed.")
    return 0


def cmd_run(args) -> int:
    manifest, config, graph, governor, scheduler, mock = _setup(args)

    if not config.dry_run:
        print("Live mode is not wired yet — real providers arrive at Milestone 2.")
        return 2

    print(f"Task: {config.task}")
    print(f"Run:  {config.run_id}  (dry run — mock provider, no network)")

    plan = scheduler.plan()
    print(render_plan(plan, graph, explain=args.explain))

    outcome = asyncio.run(scheduler.run(plan))
    print(render_outcome(outcome, graph))
    print(render_spend(outcome))

    # Seed summaries the mock cannot produce itself.
    for node_id, result in outcome.results.items():
        if node_id in SUMMARIES and not result.summary:
            result.summary = SUMMARIES[node_id]

    report = materialize(config.output_dir, graph.nodes, outcome.results)

    print("\nOutput")
    print("=" * 78)
    print(f"  {config.output_dir}")
    print(f"  {len(report.written)} written, {len(report.stubbed)} stubbed")
    for path, reason in report.rejected:
        print(f"  ! rejected {path}: {reason}")

    print(render_warnings(outcome.warnings))

    print("\nRun the result:")
    print(f"  python {config.output_dir / 'server.py'}")
    print("  then open http://localhost:8000")

    return 0 if outcome.all_succeeded else 1


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmorch",
        description="Split a build across models from different vendors.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="plan, execute, and write the output folder")
    run.add_argument("task", nargs="?", default=None)
    run.add_argument("--dry-run", action="store_true", default=True)
    run.add_argument("--live", action="store_true", help="use real providers [M2]")
    run.add_argument("--explain", action="store_true", help="show the scoring maths")
    run.add_argument("--review", choices=["off", "code", "all"], default="code")
    run.add_argument("--allow-paid", action="store_true")
    run.add_argument("--max-usd", type=float, default=0.0)
    run.add_argument("--max-nodes", type=int, default=10)
    run.add_argument("--concurrency", type=int, default=4)
    run.set_defaults(func=cmd_run)

    plan = sub.add_parser("plan", help="show the assignment without executing")
    plan.add_argument("task", nargs="?", default=None)
    plan.add_argument("--explain", action="store_true")
    plan.add_argument("--max-nodes", type=int, default=10)
    plan.set_defaults(func=cmd_plan)

    quota = sub.add_parser("quota", help="show per-provider headroom")
    quota.set_defaults(func=cmd_quota, task=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except LLMOrchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
