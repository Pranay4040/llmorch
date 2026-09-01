"""Command line entry point.

    llmorch run "build a notes app"          # mock provider, no network
    llmorch run --live "build a notes app"   # real requests, Groq only [M2]
    llmorch resume [<run_id>]                # continue after a quota wall
    llmorch plan --explain "build a notes app"
    llmorch quota
    llmorch ledger --days 3
    llmorch doctor [--probe]
    llmorch dashboard                        # read-only view on localhost
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from .config import RunConfig, load_dotenv, state_db_path
from .dashboard.server import DEFAULT_PORT, serve
from .demo.website import ARTIFACTS, INTERFACE, SUMMARIES, TASK, build_nodes
from .discover import discover_all
from .doctor import run_doctor
from .engine.blackboard import Blackboard
from .engine.health import HealthTracker
from .engine.checkpoint import (
    Checkpoint,
    check_applies,
    plan_from_dict,
    latest_resumable,
    list_checkpoints,
    load_run,
)
from .engine.contracts import artifacts_from_results, check_contract
from .engine.graph import TaskGraph
from .engine.materialize import materialize
from .engine.scheduler import Scheduler
from .engine.worker import WorkerDeps
from .errors import LLMOrchError
from .providers.base import ProviderRegistry
from .providers.mock import MockProvider
from .providers.openai_compat import build_live_registry
from .quota.estimator import TokenEstimator
from .negotiate import plancache
from .negotiate.bidding import collect_bids, should_bid
from .negotiate.decompose import DecomposeError, decompose, pick_planner, plan_signature
from .negotiate.profiles import Profiles
from .quota.governor import Governor
from .quota.store import DayUsage, LedgerStore, restore_governor
from .registry.manifest import Manifest, load_manifest
from .types import InterfaceContract
from .report.ledger import render_day_usage, render_recent, render_restored
from .report.render import (
    render_contracts,
    render_discovery,
    render_doctor,
    render_resume_list,
    render_outcome,
    render_plan,
    render_quota,
    render_spend,
    render_warnings,
)

# Milestone 2 deliberately runs one vendor. Groq allows 14,400 requests a day,
# which makes it the only sane place to debug header parsing and counter sync;
# Gemini's 250 would be exhausted by a bad afternoon.
DEFAULT_LIVE_PROVIDERS = "groq"


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _providers_arg(value: str | None) -> set[str] | None:
    if not value or value.strip().lower() == "all":
        return None
    return {p.strip() for p in value.split(",") if p.strip()}


@dataclass(slots=True)
class Session:
    """Everything a command needs, assembled once."""

    manifest: Manifest
    config: RunConfig
    graph: TaskGraph
    governor: Governor
    scheduler: Scheduler
    store: LedgerStore | None
    estimator: TokenEstimator
    restored: dict[str, DayUsage]
    mock: MockProvider | None
    profiles: Profiles

    def close(self) -> None:
        # The track record is the only thing here that took live requests to
        # learn, so it is saved whether or not the run went well.
        if not self.config.dry_run:
            self.profiles.save()
        if self.store is not None:
            # The estimator's learned ratios take ~20 live calls to converge.
            # Each one costs a request, so they are far too expensive to throw
            # away at process exit.
            self.store.save_calibration(self.estimator.to_dict())
            self.store.close()


def _mock_registry(manifest) -> tuple[ProviderRegistry, MockProvider]:
    """One mock instance shared by every model id, so a single call log covers
    the whole run and fault injection can target any model."""
    provider = MockProvider(responses=dict(ARTIFACTS))
    registry = ProviderRegistry()
    for model in manifest.enabled_models:
        registry.register(model.id, provider)
    return registry, provider


def _worker_deps(
    *, manifest, governor, registry, estimator, profiles, store, config, interface
):
    """The same dependency bundle the executor uses.

    Negotiation goes through it deliberately: planning and bidding are requests
    like any other, and routing them around admission control would let the one
    request the run depends on be the one that blows the daily cap.
    """
    return WorkerDeps(
        manifest=manifest,
        governor=governor,
        registry=registry,
        estimator=estimator,
        health=HealthTracker(threshold=config.circuit_breaker_threshold),
        blackboard=Blackboard(interface=interface),
        ledger=store,
        run_id=config.run_id,
    )


def _plan(
    args, *, config, manifest, governor, registry, estimator, profiles, store,
    stored_plan: dict | None = None,
):
    """Decide the task graph, spending as little as possible to get it.

    Three sources, cheapest first:

    1. **The demo graph** — for the reference task, hand-written and free. It is
       the fixture the whole test suite is built on; re-planning it every run
       would spend the scarcest request in the system to rediscover a known
       answer.
    2. **The plan cache** — same task, same roster, zero requests.
    3. **A live decomposition** — one HIGH-priority request to whichever model
       has the best planning affinity.

    Returns (nodes, interface, note).
    """
    task = config.task.strip()
    force = getattr(args, "decompose", False)

    if stored_plan:
        # A resume runs the graph it was interrupted in the middle of. Planning
        # again would spend the scarcest request in the system to rediscover a
        # known answer — and could return a different graph, since the plan
        # signature includes the roster.
        nodes, interface = plan_from_dict(stored_plan)
        if nodes:
            return nodes, interface, "graph restored from the checkpoint"

    if not force and task.lower() == TASK.strip().lower():
        return build_nodes(), INTERFACE, ""

    signature = plan_signature(task, manifest)
    if not getattr(args, "no_cache", False):
        cached = plancache.load(signature)
        if cached is not None:
            return (
                cached.nodes,
                cached.interface,
                f"plan reused from cache ({signature}) — no planning request spent",
            )

    deps = _worker_deps(
        manifest=manifest, governor=governor, registry=registry,
        estimator=estimator, profiles=profiles, store=store, config=config,
        interface=InterfaceContract(),
    )
    planner = pick_planner(manifest, sorted(registry.model_ids))
    if planner is None:
        raise DecomposeError("no model available to plan this task")

    plan = asyncio.run(
        decompose(task, deps=deps, model_id=planner, max_nodes=config.max_nodes)
    )
    plancache.save(signature, plan, task=task)
    return (
        plan.nodes,
        plan.interface,
        f"planned by {planner} into {len(plan.nodes)} node(s), cached as {signature}",
    )


def _setup(
    args, *, run_id: str | None = None, stored_plan: dict | None = None
) -> Session:
    manifest = load_manifest()
    config = RunConfig(
        task=args.task or TASK,
        # A resume writes back into the run it is continuing, so the output
        # folder and the checkpoint stay in one place.
        run_id=run_id or _run_id(),
        dry_run=not getattr(args, "live", False),
        allow_paid=getattr(args, "allow_paid", False),
        max_usd=Decimal(str(getattr(args, "max_usd", 0) or 0)),
        review=getattr(args, "review", "code"),
        max_nodes=getattr(args, "max_nodes", 10),
        max_concurrency=getattr(args, "concurrency", 4),
    )

    governor = Governor(
        manifest,
        max_usd=config.max_usd,
        allow_paid=config.allow_paid,
        safety_factor=config.token_safety_factor,
    )

    # A dry run touches neither the ledger nor the network. Recording mock
    # calls would tell tomorrow's admission control that quota was spent which
    # never was — the ledger is only ever a record of real requests.
    store: LedgerStore | None = None
    profiles = Profiles.load()
    estimator = TokenEstimator()
    restored: dict[str, DayUsage] = {}
    mock: MockProvider | None = None

    if config.dry_run:
        registry, mock = _mock_registry(manifest)
    else:
        store = LedgerStore(state_db_path()).open()
        estimator = TokenEstimator.from_dict(store.load_calibration())
        restored = restore_governor(governor, store, manifest)
        registry, _ = build_live_registry(
            manifest,
            only_providers=_providers_arg(getattr(args, "providers", None)),
        )

    nodes, interface, plan_note = _plan(
        args,
        config=config,
        manifest=manifest,
        governor=governor,
        registry=registry,
        estimator=estimator,
        profiles=profiles,
        store=store,
        stored_plan=stored_plan,
    )
    graph = TaskGraph.build(nodes)
    graph.prune_to_budget(config.max_nodes)
    if plan_note:
        graph.warnings.append(plan_note)

    scheduler = Scheduler(
        graph,
        manifest,
        governor,
        registry,
        config=config,
        blackboard=Blackboard(interface=interface),
        estimator=estimator,
        ledger=store,
        profiles=profiles,
        checkpoints=True,
    )
    return Session(
        manifest=manifest,
        config=config,
        graph=graph,
        governor=governor,
        scheduler=scheduler,
        store=store,
        estimator=estimator,
        restored=restored,
        mock=mock,
        profiles=profiles,
    )


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_plan(args) -> int:
    session = _setup(args)
    try:
        plan = session.scheduler.plan()
        print(render_plan(plan, session.graph, explain=args.explain))
        print(render_warnings(session.graph.warnings))
    finally:
        session.close()
    return 0


def cmd_quota(args) -> int:
    """Headroom as it actually stands, ledger included."""
    manifest = load_manifest()
    governor = Governor(manifest)
    with LedgerStore(state_db_path()) as store:
        restored = restore_governor(governor, store, manifest)
        print(render_quota(governor.headroom()))
        print(render_restored(restored))
        print(render_day_usage(store.day_table(days=1), title="Ledger — today"))
    return 0


def cmd_ledger(args) -> int:
    with LedgerStore(state_db_path()) as store:
        print(render_day_usage(store.day_table(days=args.days)))
        if args.recent:
            print(render_recent(store.recent(args.recent, run_id=args.run)))
        runs = store.runs(limit=5)
        if runs:
            print("\nRecent runs")
            print("=" * 78)
            for run_id, calls, last in runs:
                print(f"  {run_id:<24} {calls:>4} calls   last {last[:19]}")
    return 0


def cmd_dashboard(args) -> int:
    """Serve the read-only view until interrupted."""
    serve(host=args.host, port=args.port)
    return 0


def cmd_discover(args) -> int:
    """Ask every spare key what it is worth, without spending a token."""
    found = discover_all(only=_providers_arg(args.providers))
    print(render_discovery(found))
    return 0


def cmd_doctor(args) -> int:
    checks = run_doctor(
        probe=args.probe,
        providers=_providers_arg(args.providers),
    )
    print(render_doctor(checks))
    if not args.probe:
        print(
            "\n  The wire names in models.yaml are still unverified. "
            "`llmorch doctor --probe` confirms each with one live call."
        )
    return 1 if any(c.failed for c in checks) else 0


def cmd_run(args) -> int:
    session = _setup(args)
    try:
        return _execute(session, args)
    finally:
        session.close()


def cmd_resume(args) -> int:
    """Pick a run back up after a quota wall, without re-buying finished work."""
    if args.list:
        print(render_resume_list(list_checkpoints()))
        return 0

    try:
        book = load_run(args.run_id) if args.run_id else latest_resumable()
    except LLMOrchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if book is None:
        print("Nothing to resume — no run has unfinished work.")
        return 0
    if book.is_complete:
        print(f"{book.run_id} is already complete ({len(book.completed)} nodes).")
        return 0

    # The checkpoint carries the task, so the graph it is replayed onto is the
    # graph it was written for.
    args.task = book.task
    session = _setup(args, run_id=book.run_id, stored_plan=book.plan)
    try:
        check_applies(book, session.config.task, session.graph.nodes)

        waiting = book.seconds_until_resumable()
        if waiting > 0 and not args.force:
            hours, rest = divmod(int(waiting), 3600)
            print(
                f"{book.run_id} is still blocked for {hours}h{rest // 60:02d}m: "
                f"{', '.join(sorted(book.blocked_until))} has not reset yet.\n"
                "Resuming now would spend the remaining models on work that is "
                "waiting for a specific one. Use --force to go anyway."
            )
            return 2

        print(
            f"Resuming {book.run_id}: {len(book.completed)} node(s) already "
            f"done, {len(book.unfinished)} to go"
        )
        return _execute(session, args, resume=book)
    finally:
        session.close()


def _bid(session: Session, args) -> list:
    """Run a bidding round if it would actually inform the assignment.

    Skipped silently when it would not: one model, or fewer nodes than models.
    Spending four requests to allocate two nodes buys nothing the capability
    sheet does not already say.
    """
    policy = getattr(args, "negotiate", session.config.negotiate)
    candidates = sorted(session.scheduler.registry.model_ids)
    nodes = list(session.graph.nodes.values())
    if not should_bid(policy, nodes, candidates):
        return []

    deps = _worker_deps(
        manifest=session.manifest,
        governor=session.governor,
        registry=session.scheduler.registry,
        estimator=session.estimator,
        profiles=session.profiles,
        store=session.store,
        config=session.config,
        interface=session.scheduler.blackboard.interface,
    )
    bids = asyncio.run(collect_bids(nodes, deps=deps, candidates=candidates))
    if bids:
        bidders = len({b.model_id for b in bids})
        print(f"  {len(bids)} bids from {bidders} model(s)")
    return bids


def _execute(session: Session, args, *, resume: Checkpoint | None = None) -> int:
    """Plan, run, report, and write the output folder."""
    config = session.config
    mode = (
        "dry run — mock provider, no network"
        if config.dry_run
        else f"LIVE — {', '.join(sorted(session.scheduler.registry.model_ids))}"
    )
    print(f"Task: {config.task}")
    print(f"Run:  {config.run_id}  ({mode})")
    if session.restored:
        print(render_restored(session.restored))

    bids = _bid(session, args)
    plan = session.scheduler.plan(bids=bids)
    print(render_plan(plan, session.graph, explain=getattr(args, "explain", False)))

    outcome = asyncio.run(session.scheduler.run(plan, resume=resume))
    print(render_outcome(outcome, session.graph))
    print(render_spend(outcome))

    # Seed summaries the mock cannot produce itself.
    if config.dry_run:
        for node_id, result in outcome.results.items():
            if node_id in SUMMARIES and not result.summary:
                result.summary = SUMMARIES[node_id]

    report = materialize(config.output_dir, session.graph.nodes, outcome.results)

    print("\nOutput")
    print("=" * 78)
    print(f"  {config.output_dir}")
    print(f"  {len(report.written)} written, {len(report.stubbed)} stubbed")
    for path, reason in report.rejected:
        print(f"  ! rejected {path}: {reason}")

    # Do the pieces fit each other? Free, deterministic, and the only check
    # that looks across artifacts rather than at one in isolation.
    contract = check_contract(
        # The contract this run was planned against, not the demo's: checking a
        # CSV tool for the notes app's routes reports five faults that are only
        # the checker looking at the wrong document.
        session.scheduler.blackboard.interface,
        artifacts_from_results(session.graph.nodes, outcome.results),
    )
    print(render_contracts(contract))

    print(render_warnings(outcome.warnings))

    if outcome.degraded:
        print(
            f"\n  {len(outcome.degraded)} node(s) degraded. Their work is "
            f"checkpointed — `llmorch resume {config.run_id}` picks up only "
            "what is missing."
        )

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
    run.add_argument("--live", action="store_true", help="use real providers")
    run.add_argument(
        "--providers",
        default=DEFAULT_LIVE_PROVIDERS,
        help="comma-separated providers to use when live, or 'all' "
        f"(default: {DEFAULT_LIVE_PROVIDERS})",
    )
    run.add_argument("--explain", action="store_true", help="show the scoring maths")
    run.add_argument(
        "--decompose",
        action="store_true",
        help="plan with a model even for the built-in demo task",
    )
    run.add_argument(
        "--no-cache", action="store_true", help="ignore any cached plan for this task"
    )
    run.add_argument(
        "--negotiate",
        choices=["auto", "always", "never"],
        default="auto",
        help="whether models bid on nodes before assignment (default: auto)",
    )
    run.add_argument("--review", choices=["off", "code", "all"], default="code")
    run.add_argument("--allow-paid", action="store_true")
    run.add_argument("--max-usd", type=float, default=0.0)
    run.add_argument("--max-nodes", type=int, default=10)
    run.add_argument("--concurrency", type=int, default=4)
    run.set_defaults(func=cmd_run)

    resume = sub.add_parser(
        "resume", help="continue a run that hit a quota wall, skipping finished work"
    )
    resume.add_argument("run_id", nargs="?", default=None, help="default: most recent")
    resume.add_argument("--list", action="store_true", help="show resumable runs")
    resume.add_argument(
        "--force", action="store_true", help="resume before the blocked model resets"
    )
    resume.add_argument("--live", action="store_true")
    resume.add_argument("--providers", default=DEFAULT_LIVE_PROVIDERS)
    resume.add_argument("--explain", action="store_true")
    resume.add_argument("--review", choices=["off", "code", "all"], default="code")
    resume.add_argument("--max-nodes", type=int, default=10)
    resume.add_argument("--concurrency", type=int, default=4)
    resume.set_defaults(func=cmd_resume, task=None)

    plan = sub.add_parser("plan", help="show the assignment without executing")
    plan.add_argument("task", nargs="?", default=None)
    plan.add_argument("--explain", action="store_true")
    plan.add_argument("--decompose", action="store_true")
    plan.add_argument("--no-cache", action="store_true")
    plan.add_argument("--live", action="store_true", help="plan with a real model")
    plan.add_argument("--providers", default=DEFAULT_LIVE_PROVIDERS)
    plan.add_argument("--negotiate", choices=["auto", "always", "never"], default="never")
    plan.add_argument("--max-nodes", type=int, default=10)
    plan.set_defaults(func=cmd_plan)

    quota = sub.add_parser("quota", help="show per-provider headroom")
    quota.set_defaults(func=cmd_quota, task=None)

    ledger = sub.add_parser("ledger", help="show recorded usage across runs")
    ledger.add_argument("--days", type=int, default=3, help="how many recorded days")
    ledger.add_argument("--recent", type=int, default=0, help="also list the last N calls")
    ledger.add_argument("--run", default=None, help="restrict --recent to one run id")
    ledger.set_defaults(func=cmd_ledger, task=None)

    dashboard = sub.add_parser(
        "dashboard", help="serve a read-only view of quota, runs and spend"
    )
    dashboard.add_argument("--port", type=int, default=DEFAULT_PORT)
    dashboard.add_argument(
        "--host", default="127.0.0.1", help="loopback addresses only"
    )
    dashboard.set_defaults(func=cmd_dashboard, task=None)

    discover = sub.add_parser(
        "discover", help="ask each configured key which models it can reach"
    )
    discover.add_argument(
        "--providers", default="all", help="comma-separated candidates, or 'all'"
    )
    discover.set_defaults(func=cmd_discover, task=None)

    doctor = sub.add_parser("doctor", help="pre-flight checks before spending quota")
    doctor.add_argument(
        "--probe",
        action="store_true",
        help="confirm each wire name with one live call (costs quota)",
    )
    doctor.add_argument(
        "--providers",
        default=DEFAULT_LIVE_PROVIDERS,
        help=f"which providers to probe, or 'all' (default: {DEFAULT_LIVE_PROVIDERS})",
    )
    doctor.set_defaults(func=cmd_doctor, task=None)

    return parser


def _writable_console() -> None:
    """Make stdout accept the report tables on a legacy console.

    The plan and quota tables use box-drawing and bullet characters. On a
    Windows console still defaulting to cp1252, printing them raises
    UnicodeEncodeError *after* the work is done — the run completes, the
    artifacts are written, and the summary that says so is lost.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # already wrapped, or not a tty
            pass


def main(argv: list[str] | None = None) -> int:
    _writable_console()
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except LLMOrchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
