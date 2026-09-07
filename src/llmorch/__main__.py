"""Command line entry point.

    llmorch run "build a notes app"          # mock provider, no network
    llmorch run --live "build a notes app"   # real requests, Groq only [M2]
    llmorch                                  # the setup page, in a browser
    llmorch start                            # a session, using what it saved
    llmorch ask "what serves /api/items?"    # a question, not an instruction
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
import dataclasses
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from . import settings as settings_module
from .config import RunConfig, load_dotenv, runs_dir, state_db_path
from .configure.server import DEFAULT_PORT as CONFIGURE_PORT
from .configure.server import serve as configure_serve
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
from .chat import Conversation, latest_session, merge_interfaces
from .engine.materialize import materialize
from .engine.scheduler import Scheduler
from .engine.smoke import smoke_run
from .engine.worker import WorkerDeps
from .errors import LLMOrchError
from .providers.base import ProviderRegistry
from .providers.mock import MockProvider
from .providers.openai_compat import build_live_registry
from .quota.estimator import TokenEstimator
from .negotiate import plancache
from .negotiate.bidding import collect_bids, should_bid
from .negotiate.answer import (
    answer,
    files_named,
    pick_answerer,
    take_excerpts,
)
from .negotiate.decompose import (
    DecomposeError,
    decompose,
    pick_planner,
    plan_signature,
    revise,
)
from .negotiate.intent import Intent, classify
from .negotiate.profiles import Profiles
from .quota.governor import Governor
from .quota.store import DayUsage, LedgerStore, restore_governor
from .registry.manifest import Manifest, load_manifest
from .types import InterfaceContract
from .report.document import REPORT_NAME, render_run_report
from .report.ledger import render_day_usage, render_recent, render_restored
from .report.render import (
    render_contracts,
    render_discovery,
    render_doctor,
    render_resume_list,
    render_outcome,
    render_plan,
    render_quota,
    render_smoke,
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
class Roster:
    """Who can be called, under what accounting — assembled without a plan.

    Split out from `Session` because a question is not a build. Answering one
    needs the models, the governor and the ledger and nothing else; planning it
    would spend the request that asking a question exists to avoid.
    """

    manifest: Manifest
    config: RunConfig
    governor: Governor
    registry: ProviderRegistry
    estimator: TokenEstimator
    store: LedgerStore | None
    restored: dict[str, DayUsage]
    mock: MockProvider | None
    profiles: Profiles
    warnings: list[str] = field(default_factory=list)

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


@dataclass(slots=True)
class Session:
    """A roster with a plan on top: everything a build needs."""

    roster: Roster
    graph: TaskGraph
    scheduler: Scheduler

    # The roster's parts are reached through the session, so a command that
    # holds one does not have to know which of the two assembled what.
    @property
    def manifest(self) -> Manifest:
        return self.roster.manifest

    @property
    def config(self) -> RunConfig:
        return self.roster.config

    @property
    def governor(self) -> Governor:
        return self.roster.governor

    @property
    def store(self) -> LedgerStore | None:
        return self.roster.store

    @property
    def estimator(self) -> TokenEstimator:
        return self.roster.estimator

    @property
    def restored(self) -> dict[str, DayUsage]:
        return self.roster.restored

    @property
    def mock(self) -> MockProvider | None:
        return self.roster.mock

    @property
    def profiles(self) -> Profiles:
        return self.roster.profiles

    def close(self) -> None:
        self.roster.close()


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

    # A change to a project that already exists is planned against what that
    # project is, and never against the cache or the demo graph: the same
    # sentence means something different on turn four than it did on turn one.
    history: Conversation | None = getattr(args, "history", None)
    if history is not None and history.started:
        deps = _worker_deps(
            manifest=manifest, governor=governor, registry=registry,
            estimator=estimator, profiles=profiles, store=store, config=config,
            interface=history.interface,
        )
        planner = pick_planner(manifest, sorted(registry.model_ids))
        if planner is None:
            raise DecomposeError("no model available to plan this change")
        change = asyncio.run(
            revise(
                task,
                deps=deps,
                model_id=planner,
                memory=history.render_memory(),
                interface_text=Blackboard(interface=history.interface).interface_text(),
                max_nodes=config.max_nodes,
            )
        )
        return (
            change.nodes,
            merge_interfaces(history.interface, change.interface),
            f"{planner} planned {len(change.nodes)} change(s) against "
            f"{len(history.files)} existing file(s)",
        )

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


def _roster(args, *, run_id: str | None = None) -> Roster:
    """Assemble who can be called and how their usage is accounted for.

    No planning happens here: this is everything a single request needs, which
    is all a question needs.
    """
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

    warnings: list[str] = []
    only = _providers_arg(getattr(args, "providers", None))

    # Which models the setup page left ticked. Applied in both modes, so a mock
    # run rehearses the roster a live one would use.
    #
    # Unknown ids are dropped rather than fatal: this list is a file written
    # when `models.yaml` said something else, and a model retired from the
    # manifest must not be the reason a session refuses to start.
    wanted = {m for m in getattr(args, "models", ()) or ()}
    if wanted:
        known = {m.id for m in manifest.models}
        stale = sorted(wanted - known)
        if stale:
            warnings.append(
                f"settings name {', '.join(stale)}, which the manifest no longer "
                "declares — ignored"
            )
        wanted &= known
    if wanted:
        manifest = manifest.restricted_to_models(wanted)
        empty = manifest.unstaffed_roles()
        if empty:
            warnings.append(
                f"no model left to serve {', '.join(sorted(r.value for r in empty))} "
                "— a node with that role cannot run"
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

        # The roster is narrowed to what can actually be called, *before*
        # anything is built on it. The assignment, the failover chains and the
        # governor all read the manifest, and a model left enabled there that no
        # client serves is a node assigned to nobody: the run fails looking for
        # a provider that was never built, after the plan has been printed and
        # the planning request already spent.
        #
        # Two ways a model gets left behind, and they are the same fault: named
        # out by `--providers`, or enabled in the manifest with no key in the
        # environment. `restricted_to` is what makes the roster agree with the
        # registry in both cases.
        if only:
            manifest = manifest.restricted_to(only)  # also rejects an unknown name
        registry, _ = build_live_registry(manifest, only_providers=only)

        reachable = {manifest.vendor_of(m) for m in registry.model_ids}
        keyless = sorted(
            {m.provider for m in manifest.enabled_models} - reachable
        )
        if keyless:
            warnings.append(
                f"no key for {', '.join(keyless)} — left out of this run's roster"
            )
        manifest = manifest.restricted_to(reachable)

        stranded = manifest.single_vendor_roles()
        if stranded:
            warnings.append(
                f"{', '.join(sorted(r.value for r in stranded))} can only be served "
                f"by {', '.join(sorted(reachable))} in this run — a failure there "
                "has nowhere to fail over to"
            )

    governor = Governor(
        manifest,
        max_usd=config.max_usd,
        allow_paid=config.allow_paid,
        safety_factor=config.token_safety_factor,
    )
    if store is not None:
        restored = restore_governor(governor, store, manifest)

    return Roster(
        manifest=manifest,
        config=config,
        governor=governor,
        registry=registry,
        estimator=estimator,
        store=store,
        restored=restored,
        mock=mock,
        profiles=profiles,
        warnings=warnings,
    )


def _setup(
    args, *, run_id: str | None = None, stored_plan: dict | None = None
) -> Session:
    roster = _roster(args, run_id=run_id)
    manifest, config = roster.manifest, roster.config
    governor, registry = roster.governor, roster.registry
    estimator, store, profiles = roster.estimator, roster.store, roster.profiles

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
    graph.warnings.extend(roster.warnings)
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
    return Session(roster=roster, graph=graph, scheduler=scheduler)


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


CHAT_HELP = """\
  <a question>      answered from what this session knows; nothing is built
  <anything else>   an instruction: the first builds, the rest change what is built
  /ask <question>   answer it, whatever it looks like
  /build <thing>    build it, whatever it looks like
  /files            what the project consists of now
  /history          what you have asked so far
  /report           where this session's report.md is
  /quit             leave (the conversation is saved after every turn)

  A line ending in "?" is a question; "add tags" is an instruction; "thanks" is
  neither and costs nothing. /ask and /build settle it when the guess is wrong.
"""


# What the contract checker can read, and nothing else. `--smoke-install` may
# have put a node_modules of tens of thousands of files in this folder, and a
# smoke run may have left a SQLite file: walking either is a waste at best.
_READABLE = (".py", ".js", ".mjs", ".ts", ".html", ".htm", ".css", ".sql", ".json")
_SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "vendor", "dist"}
_MAX_READ_BYTES = 400_000


def _read_output(output_dir: Path) -> dict[str, str]:
    """What is on disk now, as the contract checker wants it.

    Read from the folder rather than from memory because the folder is the
    truth: it holds what every earlier turn wrote, including the turns that ran
    in a previous process.
    """
    if not output_dir.is_dir():
        return {}

    files: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "README.md":
            continue
        relative = path.relative_to(output_dir)
        if _SKIP_DIRS.intersection(relative.parts):
            continue
        if path.suffix.lower() not in _READABLE:
            continue
        try:
            if path.stat().st_size > _MAX_READ_BYTES:
                continue
            files[relative.as_posix()] = path.read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
    return files


def _chat_command(line: str, history: Conversation, config_dir: Path) -> bool:
    """Handle a `/command`. Returns False when the session should end."""
    command = line.split()[0].lower()

    if command in ("/quit", "/exit", "/q"):
        return False
    if command in ("/help", "/?"):
        print(CHAT_HELP)
    elif command == "/files":
        if not history.files:
            print("  nothing built yet")
        for path in sorted(history.files):
            note = history.files[path]
            print(f"  {path:<20} {note.role:<10} {note.summary.splitlines()[0][:44]}")
    elif command == "/history":
        for index, turn in enumerate(history.turns, start=1):
            degraded = f"  ({len(turn.degraded)} degraded)" if turn.degraded else ""
            mark = {"ask": "?", "remark": "·"}.get(turn.kind, " ")
            print(f"  {index}. {mark} {turn.instruction}{degraded}")
    elif command == "/report":
        print(f"  {config_dir / REPORT_NAME}")
    else:
        print(f"  unknown command {command}; /help lists them")
    return True


def _answer_question(
    args, history: Conversation, run_dir: Path, question: str
) -> None:
    """One question, answered against what this session already knows.

    No graph, no plan, no files. The whole point of the lane is that a question
    costs one request and leaves the project exactly as it was.
    """
    roster = _roster(args, run_id=history.session_id)
    try:
        model_id = pick_answerer(roster.manifest, sorted(roster.registry.model_ids))
        if model_id is None:
            print("  no model available to answer")
            return

        # Only what the question named. A question that names no file is
        # answered from summaries, which is what the session remembers anyway.
        on_disk = _read_output(run_dir / "output")
        named = files_named(question, sorted(on_disk))
        excerpts = take_excerpts(on_disk, named)

        deps = _worker_deps(
            manifest=roster.manifest,
            governor=roster.governor,
            registry=roster.registry,
            estimator=roster.estimator,
            profiles=roster.profiles,
            store=roster.store,
            config=roster.config,
            interface=history.interface,
        )
        reply = asyncio.run(
            answer(
                question,
                deps=deps,
                model_id=model_id,
                memory=history.render_for_answer(),
                interface_text=Blackboard(interface=history.interface).interface_text(),
                excerpts=excerpts,
                exchanges=history.render_exchanges(),
            )
        )
    except LLMOrchError as exc:
        print(f"  {exc}")
        return
    finally:
        roster.close()

    for line in reply.text.splitlines():
        print(f"  {line}" if line.strip() else "")
    read = f", reading {', '.join(reply.files_read)}" if reply.files_read else ""
    print(f"  — {reply.model_id}{read}\n")

    history.record_said(question, kind="ask", answer=reply.text)
    history.save()


def _chat_turn(
    args,
    history: Conversation,
    run_dir: Path,
    line: str,
    *,
    intent: Intent | None = None,
) -> None:
    """One line of the conversation, in whichever lane it belongs to.

    The classification is free and happens before anything is spent, which is
    the entire point: "looks good" used to buy a planning request in order to be
    told there was nothing to plan.
    """
    lane = intent or classify(line)

    if lane is Intent.REMARK:
        print("  noted — nothing to build\n")
        history.record_said(line, kind="remark")
        history.save()
        return

    if lane is Intent.ASK:
        _answer_question(args, history, run_dir, line)
        return

    args.task = line
    args.history = history
    prior = _read_output(run_dir / "output")

    try:
        session = _setup(args, run_id=history.session_id)
    except LLMOrchError as exc:
        print(f"  {exc}")
        return

    try:
        if session.graph.nodes:
            _execute(session, args, prior=prior, history=history)
        else:
            # The planner read the instruction and concluded the project already
            # satisfies it. Recorded, so the next turn knows it was said and
            # `/history` shows it.
            print("  nothing to change")
            history.record(line, {}, {}, session.scheduler.blackboard.interface)
            history.save()
    except LLMOrchError as exc:
        print(f"  {exc}")
    finally:
        session.close()


def cmd_chat(args) -> int:
    """A session with the orchestrator rather than one shot at it."""
    session_id = args.session or (latest_session() if args.continue_ else None)
    history = Conversation.load(session_id) if session_id else None
    if history is None:
        history = Conversation(session_id=session_id or _run_id())
        print(f"New session {history.session_id}.")
    else:
        print(
            f"Resuming {history.session_id}: {len(history.turns)} turn(s), "
            f"{len(history.files)} file(s)."
        )

    print("Type an instruction, or /help. Ctrl-D to leave.\n")
    run_dir = runs_dir() / history.session_id

    # `llmorch "build a notes app"` starts the session with that already said,
    # so the shortest way in is one line rather than two.
    if getattr(args, "first", None):
        print(f"> {args.first}")
        _chat_turn(args, history, run_dir, args.first)

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            # `/ask` and `/build` are not queries about the session, they are
            # turns with the lane named outright — which is what lets the
            # classifier stay small instead of growing a case per phrasing.
            verb, _, rest = line.partition(" ")
            forced = {"/ask": Intent.ASK, "/build": Intent.BUILD}.get(verb.lower())
            if forced is not None:
                if rest.strip():
                    _chat_turn(args, history, run_dir, rest.strip(), intent=forced)
                else:
                    print(f"  {verb} needs something after it")
                continue
            if not _chat_command(line, history, run_dir):
                break
            continue

        _chat_turn(args, history, run_dir, line)

    print(f"Session {history.session_id} saved to {history.path}")
    return 0


def cmd_configure(args) -> int:
    """Open the setup page and stay up until it is closed."""
    configure_serve(
        port=getattr(args, "port", CONFIGURE_PORT),
        open_browser=not getattr(args, "no_browser", False),
    )
    return 0


def cmd_start(args) -> int:
    """A session using what the setup page saved.

    The file supplies the defaults and the command line still wins, because a
    stored preference should never be the reason you cannot do something once.
    """
    chosen, complaint = settings_module.read()
    if complaint:
        print(f"warning: {complaint}")
    if getattr(args, "mock", False):
        chosen = dataclasses.replace(chosen, live=False)
    elif getattr(args, "live", False):
        chosen = dataclasses.replace(chosen, live=True)

    args.live = chosen.live
    args.providers = ",".join(chosen.providers) if chosen.providers else "all"
    args.models = chosen.models
    args.review = chosen.review
    args.smoke = chosen.smoke
    args.smoke_install = chosen.smoke_install
    args.max_nodes = chosen.max_nodes
    args.concurrency = chosen.concurrency

    if not chosen.configured:
        print("Nothing saved yet — running on defaults. `llmorch` opens the setup page.")
    print(f"Settings: {chosen.summary()}")
    return cmd_chat(args)


def cmd_ask(args) -> int:
    """One question about a session, without opening one.

    The same lane `chat` routes a question into, reached from a shell prompt —
    for the case where the project is already built and the question is the only
    thing being said.
    """
    session_id = args.session or latest_session()
    history = Conversation.load(session_id) if session_id else None
    if history is None:
        print(
            "No session to ask about yet. `llmorch \"build a notes app\"` starts one.",
            file=sys.stderr,
        )
        return 2

    print(f"Session {history.session_id}: {len(history.files)} file(s).\n")
    _answer_question(args, history, runs_dir() / history.session_id, args.question)
    return 0


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


def _execute(
    session: Session,
    args,
    *,
    resume: Checkpoint | None = None,
    prior: dict[str, str] | None = None,
    history: Conversation | None = None,
) -> int:
    """Plan, run, report, and write the output folder.

    `prior` is what earlier turns of a conversation already wrote. It matters
    only to the contract check, which asks whether the artifacts agree with each
    other: given one turn's three changed files it would report the other five
    pages as missing, which is the checker looking at a fragment and describing
    it as the project.
    """
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
        {**(prior or {}), **artifacts_from_results(session.graph.nodes, outcome.results)},
    )
    print(render_contracts(contract))

    # The only check that runs the code rather than reading it. Opt-in: every
    # other step treats model output as untrusted data, and this one hands it
    # the interpreter.
    smoke = None
    wants_install = getattr(args, "smoke_install", False)
    if getattr(args, "smoke", False) or wants_install:
        smoke = smoke_run(
            config.output_dir,
            session.scheduler.blackboard.interface,
            install=wants_install,
        )
        print(render_smoke(smoke))

    # The same findings, written down. Everything above this line lives in
    # scrollback; the artifacts it describes live on disk indefinitely.
    report_path = config.run_dir / REPORT_NAME
    report_path.write_text(
        render_run_report(
            config=config,
            graph=session.graph,
            plan=plan,
            outcome=outcome,
            materialized=report,
            contract=contract,
            smoke=smoke,
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(f"  {report_path}")

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

    if history is not None:
        history.record(
            config.task,
            session.graph.nodes,
            outcome.results,
            session.scheduler.blackboard.interface,
        )
        history.save()

    # A project that was executed and failed is a failed run, even when every
    # node reported success — that gap is the whole reason this step exists.
    if smoke is not None and smoke.ran and smoke.errors:
        return 1
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
    run.add_argument(
        "--smoke",
        action="store_true",
        help="start the generated project and drive its routes and pages "
        "(runs model-written code; off by default)",
    )
    run.add_argument(
        "--smoke-install",
        action="store_true",
        help="--smoke, plus a lockfile-pinned dependency install first "
        "(reaches the network; package install scripts stay disabled)",
    )
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
    resume.add_argument(
        "--smoke",
        action="store_true",
        help="start the completed project and drive it (runs model-written code)",
    )
    resume.add_argument(
        "--smoke-install",
        action="store_true",
        help="--smoke, plus a lockfile-pinned dependency install first",
    )
    resume.add_argument("--max-nodes", type=int, default=10)
    resume.add_argument("--concurrency", type=int, default=4)
    resume.set_defaults(func=cmd_resume, task=None)

    chat = sub.add_parser(
        "chat",
        aliases=["cli"],
        help="a session: the first instruction builds, the rest change it",
    )
    chat.add_argument(
        "first",
        nargs="?",
        default=None,
        help="an opening instruction, said for you as the first turn",
    )
    chat.add_argument("--session", default=None, help="continue a specific session id")
    chat.add_argument(
        "--continue",
        dest="continue_",
        action="store_true",
        help="continue the most recent session",
    )
    chat.add_argument("--live", action="store_true", help="use real providers")
    chat.add_argument("--providers", default=DEFAULT_LIVE_PROVIDERS)
    chat.add_argument("--review", choices=["off", "code", "all"], default="code")
    chat.add_argument(
        "--smoke",
        action="store_true",
        help="after each turn, start the project and drive it",
    )
    chat.add_argument("--smoke-install", action="store_true")
    chat.add_argument("--max-nodes", type=int, default=10)
    chat.add_argument("--concurrency", type=int, default=4)
    chat.add_argument("--explain", action="store_true")
    chat.set_defaults(func=cmd_chat, task=None, negotiate="auto")

    configure = sub.add_parser(
        "configure",
        aliases=["config"],
        help="open the setup page in a browser (what a bare `llmorch` does)",
    )
    configure.add_argument("--port", type=int, default=CONFIGURE_PORT)
    configure.add_argument(
        "--no-browser",
        action="store_true",
        help="serve the page but do not open it",
    )
    configure.set_defaults(func=cmd_configure, task=None)

    start = sub.add_parser(
        "start",
        help="a session using whatever the setup page saved",
    )
    start.add_argument(
        "first",
        nargs="?",
        default=None,
        help="an opening instruction, said for you as the first turn",
    )
    start.add_argument("--session", default=None, help="continue a specific session id")
    start.add_argument(
        "--continue",
        dest="continue_",
        action="store_true",
        help="continue the most recent session",
    )
    start.add_argument(
        "--live", action="store_true", help="override the saved mode for this session"
    )
    start.add_argument(
        "--mock",
        action="store_true",
        help="override the saved mode: mock provider, no network",
    )
    start.add_argument("--explain", action="store_true")
    start.set_defaults(func=cmd_start, task=None, negotiate="auto")

    ask = sub.add_parser(
        "ask",
        help="answer a question about a session, building nothing",
    )
    ask.add_argument("question")
    ask.add_argument(
        "--session", default=None, help="which session (default: the most recent)"
    )
    ask.add_argument("--live", action="store_true", help="use real providers")
    ask.add_argument("--providers", default=DEFAULT_LIVE_PROVIDERS)
    ask.set_defaults(func=cmd_ask, task=None)

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

    # Recorded so `main` can tell "llmorch run …" from "llmorch build a thing".
    parser.subcommand_names = frozenset(sub.choices)

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

    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)

    # `llmorch` on its own opens the setup page: the choice of models and of
    # live-versus-mock is the thing a person needs to make first, and making it
    # on the command line means remaking it every time — with a quiet failure
    # (a session replaying fixtures) as the cost of forgetting.
    #
    # `llmorch "build a notes app"` and `llmorch --live` still open a session,
    # because both name something to do. Anything naming a subcommand is
    # untouched, so `run`, `start`, `doctor` and the rest behave as they did.
    if not argv:
        argv = ["configure"]
    elif argv[0] not in parser.subcommand_names and argv[0] not in ("-h", "--help"):
        argv = ["chat", *argv]

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except LLMOrchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
