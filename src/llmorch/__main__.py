"""Command line entry point.

    llmorch run "build a notes app"        dry run against the mock
    llmorch run --live "build a notes app" real providers, real quota
    llmorch plan --explain "build a notes app"
    llmorch quota
    llmorch ledger
    llmorch doctor [--live]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import (
    RunConfig,
    get_api_key,
    has_api_key,
    load_dotenv,
    project_root,
    runs_dir,
    state_db_path,
)
from .demo.website import ARTIFACTS, INTERFACE, SUMMARIES, TASK, build_nodes
from .engine.blackboard import Blackboard
from .engine.graph import TaskGraph
from .engine.health import HealthTracker
from .engine.materialize import materialize
from .engine.scheduler import Scheduler
from .errors import LLMOrchError, ProviderError
from .providers.base import ProviderRegistry
from .providers.mock import MockProvider
from .providers.openai_compat import build_provider
from .quota.governor import Governor
from .quota.store import LedgerStore, build_event
from .registry.manifest import load_manifest
from .report.ledger import (
    render_run_detail,
    render_runs,
    render_today,
    render_totals,
)
from .report.render import (
    render_outcome,
    render_plan,
    render_quota,
    render_spend,
    render_warnings,
)
from .types import ChatRequest, Message, Usage


def _force_utf8_console() -> None:
    """Windows consoles default to cp1252, which cannot encode the box glyphs
    and em dashes the report tables use — printing one raises mid-render and
    kills the run. Reconfiguring is cheap and a no-op where stdout is already
    UTF-8; `errors="replace"` keeps a redirected pipe from ever taking us down.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass  # detached or already-wrapped stream; render as-is


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


# --------------------------------------------------------------------------
# Provider wiring
# --------------------------------------------------------------------------


def _mock_registry(manifest) -> tuple[ProviderRegistry, MockProvider]:
    """One mock instance shared by every model id, so a single call log covers
    the whole run and fault injection can target any model."""
    provider = MockProvider(responses=dict(ARTIFACTS))
    registry = ProviderRegistry()
    for model in manifest.enabled_models:
        registry.register(model.id, provider)
    return registry, provider


def _live_registry(manifest, *, only: str | None = None) -> tuple[ProviderRegistry, list[str]]:
    """Real adapters for every enabled provider that has a key.

    A missing key disables that provider rather than failing the run: with the
    fallback chains spanning two vendors, one key is enough to work, and M2
    exists specifically to be run on Groq alone.
    """
    registry = ProviderRegistry()
    notes: list[str] = []

    for name, spec in manifest.providers.items():
        if not spec.enabled or (only and name != only):
            continue
        if not has_api_key(spec.api_key_env):
            notes.append(f"{name} is enabled but {spec.api_key_env} is unset — skipped")
            continue

        provider = build_provider(
            spec, manifest.models, get_api_key(spec.api_key_env, provider=name)
        )
        for model in manifest.models:
            if model.provider == name:
                registry.register(model.id, provider)

    return registry, notes


def _restore_governor(governor: Governor, store: LedgerStore) -> None:
    """Seed today's counters from the ledger before admitting anything."""
    governor.restore_daily(store.usage_by_model_today(governor.reset_timezones()))


def _setup(args) -> tuple:
    manifest = load_manifest()
    live = bool(getattr(args, "live", False))

    # Build the registry first when live: a model whose provider has no key
    # cannot be called, so it must be kept out of the candidate set before the
    # planner ever scores it. Otherwise the plan assigns work to it and the
    # run discovers the gap at call time, one node at a time.
    store: LedgerStore | None = None
    notes: list[str] = []
    excluded: set[str] = set()
    if live:
        registry, notes = _live_registry(manifest)
        mock = None
        excluded = {m.id for m in manifest.enabled_models if m.id not in registry}
        notes += [f"{m} has no key — excluded from planning" for m in sorted(excluded)]
    else:
        registry, mock = _mock_registry(manifest)

    config = RunConfig(
        task=args.task or TASK,
        run_id=_run_id(),
        dry_run=not live,
        allow_paid=getattr(args, "allow_paid", False),
        max_usd=Decimal(str(getattr(args, "max_usd", 0) or 0)),
        review=getattr(args, "review", "code"),
        max_nodes=getattr(args, "max_nodes", 10),
        max_concurrency=getattr(args, "concurrency", 4),
        excluded_models=frozenset(excluded),
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

    if live:
        store = LedgerStore()
        _restore_governor(governor, store)
        notes += [
            f"ledger holds {m}, which models.yaml no longer declares — ignored"
            for m in governor.unknown_restored
        ]

    # Excluding a model from planning is not enough: the failover chains come
    # straight from the manifest, so a keyless model stays a valid rung and
    # every node rediscovers the missing key on its own.
    health = HealthTracker(threshold=config.circuit_breaker_threshold)
    for model_id in sorted(excluded):
        health.mark_unconfigured(model_id, "no API key — excluded for this run")

    scheduler = Scheduler(
        graph,
        manifest,
        governor,
        registry,
        config=config,
        blackboard=blackboard,
        health=health,
        store=store,
    )
    return manifest, config, graph, governor, scheduler, mock, store, notes


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_plan(args) -> int:
    _, _, graph, _, scheduler, _, store, _ = _setup(args)
    plan = scheduler.plan()
    print(render_plan(plan, graph, explain=args.explain))
    print(render_warnings(graph.warnings))
    if store is not None:
        store.close()
    return 0


def cmd_quota(args) -> int:
    manifest = load_manifest()
    governor = Governor(manifest)

    with LedgerStore() as store:
        _restore_governor(governor, store)
        print(render_quota(governor.headroom()))
        print(
            render_today(
                store.usage_by_model_today(governor.reset_timezones()), manifest
            )
        )

    missing = [
        f"{name} ({spec.api_key_env})"
        for name, spec in manifest.providers.items()
        if spec.enabled and not has_api_key(spec.api_key_env)
    ]
    if missing:
        print(f"\n  No key for: {', '.join(missing)}. Dry runs are unaffected.")
    return 0


def cmd_ledger(args) -> int:
    manifest = load_manifest()
    with LedgerStore() as store:
        if args.run:
            print(render_run_detail(store.events_for_run(args.run), run_id=args.run))
            return 0

        print(render_runs(store.runs(limit=args.limit), limit=args.limit))
        governor = Governor(manifest)
        print(
            render_today(
                store.usage_by_model_today(governor.reset_timezones()), manifest
            )
        )
        if args.totals:
            print(render_totals(store.totals_by_model()))
        print(f"\n  Ledger: {store.path}")
    return 0


def cmd_run(args) -> int:
    manifest, config, graph, governor, scheduler, mock, store, notes = _setup(args)

    try:
        for note in notes:
            print(f"  ! {note}")

        if not config.dry_run and not scheduler.registry.model_ids:
            print(
                "No provider key is set, so --live has nothing to call.\n"
                "Add GROQ_API_KEY to .env (see .env.example), or drop --live to "
                "run against the mock.",
                file=sys.stderr,
            )
            return 2

        mode = "dry run — mock provider, no network" if config.dry_run else "LIVE"
        print(f"Task: {config.task}")
        print(f"Run:  {config.run_id}  ({mode})")

        plan = scheduler.plan()
        print(render_plan(plan, graph, explain=args.explain))

        outcome = asyncio.run(scheduler.run(plan))
        print(render_outcome(outcome, graph))
        print(render_spend(outcome))

        # Seed summaries the mock cannot produce itself.
        if config.dry_run:
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

        if not config.dry_run:
            print(f"\n  Recorded to the ledger as run {config.run_id}.")

        print("\nRun the result:")
        print(f"  python {config.output_dir / 'server.py'}")
        print("  then open http://localhost:8000")

        return 0 if outcome.all_succeeded else 1
    finally:
        if store is not None:
            store.close()


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def _check(ok: bool, label: str, detail: str = "") -> bool:
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def _note(label: str, detail: str = "") -> None:
    """A fact worth stating that is not a pass or a failure.

    A missing key is the common case here: it makes one provider unavailable to
    `--live` without anything being wrong, and reporting it as `ok` would make
    the check meaningless while `FAIL` would be plainly false.
    """
    print(f"  [--  ] {label}" + (f" — {detail}" if detail else ""))


def cmd_doctor(args) -> int:
    """Preflight. Offline by default; `--live` confirms the wire names.

    The offline half is free and catches most of what goes wrong. The live half
    exists for one specific reason: the wire names in models.yaml have never
    been checked against a real API, and a wrong one fails identically to a dead
    model — mid-run, after quota has already been spent on its neighbours. One
    trivial call each settles it for a few tokens.
    """
    print("Preflight")
    print("=" * 78)
    healthy = True

    # -- manifest ---------------------------------------------------------
    try:
        manifest = load_manifest()
        _check(True, "models.yaml", f"{len(manifest.enabled_models)} models enabled")
    except LLMOrchError as exc:
        _check(False, "models.yaml", str(exc))
        return 1

    # -- timezone database ------------------------------------------------
    # Windows ships none; without tzdata, Gemini's Pacific reset cannot be
    # resolved and every daily counter for it silently uses the wrong day.
    for name, spec in manifest.providers.items():
        if not spec.enabled:
            continue
        try:
            ZoneInfo(spec.reset_tz)
            _check(True, f"timezone {spec.reset_tz}", f"for {name}")
        except (ZoneInfoNotFoundError, ValueError) as exc:
            healthy = _check(False, f"timezone {spec.reset_tz}", f"{exc}; pip install tzdata")

    # -- writable state ---------------------------------------------------
    try:
        with LedgerStore() as store:
            runs = len(store.runs(limit=1))
        _check(True, "ledger", f"{state_db_path()}" + (" (has history)" if runs else " (empty)"))
    except Exception as exc:
        healthy = _check(False, "ledger", f"{state_db_path()}: {exc}")

    try:
        runs_dir().mkdir(parents=True, exist_ok=True)
        _check(True, "runs directory", str(runs_dir()))
    except OSError as exc:
        healthy = _check(False, "runs directory", str(exc))

    if not (project_root() / ".env").is_file():
        _note(".env", "absent — fine for dry runs; keys may come from the environment")

    # -- keys -------------------------------------------------------------
    # Presence only. A key's value is never printed, logged, or echoed.
    have_any = False
    for name, spec in manifest.providers.items():
        if not spec.enabled:
            continue
        present = has_api_key(spec.api_key_env)
        have_any = have_any or present
        if present:
            _check(True, f"key {spec.api_key_env}", "set")
        else:
            _note(f"key {spec.api_key_env}", f"unset — {name} unavailable to --live")

    if not have_any:
        print("\n  No provider keys are set. Dry runs work; --live has nothing to call.")

    if not args.live:
        print("\n  Offline checks only. Add --live to confirm the wire names,")
        print("  which have never been verified against a real API.")
        return 0 if healthy else 1

    if not have_any:
        return 1

    print()
    print("Live wire-name check")
    print("=" * 78)
    return 0 if _live_probe(manifest, only=args.provider) and healthy else 1


def _live_probe(manifest, *, only: str | None = None) -> bool:
    """One minimal call per model, confirming the wire name really resolves.

    Deliberately the smallest request that can succeed. It still costs a
    request against the daily cap — which is why Groq, at 14,400 a day, is the
    right place to debug this and Gemini, at 250, is checked but not iterated
    on.
    """
    registry, notes = _live_registry(manifest, only=only)
    for note in notes:
        print(f"  ! {note}")

    if not registry.model_ids:
        print("  nothing to probe")
        return False

    governor = Governor(manifest)
    healthy = True

    with LedgerStore() as store:
        _restore_governor(governor, store)

        for model_id in registry.model_ids:
            model = manifest.model(model_id)
            provider_spec = manifest.provider_of(model_id)

            ticket = governor.try_acquire(model_id, 16, 1)
            if not hasattr(ticket, "ticket_id"):
                healthy = _check(False, model_id, f"quota: {ticket.reason}")
                continue

            request = ChatRequest(
                model_id=model_id,
                messages=(Message("user", "Reply with the single word: ok"),),
                max_tokens=1,
                temperature=0.0,
            )

            try:
                response = asyncio.run(registry.get(model_id).chat(request))
            except ProviderError as exc:
                governor.release(ticket, "probe failed")
                healthy = _check(False, model_id, f"{model.wire_name}: {exc}")
                store.record(
                    build_event(
                        run_id="doctor",
                        node_id=None,
                        purpose="doctor",
                        provider=provider_spec.name,
                        model_id=model_id,
                        reset_tz=provider_spec.reset_tz,
                        est_prompt_tokens=16,
                        est_completion_tokens=1,
                        usage=Usage(),
                        ok=False,
                        http_status=int(getattr(exc, "status", 0) or 0),
                        error=str(exc)[:500],
                    )
                )
                continue

            governor.commit(ticket, response.usage)
            if response.rate_limit:
                governor.sync_from_headers(model_id, response.rate_limit)

            store.record(
                build_event(
                    run_id="doctor",
                    node_id=None,
                    purpose="doctor",
                    provider=provider_spec.name,
                    model_id=model_id,
                    reset_tz=provider_spec.reset_tz,
                    est_prompt_tokens=16,
                    est_completion_tokens=1,
                    usage=response.usage,
                    latency_ms=response.latency_ms,
                )
            )

            remaining = ""
            if response.rate_limit and response.rate_limit.remaining_requests is not None:
                remaining = f", {response.rate_limit.remaining_requests} requests left today"
            _check(
                True,
                model_id,
                f"{model.wire_name} answered in {response.latency_ms}ms{remaining}",
            )

    return healthy


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
    run.add_argument("--live", action="store_true", help="call real providers")
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

    ledger = sub.add_parser("ledger", help="show recorded usage across runs")
    ledger.add_argument("--run", help="show every request one run made")
    ledger.add_argument("--limit", type=int, default=20)
    ledger.add_argument("--totals", action="store_true", help="lifetime per model")
    ledger.set_defaults(func=cmd_ledger, task=None)

    doctor = sub.add_parser("doctor", help="preflight checks before a live run")
    doctor.add_argument(
        "--live",
        action="store_true",
        help="also confirm each wire name with one trivial call (costs quota)",
    )
    doctor.add_argument("--provider", help="probe only this provider")
    doctor.set_defaults(func=cmd_doctor, task=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except LLMOrchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
