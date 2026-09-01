"""Pre-flight diagnostics.

The point of `llmorch doctor` is to move every discoverable failure *before* the
first real request. Milestone 1 ran entirely on mocks, which means a set of
facts remain unverified on purpose: whether the keys exist, whether the wire
names in models.yaml are what the providers actually call those models, whether
the timezone database can resolve Pacific midnight, whether the ledger is
writable. Each of those, discovered mid-run, costs live quota — and on Gemini,
250 requests a day makes that expensive.

Offline checks always run. The live probe is opt-in (`--probe`) and defaults to
Groq alone, whose 14,400 requests/day makes it the safe place to be wrong.
Every probe goes through the governor and lands in the ledger like any other
call, because a diagnostic that bypasses admission control is a diagnostic that
can break the thing it is checking.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .config import has_api_key, state_db_path
from .errors import LLMOrchError
from .providers.openai_compat import Transport, build_live_registry
from .quota.governor import Governor
from .quota.store import LedgerStore, make_event, restore_governor
from .quota.windows import next_reset_utc
from .registry.manifest import Manifest, load_manifest
from .types import ChatRequest, Message, Ticket, Usage

OK = "ok"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"

PROBE_MAX_TOKENS = 16
PROBE_PROMPT = "Reply with the single word: ok"

# Below this, a model cannot be sent a prompt of any substance — the entire
# per-minute allowance is consumed by the reply it is allowed to write.
MIN_PROMPT_HEADROOM = 1000


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: str
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status == FAIL


# --------------------------------------------------------------------------
# Offline checks
# --------------------------------------------------------------------------


def check_manifest(manifest: Manifest) -> list[Check]:
    enabled = manifest.enabled_models
    vendors = sorted({m.provider for m in enabled})
    checks = [
        Check(
            "manifest",
            OK if enabled else FAIL,
            f"{len(manifest.models)} models declared, {len(enabled)} enabled "
            f"across {len(vendors)} vendor(s): {', '.join(vendors) or 'none'}",
        )
    ]

    # Every role's chain must still span two vendors *among enabled providers*,
    # or cross-vendor failover is decorative in practice even though the
    # manifest validated at load.
    single_vendor = []
    for role in manifest.roles:
        chain = manifest.chain(role)
        if len({manifest.vendor_of(m) for m in chain}) < 2:
            single_vendor.append(role.value)
    checks.append(
        Check(
            "failover reach",
            WARN if single_vendor else OK,
            f"role(s) with only one enabled vendor: {', '.join(single_vendor)}"
            if single_vendor
            else "every role can fail over to a second vendor right now",
        )
    )
    return checks


def check_request_ceilings(manifest: Manifest) -> list[Check]:
    """How much prompt room each model actually has under its provider's TPM."""
    checks: list[Check] = []
    for model in manifest.enabled_models:
        ceiling = manifest.max_request_tokens(model.id)
        headroom = ceiling - model.max_output
        status = OK if headroom >= MIN_PROMPT_HEADROOM else WARN
        checks.append(
            Check(
                f"ceiling {model.id}",
                status,
                f"{ceiling} tokens per request, {model.max_output} reserved for "
                f"output, leaving ~{headroom} for the prompt",
            )
        )
    return checks


def check_keys(manifest: Manifest) -> list[Check]:
    """Presence only. A key's value is never read into a diagnostic."""
    checks: list[Check] = []
    for name in sorted({m.provider for m in manifest.enabled_models}):
        spec = manifest.providers[name]
        present = has_api_key(spec.api_key_env)
        checks.append(
            Check(
                f"key {name}",
                OK if present else WARN,
                f"{spec.api_key_env} is set"
                if present
                else f"{spec.api_key_env} is unset — {name} cannot be called",
            )
        )
    return checks


def check_timezones(manifest: Manifest, *, now: datetime | None = None) -> list[Check]:
    """Resolve every reset timezone and say when each provider's day ends.

    On Windows this is a real failure mode rather than a formality: there is no
    system timezone database, so without `tzdata` installed Gemini's Pacific
    midnight cannot be resolved at all and its daily counter would never reset.
    """
    moment = now or datetime.now(timezone.utc)
    checks: list[Check] = []
    for name in sorted({m.provider for m in manifest.enabled_models}):
        spec = manifest.providers[name]
        try:
            ZoneInfo(spec.reset_tz)
            reset = next_reset_utc(moment, spec.reset_tz)
            hours = (reset - moment).total_seconds() / 3600
            checks.append(
                Check(
                    f"reset tz {name}",
                    OK,
                    f"{spec.reset_tz}; next reset in {hours:.1f}h "
                    f"({reset:%Y-%m-%d %H:%M} UTC)",
                )
            )
        except Exception as exc:  # ZoneInfoNotFoundError and friends
            checks.append(
                Check(
                    f"reset tz {name}",
                    FAIL,
                    f"cannot resolve {spec.reset_tz!r} ({exc}). Install tzdata: "
                    "without it the daily counter never rolls over.",
                )
            )
    return checks


def check_ledger(manifest: Manifest, store: LedgerStore) -> list[Check]:
    """Confirm the ledger is writable and report what it already knows."""
    checks: list[Check] = []
    try:
        store.open()
        version = store.get_meta("schema_version")
        checks.append(
            Check(
                "ledger",
                OK,
                f"{store.path} (schema v{version}, {store.total_events} events)",
            )
        )
    except Exception as exc:
        return [
            Check(
                "ledger",
                FAIL,
                f"cannot open {store.path}: {exc}. Quota would reset to zero on "
                "every process start, which is how a daily cap gets blown.",
            )
        ]

    governor = Governor(manifest)
    restored = restore_governor(governor, store, manifest)
    if restored:
        summary = ", ".join(
            f"{model_id} {usage.requests} req" for model_id, usage in sorted(restored.items())
        )
        checks.append(Check("spent today", WARN, f"already used: {summary}"))
    else:
        checks.append(
            Check("spent today", OK, "no calls recorded against today's quota yet")
        )
    return checks


def offline_checks(
    manifest: Manifest, store: LedgerStore, *, now: datetime | None = None
) -> list[Check]:
    return [
        *check_manifest(manifest),
        *check_keys(manifest),
        *check_timezones(manifest, now=now),
        *check_request_ceilings(manifest),
        *check_ledger(manifest, store),
    ]


# --------------------------------------------------------------------------
# Live probe
# --------------------------------------------------------------------------


async def probe_models(
    manifest: Manifest,
    *,
    providers: set[str] | None = None,
    store: LedgerStore | None = None,
    governor: Governor | None = None,
    transport: Transport | None = None,
    run_id: str = "doctor",
) -> list[Check]:
    """Confirm each wire name with one trivial call.

    This is the check that cannot be done offline and that nothing else in the
    system does for free: models.yaml carries an *unverified* guess at what each
    provider calls its models, and a wrong guess surfaces as a 404 in the middle
    of a run, after the planning request has already been spent.

    One call per model, sixteen output tokens, through the governor, recorded in
    the ledger. Header support is reported too — a provider that returns
    rate-limit headers can be tracked exactly rather than inferred.
    """
    try:
        registry, _ = build_live_registry(
            manifest, transport=transport, only_providers=providers
        )
    except LLMOrchError as exc:
        return [Check("probe", SKIP, str(exc))]

    governor = governor or Governor(manifest)
    checks: list[Check] = []

    for model_id in sorted(registry.model_ids):
        model = manifest.model(model_id)
        client = registry.get(model_id)

        ticket = governor.try_acquire(model_id, 32, PROBE_MAX_TOKENS)
        if not isinstance(ticket, Ticket):
            checks.append(
                Check(f"probe {model_id}", SKIP, f"not admitted: {ticket.reason}")
            )
            continue

        request = ChatRequest(
            model_id=model_id,
            messages=(Message("user", PROBE_PROMPT),),
            max_tokens=PROBE_MAX_TOKENS,
            timeout_s=30.0,
        )
        try:
            response = await client.chat(request)
        except LLMOrchError as exc:
            governor.release(ticket, "probe failed")
            status = getattr(exc, "status", None) or 0
            checks.append(
                Check(
                    f"probe {model_id}",
                    FAIL,
                    f"wire name {model.wire_name!r} did not answer: {exc}",
                )
            )
            if store is not None:
                store.record(
                    make_event(
                        run_id=run_id,
                        node_id=None,
                        purpose="doctor",
                        manifest=manifest,
                        model_id=model_id,
                        usage=Usage(),
                        est_prompt_tokens=32,
                        est_completion_tokens=PROBE_MAX_TOKENS,
                        ok=False,
                        http_status=status,
                        error=str(exc),
                    )
                )
            continue

        governor.commit(ticket, response.usage)
        if store is not None:
            store.record(
                make_event(
                    run_id=run_id,
                    node_id=None,
                    purpose="doctor",
                    manifest=manifest,
                    model_id=model_id,
                    usage=response.usage,
                    est_prompt_tokens=32,
                    est_completion_tokens=PROBE_MAX_TOKENS,
                    latency_ms=response.latency_ms,
                )
            )

        snapshot = response.rate_limit
        header_note = (
            f"headers: {snapshot.remaining_requests} requests left"
            if snapshot and snapshot.remaining_requests is not None
            else "no rate-limit headers (counting stays local)"
        )
        reported = response.model_reported
        drift = "" if reported.endswith(model.wire_name) else f" (server said {reported!r})"
        checks.append(
            Check(
                f"probe {model_id}",
                OK,
                f"{model.wire_name} answered in {response.latency_ms}ms, "
                f"{response.usage.total_tokens} tokens; {header_note}{drift}",
            )
        )

    return checks


def run_doctor(
    *,
    probe: bool = False,
    providers: set[str] | None = None,
    store: LedgerStore | None = None,
    transport: Transport | None = None,
) -> list[Check]:
    """Full diagnostic sweep. Offline always; live probe only when asked."""
    manifest = load_manifest()
    owned = store is None
    store = store or LedgerStore(state_db_path())

    try:
        checks = offline_checks(manifest, store)
        if probe:
            governor = Governor(manifest)
            restore_governor(governor, store, manifest)
            checks += asyncio.run(
                probe_models(
                    manifest,
                    providers=providers,
                    store=store,
                    governor=governor,
                    transport=transport,
                )
            )
        else:
            checks.append(
                Check(
                    "probe",
                    SKIP,
                    "wire names unverified — run `llmorch doctor --probe` to "
                    "confirm each with one live call",
                )
            )
        return checks
    finally:
        if owned:
            store.close()
