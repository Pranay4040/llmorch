"""Admission control.

Nothing calls a provider without a ticket from here. The governor answers one
question — *may this request proceed right now?* — and the value is in how
precisely it distinguishes the ways the answer can be "no":

    GRANTED          go
    WAIT             not now, but soon; `retry_after_s` says when
    EXHAUSTED_TODAY  not until the provider's next local midnight
    UNSERVABLE       never, at this size, on this model
    COST_BLOCKED     would spend real money without authorisation

`UNSERVABLE` versus `WAIT` is the distinction that matters most. A request
larger than a provider's per-minute token ceiling will not fit however long the
scheduler waits, so treating it as a wait condition hangs the run forever.
Against a 6,000 TPM provider this case is routine, not exotic.

Reservation model: capacity is reserved against an *estimate* at acquire time
and reconciled to the true count at commit. Without that, concurrent fan-out
would race — several requests would each check a counter that none of them had
yet incremented, and collectively blow the limit.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from ..errors import CostBlocked, QuotaExhausted, Unservable
from ..registry.manifest import Manifest, ProviderSpec
from ..types import (
    Admission,
    Denial,
    Headroom,
    LimitKind,
    LimitScope,
    Priority,
    RateLimitSnapshot,
    Ticket,
    Usage,
)
from .windows import Clock, DayCounter, SlidingWindow, SystemClock

_ticket_ids = itertools.count(1)


@dataclass(slots=True)
class _LimitState:
    """Live state for one (provider, scope-key, limit) triple."""

    rpm: SlidingWindow = field(default_factory=lambda: SlidingWindow(60.0))
    tpm: SlidingWindow = field(default_factory=lambda: SlidingWindow(60.0))
    rpd: DayCounter | None = None
    tpd: DayCounter | None = None
    cooldown_until: float | None = None
    """Server-asserted pause, from a 429's Retry-After."""
    exhausted_day: str | None = None
    """Provider-local day on which a daily cap was hit, per the server."""


class Governor:
    """Multi-limit admission controller across every provider in the manifest."""

    def __init__(
        self,
        manifest: Manifest,
        *,
        clock: Clock | None = None,
        max_usd: Decimal = Decimal("0"),
        allow_paid: bool = False,
        safety_factor: float = 1.25,
    ) -> None:
        self.manifest = manifest
        self.clock = clock or SystemClock()
        self.max_usd = max_usd
        self.allow_paid = allow_paid
        self.safety_factor = safety_factor

        self._states: dict[str, _LimitState] = {}
        self._spent_usd = Decimal("0")
        self._run_requests: dict[str, int] = {}
        self._tickets: dict[str, Ticket] = {}
        self._wakeup = asyncio.Event()

    # -- state keys -------------------------------------------------------

    def _key(self, provider_name: str, model_id: str, scope: LimitScope) -> str:
        """Account-scoped limits share one bucket across every model.

        This is why falling back to a different model on the same provider can
        buy nothing: if the limit is account-scoped, both models draw down the
        same counter.
        """
        if scope is LimitScope.ACCOUNT:
            return f"{provider_name}:*"
        return f"{provider_name}:{model_id}"

    def _state(self, key: str, provider: ProviderSpec) -> _LimitState:
        state = self._states.get(key)
        if state is None:
            state = _LimitState()
            if provider.limit(LimitKind.RPD):
                state.rpd = DayCounter(provider.reset_tz)
            if provider.limit(LimitKind.TPD):
                state.tpd = DayCounter(provider.reset_tz)
            self._states[key] = state
        return state

    def _states_for(self, model_id: str) -> list[tuple[_LimitState, ProviderSpec]]:
        """Every limit bucket a request against this model draws from."""
        provider = self.manifest.provider_of(model_id)
        scopes = {lim.scope for lim in provider.limits} or {LimitScope.MODEL}
        return [
            (self._state(self._key(provider.name, model_id, s), provider), provider)
            for s in scopes
        ]

    # -- admission --------------------------------------------------------

    def try_acquire(
        self,
        model_id: str,
        est_prompt_tokens: int,
        est_completion_tokens: int,
        *,
        priority: Priority = Priority.NORMAL,
    ) -> Ticket | Denial:
        provider = self.manifest.provider_of(model_id)
        now_m = self.clock.monotonic()
        now_w = self.clock.now_utc()

        reserve_tokens = int(
            (est_prompt_tokens + est_completion_tokens) * self.safety_factor
        )

        # 1. Paid providers need explicit authorisation before anything else.
        if provider.paid:
            if not self.allow_paid or self.max_usd <= 0:
                return Denial(
                    Admission.COST_BLOCKED,
                    model_id,
                    f"{provider.name} is a paid provider; needs --allow-paid "
                    "and a non-zero budget",
                )
            if self._spent_usd >= self.max_usd:
                return Denial(
                    Admission.COST_BLOCKED,
                    model_id,
                    f"run budget of ${self.max_usd} is spent",
                )

        # 2. Structurally impossible? Decide before any waiting logic, because
        #    no amount of waiting changes the answer.
        ceiling = self.manifest.max_request_tokens(model_id)
        if est_prompt_tokens + est_completion_tokens > ceiling:
            return Denial(
                Admission.UNSERVABLE,
                model_id,
                f"request of ~{est_prompt_tokens + est_completion_tokens} tokens "
                f"exceeds the {ceiling}-token ceiling for {model_id}; "
                "this can never be served at this size",
            )

        # 3. Per-run request cap (used to fence paid providers).
        per_run = provider.limit(LimitKind.REQUESTS_PER_RUN)
        if per_run and self._run_requests.get(provider.name, 0) >= per_run.value:
            return Denial(
                Admission.COST_BLOCKED,
                model_id,
                f"{provider.name} per-run request cap ({per_run.value}) reached",
            )

        # 4. Every applicable limit must have room simultaneously.
        waits: list[float] = []
        for state, prov in self._states_for(model_id):
            if state.cooldown_until and now_m < state.cooldown_until:
                waits.append(state.cooldown_until - now_m)
                continue

            rpd_spec = prov.limit(LimitKind.RPD)
            if rpd_spec and state.rpd is not None:
                budget = rpd_spec.value - (
                    rpd_spec.reserve if priority is Priority.NORMAL else 0
                )
                if state.rpd.current(now_w) >= budget:
                    if state.rpd.current(now_w) >= rpd_spec.value:
                        return Denial(
                            Admission.EXHAUSTED_TODAY,
                            model_id,
                            f"{prov.name} daily request cap ({rpd_spec.value}) reached",
                            retry_after_s=state.rpd.seconds_until_reset(now_w),
                        )
                    # Only the reserve remains: available to HIGH priority only.
                    return Denial(
                        Admission.WAIT,
                        model_id,
                        f"{prov.name} daily budget down to its reserve; "
                        "held back for critical-path retries",
                        retry_after_s=state.rpd.seconds_until_reset(now_w),
                    )

            tpd_spec = prov.limit(LimitKind.TPD)
            if tpd_spec and state.tpd is not None:
                if state.tpd.current(now_w) + reserve_tokens > tpd_spec.value:
                    return Denial(
                        Admission.EXHAUSTED_TODAY,
                        model_id,
                        f"{prov.name} daily token cap ({tpd_spec.value}) reached",
                        retry_after_s=state.tpd.seconds_until_reset(now_w),
                    )

            rpm_spec = prov.limit(LimitKind.RPM)
            if rpm_spec:
                wait = state.rpm.seconds_until_room(now_m, 1, rpm_spec.value)
                if wait is None:
                    return Denial(
                        Admission.UNSERVABLE, model_id, f"{prov.name} rpm limit is zero"
                    )
                if wait > 0:
                    waits.append(wait)

            tpm_spec = prov.limit(LimitKind.TPM)
            if tpm_spec:
                wait = state.tpm.seconds_until_room(
                    now_m, reserve_tokens, tpm_spec.value
                )
                if wait is None:
                    # Larger than the entire per-minute window: never fits.
                    return Denial(
                        Admission.UNSERVABLE,
                        model_id,
                        f"~{reserve_tokens} tokens exceeds {prov.name}'s "
                        f"{tpm_spec.value} tokens/minute window entirely",
                    )
                if wait > 0:
                    waits.append(wait)

        if waits:
            return Denial(
                Admission.WAIT,
                model_id,
                f"{model_id} rate limited",
                retry_after_s=max(waits),
            )

        # 5. Granted — reserve capacity now, before returning.
        ticket = Ticket(
            ticket_id=f"t{next(_ticket_ids)}",
            model_id=model_id,
            provider=provider.name,
            est_prompt_tokens=est_prompt_tokens,
            est_completion_tokens=est_completion_tokens,
            reserved_tokens=reserve_tokens,
            acquired_monotonic=now_m,
            priority=priority,
        )
        for state, _ in self._states_for(model_id):
            state.rpm.add(now_m, 1)
            state.tpm.add(now_m, reserve_tokens)
            if state.rpd is not None:
                state.rpd.add(now_w, 1)
            if state.tpd is not None:
                state.tpd.add(now_w, reserve_tokens)

        self._run_requests[provider.name] = self._run_requests.get(provider.name, 0) + 1
        self._tickets[ticket.ticket_id] = ticket
        return ticket

    async def acquire(
        self,
        model_id: str,
        est_prompt_tokens: int,
        est_completion_tokens: int,
        *,
        priority: Priority = Priority.NORMAL,
        deadline_s: float | None = None,
    ) -> Ticket:
        """Block until granted, or raise.

        Only `WAIT` is worth waiting on. The other refusals are terminal for
        this attempt and are raised immediately so the caller can fail over to
        a different vendor instead of sleeping pointlessly.
        """
        start = self.clock.monotonic()
        while True:
            result = self.try_acquire(
                model_id, est_prompt_tokens, est_completion_tokens, priority=priority
            )
            if isinstance(result, Ticket):
                return result

            if result.verdict is Admission.UNSERVABLE:
                raise Unservable(result.reason)
            if result.verdict is Admission.EXHAUSTED_TODAY:
                raise QuotaExhausted(result.reason)
            if result.verdict is Admission.COST_BLOCKED:
                raise CostBlocked(result.reason)

            wait = result.retry_after_s or 1.0
            elapsed = self.clock.monotonic() - start
            if deadline_s is not None and elapsed + wait > deadline_s:
                raise QuotaExhausted(
                    f"{model_id} would need {wait:.1f}s, past the "
                    f"{deadline_s:.1f}s deadline: {result.reason}"
                )
            await asyncio.sleep(wait)

    # -- settlement -------------------------------------------------------

    def commit(
        self, ticket: Ticket, usage: Usage, cost: Decimal = Decimal("0")
    ) -> None:
        """Replace the reservation with what was actually consumed."""
        actual = usage.total_tokens
        delta = actual - ticket.reserved_tokens
        for state, _ in self._states_for(ticket.model_id):
            state.tpm.adjust_last(delta)
            if state.tpd is not None:
                if delta >= 0:
                    state.tpd.add(self.clock.now_utc(), delta)
                else:
                    state.tpd.remove(-delta)
        self._spent_usd += cost
        self._tickets.pop(ticket.ticket_id, None)

    def release(self, ticket: Ticket, reason: str = "") -> None:
        """Refund a reservation for a request that never reached the provider.

        Without this, a transport failure would permanently consume quota that
        was never actually spent.
        """
        for state, _ in self._states_for(ticket.model_id):
            state.rpm.remove(1)
            state.tpm.remove(ticket.reserved_tokens)
            if state.rpd is not None:
                state.rpd.remove(1)
            if state.tpd is not None:
                state.tpd.remove(ticket.reserved_tokens)
        provider = self.manifest.provider_of(ticket.model_id).name
        self._run_requests[provider] = max(0, self._run_requests.get(provider, 1) - 1)
        self._tickets.pop(ticket.ticket_id, None)

    # -- server truth -----------------------------------------------------

    def sync_from_headers(self, model_id: str, snap: RateLimitSnapshot) -> None:
        """Overwrite local counters with the provider's own numbers.

        Local counting is inference; these headers are fact, and they cost
        nothing to read. This is also how providers with undocumented limits get
        characterised over time.
        """
        now_w = self.clock.now_utc()
        now_m = self.clock.monotonic()

        for state, prov in self._states_for(model_id):
            rpd_spec = prov.limit(LimitKind.RPD)
            if snap.remaining_requests is not None and rpd_spec and state.rpd:
                used = max(0, rpd_spec.value - snap.remaining_requests)
                state.rpd.set_count(now_w, used)

            if snap.retry_after_s:
                state.cooldown_until = now_m + snap.retry_after_s

            if snap.daily_limit_hit and state.rpd and rpd_spec:
                # Believe the server: stop trying until the next local midnight.
                state.rpd.set_count(now_w, rpd_spec.value)
                state.exhausted_day = state.rpd.day

    def note_daily_exhausted(self, model_id: str) -> None:
        """Mark a model out of daily quota after a 429 identifying a daily cap."""
        self.sync_from_headers(model_id, RateLimitSnapshot(daily_limit_hit=True))

    # -- reporting --------------------------------------------------------

    def wait_time(self, model_id: str, est_tokens: int) -> float | None:
        """Seconds until this request could proceed.

        `None` means never today — either structurally impossible or out of
        daily quota. Callers must treat `None` as "pick another model", not as
        "wait indefinitely".
        """
        result = self.try_acquire(model_id, est_tokens, 0)
        if isinstance(result, Ticket):
            self.release(result)
            return 0.0
        if result.is_permanent_today:
            return None
        return result.retry_after_s

    def headroom(self) -> dict[str, Headroom]:
        now_w = self.clock.now_utc()
        now_m = self.clock.monotonic()
        out: dict[str, Headroom] = {}

        for model in self.manifest.enabled_models:
            provider = self.manifest.providers[model.provider]
            rpd_spec = provider.limit(LimitKind.RPD)
            tpm_spec = provider.limit(LimitKind.TPM)

            requests_used = 0
            seconds_to_reset = None
            tokens_minute = 0
            for state, _ in self._states_for(model.id):
                if state.rpd is not None:
                    requests_used = max(requests_used, state.rpd.current(now_w))
                    seconds_to_reset = state.rpd.seconds_until_reset(now_w)
                tokens_minute = max(tokens_minute, state.tpm.current(now_m))

            out[model.id] = Headroom(
                model_id=model.id,
                provider=model.provider,
                requests_used=requests_used,
                requests_limit=rpd_spec.value if rpd_spec else None,
                tokens_used_minute=tokens_minute,
                tokens_limit_minute=tpm_spec.value if tpm_spec else None,
                seconds_to_reset=seconds_to_reset,
                reset_tz=provider.reset_tz,
            )
        return out

    @property
    def spent_usd(self) -> Decimal:
        return self._spent_usd
