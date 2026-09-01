"""Quota governance for applications that call several LLM providers.

The problem this solves is not cost. It is that free and low tier APIs are
rationed along several axes at once — requests per minute, tokens per minute,
requests per day — with different numbers per provider, days that end at
different midnights, and limits that are sometimes shared across every model on
an account. Getting that arithmetic wrong does not produce a slightly worse
result; it produces a 429 in the middle of work you have already partly paid
for.

Four pieces, usable independently of the orchestrator around them:

`Governor`
    Admission control. Nothing calls a provider without a ticket. It answers
    one question — may this request proceed right now? — and the value is in
    how precisely it separates the ways the answer can be no:

        GRANTED          go
        WAIT             not now, but soon; retry_after_s says when
        EXHAUSTED_TODAY  not until the provider's next local midnight
        UNSERVABLE       never, at this size, on this model
        COST_BLOCKED     would spend real money without authorisation

    `UNSERVABLE` versus `WAIT` is the distinction that matters most: a request
    larger than a provider's per-minute ceiling will not fit however long you
    wait, so treating it as a wait condition hangs the caller forever.

`LedgerStore`
    An append-only SQLite record of every call, stamped with the day key of the
    *provider's own* timezone. `restore_governor` replays it at startup, which
    is what stops a fresh process believing it holds the whole daily allowance.

`TokenEstimator`
    Character-based estimation with a per-provider correction learned from
    actual usage. No tokenizer dependency: every provider here uses a different
    one, so shipping a single tokenizer would be precise for one and misleading
    for the rest.

`quota_manifest`
    Declares providers and models in keyword arguments rather than this
    project's YAML, so using the governor does not mean adopting its config.

A worked example lives in the README, and a test runs that example verbatim so
it cannot drift from the code.
"""

from __future__ import annotations

from ..types import Admission, Denial, Headroom, Priority, RateLimitSnapshot, Ticket, Usage
from .estimator import TokenEstimator
from .governor import Governor
from .spec import model_spec, provider_spec, quota_manifest
from .store import LedgerStore, cost_of, make_event, restore_governor
from .windows import Clock, DayCounter, FakeClock, SlidingWindow, SystemClock, day_key

__all__ = [
    # Admission control
    "Governor",
    "Ticket",
    "Denial",
    "Admission",
    "Priority",
    "Headroom",
    # Declaring a roster
    "quota_manifest",
    "provider_spec",
    "model_spec",
    # The persistent record
    "LedgerStore",
    "restore_governor",
    "make_event",
    "cost_of",
    "Usage",
    # Estimation and time
    "TokenEstimator",
    "Clock",
    "SystemClock",
    "FakeClock",
    "SlidingWindow",
    "DayCounter",
    "day_key",
    "RateLimitSnapshot",
]
