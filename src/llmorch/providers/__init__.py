"""Talking to OpenAI-shaped endpoints, and reading what they say back.

`OpenAICompatProvider` is a deliberately thin client: one POST, no retry policy,
no rate-limit handling of its own. That thinness is the point. Every vendor SDK
ships its own retry and backoff machinery, and any of it would sit *underneath*
the governor, quietly re-sending requests admission control never granted —
which would make the whole quota story a fiction.

`parse_rate_limit_headers` is the other half. Local counting is an estimate of
what a provider will allow; its headers are fact, and free with every response.
Where a provider sends none, the 429 body is read instead — including prose like
"Please retry in 12.398389909s", which is the only number Gemini ever gives.
"""

from __future__ import annotations

from .base import Provider, ProviderRegistry
from .headers import (
    looks_like_daily_limit,
    parse_duration,
    parse_rate_limit_headers,
    retry_after_from_body,
)
from .mock import FaultMode, MockProvider
from .openai_compat import (
    HttpResponse,
    OpenAICompatProvider,
    Transport,
    UrllibTransport,
    build_live_registry,
)

__all__ = [
    "Provider",
    "ProviderRegistry",
    "OpenAICompatProvider",
    "build_live_registry",
    # Injectable transport: the failure paths — a 429 naming a daily cap, a 404
    # on an unverified wire name — have to be testable without a network.
    "Transport",
    "UrllibTransport",
    "HttpResponse",
    # Deterministic fake, with the failure shapes real free models produce.
    "MockProvider",
    "FaultMode",
    # Header and body parsing.
    "parse_rate_limit_headers",
    "retry_after_from_body",
    "looks_like_daily_limit",
    "parse_duration",
]
