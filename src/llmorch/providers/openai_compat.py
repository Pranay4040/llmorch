"""HTTP adapter for OpenAI-compatible chat endpoints.

Every provider in the roster speaks this wire format — Groq natively, Gemini
through its compatibility endpoint — so one adapter covers all of them and a
new vendor is a manifest entry rather than a module.

**Why stdlib rather than the `openai` SDK.** The SDK retries on its own. A retry
it performs is a request the governor never reserved and never counted, which
breaks the one invariant admission control rests on: that every call to a
provider passed through a ticket. Its backoff would also race the failover
ladder in `engine/health.py`, re-trying the same model while this package was
deliberately moving to a different vendor. Raw `urllib` keeps retry policy in
exactly one place, keeps response headers reachable — the SDK buries them —
and costs no dependency.

Transport is injectable, so the whole adapter is testable offline against
recorded responses. Nothing here touches the network unless a real transport is
supplied, which is what lets M2 be built and verified before a key exists.

Secret hygiene: the API key is read once into an Authorization header and never
appears in a log line, an exception message, or the ledger.
"""

from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from ..errors import (
    ProviderError,
    RateLimited,
    SchemaInvalid,
    TransportError,
)
from ..types import ChatRequest, ChatResponse, Message, Usage
from .headers import parse_rate_limit

USER_AGENT = "llmorch/0.1"

# Bodies longer than this are truncated before going into an exception message.
# Provider errors are occasionally enormous, and an error is meant to be read.
MAX_ERROR_BODY = 500


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HttpRequest:
    url: str
    headers: Mapping[str, str]
    body: bytes
    timeout_s: float


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: str


Transport = Callable[[HttpRequest], HttpResponse]
"""Synchronous send. Runs on a worker thread so the event loop stays free."""


def urllib_transport(request: HttpRequest) -> HttpResponse:
    """The real one. Blocking — always called through `asyncio.to_thread`.

    An HTTP error status is a response, not an exception: a 429 carries the
    rate-limit headers that are the most valuable thing in the exchange, so it
    is read and returned rather than raised past.
    """
    req = urllib.request.Request(
        request.url, data=request.body, headers=dict(request.headers), method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=request.timeout_s) as response:
            return HttpResponse(
                status=response.status,
                headers=dict(response.headers.items()),
                body=response.read().decode("utf-8", errors="replace"),
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return HttpResponse(
            status=exc.code, headers=dict(exc.headers.items()), body=body
        )
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
        # Never reached the provider. Retryable, and the reservation gets
        # refunded by the caller.
        raise TransportError(f"could not reach {_host_of(request.url)}: {exc}") from exc


def _host_of(url: str) -> str:
    """Host portion, for error messages. Keeps any query string out of a log."""
    without_scheme = url.split("://", 1)[-1]
    return without_scheme.split("/", 1)[0]


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------


@dataclass(slots=True)
class OpenAICompatProvider:
    """One vendor's chat endpoint.

    `wire_names` maps this package's model ids onto whatever the vendor calls
    them. The indirection is deliberate: vendors rename models, and the id is
    what `profiles.json` keys a track record by, so it has to stay stable.
    """

    name: str
    base_url: str
    api_key: str
    wire_names: Mapping[str, str] = field(default_factory=dict)
    transport: Transport = urllib_transport
    max_tokens_field: str = "max_tokens"
    """Some endpoints have renamed this to `max_completion_tokens`. Which one a
    provider wants is a wire detail, not a behaviour change."""
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    # -- Provider protocol ------------------------------------------------

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = self._payload(request)
        http = HttpRequest(
            url=self._endpoint(),
            headers=self._headers(),
            body=json.dumps(payload).encode("utf-8"),
            timeout_s=request.timeout_s,
        )

        started = perf_counter()
        response = await asyncio.to_thread(self.transport, http)
        latency_ms = int((perf_counter() - started) * 1000)

        snapshot = parse_rate_limit(
            response.headers, status=response.status, body=response.body
        )

        if response.status != 200:
            self._raise_for_status(response, snapshot)

        return self._parse_success(request, response, snapshot, latency_ms)

    async def count_tokens(self, request: ChatRequest) -> int | None:
        """Neither Groq nor Gemini's compatibility endpoint offers a free
        counting call, so the estimator's heuristic stands."""
        return None

    # -- request construction ---------------------------------------------

    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            **dict(self.extra_headers),
        }

    def wire_name(self, model_id: str) -> str:
        return self.wire_names.get(model_id, model_id)

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        """Build the JSON body.

        `max_tokens` is always present. It is what turns the completion half of
        the token estimate into a hard bound, and admission control is only
        sound because that bound exists.
        """
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages += [{"role": m.role, "content": m.content} for m in request.messages]

        payload: dict[str, Any] = {
            "model": self.wire_name(request.model_id),
            "messages": messages,
            "temperature": request.temperature,
            self.max_tokens_field: request.max_tokens,
            "stream": False,
        }

        if request.json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": request.json_schema,
            }
        return payload

    # -- response handling ------------------------------------------------

    def _raise_for_status(self, response: HttpResponse, snapshot) -> None:
        """Map an HTTP status onto the error hierarchy.

        The split that matters is `is_retryable`: it is what the failover ladder
        branches on. A 400 means the request itself is wrong and will be wrong
        again, so retrying it anywhere is wasted quota.
        """
        detail = _error_message(response.body)
        status = response.status

        if status == 429:
            raise RateLimited(
                f"{self.name}: {detail}",
                retry_after_s=snapshot.retry_after_s,
                daily=snapshot.daily_limit_hit,
            )

        if status in (401, 403):
            # Never echo the key — only the fact that it was rejected.
            raise ProviderError(
                f"{self.name} rejected the API key ({status}): {detail}", status=status
            )

        if status == 404:
            raise ProviderError(
                f"{self.name} has no such model ({status}): {detail}. "
                "Check wire_name in models.yaml against the provider's model list.",
                status=status,
            )

        if 400 <= status < 500:
            raise ProviderError(f"{self.name} rejected the request ({status}): {detail}", status=status)

        # 5xx: the provider's problem, and usually transient.
        raise TransportError(f"{self.name} returned {status}: {detail}", status=status)

    def _parse_success(
        self, request: ChatRequest, response: HttpResponse, snapshot, latency_ms: int
    ) -> ChatResponse:
        try:
            data = json.loads(response.body)
        except json.JSONDecodeError as exc:
            raise SchemaInvalid(
                f"{self.name} returned a 200 that was not JSON: {exc}"
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            # A 200 with no choices is a malformed response, not an empty one.
            raise SchemaInvalid(f"{self.name} returned a 200 with no choices")

        choice = choices[0] or {}
        message = choice.get("message") or {}
        text = message.get("content") or ""
        finish_reason = choice.get("finish_reason") or ""

        return ChatResponse(
            text=text,
            usage=_usage_of(data.get("usage")),
            model_reported=str(data.get("model") or request.model_id),
            latency_ms=latency_ms,
            raw_status=response.status,
            rate_limit=snapshot,
            # `length` is the provider stating outright that it stopped at the
            # cap. Cheaper and more reliable than inspecting the text.
            truncated=finish_reason == "length",
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _usage_of(raw: Any) -> Usage:
    """Read the usage block, tolerating the fields a provider omits.

    Reasoning tokens are billed against the token budget but excluded from
    `completion_tokens` by every provider that emits them, so they have to be
    added explicitly or the governor under-counts exactly the models that spend
    the most.
    """
    if not isinstance(raw, dict):
        return Usage()

    completion_details = raw.get("completion_tokens_details") or {}
    prompt_details = raw.get("prompt_tokens_details") or {}

    return Usage(
        prompt_tokens=_non_negative_int(raw.get("prompt_tokens")),
        completion_tokens=_non_negative_int(raw.get("completion_tokens")),
        reasoning_tokens=_non_negative_int(completion_details.get("reasoning_tokens")),
        cached_tokens=_non_negative_int(prompt_details.get("cached_tokens")),
    )


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _error_message(body: str) -> str:
    """Pull the human-readable part out of an error body.

    Providers wrap the message differently and sometimes return HTML from a
    load balancer rather than JSON, so this degrades to a trimmed raw body
    rather than failing while trying to report a failure.
    """
    text = (body or "").strip()
    if not text:
        return "no response body"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text[:MAX_ERROR_BODY]

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)[:MAX_ERROR_BODY]
        if isinstance(error, str):
            return error[:MAX_ERROR_BODY]
        if "message" in data:
            return str(data["message"])[:MAX_ERROR_BODY]
    return text[:MAX_ERROR_BODY]


def build_provider(
    provider_spec,
    models,
    api_key: str,
    *,
    transport: Transport = urllib_transport,
) -> OpenAICompatProvider:
    """Construct an adapter from the manifest entries for one provider."""
    return OpenAICompatProvider(
        name=provider_spec.name,
        base_url=provider_spec.base_url,
        api_key=api_key,
        wire_names={m.id: m.wire_name for m in models if m.provider == provider_spec.name},
        transport=transport,
    )
