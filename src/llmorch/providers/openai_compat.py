"""HTTP adapter for OpenAI-shaped chat endpoints.

Every provider in the roster speaks this wire format — Groq natively, Gemini
through its `/v1beta/openai/` compatibility endpoint — so one adapter covers the
whole roster and adding a vendor stays a manifest entry rather than a rewrite.

**No HTTP dependency.** The client is stdlib `urllib` executed on a worker
thread. That is not asceticism: the vendor SDKs each ship their own retry and
rate-limit machinery, which would sit *underneath* the governor and silently
retry requests the governor never admitted. Admission control only works if
every call goes through it, so the transport stays thin and dumb.

Transport is injectable for the same reason the clock is: the failure paths
(a 429 whose body names a daily cap, a 404 on an unverified wire name, a
completion cut off at max_tokens) have to be exercisable without a network and
without spending a quota day.

Error mapping follows `errors.is_retryable`, which is what the failover ladder
branches on:

    timeout / connection reset / 5xx -> TransportError   (retry same model)
    429                              -> RateLimited      (resync, then re-select)
    400 / 401 / 403 / 404            -> ProviderError    (fail over; config is wrong)
    unparseable body                 -> SchemaInvalid    (salvage, then re-select)
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import get_api_key, has_api_key
from ..errors import (
    ConfigError,
    ProviderError,
    RateLimited,
    SchemaInvalid,
    TransportError,
)
from ..registry.manifest import Manifest
from ..types import ChatRequest, ChatResponse, Usage
from .base import ProviderRegistry
from .headers import looks_like_daily_limit, parse_rate_limit_headers

USER_AGENT = "llmorch/0.1"


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: str


class Transport(Protocol):
    """Minimal POST-and-keep-the-headers interface."""

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_s: float,
    ) -> HttpResponse:
        ...


class UrllibTransport:
    """Default transport: `urllib` on a worker thread.

    HTTP error statuses are returned rather than raised, because a 429's
    *headers* are the most valuable thing a provider ever sends and discarding
    them would leave the governor guessing at exactly the moment it must not.
    """

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_s: float,
    ) -> HttpResponse:
        return await asyncio.to_thread(self._post, url, dict(headers), body, timeout_s)

    @staticmethod
    def _post(
        url: str, headers: dict[str, str], body: bytes, timeout_s: float
    ) -> HttpResponse:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return HttpResponse(
                    status=response.status,
                    headers=dict(response.headers.items()),
                    body=response.read().decode("utf-8", errors="replace"),
                )
        except urllib.error.HTTPError as exc:
            return HttpResponse(
                status=exc.code,
                headers=dict(exc.headers.items() if exc.headers else {}),
                body=exc.read().decode("utf-8", errors="replace"),
            )
        except urllib.error.URLError as exc:
            raise TransportError(f"{url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise TransportError(f"{url}: timed out after {timeout_s:.0f}s") from exc


@dataclass(slots=True)
class OpenAICompatProvider:
    """One provider endpoint, serving every model the manifest maps to it."""

    name: str
    base_url: str
    api_key: str
    wire_names: dict[str, str] = field(default_factory=dict)
    """model id -> the string this provider expects on the wire."""
    transport: Transport = field(default_factory=UrllibTransport)
    extra_headers: dict[str, str] = field(default_factory=dict)
    supports_json_schema: bool = True

    # -- Provider protocol ------------------------------------------------

    async def chat(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        response = await self.transport.post(
            self.endpoint("chat/completions"),
            headers=self._headers(),
            body=json.dumps(self._payload(request)).encode("utf-8"),
            timeout_s=request.timeout_s,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        if response.status != 200:
            self._raise_for_status(response)
        return self._parse(response, request, latency_ms)

    async def count_tokens(self, request: ChatRequest) -> int | None:
        """No free exact count on these endpoints — the estimator's job stands."""
        return None

    # -- request ----------------------------------------------------------

    def endpoint(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            **self.extra_headers,
        }

    def wire_name(self, model_id: str) -> str:
        return self.wire_names.get(model_id, model_id)

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(
            {"role": m.role, "content": m.content} for m in request.messages
        )

        payload: dict[str, Any] = {
            "model": self.wire_name(request.model_id),
            "messages": messages,
            # Never omitted. This is what turns the completion side of the
            # estimate into a hard bound, and a hard bound is what makes
            # admission control sound rather than hopeful.
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False,
        }
        if request.json_schema and self.supports_json_schema:
            payload["response_format"] = _response_format(request.json_schema)
        return payload

    # -- response ---------------------------------------------------------

    def _raise_for_status(self, response: HttpResponse) -> None:
        snapshot = parse_rate_limit_headers(response.headers, body=response.body)
        detail = _error_message(
            response.body,
            limit=RATE_LIMIT_DETAIL_CHARS
            if response.status == 429
            else ERROR_DETAIL_CHARS,
        )

        if response.status == 429:
            raise RateLimited(
                f"{self.name}: {detail}",
                retry_after_s=snapshot.retry_after_s or snapshot.reset_requests_s,
                daily=snapshot.daily_limit_hit or looks_like_daily_limit(detail),
            )
        if response.status >= 500:
            raise TransportError(f"{self.name}: {detail}", status=response.status)
        if response.status in (401, 403):
            # Not a competence failure and not worth retrying: the key is wrong
            # or unauthorised for this model. The key itself is never echoed.
            raise ProviderError(
                f"{self.name}: rejected the API key ({response.status}). {detail}",
                status=response.status,
            )
        raise ProviderError(f"{self.name}: {detail}", status=response.status)

    def _parse(
        self, response: HttpResponse, request: ChatRequest, latency_ms: int
    ) -> ChatResponse:
        try:
            data = json.loads(response.body)
        except (json.JSONDecodeError, TypeError) as exc:
            raise SchemaInvalid(f"{self.name}: response body was not JSON") from exc

        choices = data.get("choices") or []
        if not choices:
            raise SchemaInvalid(f"{self.name}: response contained no choices")

        choice = choices[0]
        text = (choice.get("message") or {}).get("content") or ""
        finish = str(choice.get("finish_reason") or "")

        raw_usage = data.get("usage") or {}
        completion_details = raw_usage.get("completion_tokens_details") or {}
        prompt_details = raw_usage.get("prompt_tokens_details") or {}
        usage = Usage(
            prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
            completion_tokens=int(raw_usage.get("completion_tokens") or 0),
            reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
            cached_tokens=int(prompt_details.get("cached_tokens") or 0),
        )

        return ChatResponse(
            text=text,
            usage=usage,
            model_reported=str(data.get("model") or request.model_id),
            latency_ms=latency_ms,
            raw_status=response.status,
            rate_limit=parse_rate_limit_headers(response.headers),
            # `length` is the provider stating outright that it stopped at the
            # cap rather than finishing. Free to detect, and among the commonest
            # free-model failures.
            truncated=finish == "length",
        )


def _response_format(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a JSON Schema the way the API expects it.

    The wire format is `{"type": "json_schema", "json_schema": {"name": ...,
    "schema": ...}}` — the schema goes *inside* a named envelope. Passing the
    bare schema instead puts `type` and `properties` where the server looks for
    `name` and `schema`, and it answers 400. A caller that already built the
    envelope is passed through untouched.
    """
    if "schema" in schema and "name" in schema:
        return {"type": "json_schema", "json_schema": schema}
    return {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema},
    }


# A 429 body carries the machine-readable quota metric and value — the only
# place a provider ever states its real limit. Truncating that to the same
# length as an ordinary error throws away the one diagnostic worth having.
RATE_LIMIT_DETAIL_CHARS = 2000
ERROR_DETAIL_CHARS = 400


def _error_message(body: str | None, *, limit: int = ERROR_DETAIL_CHARS) -> str:
    """Pull the human-readable part out of an error body.

    Providers nest it differently (`error.message`, `message`, `detail`), and an
    edge proxy may return HTML instead of JSON, so the raw body is the last
    resort — trimmed, because a proxy error page is not worth a screenful.
    """
    if not body:
        return "no response body"
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return body.strip()[:limit]

    # Gemini wraps its error object in a single-element array; OpenAI-shaped
    # providers do not. Unwrap before looking for the message, or the whole
    # thing degrades to a stringified list and the quota detail is lost.
    if isinstance(data, list) and data:
        data = data[0]

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)[:limit]
        if isinstance(error, str):
            return error[:limit]
        for key in ("message", "detail"):
            if key in data:
                return str(data[key])[:limit]
    return str(data)[:limit]


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def build_live_registry(
    manifest: Manifest,
    *,
    transport: Transport | None = None,
    only_providers: set[str] | None = None,
) -> tuple[ProviderRegistry, dict[str, OpenAICompatProvider]]:
    """Build real endpoints for every enabled provider that has a key.

    A keyless provider is skipped rather than fatal: the roster is deliberately
    partial during rollout — Milestone 2 runs Groq alone — and the rest of the
    manifest stays declared but dormant.
    """
    registry = ProviderRegistry()
    clients: dict[str, OpenAICompatProvider] = {}

    for model in manifest.enabled_models:
        spec = manifest.providers[model.provider]
        if only_providers is not None and model.provider not in only_providers:
            continue
        if not has_api_key(spec.api_key_env):
            continue

        client = clients.get(model.provider)
        if client is None:
            client = OpenAICompatProvider(
                name=spec.name or model.provider,
                base_url=spec.base_url,
                api_key=get_api_key(spec.api_key_env, provider=model.provider),
                transport=transport or UrllibTransport(),
            )
            clients[model.provider] = client

        client.wire_names[model.id] = model.wire_name
        client.supports_json_schema = (
            client.supports_json_schema and model.supports_json_schema
        )
        registry.register(model.id, client)

    if not registry.model_ids:
        wanted = sorted(only_providers) if only_providers else "any enabled provider"
        raise ConfigError(
            f"no usable provider for {wanted}: every candidate is either disabled "
            "in models.yaml or missing its API key in .env"
        )
    return registry, clients
