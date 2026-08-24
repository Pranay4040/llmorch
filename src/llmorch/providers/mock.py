"""Deterministic fake provider.

Milestone 1's only provider, and the primary development loop thereafter. Two
jobs:

1. **Return canned-but-valid artifacts**, so the whole pipeline can be exercised
   end to end and the materialised folder actually runs.
2. **Inject faults on demand**, so failover and verification are testable. This
   is not a nicety: failover logic that has never been exercised is failover
   logic that does not work, and it cannot be rehearsed against live providers
   when one of them allows 250 requests a day.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

from ..errors import RateLimited, SchemaInvalid, TransportError
from ..types import ChatRequest, ChatResponse, Usage


class FaultMode(str, Enum):
    """Failure shapes observed from real free-tier models."""

    NONE = "none"
    TRANSPORT = "transport"
    RATE_LIMIT = "rate_limit"
    DAILY_LIMIT = "daily_limit"
    MALFORMED_JSON = "malformed_json"
    TRUNCATED = "truncated"
    UNPARSEABLE_CODE = "unparseable_code"
    EMPTY = "empty"
    PLACEHOLDER = "placeholder"
    """Returns TODO stubs instead of an implementation — passes a syntax check
    but fails review, which is exactly what Tier 1 exists to catch."""


@dataclass(slots=True)
class MockProvider:
    """Fake model endpoint.

    Responses are keyed by node id so a run is reproducible; faults are keyed by
    (node_id, model_id) so a specific model can be made to fail on a specific
    node — the setup a cross-vendor failover test needs.
    """

    name: str = "mock"
    responses: dict[str, str] = field(default_factory=dict)
    """node_id -> artifact text."""
    faults: dict[str, FaultMode] = field(default_factory=dict)
    """"node_id" or "node_id@model_id" -> fault to raise."""
    fail_times: dict[str, int] = field(default_factory=dict)
    """How many times a fault key should fire before succeeding. Absent = always."""
    default_response: str = "mock artifact"
    latency_ms: int = 5

    _calls: list[tuple[str, str]] = field(default_factory=list, init=False)
    _fault_counts: dict[str, int] = field(default_factory=dict, init=False)

    # -- inspection -------------------------------------------------------

    @property
    def calls(self) -> list[tuple[str, str]]:
        """(node_id, model_id) for every call, in order."""
        return list(self._calls)

    def calls_for(self, node_id: str) -> list[str]:
        return [m for n, m in self._calls if n == node_id]

    def reset(self) -> None:
        self._calls.clear()
        self._fault_counts.clear()

    # -- fault selection --------------------------------------------------

    def _fault_for(self, node_id: str, model_id: str) -> tuple[str, FaultMode] | None:
        """Most specific match wins: a per-model fault beats a per-node one."""
        for key in (f"{node_id}@{model_id}", node_id):
            mode = self.faults.get(key)
            if mode and mode is not FaultMode.NONE:
                budget = self.fail_times.get(key)
                fired = self._fault_counts.get(key, 0)
                if budget is None or fired < budget:
                    return key, mode
        return None

    # -- Provider protocol ------------------------------------------------

    async def chat(self, request: ChatRequest) -> ChatResponse:
        node_id = _node_id_of(request)
        self._calls.append((node_id, request.model_id))

        hit = self._fault_for(node_id, request.model_id)
        if hit is not None:
            key, mode = hit
            self._fault_counts[key] = self._fault_counts.get(key, 0) + 1
            return self._raise_or_return(mode, request)

        text = self.responses.get(node_id, self.default_response)
        return self._respond(request, text)

    async def count_tokens(self, request: ChatRequest) -> int | None:
        return None

    # -- fault behaviours -------------------------------------------------

    def _raise_or_return(self, mode: FaultMode, request: ChatRequest) -> ChatResponse:
        match mode:
            case FaultMode.TRANSPORT:
                raise TransportError("mock: connection reset", status=503)
            case FaultMode.RATE_LIMIT:
                raise RateLimited("mock: rate limited", retry_after_s=2.0)
            case FaultMode.DAILY_LIMIT:
                raise RateLimited("mock: daily cap reached", daily=True)
            case FaultMode.MALFORMED_JSON:
                raise SchemaInvalid("mock: response was not valid JSON")
            case FaultMode.EMPTY:
                return self._respond(request, "")
            case FaultMode.TRUNCATED:
                # Stops exactly at max_tokens — the authoritative truncation
                # signal, and among the commonest free-model failures.
                return self._respond(
                    request,
                    "def handler(request):\n    data = {'notes': [",
                    truncated=True,
                )
            case FaultMode.UNPARSEABLE_CODE:
                return self._respond(request, "def broken(:\n    this is not python\n")
            case FaultMode.PLACEHOLDER:
                return self._respond(
                    request, "# TODO: implement this\npass\n"
                )
            case _:
                return self._respond(request, self.default_response)

    def _respond(
        self, request: ChatRequest, text: str, *, truncated: bool = False
    ) -> ChatResponse:
        prompt_tokens = sum(len(m.content) for m in request.messages) // 4
        prompt_tokens += len(request.system or "") // 4
        completion = min(len(text) // 4 + 1, request.max_tokens)
        return ChatResponse(
            text=text,
            usage=Usage(
                prompt_tokens=max(1, prompt_tokens), completion_tokens=completion
            ),
            model_reported=request.model_id,
            latency_ms=self.latency_ms,
            truncated=truncated or completion >= request.max_tokens,
        )


def _node_id_of(request: ChatRequest) -> str:
    """Recover the node id the engine embedded in the prompt.

    The worker tags each prompt with a marker so canned responses can be keyed
    by node without threading extra state through the Provider protocol.
    """
    for message in request.messages:
        for line in message.content.splitlines():
            if line.startswith("[node:") and line.endswith("]"):
                return line[len("[node:") : -1]
    # Stable fallback so an untagged request is still deterministic.
    blob = "".join(m.content for m in request.messages)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]
