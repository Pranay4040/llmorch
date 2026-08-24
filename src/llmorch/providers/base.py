"""Provider interface.

Every provider in the roster except Gemini speaks the OpenAI wire format, and
Gemini offers a compatible endpoint, so a single adapter covers all of them.
This protocol is what the engine sees; concrete adapters arrive at Milestone 2.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import ChatRequest, ChatResponse


@runtime_checkable
class Provider(Protocol):
    """A callable model endpoint.

    Implementations raise from `llmorch.errors` rather than returning error
    sentinels, so the failover ladder can branch on `is_retryable`.
    """

    name: str

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Issue one completion.

        Raises TransportError, RateLimited, SchemaInvalid, or Truncated.
        Never returns a partial or error-shaped response.
        """
        ...

    async def count_tokens(self, request: ChatRequest) -> int | None:
        """Exact prompt token count, when the provider offers one for free.

        None means unsupported, in which case the estimator's heuristic stands.
        """
        ...


class ProviderRegistry:
    """Maps model ids to the provider that serves them."""

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}

    def register(self, model_id: str, provider: Provider) -> None:
        self._providers[model_id] = provider

    def get(self, model_id: str) -> Provider:
        try:
            return self._providers[model_id]
        except KeyError:
            raise KeyError(f"no provider registered for {model_id!r}") from None

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._providers

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(self._providers)
