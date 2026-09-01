"""Declare providers and models without a YAML file.

The orchestrator loads its roster from `models.yaml`, which is right for an
application and wrong for a library: importing this package to govern quota in
*your* program should not require adopting this project's config format, its
role taxonomy, or its fallback chains.

So this builds the same validated structure the rest of the code expects, from
ordinary keyword arguments, with the orchestration-only parts left empty.

    from llmorch.quota import Governor, quota_manifest

    manifest = quota_manifest(
        providers=[
            provider_spec("groq", base_url="https://api.groq.com/openai/v1",
                          api_key_env="GROQ_API_KEY", rpm=30, tpm=8000, rpd=1000),
        ],
        models=[
            model_spec("groq/gpt-oss-120b", provider="groq",
                       wire_name="openai/gpt-oss-120b",
                       context=131072, max_output=4096),
        ],
    )

The limits are named for what they are — requests per minute, tokens per
minute, requests per day — rather than as a list of typed records, because that
is how every provider's documentation states them and translating by hand is
where the mistakes come from.
"""

from __future__ import annotations

from typing import Any

from ..registry.manifest import Manifest
from ..types import LimitKind, LimitScope


def provider_spec(
    name: str,
    *,
    base_url: str,
    api_key_env: str,
    reset_tz: str = "UTC",
    rpm: int | None = None,
    tpm: int | None = None,
    rpd: int | None = None,
    tpd: int | None = None,
    requests_per_run: int | None = None,
    reserve_requests: int = 0,
    account_scoped: tuple[str, ...] = ("rpm",),
    paid: bool = False,
    estimated: bool = False,
    enabled: bool = True,
    cost: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One provider's limits, written the way its documentation states them.

    `account_scoped` names the limits shared across every model on the provider
    rather than counted per model. Getting this wrong is the classic error in
    this domain: if a limit is account-scoped, falling back to a *different
    model on the same provider* buys no additional quota at all.

    `reserve_requests` holds part of the daily allowance back from normal-
    priority work, so a wide fan-out cannot consume the last few requests that
    a critical retry will need.

    `estimated=True` marks limits you are guessing at — which is the honest
    setting for any provider that does not publish them.
    """
    limits: list[dict[str, Any]] = []
    for kind, value in (
        (LimitKind.RPM, rpm),
        (LimitKind.TPM, tpm),
        (LimitKind.RPD, rpd),
        (LimitKind.TPD, tpd),
        (LimitKind.REQUESTS_PER_RUN, requests_per_run),
    ):
        if not value:
            continue
        entry: dict[str, Any] = {
            "kind": kind,
            "value": value,
            "scope": LimitScope.ACCOUNT
            if kind.value in account_scoped
            else LimitScope.MODEL,
        }
        if kind is LimitKind.RPD and reserve_requests:
            entry["reserve"] = reserve_requests
        limits.append(entry)

    spec: dict[str, Any] = {
        "name": name,
        "enabled": enabled,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "reset_tz": reset_tz,
        "paid": paid,
        "limits_are_estimated": estimated,
        "limits": tuple(limits),
    }
    if cost:
        spec["cost"] = cost
    return spec


def model_spec(
    model_id: str,
    *,
    provider: str,
    context: int,
    max_output: int,
    wire_name: str | None = None,
    min_output_tokens: int = 0,
    quality_prior: float = 0.5,
    supports_json_schema: bool = True,
) -> dict[str, Any]:
    """One model. `wire_name` defaults to the id, for providers where they match."""
    return {
        "id": model_id,
        "provider": provider,
        "wire_name": wire_name or model_id,
        "context": context,
        "max_output": max_output,
        "min_output_tokens": min_output_tokens,
        "quality_prior": quality_prior,
        "supports_json_schema": supports_json_schema,
    }


def quota_manifest(
    *, providers: list[dict[str, Any]], models: list[dict[str, Any]]
) -> Manifest:
    """Assemble a validated roster for the governor.

    Role fallback chains are left empty on purpose. They belong to the
    orchestrator's routing, not to quota accounting, and requiring them would
    force a library user to invent a taxonomy they have no use for.
    """
    return Manifest.model_validate(
        {
            "version": 1,
            "providers": {spec["name"]: spec for spec in providers},
            "models": tuple(models),
            "roles": {},
        }
    )
