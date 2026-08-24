"""Load and validate models.yaml.

Two validations here are load-bearing rather than cosmetic:

* `max_request_tokens` is derived from each provider's TPM, so the governor can
  tell "too big, ever" apart from "too big, right now".
* Every role fallback chain must span at least two vendors. Failure modes
  correlate within a vendor, so a same-vendor-only chain is decorative.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config import project_root
from ..errors import ManifestError
from ..types import LimitKind, LimitScope, Role

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


class LimitSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: LimitKind
    scope: LimitScope = LimitScope.MODEL
    value: int = Field(gt=0)
    reserve: int = Field(default=0, ge=0)

    @field_validator("reserve")
    @classmethod
    def _reserve_fits_within_value(cls, v: int, info) -> int:
        value = info.data.get("value")
        if value is not None and v >= value:
            raise ValueError(f"reserve ({v}) must be less than value ({value})")
        return v


class CostSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_per_mtok: Decimal = Decimal("0")
    output_per_mtok: Decimal = Decimal("0")
    per_request: Decimal = Decimal("0")
    """Charged per call regardless of size — dominates cost for short requests."""


class ProviderSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = ""
    enabled: bool = False
    kind: str = "openai_compat"
    base_url: str
    api_key_env: str
    reset_tz: str = "UTC"
    paid: bool = False
    trains_on_prompts: bool = False
    limits_are_estimated: bool = False
    """True when published limits do not exist and must be learned at runtime."""
    limits: tuple[LimitSpec, ...] = ()
    cost: CostSpec = CostSpec()

    def limit(self, kind: LimitKind) -> LimitSpec | None:
        return next((lim for lim in self.limits if lim.kind is kind), None)

    def cost_for(self, prompt_tokens: int, completion_tokens: int) -> Decimal:
        """Money owed for one request.

        `per_request` is added whole, and it is not a rounding detail:
        Perplexity charges $5–14 per 1,000 calls *on top of* tokens, which for
        requests as short as these is the larger half of the bill. A cost model
        counting only tokens would under-report it severalfold.

        Free providers declare no cost and return exactly zero, so callers never
        have to branch on `paid`.
        """
        million = Decimal(1_000_000)
        return (
            self.cost.input_per_mtok * Decimal(prompt_tokens) / million
            + self.cost.output_per_mtok * Decimal(completion_tokens) / million
            + self.cost.per_request
        )

    @property
    def max_request_tokens(self) -> int | None:
        """Largest single request this provider can ever serve.

        None when no TPM limit applies. Requests above this are UNSERVABLE:
        no amount of waiting makes them fit.
        """
        tpm = self.limit(LimitKind.TPM)
        return int(tpm.value * 0.9) if tpm else None


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    provider: str
    wire_name: str
    context: int = Field(gt=0)
    max_output: int = Field(gt=0)
    quality_prior: float = Field(default=0.5, ge=0.0, le=1.0)
    supports_json_schema: bool = True
    role_affinity: dict[Role, float] = Field(default_factory=dict)

    @field_validator("role_affinity")
    @classmethod
    def _affinities_in_range(cls, v: dict[Role, float]) -> dict[Role, float]:
        for role, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"role_affinity[{role}] = {score} outside 0..1")
        return v

    def affinity(self, role: Role) -> float:
        """Prior for this role. Unlisted roles score neutral-low rather than
        zero, so an unscored model stays eligible but unfavoured."""
        return self.role_affinity.get(role, 0.3)


class Defaults(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_request_tokens_ratio: float = Field(default=0.9, gt=0.0, le=1.0)
    review_prefers_provider: str = "groq"


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    providers: dict[str, ProviderSpec]
    models: tuple[ModelSpec, ...]
    roles: dict[Role, tuple[str, ...]]
    defaults: Defaults = Defaults()

    # -- lookups ----------------------------------------------------------

    def model(self, model_id: str) -> ModelSpec:
        for m in self.models:
            if m.id == model_id:
                return m
        raise ManifestError(f"unknown model {model_id!r}")

    def knows(self, model_id: str) -> bool:
        """Whether this model is still declared.

        The ledger outlives the manifest: it holds every model ever used on
        this account, including ones since renamed or retired by the vendor.
        Callers reading history have to ask before assuming.
        """
        return any(m.id == model_id for m in self.models)

    def provider_of(self, model_id: str) -> ProviderSpec:
        return self.providers[self.model(model_id).provider]

    def vendor_of(self, model_id: str) -> str:
        """Vendor identity used for cross-vendor failover decisions."""
        return self.model(model_id).provider

    @property
    def enabled_models(self) -> tuple[ModelSpec, ...]:
        return tuple(m for m in self.models if self.providers[m.provider].enabled)

    def chain(self, role: Role, *, enabled_only: bool = True) -> tuple[str, ...]:
        """Fallback chain for a role, preference-ordered."""
        ids = self.roles.get(role, ())
        if not enabled_only:
            return ids
        return tuple(i for i in ids if self.providers[self.model(i).provider].enabled)

    def max_request_tokens(self, model_id: str) -> int:
        """Ceiling for a single request: the tighter of the provider's TPM share
        and the model's own output cap plus context."""
        provider = self.provider_of(model_id)
        model = self.model(model_id)
        tpm = provider.limit(LimitKind.TPM)
        if tpm is None:
            return model.context
        ratio = self.defaults.max_request_tokens_ratio
        return min(model.context, int(tpm.value * ratio))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge; overlay wins. Lists replace rather than concatenate,
    so an override can shorten a fallback chain."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _validate(manifest: Manifest) -> None:
    """Cross-field checks pydantic cannot express on its own."""
    # Every model points at a declared provider.
    for m in manifest.models:
        if m.provider not in manifest.providers:
            raise ManifestError(
                f"model {m.id!r} references undeclared provider {m.provider!r}"
            )

    seen: set[str] = set()
    for m in manifest.models:
        if m.id in seen:
            raise ManifestError(f"duplicate model id {m.id!r}")
        seen.add(m.id)

    known = {m.id for m in manifest.models}

    for role, chain in manifest.roles.items():
        if not chain:
            raise ManifestError(f"role {role.value!r} has an empty fallback chain")

        for model_id in chain:
            if model_id not in known:
                raise ManifestError(
                    f"role {role.value!r} chain references unknown model {model_id!r}"
                )

        # The cross-vendor rule. Checked against the *declared* chain rather
        # than the enabled subset, so the manifest stays correct even while
        # some providers are switched off.
        vendors = {manifest.model(mid).provider for mid in chain}
        if len(vendors) < 2:
            raise ManifestError(
                f"role {role.value!r} chain uses only vendor {vendors.pop()!r}. "
                "Fallback needs at least two distinct vendors: failure modes "
                "correlate within a vendor, so a same-vendor retry tends to "
                "fail the same way."
            )

    # A model must be able to emit at least something within its provider's
    # per-minute ceiling, or every request to it is unservable by construction.
    for m in manifest.models:
        provider = manifest.providers[m.provider]
        ceiling = provider.max_request_tokens
        if ceiling is not None and m.max_output > ceiling:
            raise ManifestError(
                f"model {m.id!r} has max_output={m.max_output} but provider "
                f"{m.provider!r} caps a single request at ~{ceiling} tokens. "
                "Lower max_output or the model can never serve a full-size request."
            )

    # Paid providers must declare pricing, or spend cannot be tracked.
    for name, p in manifest.providers.items():
        if p.paid and p.cost == CostSpec():
            raise ManifestError(
                f"provider {name!r} is marked paid but declares no cost — "
                "spend would be silently untracked."
            )


def load_manifest(
    path: Path | None = None, *, overlay: Path | None = None
) -> Manifest:
    """Load models.yaml, apply models.local.yaml, and validate."""
    root = project_root()
    manifest_path = path or (root / "models.yaml")
    overlay_path = overlay if overlay is not None else (root / "models.local.yaml")

    if not manifest_path.is_file():
        raise ManifestError(f"manifest not found at {manifest_path}")

    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ManifestError(f"{manifest_path} is not valid YAML: {exc}") from exc

    if overlay_path.is_file():
        try:
            extra = yaml.safe_load(overlay_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ManifestError(f"{overlay_path} is not valid YAML: {exc}") from exc
        data = _deep_merge(data, extra)

    # Providers are keyed by name in YAML; carry that name onto the spec.
    for name, spec in (data.get("providers") or {}).items():
        if isinstance(spec, dict):
            spec.setdefault("name", name)

    try:
        manifest = Manifest.model_validate(data)
    except Exception as exc:  # pydantic ValidationError
        raise ManifestError(f"{manifest_path} failed validation: {exc}") from exc

    _validate(manifest)
    return manifest
