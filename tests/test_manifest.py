"""Manifest loading and validation. Reads the real models.yaml, no network."""

from __future__ import annotations

import pytest
import yaml

from llmorch.errors import ManifestError
from llmorch.registry.manifest import load_manifest
from llmorch.types import LimitKind, LimitScope, Role


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


# --------------------------------------------------------------------------
# The shipped manifest
# --------------------------------------------------------------------------


def test_real_manifest_loads(manifest):
    assert manifest.version == 1
    assert {"groq", "gemini"} <= set(manifest.providers)


def test_only_groq_and_gemini_are_enabled_in_v1(manifest):
    enabled = {n for n, p in manifest.providers.items() if p.enabled}
    assert enabled == {"groq", "gemini"}


def test_deferred_providers_are_documented_but_inactive(manifest):
    """NIM, Mistral and Perplexity ship as inactive entries so the schema is
    proven against real shapes before those milestones arrive."""
    for name in ("nvidia_nim", "mistral", "perplexity"):
        assert not manifest.providers[name].enabled


def test_groq_tpm_is_the_binding_constraint(manifest):
    # Both figures read from live x-ratelimit headers, not documentation.
    groq = manifest.providers["groq"]
    assert groq.limit(LimitKind.TPM).value == 8000
    # ~7,200 tokens: anything larger can never be served by Groq.
    assert groq.max_request_tokens == 7200


def test_groq_rpm_is_account_scoped(manifest):
    """Org-scoped in reality — extra keys do not raise it. If this were modelled
    as per-model, the governor would over-admit."""
    assert manifest.providers["groq"].limit(LimitKind.RPM).scope is LimitScope.ACCOUNT


def test_gemini_resets_on_pacific_time_not_utc(manifest):
    assert manifest.providers["gemini"].reset_tz == "America/Los_Angeles"
    assert manifest.providers["groq"].reset_tz == "UTC"


def test_gemini_reserves_headroom_for_critical_path_retries(manifest):
    rpd = manifest.providers["gemini"].limit(LimitKind.RPD)
    assert rpd.value == 250
    assert rpd.reserve == 30


def test_capability_sheet_matches_the_intended_split(manifest):
    """Research should favour Gemini; backend should favour a Groq model."""
    gemini = manifest.model("gemini/3.6-flash")
    llama = manifest.model("groq/gpt-oss-120b")

    assert gemini.affinity(Role.RESEARCH) > llama.affinity(Role.RESEARCH)
    assert llama.affinity(Role.BACKEND) > llama.affinity(Role.RESEARCH)


def test_every_chain_spans_at_least_two_vendors(manifest):
    for role in manifest.roles:
        vendors = {manifest.vendor_of(mid) for mid in manifest.chain(role, enabled_only=False)}
        assert len(vendors) >= 2, f"{role} chain is single-vendor"


def test_max_request_tokens_is_the_tighter_of_context_and_tpm(manifest):
    # Groq: 131k context but an 8k TPM ceiling -> TPM wins.
    assert manifest.max_request_tokens("groq/gpt-oss-120b") == 7200
    # Gemini: 250k TPM against a 1M context -> context is not the limit.
    assert manifest.max_request_tokens("gemini/3.6-flash") == 225000


def test_unknown_model_lookup_raises(manifest):
    with pytest.raises(ManifestError):
        manifest.model("nope/does-not-exist")


# --------------------------------------------------------------------------
# Validation rules
# --------------------------------------------------------------------------


def _write(tmp_path, data):
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def _minimal(**overrides):
    data = {
        "version": 1,
        "providers": {
            "alpha": {
                "enabled": True,
                "base_url": "https://a.example/v1",
                "api_key_env": "ALPHA_KEY",
                "limits": [{"kind": "tpm", "scope": "model", "value": 10000}],
            },
            "beta": {
                "enabled": True,
                "base_url": "https://b.example/v1",
                "api_key_env": "BETA_KEY",
                "limits": [{"kind": "tpm", "scope": "model", "value": 10000}],
            },
        },
        "models": [
            {
                "id": "alpha/one",
                "provider": "alpha",
                "wire_name": "one",
                "context": 32000,
                "max_output": 4096,
            },
            {
                "id": "beta/two",
                "provider": "beta",
                "wire_name": "two",
                "context": 32000,
                "max_output": 4096,
            },
        ],
        "roles": {"backend": ["alpha/one", "beta/two"]},
    }
    data.update(overrides)
    return data


def test_single_vendor_chain_is_rejected(tmp_path):
    """The core failover guarantee. A chain that never leaves one vendor cannot
    survive that vendor failing."""
    data = _minimal()
    data["models"].append(
        {
            "id": "alpha/three",
            "provider": "alpha",
            "wire_name": "three",
            "context": 32000,
            "max_output": 4096,
        }
    )
    data["roles"] = {"backend": ["alpha/one", "alpha/three"]}

    with pytest.raises(ManifestError, match="two distinct vendors"):
        load_manifest(_write(tmp_path, data), overlay=tmp_path / "none.yaml")


def test_chain_referencing_unknown_model_is_rejected(tmp_path):
    data = _minimal()
    data["roles"] = {"backend": ["alpha/one", "ghost/model"]}
    with pytest.raises(ManifestError, match="unknown model"):
        load_manifest(_write(tmp_path, data), overlay=tmp_path / "none.yaml")


def test_model_with_undeclared_provider_is_rejected(tmp_path):
    data = _minimal()
    data["models"].append(
        {
            "id": "gamma/x",
            "provider": "gamma",
            "wire_name": "x",
            "context": 1000,
            "max_output": 500,
        }
    )
    with pytest.raises(ManifestError, match="undeclared provider"):
        load_manifest(_write(tmp_path, data), overlay=tmp_path / "none.yaml")


def test_model_that_can_never_fit_its_provider_tpm_is_rejected(tmp_path):
    """max_output above the provider's per-request ceiling means every full-size
    request is unservable by construction — better caught at load than at run."""
    data = _minimal()
    data["models"][0]["max_output"] = 50000  # provider ceiling is 9000
    with pytest.raises(ManifestError, match="can never serve"):
        load_manifest(_write(tmp_path, data), overlay=tmp_path / "none.yaml")


def test_paid_provider_without_pricing_is_rejected(tmp_path):
    data = _minimal()
    data["providers"]["beta"]["paid"] = True
    with pytest.raises(ManifestError, match="no cost"):
        load_manifest(_write(tmp_path, data), overlay=tmp_path / "none.yaml")


def test_duplicate_model_ids_are_rejected(tmp_path):
    data = _minimal()
    data["models"].append(dict(data["models"][0]))
    with pytest.raises(ManifestError, match="duplicate model id"):
        load_manifest(_write(tmp_path, data), overlay=tmp_path / "none.yaml")


def test_reserve_must_be_smaller_than_the_limit(tmp_path):
    data = _minimal()
    data["providers"]["alpha"]["limits"].append(
        {"kind": "rpd", "scope": "model", "value": 100, "reserve": 100}
    )
    with pytest.raises(ManifestError):
        load_manifest(_write(tmp_path, data), overlay=tmp_path / "none.yaml")


def test_local_overlay_overrides_base_values(tmp_path):
    """Per-machine overrides exist so a user can correct a limit without
    editing tracked source."""
    base = _write(tmp_path, _minimal())
    overlay = tmp_path / "models.local.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {"providers": {"alpha": {"limits": [{"kind": "tpm", "value": 99000}]}}}
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(base, overlay=overlay)
    assert manifest.providers["alpha"].limit(LimitKind.TPM).value == 99000
    # Untouched fields survive the merge.
    assert manifest.providers["alpha"].api_key_env == "ALPHA_KEY"


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "absent.yaml", overlay=tmp_path / "none.yaml")
