"""Foundation tests: types, errors, and config. No network, no clock reliance."""

from __future__ import annotations

from decimal import Decimal

import pytest

from llmorch.config import RunConfig, load_dotenv
from llmorch.errors import (
    ConfigError,
    MissingKeyError,
    QuotaExhausted,
    SchemaInvalid,
    TransportError,
    Unservable,
)
from llmorch.types import (
    Admission,
    Bid,
    ChatRequest,
    Denial,
    Message,
    Role,
    ScoreBreakdown,
    Usage,
)

# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------


def test_usage_total_includes_reasoning_tokens():
    u = Usage(prompt_tokens=100, completion_tokens=50, reasoning_tokens=25)
    assert u.total_tokens == 175


def test_score_breakdown_weights_sum_as_specified():
    b = ScoreBreakdown(
        z_confidence=1.0,
        role_affinity=1.0,
        track_record=1.0,
        quality_prior=1.0,
        quota_pressure=0.0,
    )
    assert b.total == pytest.approx(0.90)


def test_quota_pressure_penalises_a_model_near_its_wall():
    """A model close to its daily limit should score below an identical one that
    is not, with no special-casing anywhere else in the pipeline."""
    fresh = ScoreBreakdown(z_confidence=1.0, role_affinity=0.8, quota_pressure=0.0)
    nearly_spent = ScoreBreakdown(
        z_confidence=1.0, role_affinity=0.8, quota_pressure=1.0
    )
    assert nearly_spent.total < fresh.total


@pytest.mark.parametrize(
    "verdict,permanent",
    [
        (Admission.UNSERVABLE, True),
        (Admission.EXHAUSTED_TODAY, True),
        (Admission.COST_BLOCKED, True),
        (Admission.WAIT, False),
    ],
)
def test_denial_distinguishes_waiting_from_hopeless(verdict, permanent):
    """The scheduler branches on this. Treating UNSERVABLE as a wait condition
    would hang the run forever."""
    assert Denial(verdict, "m", "r").is_permanent_today is permanent


def test_roles_are_a_closed_set():
    assert Role("frontend") is Role.FRONTEND
    with pytest.raises(ValueError):
        Role("freeform-role-name")


def test_chat_request_requires_explicit_max_tokens():
    """max_tokens has no default on purpose: it is the hard upper bound that
    makes token reservation sound."""
    with pytest.raises(TypeError):
        ChatRequest(model_id="m", messages=())  # type: ignore[call-arg]

    req = ChatRequest(model_id="m", messages=(Message("user", "hi"),), max_tokens=512)
    assert req.max_tokens == 512


def test_bids_are_stored_raw_for_later_normalisation():
    bid = Bid(model_id="m", node_id="n1", confidence=0.99)
    assert bid.confidence == 0.99


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_transport_and_schema_failures_are_retryable():
    assert TransportError("boom").is_retryable
    assert SchemaInvalid("bad json").is_retryable


def test_unservable_and_exhausted_are_not_retryable():
    """Retrying either just burns quota walking back into the same wall."""
    assert not Unservable("request exceeds TPM ceiling").is_retryable
    assert not QuotaExhausted("daily cap reached").is_retryable


def test_missing_key_error_names_the_variable_but_never_a_value():
    err = MissingKeyError("groq needs GROQ_API_KEY, which is unset or blank.")
    assert "GROQ_API_KEY" in str(err)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def _cfg(**kw):
    return RunConfig(task="build a notes app", run_id="test-run", **kw)


def test_run_config_rejects_invalid_enums():
    with pytest.raises(ConfigError):
        _cfg(review="sometimes")
    with pytest.raises(ConfigError):
        _cfg(negotiate="maybe")


def test_paid_requires_both_flag_and_budget():
    """Two independent gates, so neither a stray flag nor a stray budget alone
    can start spending money."""
    assert not _cfg().paid_enabled
    assert not _cfg(allow_paid=True).paid_enabled
    assert not _cfg(max_usd=Decimal("5")).paid_enabled
    assert _cfg(allow_paid=True, max_usd=Decimal("5")).paid_enabled


def test_output_dir_is_nested_inside_the_run_dir():
    cfg = _cfg()
    assert cfg.output_dir.parent == cfg.run_dir


def test_dotenv_parses_comments_quotes_and_export_prefix(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "PLAIN=value1",
                'QUOTED="value2"',
                "SINGLE='value3'",
                "export EXPORTED=value4",
                "BLANK=",
                "no_equals_sign",
            ]
        ),
        encoding="utf-8",
    )
    for k in ("PLAIN", "QUOTED", "SINGLE", "EXPORTED", "BLANK"):
        monkeypatch.delenv(k, raising=False)

    assert load_dotenv(env) == 4
    import os

    assert os.environ["PLAIN"] == "value1"
    assert os.environ["QUOTED"] == "value2"
    assert os.environ["SINGLE"] == "value3"
    assert os.environ["EXPORTED"] == "value4"
    assert "BLANK" not in os.environ


def test_dotenv_does_not_override_real_env_by_default(tmp_path, monkeypatch):
    """An unfilled placeholder in .env must never shadow a key that is already
    set in the real environment."""
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=from_file", encoding="utf-8")

    monkeypatch.setenv("GROQ_API_KEY", "from_environment")
    load_dotenv(env)
    import os

    assert os.environ["GROQ_API_KEY"] == "from_environment"

    load_dotenv(env, override=True)
    assert os.environ["GROQ_API_KEY"] == "from_file"


def test_missing_dotenv_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "nonexistent.env") == 0
