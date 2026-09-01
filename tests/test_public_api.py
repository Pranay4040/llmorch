"""The public surface, and the README that documents it.

A library's promise is its import list. These tests pin the names other code is
allowed to depend on, and — more usefully — execute the README's example
verbatim, so the documentation cannot drift away from the code while still
looking correct.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


# ==========================================================================
# The documented example must actually run
# ==========================================================================


def _first_python_block(text: str) -> str:
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    assert match, "the README no longer contains a python example"
    return match.group(1)


def test_the_readme_example_runs_exactly_as_written():
    """Executed, not eyeballed. A quickstart that has quietly stopped working
    is worse than none: it is the first thing anyone tries."""
    source = _first_python_block(README.read_text(encoding="utf-8"))
    namespace: dict = {}
    exec(compile(source, "README.md", "exec"), namespace)  # noqa: S102

    governor = namespace["governor"]
    assert governor.headroom()["groq/gpt-oss-120b"].requests_limit == 1000


def test_the_example_needs_no_api_key_and_no_network():
    """`api_key_env` names a variable; it does not read one. Quota arithmetic
    is offline work, and the quickstart should not demand credentials to try."""
    source = _first_python_block(README.read_text(encoding="utf-8"))
    assert "GROQ_API_KEY" in source

    import os

    saved = os.environ.pop("GROQ_API_KEY", None)
    try:
        exec(compile(source, "README.md", "exec"), {})  # noqa: S102
    finally:
        if saved is not None:
            os.environ["GROQ_API_KEY"] = saved


# ==========================================================================
# The import surface
# ==========================================================================


def test_quota_exports_what_the_readme_promises():
    import llmorch.quota as quota

    for name in (
        "Governor", "Ticket", "Denial", "Admission", "Priority", "Headroom",
        "LedgerStore", "restore_governor", "make_event", "cost_of",
        "TokenEstimator", "quota_manifest", "provider_spec", "model_spec",
        "Clock", "SystemClock", "FakeClock", "SlidingWindow", "DayCounter",
    ):
        assert name in quota.__all__, f"{name} missing from llmorch.quota.__all__"
        assert hasattr(quota, name), f"{name} not importable"


def test_providers_exports_the_client_and_the_header_parser():
    import llmorch.providers as providers

    for name in (
        "OpenAICompatProvider", "Transport", "UrllibTransport", "HttpResponse",
        "parse_rate_limit_headers", "retry_after_from_body", "MockProvider",
    ):
        assert name in providers.__all__, name
        assert hasattr(providers, name), name


def test_importing_the_quota_layer_does_not_drag_in_the_orchestrator():
    """Someone governing quota in their own program should not be paying for
    the scheduler, the demo, or anything that reads models.yaml."""
    import subprocess
    import sys

    probe = (
        "import sys; import llmorch.quota; "
        "orchestration = [m for m in sys.modules "
        "if m.startswith('llmorch.engine') or m.startswith('llmorch.negotiate') "
        "or m.startswith('llmorch.demo')]; "
        "print(','.join(sorted(orchestration)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"quota pulled in {out.stdout.strip()}"


# ==========================================================================
# Usable without this project's config
# ==========================================================================


def test_a_roster_can_be_declared_without_yaml_or_roles():
    """Role chains belong to the orchestrator's routing. Requiring them would
    force a library user to invent a taxonomy they have no use for."""
    from llmorch.quota import Governor, Ticket, model_spec, provider_spec, quota_manifest

    manifest = quota_manifest(
        providers=[
            provider_spec("solo", base_url="https://example.test/v1",
                          api_key_env="SOLO_KEY", rpm=2, tpm=1000)
        ],
        models=[model_spec("solo/small", provider="solo", context=8000, max_output=512)],
    )
    governor = Governor(manifest)

    assert isinstance(governor.try_acquire("solo/small", 100, 200), Ticket)
    assert manifest.roles == {}


def test_account_scoped_limits_are_shared_across_models():
    """The classic error in this domain: if a limit is account-scoped, falling
    back to a different model on the same provider buys no extra quota."""
    from llmorch.quota import Admission, Governor, model_spec, provider_spec, quota_manifest

    manifest = quota_manifest(
        providers=[
            provider_spec("shared", base_url="https://example.test/v1",
                          api_key_env="K", rpm=1, account_scoped=("rpm",))
        ],
        models=[
            model_spec("shared/a", provider="shared", context=8000, max_output=512),
            model_spec("shared/b", provider="shared", context=8000, max_output=512),
        ],
    )
    governor = Governor(manifest)
    governor.try_acquire("shared/a", 10, 10)

    denial = governor.try_acquire("shared/b", 10, 10)
    assert denial.verdict is Admission.WAIT


def test_a_reserve_is_held_back_from_normal_priority_work():
    from llmorch.quota import (
        Admission, Governor, Priority, Ticket, model_spec, provider_spec, quota_manifest,
    )

    manifest = quota_manifest(
        providers=[
            provider_spec("p", base_url="https://example.test/v1", api_key_env="K",
                          rpd=3, reserve_requests=2)
        ],
        models=[model_spec("p/m", provider="p", context=8000, max_output=512)],
    )
    governor = Governor(manifest)
    assert isinstance(governor.try_acquire("p/m", 10, 10), Ticket)

    # One request of three is spent; two are reserved.
    assert governor.try_acquire("p/m", 10, 10).verdict is Admission.WAIT
    assert isinstance(
        governor.try_acquire("p/m", 10, 10, priority=Priority.HIGH), Ticket
    )


def test_estimated_limits_are_recorded_as_estimates():
    """A guessed ceiling should never read as a measurement."""
    from llmorch.quota import provider_spec, quota_manifest, model_spec

    manifest = quota_manifest(
        providers=[
            provider_spec("guess", base_url="https://example.test/v1",
                          api_key_env="K", rpm=10, estimated=True)
        ],
        models=[model_spec("guess/m", provider="guess", context=8000, max_output=512)],
    )
    assert manifest.providers["guess"].limits_are_estimated is True
