"""Test-wide isolation from the developer's own environment.

The suite is meant to prove the same thing on every machine. It did not: any
test that runs the CLI reaches `main`, which calls `load_dotenv`, which copies
the real `.env` into `os.environ` for the rest of the process. From then on,
tests that clear a key or two to describe a keyless provider were describing a
machine with the *other* keys still set — and passed or failed according to
which providers the developer happens to hold keys for.

It never showed up in CI, because CI has no `.env` at all. That is the shape of
the fault worth naming: a suite that is green on the machine with no secrets and
red on the machine with them is testing the machine, not the code.

So every provider key is removed before each test. A test that needs one sets it
itself, which they already do.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _no_ambient_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset every `*_API_KEY` for the duration of a test.

    Matched by suffix rather than against the manifest: `.env` carries keys for
    providers that are discovered but not yet declared, and one of those
    becoming declared should not quietly re-attach the whole suite to the
    developer's account.
    """
    for name in [k for k in os.environ if k.endswith("_API_KEY")]:
        monkeypatch.delenv(name, raising=False)
