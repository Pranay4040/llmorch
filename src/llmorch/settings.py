"""What you chose once, so you do not have to choose it every time.

`llmorch --live --providers groq --smoke` is four decisions restated on every
invocation, and the cost of forgetting one of them is not an error message: it
is a session that silently replays canned fixtures and answers the same thing
whatever you type. That failure is quiet, which is what makes it worth removing
from the command line entirely.

So the choices live in a file, the browser page writes it, and `llmorch start`
reads it. Three properties hold it together:

**A missing or broken file is not an error.** It is the defaults. Settings are a
convenience over flags that already work, and a corrupt JSON file must not be
the reason a build cannot start.

**Every value is validated on the way in, not trusted on the way out.** The file
is written by a web request, so an unknown provider name, a review mode that
does not exist, or a concurrency of two million are all things this module
expects to see and clamps. The rest of the system gets a `Settings` it can use
without checking.

**It holds no secrets.** Keys stay in `.env`, which is gitignored and already
the one place the project looks for them. What this file records is which
providers and models to use — the part that is safe to read, copy and diff.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import state_db_path

SETTINGS_NAME = "settings.json"
SETTINGS_VERSION = 1

REVIEW_MODES = ("off", "code", "all")

MAX_NODES_RANGE = (1, 25)
CONCURRENCY_RANGE = (1, 16)


def settings_path() -> Path:
    """Beside the ledger, not inside the checkout.

    Same argument as the ledger's: what models this account uses is a property
    of the account, and two clones of the repository should not disagree about
    it. `LLMORCH_SETTINGS` overrides, which is what the tests use.
    """
    if override := os.environ.get("LLMORCH_SETTINGS"):
        return Path(override).expanduser().resolve()
    return state_db_path().parent / SETTINGS_NAME


@dataclass(slots=True)
class Settings:
    """The standing answer to the flags `start` would otherwise need."""

    live: bool = True
    """Real providers. The default, because the alternative is the mock — and a
    session that quietly replays fixtures is the single most confusing state
    this program has."""

    providers: tuple[str, ...] = ()
    """Which vendors to call. Empty means every one that has a key."""

    models: tuple[str, ...] = ()
    """Which models to use. Empty means every model the manifest enables."""

    review: str = "code"
    smoke: bool = False
    smoke_install: bool = False
    max_nodes: int = 10
    concurrency: int = 4

    configured: bool = False
    """Whether a person has ever saved this file.

    Distinct from "the file exists": it is how `start` can say "you have not set
    anything up yet" rather than silently running on defaults nobody chose."""

    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SETTINGS_VERSION,
            "live": self.live,
            "providers": list(self.providers),
            "models": list(self.models),
            "review": self.review,
            "smoke": self.smoke,
            "smoke_install": self.smoke_install,
            "max_nodes": self.max_nodes,
            "concurrency": self.concurrency,
            "configured": self.configured,
        }

    def save(self, path: Path | None = None) -> Path:
        """Written atomically, with `configured` set: saving is the act that
        makes these choices somebody's rather than the defaults'."""
        target = path or settings_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = replace(self, configured=True).to_dict()
        temp = target.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(target)
        return target

    def summary(self) -> str:
        """One line for the top of a session, so the mode is never a surprise."""
        mode = "live" if self.live else "mock (no network)"
        who = ", ".join(self.providers) if self.providers else "every keyed provider"
        extras = [name for name, on in (("smoke", self.smoke),
                                        ("smoke-install", self.smoke_install)) if on]
        tail = f", {' + '.join(extras)}" if extras else ""
        return f"{mode} — {who}, review {self.review}{tail}"


def _clamp(value: Any, low: int, high: int, fallback: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return fallback


def _names(value: Any) -> tuple[str, ...]:
    """A list of identifiers from whatever the file or the browser sent.

    Deliberately forgiving about the container and strict about the contents:
    anything that is not a plain non-empty string is dropped rather than
    stringified, because a stringified dict would become a provider name that
    matches nothing and disables the roster.
    """
    if not isinstance(value, (list, tuple)):
        return ()
    out = [v.strip() for v in value if isinstance(v, str) and v.strip()]
    return tuple(dict.fromkeys(out))


def from_dict(raw: Any) -> Settings:
    """Build settings from untrusted JSON, keeping whatever is usable.

    Never raises. A field that cannot be read falls back to its default, and the
    rest of the file still applies — losing a whole configuration because one
    number was wrong is worse than the number being wrong.
    """
    if not isinstance(raw, dict):
        return Settings()

    review = raw.get("review")
    return Settings(
        live=bool(raw.get("live", True)),
        providers=_names(raw.get("providers")),
        models=_names(raw.get("models")),
        review=review if review in REVIEW_MODES else "code",
        smoke=bool(raw.get("smoke", False)),
        smoke_install=bool(raw.get("smoke_install", False)),
        max_nodes=_clamp(raw.get("max_nodes"), *MAX_NODES_RANGE, 10),
        concurrency=_clamp(raw.get("concurrency"), *CONCURRENCY_RANGE, 4),
        configured=bool(raw.get("configured", False)),
    )


def read(path: Path | None = None) -> tuple[Settings, str]:
    """The saved choices and, if they could not be read, why.

    Falling back to the defaults is right — a settings file must never be the
    reason a build cannot start — but doing it *silently* is not. A file that
    exists and cannot be parsed looks exactly like a file nobody has written,
    and the difference matters: one of them is a configuration somebody made and
    is not getting.
    """
    target = path or settings_path()
    if not target.is_file():
        return Settings(), ""
    try:
        # utf-8-sig, not utf-8. On Windows both Notepad and PowerShell's
        # `Set-Content -Encoding utf8` write a byte-order mark, and a leading
        # BOM makes `json.loads` fail on a file that is otherwise perfect —
        # which lands as "your settings were ignored and nothing said so".
        raw = json.loads(target.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        return Settings(), f"{target} could not be read ({exc.strerror or exc}); using defaults"
    except json.JSONDecodeError as exc:
        return Settings(), f"{target} is not valid JSON (line {exc.lineno}); using defaults"
    return from_dict(raw), ""


def load(path: Path | None = None) -> Settings:
    """The saved choices, or the defaults. Never an error."""
    return read(path)[0]
