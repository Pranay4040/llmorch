"""Paths, environment loading, and run configuration.

Secret handling rule for this module: an API key may be *read* and handed to a
provider client, and must never be logged, printed, written to the ledger, or
included in an error message. Only variable names appear in diagnostics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from .errors import ConfigError, MissingKeyError

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def project_root() -> Path:
    """Repository root — three levels up from src/llmorch/config.py."""
    return Path(__file__).resolve().parents[2]


def state_db_path() -> Path:
    """Quota ledger location.

    Lives outside the project by default: quota is a property of the *account*,
    not the checkout, so two clones must share one ledger or they will both
    believe they hold the full daily allowance.
    """
    if override := os.environ.get("LLMORCH_STATE_DB"):
        return Path(override).expanduser().resolve()

    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "llmorch" / "state.db"


def runs_dir() -> Path:
    if override := os.environ.get("LLMORCH_RUNS_DIR"):
        return Path(override).expanduser().resolve()
    return project_root() / "runs"


def profiles_path() -> Path:
    """Learned per-(model, role) track record. Machine-local, gitignored."""
    return state_db_path().parent / "profiles.json"


# --------------------------------------------------------------------------
# .env loading
# --------------------------------------------------------------------------


def env_path() -> Path:
    """Where the keys live.

    Overridable for the same reason the ledger and the runs directory are: this
    file is now *written* as well as read, and a test that exercised that
    against the developer's own `.env` would be a test that edits their keys.
    """
    if override := os.environ.get("LLMORCH_ENV"):
        return Path(override).expanduser().resolve()
    return project_root() / ".env"


def load_dotenv(path: Path | None = None, *, override: bool = False) -> int:
    """Minimal .env parser. Returns the number of variables set.

    Deliberately hand-rolled rather than pulling in python-dotenv: the format
    needed here is a dozen lines, and the dependency budget is better spent
    elsewhere. Supports `KEY=value`, `#` comments, blank lines, `export `
    prefixes, and single/double quoted values. Blank values are skipped, so an
    unfilled placeholder never shadows a real environment variable.
    """
    target = path or env_path()
    if not target.is_file():
        return 0

    count = 0
    # utf-8-sig: a .env saved by Notepad carries a byte-order mark, and it
    # would otherwise become part of the first variable's name — a key that
    # is set and unreadable, which reads as a wrong key rather than a
    # mis-encoded file.
    for raw in target.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key or not value:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            count += 1
    return count


def get_api_key(env_var: str, *, provider: str) -> str:
    """Read a provider key, raising if absent.

    The error names the variable but never echoes any value.
    """
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise MissingKeyError(
            f"{provider} needs {env_var}, which is unset or blank. "
            f"Add it to .env (see .env.example)."
        )
    return key


def has_api_key(env_var: str) -> bool:
    return bool(os.environ.get(env_var, "").strip())


class KeyRejected(ConfigError):
    """A key value that must not be written. The message never quotes it."""


def _reject_bad_key(env_var: str, value: str) -> None:
    """Refuse anything that would not survive the round trip.

    `.env` is line-oriented and parsed by `load_dotenv` above, so a value
    carrying a newline would not be one variable with a strange value: it would
    be that variable plus whatever the next line parsed as. A key arriving from
    a web form is exactly the input that has to be checked for it.
    """
    if not env_var.isidentifier() or not env_var.isascii():
        raise KeyRejected(f"{env_var!r} is not a valid environment variable name")
    if len(value) > 512:
        raise KeyRejected(f"the value for {env_var} is too long to be a key")
    if any(ch in value for ch in "\r\n\x00") or value != value.strip():
        raise KeyRejected(
            f"the value for {env_var} has surrounding space or a line break in it"
        )


def write_env_key(env_var: str, value: str, *, path: Path | None = None) -> Path:
    """Set one variable in `.env`, leaving every other line as it was.

    Rewritten in place rather than regenerated, because this file is hand-edited
    and its comments say which provider each key is for and what its free tier
    allows. Regenerating it from a template would be the tidier implementation
    and would throw all of that away.

    An empty value clears the variable rather than deleting the line: a blank is
    already skipped by `load_dotenv`, so the effect is the same and the comment
    above it survives to say what belongs there.

    The value is never logged, never echoed and never returned. Only the path is.
    """
    value = value.strip()
    _reject_bad_key(env_var, value)

    target = path or env_path()
    lines = (
        target.read_text(encoding="utf-8").splitlines() if target.is_file() else []
    )

    replaced = False
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        candidate = stripped.removeprefix("export ").lstrip()
        name, sep, _ = candidate.partition("=")
        if sep and name.strip() == env_var:
            lines[index] = f"{env_var}={value}"
            replaced = True
            break

    if not replaced:
        lines.append(f"{env_var}={value}")

    temp = target.with_name(target.name + ".tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Best effort on Windows, where the mode bits are mostly advisory. Worth
    # doing anyway: on any POSIX machine this file holds every key the account
    # has, and it is created here rather than by the person who knows that.
    try:
        os.chmod(temp, 0o600)
    except OSError:
        pass
    temp.replace(target)
    return target


# --------------------------------------------------------------------------
# Run configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Settings for a single orchestration run."""

    task: str
    run_id: str

    dry_run: bool = True
    """Use the mock provider and make no network calls."""

    allow_paid: bool = False
    """Required before any provider marked `paid: true` may be called."""

    max_usd: Decimal = Decimal("0.00")
    """Hard ceiling on real spend. 0 disables paid providers entirely."""

    review: str = "code"
    """off | code | all — which nodes get Tier 1 cross-vendor review."""

    negotiate: str = "auto"
    """auto | always | never — whether to run the bidding round."""

    max_nodes: int = 10
    max_concurrency: int = 4
    max_retries: int = 2

    circuit_breaker_threshold: int = 2
    """Consecutive hard failures before a model is unhealthy for the run."""

    imbalance_tolerance: float = 0.35
    """How far a model's token share may exceed an even split."""

    token_safety_factor: float = 1.25
    """Multiplier on estimated tokens when reserving, to absorb estimator error."""

    excluded_models: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.review not in ("off", "code", "all"):
            raise ConfigError(f"review must be off|code|all, got {self.review!r}")
        if self.negotiate not in ("auto", "always", "never"):
            raise ConfigError(
                f"negotiate must be auto|always|never, got {self.negotiate!r}"
            )
        if self.max_nodes < 1:
            raise ConfigError("max_nodes must be at least 1")
        if self.max_concurrency < 1:
            raise ConfigError("max_concurrency must be at least 1")
        if self.token_safety_factor < 1.0:
            raise ConfigError("token_safety_factor must be >= 1.0")

    @property
    def run_dir(self) -> Path:
        return runs_dir() / self.run_id

    @property
    def output_dir(self) -> Path:
        """Root that every materialized artifact must resolve inside."""
        return self.run_dir / "output"

    @property
    def paid_enabled(self) -> bool:
        """Paid providers need both the flag and a non-zero budget.

        Two independent gates, so neither a stray flag nor a stray budget value
        alone is enough to start spending money.
        """
        return self.allow_paid and self.max_usd > 0


def new_run_id(now_utc: str, suffix: str) -> str:
    """Build a sortable run id.

    Time is passed in rather than read here so runs stay reproducible under a
    fake clock in tests.
    """
    return f"{now_utc}-{suffix}"
