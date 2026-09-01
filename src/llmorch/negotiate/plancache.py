"""Remember decompositions, so planning the same build twice costs nothing.

Planning is one request, but it is the *most expensive* request in the system to
lose: it runs at HIGH priority, it comes from the model with the best planning
affinity — which here is also the model with 250 requests a day — and every node
downstream is shaped by it. Re-running `llmorch run "build a notes app"` while
iterating on the executor should not spend that request again.

Keyed on the task text *and the roster*, because a plan is sized against the
models that will execute it: a graph split for a 4,096-token ceiling is the
wrong graph once a 65,536-token model joins.

Entries are plain JSON files a person can read, edit, or delete. A cache that
cannot be inspected is a cache that gets blamed for every strange result.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import state_db_path
from .decompose import Decomposition, plan_signature

CACHE_VERSION = 1


def cache_dir() -> Path:
    """Beside the ledger, not inside the checkout: a plan is a property of the
    task and the account, not of one clone."""
    return state_db_path().parent / "plans"


def cache_path(signature: str, *, root: Path | None = None) -> Path:
    return (root or cache_dir()) / f"{signature}.json"


def load(signature: str, *, root: Path | None = None) -> Decomposition | None:
    """Return the cached plan, or None if there is none worth trusting."""
    path = cache_path(signature, root=root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if int(data.get("version", 0)) != CACHE_VERSION:
        return None
    try:
        return Decomposition.from_dict(data.get("plan") or {})
    except Exception:
        # A cached plan that no longer parses is not worth failing over; the
        # run simply pays for a fresh one.
        return None


def save(
    signature: str,
    decomposition: Decomposition,
    *,
    task: str = "",
    root: Path | None = None,
) -> Path:
    """Write a plan atomically, with enough context to be readable later."""
    directory = root or cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = cache_path(signature, root=directory)

    payload = {
        "version": CACHE_VERSION,
        "signature": signature,
        "task": task,
        "cached_utc": datetime.now(timezone.utc).isoformat(),
        "plan": decomposition.to_dict(),
    }
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)
    return path


def forget(signature: str, *, root: Path | None = None) -> bool:
    path = cache_path(signature, root=root)
    if path.is_file():
        path.unlink()
        return True
    return False


def entries(*, root: Path | None = None) -> list[tuple[str, str, str]]:
    """(signature, task, cached_utc) for every readable entry, newest first."""
    directory = root or cache_dir()
    if not directory.is_dir():
        return []
    found: list[tuple[str, str, str]] = []
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        found.append(
            (
                str(data.get("signature") or path.stem),
                str(data.get("task") or ""),
                str(data.get("cached_utc") or ""),
            )
        )
    return sorted(found, key=lambda row: row[2], reverse=True)


__all__ = [
    "cache_dir",
    "cache_path",
    "entries",
    "forget",
    "load",
    "plan_signature",
    "save",
]
