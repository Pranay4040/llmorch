"""Everything the dashboard shows, gathered into one JSON-able snapshot.

Kept separate from the server on purpose. The numbers here are the same ones
`llmorch quota`, `llmorch ledger` and `llmorch resume --list` print — a
dashboard that computed its own would eventually disagree with the CLI, and
then nobody would know which to believe.

Nothing in here writes. The dashboard is a window onto state that other
commands produce, which is what makes it safe to leave running.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..config import profiles_path, runs_dir, state_db_path
from ..engine.checkpoint import list_checkpoints
from ..negotiate import plancache
from ..negotiate.profiles import Profiles
from ..quota.governor import Governor
from ..quota.store import LedgerStore, restore_governor
from ..registry.manifest import load_manifest


def _quota(manifest, store: LedgerStore) -> list[dict[str, Any]]:
    """Headroom per model, with today's ledger replayed in first.

    Without the replay this would show a fresh process's empty counters, which
    is exactly the misleading number the ledger exists to prevent.
    """
    governor = Governor(manifest)
    restore_governor(governor, store, manifest)

    rows = []
    for model_id, head in sorted(governor.headroom().items()):
        limit = head.requests_limit or 0
        rows.append(
            {
                "model_id": model_id,
                "provider": head.provider,
                "requests_used": head.requests_used,
                "requests_limit": head.requests_limit,
                "fraction": (head.requests_used / limit) if limit else 0.0,
                "tokens_minute": head.tokens_used_minute,
                "tokens_limit_minute": head.tokens_limit_minute,
                "seconds_to_reset": head.seconds_to_reset,
                "reset_tz": head.reset_tz,
                "estimated": manifest.providers[head.provider].limits_are_estimated,
            }
        )
    return rows


def _spend(store: LedgerStore) -> dict[str, Any]:
    by_day = store.day_table(days=7)
    by_purpose: dict[str, int] = {}
    for row in store.recent(200):
        by_purpose[row.purpose] = by_purpose.get(row.purpose, 0) + 1

    return {
        "days": [
            {
                "day": row.day_key,
                "model_id": row.model_id,
                "requests": row.requests,
                "tokens": row.tokens,
                "failures": row.failures,
                "cost_usd": str(row.cost_usd),
            }
            for row in by_day
        ],
        "by_purpose": by_purpose,
        "total_events": store.total_events,
    }


def _runs() -> list[dict[str, Any]]:
    rows = []
    for book in list_checkpoints(limit=12):
        moment = book.resumable_at()
        rows.append(
            {
                "run_id": book.run_id,
                "task": book.task,
                "done": len(book.completed),
                "left": len(book.unfinished),
                "complete": book.is_complete,
                "blocked_until": moment.isoformat() if moment else None,
                "updated_utc": book.updated_utc,
            }
        )
    return rows


def _track_record() -> list[dict[str, Any]]:
    profiles = Profiles.load()
    return [
        {
            "model_id": model_id,
            "role": role.value,
            "score": round(record.score, 3),
            "attempts": record.attempts,
            "successes": record.successes,
            "rejections": record.rejections,
        }
        for model_id, role, record in profiles.rows()
    ]


def _recent(store: LedgerStore, limit: int = 30) -> list[dict[str, Any]]:
    return [
        {
            "ts_utc": row.ts_utc,
            "model_id": row.model_id,
            "purpose": row.purpose,
            "node_id": row.node_id,
            "tokens": row.prompt_tokens + row.completion_tokens,
            "latency_ms": row.latency_ms,
            "ok": row.ok,
            "status": row.http_status,
            # Provider text, and therefore untrusted: the server never
            # interpolates it into markup, and the page sets it as text.
            "error": (row.error or "")[:300],
        }
        for row in store.recent(limit)
    ]


def snapshot() -> dict[str, Any]:
    """One read of everything, taken together so the views cannot disagree."""
    manifest = load_manifest()
    with LedgerStore(state_db_path()) as store:
        return {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "quota": _quota(manifest, store),
            "spend": _spend(store),
            "runs": _runs(),
            "track_record": _track_record(),
            "recent": _recent(store),
            "paths": {
                # Locations, never contents: no key, no artifact, nothing a
                # provider wrote.
                "ledger": str(state_db_path()),
                "profiles": str(profiles_path()),
                "runs": str(runs_dir()),
                "plans": str(plancache.cache_dir()),
            },
            "roster": [
                {
                    "model_id": m.id,
                    "provider": m.provider,
                    "wire_name": m.wire_name,
                    "quality_prior": m.quality_prior,
                    "context": m.context,
                    "max_output": m.max_output,
                }
                for m in manifest.enabled_models
            ],
        }
