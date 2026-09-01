"""Persistent usage ledger.

Quota is a property of the *account*, not of the process that happens to be
running. A second `llmorch run` an hour later starts with empty in-memory
counters and, without this file, would believe it holds the entire daily
allowance — then walk straight into a wall the earlier run already built.
Worse, that discovery costs a live request to make.

So every call that reaches a provider is appended here as a `UsageEvent`, and
`restore_governor` replays today's rows back into the counters at startup.
Events are the source of truth; the counters are a cache of them. A crash can
therefore lose at most the request in flight, never leave a counter silently
drifting.

Three deliberate choices:

* **Append-only.** Nothing is ever updated in place, so a concurrent run can
  only ever add rows, never corrupt one.
* **Rows are stamped with the provider's own `day_key`**, computed in that
  provider's reset timezone. Groq rolls over at UTC midnight and Gemini at
  Pacific midnight; a single shared date column would be wrong for one of them
  every single day.
* **Only daily counters are restored.** The per-minute windows run on the
  monotonic clock, which has no meaning across processes — and they drain in
  under a minute anyway.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import state_db_path
from ..registry.manifest import Manifest
from ..types import Usage, UsageEvent
from .windows import day_key

if TYPE_CHECKING:  # pragma: no cover - import cycle avoided at runtime
    from .governor import Governor

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                TEXT    NOT NULL,
    node_id               TEXT,
    purpose               TEXT    NOT NULL,
    provider              TEXT    NOT NULL,
    model_id              TEXT    NOT NULL,
    ts_utc                TEXT    NOT NULL,
    day_key               TEXT    NOT NULL,
    est_prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    est_completion_tokens INTEGER NOT NULL DEFAULT 0,
    prompt_tokens         INTEGER NOT NULL DEFAULT 0,
    completion_tokens     INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens      INTEGER NOT NULL DEFAULT 0,
    cached_tokens         INTEGER NOT NULL DEFAULT 0,
    cost_usd              TEXT    NOT NULL DEFAULT '0',
    ok                    INTEGER NOT NULL DEFAULT 1,
    http_status           INTEGER NOT NULL DEFAULT 200,
    latency_ms            INTEGER NOT NULL DEFAULT 0,
    error                 TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_provider_day
    ON usage_events (provider, day_key);
CREATE INDEX IF NOT EXISTS idx_events_model_day
    ON usage_events (model_id, day_key);
CREATE INDEX IF NOT EXISTS idx_events_run
    ON usage_events (run_id);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class DayUsage:
    """What one bucket has consumed within one provider-local day."""

    requests: int = 0
    tokens: int = 0
    cost_usd: Decimal = Decimal("0")
    failures: int = 0

    @property
    def is_empty(self) -> bool:
        return self.requests == 0 and self.tokens == 0


@dataclass(frozen=True, slots=True)
class ModelDayUsage:
    """Per-model row for `llmorch ledger` and `llmorch quota`."""

    model_id: str
    provider: str
    day_key: str
    requests: int
    tokens: int
    cost_usd: Decimal
    failures: int


@dataclass(frozen=True, slots=True)
class LedgerRow:
    """One recorded call, as read back out."""

    id: int
    run_id: str
    node_id: str | None
    purpose: str
    provider: str
    model_id: str
    ts_utc: str
    day_key: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: Decimal
    ok: bool
    http_status: int
    latency_ms: int
    error: str | None


class LedgerStore:
    """SQLite-backed append-only ledger.

    Usable as a context manager. `path=":memory:"` gives an isolated store for
    tests without touching the real one — which matters, because the real one
    lives outside the checkout precisely so that every clone shares it.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else state_db_path()
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle --------------------------------------------------------

    def open(self) -> LedgerStore:
        if self._conn is not None:
            return self

        is_memory = str(self.path) == ":memory:"
        if not is_memory:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        if not is_memory:
            # WAL lets a reader (`llmorch quota`) run while a writer holds the
            # ledger, which is the normal case when checking on a live run.
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        conn.commit()
        self._conn = conn

        stored = self.get_meta("schema_version")
        if stored is None:
            self.set_meta("schema_version", SCHEMA_VERSION)
        return self

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> LedgerStore:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- key/value --------------------------------------------------------

    def get_meta(self, key: str) -> Any:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else None

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def save_calibration(self, data: dict[str, dict[str, float]]) -> None:
        """Persist the estimator's learned per-provider ratios.

        The estimator needs ~20 samples to converge. Discarding that at process
        exit would mean every run starts mis-estimating again, and each sample
        costs a live request to obtain.
        """
        self.set_meta("estimator_calibration", data)

    def load_calibration(self) -> dict[str, dict[str, float]]:
        return self.get_meta("estimator_calibration") or {}

    # -- writing ----------------------------------------------------------

    def record(self, event: UsageEvent) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO usage_events (
                run_id, node_id, purpose, provider, model_id, ts_utc, day_key,
                est_prompt_tokens, est_completion_tokens,
                prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens,
                cost_usd, ok, http_status, latency_ms, error
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event.run_id,
                event.node_id,
                event.purpose,
                event.provider,
                event.model_id,
                event.ts_utc,
                event.day_key,
                event.est_prompt_tokens,
                event.est_completion_tokens,
                event.usage.prompt_tokens,
                event.usage.completion_tokens,
                event.usage.reasoning_tokens,
                event.usage.cached_tokens,
                str(event.cost_usd),
                1 if event.ok else 0,
                event.http_status,
                event.latency_ms,
                event.error,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    # -- reading ----------------------------------------------------------

    def _usage_where(self, where: str, params: tuple[Any, ...]) -> DayUsage:
        row = self.conn.execute(
            f"""
            SELECT
                -- A request that reached the provider consumed a slot even if
                -- it came back a 429 or a 400. Only calls that never left this
                -- machine (status 0) are free.
                COALESCE(SUM(CASE WHEN http_status > 0 THEN 1 ELSE 0 END), 0) AS requests,
                COALESCE(SUM(prompt_tokens + completion_tokens + reasoning_tokens), 0)
                    AS tokens,
                COALESCE(SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END), 0) AS failures,
                COALESCE(GROUP_CONCAT(cost_usd), '') AS costs
            FROM usage_events WHERE {where}
            """,
            params,
        ).fetchone()

        costs = [c for c in str(row["costs"]).split(",") if c]
        return DayUsage(
            requests=int(row["requests"]),
            tokens=int(row["tokens"]),
            cost_usd=sum((Decimal(c) for c in costs), Decimal("0")),
            failures=int(row["failures"]),
        )

    def provider_day_usage(self, provider: str, day: str) -> DayUsage:
        """Account-scoped totals: every model on this provider, this day."""
        return self._usage_where(
            "provider = ? AND day_key = ?", (provider, day)
        )

    def model_day_usage(self, model_id: str, day: str) -> DayUsage:
        return self._usage_where("model_id = ? AND day_key = ?", (model_id, day))

    def run_usage(self, run_id: str) -> DayUsage:
        return self._usage_where("run_id = ?", (run_id,))

    def day_table(self, days: int = 1, *, since: str | None = None) -> list[ModelDayUsage]:
        """Per-(model, day) totals, newest day first.

        `days` counts distinct day keys rather than calendar days, so a machine
        that sat idle for a week still shows its last few active days.
        """
        if since is not None:
            rows = self.conn.execute(
                "SELECT DISTINCT day_key FROM usage_events WHERE day_key >= ? "
                "ORDER BY day_key DESC",
                (since,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT DISTINCT day_key FROM usage_events ORDER BY day_key DESC "
                "LIMIT ?",
                (days,),
            ).fetchall()

        keys = [r["day_key"] for r in rows]
        if not keys:
            return []

        placeholders = ",".join("?" * len(keys))
        totals = self.conn.execute(
            f"""
            SELECT model_id, provider, day_key,
                   SUM(CASE WHEN http_status > 0 THEN 1 ELSE 0 END) AS requests,
                   SUM(prompt_tokens + completion_tokens + reasoning_tokens) AS tokens,
                   SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS failures,
                   GROUP_CONCAT(cost_usd) AS costs
            FROM usage_events
            WHERE day_key IN ({placeholders})
            GROUP BY model_id, provider, day_key
            ORDER BY day_key DESC, model_id ASC
            """,
            tuple(keys),
        ).fetchall()

        out: list[ModelDayUsage] = []
        for row in totals:
            costs = [c for c in str(row["costs"] or "").split(",") if c]
            out.append(
                ModelDayUsage(
                    model_id=row["model_id"],
                    provider=row["provider"],
                    day_key=row["day_key"],
                    requests=int(row["requests"] or 0),
                    tokens=int(row["tokens"] or 0),
                    cost_usd=sum((Decimal(c) for c in costs), Decimal("0")),
                    failures=int(row["failures"] or 0),
                )
            )
        return out

    def recent(self, limit: int = 20, *, run_id: str | None = None) -> list[LedgerRow]:
        if run_id:
            rows = self.conn.execute(
                "SELECT * FROM usage_events WHERE run_id = ? ORDER BY id DESC LIMIT ?",
                (run_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM usage_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_ledger(r) for r in rows]

    def runs(self, limit: int = 10) -> list[tuple[str, int, str]]:
        """(run_id, calls, last timestamp), newest first."""
        rows = self.conn.execute(
            "SELECT run_id, COUNT(*) AS calls, MAX(ts_utc) AS last "
            "FROM usage_events GROUP BY run_id ORDER BY last DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [(r["run_id"], int(r["calls"]), r["last"]) for r in rows]

    @property
    def total_events(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM usage_events").fetchone()
        return int(row["n"])

    # -- maintenance ------------------------------------------------------

    def prune(self, keep_days: int = 30) -> int:
        """Drop rows older than the newest `keep_days` distinct day keys.

        Old rows have no effect on admission — only today's matter — so this is
        purely housekeeping, and it never touches a day that is still live.
        """
        rows = self.conn.execute(
            "SELECT DISTINCT day_key FROM usage_events ORDER BY day_key DESC LIMIT ?",
            (keep_days,),
        ).fetchall()
        if len(rows) < keep_days:
            return 0
        cutoff = rows[-1]["day_key"]
        cur = self.conn.execute(
            "DELETE FROM usage_events WHERE day_key < ?", (cutoff,)
        )
        self.conn.commit()
        return cur.rowcount


def _row_to_ledger(row: sqlite3.Row) -> LedgerRow:
    return LedgerRow(
        id=int(row["id"]),
        run_id=row["run_id"],
        node_id=row["node_id"],
        purpose=row["purpose"],
        provider=row["provider"],
        model_id=row["model_id"],
        ts_utc=row["ts_utc"],
        day_key=row["day_key"],
        prompt_tokens=int(row["prompt_tokens"]),
        completion_tokens=int(row["completion_tokens"]),
        cost_usd=Decimal(str(row["cost_usd"])),
        ok=bool(row["ok"]),
        http_status=int(row["http_status"]),
        latency_ms=int(row["latency_ms"]),
        error=row["error"],
    )


# --------------------------------------------------------------------------
# Event construction and replay
# --------------------------------------------------------------------------


def make_event(
    *,
    run_id: str,
    node_id: str | None,
    purpose: str,
    manifest: Manifest,
    model_id: str,
    usage: Usage,
    est_prompt_tokens: int = 0,
    est_completion_tokens: int = 0,
    now: datetime | None = None,
    ok: bool = True,
    http_status: int = 200,
    latency_ms: int = 0,
    error: str | None = None,
) -> UsageEvent:
    """Build a `UsageEvent`, stamping it with the provider's own day key.

    The day key must be computed in the provider's reset timezone, not the
    machine's: Groq's day ends at UTC midnight and Gemini's at Pacific midnight,
    so a single shared date would be wrong for one of them every day.
    """
    provider = manifest.provider_of(model_id)
    moment = now or datetime.now(timezone.utc)
    return UsageEvent(
        run_id=run_id,
        node_id=node_id,
        purpose=purpose,
        provider=provider.name,
        model_id=model_id,
        ts_utc=moment.isoformat(),
        day_key=day_key(moment, provider.reset_tz),
        est_prompt_tokens=est_prompt_tokens,
        est_completion_tokens=est_completion_tokens,
        usage=usage,
        cost_usd=cost_of(provider, usage),
        ok=ok,
        http_status=http_status,
        latency_ms=latency_ms,
        error=error,
    )


def cost_of(provider: Any, usage: Usage) -> Decimal:
    """What one call cost, in USD.

    Zero for every provider currently enabled — the scarce resource here is
    requests, not money — but Perplexity charges a flat fee *per request* on top
    of per-token pricing, which dominates for the short calls this system makes.
    Tracking it from the first milestone means the number is already right when
    a paid provider is switched on.
    """
    cost = provider.cost
    per_token = (
        Decimal(usage.prompt_tokens) * cost.input_per_mtok
        + Decimal(usage.completion_tokens + usage.reasoning_tokens)
        * cost.output_per_mtok
    ) / Decimal(1_000_000)
    return (per_token + cost.per_request).quantize(Decimal("0.000001"))


def restore_governor(
    governor: Governor,
    store: LedgerStore,
    manifest: Manifest,
    *,
    now: datetime | None = None,
) -> dict[str, DayUsage]:
    """Replay today's ledger into the governor's daily counters.

    Returns what was restored per model, so the CLI can say so out loud —
    silently inheriting 200 of Gemini's 250 daily requests is exactly the kind
    of thing a person needs told before a run starts.
    """
    moment = now or governor.clock.now_utc()
    restored: dict[str, DayUsage] = {}

    for model in manifest.enabled_models:
        provider = manifest.providers[model.provider]
        today = day_key(moment, provider.reset_tz)

        model_usage = store.model_day_usage(model.id, today)
        account_usage = store.provider_day_usage(provider.name, today)
        if model_usage.is_empty and account_usage.is_empty:
            continue

        governor.restore_day_usage(
            model.id,
            model_requests=model_usage.requests,
            model_tokens=model_usage.tokens,
            account_requests=account_usage.requests,
            account_tokens=account_usage.tokens,
        )
        if not model_usage.is_empty:
            restored[model.id] = model_usage

    return restored
