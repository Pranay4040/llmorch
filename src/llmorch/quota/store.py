"""Durable usage ledger.

One append-only table of `UsageEvent` rows is the single source of truth for
all accounting. Quota counters are *derived* from it rather than stored beside
it, so a crash mid-run cannot leave a saved counter disagreeing with the events
that produced it. The worst a crash can cost is the one event in flight.

Why this has to be durable at all: the governor's counters live in memory, and
a daily cap outlives the process that hit it. Without a ledger, a second
`llmorch run` on the same day starts believing it holds all 250 of Gemini's
requests, walks straight into a wall the first run already hit, and spends
several of the few remaining ones discovering that.

The database deliberately lives outside the checkout (see `config.state_db_path`)
because quota is a property of the *account*, not the working copy. Two clones
sharing one key must share one ledger or both will double-count.

Concurrency: WAL plus a busy timeout. Two runs against the same account are
expected to overlap, and readers must never block the writer.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from ..config import state_db_path
from ..types import Usage, UsageEvent
from .windows import day_key

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

-- The hot query is "how much has this provider used today", so the index
-- matches it exactly. Admission control runs it on every process start.
CREATE INDEX IF NOT EXISTS ix_usage_provider_day
    ON usage_events (provider, day_key);

CREATE INDEX IF NOT EXISTS ix_usage_run
    ON usage_events (run_id);

-- Small key/value side table for state that is genuinely a running total
-- rather than an event: the estimator's learned per-provider ratio.
CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class DayUsage:
    """What one model consumed within one provider-local day."""

    provider: str
    model_id: str
    day_key: str
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: Decimal = Decimal("0")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens


@dataclass(frozen=True, slots=True)
class RunSummary:
    """One past run, as the ledger remembers it."""

    run_id: str
    started_utc: str
    ended_utc: str
    requests: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: Decimal
    failures: int
    models: int


class LedgerStore:
    """SQLite-backed usage ledger.

    Cheap to construct and safe to keep open for a whole run. Every write
    commits immediately: an event that is not yet durable is quota that a crash
    would hand back for free.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else state_db_path()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._configure()

    def _configure(self) -> None:
        cur = self._conn.cursor()
        # WAL keeps a concurrent reader from blocking the run that is writing.
        # It is unavailable for :memory:, which is fine — nothing shares one.
        if str(self.path) != ":memory:":
            cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.executescript(_SCHEMA)
        cur.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> LedgerStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- writing ----------------------------------------------------------

    def record(self, event: UsageEvent) -> int:
        """Append one event. Returns its row id.

        Called for every request that *reached* a provider, including the ones
        that came back 429 or 500. A rejected request still consumed a slot in
        the eyes of most rate limiters, and over-counting is the safe direction:
        the cost of it is a little unused quota, whereas under-counting is a
        wall hit without warning.
        """
        cur = self._conn.execute(
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
        return int(cur.lastrowid or 0)

    def record_many(self, events: Iterable[UsageEvent]) -> int:
        return sum(1 for event in events if self.record(event))

    # -- reading: quota ---------------------------------------------------

    def day_usage(self, provider: str, day: str) -> tuple[DayUsage, ...]:
        """Per-model totals for one provider-local day."""
        rows = self._conn.execute(
            """
            SELECT model_id,
                   COUNT(*)                    AS requests,
                   SUM(prompt_tokens)          AS prompt_tokens,
                   SUM(completion_tokens)      AS completion_tokens,
                   SUM(reasoning_tokens)       AS reasoning_tokens,
                   SUM(CAST(cost_usd AS REAL)) AS cost
            FROM usage_events
            WHERE provider = ? AND day_key = ?
            GROUP BY model_id
            """,
            (provider, day),
        ).fetchall()

        return tuple(
            DayUsage(
                provider=provider,
                model_id=row["model_id"],
                day_key=day,
                requests=row["requests"] or 0,
                prompt_tokens=row["prompt_tokens"] or 0,
                completion_tokens=row["completion_tokens"] or 0,
                reasoning_tokens=row["reasoning_tokens"] or 0,
                cost_usd=Decimal(str(row["cost"] or 0)),
            )
            for row in rows
        )

    def usage_by_model_today(
        self, providers: Mapping[str, str], *, now: datetime | None = None
    ) -> dict[str, tuple[int, int]]:
        """`{model_id: (requests, tokens)}` used so far in each provider's own day.

        `providers` maps provider name to its reset timezone. The day boundary
        differs per provider — Groq rolls over at UTC midnight, Gemini at
        Pacific — so each one is asked about its own calendar date, never a
        shared one.
        """
        moment = now or datetime.now(timezone.utc)
        out: dict[str, tuple[int, int]] = {}
        for provider, tz_name in providers.items():
            for usage in self.day_usage(provider, day_key(moment, tz_name)):
                out[usage.model_id] = (usage.requests, usage.total_tokens)
        return out

    # -- reading: reporting -----------------------------------------------

    def runs(self, limit: int = 20) -> tuple[RunSummary, ...]:
        rows = self._conn.execute(
            """
            SELECT run_id,
                   MIN(ts_utc)                     AS started,
                   MAX(ts_utc)                     AS ended,
                   COUNT(*)                        AS requests,
                   SUM(prompt_tokens)              AS prompt_tokens,
                   SUM(completion_tokens)          AS completion_tokens,
                   SUM(CAST(cost_usd AS REAL))     AS cost,
                   SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS failures,
                   COUNT(DISTINCT model_id)        AS models
            FROM usage_events
            GROUP BY run_id
            ORDER BY started DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return tuple(
            RunSummary(
                run_id=row["run_id"],
                started_utc=row["started"] or "",
                ended_utc=row["ended"] or "",
                requests=row["requests"] or 0,
                prompt_tokens=row["prompt_tokens"] or 0,
                completion_tokens=row["completion_tokens"] or 0,
                cost_usd=Decimal(str(row["cost"] or 0)),
                failures=row["failures"] or 0,
                models=row["models"] or 0,
            )
            for row in rows
        )

    def events_for_run(self, run_id: str) -> tuple[UsageEvent, ...]:
        rows = self._conn.execute(
            "SELECT * FROM usage_events WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
        return tuple(_event_of(row) for row in rows)

    def totals_by_model(self, *, since: str | None = None) -> dict[str, DayUsage]:
        """Lifetime (or since-date) totals per model, for the track record."""
        clause = "WHERE ts_utc >= ?" if since else ""
        params = (since,) if since else ()
        rows = self._conn.execute(
            f"""
            SELECT provider, model_id,
                   COUNT(*)                    AS requests,
                   SUM(prompt_tokens)          AS prompt_tokens,
                   SUM(completion_tokens)      AS completion_tokens,
                   SUM(reasoning_tokens)       AS reasoning_tokens,
                   SUM(CAST(cost_usd AS REAL)) AS cost
            FROM usage_events {clause}
            GROUP BY provider, model_id
            """,
            params,
        ).fetchall()

        return {
            row["model_id"]: DayUsage(
                provider=row["provider"],
                model_id=row["model_id"],
                day_key="",
                requests=row["requests"] or 0,
                prompt_tokens=row["prompt_tokens"] or 0,
                completion_tokens=row["completion_tokens"] or 0,
                reasoning_tokens=row["reasoning_tokens"] or 0,
                cost_usd=Decimal(str(row["cost"] or 0)),
            )
            for row in rows
        }

    def spent_usd(self, *, run_id: str | None = None) -> Decimal:
        clause = "WHERE run_id = ?" if run_id else ""
        params = (run_id,) if run_id else ()
        row = self._conn.execute(
            f"SELECT SUM(CAST(cost_usd AS REAL)) AS total FROM usage_events {clause}",
            params,
        ).fetchone()
        return Decimal(str(row["total"] or 0))

    # -- key/value --------------------------------------------------------

    def get_kv(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_kv(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # -- maintenance ------------------------------------------------------

    def prune(self, *, keep_days: int = 90) -> int:
        """Drop events older than `keep_days`. Returns rows removed.

        Only the current provider-local day matters to admission control; the
        rest is history for reporting, and history need not be unbounded.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - keep_days * 86400
        cutoff_iso = datetime.fromtimestamp(cutoff, timezone.utc).isoformat()
        cur = self._conn.execute(
            "DELETE FROM usage_events WHERE ts_utc < ?", (cutoff_iso,)
        )
        return cur.rowcount or 0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _event_of(row: sqlite3.Row) -> UsageEvent:
    return UsageEvent(
        run_id=row["run_id"],
        node_id=row["node_id"],
        purpose=row["purpose"],
        provider=row["provider"],
        model_id=row["model_id"],
        ts_utc=row["ts_utc"],
        day_key=row["day_key"],
        est_prompt_tokens=row["est_prompt_tokens"],
        est_completion_tokens=row["est_completion_tokens"],
        usage=Usage(
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            reasoning_tokens=row["reasoning_tokens"],
            cached_tokens=row["cached_tokens"],
        ),
        cost_usd=Decimal(row["cost_usd"]),
        ok=bool(row["ok"]),
        http_status=row["http_status"],
        latency_ms=row["latency_ms"],
        error=row["error"],
    )


def build_event(
    *,
    run_id: str,
    node_id: str | None,
    purpose: str,
    provider: str,
    model_id: str,
    reset_tz: str,
    est_prompt_tokens: int,
    est_completion_tokens: int,
    usage: Usage,
    cost_usd: Decimal = Decimal("0"),
    ok: bool = True,
    http_status: int = 200,
    latency_ms: int = 0,
    error: str | None = None,
    now: datetime | None = None,
) -> UsageEvent:
    """Stamp an event with both timestamps it needs.

    `ts_utc` orders events globally; `day_key` buckets them into the provider's
    own quota day. Both are required, and they are not derivable from each
    other without knowing the provider's reset timezone — which is exactly the
    thing that differs between Groq and Gemini.
    """
    moment = now or datetime.now(timezone.utc)
    return UsageEvent(
        run_id=run_id,
        node_id=node_id,
        purpose=purpose,
        provider=provider,
        model_id=model_id,
        ts_utc=moment.isoformat(),
        day_key=day_key(moment, reset_tz),
        est_prompt_tokens=est_prompt_tokens,
        est_completion_tokens=est_completion_tokens,
        usage=usage,
        cost_usd=cost_usd,
        ok=ok,
        http_status=http_status,
        latency_ms=latency_ms,
        error=error,
    )


@contextmanager
def open_store(path: Path | str | None = None):
    """Context-managed store, closed on exit."""
    store = LedgerStore(path)
    try:
        yield store
    finally:
        store.close()
