"""SQLite event log.

Device-independent: pure stdlib, no camera, no model. This is what the agent
will later query, so the schema matters more than the code around it.

Design notes
------------
* One row per detection, never per frame. A 15 FPS pipeline writing every frame
  would produce 1.3 M rows/day on an SD card; the pipeline debounces first.
* Timestamps are stored as ISO-8601 UTC strings. SQLite has no date type, and
  text sorts correctly in this format — unlike Unix floats, it stays readable
  when a human (or an LLM) reads the table directly.
* WAL mode so a reader (the agent) never blocks the writer (the vision loop).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,          -- ISO-8601 UTC
    label       TEXT    NOT NULL,
    score       REAL    NOT NULL,
    source      TEXT    NOT NULL DEFAULT 'camera0',
    duration_ms REAL,                      -- how long the detection persisted
    model       TEXT,                      -- which weights produced this row
    note        TEXT
);

-- The agent's most common query is "what happened between X and Y",
-- so ts leads every index.
CREATE INDEX IF NOT EXISTS idx_events_ts        ON events (ts);
CREATE INDEX IF NOT EXISTS idx_events_label_ts  ON events (label, ts);
"""

# Columns added after the first release. SQLite has no "ADD COLUMN IF NOT
# EXISTS", so each migration is guarded by a lookup in pragma table_info.
MIGRATIONS: dict[str, str] = {
    "model": "ALTER TABLE events ADD COLUMN model TEXT",
}


@dataclass(frozen=True)
class Event:
    """One detection, as read back from the database."""

    id: int
    ts: datetime
    label: str
    score: float
    source: str
    duration_ms: float | None
    model: str | None
    note: str | None


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _to_iso(moment: datetime) -> str:
    """Normalise to UTC ISO-8601 with a trailing Z."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="milliseconds")


def _from_iso(text: str) -> datetime:
    return datetime.fromisoformat(text)


class EventStore:
    """Thin wrapper over a SQLite file. Safe to open once and keep."""

    def __init__(self, path: Path | str = "events.db"):
        self.path = Path(path)
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        # WAL: readers and the writer no longer block each other.
        # NORMAL: one fsync per checkpoint instead of per commit — a meaningful
        # difference in SD-card lifetime, at the cost of losing the last few
        # events on a hard power cut. Acceptable for telemetry.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns that older databases predate. Cheap and idempotent."""
        existing = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(events)")
        }
        for column, statement in MIGRATIONS.items():
            if column not in existing:
                self.conn.execute(statement)

    # -- writing ----------------------------------------------------------- #

    def record(self, label: str, score: float, *, source: str = "camera0",
               duration_ms: float | None = None, model: str | None = None,
               note: str | None = None, ts: datetime | None = None) -> int:
        """Insert one event and return its id.

        `model` is worth filling in: without it, rows from two different sets
        of weights are indistinguishable, and any accuracy comparison across a
        model swap becomes guesswork.
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {score}")

        cur = self.conn.execute(
            "INSERT INTO events (ts, label, score, source, duration_ms, model, "
            "note) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_to_iso(ts or _utcnow()), label, float(score), source,
             duration_ms, model, note),
        )
        return int(cur.lastrowid)

    # -- reading ----------------------------------------------------------- #

    def query(self, *, since: datetime | None = None,
              until: datetime | None = None, label: str | None = None,
              model: str | None = None,
              min_score: float = 0.0, limit: int = 100) -> list[Event]:
        """Fetch events, newest first. This is what the agent's tool calls."""
        sql = ["SELECT * FROM events WHERE score >= ?"]
        params: list[object] = [min_score]

        if since is not None:
            sql.append("AND ts >= ?")
            params.append(_to_iso(since))
        if until is not None:
            sql.append("AND ts < ?")
            params.append(_to_iso(until))
        if label is not None:
            sql.append("AND label = ?")
            params.append(label)
        if model is not None:
            sql.append("AND model = ?")
            params.append(model)

        sql.append("ORDER BY ts DESC, id DESC LIMIT ?")
        params.append(limit)

        rows = self.conn.execute(" ".join(sql), params).fetchall()
        return [
            Event(
                id=row["id"],
                ts=_from_iso(row["ts"]),
                label=row["label"],
                score=row["score"],
                source=row["source"],
                duration_ms=row["duration_ms"],
                model=row["model"],
                note=row["note"],
            )
            for row in rows
        ]

    def summarise(self, *, since: datetime | None = None,
                  until: datetime | None = None) -> list[dict[str, object]]:
        """Counts per label — the cheap answer to 'what did you see today'.

        Aggregating in SQL rather than shipping raw rows to the model keeps the
        prompt small and the token cost flat as the log grows.
        """
        sql = ["SELECT label, COUNT(*) AS n, AVG(score) AS avg_score,",
               "MIN(ts) AS first_seen, MAX(ts) AS last_seen,",
               "COUNT(DISTINCT model) AS n_models,",
               "MAX(model) AS model",
               "FROM events WHERE 1=1"]
        params: list[object] = []

        if since is not None:
            sql.append("AND ts >= ?")
            params.append(_to_iso(since))
        if until is not None:
            sql.append("AND ts < ?")
            params.append(_to_iso(until))

        sql.append("GROUP BY label ORDER BY n DESC")
        return [dict(row) for row in self.conn.execute(" ".join(sql), params)]

    def prune(self, older_than_days: int = 30) -> int:
        """Delete old events. An SD card is not a data lake."""
        cutoff = _to_iso(_utcnow() - timedelta(days=older_than_days))
        cur = self.conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
        return cur.rowcount

    def models_used(self, *, since: datetime | None = None) -> list[dict[str, object]]:
        """Which weights produced the rows in this log, and when.

        The first thing to check when results look different from yesterday.
        """
        sql = ["SELECT COALESCE(model, '(unrecorded)') AS model, COUNT(*) AS n,",
               "MIN(ts) AS first_seen, MAX(ts) AS last_seen FROM events"]
        params: list[object] = []
        if since is not None:
            sql.append("WHERE ts >= ?")
            params.append(_to_iso(since))
        sql.append("GROUP BY model ORDER BY last_seen DESC")
        return [dict(row) for row in self.conn.execute(" ".join(sql), params)]

    def close(self) -> None:
        self.conn.close()


@contextmanager
def open_store(path: Path | str = "events.db") -> Iterator[EventStore]:
    store = EventStore(path)
    try:
        yield store
    finally:
        store.close()