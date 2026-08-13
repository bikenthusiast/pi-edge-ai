"""Tests for the SQLite event log. No hardware, no network, no model."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from edge.events.store import Event, EventStore, open_store


@pytest.fixture
def store(tmp_path) -> EventStore:
    return EventStore(tmp_path / "events.db")


def test_record_returns_increasing_ids(store):
    first = store.record("person", 0.91)
    second = store.record("cat", 0.77)
    assert second > first


def test_roundtrip_preserves_fields(store):
    store.record("person", 0.91, source="camera1", duration_ms=1200.5,
                 note="front door")
    (event,) = store.query()

    assert isinstance(event, Event)
    assert event.label == "person"
    assert event.score == pytest.approx(0.91)
    assert event.source == "camera1"
    assert event.duration_ms == pytest.approx(1200.5)
    assert event.note == "front door"


def test_timestamps_come_back_timezone_aware(store):
    """Naive datetimes are a classic source of off-by-hours bugs."""
    store.record("person", 0.5)
    (event,) = store.query()
    assert event.ts.tzinfo is not None
    assert event.ts.utcoffset() == timedelta(0)


def test_naive_input_is_treated_as_utc(store):
    naive = datetime(2026, 8, 12, 9, 0, 0)  # noqa: DTZ001 - that is the point
    store.record("person", 0.5, ts=naive)
    (event,) = store.query()
    assert event.ts == naive.replace(tzinfo=UTC)


def test_local_time_is_converted_not_truncated(store):
    berlin = timezone(timedelta(hours=2))
    store.record("person", 0.5, ts=datetime(2026, 8, 12, 11, 0, tzinfo=berlin))
    (event,) = store.query()
    assert event.ts.hour == 9  # 11:00+02:00 == 09:00Z


@pytest.mark.parametrize("bad", [-0.1, 1.1, 42.0])
def test_invalid_scores_are_rejected(store, bad):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        store.record("person", bad)


# --- querying -------------------------------------------------------------- #

@pytest.fixture
def populated(store) -> EventStore:
    base = datetime(2026, 8, 12, 8, 0, tzinfo=UTC)
    for minutes, label, score in [
        (0, "person", 0.95),
        (30, "person", 0.88),
        (90, "cat", 0.61),
        (240, "person", 0.42),
        (300, "dog", 0.79),
    ]:
        store.record(label, score, ts=base + timedelta(minutes=minutes))
    return store


def test_query_returns_newest_first(populated):
    events = populated.query()
    timestamps = [e.ts for e in events]
    assert timestamps == sorted(timestamps, reverse=True)


def test_time_window_is_half_open(populated):
    """since is inclusive, until exclusive — so adjacent windows don't
    double-count the boundary event."""
    base = datetime(2026, 8, 12, 8, 0, tzinfo=UTC)
    early = populated.query(since=base, until=base + timedelta(hours=2))
    late = populated.query(since=base + timedelta(hours=2))

    assert len(early) == 3
    assert len(late) == 2
    assert len(early) + len(late) == 5


def test_label_and_score_filters_combine(populated):
    events = populated.query(label="person", min_score=0.5)
    assert [e.score for e in events] == pytest.approx([0.88, 0.95], rel=1e-6)


def test_limit_caps_results(populated):
    assert len(populated.query(limit=2)) == 2


def test_summarise_aggregates_per_label(populated):
    summary = {row["label"]: row for row in populated.summarise()}

    assert summary["person"]["n"] == 3
    assert summary["cat"]["n"] == 1
    assert summary["person"]["avg_score"] == pytest.approx(0.75, abs=0.01)
    # ordered by count, descending
    assert populated.summarise()[0]["label"] == "person"


def test_summarise_respects_the_window(populated):
    base = datetime(2026, 8, 12, 8, 0, tzinfo=UTC)
    rows = populated.summarise(since=base + timedelta(hours=3))
    assert {row["label"] for row in rows} == {"person", "dog"}


# --- housekeeping ---------------------------------------------------------- #

def test_prune_deletes_only_old_rows(store):
    now = datetime.now(UTC)
    store.record("old", 0.5, ts=now - timedelta(days=40))
    store.record("new", 0.5, ts=now - timedelta(days=1))

    assert store.prune(older_than_days=30) == 1
    assert [e.label for e in store.query()] == ["new"]


def test_schema_survives_reopening(tmp_path):
    path = tmp_path / "events.db"
    with open_store(path) as first:
        first.record("person", 0.9)
    with open_store(path) as second:
        assert len(second.query()) == 1


def test_wal_mode_is_active(store):
    mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_creates_parent_directory(tmp_path):
    store = EventStore(tmp_path / "nested" / "deeper" / "events.db")
    store.record("person", 0.5)
    assert store.path.is_file()


# --- model provenance ------------------------------------------------------ #

def test_model_is_stored_and_returned(store):
    store.record("person", 0.9, model="mobilenet_v2_1.0_224_quant.tflite")
    (event,) = store.query()
    assert event.model == "mobilenet_v2_1.0_224_quant.tflite"


def test_model_defaults_to_none(store):
    store.record("person", 0.9)
    (event,) = store.query()
    assert event.model is None


def test_query_can_filter_by_model(store):
    store.record("person", 0.9, model="v2.tflite")
    store.record("person", 0.5, model="v1_tiny.tflite")
    assert len(store.query(model="v2.tflite")) == 1


def test_models_used_groups_and_orders_by_recency(store):
    now = datetime.now(UTC)
    store.record("person", 0.9, model="v1_tiny.tflite", ts=now - timedelta(hours=3))
    store.record("person", 0.9, model="v2.tflite", ts=now - timedelta(hours=1))
    store.record("cat", 0.8, model="v2.tflite", ts=now)

    rows = store.models_used()
    assert [row["model"] for row in rows] == ["v2.tflite", "v1_tiny.tflite"]
    assert rows[0]["n"] == 2


def test_models_used_labels_rows_from_before_the_column_existed(store):
    store.record("person", 0.9)                      # no model given
    (row,) = store.models_used()
    assert row["model"] == "(unrecorded)"


def test_migration_adds_the_column_to_an_old_database(tmp_path):
    """A database created before the model column must keep working."""
    import sqlite3

    path = tmp_path / "old.db"
    legacy = sqlite3.connect(path)
    legacy.executescript("""
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, label TEXT NOT NULL, score REAL NOT NULL,
            source TEXT NOT NULL DEFAULT 'camera0',
            duration_ms REAL, note TEXT
        );
        INSERT INTO events (ts, label, score, source)
        VALUES ('2026-08-01T10:00:00.000+00:00', 'person', 0.9, 'camera0');
    """)
    legacy.commit()
    legacy.close()

    store = EventStore(path)                          # runs the migration
    (event,) = store.query()
    assert event.label == "person"
    assert event.model is None

    store.record("cat", 0.7, model="v2.tflite")
    assert len(store.query(model="v2.tflite")) == 1


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "events.db"
    EventStore(path).record("person", 0.9, model="v2.tflite")
    reopened = EventStore(path)                       # must not raise
    assert len(reopened.query()) == 1