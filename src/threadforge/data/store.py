"""SQLite feature store.

Persists raw stream values and computed signal scores to a local SQLite
database as the pipeline runs. The schema is intentionally simple and
append-only — rows are never updated or deleted here.

Tables
------
runs            one row per pipeline execution (run_id, source, started_at)
stream_values   one row per (run_id, timestamp, channel, value) from raw input
signal_scores   one row per (run_id, timestamp, channel, signal_name, value)
                from the SignalEngine

A ``run_id`` groups all rows from a single pipeline execution so multiple
files can share one database without collisions.

MULTIVARIATE READINESS
----------------------
The pipeline today is univariate: one (timestamp, value) series per run, so
every row carries the same ``channel`` (DEFAULT_CHANNEL = "value"). The schema,
however, keys on ``channel`` deliberately. When the multivariate path arrives
(many correlated series per timestamp, e.g. the Server Machine Dataset), each
series is written under its own channel name and the rows coexist without any
schema migration — the storage layer is general even though the current engine
and detector are not. This is a persistence concern only; it adds no
multivariate behaviour to signals/, detection/, or engine.py.

Usage
-----
    store = FeatureStore("threadforge.db")
    with store:
        store.begin_run("ec2_cpu_utilization_5f5533.csv")
        # univariate (channel defaults to "value")
        store.write_stream_value(ts, raw_val)
        store.write_signal_scores(ts, {"volatility": 1.23, "zscore": 0.45})
        # multivariate (explicit channel per series) — future use
        store.write_stream_value(ts, cpu_val,  channel="cpu")
        store.write_stream_value(ts, mem_val,  channel="mem")
    # context manager commits and closes on exit
"""

import sqlite3
from pathlib import Path


# Channel name used when the caller doesn't specify one. Univariate streams
# write everything under this single channel.
DEFAULT_CHANNEL = "value"


_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT    NOT NULL,
    started_at  TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

_CREATE_STREAM = """
CREATE TABLE IF NOT EXISTS stream_values (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    timestamp   TEXT    NOT NULL,
    channel     TEXT    NOT NULL DEFAULT 'value',
    value       REAL    NOT NULL
)
"""

_CREATE_SIGNALS = """
CREATE TABLE IF NOT EXISTS signal_scores (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    timestamp   TEXT    NOT NULL,
    channel     TEXT    NOT NULL DEFAULT 'value',
    signal_name TEXT    NOT NULL,
    value       REAL
)
"""

# Composite indexes keyed on channel so per-series time queries stay fast
# whether there is one channel (univariate) or many (multivariate).
_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_stream_run_chan_ts "
    "ON stream_values (run_id, channel, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_signal_run_chan_ts "
    "ON signal_scores (run_id, channel, timestamp)",
]


class FeatureStore:
    def __init__(self, db_path: str | Path):
        self._path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._run_id: int | None = None

    # --- context manager ---

    def __enter__(self) -> "FeatureStore":
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_RUNS)
        self._conn.execute(_CREATE_STREAM)
        self._conn.execute(_CREATE_SIGNALS)
        for stmt in _CREATE_INDEXES:
            self._conn.execute(stmt)
        self._conn.commit()
        return self

    def __exit__(self, *_) -> None:
        if self._conn:
            self._conn.commit()
            self._conn.close()
            self._conn = None
        self._run_id = None

    # --- run management ---

    def begin_run(self, source: str) -> int:
        """Register a new pipeline run and return its run_id."""
        cur = self._conn.execute(
            "INSERT INTO runs (source) VALUES (?)", (source,)
        )
        self._conn.commit()
        self._run_id = cur.lastrowid
        return self._run_id

    @property
    def run_id(self) -> int | None:
        return self._run_id

    # --- write helpers ---

    def write_stream_value(
        self, timestamp: str, value: float, channel: str = DEFAULT_CHANNEL
    ) -> None:
        self._conn.execute(
            "INSERT INTO stream_values (run_id, timestamp, channel, value) "
            "VALUES (?, ?, ?, ?)",
            (self._run_id, timestamp, channel, value),
        )

    def write_signal_scores(
        self,
        timestamp: str,
        scores: dict[str, float | None],
        channel: str = DEFAULT_CHANNEL,
    ) -> None:
        self._conn.executemany(
            "INSERT INTO signal_scores (run_id, timestamp, channel, signal_name, value) "
            "VALUES (?, ?, ?, ?, ?)",
            [(self._run_id, timestamp, channel, name, val) for name, val in scores.items()],
        )
