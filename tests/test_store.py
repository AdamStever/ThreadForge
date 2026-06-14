"""Tests for the SQLite feature store."""

import sqlite3
import tempfile
from pathlib import Path

from threadforge.data import FeatureStore


def _tmp_db():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return f.name


def test_store_creates_tables():
    path = _tmp_db()
    with FeatureStore(path):
        pass
    conn = sqlite3.connect(path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert {"runs", "stream_values", "signal_scores"} <= tables


def test_begin_run_returns_id():
    path = _tmp_db()
    with FeatureStore(path) as s:
        run_id = s.begin_run("test.csv")
    assert isinstance(run_id, int)
    assert run_id >= 1


def test_write_stream_value_persisted():
    path = _tmp_db()
    with FeatureStore(path) as s:
        s.begin_run("test.csv")
        s.write_stream_value("2024-01-01 00:00:00", 42.5)

    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT timestamp, value FROM stream_values").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0] == ("2024-01-01 00:00:00", 42.5)


def test_write_signal_scores_persisted():
    path = _tmp_db()
    with FeatureStore(path) as s:
        s.begin_run("test.csv")
        s.write_signal_scores("2024-01-01 00:00:00", {"volatility": 1.23, "zscore": 0.45})

    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT signal_name, value FROM signal_scores ORDER BY signal_name"
    ).fetchall()
    conn.close()
    assert ("volatility", 1.23) in rows
    assert ("zscore", 0.45) in rows


def test_multiple_runs_isolated_by_run_id():
    path = _tmp_db()
    with FeatureStore(path) as s:
        id1 = s.begin_run("file_a.csv")
        s.write_stream_value("2024-01-01 00:00:00", 1.0)
        id2 = s.begin_run("file_b.csv")
        s.write_stream_value("2024-01-01 00:01:00", 2.0)

    assert id1 != id2
    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT run_id, value FROM stream_values ORDER BY run_id").fetchall()
    conn.close()
    assert rows[0][0] == id1 and rows[0][1] == 1.0
    assert rows[1][0] == id2 and rows[1][1] == 2.0


def test_none_signal_value_stored_as_null():
    path = _tmp_db()
    with FeatureStore(path) as s:
        s.begin_run("test.csv")
        s.write_signal_scores("2024-01-01 00:00:00", {"momentum": None})

    conn = sqlite3.connect(path)
    row = conn.execute("SELECT value FROM signal_scores WHERE signal_name='momentum'").fetchone()
    conn.close()
    assert row[0] is None
