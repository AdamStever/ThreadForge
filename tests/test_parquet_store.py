"""Tests for the columnar (Parquet) feature store.

Covers the read/write round-trip, run/channel isolation, batch-flush behaviour,
and that it is a drop-in replacement for the SQLite store in the Detector.
"""

import tempfile
from pathlib import Path

import pytest

from threadforge.data import ParquetFeatureStore
from threadforge.engine import SignalEngine
from threadforge.signals import Volatility, Momentum
from threadforge.detection import RobustCalibrator, Detector, Scorer


def _tmp_root() -> str:
    return tempfile.mkdtemp(prefix="tf_parquet_")


def test_round_trip_stream_in_order():
    root = _tmp_root()
    with ParquetFeatureStore(root) as s:
        rid = s.begin_run("a.csv")
        s.write_stream_value("2024-01-01 00:00:00", 1.0)
        s.write_stream_value("2024-01-01 00:01:00", 2.0)
        s.write_stream_value("2024-01-01 00:02:00", 3.0)
    # data is only on disk after the context exits (batch flush)
    with ParquetFeatureStore(root) as s:
        assert s.read_stream(rid) == [
            ("2024-01-01 00:00:00", 1.0),
            ("2024-01-01 00:01:00", 2.0),
            ("2024-01-01 00:02:00", 3.0),
        ]


def test_run_ids_increment_and_persist():
    root = _tmp_root()
    with ParquetFeatureStore(root) as s:
        r1 = s.begin_run("a.csv")
        s.write_stream_value("t", 1.0)
        r2 = s.begin_run("b.csv")  # begin_run flushes the previous run
        s.write_stream_value("t", 2.0)
    assert (r1, r2) == (1, 2)
    # a fresh session keeps numbering from the persisted runs
    with ParquetFeatureStore(root) as s:
        r3 = s.begin_run("c.csv")
    assert r3 == 3


def test_read_signal_preserves_nulls():
    root = _tmp_root()
    with ParquetFeatureStore(root) as s:
        rid = s.begin_run("a.csv")
        s.write_signal_scores("t1", {"zscore": None})
        s.write_signal_scores("t2", {"zscore": 1.5})
    with ParquetFeatureStore(root) as s:
        assert s.read_signal(rid, "zscore") == [("t1", None), ("t2", 1.5)]


def test_channel_isolation():
    root = _tmp_root()
    with ParquetFeatureStore(root) as s:
        rid = s.begin_run("smd.csv")
        s.write_stream_value("t", 0.9, channel="cpu")
        s.write_stream_value("t", 0.4, channel="mem")
        s.write_signal_scores("t", {"volatility": 1.1}, channel="cpu")
    with ParquetFeatureStore(root) as s:
        assert s.read_stream(rid, channel="cpu") == [("t", 0.9)]
        assert s.read_stream(rid, channel="mem") == [("t", 0.4)]
        assert s.channels(rid) == ["cpu", "mem"]
        assert s.signal_names(rid, channel="cpu") == ["volatility"]


def test_list_and_summarize_run():
    root = _tmp_root()
    with ParquetFeatureStore(root) as s:
        rid = s.begin_run("a.csv")
        s.write_stream_value("2024-01-01 00:00:00", 1.0)
        s.write_stream_value("2024-01-01 00:05:00", 2.0)
        s.write_signal_scores("2024-01-01 00:00:00", {"volatility": 0.1})
    with ParquetFeatureStore(root) as s:
        runs = s.list_runs()
        assert [r["source"] for r in runs] == ["a.csv"]
        summary = s.summarize_run(rid)
    assert summary["stream_points"] == 2
    assert summary["signal_rows"] == 1
    assert summary["signals"] == ["volatility"]
    assert summary["time_start"] == "2024-01-01 00:00:00"
    assert summary["time_end"] == "2024-01-01 00:05:00"


def test_summarize_unknown_run_raises():
    root = _tmp_root()
    with ParquetFeatureStore(root) as s:
        s.begin_run("a.csv")
        s.write_stream_value("t", 1.0)
    with ParquetFeatureStore(root) as s:
        with pytest.raises(KeyError):
            s.summarize_run(999)


def test_drop_in_replacement_for_detector():
    # the Detector should write to a Parquet store exactly as it does to SQLite
    stream = [(f"2024-01-01 00:{i:02d}:00", 50.0 + (i % 5)) for i in range(60)]
    engine = SignalEngine()
    engine.register("volatility", Volatility(10))
    engine.register("momentum", Momentum(10))
    calibrators = {n: RobustCalibrator(3.0) for n in engine._signals}
    scorer = Scorer({"volatility": 1.0}, score_threshold=1.0)

    root = _tmp_root()
    with ParquetFeatureStore(root) as store:
        rid = store.begin_run("synthetic.csv")
        detector = Detector(
            engine=engine, calibrators=calibrators, scorer=scorer,
            calib_steps=30, gap_steps=20, store=store, min_calib_samples=0,
        )
        detector.run(stream)

    with ParquetFeatureStore(root) as store:
        # detection phase ran for the 30 post-calibration steps
        assert len(store.read_stream(rid)) == 30
        assert set(store.signal_names(rid)) == {"volatility", "momentum"}
        # each detection step recorded a value for each signal
        assert len(store.read_signal(rid, "volatility")) == 30
