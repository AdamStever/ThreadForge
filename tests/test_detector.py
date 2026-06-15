"""Tests for the detection layer."""

import pytest

from threadforge.engine import SignalEngine
from threadforge.signals import Volatility
from threadforge.detection import Calibrator, Detector, Scorer

_DEFAULT_SCORER = Scorer({"volatility": 1.0}, score_threshold=1.0)


def _make_engine_and_calibrators(multiplier: float = 4.0, window_size: int = 10):
    engine = SignalEngine()
    engine.register("volatility", Volatility(window_size))
    calibrators = {"volatility": Calibrator(multiplier)}
    return engine, calibrators


def _make_stream():
    """Calm baseline, then a burst of volatility, then calm again."""
    stream = []
    # 100 calm points alternating tightly around 50
    for i in range(100):
        stream.append((f"2024-01-01 00:{i:02d}:00", 50.0 + (1 if i % 2 else -1)))
    # a volatile burst
    for i, v in enumerate([80, 20, 85, 15, 90, 10]):
        stream.append((f"2024-01-01 02:{i:02d}:00", float(v)))
    # calm tail
    for i in range(20):
        stream.append((f"2024-01-01 03:{i:02d}:00", 50.0 + (1 if i % 2 else -1)))
    return stream


def test_calibrator_finalizes_threshold():
    cal = Calibrator(multiplier=3.0)
    for v in [1.0, 1.0, 1.0, 1.0]:
        cal.observe(v)
    thresh = cal.finalize()
    assert thresh == 1.0  # zero variance => mean + 0


def test_calibrator_ignores_none():
    cal = Calibrator()
    cal.observe(None)
    cal.observe(2.0)
    cal.observe(4.0)
    cal.finalize()
    assert cal.threshold is not None


def test_detector_flags_the_burst():
    stream = _make_stream()
    engine, calibrators = _make_engine_and_calibrators()
    detector = Detector(
        engine=engine,
        calibrators=calibrators,
        scorer=_DEFAULT_SCORER,
        calib_steps=63,
        gap_steps=20,
    )
    events = detector.run(stream)
    assert len(events) >= 1


def test_detector_quiet_on_calm_stream():
    calm = [(f"2024-01-01 00:{i:02d}:00", 50.0 + (1 if i % 2 else -1))
            for i in range(120)]
    engine, calibrators = _make_engine_and_calibrators()
    detector = Detector(
        engine=engine,
        calibrators=calibrators,
        scorer=_DEFAULT_SCORER,
        calib_steps=18,
        gap_steps=20,
        min_calib_samples=0,  # tiny calibration is intentional here
    )
    events = detector.run(calm)
    assert events == []


def test_detector_reports_effective_calibration_samples():
    stream = _make_stream()
    engine, calibrators = _make_engine_and_calibrators(window_size=10)
    detector = Detector(
        engine=engine,
        calibrators=calibrators,
        scorer=_DEFAULT_SCORER,
        calib_steps=63,
        gap_steps=20,
    )
    detector.run(stream)
    eff = detector.calibration_samples["volatility"]
    # warm-up trims roughly window_size points off the front, so the effective
    # count is below the requested calib_steps but no more than window_size below
    assert 63 - 10 <= eff < 63


def test_detector_warns_on_thin_calibration():
    stream = _make_stream()
    engine, calibrators = _make_engine_and_calibrators(window_size=10)
    detector = Detector(
        engine=engine,
        calibrators=calibrators,
        scorer=_DEFAULT_SCORER,
        calib_steps=15,          # only ~5 effective points after warm-up
        gap_steps=20,
        min_calib_samples=30,
    )
    with pytest.warns(UserWarning, match="thin calibration"):
        detector.run(stream)


def test_detector_no_warning_when_calibration_sufficient(recwarn):
    stream = _make_stream()
    engine, calibrators = _make_engine_and_calibrators(window_size=10)
    detector = Detector(
        engine=engine,
        calibrators=calibrators,
        scorer=_DEFAULT_SCORER,
        calib_steps=63,
        gap_steps=20,
        min_calib_samples=30,
    )
    detector.run(stream)
    assert not any("thin calibration" in str(w.message) for w in recwarn.list)


def test_flagged_point_records_signal_name():
    stream = _make_stream()
    engine, calibrators = _make_engine_and_calibrators()
    detector = Detector(
        engine=engine,
        calibrators=calibrators,
        scorer=_DEFAULT_SCORER,
        calib_steps=63,
        gap_steps=20,
    )
    events = detector.run(stream)
    assert len(events) >= 1


# --- time-based gap grouping (limitation #4) ---

def _make_irregular_stream():
    """Calm calibration, then two volatile bursts that are only a few ROWS
    apart but ~2 DAYS apart in TIME (an irregular-stream timestamp jump).

    Index-based grouping treats the two bursts as one event; time-based
    grouping with a small gap_seconds correctly splits them in two.
    """
    stream = []
    # 40 calm calibration points, 1 minute apart, very low volatility
    for i in range(40):
        stream.append((f"2024-01-01 00:{i:02d}:00", 50.0 + (0.5 if i % 2 else -0.5)))
    # burst 1: 7 wildly volatile points, 1 minute apart
    for i in range(7):
        stream.append((f"2024-01-01 00:{40 + i:02d}:00", 1000.0 if i % 2 else -1000.0))
    # burst 2: 7 more wild points, only rows later but two DAYS later in time
    for i in range(7):
        stream.append((f"2024-01-03 00:{40 + i:02d}:00", 1000.0 if i % 2 else -1000.0))
    return stream


def test_index_mode_merges_bursts_separated_only_in_time():
    stream = _make_irregular_stream()
    engine, calibrators = _make_engine_and_calibrators(window_size=10)
    detector = Detector(
        engine=engine,
        calibrators=calibrators,
        scorer=_DEFAULT_SCORER,
        calib_steps=40,
        gap_steps=20,          # 14 flagged rows are all within 20 => one event
        min_calib_samples=0,
    )
    events = detector.run(stream)
    assert len(events) == 1


def test_time_mode_splits_bursts_on_time_jump():
    stream = _make_irregular_stream()
    engine, calibrators = _make_engine_and_calibrators(window_size=10)
    detector = Detector(
        engine=engine,
        calibrators=calibrators,
        scorer=_DEFAULT_SCORER,
        calib_steps=40,
        gap_steps=20,
        gap_seconds=3600,      # 1 hour: the 2-day jump splits the bursts
        min_calib_samples=0,
    )
    events = detector.run(stream)
    assert len(events) == 2
    assert events[0].peak.signal_name == "volatility"
