"""Tests for the detection layer."""

from threadforge.engine import SignalEngine
from threadforge.signals import Volatility
from threadforge.detection import Calibrator, Detector


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
        calib_steps=63,  # calm first half calibrates "normal"
        gap_steps=20,
    )
    events = detector.run(stream)
    assert len(events) >= 1  # the volatile burst should be caught


def test_detector_quiet_on_calm_stream():
    calm = [(f"2024-01-01 00:{i:02d}:00", 50.0 + (1 if i % 2 else -1))
            for i in range(120)]
    engine, calibrators = _make_engine_and_calibrators()
    detector = Detector(
        engine=engine,
        calibrators=calibrators,
        calib_steps=18,
        gap_steps=20,
    )
    events = detector.run(calm)
    assert events == []


def test_flagged_point_records_signal_name():
    stream = _make_stream()
    engine, calibrators = _make_engine_and_calibrators()
    detector = Detector(engine=engine, calibrators=calibrators, calib_steps=63, gap_steps=20)
    events = detector.run(stream)
    assert len(events) >= 1
    assert events[0].peak.signal_name == "volatility"
