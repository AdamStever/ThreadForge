"""Tests for the detection layer."""

from threadforge.signals import Volatility
from threadforge.detection import Calibrator, Detector


def _make_stream():
    """Calm baseline, then a burst of volatility, then calm again."""
    stream = []
    t = 0
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
    detector = Detector(
        signal=Volatility(window_size=10),
        calibrator=Calibrator(multiplier=4.0),
        calib_steps=63,  # first 63 of 126 points = calm baseline
        gap_steps=20,
    )
    events = detector.run(stream)
    assert len(events) >= 1  # the volatile burst should be caught


def test_detector_quiet_on_calm_stream():
    calm = [(f"2024-01-01 00:{i:02d}:00", 50.0 + (1 if i % 2 else -1))
            for i in range(120)]
    detector = Detector(
        signal=Volatility(window_size=10),
        calibrator=Calibrator(multiplier=4.0),
        calib_steps=18,  # first 18 of 120 calm points
        gap_steps=20,
    )
    events = detector.run(calm)
    assert events == []  # no false alarms on steady data
