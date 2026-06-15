"""Tests for statistical signals and robust calibration."""

import math
import pytest

from threadforge.signals import ZScore, EntropyFine, EntropyCoarse
from threadforge.detection import RobustCalibrator


# --- ZScore ---

def test_zscore_zero_for_constant_window():
    sig = ZScore(window_size=4)
    for _ in range(4):
        out = sig.update(5.0)
    assert out == 0.0


def test_zscore_positive_when_current_above_mean():
    sig = ZScore(window_size=4)
    sig.update(0.0)
    sig.update(0.0)
    sig.update(0.0)
    out = sig.update(3.0)  # mean=0.75, std>0, z > 0
    assert out > 0.0


def test_zscore_negative_when_current_below_mean():
    sig = ZScore(window_size=4)
    sig.update(10.0)
    sig.update(10.0)
    sig.update(10.0)
    out = sig.update(0.0)  # current below mean => negative z
    assert out < 0.0


def test_zscore_unbounded():
    # extreme outlier should produce a large z-score well above 1
    sig = ZScore(window_size=10)
    for _ in range(9):
        sig.update(0.0)
    out = sig.update(100.0)
    assert out > 2.0


# --- EntropyFine / EntropyCoarse ---

def test_entropy_fine_uses_16_bins():
    sig = EntropyFine(window_size=4)
    assert sig.bins == 16


def test_entropy_coarse_uses_4_bins():
    sig = EntropyCoarse(window_size=4)
    assert sig.bins == 4


def test_entropy_variants_zero_for_constant_window():
    for cls in (EntropyFine, EntropyCoarse):
        sig = cls(window_size=4)
        for _ in range(4):
            out = sig.update(5.0)
        assert out == 0.0


# --- RobustCalibrator ---

def test_robust_calibrator_sets_upper_and_lower():
    cal = RobustCalibrator(multiplier=1.5)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        cal.observe(v)
    cal.finalize()
    assert cal.upper is not None
    assert cal.lower is not None
    assert cal.upper > cal.lower


def test_robust_calibrator_not_anomalous_within_band():
    cal = RobustCalibrator(multiplier=3.0)
    for v in [10.0] * 20:
        cal.observe(v)
    cal.finalize()
    assert not cal.is_anomalous(10.0)


def test_robust_calibrator_flags_high_outlier():
    cal = RobustCalibrator(multiplier=1.5)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        cal.observe(v)
    cal.finalize()
    assert cal.is_anomalous(100.0)


def test_robust_calibrator_flags_low_outlier():
    cal = RobustCalibrator(multiplier=1.5)
    for v in [10.0, 11.0, 12.0, 13.0, 14.0]:
        cal.observe(v)
    cal.finalize()
    assert cal.is_anomalous(-100.0)


def test_robust_calibrator_ignores_none():
    cal = RobustCalibrator()
    cal.observe(None)
    cal.observe(5.0)
    cal.observe(6.0)
    cal.finalize()
    assert cal.threshold is not None


def test_robust_calibrator_raises_before_finalize():
    cal = RobustCalibrator()
    cal.observe(1.0)
    with pytest.raises(RuntimeError):
        cal.is_anomalous(1.0)


def test_robust_calibrator_sample_size_counts_real_values():
    cal = RobustCalibrator()
    for v in [1.0, 2.0, 3.0]:
        cal.observe(v)
    assert cal.sample_size == 3


def test_robust_calibrator_sample_size_excludes_none():
    cal = RobustCalibrator()
    cal.observe(None)
    cal.observe(5.0)
    cal.observe(None)
    cal.observe(6.0)
    assert cal.sample_size == 2
