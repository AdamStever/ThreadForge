"""Tests for the AdaptiveRobustCalibrator (rolling, re-baselining)."""

import numpy as np
import pytest

from threadforge.detection import AdaptiveRobustCalibrator, RobustCalibrator


def test_requires_finalize_before_use():
    with pytest.raises(RuntimeError):
        AdaptiveRobustCalibrator().is_anomalous(1.0)


def test_no_values_raises():
    with pytest.raises(ValueError):
        AdaptiveRobustCalibrator().finalize()


def test_window_caps_sample_size():
    c = AdaptiveRobustCalibrator(window=30)
    for v in range(100):
        c.observe(float(v))
    assert c.sample_size == 30


def test_flags_spike_against_rolling_band():
    c = AdaptiveRobustCalibrator(multiplier=3.0, window=50)
    rng = np.random.RandomState(1)
    for v in rng.normal(0, 1, 100):
        c.observe(float(v))
    c.finalize()
    assert c.is_anomalous(50.0) is True     # huge spike vs ~N(0,1)
    assert c.is_anomalous(0.0) is False      # squarely normal


def test_rebaselines_after_level_shift_unlike_frozen():
    """The adaptive band re-learns a sustained new level; the frozen one never does."""
    rng = np.random.RandomState(0)
    calib = rng.normal(0.0, 0.1, 100)
    shifted = rng.normal(10.0, 0.1, 100)     # a sustained shift to a new normal level

    adaptive = AdaptiveRobustCalibrator(multiplier=3.0, window=50)
    frozen = RobustCalibrator(multiplier=3.0)
    for v in calib:
        adaptive.observe(float(v))
        frozen.observe(float(v))
    adaptive.finalize()
    frozen.finalize()

    a_flags = [adaptive.is_anomalous(float(v)) for v in shifted]
    f_flags = [frozen.is_anomalous(float(v)) for v in shifted]

    # frozen keeps flagging the new level forever (band stuck at the old normal)
    assert sum(f_flags[-20:]) == 20
    # adaptive catches the onset of the shift...
    assert sum(a_flags[:10]) >= 5
    # ...then re-learns the new level as normal
    assert sum(a_flags[-20:]) <= 2


def test_drop_in_interface_matches_robust():
    """Same observe/finalize/is_anomalous/threshold/sample_size surface as RobustCalibrator."""
    for name in ("observe", "finalize", "is_anomalous", "threshold", "sample_size"):
        assert hasattr(AdaptiveRobustCalibrator(), name)
