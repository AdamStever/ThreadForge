"""Tests for the signal layer.

Run with:  pytest   (from the project root)
"""

import math

from threadforge.signals import Momentum, Volatility, Entropy, Sharpness, Acceleration


def test_signal_returns_none_until_window_fills():
    sig = Momentum(window_size=3)
    assert sig.update(1.0) is None
    assert sig.update(2.0) is None
    assert sig.update(3.0) is not None  # window now full


def test_momentum_is_per_step_change():
    sig = Momentum(window_size=3)
    sig.update(10.0)
    sig.update(20.0)
    result = sig.update(30.0)  # window [10, 20, 30]
    assert result == (30.0 - 10.0) / 2  # == 10.0


def test_volatility_zero_for_constant_window():
    sig = Volatility(window_size=4)
    for _ in range(4):
        out = sig.update(5.0)
    assert out == 0.0


def test_volatility_known_value():
    sig = Volatility(window_size=3)
    sig.update(2.0)
    sig.update(4.0)
    out = sig.update(6.0)  # window [2, 4, 6], sample std = 2.0
    assert math.isclose(out, 2.0, rel_tol=1e-9)


def test_entropy_zero_for_constant_window():
    sig = Entropy(window_size=5, bins=4)
    for _ in range(5):
        out = sig.update(7.0)
    assert out == 0.0


def test_entropy_positive_for_varied_window():
    sig = Entropy(window_size=4, bins=4)
    sig.update(1.0)
    sig.update(2.0)
    sig.update(3.0)
    out = sig.update(4.0)  # spread across buckets => entropy > 0
    assert out > 0.0


def test_reset_clears_window():
    sig = Momentum(window_size=2)
    sig.update(1.0)
    sig.reset()
    assert sig.update(9.0) is None  # window emptied, needs to refill


# --- Sharpness ---

def test_sharpness_zero_for_constant_window():
    sig = Sharpness(window_size=4)
    for _ in range(4):
        out = sig.update(5.0)
    assert out == 0.0


def test_sharpness_positive_when_current_above_mean():
    sig = Sharpness(window_size=4)
    sig.update(1.0)
    sig.update(1.0)
    sig.update(1.0)
    out = sig.update(5.0)  # window [1, 1, 1, 5]: mean=2, spread=4, sharpness=(5-2)/4=0.75
    assert abs(out - 0.75) < 1e-9


def test_sharpness_negative_when_current_below_mean():
    sig = Sharpness(window_size=4)
    sig.update(5.0)
    sig.update(5.0)
    sig.update(5.0)
    out = sig.update(1.0)  # window [5, 5, 5, 1]: mean=4, spread=4, sharpness=(1-4)/4=-0.75
    assert abs(out - (-0.75)) < 1e-9


# --- Acceleration ---

def test_acceleration_requires_window_size_3():
    import pytest
    with pytest.raises(ValueError):
        Acceleration(window_size=2)


def test_acceleration_zero_for_constant_velocity():
    sig = Acceleration(window_size=3)
    sig.update(10.0)
    sig.update(20.0)
    out = sig.update(30.0)  # linear ramp => zero second difference
    assert out == 0.0


def test_acceleration_positive_when_speeding_up():
    sig = Acceleration(window_size=3)
    sig.update(0.0)
    sig.update(1.0)
    out = sig.update(4.0)  # gaps: 1, 3 => second diff = 2 => accel = 2
    assert out > 0.0


def test_acceleration_negative_when_slowing_down():
    sig = Acceleration(window_size=3)
    sig.update(0.0)
    sig.update(3.0)
    out = sig.update(4.0)  # gaps: 3, 1 => second diff = -2 => accel = -2
    assert out < 0.0
