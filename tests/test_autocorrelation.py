"""Tests for the Autocorrelation signal."""

import math

import pytest

from threadforge.signals import Autocorrelation


def test_returns_none_until_window_fills():
    sig = Autocorrelation(window_size=4)
    assert sig.update(1.0) is None
    assert sig.update(2.0) is None
    assert sig.update(3.0) is None
    assert sig.update(4.0) is not None


def test_zero_for_constant_window():
    # no spread => nothing to correlate => 0.0
    sig = Autocorrelation(window_size=5)
    for _ in range(5):
        out = sig.update(7.0)
    assert out == 0.0


def test_high_positive_for_smooth_ramp():
    # a steadily increasing ramp is strongly self-similar one step apart.
    # the finite-sample estimator yields exactly 0.5 for a perfect ramp of
    # this length (edge effects keep it below 1) — the point is it's clearly
    # positive, unlike the alternating case below.
    sig = Autocorrelation(window_size=6)
    out = None
    for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        out = sig.update(v)
    assert out >= 0.5


def test_negative_for_alternating_window():
    # values that flip up/down every step are negatively autocorrelated at lag 1
    sig = Autocorrelation(window_size=6)
    out = None
    for v in [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]:
        out = sig.update(v)
    assert out < 0.0


def test_bounded_within_unit_range():
    sig = Autocorrelation(window_size=10)
    out = None
    for i in range(10):
        out = sig.update(float(i % 3))  # arbitrary repeating pattern
    assert -1.0 - 1e-9 <= out <= 1.0 + 1e-9


def test_lag_must_be_at_least_one():
    with pytest.raises(ValueError):
        Autocorrelation(window_size=5, lag=0)


def test_lag_must_be_smaller_than_window():
    with pytest.raises(ValueError):
        Autocorrelation(window_size=5, lag=5)


def test_custom_lag_detects_period_two():
    # a period-2 pattern is perfectly correlated with itself at lag 2
    sig = Autocorrelation(window_size=6, lag=2)
    out = None
    for v in [1.0, 9.0, 1.0, 9.0, 1.0, 9.0]:
        out = sig.update(v)
    assert out > 0.5
