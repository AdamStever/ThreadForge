"""Tests for the latent-state (feature-vector) representation."""

import math

import numpy as np
import pytest

from threadforge.state import StateVector, standardize, euclidean_distance
from threadforge.engine import SignalEngine
from threadforge.signals import Volatility, Momentum


def test_dim_matches_schema():
    sv = StateVector(["a", "b", "c"])
    assert sv.dim == 3


def test_empty_schema_raises():
    with pytest.raises(ValueError):
        StateVector([])


def test_vector_follows_schema_order_not_dict_order():
    sv = StateVector(["a", "b", "c"])
    vec = sv.vector({"c": 3.0, "a": 1.0, "b": 2.0})
    assert np.array_equal(vec, np.array([1.0, 2.0, 3.0]))


def test_missing_signal_becomes_nan_by_default():
    sv = StateVector(["a", "b"])
    vec = sv.vector({"a": 1.0, "b": None})
    assert vec[0] == 1.0
    assert math.isnan(vec[1])


def test_custom_fill_value():
    sv = StateVector(["a", "b"], fill=0.0)
    vec = sv.vector({"a": 1.0})  # b absent entirely
    assert np.array_equal(vec, np.array([1.0, 0.0]))


def test_is_ready_only_when_all_present():
    sv = StateVector(["a", "b"])
    assert not sv.is_ready({"a": 1.0, "b": None})
    assert not sv.is_ready({"a": 1.0})
    assert sv.is_ready({"a": 1.0, "b": 2.0})


def test_standardize_centers_and_scales():
    vec = np.array([10.0, 20.0])
    out = standardize(vec, centers=np.array([5.0, 10.0]), scales=np.array([5.0, 2.0]))
    assert np.allclose(out, np.array([1.0, 5.0]))


def test_standardize_zero_scale_axis_is_zero():
    vec = np.array([10.0, 20.0])
    out = standardize(vec, centers=np.array([5.0, 10.0]), scales=np.array([5.0, 0.0]))
    assert out[0] == 1.0
    assert out[1] == 0.0  # constant axis -> 0, not division by zero


def test_euclidean_distance_known_value():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert euclidean_distance(a, b) == pytest.approx(5.0)


def test_euclidean_distance_zero_for_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert euclidean_distance(a, a) == 0.0


def test_integration_with_signal_engine():
    engine = SignalEngine()
    engine.register("volatility", Volatility(3))
    engine.register("momentum", Momentum(3))
    sv = StateVector(["volatility", "momentum"])

    out = None
    for v in (10.0, 20.0, 30.0):
        out = engine.update(v)
    # window full -> both signals defined -> state is a complete point in R^2
    assert sv.is_ready(out)
    vec = sv.vector(out)
    assert vec.shape == (2,)
    assert np.all(np.isfinite(vec))
