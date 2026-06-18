"""Tests for domain-adaptation (seeded) forecasting. Skipped without torch."""

import numpy as np
import pytest

pytest.importorskip("torch")

from threadforge.models.domain_adapt import train_pool_forecaster, SeededForecastDetector


def _series(n=300, seed=0):
    rng = np.random.RandomState(seed)
    return list(np.sin(np.linspace(0, 20, n)) + rng.normal(0, 0.05, n))


def _stream(n=300, spike=(150, 160), seed=1):
    rng = np.random.RandomState(seed)
    v = np.sin(np.linspace(0, 20, n)) + rng.normal(0, 0.05, n)
    for i in range(*spike):
        v[i] += 4.0
    return [(str(i), float(x)) for i, x in enumerate(v)]


def _model():
    pool = [_series(seed=s) for s in range(3)]
    return train_pool_forecaster(pool, window=10, hidden_dim=8, epochs=3, seed=0)


def test_train_pool_returns_model():
    assert _model() is not None


def test_train_pool_empty_returns_none():
    assert train_pool_forecaster([], window=10) is None
    assert train_pool_forecaster([[1.0, 2.0]], window=10) is None  # too short for any window


def test_seeded_detector_scores_shape_and_probation():
    det = SeededForecastDetector(_model(), window=10, resid_window=50, min_history=5)
    stream = _stream()
    scores = det.scores(stream)
    assert len(scores) == len(stream)
    assert all(np.isfinite(s) for s in scores)
    probation = det.probation(len(stream))
    assert all(s == 0.0 for s in scores[:probation])


def test_seeded_detector_deterministic():
    model = _model()
    a = SeededForecastDetector(model, window=10, resid_window=50, min_history=5).scores(_stream())
    b = SeededForecastDetector(model, window=10, resid_window=50, min_history=5).scores(_stream())
    assert a == pytest.approx(b, abs=1e-9)
