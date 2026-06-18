"""Tests for the neural (LSTM) forecasting-residual detector.

Skipped when torch is not installed (the optional `dl` extra). Kept tiny — a few
epochs on a short series — so it runs fast on CPU.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from threadforge.models.neural_forecast import NeuralForecastResidualDetector
from threadforge.models.torch_util import get_device


def _stream(n=200, spike=(120, 130), seed=0):
    rng = np.random.RandomState(seed)
    values = rng.rand(n)
    for i in range(*spike):
        values[i] += 5.0
    return [(str(i), float(v)) for i, v in enumerate(values)]


def _detector():
    return NeuralForecastResidualDetector(
        window=10, hidden_dim=8, epochs=3, resid_window=50, min_history=5, seed=0,
    )


def test_scores_shape_and_finiteness():
    stream = _stream()
    scores = _detector().scores(stream)
    assert len(scores) == len(stream)
    assert all(np.isfinite(s) for s in scores)


def test_probation_region_scores_zero():
    stream = _stream(n=300)
    det = _detector()
    scores = det.scores(stream)
    probation = det.probation(len(stream))
    assert all(s == 0.0 for s in scores[:probation])


def test_deterministic_with_seed():
    stream = _stream()
    a = _detector().scores(stream)
    b = _detector().scores(stream)
    assert a == pytest.approx(b, abs=1e-9)


def test_flags_threshold():
    stream = _stream()
    det = _detector()
    scores = det.scores(stream)
    flags = det.flags(stream, threshold=3.0)
    assert flags == [s >= 3.0 for s in scores]


def test_runs_on_explicit_cpu_device():
    det = NeuralForecastResidualDetector(window=10, hidden_dim=8, epochs=2,
                                         resid_window=50, min_history=5, device="cpu")
    scores = det.scores(_stream(n=150))
    assert len(scores) == 150


def test_get_device_returns_torch_device():
    import torch
    dev = get_device("cpu")
    assert isinstance(dev, torch.device) and dev.type == "cpu"
