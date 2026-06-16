"""Tests for the LSTM one-step forecaster (residual-based detection)."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from threadforge.models.torch_forecaster import LSTMForecaster, lstm_residuals


def test_forecaster_forward_shape():
    model = LSTMForecaster(hidden_dim=8)
    out = model(torch.zeros(4, 10))     # (batch, window)
    assert out.shape == (4,)


def test_residuals_length_and_warmup_zero():
    values = [math.sin(i / 5.0) for i in range(300)]
    res = lstm_residuals(values, probation=150, window=20, epochs=5)
    assert len(res) == len(values)
    assert all(r == 0.0 for r in res[:20])   # first `window` steps are 0


def test_too_short_returns_zero_residuals():
    res = lstm_residuals([1.0, 2.0, 3.0, 4.0], probation=4, window=20)
    assert res == [0.0, 0.0, 0.0, 0.0]


def test_injected_spike_raises_residual():
    # learnable sine; inject an out-of-pattern spike well after probation
    values = [math.sin(i / 5.0) for i in range(400)]
    values[330] = 8.0
    res = lstm_residuals(values, probation=200, window=20, epochs=25, seed=0)
    # the spike's residual should stand above the typical post-probation residual
    typical = np.median([r for r in res[200:] if r > 0])
    assert res[330] > 3 * typical
