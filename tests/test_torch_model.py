"""Tests for the raw-window dataset and the PyTorch encoder/score net.

The torch tests skip cleanly if the optional dependency isn't installed.
"""

import numpy as np
import pytest

from threadforge.models import build_window_examples

torch = pytest.importorskip("torch")
from threadforge.models.torch_model import (
    EncoderScorer, LSTMScorer, train_model, train_lstm, predict_proba,
)


# --- raw-window dataset ---

def test_window_examples_shape_and_warmup():
    stream = [(f"2024-01-01 00:{i:02d}:00", float(i)) for i in range(20)]
    we = build_window_examples("s.csv", stream, [], window_size=5)
    assert we.X.shape == (20 - 4, 5)         # warm-up of window_size-1 dropped
    assert we.X.shape[0] == we.y.shape[0]


def test_window_examples_normalized_per_row():
    stream = [(f"2024-01-01 00:{i:02d}:00", float(i)) for i in range(20)]
    we = build_window_examples("s.csv", stream, [], window_size=5)
    # each (non-constant) window is z-normalized: mean ~0, std ~1
    assert np.allclose(we.X.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(we.X.std(axis=1), 1.0, atol=1e-9)


def test_window_examples_labels_window():
    stream = [(f"2024-01-01 00:{i:02d}:00", float(i % 3)) for i in range(20)]
    windows = [("2024-01-01 00:10:00", "2024-01-01 00:13:00")]
    we = build_window_examples("s.csv", stream, windows, window_size=5)
    assert we.y[we.timestamps.index("2024-01-01 00:11:00")] == 1
    assert we.y[we.timestamps.index("2024-01-01 00:17:00")] == 0


# --- encoder/score net ---

def test_forward_output_shape():
    model = EncoderScorer(input_dim=8, latent_dim=4)
    out = model(torch.zeros(5, 8))
    assert out.shape == (5,)


def test_latent_dimension():
    model = EncoderScorer(input_dim=8, latent_dim=4)
    z = model.latent(torch.zeros(3, 8))
    assert z.shape == (3, 4)


def test_train_separates_classes():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (200, 8)), rng.normal(5, 1, (40, 8))])
    y = np.array([0] * 200 + [1] * 40)
    model = train_model(X, y, epochs=40, seed=0)
    proba = predict_proba(model, np.array([[5.0] * 8, [0.0] * 8]))
    assert proba.shape == (2,)
    assert proba[0] > proba[1]          # anomaly-like scored higher
    assert 0.0 <= proba.min() <= proba.max() <= 1.0


# --- temporal LSTM ---

def test_lstm_forward_output_shape():
    model = LSTMScorer(hidden_dim=8)
    out = model(torch.zeros(5, 12))     # (batch, seq_len)
    assert out.shape == (5,)


def test_lstm_train_separates_classes():
    # anomaly windows are a rising ramp; normal windows are flat noise
    rng = np.random.default_rng(0)
    seq_len = 12
    normal = rng.normal(0, 0.5, (150, seq_len))
    ramp = np.linspace(0, 4, seq_len) + rng.normal(0, 0.2, (40, seq_len))
    X = np.vstack([normal, ramp])
    y = np.array([0] * 150 + [1] * 40)
    model = train_lstm(X, y, epochs=30, seed=0)
    proba = predict_proba(model, np.vstack([np.linspace(0, 4, seq_len), np.zeros(seq_len)]))
    assert proba.shape == (2,)
    assert proba[0] > proba[1]          # ramp scored higher than flat
    assert 0.0 <= proba.min() <= proba.max() <= 1.0
