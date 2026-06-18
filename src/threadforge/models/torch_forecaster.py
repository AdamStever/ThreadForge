"""LSTM one-step forecaster for residual-based anomaly detection.

The EWMA forecaster predicts the next value as a smoothed average — fine for
level shifts, blind to dynamics. This swaps in an LSTM that learns to predict the
next value from a window of recent values. Everything downstream is unchanged:
residual = |actual - predicted| → residual z-score → threshold → NAB.

It stays unsupervised and online/per-file in NAB's spirit: the forecaster is
trained only on the file's **probationary** prefix (assumed normal), then frozen
and run forward to produce residuals over the rest of the stream. No anomaly
labels are used.

Imported on demand (not from `models/__init__`) so the core package stays
torch-free.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from threadforge.models.torch_util import get_device


class LSTMForecaster(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.unsqueeze(-1))   # (B, W) -> (B, W, 1) -> (B, W, H)
        return self.head(out[:, -1, :]).squeeze(-1)  # predicted next value


def _windows(series: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Supervised next-value pairs: X[i] = series[i-w:i], y[i] = series[i]."""
    if len(series) <= w:
        return np.empty((0, w)), np.empty((0,))
    X = np.stack([series[i - w:i] for i in range(w, len(series))])
    y = series[w:]
    return X, y


def lstm_residuals(
    values: list[float],
    probation: int,
    *,
    window: int = 20,
    hidden_dim: int = 32,
    epochs: int = 15,
    lr: float = 1e-2,
    seed: int = 0,
    device=None,
) -> list[float]:
    """Train an LSTM on the probation prefix, return one-step residuals over the stream.

    Residuals for the first `window` steps (and if training data is too small) are
    0.0, so they never trigger a detection. Runs on the GPU when one is available
    (``device`` overrides), falling back to CPU.
    """
    device = get_device(device)
    v = np.asarray(values, dtype=float)
    n = len(v)
    residuals = [0.0] * n

    # normalize using only the probation prefix (causal)
    ref = v[:probation] if probation > 0 else v
    mu = float(ref.mean())
    sd = float(ref.std()) or 1.0
    z = (v - mu) / sd

    Xtr, ytr = _windows(z[:probation], window)
    if len(Xtr) < 10:
        return residuals  # not enough history to train -> no detections

    torch.manual_seed(seed)
    model = LSTMForecaster(hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        optimizer.step()

    # predict next value across the whole (normalized) series
    Xall, _ = _windows(z, window)
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(Xall, dtype=torch.float32, device=device)).cpu().numpy()

    for k, i in enumerate(range(window, n)):
        residuals[i] = abs(z[i] - preds[k])
    return residuals
