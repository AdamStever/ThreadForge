"""Domain adaptation — seed a forecaster on a domain, apply it to unseen streams.

ThreadForge is meant to be *universal*: it must work on a brand-new domain with no
training (cold start), and get better as it sees more of that domain (seeding).
The per-file `NeuralForecastResidualDetector` is the cold-start/zero-seed case —
it trains only on the current file's prefix. This module is the *seeded* case: an
LSTM forecaster trained across a pool of a domain's files, then applied to a
held-out stream it never saw.

  - `train_pool_forecaster(series_list)` — train one LSTM on next-value pairs from
    many series, each z-normalized by its own stats so the model learns *shape*,
    not absolute level. Unsupervised: only forecasting, anomaly labels never used.
  - `SeededForecastDetector` — wraps a pretrained model in the standard
    `scores()`/`flags()` interface. On a new stream it normalizes by that stream's
    causal probation prefix, forecasts, and turns residuals into residual
    z-scores — exactly like the EWMA/per-file detectors, so it competes in the
    same loop.

No leakage: the seed pool must be disjoint from the streams being scored.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from threadforge.detection.forecast_detector import residual_zscores
from threadforge.models.torch_forecaster import LSTMForecaster, _windows
from threadforge.models.torch_util import get_device


def _normalize(series: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Z-normalize ``series`` by ``ref`` stats (default: the series itself)."""
    ref = series if ref is None else ref
    mu = float(ref.mean())
    sd = float(ref.std()) or 1.0
    return (series - mu) / sd


def train_pool_forecaster(
    series_list: list[list[float]],
    *,
    window: int = 20,
    hidden_dim: int = 32,
    epochs: int = 15,
    lr: float = 1e-2,
    seed: int = 0,
    device=None,
) -> LSTMForecaster | None:
    """Train one LSTM forecaster across many series (each self-normalized).

    Returns the trained model, or None if the pool yields too few windows. Runs on
    the GPU when available.
    """
    device = get_device(device)
    Xs, ys = [], []
    for values in series_list:
        z = _normalize(np.asarray(values, dtype=float))
        X, y = _windows(z, window)
        if len(X):
            Xs.append(X)
            ys.append(y)
    if not Xs:
        return None
    X = np.concatenate(Xs)
    y = np.concatenate(ys)

    torch.manual_seed(seed)
    model = LSTMForecaster(hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _pool_residuals(model: LSTMForecaster, values: list[float], probation: int,
                    *, window: int, device) -> list[float]:
    """One-step residuals from a pretrained model on a new (causally-normalized) series."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    residuals = [0.0] * n
    ref = v[:probation] if probation > 0 else v
    z = _normalize(v, ref)                      # normalize the new stream by its own prefix
    Xall, _ = _windows(z, window)
    if len(Xall) == 0:
        return residuals
    with torch.no_grad():
        preds = model(torch.tensor(Xall, dtype=torch.float32, device=device)).cpu().numpy()
    for k, i in enumerate(range(window, n)):
        residuals[i] = abs(z[i] - preds[k])
    return residuals


class SeededForecastDetector:
    """A forecaster pretrained on a domain, scoring unseen streams of that domain."""

    def __init__(
        self,
        model: LSTMForecaster,
        *,
        window: int = 20,
        resid_window: int = 200,
        probation_frac: float = 0.15,
        probation_max: int = 750,
        min_history: int = 20,
        device=None,
    ):
        self.model = model
        self.window = window
        self.resid_window = resid_window
        self.probation_frac = probation_frac
        self.probation_max = probation_max
        self.min_history = min_history
        self.device = get_device(device)

    def probation(self, n: int) -> int:
        return min(int(self.probation_frac * n), self.probation_max)

    def scores(self, stream: list[tuple[str, float]]) -> list[float]:
        values = [v for _, v in stream]
        probation = self.probation(len(values))
        residuals = _pool_residuals(self.model, values, probation,
                                    window=self.window, device=self.device)
        return residual_zscores(residuals, probation, self.resid_window, self.min_history)

    def flags(self, stream: list[tuple[str, float]], threshold: float) -> list[bool]:
        return [s >= threshold for s in self.scores(stream)]
