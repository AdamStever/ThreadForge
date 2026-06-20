"""Perception — turn a price stream into the signed, standardized signal state vector.

This is the "sensorium" the trading policy acts on. The anomaly path collapses the
10 signals to ``abs(z)``, deleting direction; the policy needs the opposite — the
*signed* state, standardized so signals on wildly different scales (momentum in
price units vs autocorrelation in [-1, 1]) are comparable and stationary as price
drifts over years.

Two causal transforms, chosen by what each signal means for trading:

* **Directional** signals — momentum, acceleration, zscore — carry meaning in their
  *level and sign* (positive momentum = uptrend). We scale by a trailing std but do
  NOT center, so a steady trend keeps a steady non-zero feature (centering would
  zero it out, because in a steady trend momentum sits near its own mean).
* **Regime / risk / conviction** signals — volatility, hilbert, the entropies,
  autocorrelation, spectral_flatness — carry meaning in *deviation from their
  recent normal* (vol higher than usual = de-risk). We center and scale (rolling z).

Everything is computed over trailing windows only — causal, no look-ahead. This
per-instrument normalization lives in the data/market layer by design; the core
signals stay domain-agnostic.
"""

from __future__ import annotations

import numpy as np

from threadforge.presets import default_signal_engine, default_signal_names

# directional signals: keep level + sign (scale only, no centering)
DIRECTIONAL = ("momentum", "acceleration", "zscore")
CLIP = 5.0


def _rolling_mean_std(values: np.ndarray, window: int):
    """Causal trailing mean and std for each position (NaN-safe, 0 until warm)."""
    n = len(values)
    mean = np.zeros(n)
    std = np.zeros(n)
    csum = np.concatenate([[0.0], np.cumsum(values)])
    csq = np.concatenate([[0.0], np.cumsum(values * values)])
    min_count = max(5, window // 2)
    for t in range(n):
        lo = max(0, t - window + 1)
        cnt = t + 1 - lo
        if cnt < min_count:
            continue
        s = csum[t + 1] - csum[lo]
        sq = csq[t + 1] - csq[lo]
        m = s / cnt
        var = sq / cnt - m * m
        mean[t] = m
        std[t] = np.sqrt(var) if var > 0.0 else 0.0
    return mean, std


def rolling_z(values, window: int = 60) -> np.ndarray:
    """Causal z-score of a series against its own trailing window (0 until filled)."""
    values = np.asarray(values, dtype=float)
    mean, std = _rolling_mean_std(values, window)
    out = np.zeros(len(values))
    nz = std > 1e-12
    out[nz] = (values[nz] - mean[nz]) / std[nz]
    return np.clip(out, -CLIP, CLIP)


def rolling_scale(values, window: int = 60) -> np.ndarray:
    """Causal scale-only standardization: value / trailing std (sign + level kept)."""
    values = np.asarray(values, dtype=float)
    _, std = _rolling_mean_std(values, window)
    out = np.zeros(len(values))
    nz = std > 1e-12
    out[nz] = values[nz] / std[nz]
    return np.clip(out, -CLIP, CLIP)


def signal_series(stream, name: str = "momentum", window_size: int = 30) -> np.ndarray:
    """Causal series of one raw signal value; warm-up steps are 0.0."""
    engine = default_signal_engine(window_size)
    out = [engine.update(x).get(name) for _, x in stream]
    return np.asarray([0.0 if v is None else float(v) for v in out], dtype=float)


def raw_signal_matrix(stream, window_size: int = 30):
    """All 10 raw signal series as columns ``[T, 10]`` plus their names (warm-up 0)."""
    names = default_signal_names(window_size)
    engine = default_signal_engine(window_size)
    rows = []
    for _, x in stream:
        sig = engine.update(x)
        rows.append([0.0 if sig.get(n) is None else float(sig[n]) for n in names])
    return np.asarray(rows, dtype=float), names


def signal_matrix(stream, window_size: int = 30, z_window: int = 60):
    """The standardized, signed state vector ``[T, 10]`` plus names.

    Directional signals are scaled (level + sign kept); the rest are rolling-z
    (deviation from recent normal). Comparable scale, stationary, causal.
    """
    raw, names = raw_signal_matrix(stream, window_size)
    cols = []
    for j, name in enumerate(names):
        col = raw[:, j]
        cols.append(rolling_scale(col, z_window) if name in DIRECTIONAL
                    else rolling_z(col, z_window))
    return np.column_stack(cols) if cols else raw, names
