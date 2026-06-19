"""WeightedSignalDetector — the 10-signal detector, done right.

The original signal ensemble lost to the EWMA forecaster for fixable reasons: it
voted against frozen bands (over-flagging) and weighted every signal equally (the
useful signals drowned in the noisy ones). This detector keeps the 10 signals as
the *only* evidence but combines them the way that actually works:

  1. Each step, take the 10 causal signals.
  2. Per signal, a robust z-score against a **rolling window of its recent values**
     (so "normal" tracks the stream — no frozen calibration band).
  3. Combine into one surprise as a **weighted** mean of the absolute z-scores.
     The weights say how much each signal is trusted — learned by the genetic
     search to maximise VUS-PR, so the discriminative signals carry the decision
     and the noise gets ~zero weight (rather than hand-tuned, per project policy).
  4. Normalise that surprise with the same rolling residual-z + probation
     discipline that gave the forecaster its false-positive control.

No EWMA, no forecast residual — pure signals. Standard ``scores``/``flags``/
``probation`` interface, so it competes as a challenger and is judged by VUS-PR.
``weights`` is a ``{signal_name: weight}`` dict (or a list in signal order);
``None`` means equal weights.
"""

from __future__ import annotations

import math
from collections import deque

from threadforge.detection.forecast_detector import residual_zscores
from threadforge.presets import default_signal_engine, default_signal_names


class WeightedSignalDetector:
    def __init__(
        self,
        weights=None,
        *,
        window_size: int = 30,
        feature_window: int = 200,
        resid_window: int = 200,
        probation_frac: float = 0.15,
        probation_max: int = 750,
        min_history: int = 20,
        min_window: int = 30,
    ):
        self.window_size = window_size
        self.feature_window = feature_window
        self.resid_window = resid_window
        self.probation_frac = probation_frac
        self.probation_max = probation_max
        self.min_history = min_history
        self.min_window = min_window
        self._names = default_signal_names(window_size)

        if weights is None:
            self.weights = {n: 1.0 for n in self._names}
        elif isinstance(weights, dict):
            self.weights = {n: float(weights.get(n, 0.0)) for n in self._names}
        else:  # iterable in signal order
            self.weights = {n: float(w) for n, w in zip(self._names, weights)}

    def probation(self, n: int) -> int:
        return min(int(self.probation_frac * n), self.probation_max)

    def _surprise_series(self, stream: list[tuple[str, float]]) -> list[float]:
        engine = default_signal_engine(self.window_size)
        names = self._names
        d = len(names)
        w = [self.weights[n] for n in names]
        wsum = sum(w) or 1.0

        win: deque[list[float]] = deque(maxlen=self.feature_window)
        sums = [0.0] * d
        sumsq = [0.0] * d
        surprise: list[float] = []

        for _, x in stream:
            sig = engine.update(x)
            vals = [sig.get(n) for n in names]
            if any(v is None for v in vals):     # signals still warming up
                surprise.append(0.0)
                continue
            f = [float(v) for v in vals]

            count = len(win)
            if count >= self.min_window:
                total = 0.0
                for j in range(d):
                    if w[j] == 0.0:
                        continue
                    mean = sums[j] / count
                    var = sumsq[j] / count - mean * mean
                    std = math.sqrt(var) if var > 0.0 else 0.0
                    if std > 1e-9:
                        total += w[j] * abs((f[j] - mean) / std)
                surprise.append(total / wsum)
            else:
                surprise.append(0.0)

            if len(win) == self.feature_window:
                old = win[0]
                for j in range(d):
                    sums[j] -= old[j]
                    sumsq[j] -= old[j] * old[j]
            win.append(f)
            for j in range(d):
                sums[j] += f[j]
                sumsq[j] += f[j] * f[j]

        return surprise

    def scores(self, stream: list[tuple[str, float]]) -> list[float]:
        surprise = self._surprise_series(stream)
        probation = self.probation(len(stream))
        return residual_zscores(surprise, probation, self.resid_window, self.min_history)

    def flags(self, stream: list[tuple[str, float]], threshold: float) -> list[bool]:
        return [s >= threshold for s in self.scores(stream)]
