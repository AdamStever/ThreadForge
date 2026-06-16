"""Forecasting-based anomaly detector — the paradigm NAB's best detectors use.

Instead of training a classifier on labels (cross-file), this runs **online and
unsupervised** on each stream, the way NAB intends: predict the next value from
recent history, and flag when the prediction error is unusual relative to the
errors seen lately.

  1. one-step prediction via EWMA of past values
  2. residual  r_t = |x_t - prediction|
  3. normalize the residual against a rolling window of recent residuals:
         score_t = (r_t - mean(recent r)) / std(recent r)     (a residual z-score)
  4. flag when score_t exceeds a threshold, after a probationary period

The residual z-score is the key to controlling false positives: a perpetually
noisy stream has large residuals *and* a large residual spread, so its score
stays moderate — only a residual that is unusual *for that stream* fires. No
labels are used; labels only enter at scoring time.
"""

from __future__ import annotations

import math
from collections import deque


def residual_zscores(
    residuals: list[float],
    probation: int,
    resid_window: int = 200,
    min_history: int = 20,
) -> list[float]:
    """Normalize a residual series against a rolling window of recent residuals.

    Each residual is scored as (r - mean) / std over the *preceding* residuals, so
    only an error unusual *for that stream* gets a high score. Probation rows score
    0. Shared by every forecaster (EWMA, LSTM, …) — only the residuals differ.
    """
    history: deque[float] = deque(maxlen=resid_window)
    out: list[float] = []
    for i, r in enumerate(residuals):
        score = 0.0
        if i >= probation and len(history) >= min_history:
            mean = sum(history) / len(history)
            var = sum((v - mean) ** 2 for v in history) / len(history)
            std = math.sqrt(var)
            if std > 0.0:
                score = (r - mean) / std
        history.append(r)
        out.append(score)
    return out


class ForecastResidualDetector:
    def __init__(
        self,
        ewma_alpha: float = 0.2,
        resid_window: int = 200,
        probation_frac: float = 0.15,
        probation_max: int = 750,
        min_history: int = 20,
    ):
        self.ewma_alpha = ewma_alpha
        self.resid_window = resid_window
        self.probation_frac = probation_frac
        self.probation_max = probation_max
        self.min_history = min_history

    def probation(self, n: int) -> int:
        return min(int(self.probation_frac * n), self.probation_max)

    def residuals(self, stream: list[tuple[str, float]]) -> list[float]:
        """One-step EWMA prediction residuals |x - prediction| over the stream."""
        ewma: float | None = None
        out: list[float] = []
        for _, x in stream:
            prediction = x if ewma is None else ewma
            out.append(abs(x - prediction))
            ewma = x if ewma is None else self.ewma_alpha * x + (1 - self.ewma_alpha) * ewma
        return out

    def scores(self, stream: list[tuple[str, float]]) -> list[float]:
        """Per-step anomaly score (residual z-score). 0.0 during warm-up/probation."""
        probation = self.probation(len(stream))
        return residual_zscores(self.residuals(stream), probation, self.resid_window, self.min_history)

    def flags(self, stream: list[tuple[str, float]], threshold: float) -> list[bool]:
        """Boolean detections: score >= threshold (probation rows score 0 → never flag)."""
        return [s >= threshold for s in self.scores(stream)]
