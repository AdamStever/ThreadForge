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

    def scores(self, stream: list[tuple[str, float]]) -> list[float]:
        """Per-step anomaly score (residual z-score). 0.0 during warm-up/probation."""
        n = len(stream)
        probation = self.probation(n)
        alpha = self.ewma_alpha

        ewma: float | None = None
        residuals: deque[float] = deque(maxlen=self.resid_window)
        out: list[float] = []

        for i, (_, x) in enumerate(stream):
            prediction = x if ewma is None else ewma
            r = abs(x - prediction)
            ewma = x if ewma is None else alpha * x + (1 - alpha) * ewma

            score = 0.0
            if i >= probation and len(residuals) >= self.min_history:
                mean = sum(residuals) / len(residuals)
                var = sum((v - mean) ** 2 for v in residuals) / len(residuals)
                std = math.sqrt(var)
                if std > 0.0:
                    score = (r - mean) / std

            residuals.append(r)  # the residual still informs the rolling scale
            out.append(score)

        return out

    def flags(self, stream: list[tuple[str, float]], threshold: float) -> list[bool]:
        """Boolean detections: score >= threshold (probation rows score 0 → never flag)."""
        return [s >= threshold for s in self.scores(stream)]
