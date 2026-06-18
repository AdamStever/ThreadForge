"""Online (streaming) form of the forecasting-residual detector.

The batch `ForecastResidualDetector` consumes a whole stream as a list. This is
the same algorithm reshaped for a *live* feed: one point in, one score out, O(1)
state, no knowledge of the total length.

  step(x):
    prediction = EWMA of past values
    residual   = |x - prediction|
    score      = (residual - mean(recent residuals)) / std(recent residuals)
                 once past the probation period and enough history has built up

It is mathematically identical to feeding the same points through
`ForecastResidualDetector.residuals` + `residual_zscores` with a fixed
`probation` — `tests/test_streaming.py` pins that equivalence.

The one necessary difference from the batch detector: **probation is an absolute
number of steps**, not a fraction of the total. A live stream has no known total,
so "warm up for the first N points" is the only thing that makes sense; callers
replaying a finite file can pass the batch detector's `probation(n)` to match it.
"""

from __future__ import annotations

import math
from collections import deque


class OnlineForecastResidualDetector:
    def __init__(
        self,
        ewma_alpha: float = 0.2,
        resid_window: int = 200,
        probation: int = 750,
        min_history: int = 20,
    ):
        self.ewma_alpha = ewma_alpha
        self.resid_window = resid_window
        self.probation = probation
        self.min_history = min_history
        self.reset()

    def reset(self) -> None:
        """Clear all state so the detector can be reused on a fresh stream."""
        self._ewma: float | None = None
        self._history: deque[float] = deque(maxlen=self.resid_window)
        self._i = 0

    def update(self, value: float) -> float:
        """Ingest one value; return its anomaly score (0.0 during warm-up/probation)."""
        prediction = value if self._ewma is None else self._ewma
        residual = abs(value - prediction)
        self._ewma = (
            value if self._ewma is None
            else self.ewma_alpha * value + (1 - self.ewma_alpha) * self._ewma
        )

        score = 0.0
        if self._i >= self.probation and len(self._history) >= self.min_history:
            mean = sum(self._history) / len(self._history)
            var = sum((v - mean) ** 2 for v in self._history) / len(self._history)
            std = math.sqrt(var)
            if std > 0.0:
                score = (residual - mean) / std

        self._history.append(residual)
        self._i += 1
        return score
