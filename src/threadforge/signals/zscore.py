"""ZScore: how many standard deviations the current value sits from the window mean.

A z-score of +3 means the current value is 3 standard deviations above the recent
average — statistically rare under a normal distribution. Unlike Sharpness (which
uses range as the denominator and is bounded to [-1, +1]), z-score is unbounded and
can grow large during genuine anomalies.

  z = (current - mean) / std

WHY UNBOUNDED MATTERS:
  A threshold calibrator learning from a calm window will set a threshold around
  mean + k*std of the z-scores. If the signal is bounded, the threshold may sit
  above the signal's maximum possible value — making it permanently untriggerable.
  An unbounded signal ensures extreme events can always exceed the threshold.

PERFORMANCE — O(1) PER STEP (incremental):
  Like Volatility, the mean and standard deviation are maintained from running
  sums (sum of values, sum of squares) as the window slides, so each step is
  constant time instead of O(W). `update()` is the fast path; `compute()` is the
  plain O(W) reference that update() is tested to match.
"""

import math

from threadforge.signals.base import Signal


class ZScore(Signal):
    def __init__(self, window_size: int):
        super().__init__(window_size)
        self._sum = 0.0
        self._sumsq = 0.0

    def reset(self) -> None:
        super().reset()
        self._sum = 0.0
        self._sumsq = 0.0

    def update(self, value: float) -> float | None:
        if len(self._window) == self.window_size:
            evicted = self._window[0]
            self._sum -= evicted
            self._sumsq -= evicted * evicted

        self._window.append(value)
        self._sum += value
        self._sumsq += value * value

        if len(self._window) < self.window_size:
            return None  # warm-up: window not full yet

        n = self.window_size
        mean = self._sum / n
        var = (self._sumsq - self._sum * mean) / (n - 1)
        if var < 0.0:
            var = 0.0  # clamp tiny negatives from floating-point cancellation
        std = math.sqrt(var)
        if std == 0.0:
            return 0.0  # constant window — current value is exactly at the mean
        return (value - mean) / std

    def compute(self, window: list[float]) -> float:
        # Plain O(W) reference definition; update() is the O(1) fast path.
        n = len(window)
        mean = sum(window) / n
        var = sum((x - mean) ** 2 for x in window) / (n - 1)
        std = math.sqrt(var)
        if std == 0.0:
            return 0.0
        return (window[-1] - mean) / std
