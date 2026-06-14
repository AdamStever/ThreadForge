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
"""

import math

from threadforge.signals.base import Signal


class ZScore(Signal):
    def compute(self, window: list[float]) -> float:
        n = len(window)
        mean = sum(window) / n
        var = sum((x - mean) ** 2 for x in window) / (n - 1)
        std = math.sqrt(var)
        if std == 0.0:
            return 0.0  # constant window — current value is exactly at the mean
        return (window[-1] - mean) / std
