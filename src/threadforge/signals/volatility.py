"""Volatility: sample standard deviation over the window.

High => turbulent/erratic, low => calm/steady.
Measures how spread out the values are around the window's average.

WHY STANDARD DEVIATION?
  It answers "how far from average is a typical value?" in the same units as
  the data itself. A CPU stream sitting at ~50% with tiny fluctuations has low
  volatility; one swinging between 10% and 90% has high volatility.

WHY n-1 (SAMPLE STD) INSTEAD OF n (POPULATION STD)?
  We only ever see a window of the full stream, not the whole stream. Dividing
  by (n-1) instead of n corrects for the fact that a small sample tends to
  underestimate the true spread — this is called Bessel's correction.
"""

import math

from threadforge.signals.base import Signal


class Volatility(Signal):
    def compute(self, window: list[float]) -> float:
        n = len(window)
        mean = sum(window) / n
        # sum of squared deviations from the mean, divided by (n-1)
        var = sum((x - mean) ** 2 for x in window) / (n - 1)
        return math.sqrt(var)
