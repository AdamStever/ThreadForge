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

PERFORMANCE — O(1) PER STEP (incremental):
  The obvious implementation recomputes the mean and the sum of squared
  deviations over the whole window every step — O(W) per step, O(n·W) overall.
  Instead we keep two running totals as the window slides: the sum of values and
  the sum of their squares. When a value enters we add its contribution; when a
  value leaves the window we subtract it. The variance then comes from

      var = (sumsq - sum^2 / n) / (n - 1)

  in constant time, so the whole stream is O(n). `update()` is the fast path;
  `compute()` is kept as the plain O(W) reference that update() must agree with
  (the tests assert they match).
"""

import math

from threadforge.signals.base import Signal


class Volatility(Signal):
    def __init__(self, window_size: int):
        super().__init__(window_size)
        self._sum = 0.0     # running sum of the values in the window
        self._sumsq = 0.0   # running sum of their squares

    def reset(self) -> None:
        super().reset()
        self._sum = 0.0
        self._sumsq = 0.0

    def update(self, value: float) -> float | None:
        # If the window is full, the oldest value is about to fall out — remove
        # its contribution from the running totals before the deque drops it.
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
        return math.sqrt(var)

    def compute(self, window: list[float]) -> float:
        # Plain O(W) reference definition; update() is the O(1) fast path and is
        # tested for agreement with this.
        n = len(window)
        mean = sum(window) / n
        var = sum((x - mean) ** 2 for x in window) / (n - 1)
        return math.sqrt(var)
