"""Sharpness: how far the current value sits from the window mean, in units of spread.

A high absolute sharpness means the latest value is an outlier relative to recent
history — it is the discrete analogue of a z-score computed causally over the window.

INTUITION:
  Imagine the last 30 CPU readings averaged around 50% and never left the 48-52%
  range. Sharpness at the *current* point tells you: relative to that recent band,
  how extreme is this new reading?

  sharpness = (current - mean) / (max - min)

  Result is always in the range [-1, +1]:
    +1  => current value is the highest in the window
    -1  => current value is the lowest in the window
     0  => current value sits exactly at the window mean

WHY USE (max - min) AS THE DENOMINATOR INSTEAD OF STD DEV?
  The range is simpler to reason about and requires no extra math library.
  A std-dev based z-score is a natural next step if more sensitivity is needed.
"""

from threadforge.signals.base import Signal


class Sharpness(Signal):
    def compute(self, window: list[float]) -> float:
        n = len(window)
        mean = sum(window) / n
        spread = max(window) - min(window)
        if spread == 0.0:
            return 0.0  # all values identical — no outlier possible
        return (window[-1] - mean) / spread
