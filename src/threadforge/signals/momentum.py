"""Momentum: net change across the window, per step.

Positive => trending up, negative => trending down, ~0 => flat.
This is a discrete first derivative — the rate-of-change idea from calculus.

EXAMPLE:
  window = [10, 20, 30]
  momentum = (30 - 10) / (3 - 1) = 10.0 per step

WHY DIVIDE BY (len - 1)?
  We want the *average* rate of change per time step, not the raw difference.
  Dividing by the number of gaps between points makes signals from different
  window sizes comparable.
"""

from threadforge.signals.base import Signal


class Momentum(Signal):
    def compute(self, window: list[float]) -> float:
        # (last value - first value) / number of steps between them
        return (window[-1] - window[0]) / (len(window) - 1)
