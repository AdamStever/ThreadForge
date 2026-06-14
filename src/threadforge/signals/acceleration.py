"""Acceleration: second difference across the window, per step squared.

The discrete second derivative — how fast the rate of change is itself changing.
Positive => the stream is speeding up; negative => slowing down; ~0 => steady pace.
Requires at least 3 points (window_size >= 3).

THE CALCULUS ANALOGY:
  In calculus:
    first derivative  = velocity  (how fast position changes)
    second derivative = acceleration (how fast velocity changes)

  Here we're working with discrete data (one reading per time step) so we
  use differences instead of derivatives:
    first differences  d1[i] = window[i+1] - window[i]   (like velocity)
    second differences d2[i] = d1[i+1] - d1[i]           (like acceleration)

EXAMPLE:
  window   = [0, 1, 4]        (speeding up)
  d1       = [1, 3]           (gaps growing)
  d2       = [2]              => acceleration = +2.0

  window   = [0, 3, 4]        (slowing down)
  d1       = [3, 1]           (gaps shrinking)
  d2       = [-2]             => acceleration = -2.0

  window   = [10, 20, 30]     (constant rate)
  d1       = [10, 10]
  d2       = [0]              => acceleration = 0.0
"""

from threadforge.signals.base import Signal


class Acceleration(Signal):
    def __init__(self, window_size: int):
        if window_size < 3:
            raise ValueError("Acceleration requires window_size >= 3")
        super().__init__(window_size)

    def compute(self, window: list[float]) -> float:
        # first differences (velocity)
        d1 = [window[i + 1] - window[i] for i in range(len(window) - 1)]
        # second differences (acceleration)
        d2 = [d1[i + 1] - d1[i] for i in range(len(d1) - 1)]
        # average over all second differences in the window
        return sum(d2) / len(d2)
