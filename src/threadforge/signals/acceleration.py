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

PERFORMANCE — O(1) PER STEP:
  Averaging the second differences telescopes: the sum of all second
  differences over the window equals just the last first-difference minus the
  first first-difference,

      sum(d2) = d1[-1] - d1[0]
              = (window[-1] - window[-2]) - (window[1] - window[0])

  so the whole average is a constant-time expression in four endpoints — no need
  to walk the window. update() uses that closed form (and skips the base class's
  window copy); compute() keeps the explicit O(W) version as the tested reference.
"""

from threadforge.signals.base import Signal


class Acceleration(Signal):
    def __init__(self, window_size: int):
        if window_size < 3:
            raise ValueError("Acceleration requires window_size >= 3")
        super().__init__(window_size)

    def update(self, value: float) -> float | None:
        self._window.append(value)
        if len(self._window) < self.window_size:
            return None  # warm-up: window not full yet
        w = self._window
        n = self.window_size
        # sum of second differences telescopes to d1[-1] - d1[0]; deque end
        # access (indices 0, 1, -2, -1) is O(1)
        return ((w[-1] - w[-2]) - (w[1] - w[0])) / (n - 2)

    def compute(self, window: list[float]) -> float:
        # Plain O(W) reference definition; update() is the O(1) fast path.
        # first differences (velocity)
        d1 = [window[i + 1] - window[i] for i in range(len(window) - 1)]
        # second differences (acceleration)
        d2 = [d1[i + 1] - d1[i] for i in range(len(d1) - 1)]
        # average over all second differences in the window
        return sum(d2) / len(d2)
