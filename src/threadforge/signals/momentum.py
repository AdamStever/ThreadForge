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

PERFORMANCE — O(1) PER STEP:
  Momentum only needs the first and last values of the window, so update() reads
  them directly from the deque ends (O(1)) and skips the base class's O(W) window
  copy. compute() keeps the same formula as the tested reference.
"""

from threadforge.signals.base import Signal


class Momentum(Signal):
    def update(self, value: float) -> float | None:
        self._window.append(value)
        if len(self._window) < self.window_size:
            return None  # warm-up: window not full yet
        return (self._window[-1] - self._window[0]) / (self.window_size - 1)

    def compute(self, window: list[float]) -> float:
        # (last value - first value) / number of steps between them
        return (window[-1] - window[0]) / (len(window) - 1)
