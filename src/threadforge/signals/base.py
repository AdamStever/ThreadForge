"""Base class for all signals.

A signal keeps a fixed-size rolling window of the most recent values and
computes a single number from them. Every signal is causal: it only ever
sees past and current values, never future ones.

WHY A ROLLING WINDOW?
  In a real streaming system data arrives one point at a time — you can't
  "look ahead". A rolling window lets us compute meaningful statistics
  (average, spread, trend) from only the most recent N values, which is all
  we'd ever have in production.

WHY RETURN None UNTIL THE WINDOW FILLS?
  If we only have 2 values and the window needs 30, any statistic we compute
  would be meaningless. Returning None forces callers to explicitly handle
  the warm-up period rather than silently using bad numbers.
"""

from abc import ABC, abstractmethod
from collections import deque


class Signal(ABC):
    def __init__(self, window_size: int):
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self.window_size = window_size
        # deque with maxlen automatically drops the oldest value when a new
        # one is added — that's exactly the "rolling" behaviour we want.
        self._window: deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> float | None:
        """Add a new value, then compute the signal over the current window.

        Returns None until the window has filled, so we never emit a signal
        from too little data.
        """
        self._window.append(value)
        if len(self._window) < self.window_size:
            return None
        return self.compute(list(self._window))

    def reset(self) -> None:
        """Clear the window so the signal can be reused on a new stream."""
        self._window.clear()

    @abstractmethod
    def compute(self, window: list[float]) -> float:
        # Subclasses implement this — they receive a full window and return
        # one number that summarises it.
        ...
