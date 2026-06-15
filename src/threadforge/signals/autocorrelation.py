"""Autocorrelation: how similar the window is to a time-shifted copy of itself.

Measures the *temporal structure* of a stream rather than its magnitude. Where
Volatility asks "how spread out are the values?" and ZScore asks "how far is the
latest value from normal?", autocorrelation asks "how predictable is this series
from its own recent past?"

The lag-1 autocorrelation correlates each value with the one immediately before
it. The result is bounded to roughly [-1, +1]:

  near +1  smooth and predictable — each value is close to the previous one
           (a calm, trending, or slowly-drifting stream)
  near  0  no relationship between consecutive values — looks like noise
  near -1  strongly oscillating — values alternate up/down every step

WHY THIS CATCHES A DIFFERENT KIND OF ANOMALY
  Some anomalies don't change the *size* of the values at all — they change
  their *structure*. A periodic signal (daily traffic, a heartbeat, a polling
  loop) has high autocorrelation while the rhythm holds. When that rhythm breaks
  — a sensor starts chattering, a periodic job stalls, a smooth feed turns noisy
  — the magnitude may look normal but the autocorrelation collapses. Volatility
  and z-score can miss this; autocorrelation is built to see it.

HOW IT IS COMPUTED (standard sample ACF at a given lag k)
  r_k = sum over t of (x_t - mean)(x_{t+k} - mean)        <- pairs k apart
        --------------------------------------------
        sum over t of (x_t - mean)^2                       <- total spread

  The numerator is large and positive when points k steps apart move together,
  negative when they move oppositely. Dividing by the total spread normalises
  the result into the [-1, +1] range so it is comparable across streams.

A constant window has zero spread (nothing to correlate), so we return 0.0.

NOTE ON DETECTION WIRING
  This signal is registered in the engine so it is computed, calibrated, and
  written to the feature store — but it is intentionally given no Scorer weight
  yet. Its near-term purpose is to be a richer input feature for the future ML
  layer; deliberately leaving its detection weight at zero avoids hand-tuning
  the current heuristic scorer.
"""

from threadforge.signals.base import Signal


class Autocorrelation(Signal):
    def __init__(self, window_size: int, lag: int = 1):
        super().__init__(window_size)
        if lag < 1:
            raise ValueError("lag must be at least 1")
        if lag >= window_size:
            raise ValueError("lag must be smaller than window_size")
        self.lag = lag

    def compute(self, window: list[float]) -> float:
        n = len(window)
        mean = sum(window) / n

        # total spread of the window around its mean (the normaliser)
        denom = sum((x - mean) ** 2 for x in window)
        if denom == 0.0:
            return 0.0  # constant window: no structure to correlate

        # sum of products of deviations `lag` steps apart
        num = sum(
            (window[i] - mean) * (window[i + self.lag] - mean)
            for i in range(n - self.lag)
        )
        return num / denom
