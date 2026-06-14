"""Entropy: how disordered the window is.

Bins the window's values into a few buckets and computes Shannon entropy
(-sum p*log(p)) over the bucket proportions. Low entropy => values cluster
in a few buckets (orderly); high entropy => values spread evenly (erratic).

This catches "the stream got choppy" even when the average level hasn't moved,
which volatility alone can miss. Uses only basic math — no external libraries.

WHAT IS SHANNON ENTROPY?
  Originally from information theory. Here we use it as a measure of
  "surprise" or disorder. If all values land in one bucket, entropy = 0
  (completely predictable). If values spread evenly across all buckets,
  entropy is at its maximum (maximally unpredictable).

EXAMPLE with bins=4:
  window = [1, 1, 1, 1]  => all in bucket 0 => entropy = 0  (orderly)
  window = [1, 2, 3, 4]  => one in each bucket => entropy = 2.0  (erratic)
"""

import math

from threadforge.signals.base import Signal


class Entropy(Signal):
    def __init__(self, window_size: int, bins: int = 8):
        super().__init__(window_size)
        if bins < 2:
            raise ValueError("bins must be at least 2")
        self.bins = bins

    def compute(self, window: list[float]) -> float:
        lo, hi = min(window), max(window)
        if hi == lo:
            return 0.0  # all identical => perfectly ordered, entropy = 0

        # divide the value range into equal-width buckets and count how many
        # window values fall into each one
        counts = [0] * self.bins
        span = hi - lo
        for x in window:
            idx = int((x - lo) / span * self.bins)
            if idx == self.bins:  # the max value lands exactly on the right edge
                idx = self.bins - 1
            counts[idx] += 1

        # Shannon entropy: -sum(p * log2(p)) over non-empty buckets
        # p is the proportion of window values in that bucket
        n = len(window)
        entropy = 0.0
        for c in counts:
            if c:
                p = c / n
                entropy -= p * math.log(p, 2)
        return entropy
