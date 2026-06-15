"""RobustCalibrator: a more resistant alternative to the mean+k*std calibrator.

Uses the median and inter-quartile range (IQR) instead of mean and std to set
the anomaly threshold. This makes calibration far less sensitive to outliers in
the calibration window — if an anomaly happens to occur during calibration, it
no longer drags the threshold up and blinds the detector to future anomalies.

Also supports two-tailed detection: flags values that are either too high OR
too low relative to the calibration distribution. This catches flatlines,
frozen feeds, and downward spikes that the one-tailed calibrator misses.

WHY MEDIAN AND IQR INSTEAD OF MEAN AND STD?
  Mean and std are both sensitive to outliers — one extreme value shifts both
  significantly. Median is the middle value of the sorted distribution; it
  doesn't move when one end of the distribution is pulled. IQR (the gap between
  the 25th and 75th percentile) measures spread the same way. Together they
  describe "typical" behaviour robustly, even if the calibration window is dirty.

WHY TWO-TAILED?
  mean + k*std only flags unusually HIGH signal values. But anomalies can also
  manifest as unusually LOW values — a CPU that suddenly flatlines at 0%,
  network traffic that drops to zero, a signal that freezes. The lower bound
  catches these. Each tail uses the same k multiplier applied symmetrically.
"""

import math


class RobustCalibrator:
    def __init__(self, multiplier: float = 3.0):
        self.multiplier = multiplier
        self._values: list[float] = []
        self.upper: float | None = None
        self.lower: float | None = None

    def observe(self, value: float | None) -> None:
        """Record a signal value during the calibration period."""
        if value is not None:
            self._values.append(value)

    def finalize(self) -> tuple[float, float]:
        """Compute and freeze the upper and lower thresholds."""
        if not self._values:
            raise ValueError("no values observed during calibration")

        sorted_vals = sorted(self._values)
        n = len(sorted_vals)

        # median
        mid = n // 2
        median = sorted_vals[mid] if n % 2 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2

        # inter-quartile range
        q1 = sorted_vals[n // 4]
        q3 = sorted_vals[(3 * n) // 4]
        iqr = q3 - q1

        self.upper = median + self.multiplier * iqr
        self.lower = median - self.multiplier * iqr
        return self.lower, self.upper

    def is_anomalous(self, value: float) -> bool:
        """Return True if value falls outside the [lower, upper] band."""
        if self.upper is None or self.lower is None:
            raise RuntimeError("call finalize() before is_anomalous()")
        return value > self.upper or value < self.lower

    @property
    def threshold(self) -> float | None:
        """Convenience property — returns the upper threshold for display."""
        return self.upper

    @property
    def sample_size(self) -> int:
        """Number of real (non-None) values actually used for calibration.

        This is the *effective* calibration size, which is smaller than the
        requested calib_steps because each signal emits None during its
        rolling-window warm-up (see limitation #5). Use it to tell whether the
        threshold was learned from enough data to be meaningful.
        """
        return len(self._values)
