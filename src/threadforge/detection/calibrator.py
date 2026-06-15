"""Calibrator: learns a 'normal' threshold from an initial calibration period.

Watches the first fraction of a signal's values, then freezes a threshold at
mean + k*std. This mirrors NAB's autocalibration idea: use only early data to
decide what normal looks like, so detection stays causal (no peeking ahead).

WHY CALIBRATE AT ALL?
  Different streams have wildly different scales — a CPU signal might range
  0-100, a network signal might range 0-10,000,000. Hard-coding a threshold
  number would break the moment you switch files. Instead, we let the early
  part of each stream teach us what "normal" looks like for *that* stream.

WHY mean + k*std?
  Under a roughly normal distribution, ~99.99% of values fall within 4
  standard deviations of the mean. So setting k=4 means we only flag
  something as anomalous if it would be extremely rare in a normal world.
  k is tunable in config/default.json (threshold_multiplier).

WHY FREEZE THE THRESHOLD AFTER CALIBRATION?
  If we kept updating the threshold as anomalies arrived, the system would
  gradually learn that anomalies are "normal" and stop flagging them.
  Freezing it preserves the definition of normal from the clean early data.
"""

import math


class Calibrator:
    def __init__(self, multiplier: float = 4.0):
        self.multiplier = multiplier
        self._values: list[float] = []
        self.threshold: float | None = None  # None until finalize() is called

    def observe(self, value: float) -> None:
        """Record a signal value during the calibration period."""
        if value is not None:
            self._values.append(value)

    def finalize(self) -> float:
        """Freeze and return the threshold from observed values.

        After this is called, the threshold is locked — observe() can still
        be called but its values are ignored.
        """
        if not self._values:
            raise ValueError("no values observed during calibration")
        n = len(self._values)
        mean = sum(self._values) / n
        # population variance (not sample) — we're summarising the calibration
        # window itself, not estimating an underlying distribution
        var = sum((x - mean) ** 2 for x in self._values) / n
        std = math.sqrt(var)
        self.threshold = mean + self.multiplier * std
        return self.threshold

    def is_anomalous(self, value: float) -> bool:
        """Return True if value exceeds the learned threshold."""
        if self.threshold is None:
            raise RuntimeError("call finalize() before is_anomalous()")
        return value > self.threshold

    @property
    def sample_size(self) -> int:
        """Number of real (non-None) values actually used for calibration.

        Smaller than the requested calib_steps because each signal emits None
        during its rolling-window warm-up (see limitation #5).
        """
        return len(self._values)
