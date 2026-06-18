"""AdaptiveRobustCalibrator: a rolling alternative to the frozen RobustCalibrator.

`RobustCalibrator` learns its median +/- k*IQR band from the opening calibration
window and then **freezes** it for the whole run. That assumes "normal" never
changes — fine for a short, stationary file, but wrong for a long stream that
drifts to a new level. This version keeps a rolling window of the most recent
signal values and **recomputes the band continuously**, so its idea of normal
tracks the data the way the forecasting detector's residual z-score already does.

Drop-in: same ``observe`` / ``finalize`` / ``is_anomalous`` interface as
`RobustCalibrator`, and it self-updates inside ``is_anomalous`` (which the
`Detector` already calls once per step), so no detector changes are needed —
build a `Detector` with these instead and it adapts.

CAUSALITY: each value is judged against the band computed from *prior* values,
then folded into the rolling window — never against a band that already includes
itself.

THE TRADE-OFF (documented, not a bug): because every value rolls into the window,
a *sustained* anomaly will eventually shift the band and be re-learned as the new
normal. That is the correct behaviour for a genuine level shift (drift) but means
adaptive thresholds mask very long anomalies — the classic adaptive-threshold
tension. The project's forecasting detector found that a plain rolling estimate
self-regulates better than trying to exclude anomalies, so this mirrors that:
plain rolling, masking tail accepted. It is opt-in; the benchmarked default stays
the frozen `RobustCalibrator`.
"""

from __future__ import annotations

from collections import deque


class AdaptiveRobustCalibrator:
    def __init__(self, multiplier: float = 3.0, window: int = 400):
        self.multiplier = multiplier
        self.window = window
        self._values: deque[float] = deque(maxlen=window)
        self.upper: float | None = None
        self.lower: float | None = None

    def observe(self, value: float | None) -> None:
        """Seed the rolling window during the calibration period."""
        if value is not None:
            self._values.append(value)

    def _recompute(self) -> None:
        """Recompute the median +/- k*IQR band from the current rolling window."""
        vals = sorted(self._values)
        n = len(vals)
        if n == 0:
            return
        mid = n // 2
        median = vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2
        q1 = vals[n // 4]
        q3 = vals[(3 * n) // 4]
        iqr = q3 - q1
        self.upper = median + self.multiplier * iqr
        self.lower = median - self.multiplier * iqr

    def finalize(self) -> tuple[float, float]:
        """Set the initial band from the calibration window (then it keeps adapting)."""
        if not self._values:
            raise ValueError("no values observed during calibration")
        self._recompute()
        return self.lower, self.upper

    def is_anomalous(self, value: float) -> bool:
        """Judge ``value`` against the band from prior values, then roll it into the window."""
        if self.upper is None or self.lower is None:
            raise RuntimeError("call finalize() before is_anomalous()")
        verdict = value > self.upper or value < self.lower
        self._values.append(value)   # roll the window forward (causal: judged before adding)
        self._recompute()            # band tracks recent data
        return verdict

    @property
    def threshold(self) -> float | None:
        """Convenience: the current upper threshold (for display)."""
        return self.upper

    @property
    def sample_size(self) -> int:
        """Number of values currently in the rolling window."""
        return len(self._values)
