"""Detector: ties a single signal + calibrator together over a stream.

Two-phase, fully causal:
  1. Calibration: consume the first `calib_steps` points to learn a threshold.
  2. Detection: process every subsequent point, flag those whose signal exceeds
     the threshold, and group consecutive flags into AnomalyEvents.

`gap_steps` controls grouping: flags more than this many steps apart start a
new event.

WHY A FIXED STEP COUNT INSTEAD OF A FRACTION OF THE STREAM?
  A fraction (e.g. 15%) requires knowing the total stream length upfront.
  In a real system data arrives one point at a time — you never know how long
  the stream will be. A fixed step count (e.g. 600 points) mirrors what you'd
  actually do in production: "calibrate for the first 600 readings, then go
  live." No lookahead required.

WHY NOT RESET THE SIGNAL BETWEEN PHASES?
  The signal's rolling window should carry over from calibration into
  detection — there's no gap in the real stream between those two phases.
  Resetting would throw away the last (window_size) calibration points and
  leave the signal blind at the start of detection.
"""

from threadforge.signals.base import Signal
from threadforge.detection.calibrator import Calibrator
from threadforge.detection.event import AnomalyEvent, FlaggedPoint


class Detector:
    def __init__(
        self,
        signal: Signal,
        calibrator: Calibrator,
        calib_steps: int = 600,
        gap_steps: int = 20,
    ):
        self.signal = signal
        self.calibrator = calibrator
        self.calib_steps = calib_steps
        self.gap_steps = gap_steps

    def run(self, stream: list[tuple[str, float]]) -> list[AnomalyEvent]:
        """Run both phases over the stream and return detected anomaly events."""

        # --- Phase 1: calibration ---
        # Consume exactly calib_steps points — no knowledge of stream length needed.
        self.signal.reset()
        for _, value in stream[:self.calib_steps]:
            sig = self.signal.update(value)
            self.calibrator.observe(sig)  # None values are silently ignored
        self.calibrator.finalize()  # lock the threshold

        # --- Phase 2: detection ---
        # Process every point after the calibration window, one at a time.
        events: list[AnomalyEvent] = []
        last_flag_idx: int | None = None
        for i, (ts, value) in enumerate(stream[self.calib_steps:], start=self.calib_steps):
            sig = self.signal.update(value)
            if sig is None:
                continue  # still warming up the window — skip
            if self.calibrator.is_anomalous(sig):
                point = FlaggedPoint(ts, value, sig)
                # extend the current event if this flag is close to the last one,
                # otherwise start a new event
                if last_flag_idx is not None and i - last_flag_idx <= self.gap_steps:
                    events[-1].add(point)
                else:
                    ev = AnomalyEvent()
                    ev.add(point)
                    events.append(ev)
                last_flag_idx = i
        return events
