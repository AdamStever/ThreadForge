"""Detector: ties a SignalEngine + per-signal calibrators together over a stream.

Two-phase, fully causal:
  1. Calibration: consume the first `calib_steps` points through the engine,
     feeding each signal's output to its calibrator to learn a threshold.
  2. Detection: process every subsequent point, flag those where any signal
     exceeds its learned threshold, and group consecutive flags into AnomalyEvents.

`gap_steps` controls grouping: flags more than this many steps apart start a
new event.

WHY ONE CALIBRATOR PER SIGNAL?
  Each signal operates on a different scale — volatility might sit around 2.0
  while entropy sits around 1.5. A single shared threshold would make no sense.
  Each signal learns its own definition of "normal" independently.

WHY FLAG IF ANY SIGNAL EXCEEDS ITS THRESHOLD?
  Different anomaly types show up in different signals. A slow drift might only
  move momentum; a sudden spike might only move sharpness. Using all signals
  as independent detectors and flagging on any hit gives broader coverage than
  a single signal alone.

WHY NOT RESET THE SIGNAL WINDOWS BETWEEN PHASES?
  The signal's rolling window should carry over from calibration into
  detection — there's no gap in the real stream between those two phases.
  Resetting would throw away the last (window_size) calibration points and
  leave signals blind at the start of detection.
"""

from threadforge.engine import SignalEngine
from threadforge.detection.calibrator import Calibrator
from threadforge.detection.event import AnomalyEvent, FlaggedPoint


class Detector:
    def __init__(
        self,
        engine: SignalEngine,
        calibrators: dict[str, Calibrator],
        calib_steps: int = 600,
        gap_steps: int = 20,
    ):
        self.engine = engine
        self.calibrators = calibrators
        self.calib_steps = calib_steps
        self.gap_steps = gap_steps

    def run(self, stream: list[tuple[str, float]]) -> list[AnomalyEvent]:
        """Run both phases over the stream and return detected anomaly events."""

        # --- Phase 1: calibration ---
        # Feed calib_steps points through every signal and observe each output.
        self.engine.reset()
        for _, value in stream[:self.calib_steps]:
            outputs = self.engine.update(value)
            for name, sig_val in outputs.items():
                self.calibrators[name].observe(sig_val)
        for cal in self.calibrators.values():
            cal.finalize()

        # --- Phase 2: detection ---
        # For each point, check every signal. Flag if any exceeds its threshold.
        # The triggering signal with the highest value is recorded on the point.
        events: list[AnomalyEvent] = []
        last_flag_idx: int | None = None
        for i, (ts, value) in enumerate(stream[self.calib_steps:], start=self.calib_steps):
            outputs = self.engine.update(value)

            # find the most anomalous signal at this step, if any
            best_name: str | None = None
            best_val: float = 0.0
            for name, sig_val in outputs.items():
                if sig_val is None:
                    continue
                if self.calibrators[name].is_anomalous(sig_val):
                    if sig_val > best_val:
                        best_name = name
                        best_val = sig_val

            if best_name is None:
                continue  # no signal flagged this point

            point = FlaggedPoint(ts, value, best_name, best_val)
            if last_flag_idx is not None and i - last_flag_idx <= self.gap_steps:
                events[-1].add(point)
            else:
                ev = AnomalyEvent()
                ev.add(point)
                events.append(ev)
            last_flag_idx = i

        return events
