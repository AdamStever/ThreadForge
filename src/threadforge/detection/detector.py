"""Detector: ties a SignalEngine + per-signal calibrators + a Scorer together
over a stream.

Two-phase, fully causal:
  1. Calibration: consume the first `calib_steps` points through the engine,
     feeding each signal's output to its calibrator to learn a threshold.
  2. Detection: process every subsequent point, build a per-signal anomaly flag
     dict, pass it through the Scorer, and group scoring steps into AnomalyEvents.

`gap_steps` controls grouping: flagged steps more than this many apart start a
new event.

An optional `store` (FeatureStore) can be passed to persist raw values and
signal scores to SQLite during the detection phase. If None, nothing is written.

WHY ONE CALIBRATOR PER SIGNAL?
  Each signal operates on a different scale — volatility might sit around 2.0
  while entropy sits around 1.5. A single shared threshold would make no sense.
  Each signal learns its own definition of "normal" independently.

WHY A SCORER INSTEAD OF "ANY SIGNAL TRIGGERS"?
  Different signal combinations carry different evidence strength. A Scorer
  assigns weights to solo signals and signal pairs so that high-confidence
  combinations (e.g. volatility + zscore both firing) contribute more than a
  single noisy signal firing alone. The weights are naive defaults for now and
  are designed to be replaced by learned values when the modeling layer arrives.

WHY NOT RESET THE SIGNAL WINDOWS BETWEEN PHASES?
  The signal's rolling window should carry over from calibration into
  detection — there's no gap in the real stream between those two phases.
  Resetting would throw away the last (window_size) calibration points and
  leave signals blind at the start of detection.
"""

from __future__ import annotations
import warnings
from typing import TYPE_CHECKING

from threadforge.engine import SignalEngine
from threadforge.data.stream import parse_timestamp
from threadforge.detection.calibrator import Calibrator
from threadforge.detection.scorer import Scorer
from threadforge.detection.event import AnomalyEvent, FlaggedPoint

if TYPE_CHECKING:
    from threadforge.data.store import FeatureStore


class Detector:
    def __init__(
        self,
        engine: SignalEngine,
        calibrators: dict[str, Calibrator],
        scorer: Scorer,
        calib_steps: int = 600,
        gap_steps: int = 20,
        store: "FeatureStore | None" = None,
        min_calib_samples: int = 30,
        gap_seconds: float | None = None,
    ):
        self.engine = engine
        self.calibrators = calibrators
        self.scorer = scorer
        self.calib_steps = calib_steps
        self.gap_steps = gap_steps
        self.store = store
        self.min_calib_samples = min_calib_samples
        # Gap grouping mode (limitation #4):
        #   gap_seconds is None  -> index-based: flags within gap_steps ROWS
        #                           are one event (the original behaviour).
        #   gap_seconds is set   -> time-based: flags within gap_seconds of
        #                           ELAPSED TIME are one event. Correct for
        #                           irregular streams (market calendars, sparse
        #                           telemetry) where rows aren't evenly spaced.
        self.gap_seconds = gap_seconds
        # Effective (non-None) calibration sample count per signal, populated
        # by run(). Smaller than calib_steps because of warm-up — limitation #5.
        self.calibration_samples: dict[str, int] = {}

    def run(self, stream: list[tuple[str, float]]) -> list[AnomalyEvent]:
        """Run both phases over the stream and return detected anomaly events."""

        # --- Phase 1: calibration ---
        self.engine.reset()
        for _, value in stream[:self.calib_steps]:
            outputs = self.engine.update(value)
            for name, sig_val in outputs.items():
                self.calibrators[name].observe(sig_val)
        for cal in self.calibrators.values():
            cal.finalize()

        # Record effective calibration size per signal and warn if any signal
        # was calibrated on too few real points. The first window_size steps of
        # each signal are None during warm-up, so the effective sample count is
        # always below calib_steps (limitation #5).
        self.calibration_samples = {
            name: cal.sample_size for name, cal in self.calibrators.items()
        }
        thin = {
            name: n
            for name, n in self.calibration_samples.items()
            if n < self.min_calib_samples
        }
        if thin:
            worst = min(thin.values())
            warnings.warn(
                f"thin calibration: {len(thin)} signal(s) calibrated on fewer "
                f"than min_calib_samples={self.min_calib_samples} points "
                f"(worst={worst}). Effective calibration is calib_steps minus "
                f"each signal's warm-up; raise calib_steps or lower window_size. "
                f"Thin signals: {thin}",
                stacklevel=2,
            )

        # --- Phase 2: detection ---
        events: list[AnomalyEvent] = []
        last_flag_idx: int | None = None
        last_flag_time = None  # parsed datetime of the previous flag (time mode)
        for i, (ts, value) in enumerate(stream[self.calib_steps:], start=self.calib_steps):
            outputs = self.engine.update(value)

            if self.store is not None:
                self.store.write_stream_value(ts, value)
                self.store.write_signal_scores(ts, outputs)

            flags: dict[str, bool] = {}
            for name, sig_val in outputs.items():
                if sig_val is None:
                    flags[name] = False
                else:
                    flags[name] = self.calibrators[name].is_anomalous(sig_val)

            if not self.scorer.is_anomalous(flags):
                continue

            best_name = max(
                (n for n, f in flags.items() if f),
                key=lambda n: abs(outputs.get(n) or 0.0),
                default=None,
            )
            best_val = abs(outputs.get(best_name) or 0.0) if best_name else 0.0

            point = FlaggedPoint(ts, value, best_name or "composite", best_val)

            # Decide whether this flag continues the previous event or starts a
            # new one — by elapsed time if gap_seconds is set, else by row count.
            this_time = None
            if self.gap_seconds is not None:
                try:
                    this_time = parse_timestamp(ts)
                except ValueError:
                    this_time = None  # unparseable ts => treat as a fresh event

            if self.gap_seconds is not None:
                same_event = (
                    last_flag_time is not None
                    and this_time is not None
                    and (this_time - last_flag_time).total_seconds() <= self.gap_seconds
                )
            else:
                same_event = (
                    last_flag_idx is not None
                    and i - last_flag_idx <= self.gap_steps
                )

            if same_event:
                events[-1].add(point)
            else:
                ev = AnomalyEvent()
                ev.add(point)
                events.append(ev)

            last_flag_idx = i
            last_flag_time = this_time

        return events
