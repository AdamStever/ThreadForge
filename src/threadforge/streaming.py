"""Streaming runtime — drive the detector over a live or replayed feed.

Everything else in the pipeline is batch: read a whole file, score it, stop. This
module is the long-running form — points arrive one at a time and are scored,
grouped, and dispatched as they come, with O(1) state and no whole-stream buffer.
It is the substrate the later shadow-detection and champion/challenger work needs.

Three pieces:

  - `StreamRuntime` — the engine. `feed(timestamp, value)` pushes one point (for a
    live source that calls in); `run(source)` pulls an iterable to exhaustion (for
    replay). Each point yields a `StreamResult`; flagged points are grouped into
    `AnomalyEvent`s online; `on_result` / `on_event` callbacks are the sinks.
  - `OnlineEventGrouper` — groups consecutive flags into events by row gap, with
    the same rule as the batch `Detector` (a flag within `gap_steps` of the last
    continues the event; a later flag closes it and starts a new one).
  - `replay_csv` — a generator source that reads a NAB-style CSV row by row,
    optionally rate-limited to simulate a real-time feed.

Domain-agnostic: any iterable of `(timestamp, value)` is a valid source.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from threadforge.detection.event import AnomalyEvent, FlaggedPoint

SIGNAL_NAME = "forecast_residual"


@dataclass
class StreamResult:
    """The outcome of scoring one streamed point."""
    index: int
    timestamp: str
    value: float
    score: float
    is_anomaly: bool


class OnlineEventGrouper:
    """Group consecutive flagged points into events by row gap, incrementally.

    Mirrors the batch `Detector`: a flag within `gap_steps` rows of the previous
    flag continues the current event; a flag further out closes it (returned by
    `update`) and opens a new one. `flush` returns the final still-open event.
    """

    def __init__(self, gap_steps: int = 20):
        self.gap_steps = gap_steps
        self._current: AnomalyEvent | None = None
        self._last_flag_index: int | None = None

    def update(self, index: int, point: FlaggedPoint) -> AnomalyEvent | None:
        """Add a flagged point. Returns a just-closed event if this flag started a new one."""
        closed: AnomalyEvent | None = None
        if self._current is not None and index - self._last_flag_index > self.gap_steps:
            closed = self._current
            self._current = None
        if self._current is None:
            self._current = AnomalyEvent()
        self._current.add(point)
        self._last_flag_index = index
        return closed

    def flush(self) -> AnomalyEvent | None:
        """Close and return the final open event (call once the stream ends)."""
        ev = self._current
        self._current = None
        self._last_flag_index = None
        return ev


class StreamRuntime:
    """Score a stream point-by-point, grouping flags into events as they arrive."""

    def __init__(
        self,
        detector,
        threshold: float,
        gap_steps: int = 20,
        on_result: Callable[[StreamResult], None] | None = None,
        on_event: Callable[[AnomalyEvent], None] | None = None,
    ):
        """
        Args:
            detector: an online detector exposing ``update(value) -> score`` (e.g.
                `OnlineForecastResidualDetector`).
            threshold: a step is flagged when ``score >= threshold``.
            gap_steps: rows of separation that split one event from the next.
            on_result: called with every `StreamResult` (the per-point sink).
            on_event: called with each `AnomalyEvent` as it closes (the alert sink).
        """
        self.detector = detector
        self.threshold = threshold
        self.grouper = OnlineEventGrouper(gap_steps)
        self.on_result = on_result
        self.on_event = on_event
        self.events: list[AnomalyEvent] = []
        self._index = 0

    def feed(self, timestamp: str, value: float) -> StreamResult:
        """Push one point through the detector; fire sinks; return its result."""
        score = self.detector.update(value)
        is_anomaly = score >= self.threshold
        result = StreamResult(self._index, timestamp, value, score, is_anomaly)
        if self.on_result is not None:
            self.on_result(result)
        if is_anomaly:
            point = FlaggedPoint(timestamp, value, SIGNAL_NAME, score)
            closed = self.grouper.update(self._index, point)
            if closed is not None:
                self._emit(closed)
        self._index += 1
        return result

    def run(self, source: Iterable[tuple[str, float]]) -> list[AnomalyEvent]:
        """Pull a `(timestamp, value)` source to exhaustion, then flush. Returns events."""
        for timestamp, value in source:
            self.feed(timestamp, value)
        self.close()
        return self.events

    def close(self) -> None:
        """Flush the final open event (call after a `run`, or to end a `feed` session)."""
        ev = self.grouper.flush()
        if ev is not None:
            self._emit(ev)

    def _emit(self, event: AnomalyEvent) -> None:
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)


def replay_csv(path: str, rate: float | None = None) -> Iterator[tuple[str, float]]:
    """Yield ``(timestamp, value)`` from a NAB-style CSV one row at a time.

    Reads incrementally (a generator), so it models a real feed rather than
    loading the file. ``rate`` (rows per second), if set, sleeps between rows to
    simulate real time; the default streams as fast as possible.
    """
    delay = 1.0 / rate if rate else 0.0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts = row.get("timestamp")
            raw = row.get("value")
            if ts is None or raw is None:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if delay:
                time.sleep(delay)
            yield (ts, value)
