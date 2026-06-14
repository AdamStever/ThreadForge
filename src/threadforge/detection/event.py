"""AnomalyEvent: a contiguous run of flagged points grouped into one event.

Grouping consecutive flags into events (rather than reporting every single
flagged point) is the simplest form of alert de-duplication — it keeps one
sustained anomaly from producing dozens of separate alerts.

ANALOGY:
  Think of a fire alarm. If smoke lingers for 10 minutes you want one alert
  saying "fire from 3:00 to 3:10", not 600 individual "smoke detected" pings
  (one per second). AnomalyEvent is that grouping container.

WHAT IS THE PEAK?
  Within one event some points will have a higher signal value than others —
  the peak is the single most extreme flagged point. It's useful both for
  reporting ("the worst moment was at 2:47") and for evaluation (we check
  whether the peak timestamp falls inside a labeled anomaly window).
"""

from dataclasses import dataclass, field


@dataclass
class FlaggedPoint:
    """One individual time step that was flagged as anomalous."""
    timestamp: str
    value: float        # the raw stream value at this point
    signal_value: float # the signal (e.g. volatility) that triggered the flag


@dataclass
class AnomalyEvent:
    """A group of consecutive FlaggedPoints treated as a single anomaly."""
    points: list[FlaggedPoint] = field(default_factory=list)

    def add(self, point: FlaggedPoint) -> None:
        self.points.append(point)

    @property
    def start(self) -> str:
        """Timestamp of the first flagged point in this event."""
        return self.points[0].timestamp

    @property
    def end(self) -> str:
        """Timestamp of the last flagged point in this event."""
        return self.points[-1].timestamp

    @property
    def size(self) -> int:
        """Number of flagged points in this event."""
        return len(self.points)

    @property
    def peak(self) -> FlaggedPoint:
        """The flagged point with the largest signal value."""
        return max(self.points, key=lambda p: p.signal_value)
