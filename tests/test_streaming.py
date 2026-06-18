"""Tests for the streaming runtime and the online forecasting detector."""

import numpy as np
import pytest

from threadforge.detection.forecast_detector import ForecastResidualDetector
from threadforge.detection.online_forecast import OnlineForecastResidualDetector
from threadforge.detection.event import FlaggedPoint
from threadforge.streaming import OnlineEventGrouper, StreamRuntime, replay_csv


# --- online detector equivalence -------------------------------------------

def test_online_matches_batch_exactly():
    """The online detector must reproduce the batch detector's scores point-for-point."""
    rng = np.random.RandomState(0)
    values = rng.rand(1200).tolist()
    values[600] += 8.0  # plant an anomaly
    stream = [(str(i), v) for i, v in enumerate(values)]

    batch = ForecastResidualDetector()
    probation = batch.probation(len(stream))
    batch_scores = batch.scores(stream)

    online = OnlineForecastResidualDetector(
        ewma_alpha=batch.ewma_alpha,
        resid_window=batch.resid_window,
        probation=probation,
        min_history=batch.min_history,
    )
    online_scores = [online.update(v) for _, v in stream]

    assert online_scores == pytest.approx(batch_scores, abs=1e-12)


def test_online_reset_clears_state():
    d = OnlineForecastResidualDetector(probation=0, min_history=2, resid_window=10)
    for v in [1.0, 2.0, 3.0, 50.0]:
        d.update(v)
    d.reset()
    assert d._i == 0 and d._ewma is None and len(d._history) == 0


# --- event grouping ---------------------------------------------------------

def _pt(i: int) -> FlaggedPoint:
    return FlaggedPoint(str(i), 1.0, "forecast_residual", 5.0)


def test_grouper_merges_within_gap_and_splits_beyond():
    g = OnlineEventGrouper(gap_steps=5)
    assert g.update(0, _pt(0)) is None
    assert g.update(3, _pt(3)) is None        # within gap -> same event
    closed = g.update(20, _pt(20))            # beyond gap -> closes the first
    assert closed is not None
    assert closed.size == 2 and closed.start == "0" and closed.end == "3"
    tail = g.flush()
    assert tail.size == 1 and tail.start == "20"


def test_grouper_flush_when_empty():
    assert OnlineEventGrouper().flush() is None


# --- runtime ----------------------------------------------------------------

class _FakeDetector:
    """Replays a fixed score sequence so runtime behaviour is deterministic."""
    def __init__(self, scores):
        self._scores = list(scores)
        self._i = 0

    def update(self, value: float) -> float:
        s = self._scores[self._i]
        self._i += 1
        return s


def test_runtime_groups_events_and_fires_callbacks():
    # event A at idx 5-7, long quiet gap, event B at idx 38
    scores = [0.0] * 5 + [10.0, 10.0, 10.0] + [0.0] * 30 + [10.0]
    results, events = [], []
    rt = StreamRuntime(
        _FakeDetector(scores), threshold=5.0, gap_steps=20,
        on_result=results.append, on_event=events.append,
    )
    returned = rt.run([(str(i), float(i)) for i in range(len(scores))])

    assert len(results) == len(scores)
    assert sum(r.is_anomaly for r in results) == 4
    assert returned == events                 # run() returns the same events the sink saw
    assert [e.size for e in events] == [3, 1]
    assert events[0].start == "5" and events[0].end == "7"
    assert events[1].start == "38"


def test_runtime_feed_is_incremental():
    """feed() can be called point-by-point (push API) and returns each result."""
    rt = StreamRuntime(_FakeDetector([0.0, 9.0]), threshold=5.0)
    r0 = rt.feed("t0", 1.0)
    r1 = rt.feed("t1", 2.0)
    assert (r0.index, r0.is_anomaly) == (0, False)
    assert (r1.index, r1.is_anomaly) == (1, True)
    rt.close()
    assert len(rt.events) == 1


# --- replay source ----------------------------------------------------------

def test_replay_csv_streams_rows_and_skips_bad(tmp_path):
    p = tmp_path / "feed.csv"
    p.write_text(
        "timestamp,value\n"
        "2014-01-01 00:00:00,1.0\n"
        "2014-01-01 00:05:00,2.5\n"
        "bad,notanumber\n"          # unparseable value -> skipped
        "2014-01-01 00:10:00,3.0\n",
        encoding="utf-8",
    )
    rows = list(replay_csv(str(p)))
    assert rows == [
        ("2014-01-01 00:00:00", 1.0),
        ("2014-01-01 00:05:00", 2.5),
        ("2014-01-01 00:10:00", 3.0),
    ]
