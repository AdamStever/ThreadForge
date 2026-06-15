"""Tests for the evaluation layer."""

import io
import sys

import pytest

from threadforge.evaluation import evaluate, print_report, PEAK, OVERLAP
from threadforge.detection.event import AnomalyEvent, FlaggedPoint


def _event(ts: str, signal_value: float = 1.0) -> AnomalyEvent:
    ev = AnomalyEvent()
    ev.add(FlaggedPoint(ts, 0.0, "volatility", signal_value))
    return ev


def _span_event(points: list[tuple[str, float]]) -> AnomalyEvent:
    """Build a multi-point event from (timestamp, signal_value) pairs.

    The point with the largest signal_value becomes the peak; start/end are
    the first/last timestamps in order.
    """
    ev = AnomalyEvent()
    for ts, sv in points:
        ev.add(FlaggedPoint(ts, 0.0, "volatility", sv))
    return ev


WINDOWS = [
    ("2024-01-01 00:10:00", "2024-01-01 00:20:00"),
    ("2024-01-01 01:00:00", "2024-01-01 01:10:00"),
]


def test_evaluate_perfect():
    events = [
        _event("2024-01-01 00:15:00"),
        _event("2024-01-01 01:05:00"),
    ]
    r = evaluate(events, WINDOWS)
    assert r["true_positives"] == 2
    assert r["false_positives"] == 0
    assert r["misses"] == 0
    assert r["precision"] == 1.0
    assert r["recall"] == 1.0


def test_evaluate_false_positive():
    events = [_event("2024-01-01 00:15:00"), _event("2024-01-01 00:30:00")]
    r = evaluate(events, WINDOWS)
    assert r["false_positives"] == 1
    assert r["misses"] == 1


def test_evaluate_miss():
    events = [_event("2024-01-01 00:15:00")]
    r = evaluate(events, WINDOWS)
    assert r["misses"] == 1
    assert r["recall"] < 1.0


def test_default_mode_is_peak():
    # default call should behave identically to explicit peak mode
    events = [_event("2024-01-01 00:15:00")]
    assert evaluate(events, WINDOWS) == evaluate(events, WINDOWS, mode=PEAK)


def test_overlap_credits_event_that_peaks_outside_window():
    # span 00:18 -> 00:30 overlaps window 00:10-00:20, but the peak (00:30)
    # falls outside it. peak mode = FP; overlap mode = TP.
    ev = _span_event([("2024-01-01 00:18:00", 1.0), ("2024-01-01 00:30:00", 9.0)])

    peak_r = evaluate([ev], WINDOWS, mode=PEAK)
    assert peak_r["true_positives"] == 0
    assert peak_r["false_positives"] == 1

    overlap_r = evaluate([ev], WINDOWS, mode=OVERLAP)
    assert overlap_r["true_positives"] == 1
    assert overlap_r["false_positives"] == 0
    assert overlap_r["recall"] == pytest.approx(0.5)


def test_overlap_no_intersection_is_false_positive():
    # span 00:30 -> 00:40 touches neither window
    ev = _span_event([("2024-01-01 00:30:00", 1.0), ("2024-01-01 00:40:00", 2.0)])
    r = evaluate([ev], WINDOWS, mode=OVERLAP)
    assert r["false_positives"] == 1
    assert r["true_positives"] == 0


def test_overlap_single_event_spanning_two_windows_counts_once_but_covers_both():
    # one long event from 00:15 to 01:05 overlaps BOTH windows
    ev = _span_event([("2024-01-01 00:15:00", 1.0), ("2024-01-01 01:05:00", 2.0)])
    r = evaluate([ev], WINDOWS, mode=OVERLAP)
    assert r["true_positives"] == 1   # one event => one true positive
    assert r["misses"] == 0           # but it covers both windows for recall
    assert r["recall"] == 1.0


def test_overlap_boundary_touch_counts():
    # event ends exactly at the window start — spans touch at a single instant
    ev = _span_event([("2024-01-01 00:05:00", 1.0), ("2024-01-01 00:10:00", 2.0)])
    r = evaluate([ev], WINDOWS, mode=OVERLAP)
    assert r["true_positives"] == 1


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        evaluate([_event("2024-01-01 00:15:00")], WINDOWS, mode="nonsense")


def test_print_report_outputs_table(capsys):
    report = {"true_positives": 2, "false_positives": 0, "misses": 0,
               "precision": 1.0, "recall": 1.0}
    print_report(report)
    out = capsys.readouterr().out
    assert "Precision" in out
    assert "1.000" in out
    assert "True positives" in out
