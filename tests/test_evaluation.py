"""Tests for the evaluation layer."""

import io
import sys

from threadforge.evaluation import evaluate, print_report
from threadforge.detection.event import AnomalyEvent, FlaggedPoint


def _event(ts: str, signal_value: float = 1.0) -> AnomalyEvent:
    ev = AnomalyEvent()
    ev.add(FlaggedPoint(ts, 0.0, "volatility", signal_value))
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


def test_print_report_outputs_table(capsys):
    report = {"true_positives": 2, "false_positives": 0, "misses": 0,
               "precision": 1.0, "recall": 1.0}
    print_report(report)
    out = capsys.readouterr().out
    assert "Precision" in out
    assert "1.000" in out
    assert "True positives" in out
