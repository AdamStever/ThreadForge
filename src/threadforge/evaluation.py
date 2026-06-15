"""Evaluation: compare detected events against labeled anomaly windows.

Two matching modes are supported:

  "peak"     An event matches a labeled window if its PEAK timestamp falls
             inside that window. Strict: an event that overlaps a window but
             whose single most-extreme point lands just outside is counted as
             a false positive. This is the original, conservative mode.

  "overlap"  An event matches a labeled window if its SPAN [start, end]
             intersects the window span at all. More forgiving and closer to
             NAB's own window-based scoring philosophy: catching any part of a
             labeled anomaly counts as catching it.

Both modes share identical bookkeeping — only the per-event match test differs.

WHAT IS PRECISION?
  Of all the events we flagged, what fraction were real anomalies?
  precision = true_positives / (true_positives + false_positives)
  High precision => we don't cry wolf often.

WHAT IS RECALL?
  Of all the real anomaly windows in the data, what fraction did we catch?
  recall = matched_windows / total_labeled_windows
  High recall => we don't miss real problems.

WHY BOTH?
  A detector that flags *everything* gets recall=1.0 but terrible precision.
  A detector that never flags anything gets precision=1.0 but recall=0.0.
  You need both to be high for the detector to be useful.

WHY OFFER OVERLAP MATCHING?
  Peak-only matching penalizes a detection that genuinely covered an anomaly
  just because its most-extreme point sat a few steps outside the labeled
  window. Overlap matching credits the detection for the part it caught, which
  is both fairer and consistent with how NAB scores against windows rather than
  single points.
"""

from datetime import datetime

from threadforge.detection.event import AnomalyEvent

_FMT = "%Y-%m-%d %H:%M:%S"

PEAK = "peak"
OVERLAP = "overlap"


def _parse(ts: str) -> datetime:
    # tolerate optional fractional seconds in label files
    ts = ts.split(".")[0]
    return datetime.strptime(ts, _FMT)


def _event_matches_window(
    ev: AnomalyEvent,
    window: tuple[datetime, datetime],
    mode: str,
) -> bool:
    """Return True if `ev` is considered a hit on `window` under `mode`."""
    a, b = window
    if mode == PEAK:
        peak_t = _parse(ev.peak.timestamp)
        return a <= peak_t <= b
    if mode == OVERLAP:
        start_t = _parse(ev.start)
        end_t = _parse(ev.end)
        # two spans [start, end] and [a, b] intersect iff each starts before
        # the other ends
        return start_t <= b and a <= end_t
    raise ValueError(f"unknown match mode: {mode!r} (use {PEAK!r} or {OVERLAP!r})")


def evaluate(
    events: list[AnomalyEvent],
    windows: list[tuple[str, str]],
    mode: str = PEAK,
) -> dict[str, float]:
    """Return a small report dict: tp, fp, misses, precision, recall.

    Args:
        events: detected anomaly events.
        windows: labeled (start, end) timestamp pairs.
        mode: "peak" (peak timestamp inside window) or "overlap" (event span
              intersects window span).
    """
    parsed_windows = [(_parse(a), _parse(b)) for a, b in windows]

    matched_windows: set[int] = set()  # track which windows have been hit
    true_positives = 0
    false_positives = 0

    for ev in events:
        matched_here: list[int] = []
        for i, window in enumerate(parsed_windows):
            if _event_matches_window(ev, window, mode):
                matched_here.append(i)
                if mode == PEAK:
                    break  # a single peak point sits in at most one window
        if not matched_here:
            false_positives += 1  # flagged something outside any known window
        else:
            true_positives += 1  # one event => at most one true positive
            matched_windows.update(matched_here)  # but it may cover several windows

    # windows we never matched = anomalies we missed
    misses = len(parsed_windows) - len(matched_windows)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    recall = (
        len(matched_windows) / len(parsed_windows) if parsed_windows else 0.0
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "misses": misses,
        "precision": precision,
        "recall": recall,
    }


def print_report(report: dict[str, float | int]) -> None:
    """Print an evaluation report as a small aligned table."""
    rows = [
        ("Metric", "Value"),
        ("-" * 16, "-" * 8),
        ("True positives", str(int(report["true_positives"]))),
        ("False positives", str(int(report["false_positives"]))),
        ("Misses", str(int(report["misses"]))),
        ("Precision", f"{report['precision']:.3f}"),
        ("Recall", f"{report['recall']:.3f}"),
    ]
    col_w = max(len(r[0]) for r in rows)
    for label, value in rows:
        print(f"  {label:<{col_w}}  {value}")
