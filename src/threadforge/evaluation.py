"""Evaluation: compare detected events against labeled anomaly windows.

An event 'matches' a labeled window if its peak timestamp falls inside that
window. No external libraries — precision and recall are computed directly.

WHAT IS PRECISION?
  Of all the events we flagged, what fraction were real anomalies?
  precision = true_positives / (true_positives + false_positives)
  High precision => we don't cry wolf often.

WHAT IS RECALL?
  Of all the real anomaly windows in the data, what fraction did we catch?
  recall = true_positives / total_labeled_windows
  High recall => we don't miss real problems.

WHY BOTH?
  A detector that flags *everything* gets recall=1.0 but terrible precision.
  A detector that never flags anything gets precision=1.0 but recall=0.0.
  You need both to be high for the detector to be useful.

WHY USE THE PEAK TIMESTAMP FOR MATCHING?
  The peak is the single most extreme point in an event — it's the clearest
  signal that something was wrong. If it falls inside a labeled window, we
  count that as a hit. This is simpler than checking overlap between event
  spans and window spans, and avoids edge cases where events partially overlap.
"""

from datetime import datetime

from threadforge.detection.event import AnomalyEvent

_FMT = "%Y-%m-%d %H:%M:%S"


def _parse(ts: str) -> datetime:
    # tolerate optional fractional seconds in label files
    ts = ts.split(".")[0]
    return datetime.strptime(ts, _FMT)


def evaluate(
    events: list[AnomalyEvent],
    windows: list[tuple[str, str]],
) -> dict[str, float]:
    """Return a small report dict: tp, fp, misses, precision, recall."""
    parsed_windows = [(_parse(a), _parse(b)) for a, b in windows]

    matched_windows: set[int] = set()  # track which windows have been hit
    true_positives = 0
    false_positives = 0

    for ev in events:
        peak_t = _parse(ev.peak.timestamp)
        hit_idx = None
        # check if this event's peak falls inside any labeled window
        for i, (a, b) in enumerate(parsed_windows):
            if a <= peak_t <= b:
                hit_idx = i
                break
        if hit_idx is None:
            false_positives += 1  # flagged something outside any known window
        else:
            true_positives += 1
            matched_windows.add(hit_idx)  # mark this window as found

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
