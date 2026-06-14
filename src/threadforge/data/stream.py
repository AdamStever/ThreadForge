"""Data layer: turn a CSV file into a stream of (timestamp, value) pairs.

Reads one row at a time, the way data would arrive from a live feed. Handles
the messiness real NAB files can have: blank lines and unparseable rows are
skipped rather than crashing the run.

WHY RETURN A LIST INSTEAD OF A GENERATOR?
  The target files are small (~1K-22K rows). Returning a list lets the caller
  pass the same stream to multiple phases (calibration then detection) without
  re-reading the file. For very large files a generator would be more memory
  efficient.
"""

import csv
from datetime import datetime

_FMT = "%Y-%m-%d %H:%M:%S"


def stream_csv(path: str) -> list[tuple[str, float]]:
    """Read a NAB-style CSV (columns: timestamp,value) into a list of rows."""
    rows: list[tuple[str, float]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp")
            raw = row.get("value")
            if ts is None or raw is None:
                continue  # skip rows missing either column
            try:
                value = float(raw)
            except ValueError:
                continue  # skip header repeats or malformed rows
            rows.append((ts, value))
    return rows


def check_timestamps(
    rows: list[tuple[str, float]],
    gap_multiplier: float = 3.0,
) -> list[dict]:
    """Scan a stream for missing or irregularly-spaced timestamps.

    HOW IT WORKS:
      1. Parse every timestamp into a datetime object.
      2. Compute the gap (in seconds) between every consecutive pair of rows.
      3. Find the median gap — that's our definition of the "expected" interval.
      4. Flag any gap that is more than gap_multiplier × median as irregular.

    WHY THE MEDIAN AND NOT THE MEAN?
      A single huge gap (like a server outage) would drag the mean up, making
      everything look normal by comparison. The median is resistant to that
      kind of outlier — it stays close to the typical interval.

    Returns a list of warning dicts — empty if the stream looks uniform.
    """
    if len(rows) < 2:
        return []

    def _parse(ts: str) -> datetime:
        return datetime.strptime(ts.split(".")[0], _FMT)

    try:
        times = [_parse(ts) for ts, _ in rows]
    except ValueError:
        return [{"type": "parse_error", "detail": "one or more timestamps could not be parsed"}]

    # compute the gap in seconds between each consecutive pair
    gaps = [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]

    # find the median gap
    sorted_gaps = sorted(gaps)
    mid = len(sorted_gaps) // 2
    median = sorted_gaps[mid] if len(sorted_gaps) % 2 else (sorted_gaps[mid - 1] + sorted_gaps[mid]) / 2

    warnings = []
    if median == 0:
        return [{"type": "parse_error", "detail": "all timestamps are identical"}]

    threshold = gap_multiplier * median
    for i, g in enumerate(gaps):
        if g > threshold:
            warnings.append({
                "type": "gap",
                "after_index": i,
                "after_timestamp": rows[i][0],
                "before_timestamp": rows[i + 1][0],
                "gap_seconds": g,
                "median_seconds": median,
                "multiple": round(g / median, 1),  # e.g. 24.0 => gap is 24x the normal interval
            })

    return warnings
