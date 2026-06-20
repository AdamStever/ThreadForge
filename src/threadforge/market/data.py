"""Load a real OHLCV price CSV into the standard (timestamp, value) stream.

Univariate by design (the engine is univariate): pick one column, default the
close. Calendar gaps (weekends/holidays) are expected for market data and are not
treated as errors — the detectors work on the sequence, not wall-clock spacing.
"""

from __future__ import annotations

import csv
from pathlib import Path


def load_ohlcv_csv(path: str | Path, column: str = "Close") -> list[tuple[str, float]]:
    """Read an OHLCV CSV into ``[(date, value)]`` using one column (default Close).

    Column and date headers are matched case-insensitively; rows with a missing or
    unparseable value are skipped.
    """
    rows: list[tuple[str, float]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:   # utf-8-sig strips a BOM
        reader = csv.DictReader(f)
        fields = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        date_key = fields.get("date") or fields.get("datetime") or fields.get("timestamp")
        val_key = fields.get(column.lower())
        if date_key is None or val_key is None:
            raise ValueError(f"{path}: need a date and a '{column}' column; found {reader.fieldnames}")
        for row in reader:
            try:
                value = float(row[val_key])
            except (TypeError, ValueError):
                continue
            rows.append((row[date_key], value))
    return rows
