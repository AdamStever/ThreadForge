"""Inspect a feature-store database (SQLite or Parquet).

Usage:
    python scripts/inspect_store.py threadforge.db           # list all runs
    python scripts/inspect_store.py threadforge.db 1         # summarize run 1
    python scripts/inspect_store.py threadforge.db 1 zscore  # dump a signal series

The backend is auto-detected from the path: a directory is read as a Parquet
store, a file as a SQLite store. This is the read-side counterpart to
`run_detection.py --store`.
"""

import sys
from pathlib import Path

from threadforge.data import FeatureStore, ParquetFeatureStore


def open_store(path: str):
    """Pick the backend by path shape: directory => Parquet, file => SQLite."""
    if Path(path).is_dir():
        return ParquetFeatureStore(path)
    return FeatureStore(path)


def list_runs(store) -> None:
    runs = store.list_runs()
    if not runs:
        print("No runs recorded.")
        return
    print(f"{'run_id':>6}  {'started_at':<20}  source")
    print("-" * 60)
    for r in runs:
        print(f"{r['run_id']:>6}  {r['started_at']:<20}  {r['source']}")


def summarize(store, run_id: int) -> None:
    s = store.summarize_run(run_id)
    print(f"Run {s['run_id']}: {s['source']}")
    print(f"  started_at    {s['started_at']}")
    print(f"  time range    {s['time_start']} -> {s['time_end']}")
    print(f"  channels      {', '.join(s['channels'])}")
    print(f"  stream points {s['stream_points']}")
    print(f"  signal rows   {s['signal_rows']}")
    print(f"  signals       {', '.join(s['signals'])}")


def dump_signal(store, run_id: int, signal_name: str) -> None:
    series = store.read_signal(run_id, signal_name)
    if not series:
        print(f"No data for signal '{signal_name}' in run {run_id}.")
        return
    print(f"Run {run_id}, signal '{signal_name}' ({len(series)} points):")
    for ts, val in series:
        shown = "None" if val is None else f"{val:.4f}"
        print(f"  {ts}  {shown}")


def main(argv: list[str]) -> None:
    store_path = argv[0]
    with open_store(store_path) as store:
        if len(argv) == 1:
            list_runs(store)
        elif len(argv) == 2:
            summarize(store, int(argv[1]))
        else:
            dump_signal(store, int(argv[1]), argv[2])


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("usage: python scripts/inspect_store.py <db-path> [run_id] [signal_name]")
        raise SystemExit(1)
    main(sys.argv[1:])
