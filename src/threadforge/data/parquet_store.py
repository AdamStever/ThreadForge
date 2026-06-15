"""Columnar feature store backed by Apache Arrow / Parquet.

A scalable, drop-in-compatible alternative to the row-oriented SQLite
`FeatureStore`. It exposes the same write/read interface, so the `Detector` (and
anything else that takes a store) can target either backend unchanged.

WHY A SECOND BACKEND — ROW vs COLUMNAR
  SQLite stores data row by row and accepts a new row on every step — great for
  streaming inserts and ad-hoc queries. Parquet is *columnar*: each column's
  values sit together on disk, which compresses well and makes "read one signal
  across an entire run" cheap (only that column is read). The trade-off is that
  Parquet is written in batches, not row-by-row.

WHY BATCH WRITES (BUFFER THEN FLUSH)
  Because Parquet is batch-oriented, this store buffers a run's rows in memory
  and writes them out when the run ends — on the next `begin_run`, or when the
  context manager exits. The public interface is identical to the SQLite store;
  only the on-disk strategy differs.

LAYOUT
  <root>/
    runs.parquet                 run_id, source, started_at
    stream/run_<id>.parquet      timestamp, channel, value
    signals/run_<id>.parquet     timestamp, channel, signal_name, value

  Partitioning per run keeps each file self-contained and lets a run be read
  without scanning the others.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from threadforge.data.store import DEFAULT_CHANNEL


class ParquetFeatureStore:
    def __init__(self, root_dir: str | Path):
        self._root = Path(root_dir)
        self._run_id: int | None = None
        self._source: str | None = None
        self._started: str | None = None
        # in-memory buffers for the current (not-yet-flushed) run
        self._stream_rows: list[tuple[str, str, float]] = []
        self._signal_rows: list[tuple[str, str, str, float | None]] = []

    # --- paths ---

    @property
    def _runs_path(self) -> Path:
        return self._root / "runs.parquet"

    def _stream_path(self, run_id: int) -> Path:
        return self._root / "stream" / f"run_{run_id}.parquet"

    def _signals_path(self, run_id: int) -> Path:
        return self._root / "signals" / f"run_{run_id}.parquet"

    # --- context manager ---

    def __enter__(self) -> "ParquetFeatureStore":
        (self._root / "stream").mkdir(parents=True, exist_ok=True)
        (self._root / "signals").mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *_) -> None:
        self._flush()

    # --- run management ---

    def begin_run(self, source: str) -> int:
        self._flush()  # persist any previous run before starting a new one
        self._run_id = self._next_run_id()
        self._source = source
        self._started = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._stream_rows = []
        self._signal_rows = []
        return self._run_id

    @property
    def run_id(self) -> int | None:
        return self._run_id

    def _next_run_id(self) -> int:
        if self._runs_path.exists():
            ids = pq.read_table(self._runs_path, columns=["run_id"]).column("run_id").to_pylist()
            return (max(ids) + 1) if ids else 1
        return 1

    # --- writes (buffered in memory until flush) ---

    def write_stream_value(
        self, timestamp: str, value: float, channel: str = DEFAULT_CHANNEL
    ) -> None:
        self._stream_rows.append((timestamp, channel, float(value)))

    def write_signal_scores(
        self,
        timestamp: str,
        scores: dict[str, float | None],
        channel: str = DEFAULT_CHANNEL,
    ) -> None:
        for name, val in scores.items():
            self._signal_rows.append(
                (timestamp, channel, name, None if val is None else float(val))
            )

    def _flush(self) -> None:
        """Write the buffered current run to Parquet, then clear the buffers."""
        if self._run_id is None:
            return

        if self._stream_rows:
            ts, ch, val = zip(*self._stream_rows)
            pq.write_table(
                pa.table({"timestamp": list(ts), "channel": list(ch), "value": list(val)}),
                self._stream_path(self._run_id),
            )
        if self._signal_rows:
            ts, ch, name, val = zip(*self._signal_rows)
            pq.write_table(
                pa.table({
                    "timestamp": list(ts),
                    "channel": list(ch),
                    "signal_name": list(name),
                    "value": list(val),
                }),
                self._signals_path(self._run_id),
            )

        self._append_run_meta()
        self._stream_rows = []
        self._signal_rows = []
        self._run_id = None  # mark flushed so __exit__ won't write it twice

    def _append_run_meta(self) -> None:
        row = pa.table({
            "run_id": [self._run_id],
            "source": [self._source],
            "started_at": [self._started],
        })
        if self._runs_path.exists():
            row = pa.concat_tables([pq.read_table(self._runs_path), row])
        pq.write_table(row, self._runs_path)

    # --- read helpers (mirror the SQLite FeatureStore) ---

    def list_runs(self) -> list[dict]:
        if not self._runs_path.exists():
            return []
        t = pq.read_table(self._runs_path).sort_by("run_id")
        return [
            {"run_id": r, "source": s, "started_at": a}
            for r, s, a in zip(
                t.column("run_id").to_pylist(),
                t.column("source").to_pylist(),
                t.column("started_at").to_pylist(),
            )
        ]

    def read_stream(
        self, run_id: int, channel: str = DEFAULT_CHANNEL
    ) -> list[tuple[str, float]]:
        p = self._stream_path(run_id)
        if not p.exists():
            return []
        t = pq.read_table(p, columns=["timestamp", "channel", "value"])
        return [
            (ts, v)
            for ts, ch, v in zip(
                t.column("timestamp").to_pylist(),
                t.column("channel").to_pylist(),
                t.column("value").to_pylist(),
            )
            if ch == channel
        ]

    def read_signal(
        self, run_id: int, signal_name: str, channel: str = DEFAULT_CHANNEL
    ) -> list[tuple[str, float | None]]:
        p = self._signals_path(run_id)
        if not p.exists():
            return []
        t = pq.read_table(p, columns=["timestamp", "channel", "signal_name", "value"])
        return [
            (ts, v)
            for ts, ch, nm, v in zip(
                t.column("timestamp").to_pylist(),
                t.column("channel").to_pylist(),
                t.column("signal_name").to_pylist(),
                t.column("value").to_pylist(),
            )
            if ch == channel and nm == signal_name
        ]

    def signal_names(self, run_id: int, channel: str = DEFAULT_CHANNEL) -> list[str]:
        p = self._signals_path(run_id)
        if not p.exists():
            return []
        t = pq.read_table(p, columns=["channel", "signal_name"])
        names = {
            nm
            for ch, nm in zip(t.column("channel").to_pylist(), t.column("signal_name").to_pylist())
            if ch == channel
        }
        return sorted(names)

    def channels(self, run_id: int) -> list[str]:
        p = self._stream_path(run_id)
        if not p.exists():
            return []
        chans = set(pq.read_table(p, columns=["channel"]).column("channel").to_pylist())
        return sorted(chans)

    def summarize_run(self, run_id: int) -> dict:
        runs = {r["run_id"]: r for r in self.list_runs()}
        if run_id not in runs:
            raise KeyError(f"no run with id {run_id}")
        meta = runs[run_id]
        stream = self.read_stream(run_id)
        timestamps = [ts for ts, _ in stream]
        n_signal = 0
        sp = self._signals_path(run_id)
        if sp.exists():
            n_signal = pq.read_metadata(sp).num_rows
        return {
            "run_id": run_id,
            "source": meta["source"],
            "started_at": meta["started_at"],
            "channels": self.channels(run_id),
            "signals": self.signal_names(run_id),
            "stream_points": len(stream),
            "signal_rows": n_signal,
            "time_start": timestamps[0] if timestamps else None,
            "time_end": timestamps[-1] if timestamps else None,
        }
