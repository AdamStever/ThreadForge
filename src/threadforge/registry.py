"""Detector registry — the versioned ledger of detectors and their scores.

Champion–challenger promotion needs somewhere to record *which* detectors exist,
how each is configured, how each scored on the trusted metrics, and which one is
currently live (the champion). This is that ledger.

  - `register(...)`  records a detector version: its family name, hyperparameters,
    and metric scores (e.g. VUS-PR / Aff-F1), with an auto-assigned id + timestamp.
  - `best(metric)`   the highest-scoring record on a metric — the obvious promotion
    candidate.
  - `promote(id)`    points the **champion** at a record; `champion()` reads it back.
    Promotion is just moving this pointer, and any past record can be re-promoted,
    so rollback is free.

Persistence is a single human-readable JSON file written atomically (temp file +
rename), so a half-written registry can never be left behind. No new dependencies.

The registry versions the *detectors*; the feature store (`data/store.py`) remains
the system-of-record for the *data* they replay and train on — its `read_stream` /
`read_signal` already return replayable series. The two together let a challenger
be re-evaluated on exactly the data the champion saw.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DetectorRecord:
    """One versioned detector: what it is, how it's configured, how it scored."""
    id: int
    name: str                       # detector family, e.g. "ewma_forecast"
    params: dict = field(default_factory=dict)   # hyperparameters
    metrics: dict = field(default_factory=dict)  # {"VUS_PR": .., "Aff_F1": ..}
    created_at: str = ""
    notes: str = ""


class DetectorRegistry:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._records: list[DetectorRecord] = []
        self._champion_id: int | None = None
        self._load()

    # --- persistence ---

    def _load(self) -> None:
        if not self._path.exists():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._records = [DetectorRecord(**r) for r in data.get("records", [])]
        self._champion_id = data.get("champion_id")

    def _save(self) -> None:
        payload = {
            "records": [asdict(r) for r in self._records],
            "champion_id": self._champion_id,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)  # atomic: never leaves a half-written registry

    # --- writes ---

    def register(
        self,
        name: str,
        params: dict | None = None,
        metrics: dict | None = None,
        notes: str = "",
    ) -> DetectorRecord:
        """Record a new detector version and return it (with its assigned id)."""
        next_id = (max((r.id for r in self._records), default=0)) + 1
        record = DetectorRecord(
            id=next_id,
            name=name,
            params=params or {},
            metrics=metrics or {},
            created_at=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            notes=notes,
        )
        self._records.append(record)
        self._save()
        return record

    def promote(self, record_id: int) -> DetectorRecord:
        """Make ``record_id`` the champion (the live detector). Re-promotable for rollback."""
        record = self.get(record_id)
        self._champion_id = record.id
        self._save()
        return record

    # --- reads ---

    def all(self) -> list[DetectorRecord]:
        """All records in registration order."""
        return list(self._records)

    def get(self, record_id: int) -> DetectorRecord:
        for r in self._records:
            if r.id == record_id:
                return r
        raise KeyError(f"no detector record with id {record_id}")

    def latest(self) -> DetectorRecord | None:
        """The most recently registered record, or None if empty."""
        return self._records[-1] if self._records else None

    def best(self, metric: str = "VUS_PR") -> DetectorRecord | None:
        """The record with the highest value for ``metric`` (records lacking it are ignored)."""
        scored = [r for r in self._records if metric in r.metrics]
        return max(scored, key=lambda r: r.metrics[metric], default=None)

    def champion(self) -> DetectorRecord | None:
        """The currently-promoted (live) record, or None if nothing is promoted."""
        return self.get(self._champion_id) if self._champion_id is not None else None
