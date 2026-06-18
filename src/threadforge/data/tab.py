"""TAB benchmark loader: long-format CSV -> (stream, labels) + corpus metadata.

The TAB anomaly-detection corpus stores each series in **long format**: a single
CSV stacks one or more data channels followed by a ``label`` channel, told apart
by the ``cols`` column. Each channel's ``date`` index restarts at 1.

    date,data,cols
    1,-142.9,channel_1     <- data channel
    2,-164.9,channel_1
    ...
    1,0.0,label            <- 0/1 anomaly label, same length as the data channel
    2,0.0,label

`DETECT_META.csv` indexes every file (dataset, train/test lengths, whether the
series is univariate, anomaly rate). This module reads both: the metadata index
and a single series into the ``[(timestamp, value)]`` stream the detectors
already consume, plus the aligned 0/1 label list the TAB scorers need.

Domain concerns stay here in the data layer — the core pipeline never sees the
TAB-specific file shape.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

LABEL_CHANNEL = "label"


@dataclass(frozen=True)
class TabMeta:
    """One row of ``DETECT_META.csv`` — describes a single series."""
    file_name: str
    dataset_name: str
    if_univariate: bool
    train_lens: int
    test_lens: int
    time_steps: int
    anomaly_rate: float


def load_tab_meta(meta_path: str | Path) -> list[TabMeta]:
    """Read ``DETECT_META.csv`` into a list of :class:`TabMeta` records."""
    out: list[TabMeta] = []
    with open(meta_path, newline="") as f:
        for row in csv.DictReader(f):
            out.append(TabMeta(
                file_name=row["file_name"],
                dataset_name=row["dataset_name"],
                if_univariate=row["if_univariate"].strip().upper() == "TRUE",
                train_lens=int(row["train_lens"]),
                test_lens=int(row["test_lens"]),
                time_steps=int(row["time_steps"]),
                anomaly_rate=float(row["anomaly_rate"]),
            ))
    return out


def _date_key(d: str):
    """Sort key for the ``date`` column, robust to either form TAB uses.

    Most files use an integer index (1, 2, 3, …) but some (e.g. GAIA) use ISO
    timestamps like ``2019-11-16 22:00:00``. Returning a ``(rank, value)`` tuple
    keeps integers and strings from ever being compared to each other, and ISO
    timestamps sort correctly lexically.
    """
    s = d.strip()
    return (0, int(s)) if s.lstrip("-").isdigit() else (1, s)


def load_tab_csv(path: str | Path) -> tuple[dict[str, list[float]], list[int]]:
    """Parse a long-format TAB CSV into ``{channel: values}`` plus the label list.

    Channel values are returned ordered by their ``date`` column (integer index or
    ISO timestamp). Raises if the file has no ``label`` channel.
    """
    raw: dict[str, list[tuple[str, float]]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["cols"]
            raw.setdefault(name, []).append((row["date"], float(row["data"])))

    channels: dict[str, list[float]] = {}
    labels: list[int] | None = None
    for name, pairs in raw.items():
        pairs.sort(key=lambda p: _date_key(p[0]))
        values = [v for _, v in pairs]
        if name == LABEL_CHANNEL:
            labels = [int(round(v)) for v in values]
        else:
            channels[name] = values

    if labels is None:
        raise ValueError(f"{path}: no '{LABEL_CHANNEL}' channel found")
    return channels, labels


def load_tab_univariate(path: str | Path) -> tuple[list[tuple[str, float]], list[int]]:
    """Load one univariate TAB series as ``([(timestamp, value)], labels)``.

    The timestamp is the 1-based integer index rendered as a string, matching the
    ``[(timestamp, value)]`` contract the detectors expect. Raises if the file
    holds more than one data channel (i.e. it is multivariate).
    """
    channels, labels = load_tab_csv(path)
    data_names = [n for n in channels if n != LABEL_CHANNEL]
    if len(data_names) != 1:
        raise ValueError(
            f"{path}: expected one data channel, found {len(data_names)} {data_names} "
            "(use a multivariate loader for this series)"
        )
    values = channels[data_names[0]]
    if len(values) != len(labels):
        raise ValueError(f"{path}: value/label length mismatch {len(values)} vs {len(labels)}")
    stream = [(str(i + 1), v) for i, v in enumerate(values)]
    return stream, labels
