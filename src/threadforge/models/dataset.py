"""Build a labeled supervised dataset from streams + anomaly windows.

For each file the signal engine is run over the stream, every fully-warmed-up
step becomes a feature row (the StateVector), and each row is labeled 1 if its
timestamp falls inside a labeled anomaly window, else 0. Feature extraction is
causal (the signals only see past/current values); the labels come from the
ground-truth windows.

Splitting is done **across files** (cross-file), not by shuffling rows: a whole
file is either train or test. This avoids leaking a file's own future into its
past and measures generalization to unseen streams.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from threadforge.data import parse_timestamp
from threadforge.presets import default_signal_engine
from threadforge.state import StateVector


@dataclass
class FileExamples:
    """Feature rows and labels extracted from one file, aligned by index."""
    filename: str
    timestamps: list[str]
    values: list[float]
    X: np.ndarray          # (n_rows, n_signals) feature matrix
    y: np.ndarray          # (n_rows,) 0/1 anomaly labels
    signal_names: list[str]


def build_file_examples(
    filename: str,
    stream: list[tuple[str, float]],
    windows: list[tuple[str, str]],
    window_size: int = 30,
) -> FileExamples:
    """Run the engine over one stream and produce labeled feature rows."""
    engine = default_signal_engine(window_size)
    names = list(engine._signals.keys())
    sv = StateVector(names)
    parsed_windows = [(parse_timestamp(a), parse_timestamp(b)) for a, b in windows]

    timestamps: list[str] = []
    values: list[float] = []
    rows: list[np.ndarray] = []
    labels: list[int] = []

    for ts, value in stream:
        outputs = engine.update(value)
        if not sv.is_ready(outputs):
            continue  # skip warm-up steps — features only partially defined
        timestamps.append(ts)
        values.append(value)
        rows.append(sv.vector(outputs))
        t = parse_timestamp(ts)
        labels.append(1 if any(a <= t <= b for a, b in parsed_windows) else 0)

    X = np.array(rows, dtype=float) if rows else np.empty((0, len(names)), dtype=float)
    y = np.array(labels, dtype=int)
    return FileExamples(filename, timestamps, values, X, y, names)


def cross_file_split(
    filenames: list[str], test_fraction: float = 0.3, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Partition files into (train, test) — disjoint, deterministic for a seed.

    At least one file is kept on each side.
    """
    files = sorted(filenames)
    if len(files) < 2:
        raise ValueError("need at least 2 files for a cross-file split")

    shuffled = files[:]
    random.Random(seed).shuffle(shuffled)

    n_test = max(1, round(len(shuffled) * test_fraction))
    n_test = min(n_test, len(shuffled) - 1)  # always leave at least one for train

    test = sorted(shuffled[:n_test])
    train = sorted(shuffled[n_test:])
    return train, test
