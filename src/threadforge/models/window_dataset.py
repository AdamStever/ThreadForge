"""Raw-window dataset for the learned encoder.

The baseline model consumes 10 hand-crafted signals per step. The encoder
instead consumes the **raw value window** — the last `window_size` stream values,
z-normalized within the window — and learns its own features from them. This is
the "returns -> latent state" input the deep model is meant to digest.

Each warmed-up step becomes one row: a length-`window_size` normalized window,
labeled 1 if the step's timestamp falls in an anomaly window. Normalization is
per-window (subtract the window mean, divide by its std), which is causal and
makes the encoder scale-invariant across streams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from threadforge.data import parse_timestamp


@dataclass
class WindowExamples:
    filename: str
    timestamps: list[str]
    values: list[float]
    X: np.ndarray          # (n_rows, window_size) z-normalized raw windows
    y: np.ndarray          # (n_rows,) 0/1 labels
    window_size: int


def build_window_examples(
    filename: str,
    stream: list[tuple[str, float]],
    windows: list[tuple[str, str]],
    window_size: int = 30,
) -> WindowExamples:
    """Slice a stream into per-step normalized raw windows with labels."""
    parsed_windows = [(parse_timestamp(a), parse_timestamp(b)) for a, b in windows]
    values_all = [v for _, v in stream]
    ts_all = [t for t, _ in stream]

    rows: list[np.ndarray] = []
    timestamps: list[str] = []
    values: list[float] = []
    labels: list[int] = []

    for i in range(window_size - 1, len(stream)):
        w = np.asarray(values_all[i - window_size + 1: i + 1], dtype=float)
        sd = w.std()
        normalized = (w - w.mean()) / sd if sd > 0 else np.zeros_like(w)
        rows.append(normalized)
        timestamps.append(ts_all[i])
        values.append(values_all[i])
        t = parse_timestamp(ts_all[i])
        labels.append(1 if any(a <= t <= b for a, b in parsed_windows) else 0)

    X = np.array(rows, dtype=float) if rows else np.empty((0, window_size), dtype=float)
    y = np.array(labels, dtype=int)
    return WindowExamples(filename, timestamps, values, X, y, window_size)
