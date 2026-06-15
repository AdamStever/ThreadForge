"""Baseline supervised anomaly model over the signal features.

A scikit-learn pipeline (standardize -> logistic regression) trained on the
labeled feature rows. Logistic regression over the signals is the learned
counterpart of the hand-weighted Scorer: instead of guessing weights, it fits
them from data. `class_weight="balanced"` handles the heavy normal-vs-anomaly
imbalance.

To stay comparable with the heuristic pipeline, per-step predictions are grouped
into AnomalyEvents (consecutive positives within `gap_steps`) and scored with the
same `evaluate()` harness and overlap matching.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from threadforge.detection.event import AnomalyEvent, FlaggedPoint
from threadforge.evaluation import evaluate, OVERLAP
from threadforge.models.dataset import FileExamples


def train(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> Pipeline:
    """Fit the baseline classifier on feature rows X with labels y.

    `C` is the inverse-regularization strength of the logistic regression
    (smaller = stronger regularization); the hyperparameter search tunes it.
    """
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=C)),
    ])
    model.fit(X, y)
    return model


def group_predictions(
    timestamps: list[str],
    values: list[float],
    preds: np.ndarray,
    scores: np.ndarray,
    gap_steps: int = 20,
) -> list[AnomalyEvent]:
    """Group consecutive positive predictions into anomaly events.

    Mirrors the Detector's index-based grouping so model output is scored the
    same way as the heuristic's. `scores` (predicted anomaly probability) becomes
    the flagged point's value so the event peak is its most confident point.
    """
    events: list[AnomalyEvent] = []
    last_idx: int | None = None
    for i, (ts, val, pred, score) in enumerate(zip(timestamps, values, preds, scores)):
        if not pred:
            continue
        point = FlaggedPoint(ts, val, "model", float(score))
        if last_idx is not None and i - last_idx <= gap_steps:
            events[-1].add(point)
        else:
            ev = AnomalyEvent()
            ev.add(point)
            events.append(ev)
        last_idx = i
    return events


def predict_events(model: Pipeline, fe: FileExamples, gap_steps: int = 20) -> list[AnomalyEvent]:
    """Predict on one file's rows and return grouped anomaly events."""
    if fe.X.shape[0] == 0:
        return []
    preds = model.predict(fe.X)
    scores = model.predict_proba(fe.X)[:, 1]
    return group_predictions(fe.timestamps, fe.values, preds, scores, gap_steps)


def evaluate_model(
    model: Pipeline,
    file_examples: list[FileExamples],
    windows_by_file: dict[str, list[tuple[str, str]]],
    gap_steps: int = 20,
    mode: str = OVERLAP,
) -> dict[str, float]:
    """Aggregate precision / recall / F1 of the model across files."""
    precisions: list[float] = []
    recalls: list[float] = []
    for fe in file_examples:
        events = predict_events(model, fe, gap_steps)
        windows = windows_by_file.get(fe.filename, [])
        r = evaluate(events, windows, mode=mode)
        precisions.append(r["precision"])
        recalls.append(r["recall"])

    if not precisions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "files": 0}

    p = sum(precisions) / len(precisions)
    rec = sum(recalls) / len(recalls)
    f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
    return {"precision": p, "recall": rec, "f1": f1, "files": len(precisions)}
