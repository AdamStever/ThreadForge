"""Hyperparameter search for the baseline model, driven by the genetic algorithm.

The GA searches two knobs, scoring each candidate on a held-out **validation**
split (never the test set):

  - log_C       log10 of the logistic-regression inverse-regularization C
  - threshold   probability cutoff for calling a step anomalous

WHY POINT-LEVEL F1 AS THE OBJECTIVE
  The event/overlap F1 used for operational reporting is reward-hackable here:
  flagging a large fraction of a stream merges (via gap grouping) into one
  file-spanning "event" that overlaps every window — a perfect 1.0 with no real
  skill. Per-step (point-level) F1 can't be gamed that way: flagging everything
  drives point-precision down to the anomaly base rate. We already have a label
  per step (FileExamples.y), so point scoring is exact and direct. Event grouping
  (gap_steps) doesn't change per-step predictions, so it isn't part of the search.

Feature extraction is done once up front and reused across candidates; only the
fast model fit and scoring repeat.
"""

from __future__ import annotations

import random

import numpy as np

from threadforge.models.baseline import train
from threadforge.optimization.genetic import Gene, evolve


SEARCH_SPACE = [
    Gene("log_C", -3.0, 2.0),       # C = 10**log_C, i.e. 1e-3 .. 1e2
    Gene("threshold", 0.1, 0.9),
]


def decode(genome: dict) -> dict:
    """Turn a raw genome into usable hyperparameters."""
    return {
        "C": 10.0 ** genome["log_C"],
        "threshold": float(genome["threshold"]),
    }


def point_scores(model, examples: dict, files: list[str], threshold: float) -> dict:
    """Micro-averaged per-step precision / recall / F1 over the given files.

    Predictions are compared against the per-step labels (FileExamples.y) pooled
    across files — not gameable by event-spanning.
    """
    tp = fp = fn = flagged = total = 0
    for f in files:
        fe = examples[f]
        if fe.X.shape[0] == 0:
            continue
        proba = model.predict_proba(fe.X)[:, 1]
        preds = proba >= threshold
        y = fe.y.astype(bool)
        tp += int((preds & y).sum())
        fp += int((preds & ~y).sum())
        fn += int((~preds & y).sum())
        flagged += int(preds.sum())
        total += len(preds)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    alert_rate = flagged / total if total else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "alert_rate": alert_rate}


def make_fitness(examples, train_files, val_files):
    """Build a fitness function: train on train_files, score point-F1 on val_files."""
    X = np.vstack([examples[f].X for f in train_files if examples[f].X.shape[0] > 0])
    y = np.concatenate([examples[f].y for f in train_files if examples[f].X.shape[0] > 0])

    def fitness(genome: dict) -> float:
        hp = decode(genome)
        model = train(X, y, C=hp["C"])
        return point_scores(model, examples, val_files, hp["threshold"])["f1"]

    return fitness


def run_search(
    examples, train_files, val_files,
    *, seed: int = 0, pop_size: int = 12, generations: int = 8,
) -> tuple[dict, float, list[float]]:
    """Run the GA and return (best hyperparameters, best val point-F1, history)."""
    fitness = make_fitness(examples, train_files, val_files)
    rng = random.Random(seed)
    best_genome, best_fit, history = evolve(
        SEARCH_SPACE, fitness, pop_size=pop_size, generations=generations, rng=rng
    )
    return decode(best_genome), best_fit, history
