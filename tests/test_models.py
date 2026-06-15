"""Tests for the baseline modeling layer."""

import numpy as np
import pytest

from threadforge.models import (
    FileExamples,
    build_file_examples,
    cross_file_split,
    train,
    group_predictions,
    evaluate_model,
)


# --- dataset construction ---

def test_build_file_examples_labels_window():
    stream = [(f"2024-01-01 00:{i:02d}:00", float(i % 7)) for i in range(30)]
    windows = [("2024-01-01 00:10:00", "2024-01-01 00:15:00")]
    fe = build_file_examples("synthetic.csv", stream, windows, window_size=5)

    assert fe.X.shape[0] == fe.y.shape[0] > 0
    assert fe.X.shape[1] == len(fe.signal_names) == 10
    assert set(fe.y.tolist()) == {0, 1}  # both classes present
    # a timestamp inside the window is labeled 1, one outside is 0
    assert fe.y[fe.timestamps.index("2024-01-01 00:12:00")] == 1
    assert fe.y[fe.timestamps.index("2024-01-01 00:20:00")] == 0


def test_build_file_examples_skips_warmup():
    stream = [(f"2024-01-01 00:{i:02d}:00", float(i)) for i in range(20)]
    fe = build_file_examples("s.csv", stream, [], window_size=5)
    # first (window_size - 1) steps are warm-up and dropped
    assert len(fe.timestamps) == 20 - (5 - 1)


# --- cross-file split ---

def test_cross_file_split_disjoint_and_deterministic():
    files = [f"f{i}.csv" for i in range(10)]
    tr, te = cross_file_split(files, test_fraction=0.3, seed=0)
    assert set(tr).isdisjoint(te)
    assert sorted(tr + te) == sorted(files)
    assert len(tr) >= 1 and len(te) >= 1
    assert cross_file_split(files, test_fraction=0.3, seed=0) == (tr, te)


def test_cross_file_split_needs_two_files():
    with pytest.raises(ValueError):
        cross_file_split(["only.csv"])


# --- model training / grouping / evaluation ---

def test_train_separates_classes():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (100, 10)), rng.normal(8, 1, (20, 10))])
    y = np.array([0] * 100 + [1] * 20)
    model = train(X, y)
    assert model.predict([[8.0] * 10])[0] == 1
    assert model.predict([[0.0] * 10])[0] == 0


def test_group_predictions_respects_gap():
    ts = [f"2024-01-01 00:{i:02d}:00" for i in range(10)]
    vals = [0.0] * 10
    preds = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1])
    scores = np.array([0.9] * 10)
    assert len(group_predictions(ts, vals, preds, scores, gap_steps=3)) == 2
    assert len(group_predictions(ts, vals, preds, scores, gap_steps=20)) == 1


def test_evaluate_model_catches_anomaly_window():
    rng = np.random.default_rng(1)
    Xtrain = np.vstack([rng.normal(0, 1, (80, 10)), rng.normal(8, 1, (20, 10))])
    ytrain = np.array([0] * 80 + [1] * 20)
    model = train(Xtrain, ytrain)

    n = 10
    ts = [f"2024-01-01 00:{i:02d}:00" for i in range(n)]
    rows = [rng.normal(8, 1, 10) if i in (5, 6, 7) else rng.normal(0, 1, 10) for i in range(n)]
    fe = FileExamples("f.csv", ts, [0.0] * n, np.array(rows), np.zeros(n, int),
                      [f"s{i}" for i in range(10)])
    windows = {"f.csv": [("2024-01-01 00:05:00", "2024-01-01 00:07:00")]}

    res = evaluate_model(model, [fe], windows, gap_steps=2, mode="overlap")
    assert res["recall"] == 1.0
    assert res["precision"] > 0.0
    assert res["files"] == 1
