"""K-fold cross-file CV of the encoder, by NAB score.

A single train/val/test split of a small, very heterogeneous corpus gives a noisy
estimate. This rotates every file through the test fold to ask: is the held-out
NAB score *systematic* (low spread across folds) or just *split variance* (high
spread)? That decides whether "more data" is the right lever.

For each fold: the fold is the test set; the remaining files are split into
train + validation; the encoder is trained on train, its threshold chosen on
validation (best NAB), and it is scored on the test fold.

    python scripts/cross_validate.py
"""

import json
import random
from pathlib import Path

import numpy as np

from threadforge.data import stream_csv
from threadforge.models import build_window_examples, cross_file_split
from threadforge.models.torch_model import train_model, predict_proba
from threadforge.nab_scoring import score_file, normalized_score

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
K = 5


def _nab(proba, win, windows_by_file, files, threshold):
    results = []
    for f in files:
        ex = win[f]
        if ex.X.shape[0] == 0:
            continue
        flags = (proba[f] >= threshold).tolist()
        results.append(score_file(ex.timestamps, flags, windows_by_file.get(f, []), probation=0))
    return normalized_score(results)


def _best_threshold(proba, win, windows_by_file, val_files):
    best_t, best = THRESHOLDS[0], float("-inf")
    for t in THRESHOLDS:
        s = _nab(proba, win, windows_by_file, val_files, t)
        if s > best:
            best, best_t = s, t
    return best_t


def kfold(names, k, seed=0):
    shuffled = sorted(names)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i::k] for i in range(k)]  # stride partition into k folds


def main() -> None:
    with open(ROOT / "config" / "default.json") as f:
        cfg = json.load(f)
    with open(ROOT / "labels" / "windows.json") as f:
        all_labels = json.load(f)

    files = [p for p in sorted((ROOT / "data" / "raw").glob("*.csv"))
             if p.name in all_labels and all_labels[p.name]]
    names = [p.name for p in files]
    if len(names) < K + 1:
        print("Not enough labeled files for CV.")
        return

    print(f"Building raw windows for {len(files)} files...")
    win, windows_by_file = {}, {}
    for p in files:
        windows = [(w[0], w[1]) for w in all_labels[p.name]]
        windows_by_file[p.name] = windows
        win[p.name] = build_window_examples(p.name, stream_csv(str(p)), windows, cfg["window_size"])

    folds = kfold(names, K, seed=0)
    fold_scores = []
    for i, test in enumerate(folds):
        rest = [n for n in names if n not in test]
        train_files, val_files = cross_file_split(rest, test_fraction=0.2, seed=0)
        Xtr = np.vstack([win[f].X for f in train_files if win[f].X.shape[0] > 0])
        ytr = np.concatenate([win[f].y for f in train_files if win[f].X.shape[0] > 0])
        model = train_model(Xtr, ytr, seed=0)
        proba = {f: predict_proba(model, win[f].X) if win[f].X.shape[0] else np.array([]) for f in names}
        t = _best_threshold(proba, win, windows_by_file, val_files)
        test_nab = _nab(proba, win, windows_by_file, test, t)
        fold_scores.append(test_nab)
        print(f"fold {i + 1}/{K}: {len(test)} test files, thr={t:.2f}, test NAB = {test_nab:.1f}")

    arr = np.array(fold_scores)
    print("-" * 40)
    print(f"NAB across folds: mean {arr.mean():.1f}  std {arr.std():.1f}  "
          f"min {arr.min():.1f}  max {arr.max():.1f}")


if __name__ == "__main__":
    main()
