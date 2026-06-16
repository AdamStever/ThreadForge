"""Train the PyTorch encoder/score net and compare it to the linear baseline.

Head-to-head, both judged by the standardized NAB score:
  - encoder: learns features from raw value windows
  - baseline: logistic regression on the 10 hand-crafted signals

Methodology: cross-file train / validation / test. Each model is fit on train,
its decision threshold is chosen on validation (best NAB), and it is scored once
on the untouched test files.

    python scripts/train_encoder.py
"""

import json
from pathlib import Path

import numpy as np

from threadforge.data import stream_csv
from threadforge.models import (
    build_file_examples, build_window_examples, cross_file_split, train,
)
from threadforge.models.torch_model import train_model, predict_proba
from threadforge.nab_scoring import score_file, normalized_score

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def _nab(proba_by_file, ex_by_file, windows_by_file, files, threshold):
    results = []
    for f in files:
        ex = ex_by_file[f]
        if ex.X.shape[0] == 0:
            continue
        flags = (proba_by_file[f] >= threshold).tolist()
        results.append(score_file(ex.timestamps, flags, windows_by_file.get(f, []), probation=0))
    return normalized_score(results)


def _best_threshold(proba_by_file, ex_by_file, windows_by_file, val_files):
    best_t, best = THRESHOLDS[0], float("-inf")
    for t in THRESHOLDS:
        s = _nab(proba_by_file, ex_by_file, windows_by_file, val_files, t)
        if s > best:
            best, best_t = s, t
    return best_t, best


def main() -> None:
    with open(ROOT / "config" / "default.json") as f:
        cfg = json.load(f)
    with open(ROOT / "labels" / "windows.json") as f:
        all_labels = json.load(f)

    files = [p for p in sorted((ROOT / "data" / "raw").glob("*.csv"))
             if p.name in all_labels and all_labels[p.name]]
    if len(files) < 4:
        print("Need at least 4 labeled files.")
        return

    names = [p.name for p in files]
    trainval, test = cross_file_split(names, test_fraction=0.3, seed=0)
    train_files, val_files = cross_file_split(trainval, test_fraction=0.25, seed=0)

    print(f"Building features for {len(files)} files...")
    win, feat, windows_by_file = {}, {}, {}
    for p in files:
        stream = stream_csv(str(p))
        windows = [(w[0], w[1]) for w in all_labels[p.name]]
        windows_by_file[p.name] = windows
        win[p.name] = build_window_examples(p.name, stream, windows, cfg["window_size"])
        feat[p.name] = build_file_examples(p.name, stream, windows, cfg["window_size"])

    ws = cfg["window_size"]
    print(f"Split: {len(train_files)} train / {len(val_files)} val / {len(test)} test")

    # --- encoder (learned features from raw windows) ---
    print("Training encoder...")
    Xtr = np.vstack([win[f].X for f in train_files if win[f].X.shape[0] > 0])
    ytr = np.concatenate([win[f].y for f in train_files if win[f].X.shape[0] > 0])
    enc = train_model(Xtr, ytr, seed=0)
    enc_proba = {f: predict_proba(enc, win[f].X) if win[f].X.shape[0] else np.array([]) for f in names}
    enc_t, enc_val = _best_threshold(enc_proba, win, windows_by_file, val_files)
    enc_test = _nab(enc_proba, win, windows_by_file, test, enc_t)

    # --- linear baseline (hand-crafted signals) ---
    print("Training linear baseline...")
    Xb = np.vstack([feat[f].X for f in train_files if feat[f].X.shape[0] > 0])
    yb = np.concatenate([feat[f].y for f in train_files if feat[f].X.shape[0] > 0])
    lin = train(Xb, yb)
    lin_proba = {f: lin.predict_proba(feat[f].X)[:, 1] if feat[f].X.shape[0] else np.array([]) for f in names}
    lin_t, lin_val = _best_threshold(lin_proba, feat, windows_by_file, val_files)
    lin_test = _nab(lin_proba, feat, windows_by_file, test, lin_t)

    print()
    print(f"{'Model':<28}{'val NAB':>9}{'thr':>6}{'test NAB':>10}")
    print("-" * 54)
    print(f"{'linear (hand features)':<28}{lin_val:>9.1f}{lin_t:>6.2f}{lin_test:>10.1f}")
    print(f"{'encoder (raw windows)':<28}{enc_val:>9.1f}{enc_t:>6.2f}{enc_test:>10.1f}")


if __name__ == "__main__":
    main()
