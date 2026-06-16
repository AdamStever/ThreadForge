"""Compare the temporal LSTM to the flat-window encoder, by NAB score.

Both models read the *same* raw value windows; the only difference is how:
  - encoder: flattens the window into one vector (MLP)
  - LSTM:    reads it as a sequence, carrying memory across steps

Methodology mirrors train_encoder.py: cross-file train / validation / test, the
decision threshold chosen on validation (best NAB), scored once on the untouched
test files.

    python scripts/train_temporal.py
"""

import json
from pathlib import Path

import numpy as np

from threadforge.data import stream_csv
from threadforge.models import build_window_examples, cross_file_split
from threadforge.models.torch_model import train_model, train_lstm, predict_proba
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


def _evaluate(model, win, windows_by_file, names, val_files, test):
    proba = {f: predict_proba(model, win[f].X) if win[f].X.shape[0] else np.array([]) for f in names}
    t, val = _best_threshold(proba, win, windows_by_file, val_files)
    return val, t, _nab(proba, win, windows_by_file, test, t)


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

    print(f"Building raw windows for {len(files)} files...")
    win, windows_by_file = {}, {}
    for p in files:
        windows = [(w[0], w[1]) for w in all_labels[p.name]]
        windows_by_file[p.name] = windows
        win[p.name] = build_window_examples(p.name, stream_csv(str(p)), windows, cfg["window_size"])

    Xtr = np.vstack([win[f].X for f in train_files if win[f].X.shape[0] > 0])
    ytr = np.concatenate([win[f].y for f in train_files if win[f].X.shape[0] > 0])
    print(f"Split: {len(train_files)} train / {len(val_files)} val / {len(test)} test")

    print("Training encoder (flat window)...")
    enc_val, enc_t, enc_test = _evaluate(train_model(Xtr, ytr, seed=0), win, windows_by_file, names, val_files, test)
    print("Training LSTM (sequence)...")
    lstm_val, lstm_t, lstm_test = _evaluate(train_lstm(Xtr, ytr, seed=0), win, windows_by_file, names, val_files, test)

    print()
    print(f"{'Model':<26}{'val NAB':>9}{'thr':>6}{'test NAB':>10}")
    print("-" * 52)
    print(f"{'encoder (flat window)':<26}{enc_val:>9.1f}{enc_t:>6.2f}{enc_test:>10.1f}")
    print(f"{'LSTM (temporal)':<26}{lstm_val:>9.1f}{lstm_t:>6.2f}{lstm_test:>10.1f}")


if __name__ == "__main__":
    main()
