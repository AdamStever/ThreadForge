"""Tune the baseline model's hyperparameters with a genetic algorithm.

Methodology (no tuning on the test set):
  - split files three ways: train / validation / test (cross-file)
  - the GA searches C, probability threshold, and gap_steps, scoring each
    candidate's F1 on the *validation* files
  - the winning hyperparameters are then evaluated **once** on the untouched
    test files, against the default baseline trained on the same data

    python scripts/tune_hyperparams.py
"""

import json
from pathlib import Path

import numpy as np

from threadforge.data import stream_csv
from threadforge.models import build_file_examples, cross_file_split, train
from threadforge.optimization import run_search, point_scores

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config" / "default.json") as f:
        return json.load(f)


def load_labels() -> dict:
    with open(ROOT / "labels" / "windows.json") as f:
        return json.load(f)


def main() -> None:
    cfg = load_config()
    all_labels = load_labels()

    files = [
        p for p in sorted((ROOT / "data" / "raw").glob("*.csv"))
        if p.name in all_labels and all_labels[p.name]
    ]
    if len(files) < 4:
        print("Need at least 4 labeled files for a train/val/test split.")
        return

    filenames = [p.name for p in files]
    trainval, test = cross_file_split(filenames, test_fraction=0.3, seed=0)
    train_files, val_files = cross_file_split(trainval, test_fraction=0.25, seed=0)

    print(f"Building features for {len(files)} files...")
    examples, windows_by_file = {}, {}
    for p in files:
        windows = [(w[0], w[1]) for w in all_labels[p.name]]
        windows_by_file[p.name] = windows
        examples[p.name] = build_file_examples(p.name, stream_csv(str(p)), windows, cfg["window_size"])

    print(f"Split: {len(train_files)} train / {len(val_files)} val / {len(test)} test files")
    print("Running genetic search (objective = per-step / point F1 on validation)...")
    best_hp, best_val_f1, history = run_search(examples, train_files, val_files, seed=0)

    # retrain on train+val, evaluate once on the untouched test set (point-level)
    Xtv = np.vstack([examples[f].X for f in trainval if examples[f].X.shape[0] > 0])
    ytv = np.concatenate([examples[f].y for f in trainval if examples[f].X.shape[0] > 0])

    default_model = train(Xtv, ytv, C=1.0)
    default = point_scores(default_model, examples, test, threshold=0.5)

    tuned_model = train(Xtv, ytv, C=best_hp["C"])
    tuned = point_scores(tuned_model, examples, test, threshold=best_hp["threshold"])

    print()
    print(f"GA best validation point-F1: {best_val_f1:.3f}   (history: "
          f"{' -> '.join(f'{h:.3f}' for h in history)})")
    print(f"Best hyperparameters:  C={best_hp['C']:.4g}  threshold={best_hp['threshold']:.3f}")
    print("-" * 60)
    print(f"{'On held-out test':<24}{'precision':>10}{'recall':>9}{'F1':>7}{'alert':>8}")
    for label, s in (("default (C=1,thr=.5)", default), ("GA-tuned", tuned)):
        print(f"{label:<24}{s['precision']:>10.3f}{s['recall']:>9.3f}{s['f1']:>7.3f}{s['alert_rate']:>8.3f}")


if __name__ == "__main__":
    main()
