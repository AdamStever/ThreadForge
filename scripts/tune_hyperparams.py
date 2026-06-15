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
from threadforge.optimization import run_search, point_scores, nab_score_on_files

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
    print("Running genetic search (objective = standardized NAB score on validation)...")
    best_hp, best_val_nab, history = run_search(
        examples, train_files, val_files, windows_by_file, objective="nab", seed=0
    )

    # retrain on train+val, evaluate once on the untouched test set
    Xtv = np.vstack([examples[f].X for f in trainval if examples[f].X.shape[0] > 0])
    ytv = np.concatenate([examples[f].y for f in trainval if examples[f].X.shape[0] > 0])

    default_model = train(Xtv, ytv, C=1.0)
    default_nab = nab_score_on_files(default_model, examples, test, windows_by_file, threshold=0.5)
    default_pt = point_scores(default_model, examples, test, threshold=0.5)

    tuned_model = train(Xtv, ytv, C=best_hp["C"])
    tuned_nab = nab_score_on_files(tuned_model, examples, test, windows_by_file, threshold=best_hp["threshold"])
    tuned_pt = point_scores(tuned_model, examples, test, threshold=best_hp["threshold"])

    print()
    print(f"GA best validation NAB score: {best_val_nab:.1f}   (history: "
          f"{' -> '.join(f'{h:.1f}' for h in history)})")
    print(f"Best hyperparameters:  C={best_hp['C']:.4g}  threshold={best_hp['threshold']:.3f}")
    print("-" * 64)
    print(f"{'On held-out test':<24}{'NAB':>10}{'point-F1':>10}{'alert':>8}")
    print(f"{'default (C=1,thr=.5)':<24}{default_nab:>10.1f}{default_pt['f1']:>10.3f}{default_pt['alert_rate']:>8.3f}")
    print(f"{'GA-tuned (NAB)':<24}{tuned_nab:>10.1f}{tuned_pt['f1']:>10.3f}{tuned_pt['alert_rate']:>8.3f}")


if __name__ == "__main__":
    main()
