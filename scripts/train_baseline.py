"""Train the baseline ML model and compare it to the heuristic pipeline.

Uses a cross-file split: some labeled NAB files train the model, the rest are
held out for testing. Both the learned model and the heuristic detector are
scored on the *same held-out files* with overlap matching, so the comparison is
apples-to-apples.

    python scripts/train_baseline.py

Reads settings from config/default.json and labels from labels/windows.json.
"""

import json
from pathlib import Path

import numpy as np

from threadforge.data import stream_csv
from threadforge.presets import default_signal_engine
from threadforge.detection import RobustCalibrator, Detector, Scorer
from threadforge.evaluation import evaluate, OVERLAP
from threadforge.models import build_file_examples, cross_file_split, train, evaluate_model

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config" / "default.json") as f:
        return json.load(f)


def load_labels() -> dict:
    with open(ROOT / "labels" / "windows.json") as f:
        return json.load(f)


def heuristic_scores(file_paths, cfg, all_labels) -> dict:
    """Run the heuristic detector on each file and aggregate P/R/F1 (overlap)."""
    scorer = Scorer(cfg["scorer_weights"], cfg["score_threshold"])
    precisions, recalls = [], []
    for p in file_paths:
        stream = stream_csv(str(p))
        engine = default_signal_engine(cfg["window_size"])
        calibrators = {n: RobustCalibrator(cfg["threshold_multiplier"]) for n in engine._signals}
        detector = Detector(
            engine=engine,
            calibrators=calibrators,
            scorer=scorer,
            calib_steps=cfg["calibration_steps"],
            gap_steps=cfg["gap_steps"],
            min_calib_samples=cfg.get("min_calibration_samples", 30),
            gap_seconds=cfg.get("gap_seconds"),
        )
        events = detector.run(stream)
        windows = [(w[0], w[1]) for w in all_labels[p.name]]
        r = evaluate(events, windows, mode=OVERLAP)
        precisions.append(r["precision"])
        recalls.append(r["recall"])
    p = sum(precisions) / len(precisions)
    rec = sum(recalls) / len(recalls)
    f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
    return {"precision": p, "recall": rec, "f1": f1, "files": len(precisions)}


def main() -> None:
    cfg = load_config()
    all_labels = load_labels()

    files = [
        p for p in sorted((ROOT / "data" / "raw").glob("*.csv"))
        if p.name in all_labels and all_labels[p.name]
    ]
    if len(files) < 2:
        print("Need at least 2 labeled files in data/raw.")
        return

    filenames = [p.name for p in files]
    train_files, test_files = cross_file_split(filenames, test_fraction=0.3, seed=0)

    print(f"Building features for {len(files)} files...")
    examples = {}
    windows_by_file = {}
    for p in files:
        stream = stream_csv(str(p))
        windows = [(w[0], w[1]) for w in all_labels[p.name]]
        windows_by_file[p.name] = windows
        examples[p.name] = build_file_examples(p.name, stream, windows, cfg["window_size"])

    train_X = [examples[f].X for f in train_files if examples[f].X.shape[0] > 0]
    train_y = [examples[f].y for f in train_files if examples[f].X.shape[0] > 0]
    if not train_X:
        print("No training rows produced.")
        return
    X = np.vstack(train_X)
    y = np.concatenate(train_y)

    model = train(X, y)

    test_examples = [examples[f] for f in test_files]
    ml = evaluate_model(model, test_examples, windows_by_file,
                        gap_steps=cfg["gap_steps"], mode=OVERLAP)
    heur = heuristic_scores([p for p in files if p.name in test_files], cfg, all_labels)

    print()
    print(f"Cross-file split (seed 0):  {len(train_files)} train / {len(test_files)} test files")
    print(f"Training rows: {len(y)}  |  anomaly fraction: {y.mean():.3f}")
    print(f"Match mode: overlap  |  held-out test files: {ml['files']}")
    print("-" * 52)
    print(f"{'':<16}{'Precision':>10}{'Recall':>9}{'F1':>8}")
    print(f"{'heuristic':<16}{heur['precision']:>10.3f}{heur['recall']:>9.3f}{heur['f1']:>8.3f}")
    print(f"{'model (LogReg)':<16}{ml['precision']:>10.3f}{ml['recall']:>9.3f}{ml['f1']:>8.3f}")


if __name__ == "__main__":
    main()
