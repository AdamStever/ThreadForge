"""Score the forecasting-based detector over the full NAB corpus, by NAB score.

This detector is unsupervised and online, so it follows NAB's own protocol: run
on every file with a probationary period, no train/test split. We scan the
detection threshold (a single corpus-wide hyperparameter, as NAB detectors tune)
and report the resulting standardized NAB score per threshold.

    python scripts/run_forecast.py
"""

import json
from pathlib import Path

from threadforge.data import stream_csv
from threadforge.detection import ForecastResidualDetector
from threadforge.nab_scoring import score_file, normalized_score

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]


def main() -> None:
    with open(ROOT / "labels" / "windows.json") as f:
        all_labels = json.load(f)

    files = [p for p in sorted((ROOT / "data" / "raw").glob("*.csv"))
             if p.name in all_labels and all_labels[p.name]]
    if not files:
        print("No labeled files found.")
        return

    detector = ForecastResidualDetector()

    # score each file once, then threshold the per-step scores at each candidate
    per_file = []
    for p in files:
        stream = stream_csv(str(p))
        scores = detector.scores(stream)
        per_file.append((
            [ts for ts, _ in stream],
            scores,
            [(w[0], w[1]) for w in all_labels[p.name]],
            detector.probation(len(stream)),
        ))

    print(f"Forecasting detector over {len(files)} files (online, unsupervised)")
    print(f"{'threshold':>10}{'NAB':>10}")
    print("-" * 20)
    best_t, best = None, float("-inf")
    for thr in THRESHOLDS:
        results = []
        for timestamps, scores, windows, probation in per_file:
            flags = [s >= thr for s in scores]
            results.append(score_file(timestamps, flags, windows, probation=probation))
        nab = normalized_score(results)
        print(f"{thr:>10.1f}{nab:>10.1f}")
        if nab > best:
            best, best_t = nab, thr

    print("-" * 20)
    print(f"best: threshold {best_t:.1f}  ->  NAB {best:.1f}")


if __name__ == "__main__":
    main()
