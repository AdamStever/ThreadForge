"""Score the LSTM forecaster over the full NAB corpus, vs the EWMA forecaster.

Per file (online, unsupervised): an LSTM is trained on the probationary prefix to
predict the next value, then run forward to produce residuals; those go through
the same residual-z-score → threshold → NAB pipeline as the EWMA forecaster, so
the two are directly comparable.

    python scripts/run_lstm_forecast.py
"""

import json
from pathlib import Path

from threadforge.data import stream_csv
from threadforge.detection import ForecastResidualDetector, residual_zscores
from threadforge.models.torch_forecaster import lstm_residuals
from threadforge.nab_scoring import score_file, normalized_score

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]


def _scan(per_file, label):
    print(f"\n{label}")
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
    print(f"best: threshold {best_t:.1f} -> NAB {best:.1f}")
    return best


def main() -> None:
    with open(ROOT / "labels" / "windows.json") as f:
        all_labels = json.load(f)

    files = [p for p in sorted((ROOT / "data" / "raw").glob("*.csv"))
             if p.name in all_labels and all_labels[p.name]]
    if not files:
        print("No labeled files found.")
        return

    ewma = ForecastResidualDetector()
    lstm_pf, ewma_pf = [], []
    print(f"Scoring {len(files)} files (training one LSTM per file)...")
    for p in files:
        stream = stream_csv(str(p))
        timestamps = [ts for ts, _ in stream]
        values = [v for _, v in stream]
        windows = [(w[0], w[1]) for w in all_labels[p.name]]
        probation = ewma.probation(len(stream))

        lstm_r = lstm_residuals(values, probation)
        lstm_pf.append((timestamps, residual_zscores(lstm_r, probation), windows, probation))
        ewma_pf.append((timestamps, ewma.scores(stream), windows, probation))

    ewma_best = _scan(ewma_pf, "EWMA forecaster")
    lstm_best = _scan(lstm_pf, "LSTM forecaster")
    print("\n" + "=" * 32)
    print(f"EWMA best NAB: {ewma_best:.1f}   LSTM best NAB: {lstm_best:.1f}")


if __name__ == "__main__":
    main()
