"""Compare the EWMA forecaster vs the neural (LSTM) forecaster on TAB by VUS-PR.

Aimed at the corpus's weak spot: the EWMA one-step predictor is near-blind to
pattern/shape anomalies (KDD21, OPPORTUNITY). This trains a per-file LSTM
forecaster as a challenger and scores both head-to-head, macro-averaged.

    python scripts/compare_neural.py                       # KDD21 subset
    python scripts/compare_neural.py --dataset OPPORTUNITY --limit 15

Uses the GPU when one is visible (the 4060), CPU otherwise — per-file LSTM
training is slow on CPU, so keep the subset small there.
"""

import argparse
import statistics
from pathlib import Path

from threadforge.data.tab import load_tab_meta, load_tab_univariate
from threadforge.detection import ForecastResidualDetector
from threadforge.models.neural_forecast import NeuralForecastResidualDetector
from threadforge.models.torch_util import describe_device
from threadforge.tab_scoring import vus

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "TAB" / "TAB_dataset" / "dataset" / "anomaly_detect"
META_PATH = DATA_DIR / "DETECT_META.csv"
FILES_DIR = DATA_DIR / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="KDD21", help="dataset_name to test (default KDD21).")
    ap.add_argument("--limit", type=int, default=20, help="files to score. Default 20.")
    ap.add_argument("--max-steps", type=int, default=6000, help="skip series longer than this. Default 6000.")
    ap.add_argument("--window", type=int, default=100, help="VUS buffer half-width. Default 100.")
    ap.add_argument("--lstm-window", type=int, default=20, help="LSTM input window. Default 20.")
    ap.add_argument("--epochs", type=int, default=15, help="LSTM training epochs. Default 15.")
    args = ap.parse_args()

    if not META_PATH.exists():
        print(f"TAB metadata not found at {META_PATH} (see data/README.md).")
        return

    meta = [m for m in load_tab_meta(META_PATH) if m.if_univariate]
    if args.dataset:
        meta = [m for m in meta if m.dataset_name.upper() == args.dataset.upper()]
    if args.max_steps:
        meta = [m for m in meta if m.time_steps <= args.max_steps]
    meta.sort(key=lambda m: m.file_name)
    if args.limit:
        meta = meta[:args.limit]
    if not meta:
        print("No matching files. Loosen --dataset / --max-steps / --limit.")
        return

    ewma = ForecastResidualDetector()
    neural = NeuralForecastResidualDetector(window=args.lstm_window, epochs=args.epochs)

    print(f"Comparing EWMA vs neural (LSTM) forecaster on {len(meta)} {args.dataset} files | "
          f"device: {describe_device()}", flush=True)
    ewma_scores, neural_scores = [], []
    for i, m in enumerate(meta, start=1):
        path = FILES_DIR / m.file_name
        if not path.exists():
            continue
        stream, labels = load_tab_univariate(path)
        if sum(labels) == 0:
            continue
        ewma_vpr = vus(labels, ewma.scores(stream), window=args.window)["VUS_PR"]
        neural_vpr = vus(labels, neural.scores(stream), window=args.window)["VUS_PR"]
        ewma_scores.append(ewma_vpr)
        neural_scores.append(neural_vpr)
        print(f"  [{i}/{len(meta)}] {m.file_name}: EWMA {ewma_vpr:.4f}  neural {neural_vpr:.4f}", flush=True)

    if not ewma_scores:
        print("Nothing scored.")
        return

    e, nu = statistics.mean(ewma_scores), statistics.mean(neural_scores)
    print("-" * 60)
    print(f"{'EWMA (champion)':<22}{e:.4f}")
    print(f"{'neural LSTM forecaster':<22}{nu:.4f}   ({nu - e:+.4f} vs EWMA)")
    print("-" * 60)
    verdict = "neural wins" if nu > e else "EWMA holds"
    print(f"{verdict} on {args.dataset} ({len(ewma_scores)} files, macro VUS-PR)")


if __name__ == "__main__":
    main()
