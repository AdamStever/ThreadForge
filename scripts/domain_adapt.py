"""Domain-adaptation experiment — does seeding help, and how fast does it learn?

For one domain, with a held-out split (seed pool disjoint from eval files), it
measures the universal-detector question end to end:

  1. EWMA cold start         -- zero training, the floor a universal detector must
                                always have and must never lose to.
  2. per-file neural          -- LSTM trained only on each eval file's own prefix
                                (the zero-seed / brand-new-domain case).
  3. seeded@k                 -- LSTM pretrained on k same-domain files, applied to
                                the eval files. Sweeping k is the learning curve:
                                how much domain data until it beats the EWMA floor.

All scored by macro VUS-PR over the *same* held-out eval files. No leakage: the
seed pool never overlaps the eval files, and labels are never used in training.

    python scripts/domain_adapt.py --dataset KDD21
    python scripts/domain_adapt.py --dataset YAHOO --limit 24 --eval 8
"""

import argparse
import statistics
from pathlib import Path

from threadforge.data.tab import load_tab_meta, load_tab_univariate
from threadforge.detection import ForecastResidualDetector
from threadforge.models.neural_forecast import NeuralForecastResidualDetector
from threadforge.models.domain_adapt import train_pool_forecaster, SeededForecastDetector
from threadforge.models.torch_util import describe_device
from threadforge.tab_scoring import vus

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "TAB" / "TAB_dataset" / "dataset" / "anomaly_detect"
META_PATH = DATA_DIR / "DETECT_META.csv"
FILES_DIR = DATA_DIR / "data"


def _macro_vpr(detector, eval_streams, window):
    return statistics.mean(
        vus(labels, detector.scores(stream), window=window)["VUS_PR"]
        for stream, labels in eval_streams
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="KDD21", help="domain to test. Default KDD21.")
    ap.add_argument("--limit", type=int, default=16, help="total files to use. Default 16.")
    ap.add_argument("--eval", type=int, default=6, help="held-out eval files. Default 6.")
    ap.add_argument("--max-steps", type=int, default=0, help="skip series longer than this (0 = no cap).")
    ap.add_argument("--window", type=int, default=100, help="VUS buffer half-width. Default 100.")
    ap.add_argument("--lstm-window", type=int, default=20, help="LSTM input window. Default 20.")
    ap.add_argument("--epochs", type=int, default=15, help="LSTM epochs. Default 15.")
    ap.add_argument("--seed-sizes", default="1,2,4,8", help="seed-pool sizes to sweep. Default 1,2,4,8.")
    args = ap.parse_args()

    if not META_PATH.exists():
        print(f"TAB metadata not found at {META_PATH} (see data/README.md).")
        return

    meta = [m for m in load_tab_meta(META_PATH) if m.if_univariate and m.dataset_name.upper() == args.dataset.upper()]
    if args.max_steps:
        meta = [m for m in meta if m.time_steps <= args.max_steps]
    meta.sort(key=lambda m: m.file_name)
    meta = meta[:args.limit]
    if len(meta) <= args.eval:
        print(f"Not enough {args.dataset} files ({len(meta)}) for eval={args.eval} + a seed pool.")
        return

    def load(m):
        return load_tab_univariate(FILES_DIR / m.file_name)

    eval_meta = meta[-args.eval:]
    pool_meta = meta[:-args.eval]
    eval_streams = [load(m) for m in eval_meta if (FILES_DIR / m.file_name).exists()]
    eval_streams = [(s, l) for s, l in eval_streams if sum(l) > 0]
    pool_series = [[v for _, v in load(m)[0]] for m in pool_meta if (FILES_DIR / m.file_name).exists()]

    sizes = [s for s in (int(x) for x in args.seed_sizes.split(",")) if s <= len(pool_series)]
    print(f"domain {args.dataset} | pool={len(pool_series)} eval={len(eval_streams)} | "
          f"device: {describe_device()}", flush=True)
    print("-" * 44)

    rows = []
    ewma = _macro_vpr(ForecastResidualDetector(), eval_streams, args.window)
    rows.append(("EWMA cold start", ewma))
    print(f"  {'EWMA cold start':<22}{ewma:.4f}", flush=True)

    perfile = _macro_vpr(
        NeuralForecastResidualDetector(window=args.lstm_window, epochs=args.epochs),
        eval_streams, args.window,
    )
    rows.append(("per-file neural", perfile))
    print(f"  {'per-file neural':<22}{perfile:.4f}  ({perfile - ewma:+.4f} vs EWMA)", flush=True)

    crossed = None
    for s in sizes:
        model = train_pool_forecaster(pool_series[:s], window=args.lstm_window, epochs=args.epochs)
        if model is None:
            continue
        det = SeededForecastDetector(model, window=args.lstm_window)
        vpr = _macro_vpr(det, eval_streams, args.window)
        rows.append((f"seeded@{s}", vpr))
        print(f"  {f'seeded@{s}':<22}{vpr:.4f}  ({vpr - ewma:+.4f} vs EWMA)", flush=True)
        if crossed is None and vpr > ewma:
            crossed = s

    print("-" * 44)
    if crossed is not None:
        print(f"seeded neural beats the EWMA floor from {crossed} domain file(s) onward.")
    else:
        print("seeded neural did not beat the EWMA floor at the sizes tried.")


if __name__ == "__main__":
    main()
