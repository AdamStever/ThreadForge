"""Learn the 10-signal weights with the genetic search, optimised on VUS-PR.

This is the project's "let search learn the weights" idea applied to the
WeightedSignalDetector: the GA searches one weight per signal to maximise macro
VUS-PR on a training split, then we report the learned-weight detector against the
equal-weight detector and the EWMA baseline on a held-out split — so we see
whether a *smart* combine of the 10 signals can actually compete.

    python scripts/learn_signal_weights.py
    python scripts/learn_signal_weights.py --train 16 --eval 8 --pop 16 --gen 8

Fitness evaluation is heavy (the 10-signal detector per file), so keep the train
set / pop / generations modest, or run it in the background.
"""

import argparse
import random
import statistics
from pathlib import Path

from threadforge.data.tab import load_tab_meta, load_tab_univariate
from threadforge.detection import ForecastResidualDetector, WeightedSignalDetector
from threadforge.optimization.genetic import Gene, evolve
from threadforge.presets import default_signal_names
from threadforge.tab_scoring import vus

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "TAB" / "TAB_dataset" / "dataset" / "anomaly_detect"
META_PATH = DATA_DIR / "DETECT_META.csv"
FILES_DIR = DATA_DIR / "data"


def _macro_vpr(detector, streams, window):
    return statistics.mean(
        vus(labels, detector.scores(stream), window=window)["VUS_PR"]
        for stream, labels in streams
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=None, help="restrict to one dataset_name.")
    ap.add_argument("--train", type=int, default=12, help="training files. Default 12.")
    ap.add_argument("--eval", type=int, default=8, help="held-out eval files. Default 8.")
    ap.add_argument("--max-steps", type=int, default=6000, help="skip series longer than this. Default 6000.")
    ap.add_argument("--window", type=int, default=100, help="VUS buffer half-width. Default 100.")
    ap.add_argument("--pop", type=int, default=12, help="GA population. Default 12.")
    ap.add_argument("--gen", type=int, default=6, help="GA generations. Default 6.")
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
    need = args.train + args.eval
    meta = meta[:need]
    if len(meta) < need:
        print(f"Only {len(meta)} files available; need {need}.")
        return

    def load(m):
        return load_tab_univariate(FILES_DIR / m.file_name)

    train = [load(m) for m in meta[:args.train]]
    train = [(s, l) for s, l in train if sum(l) > 0]
    val = [load(m) for m in meta[args.train:]]
    val = [(s, l) for s, l in val if sum(l) > 0]

    names = default_signal_names()             # signal order = gene order
    genes = [Gene(n, 0.0, 1.0) for n in names]

    evals = {"count": 0}

    def fitness(genome: dict) -> float:
        evals["count"] += 1
        score = _macro_vpr(WeightedSignalDetector(genome), train, args.window)
        print(f"  eval {evals['count']:>3}: VUS_PR(train)={score:.4f}", flush=True)
        return score

    print(f"GA weight search | train={len(train)} val={len(val)} | pop={args.pop} gen={args.gen}", flush=True)
    best, best_fit, history = evolve(
        genes, fitness, pop_size=args.pop, generations=args.gen, rng=random.Random(0),
    )

    learned = _macro_vpr(WeightedSignalDetector(best), val, args.window)
    equal = _macro_vpr(WeightedSignalDetector(), val, args.window)
    ewma = _macro_vpr(ForecastResidualDetector(), val, args.window)

    print("=" * 56)
    print("learned signal weights (high -> low):")
    for n, w in sorted(best.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {n:<18}{w:.3f}")
    print("-" * 56)
    print(f"{'EWMA baseline':<26}{ewma:.4f}   (val)")
    print(f"{'equal-weight signals':<26}{equal:.4f}   (val)")
    print(f"{'learned-weight signals':<26}{learned:.4f}   (val)   "
          f"({learned - ewma:+.4f} vs EWMA)")
    print("=" * 56)
    if learned > ewma:
        print("the 10 signals (learned weights) beat the EWMA baseline on held-out files.")
    else:
        print("learned-weight signals did not beat EWMA on held-out files.")


if __name__ == "__main__":
    main()
