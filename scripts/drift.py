"""Drift monitoring demo — watch the input distribution, spawn challengers on drift.

Streams values past a `DriftMonitor` and reports the PSI / drift level over time.
With `--retrain` it uses a `DriftRetrainer`: when drift first fires it stocks the
registry with fresh challengers, ready for the next promotion pass — the
self-feeding half of the champion-challenger loop.

    python scripts/drift.py --synthetic                       # stable -> shifted demo
    python scripts/drift.py data/raw/ec2_cpu_utilization_5f5533.csv
    python scripts/drift.py --synthetic --retrain             # spawn challengers on drift
"""

import argparse
from pathlib import Path

import numpy as np

from threadforge.drift import DriftMonitor, DriftRetrainer
from threadforge.registry import DetectorRegistry
from threadforge.streaming import replay_csv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / "registry.json"


def _synthetic(n: int = 600, seed: int = 0):
    rng = np.random.RandomState(seed)
    stable = rng.normal(0.0, 1.0, n)
    shifted = rng.normal(6.0, 1.0, n)        # a clear distribution shift
    return [(str(i), float(v)) for i, v in enumerate(np.concatenate([stable, shifted]))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="?", help="NAB-style CSV (timestamp,value). Omit with --synthetic.")
    ap.add_argument("--synthetic", action="store_true", help="use a stable->shifted synthetic stream.")
    ap.add_argument("--reference-size", type=int, default=300, help="reference window size. Default 300.")
    ap.add_argument("--window", type=int, default=300, help="current window size. Default 300.")
    ap.add_argument("--threshold", type=float, default=0.25, help="PSI drift threshold. Default 0.25.")
    ap.add_argument("--every", type=int, default=100, help="print a status line every N points. Default 100.")
    ap.add_argument("--retrain", action="store_true", help="spawn challengers into the registry on drift.")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="registry path for --retrain.")
    args = ap.parse_args()

    if args.synthetic:
        source = _synthetic()
    elif args.csv:
        source = replay_csv(args.csv)
    else:
        print("Provide a CSV path or --synthetic.")
        return

    monitor = DriftMonitor(
        reference_size=args.reference_size, window=args.window, threshold=args.threshold,
    )
    retrainer = None
    if args.retrain:
        reg = DetectorRegistry(args.registry)
        if reg.champion() is None:
            champ = reg.register("ewma_forecast", params={"ewma_alpha": 0.2, "resid_window": 200},
                                 notes="baseline champion")
            reg.promote(champ.id)
        retrainer = DriftRetrainer(reg, monitor=monitor)

    print(f"Monitoring (reference={args.reference_size}, window={args.window}, "
          f"PSI threshold={args.threshold})", flush=True)
    drift_onset = None
    spawned_so_far = 0
    n = 0
    for ts, value in source:
        status = retrainer.update(value) if retrainer else monitor.update(value)
        n += 1
        if status.drift and drift_onset is None:
            drift_onset = status.index
            print(f"  DRIFT at point {status.index}: PSI={status.psi:.3f} ({status.level})", flush=True)
        # report the challenger pool escalating as severity climbs
        if retrainer and retrainer.last_spawn_count > spawned_so_far:
            spawned_so_far = retrainer.last_spawn_count
            print(f"  -> point {status.index}: severity {retrainer.last_severity:.2f} "
                  f"-> pool now {spawned_so_far} challengers", flush=True)
        elif args.every and status.index % args.every == 0:
            print(f"  point {status.index:>6}: PSI={status.psi:.3f} ({status.level})", flush=True)

    print("-" * 60)
    if drift_onset is None:
        print(f"{n} points: no drift detected.")
    else:
        print(f"{n} points: drift first detected at point {drift_onset}.")
        if retrainer:
            print(f"challenger pool escalated to {retrainer.last_spawn_count} (peak severity "
                  f"{retrainer.last_severity:.2f}). Run `python scripts/promote.py` to evaluate them.")


if __name__ == "__main__":
    main()
