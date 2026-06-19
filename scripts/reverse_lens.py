"""Reverse-lens validation — does the non-causal oracle recover the TRUE anomalies?

The whole premise: a retrospective (non-causal) view sees anomalies a live causal
detector can't. If true, the oracle's scores should agree with the *real* TAB
labels much better than the causal EWMA does — which would make it a usable
label-free teacher/fitness signal.

This scores, per file, the matrix-profile oracle and the EWMA detector by VUS-PR
against the ground-truth labels, and reports them head-to-head.

    python scripts/reverse_lens.py
    python scripts/reverse_lens.py --m 100 --max-steps 8000 --verbose

The matrix profile is O(n^2), so --max-steps caps file length (big datasets like
KDD21 are excluded by the default cap — a follow-up can downsample them).
"""

import argparse
import statistics
from pathlib import Path

from threadforge.data.tab import load_tab_meta, load_tab_univariate
from threadforge.detection import ForecastResidualDetector
from threadforge.oracle import oracle_scores
from threadforge.tab_scoring import vus

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "TAB" / "TAB_dataset" / "dataset" / "anomaly_detect"
META_PATH = DATA_DIR / "DETECT_META.csv"
FILES_DIR = DATA_DIR / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=None, help="restrict to one dataset_name.")
    ap.add_argument("--limit", type=int, default=40, help="files to score. Default 40.")
    ap.add_argument("--max-steps", type=int, default=6000, help="skip series longer than this. Default 6000.")
    ap.add_argument("--m", type=int, default=64, help="matrix-profile subsequence length. Default 64.")
    ap.add_argument("--window", type=int, default=100, help="VUS buffer half-width. Default 100.")
    ap.add_argument("--verbose", action="store_true")
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
    per_ds_o, per_ds_e = {}, {}
    o_all, e_all = [], []
    print(f"Reverse-lens oracle vs EWMA vs TRUTH on {len(meta)} files (m={args.m})", flush=True)
    for i, m in enumerate(meta, start=1):
        path = FILES_DIR / m.file_name
        if not path.exists():
            continue
        stream, labels = load_tab_univariate(path)
        if sum(labels) == 0 or len(stream) < 2 * args.m:
            continue
        values = [v for _, v in stream]
        o = vus(labels, oracle_scores(values, args.m), window=args.window)["VUS_PR"]
        e = vus(labels, ewma.scores(stream), window=args.window)["VUS_PR"]
        per_ds_o.setdefault(m.dataset_name, []).append(o)
        per_ds_e.setdefault(m.dataset_name, []).append(e)
        o_all.append(o)
        e_all.append(e)
        if args.verbose:
            print(f"  [{i}/{len(meta)}] {m.dataset_name}/{m.file_name}: oracle {o:.4f}  EWMA {e:.4f}", flush=True)

    if not o_all:
        print("Nothing scored.")
        return

    print("-" * 60)
    print(f"{'dataset':<16}{'oracle':>10}{'EWMA':>10}{'delta':>10}")
    for ds in sorted(per_ds_o):
        o, e = statistics.mean(per_ds_o[ds]), statistics.mean(per_ds_e[ds])
        print(f"{ds:<16}{o:>10.4f}{e:>10.4f}{o - e:>+10.4f}")
    print("-" * 60)
    o, e = statistics.mean(o_all), statistics.mean(e_all)
    print(f"{'ORACLE (reverse lens)':<22}{o:.4f}   (vs TRUTH)")
    print(f"{'EWMA (causal)':<22}{e:.4f}   (vs TRUTH)")
    print("-" * 60)
    if o > e:
        print(f"the reverse lens recovers true anomalies better than causal EWMA "
              f"(+{o - e:.4f}) -> usable as a teacher.")
    else:
        print(f"the reverse lens does NOT beat causal EWMA at recovering truth "
              f"({o - e:+.4f}) -> not a better teacher here.")


if __name__ == "__main__":
    main()
