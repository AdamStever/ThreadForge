"""Score the forecasting detector on the TAB univariate corpus by VUS-PR.

TAB's headline metric is VUS-PR, computed per series and averaged across the
corpus. The forecasting detector is unsupervised and online, so it follows TAB's
intent directly: emit a per-step anomaly score, then let VUS-PR sweep the
threshold internally — no decision threshold to pick.

    python scripts/run_tab.py                      # quick subset (small files)
    python scripts/run_tab.py --limit 0            # whole univariate corpus (slow)
    python scripts/run_tab.py --dataset NAB        # one source only
    python scripts/run_tab.py --limit 40 --max-steps 10000 --window 100

The VUS computation is a faithful (unoptimised) port, so large files are slow;
``--max-steps`` skips series longer than the cap and ``--limit`` bounds the count
so a meaningful number comes back quickly. Aligning the buffer ``--window`` with
TAB's per-series window selection is a later refinement.
"""

import argparse
import statistics
from pathlib import Path

from threadforge.data.tab import load_tab_meta, load_tab_univariate
from threadforge.detection import ForecastResidualDetector
from threadforge.tab_scoring import vus

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "TAB" / "TAB_dataset" / "dataset" / "anomaly_detect"
META_PATH = DATA_DIR / "DETECT_META.csv"
FILES_DIR = DATA_DIR / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=20,
                    help="max files to score (0 = no limit). Default 20.")
    ap.add_argument("--dataset", type=str, default=None,
                    help="restrict to one dataset_name (e.g. NAB, YAHOO, KDD21).")
    ap.add_argument("--max-steps", type=int, default=8000,
                    help="skip series longer than this (0 = no cap). Default 8000.")
    ap.add_argument("--window", type=int, default=100, help="VUS buffer half-width. Default 100.")
    ap.add_argument("--thre", type=int, default=250, help="VUS threshold sweep size. Default 250.")
    ap.add_argument("--verbose", action="store_true", help="print a line per file.")
    args = ap.parse_args()

    if not META_PATH.exists():
        print(f"TAB metadata not found at {META_PATH}")
        print("Place the TAB dataset bundle under data/TAB/ (see data/README.md).")
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
        print("No matching univariate files. Loosen --dataset / --max-steps / --limit.")
        return

    detector = ForecastResidualDetector()
    per_dataset: dict[str, list[float]] = {}
    all_scores: list[float] = []
    skipped = 0

    print(f"Scoring {len(meta)} univariate files  (window={args.window}, thre={args.thre})")
    if args.verbose:
        print(f"{'VUS_PR':>8}  {'steps':>7}  dataset / file")
        print("-" * 60)

    for m in meta:
        path = FILES_DIR / m.file_name
        if not path.exists():
            skipped += 1
            continue
        stream, labels = load_tab_univariate(path)
        if sum(labels) == 0:
            skipped += 1
            continue
        scores = detector.scores(stream)
        vpr = vus(labels, scores, window=args.window, thre=args.thre)["VUS_PR"]
        per_dataset.setdefault(m.dataset_name, []).append(vpr)
        all_scores.append(vpr)
        if args.verbose:
            print(f"{vpr:>8.4f}  {m.time_steps:>7}  {m.dataset_name} / {m.file_name}")

    if not all_scores:
        print("Nothing scored (files missing or unlabeled).")
        return

    print("-" * 60)
    print(f"{'dataset':<18}{'files':>7}{'mean VUS_PR':>14}")
    for name in sorted(per_dataset):
        vals = per_dataset[name]
        print(f"{name:<18}{len(vals):>7}{statistics.mean(vals):>14.4f}")
    print("-" * 60)
    print(f"{'CORPUS (macro)':<18}{len(all_scores):>7}{statistics.mean(all_scores):>14.4f}")
    if skipped:
        print(f"({skipped} files skipped: missing or unlabeled)")


if __name__ == "__main__":
    main()
