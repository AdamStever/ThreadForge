"""Score the forecasting detector on the TAB univariate corpus by VUS-PR.

TAB's headline metric is VUS-PR, computed per series and averaged across the
corpus. The forecasting detector is unsupervised and online, so it follows TAB's
intent directly: emit a per-step anomaly score, then let VUS-PR sweep the
threshold internally — no decision threshold to pick.

    python scripts/run_tab.py                      # quick subset (small files)
    python scripts/run_tab.py --limit 0            # whole univariate corpus (slow)
    python scripts/run_tab.py --dataset NAB        # one source only
    python scripts/run_tab.py --limit 40 --max-steps 10000 --window 100

Scoring fans out across CPU cores (the files are independent): ``--workers``
defaults to all cores, ``--workers 1`` forces the serial path. A step-keyed
progress line (percent done + ETA) prints every ``--progress-every`` files.
``--max-steps`` skips series longer than the cap and ``--limit`` bounds the count
so a meaningful number comes back quickly. Aligning the buffer ``--window`` with
TAB's per-series window selection is a later refinement.

    python scripts/run_tab.py --limit 0 --max-steps 0          # full corpus, all cores
    python scripts/run_tab.py --limit 0 --max-steps 0 --workers 4
"""

import argparse
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Pin each process to single-threaded BLAS so the process pool — not numpy's
# internal threads — provides the parallelism. Otherwise N workers each spawning
# M BLAS threads oversubscribes the cores and runs slower. Must be set before
# numpy is imported (which happens via threadforge.tab_scoring below).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from threadforge.data.tab import load_tab_meta, load_tab_univariate
from threadforge.detection import ForecastResidualDetector
from threadforge.tab_scoring import vus

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "TAB" / "TAB_dataset" / "dataset" / "anomaly_detect"
META_PATH = DATA_DIR / "DETECT_META.csv"
FILES_DIR = DATA_DIR / "data"


def _fmt_dur(seconds: float) -> str:
    """Human-friendly duration: 45s, 3m12s, 1h04m."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _progress(done_steps: int, total_steps: int, files_done: int, n_files: int, t0: float) -> str:
    """A progress line keyed on steps processed (robust to wildly varying file sizes)."""
    elapsed = time.time() - t0
    frac = done_steps / total_steps if total_steps else 0.0
    eta = elapsed * (1 - frac) / frac if frac > 0 else 0.0
    return (f"[{frac * 100:5.1f}%] {files_done}/{n_files} files | "
            f"{done_steps:,}/{total_steps:,} steps | "
            f"elapsed {_fmt_dur(elapsed)} | ETA {_fmt_dur(eta)}")


def _score_one(task: tuple) -> tuple:
    """Worker: score one file by VUS-PR. Runs in a separate process.

    Returns ``(file_name, dataset_name, vpr_or_None, steps, status)`` where status
    is "ok", "missing", or "unlabeled".
    """
    file_name, dataset_name, steps, window, thre = task
    path = FILES_DIR / file_name
    if not path.exists():
        return (file_name, dataset_name, None, steps, "missing")
    stream, labels = load_tab_univariate(path)
    if sum(labels) == 0:
        return (file_name, dataset_name, None, steps, "unlabeled")
    scores = ForecastResidualDetector().scores(stream)
    vpr = vus(labels, scores, window=window, thre=thre)["VUS_PR"]
    return (file_name, dataset_name, vpr, steps, "ok")


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
    ap.add_argument("--progress-every", type=int, default=25,
                    help="print a %%-done + ETA line every N files (0 to disable). Default 25.")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel worker processes (0 = all CPU cores, 1 = serial). "
                         "Default 0. Lower it if memory is tight on large files.")
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

    per_dataset: dict[str, list[float]] = {}
    all_scores: list[float] = []
    skipped = 0

    tasks = [(m.file_name, m.dataset_name, m.time_steps, args.window, args.thre) for m in meta]
    total_steps = sum(m.time_steps for m in meta)
    workers = min(args.workers or (os.cpu_count() or 1), len(tasks))
    print(f"Scoring {len(meta)} univariate files, {total_steps:,} steps  "
          f"(window={args.window}, thre={args.thre}, workers={workers})", flush=True)
    if args.verbose:
        print(f"{'VUS_PR':>8}  {'steps':>7}  dataset / file")
        print("-" * 60)

    t0 = time.time()
    done_steps = 0

    def handle(res: tuple, files_done: int) -> None:
        nonlocal done_steps, skipped
        file_name, dataset_name, vpr, steps, status = res
        done_steps += steps
        if status == "ok":
            per_dataset.setdefault(dataset_name, []).append(vpr)
            all_scores.append(vpr)
            if args.verbose:
                print(f"{vpr:>8.4f}  {steps:>7}  {dataset_name} / {file_name}", flush=True)
        else:
            skipped += 1
        if args.progress_every and (files_done % args.progress_every == 0 or files_done == len(tasks)):
            print(_progress(done_steps, total_steps, files_done, len(tasks), t0), flush=True)

    # Files are independent, so scoring fans out across processes. Results arrive
    # in completion order (not input order), which only affects --verbose line
    # order; the per-dataset and corpus aggregates are order-independent.
    if workers == 1:
        for files_done, task in enumerate(tasks, start=1):
            handle(_score_one(task), files_done)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_score_one, t) for t in tasks]
            for files_done, fut in enumerate(as_completed(futures), start=1):
                handle(fut.result(), files_done)

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
