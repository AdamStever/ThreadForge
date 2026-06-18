"""Online champion-challenger promotion over the full TAB corpus.

Unlike `promote.py` (one decision after a full batch), this promotes *as the run
goes*: it walks the corpus file by file and switches champion the moment a
challenger's lead is statistically established, so gains accrue on the remaining
files. It reports the resulting **adaptive** VUS-PR against the static champion
(never switches) and the best single fixed detector (the offline oracle).

    python scripts/run_promote_online.py                       # quick subset
    python scripts/run_promote_online.py --limit 0 --max-steps 0   # full corpus (slow)

Scoring every candidate on every file is the cost; it fans out across cores.
"""

import argparse
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from threadforge.data.tab import load_tab_meta, load_tab_univariate
from threadforge.detection import ForecastResidualDetector
from threadforge.promotion import sequential_promotion
from threadforge.tab_scoring import vus

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "TAB" / "TAB_dataset" / "dataset" / "anomaly_detect"
META_PATH = DATA_DIR / "DETECT_META.csv"
FILES_DIR = DATA_DIR / "data"

# Candidate detectors. The champion is the registered baseline (alpha 0.2).
CANDIDATES = {
    "ewma_a0.2": {"ewma_alpha": 0.2, "resid_window": 200},   # champion
    "ewma_a0.3": {"ewma_alpha": 0.3, "resid_window": 200},
    "ewma_a0.4": {"ewma_alpha": 0.4, "resid_window": 200},
    "ewma_a0.5": {"ewma_alpha": 0.5, "resid_window": 200},
}
CHAMPION_KEY = "ewma_a0.2"


def _fmt_dur(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def _score_one(task: tuple) -> tuple:
    index, file_name, window, thre = task
    try:
        path = FILES_DIR / file_name
        if not path.exists():
            return (index, None, "missing")
        stream, labels = load_tab_univariate(path)
        if sum(labels) == 0:
            return (index, None, "unlabeled")
        out = {}
        for name, p in CANDIDATES.items():
            det = ForecastResidualDetector(ewma_alpha=p["ewma_alpha"], resid_window=p["resid_window"])
            scores = det.scores(stream)
            out[name] = vus(labels, scores, window=window, thre=thre)["VUS_PR"]
        return (index, out, "ok")
    except Exception as exc:
        return (index, None, f"error: {type(exc).__name__}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=40, help="files to score (0 = no limit). Default 40.")
    ap.add_argument("--max-steps", type=int, default=8000, help="skip series longer than this. Default 8000.")
    ap.add_argument("--window", type=int, default=100, help="VUS buffer half-width. Default 100.")
    ap.add_argument("--thre", type=int, default=250, help="VUS threshold sweep size. Default 250.")
    ap.add_argument("--workers", type=int, default=0, help="worker processes (0 = all cores).")
    ap.add_argument("--min-files", type=int, default=30, help="evidence before any promotion. Default 30.")
    ap.add_argument("--min-delta", type=float, default=0.01, help="min mean improvement to promote. Default 0.01.")
    ap.add_argument("--alpha", type=float, default=0.05, help="sign-test significance. Default 0.05.")
    ap.add_argument("--cooldown", type=int, default=20, help="min files between promotions. Default 20.")
    ap.add_argument("--progress-every", type=int, default=100, help="progress line cadence. Default 100.")
    args = ap.parse_args()

    if not META_PATH.exists():
        print(f"TAB metadata not found at {META_PATH} (see data/README.md).")
        return

    meta = [m for m in load_tab_meta(META_PATH) if m.if_univariate]
    if args.max_steps:
        meta = [m for m in meta if m.time_steps <= args.max_steps]
    meta.sort(key=lambda m: m.file_name)          # deterministic "run order"
    if args.limit:
        meta = meta[:args.limit]
    if not meta:
        print("No matching univariate files.")
        return

    tasks = [(i, m.file_name, args.window, args.thre) for i, m in enumerate(meta)]
    total_steps = sum(m.time_steps for m in meta)
    done_steps = 0
    steps_by_index = {i: m.time_steps for i, m in enumerate(meta)}
    workers = min(args.workers or (os.cpu_count() or 1), len(tasks))
    print(f"Scoring {len(tasks)} files x {len(CANDIDATES)} detectors "
          f"(window={args.window}, workers={workers})", flush=True)

    results: dict[int, dict] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_score_one, t) for t in tasks]
        for done, fut in enumerate(as_completed(futures), start=1):
            index, out, status = fut.result()
            done_steps += steps_by_index[index]
            if status == "ok":
                results[index] = out
            if args.progress_every and (done % args.progress_every == 0 or done == len(tasks)):
                frac = done_steps / total_steps if total_steps else 0
                eta = (time.time() - t0) * (1 - frac) / frac if frac > 0 else 0
                print(f"[{frac*100:5.1f}%] {done}/{len(tasks)} files | "
                      f"elapsed {_fmt_dur(time.time()-t0)} | ETA {_fmt_dur(eta)}", flush=True)

    order = sorted(results)                         # successfully-scored files, in run order
    if not order:
        print("Nothing scored.")
        return
    per_file = {name: [results[i][name] for i in order] for name in CANDIDATES}

    res = sequential_promotion(
        per_file, CHAMPION_KEY,
        min_files=args.min_files, min_delta=args.min_delta, alpha=args.alpha, cooldown=args.cooldown,
    )

    static_mean = statistics.mean(res.static_scores)
    adaptive_mean = statistics.mean(res.adaptive_scores)
    best_name = max(CANDIDATES, key=lambda n: statistics.mean(per_file[n]))
    best_mean = statistics.mean(per_file[best_name])

    print("=" * 64)
    print(f"files scored: {len(order)}   metric: VUS-PR (window={args.window})")
    print("-" * 64)
    print(f"static champion ({CHAMPION_KEY}):     {static_mean:.4f}")
    print(f"online adaptive (promotes as it goes): {adaptive_mean:.4f}   "
          f"({adaptive_mean - static_mean:+.4f} vs static)")
    print(f"best fixed detector ({best_name}, oracle): {best_mean:.4f}")
    print("-" * 64)
    print(f"promotions: {len(res.events)}   final champion: {res.final_champion}")
    for e in res.events[:12]:
        print(f"  @file {e.file_index:>4}: {e.from_champion} -> {e.to_champion} "
              f"(delta {e.delta:+.4f}, {e.wins}-{e.losses}, p={e.p_value:.4f})")
    if len(res.events) > 12:
        print(f"  ... and {len(res.events) - 12} more")


if __name__ == "__main__":
    main()
