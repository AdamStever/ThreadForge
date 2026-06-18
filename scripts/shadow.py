"""Shadow-detection demo: score challenger detectors against the champion, silently.

Runs a champion detector plus several challengers over TAB univariate files. Only
the champion would alert in production; the challengers are scored in the shadows.
Each detector is evaluated on the trusted metrics (VUS-PR, Aff-F1) and the results
are macro-averaged across files into a head-to-head comparison.

    python scripts/shadow.py                         # built-in EWMA candidates, small subset
    python scripts/shadow.py --dataset YAHOO --limit 40
    python scripts/shadow.py --registry registry.json   # candidates from the registry

This produces the comparison; it does NOT promote anything — acting on it
(champion-challenger promotion) is the next task.
"""

import argparse
import statistics
from pathlib import Path

from threadforge.data.tab import load_tab_meta, load_tab_univariate
from threadforge.detection.online_forecast import OnlineForecastResidualDetector
from threadforge.registry import DetectorRegistry
from threadforge.shadow import ShadowRuntime, best_challenger, build_detector, CHAMPION

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "TAB" / "TAB_dataset" / "dataset" / "anomaly_detect"
META_PATH = DATA_DIR / "DETECT_META.csv"
FILES_DIR = DATA_DIR / "data"

# Built-in candidate detectors (the champion is the registered baseline config).
BUILTIN = {
    CHAMPION:      {"ewma_alpha": 0.2, "resid_window": 200},
    "alpha=0.1":   {"ewma_alpha": 0.1, "resid_window": 200},
    "alpha=0.4":   {"ewma_alpha": 0.4, "resid_window": 200},
    "window=100":  {"ewma_alpha": 0.2, "resid_window": 100},
}


def _make(params: dict, probation: int):
    return OnlineForecastResidualDetector(
        ewma_alpha=params.get("ewma_alpha", 0.2),
        resid_window=params.get("resid_window", 200),
        probation=probation,
        min_history=params.get("min_history", 20),
    )


def _candidates_from_registry(path: str) -> dict[str, dict]:
    reg = DetectorRegistry(path)
    champ = reg.champion() or reg.best("VUS_PR") or reg.latest()
    if champ is None:
        raise SystemExit(f"registry {path} is empty")
    cands = {CHAMPION: champ.params}
    for r in reg.all():
        if r.id != champ.id:
            cands[f"#{r.id} {r.name}"] = r.params
    return cands


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=None, help="restrict to one dataset_name.")
    ap.add_argument("--limit", type=int, default=30, help="files to score (0 = no limit). Default 30.")
    ap.add_argument("--max-steps", type=int, default=6000, help="skip series longer than this. Default 6000.")
    ap.add_argument("--window", type=int, default=100, help="VUS buffer half-width. Default 100.")
    ap.add_argument("--aff-threshold", type=float, default=2.5, help="Aff-F1 flag threshold. Default 2.5.")
    ap.add_argument("--registry", default=None, help="load candidates from a registry JSON instead of built-ins.")
    args = ap.parse_args()

    if not META_PATH.exists():
        print(f"TAB metadata not found at {META_PATH} (see data/README.md).")
        return

    candidates = _candidates_from_registry(args.registry) if args.registry else BUILTIN

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

    # accumulate per-detector metric lists across files
    vus_by: dict[str, list[float]] = {n: [] for n in candidates}
    aff_by: dict[str, list[float]] = {n: [] for n in candidates}
    scored = 0

    print(f"Shadow-scoring {len(meta)} files | champion + {len(candidates) - 1} challenger(s)", flush=True)
    for m in meta:
        path = FILES_DIR / m.file_name
        if not path.exists():
            continue
        stream, labels = load_tab_univariate(path)
        if sum(labels) == 0:
            continue
        probation = min(int(0.15 * len(stream)), 750)
        champion = _make(candidates[CHAMPION], probation)
        challengers = {n: _make(p, probation) for n, p in candidates.items() if n != CHAMPION}
        rt = ShadowRuntime(champion, challengers, threshold=10.0, gap_steps=20)
        rt.run(stream)
        comp = rt.evaluate(labels, window=args.window, aff_threshold=args.aff_threshold)
        for name, mets in comp.items():
            vus_by[name].append(mets["VUS_PR"])
            aff_by[name].append(mets["Aff_F1"])
        scored += 1

    if not scored:
        print("Nothing scored.")
        return

    # macro-average and present the head-to-head
    comparison = {n: {"VUS_PR": statistics.mean(vus_by[n]), "Aff_F1": statistics.mean(aff_by[n])}
                  for n in candidates}
    champ_vus = comparison[CHAMPION]["VUS_PR"]

    print("-" * 64)
    print(f"{'':2}{'detector':<16}{'VUS_PR':>10}{'Aff_F1':>10}{'vs champ':>12}")
    print("-" * 64)
    order = sorted(comparison, key=lambda n: comparison[n]["VUS_PR"], reverse=True)
    for n in order:
        mark = "*" if n == CHAMPION else " "
        d = comparison[n]["VUS_PR"] - champ_vus
        delta = "  (champion)" if n == CHAMPION else f"{d:+.4f}"
        print(f"{mark} {n:<16}{comparison[n]['VUS_PR']:>10.4f}{comparison[n]['Aff_F1']:>10.4f}{delta:>12}")
    print("-" * 64)
    print(f"(* champion; {scored} files; macro-averaged)")

    bc = best_challenger(comparison, "VUS_PR")
    if bc and bc["beats_champion"]:
        print(f"\nbest challenger: {bc['name']} beats champion on VUS-PR by {bc['delta']:+.4f} "
              f"-> a promotion candidate (decision is the next task).")
    elif bc:
        print(f"\nbest challenger: {bc['name']} ({bc['delta']:+.4f} VUS-PR) does not beat the champion.")


if __name__ == "__main__":
    main()
