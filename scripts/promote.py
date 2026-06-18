"""Autonomous champion-challenger promotion over the TAB corpus.

Shadow-scores every detector in the registry against a labeled corpus, then
promotes the champion only if a challenger beats it by a real, significant margin
(effect size >= --min-delta AND a one-sided sign test p < --alpha). This closes
the loop: shadows propose, statistics decide, the registry records the winner.

    python scripts/promote.py --seed-demo --dataset YAHOO --limit 60 --dry-run
    python scripts/promote.py --dataset YAHOO --limit 60        # actually promote

Rollback is just `python scripts/registry.py promote <old_id>`.
"""

import argparse
import statistics
from pathlib import Path

from threadforge.data.tab import load_tab_meta, load_tab_univariate
from threadforge.registry import DetectorRegistry
from threadforge.promotion import collect_per_file_scores, decide_promotion
from threadforge.shadow import CHAMPION

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "TAB" / "TAB_dataset" / "dataset" / "anomaly_detect"
META_PATH = DATA_DIR / "DETECT_META.csv"
FILES_DIR = DATA_DIR / "data"
DEFAULT_REGISTRY = ROOT / "registry.json"


def _seed_demo(reg: DetectorRegistry) -> None:
    """Populate an empty registry with a champion + a few challengers."""
    champ = reg.register("ewma_forecast", params={"ewma_alpha": 0.2, "resid_window": 200},
                         metrics={"VUS_PR": 0.196}, notes="baseline champion")
    reg.register("ewma_alpha_0.1", params={"ewma_alpha": 0.1, "resid_window": 200})
    reg.register("ewma_alpha_0.4", params={"ewma_alpha": 0.4, "resid_window": 200})
    reg.register("ewma_window_100", params={"ewma_alpha": 0.2, "resid_window": 100})
    reg.promote(champ.id)


def _load_streams(args) -> list:
    meta = [m for m in load_tab_meta(META_PATH) if m.if_univariate]
    if args.dataset:
        meta = [m for m in meta if m.dataset_name.upper() == args.dataset.upper()]
    if args.max_steps:
        meta = [m for m in meta if m.time_steps <= args.max_steps]
    meta.sort(key=lambda m: m.file_name)
    if args.limit:
        meta = meta[:args.limit]
    streams = []
    for m in meta:
        path = FILES_DIR / m.file_name
        if not path.exists():
            continue
        stream, labels = load_tab_univariate(path)
        if sum(labels) > 0:
            streams.append((stream, labels))
    return streams


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=str(DEFAULT_REGISTRY), help="registry JSON path.")
    ap.add_argument("--dataset", default=None, help="restrict to one dataset_name.")
    ap.add_argument("--limit", type=int, default=60, help="files to score (0 = no limit). Default 60.")
    ap.add_argument("--max-steps", type=int, default=6000, help="skip series longer than this. Default 6000.")
    ap.add_argument("--window", type=int, default=100, help="VUS buffer half-width. Default 100.")
    ap.add_argument("--metric", default="VUS_PR", help="decision metric. Default VUS_PR.")
    ap.add_argument("--min-delta", type=float, default=0.01, help="min mean improvement to promote. Default 0.01.")
    ap.add_argument("--alpha", type=float, default=0.05, help="sign-test significance level. Default 0.05.")
    ap.add_argument("--seed-demo", action="store_true", help="seed the registry with demo candidates if empty.")
    ap.add_argument("--dry-run", action="store_true", help="decide but do not promote.")
    args = ap.parse_args()

    if not META_PATH.exists():
        print(f"TAB metadata not found at {META_PATH} (see data/README.md).")
        return

    reg = DetectorRegistry(args.path)
    if args.seed_demo and not reg.all():
        _seed_demo(reg)
    if reg.champion() is None:
        print("registry has no champion. Seed it (--seed-demo) or `scripts/registry.py register-baseline`.")
        return

    streams = _load_streams(args)
    if not streams:
        print("No matching labeled files. Loosen --dataset / --max-steps / --limit.")
        return

    champ_rec = reg.champion()
    print(f"champion: #{champ_rec.id} {champ_rec.name} | scoring {len(streams)} files "
          f"on {args.metric} (window={args.window})", flush=True)

    champion, challengers, keys = collect_per_file_scores(
        reg, streams, metric=args.metric, window=args.window,
    )
    decision = decide_promotion(
        champion, challengers, metric=args.metric, min_delta=args.min_delta, alpha=args.alpha,
    )

    # head-to-head table
    champ_mean = statistics.mean(champion)
    print("-" * 70)
    print(f"{'':2}{'detector':<22}{args.metric:>10}{'delta':>10}{'wins-losses':>13}{'p':>9}")
    print("-" * 70)
    print(f"* {('#'+str(champ_rec.id)+' '+champ_rec.name):<22}{champ_mean:>10.4f}{'(champ)':>10}{'':>13}{'':>9}")
    rows = []
    for key, vals in challengers.items():
        diffs = [c - h for c, h in zip(vals, champion)]
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        rows.append((statistics.mean(diffs), key, statistics.mean(vals), wins, losses))
    for delta, key, mean, wins, losses in sorted(rows, reverse=True):
        from threadforge.promotion import sign_test_pvalue
        p = sign_test_pvalue(wins, wins + losses)
        print(f"  {key:<22}{mean:>10.4f}{delta:>+10.4f}{f'{wins}-{losses}':>13}{p:>9.4f}")
    print("-" * 70)
    print(f"decision: {decision.reason}")

    if decision.promote:
        if args.dry_run:
            print(f"[dry-run] would promote {decision.challenger} (#{keys[decision.challenger]}).")
        else:
            reg.promote(keys[decision.challenger])
            print(f"PROMOTED {decision.challenger} (#{keys[decision.challenger]}) -> new champion.")
    else:
        print("no promotion — champion stands.")


if __name__ == "__main__":
    main()
