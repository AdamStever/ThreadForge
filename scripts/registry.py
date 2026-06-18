"""Inspect and manage the detector registry.

    python scripts/registry.py list                 # all registered detectors
    python scripts/registry.py show 1               # one record in full
    python scripts/registry.py promote 2            # make record 2 the champion
    python scripts/registry.py register-baseline    # seed the EWMA baseline as champion

The registry lives in `registry.json` at the project root by default
(`--path` to override). It is the ledger champion-challenger promotion writes to.
"""

import argparse
import json
from pathlib import Path

from threadforge.registry import DetectorRegistry

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "registry.json"


def _fmt_metrics(metrics: dict) -> str:
    return ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()) if metrics else "-"


def cmd_list(reg: DetectorRegistry) -> None:
    champ = reg.champion()
    champ_id = champ.id if champ else None
    records = reg.all()
    if not records:
        print("(registry empty — try `register-baseline`)")
        return
    print(f"{'':2}{'id':>3}  {'name':<16}{'metrics':<34}created_at")
    print("-" * 74)
    for r in records:
        mark = "*" if r.id == champ_id else " "
        print(f"{mark} {r.id:>3}  {r.name:<16}{_fmt_metrics(r.metrics):<34}{r.created_at}")
    print("-" * 74)
    print(f"champion: {champ_id if champ_id is not None else '(none)'}  (* marks the champion)")


def cmd_show(reg: DetectorRegistry, record_id: int) -> None:
    print(json.dumps(reg.get(record_id).__dict__, indent=2))


def cmd_promote(reg: DetectorRegistry, record_id: int) -> None:
    rec = reg.promote(record_id)
    print(f"promoted #{rec.id} ({rec.name}) to champion")


def cmd_register_baseline(reg: DetectorRegistry) -> None:
    """Seed the registry with the EWMA forecasting baseline and its corpus scores."""
    rec = reg.register(
        name="ewma_forecast",
        params={"ewma_alpha": 0.2, "resid_window": 200, "window": 100},
        metrics={"VUS_PR": 0.196, "Aff_F1": 0.70},
        notes="EWMA forecast-residual detector; full TAB univariate corpus baseline.",
    )
    reg.promote(rec.id)
    print(f"registered + promoted #{rec.id} ({rec.name}) as the first champion")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=str(DEFAULT_PATH), help="registry JSON path.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p_show = sub.add_parser("show"); p_show.add_argument("id", type=int)
    p_prom = sub.add_parser("promote"); p_prom.add_argument("id", type=int)
    sub.add_parser("register-baseline")
    args = ap.parse_args()

    reg = DetectorRegistry(args.path)
    if args.cmd == "list":
        cmd_list(reg)
    elif args.cmd == "show":
        cmd_show(reg, args.id)
    elif args.cmd == "promote":
        cmd_promote(reg, args.id)
    elif args.cmd == "register-baseline":
        cmd_register_baseline(reg)


if __name__ == "__main__":
    main()
