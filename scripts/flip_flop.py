"""Flip-flop a leveraged pair (TQQQ/SQQQ) on a directional signal.

Hold the 3x-long ETF when the trend is up, switch to the 3x-inverse ETF when it's
down. Compares the flip-flop strategy to just holding each ETF. P&L settles on the
real ETF returns, so decay and switch costs are included.

    python scripts/flip_flop.py --long data/raw/tqqq.csv --short data/raw/sqqq.csv

SIMULATED ONLY.
"""

import argparse

import numpy as np

from threadforge.market.data import load_ohlcv_csv
from threadforge.market.backtest import sharpe, total_return, max_drawdown
from threadforge.market.leveraged import align_pair, flip_flop_pnl
from threadforge.market.perception import signal_series, rolling_z
from threadforge.market.trend import trend_positions
from threadforge.market.walkforward import walk_forward_splits


def _card(pnl, periods):
    return {"sharpe": sharpe(pnl, periods), "return": total_return(pnl),
            "max_drawdown": max_drawdown(pnl)}


def _row(label, c):
    print(f"{label:<26}{c['sharpe']:>8.2f}{c['return']:>12.1%}{c['max_drawdown']:>10.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long", default="data/raw/tqqq.csv", help="3x-long ETF CSV.")
    ap.add_argument("--short", default="data/raw/sqqq.csv", help="3x-inverse ETF CSV.")
    ap.add_argument("--window", type=int, default=30, help="momentum window.")
    ap.add_argument("--z-window", type=int, default=60)
    ap.add_argument("--deadband", type=float, default=0.25, help="trend deadband (sigma).")
    ap.add_argument("--long-flat", action="store_true",
                    help="long TQQQ or cash only (never hold the inverse ETF).")
    ap.add_argument("--periods", type=int, default=252)
    ap.add_argument("--fee", type=float, default=0.0001)
    ap.add_argument("--slippage", type=float, default=0.0002)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()

    long_stream = load_ohlcv_csv(args.long)
    short_stream = load_ohlcv_csv(args.short)
    dates, lp, sp = align_pair(long_stream, short_stream)
    print(f"pair: {args.long} / {args.short}  ({len(dates)} common bars, "
          f"{dates[0]}..{dates[-1]})", flush=True)

    # direction from signed momentum of the long ETF (== Nasdaq direction); +1 long, -1 short
    mom = signal_series([(d, p) for d, p in zip(dates, lp)], "momentum", args.window)
    pos = trend_positions(mom, deadband=args.deadband, allow_short=not args.long_flat, size=1.0,
                          z_window=args.z_window)
    mode = "long TQQQ / cash" if args.long_flat else "flip-flop TQQQ<->SQQQ"
    print(f"mode: {mode}", flush=True)

    flip = flip_flop_pnl(pos, lp, sp, fee=args.fee, slippage=args.slippage)
    hold_long = np.diff(lp) / lp[:-1]
    hold_short = np.diff(sp) / sp[:-1]

    print(f"\nFULL HISTORY{'':14}{'Sharpe':>8}{'return':>12}{'max_dd':>10}")
    print("-" * 56)
    _row("flip-flop (trend)", _card(flip, args.periods))
    _row("buy & hold long (TQQQ)", _card(hold_long, args.periods))
    _row("buy & hold short (SQQQ)", _card(hold_short, args.periods))
    exposure_long = float(np.mean(pos > 0))
    print(f"\nflip-flop was long {exposure_long:.0%} of bars, short {np.mean(pos < 0):.0%}, "
          f"flat {np.mean(pos == 0):.0%}")

    # walk-forward: is the flip-flop's edge over buy-and-hold-long stable across periods?
    splits = walk_forward_splits(len(dates), args.folds, 0.5)
    if splits:
        print(f"\nWALK-FORWARD ({len(splits)} folds, OOS Sharpe)")
        print(f"{'':26}{'flip-flop':>10}{'hold TQQQ':>11}")
        for i, (_, (lo, hi)) in enumerate(splits, start=1):
            f = sharpe(flip_flop_pnl(pos[lo:hi], lp[lo:hi], sp[lo:hi],
                                     fee=args.fee, slippage=args.slippage), args.periods)
            h = sharpe(np.diff(lp[lo:hi]) / lp[lo:hi][:-1], args.periods)
            print(f"{'fold ' + str(i):<26}{f:>10.2f}{h:>11.2f}")


if __name__ == "__main__":
    main()
