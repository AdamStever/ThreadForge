"""Trend overlay vs buy-and-hold — does de-risking in downtrends help?

Uses the signed momentum sensor as a continuous regime (long / flat / short),
then compares to buy-and-hold on the metrics that matter for crash avoidance:
Sharpe and max drawdown, not just total return. Reports full-history and a
walk-forward sweep over the deadband so the chosen setting isn't cherry-picked.

    python scripts/trend_overlay.py --csv data/raw/spy.csv
    python scripts/trend_overlay.py --csv data/raw/spy.csv --allow-short

SIMULATED ONLY (fees + slippage modelled).
"""

import argparse

import numpy as np

from threadforge.market.data import load_ohlcv_csv
from threadforge.market.synthetic import generate_prices
from threadforge.market.trend import evaluate_trend
from threadforge.market.backtest import sharpe, total_return, max_drawdown
from threadforge.market.walkforward import walk_forward_splits


def buy_hold(stream, periods=252) -> dict:
    prices = np.asarray([v for _, v in stream], dtype=float)
    rets = np.diff(prices) / prices[:-1]
    return {"sharpe": sharpe(rets, periods), "return": total_return(rets),
            "max_drawdown": max_drawdown(rets), "trades": 1, "exposure": 1.0}


def _row(label, s):
    print(f"{label:<16}{s['sharpe']:>8.2f}{s['return']:>10.1%}{s['max_drawdown']:>10.1%}"
          f"{s['trades']:>8}{s['exposure']:>10.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None, help="OHLCV CSV. Omit for synthetic.")
    ap.add_argument("--column", default="Close")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--window", type=int, default=30, help="momentum window. Default 30.")
    ap.add_argument("--z-window", type=int, default=60, help="rolling z window. Default 60.")
    ap.add_argument("--allow-short", action="store_true", help="short in downtrends (default long/flat).")
    ap.add_argument("--periods", type=int, default=252, help="annualisation factor.")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fee", type=float, default=0.0001)
    ap.add_argument("--slippage", type=float, default=0.0002)
    args = ap.parse_args()

    if args.csv:
        stream = load_ohlcv_csv(args.csv, column=args.column)
        print(f"data: {args.csv} ({args.column}), {len(stream)} bars", flush=True)
    else:
        stream = generate_prices(args.n, seed=0)
        print(f"data: synthetic ({len(stream)} bars)", flush=True)

    kw = dict(window_size=args.window, z_window=args.z_window, allow_short=args.allow_short,
              fee=args.fee, slippage=args.slippage, periods=args.periods)
    deadbands = [0.0, 0.25, 0.5, 1.0]

    print(f"\nFULL HISTORY  (long/{'short' if args.allow_short else 'flat'})")
    print(f"{'strategy':<16}{'Sharpe':>8}{'return':>10}{'max_dd':>10}{'trades':>8}{'exposure':>10}")
    print("-" * 62)
    _row("buy & hold", buy_hold(stream, args.periods))
    for db in deadbands:
        _row(f"trend db={db}", evaluate_trend(stream, deadband=db, **kw))

    splits = walk_forward_splits(len(stream), args.folds, 0.5)
    if not splits:
        return
    print(f"\nWALK-FORWARD  ({len(splits)} folds, OOS Sharpe)")
    print(f"{'deadband':<16}" + "".join(f"{'f'+str(i):>8}" for i in range(1, len(splits) + 1))
          + f"{'mean':>9}{'b&h mean':>10}")
    print("-" * (16 + 8 * len(splits) + 19))
    bh_means = [buy_hold(stream[te_lo:te_hi], args.periods)["sharpe"]
                for _, (te_lo, te_hi) in splits]
    for db in deadbands:
        fold_sh = [evaluate_trend(stream[te_lo:te_hi], deadband=db, **kw)["sharpe"]
                   for _, (te_lo, te_hi) in splits]
        cells = "".join(f"{s:>8.2f}" for s in fold_sh)
        print(f"trend db={db:<8}{cells}{np.mean(fold_sh):>9.2f}{np.mean(bh_means):>10.2f}")


if __name__ == "__main__":
    main()
