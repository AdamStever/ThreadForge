"""Walk-forward evolution — evolve traders per fold, judge each out-of-sample.

Anti-overfit on two fronts: (1) an **activity floor** rejects agents that "win" by
barely trading (the degenerate failure we saw on SPY), and (2) **walk-forward**
re-runs across several expanding train->test folds so one lucky split can't decide
it. Reports per-fold OOS Sharpe vs buy-and-hold and the average.

    python scripts/walk_forward.py --csv data/raw/spy_5m.csv --folds 3
    python scripts/walk_forward.py --csv data/raw/spy.csv --folds 4 --min-trades 15

SIMULATED ONLY (fees + slippage modelled).
"""

import argparse
import random
import statistics
from pathlib import Path

import numpy as np

from threadforge.market.synthetic import generate_prices
from threadforge.market.data import load_ohlcv_csv
from threadforge.market.agent import trader_genes, agent_from_genome
from threadforge.market.backtest import sharpe
from threadforge.market.walkforward import walk_forward_splits
from threadforge.optimization.genetic import evolve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None, help="OHLCV CSV. Omit for synthetic.")
    ap.add_argument("--column", default="Close")
    ap.add_argument("--n", type=int, default=4000, help="synthetic length if no CSV.")
    ap.add_argument("--folds", type=int, default=3, help="walk-forward folds. Default 3.")
    ap.add_argument("--min-train-frac", type=float, default=0.5)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--min-trades", type=int, default=20, help="activity floor (per train window).")
    ap.add_argument("--fee", type=float, default=0.0001)
    ap.add_argument("--slippage", type=float, default=0.0002)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.csv:
        stream = load_ohlcv_csv(args.csv, column=args.column)
        print(f"data: {args.csv} ({args.column}), {len(stream)} bars", flush=True)
    else:
        stream = generate_prices(args.n, seed=args.seed)
        print(f"data: synthetic ({len(stream)} bars)", flush=True)

    splits = walk_forward_splits(len(stream), args.folds, args.min_train_frac)
    if not splits:
        print("Not enough data for the requested folds.")
        return

    genes = trader_genes()
    cost_kw = dict(fee=args.fee, slippage=args.slippage)
    oos_sharpes, bh_sharpes = [], []

    print(f"{'fold':>4}{'train':>8}{'test':>8}{'OOS Sharpe':>12}{'trades':>8}{'buy&hold':>10}")
    print("-" * 52)
    for i, ((tr_lo, tr_hi), (te_lo, te_hi)) in enumerate(splits, start=1):
        train, test = stream[tr_lo:tr_hi], stream[te_lo:te_hi]

        def fitness(genome: dict) -> float:
            res = agent_from_genome(genome, **cost_kw).evaluate(train)
            if res["trades"] < args.min_trades:
                return res["sharpe"] - 5.0          # activity floor: punish near-inactive agents
            return res["sharpe"]

        best, _, _ = evolve(genes, fitness, pop_size=args.pop, generations=args.gen,
                            rng=random.Random(args.seed + i))
        te = agent_from_genome(best, **cost_kw).evaluate(test)

        prices = np.asarray([v for _, v in test], dtype=float)
        bh = sharpe(np.diff(prices) / prices[:-1])
        oos_sharpes.append(te["sharpe"])
        bh_sharpes.append(bh)
        print(f"{i:>4}{len(train):>8}{len(test):>8}{te['sharpe']:>12.2f}{te['trades']:>8}{bh:>10.2f}",
              flush=True)

    print("-" * 52)
    print(f"mean OOS Sharpe: {statistics.mean(oos_sharpes):>+.2f}   "
          f"buy&hold: {statistics.mean(bh_sharpes):>+.2f}")
    wins = sum(o > b for o, b in zip(oos_sharpes, bh_sharpes))
    print(f"folds beating buy&hold OOS: {wins}/{len(splits)}")


if __name__ == "__main__":
    main()
