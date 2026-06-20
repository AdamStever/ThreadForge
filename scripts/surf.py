"""Surf the market — champion (linear) vs challenger (neural) on out-of-sample P&L.

Each bar, a policy maps the signed 10-signal state vector to a continuous position
and rides it (no anomaly gating). Two policies compete via walk-forward:

  * champion   = LinearPolicy, weights evolved by the genetic search
  * challenger = NeuralPolicy, trained by gradient descent through the backtest

Both are selected/scored by the SAME objective: Sharpe minus a drawdown penalty,
out-of-sample. Reports per-fold OOS scorecards for both vs buy-and-hold, and which
policy wins each fold.

    python scripts/surf.py --csv data/raw/spy.csv --folds 3
    python scripts/surf.py --csv data/raw/spy_5m.csv --folds 3 --periods 19656

SIMULATED ONLY (fees + slippage modelled).
"""

import argparse
import random

import numpy as np

from threadforge.market.data import load_ohlcv_csv
from threadforge.market.synthetic import generate_prices
from threadforge.market.backtest import sharpe, total_return, max_drawdown
from threadforge.market.policy import (
    PolicyTrader, fitness_score, policy_genes, linear_from_genome,
)
from threadforge.market.walkforward import walk_forward_splits
from threadforge.optimization.genetic import evolve


def buy_hold(stream, periods):
    prices = np.asarray([v for _, v in stream], dtype=float)
    rets = np.diff(prices) / prices[:-1]
    return {"sharpe": sharpe(rets, periods), "return": total_return(rets),
            "max_drawdown": max_drawdown(rets), "exposure": 1.0, "turnover": 0.0}


def evolve_linear(train, cost_kw, dd_penalty, pop, gen, seed):
    genes = policy_genes()

    def fitness(genome):
        return fitness_score(linear_from_genome(genome, **cost_kw).evaluate(train), dd_penalty)

    best, _, _ = evolve(genes, fitness, pop_size=pop, generations=gen, rng=random.Random(seed))
    return linear_from_genome(best, **cost_kw)


def _row(label, s):
    print(f"  {label:<12}{s['sharpe']:>8.2f}{s['return']:>10.1%}{s['max_drawdown']:>10.1%}"
          f"{s['exposure']:>10.1%}{s['turnover']:>9.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None, help="OHLCV CSV. Omit for synthetic.")
    ap.add_argument("--column", default="Close")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--min-train-frac", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--z-window", type=int, default=60)
    ap.add_argument("--periods", type=int, default=252, help="annualisation factor (daily=252).")
    ap.add_argument("--dd-penalty", type=float, default=3.0, help="drawdown penalty in selection.")
    ap.add_argument("--pop", type=int, default=20, help="GA population (linear).")
    ap.add_argument("--gen", type=int, default=12, help="GA generations (linear).")
    ap.add_argument("--epochs", type=int, default=400, help="training epochs (neural).")
    ap.add_argument("--risk-penalty", type=float, default=1.0,
                    help="neural training downside penalty (lower = more exposure). Default 1.0.")
    ap.add_argument("--fee", type=float, default=0.0001)
    ap.add_argument("--slippage", type=float, default=0.0002)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-neural", action="store_true", help="skip the MLP neural challenger.")
    ap.add_argument("--recurrent", action="store_true", help="add the LSTM (memory) challenger.")
    ap.add_argument("--hidden", type=int, default=32, help="LSTM hidden size. Default 32.")
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

    cost_kw = dict(window_size=args.window, z_window=args.z_window,
                   fee=args.fee, slippage=args.slippage, periods=args.periods)
    score = lambda s: fitness_score(s, args.dd_penalty)
    agg = {"linear": [], "neural": [], "recurrent": [], "bh": []}
    wins = {"linear": 0, "neural": 0, "recurrent": 0, "buy&hold": 0}

    for i, ((tr_lo, tr_hi), (te_lo, te_hi)) in enumerate(splits, start=1):
        train, test = stream[tr_lo:tr_hi], stream[te_lo:te_hi]
        print(f"\nfold {i}: train={len(train)} test={len(test)}", flush=True)
        print(f"  {'policy':<12}{'Sharpe':>8}{'return':>10}{'max_dd':>10}{'exposure':>10}{'turnover':>9}")

        bh = buy_hold(test, args.periods)
        _row("buy & hold", bh)
        agg["bh"].append(bh)

        lin = evolve_linear(train, cost_kw, args.dd_penalty, args.pop, args.gen, args.seed + i)
        lin_te = lin.evaluate(test)
        _row("linear", lin_te)
        agg["linear"].append(lin_te)

        results = {"linear": lin_te, "buy&hold": bh}
        if not args.no_neural:
            from threadforge.market.neural_policy import neural_trader
            neu = neural_trader(train, epochs=args.epochs, risk_penalty=args.risk_penalty,
                                seed=args.seed + i, **cost_kw)
            neu_te = neu.evaluate(test)
            _row("neural", neu_te)
            agg["neural"].append(neu_te)
            results["neural"] = neu_te
        if args.recurrent:
            from threadforge.market.recurrent_policy import recurrent_trader
            rec = recurrent_trader(train, epochs=args.epochs, risk_penalty=args.risk_penalty,
                                   hidden=args.hidden, seed=args.seed + i, **cost_kw)
            rec_te = rec.evaluate(test)
            _row("recurrent", rec_te)
            agg["recurrent"].append(rec_te)
            results["recurrent"] = rec_te

        winner = max(results, key=lambda k: score(results[k]))
        wins[winner] += 1
        print(f"  -> fold winner (Sharpe-dd): {winner}", flush=True)

    def mean_sh(xs):
        return float(np.mean([s["sharpe"] for s in xs])) if xs else float("nan")

    print("\n" + "=" * 56)
    parts = [f"linear={mean_sh(agg['linear']):+.2f}", f"neural={mean_sh(agg['neural']):+.2f}"]
    if args.recurrent:
        parts.append(f"recurrent={mean_sh(agg['recurrent']):+.2f}")
    parts.append(f"buy&hold={mean_sh(agg['bh']):+.2f}")
    print("mean OOS Sharpe   " + "   ".join(parts))
    print(f"fold wins (Sharpe-dd): {dict(wins)}")


if __name__ == "__main__":
    main()
