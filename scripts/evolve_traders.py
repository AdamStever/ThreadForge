"""Evolve trading agents on synthetic market data, graded by Sharpe (paper-traded).

Each agent is a 10-signal WeightedSignalDetector + an evolvable trade rule (fade
vs follow, threshold, size). The genetic search optimises in-sample Sharpe; the
agent is then judged **out-of-sample** on a held-out segment — the honest test,
because P&L is the most overfit-prone fitness we've used.

    python scripts/evolve_traders.py
    python scripts/evolve_traders.py --n 5000 --pop 30 --gen 20

SIMULATED ONLY — no real money or live execution.
"""

import argparse
import random
from pathlib import Path

from threadforge.market.synthetic import generate_prices
from threadforge.market.agent import trader_genes, agent_from_genome
from threadforge.market.backtest import sharpe, total_return, max_drawdown
import numpy as np
from threadforge.optimization.genetic import evolve
from threadforge.presets import default_signal_names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4000, help="total series length. Default 4000.")
    ap.add_argument("--train-frac", type=float, default=0.7, help="train fraction. Default 0.7.")
    ap.add_argument("--seed", type=int, default=0, help="data + GA seed. Default 0.")
    ap.add_argument("--pop", type=int, default=24, help="GA population. Default 24.")
    ap.add_argument("--gen", type=int, default=15, help="GA generations. Default 15.")
    args = ap.parse_args()

    stream = generate_prices(args.n, seed=args.seed)
    split = int(args.n * args.train_frac)
    train, test = stream[:split], stream[split:]
    print(f"Evolving traders | train={len(train)} test={len(test)} | pop={args.pop} gen={args.gen}",
          flush=True)

    genes = trader_genes()
    evals = {"n": 0}

    def fitness(genome: dict) -> float:
        evals["n"] += 1
        s = agent_from_genome(genome).evaluate(train)["sharpe"]
        if evals["n"] % 25 == 0:
            print(f"  eval {evals['n']:>3}: best-so-far train Sharpe tracked by GA", flush=True)
        return s

    best, best_fit, _ = evolve(genes, fitness, pop_size=args.pop, generations=args.gen,
                               rng=random.Random(args.seed))

    agent = agent_from_genome(best)
    tr = agent.evaluate(train)
    te = agent.evaluate(test)

    # reference: buy-and-hold on the test segment
    prices = np.asarray([v for _, v in test], dtype=float)
    bh = np.diff(prices) / prices[:-1]
    bh_sharpe = sharpe(bh)

    print("=" * 60)
    print("evolved agent — trade rule:")
    print(f"  mode={agent.mode}  threshold={agent.threshold:.2f}  size={agent.size:.2f}")
    print("  signal weights (high -> low):")
    for n, w in sorted(((k, best[k]) for k in default_signal_names()), key=lambda kv: kv[1], reverse=True):
        print(f"    {n:<18}{w:.3f}")
    print("-" * 60)
    print(f"{'':12}{'Sharpe':>10}{'return':>10}{'max_dd':>10}")
    print(f"{'in-sample':<12}{tr['sharpe']:>10.2f}{tr['return']:>10.2%}{tr['max_drawdown']:>10.2%}")
    print(f"{'OUT-SAMPLE':<12}{te['sharpe']:>10.2f}{te['return']:>10.2%}{te['max_drawdown']:>10.2%}")
    print(f"{'buy & hold':<12}{bh_sharpe:>10.2f}   (test segment)")
    print("=" * 60)
    if te["sharpe"] > bh_sharpe and te["sharpe"] > 0:
        print("evolved agent beats buy-and-hold out-of-sample (paper).")
    else:
        print("evolved agent does NOT beat buy-and-hold out-of-sample -- likely in-sample overfit.")


if __name__ == "__main__":
    main()
