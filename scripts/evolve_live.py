"""Run the online evolving trader — bounded-lookback re-evolution with live promotion.

Unlike scripts/surf.py (fit once on deep history, then test), this walks the stream
forward like a live feed: the champion trades each block out-of-sample, challengers
are re-evolved on only the trailing `--lookback` bars, and a better one is promoted
live. Reports the adaptive champion vs the frozen initial champion vs buy-and-hold,
plus how many promotions happened (evidence it actually adapts).

    python scripts/evolve_live.py --csv data/raw/spy.csv --lookback 252 --reevolve-every 63
    python scripts/evolve_live.py --csv data/raw/spy.csv --lookback 500   # longer memory

SIMULATED ONLY (fees + slippage modelled).
"""

import argparse

import numpy as np

from threadforge.market.data import load_ohlcv_csv
from threadforge.market.synthetic import generate_prices
from threadforge.market.backtest import sharpe, total_return, max_drawdown
from threadforge.market.evolving import EvolveConfig, evolve_live


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None, help="OHLCV CSV. Omit for synthetic.")
    ap.add_argument("--column", default="Close")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--lookback", type=int, default=252, help="bounded memory per re-evolution.")
    ap.add_argument("--reevolve-every", type=int, default=63, help="bars between re-evolutions.")
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gen", type=int, default=10)
    ap.add_argument("--margin", type=float, default=0.10, help="promotion margin (Sharpe-dd).")
    ap.add_argument("--cooldown", type=int, default=126, help="min bars between promotions.")
    ap.add_argument("--dd-penalty", type=float, default=3.0)
    ap.add_argument("--target-vol", type=float, default=0.10,
                    help="annualized vol target for sizing; 0 disables vol targeting.")
    ap.add_argument("--periods", type=int, default=252)
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

    cfg = EvolveConfig(lookback=args.lookback, reevolve_every=args.reevolve_every,
                       pop=args.pop, gen=args.gen, margin=args.margin, cooldown=args.cooldown,
                       dd_penalty=args.dd_penalty, target_vol=(args.target_vol or None),
                       periods=args.periods, fee=args.fee, slippage=args.slippage, seed=args.seed)
    print(f"evolving: lookback={cfg.lookback} reevolve_every={cfg.reevolve_every} "
          f"pop={cfg.pop} gen={cfg.gen} margin={cfg.margin} cooldown={cfg.cooldown} "
          f"target_vol={cfg.target_vol}", flush=True)
    res = evolve_live(stream, cfg)

    # buy-and-hold over the same live region
    prices = np.asarray([v for _, v in stream], dtype=float)[res.start:]
    bh = np.diff(prices) / prices[:-1]

    cards = res.scorecards(args.periods)
    print(f"\nlive region: bars {res.start}..{len(stream)}  "
          f"({res.n_reevolutions} re-evolutions, {len(res.promotions)} promotions)")
    print(f"{'strategy':<22}{'Sharpe':>8}{'return':>10}{'max_dd':>10}")
    print("-" * 50)
    print(f"{'adaptive (evolving)':<22}{cards['adaptive']['sharpe']:>8.2f}"
          f"{cards['adaptive']['return']:>10.1%}{cards['adaptive']['max_drawdown']:>10.1%}")
    print(f"{'static (initial champ)':<22}{cards['static']['sharpe']:>8.2f}"
          f"{cards['static']['return']:>10.1%}{cards['static']['max_drawdown']:>10.1%}")
    print(f"{'buy & hold':<22}{sharpe(bh, args.periods):>8.2f}"
          f"{total_return(bh):>10.1%}{max_drawdown(bh):>10.1%}")
    if res.promotions:
        bars = ", ".join(f"{p.bar}(+{p.delta:.2f})" for p in res.promotions)
        print(f"\npromotions at bars: {bars}")


if __name__ == "__main__":
    main()
