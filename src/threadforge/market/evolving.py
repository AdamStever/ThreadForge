"""Online evolving trader — bounded-lookback re-evolution with live promotion.

This is the *evolving system*, as opposed to the fit-once-on-deep-history harness in
``scripts/surf.py``. It walks the stream forward like a live feed:

  1. The current champion policy trades the next block of bars **live** — those bars
     have not been seen by the policy, so their P&L is genuinely out-of-sample.
  2. Periodically it re-evolves a challenger by genetic search on **only the trailing
     ``lookback`` bars** (bounded memory — old regimes age off the back of the
     window, so the system can't overfit to a decade of history).
  3. The challenger replaces the champion only if it beats it on that recent window
     by ``margin`` (Sharpe-minus-drawdown). No fit is permanent; a champion that
     stops working is dethroned by the next challenger.

Causality holds two ways: the signal features at each bar use only past data, and a
policy is only ever applied to bars *after* the window it was evolved on. We grade
the live (adaptive) champion against the frozen initial champion and buy-and-hold to
show the adaptation is doing something. SIMULATED ONLY.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from threadforge.market.backtest import pnl_from_positions, sharpe, total_return, max_drawdown
from threadforge.market.perception import signal_matrix
from threadforge.market.policy import (
    fitness_score, policy_genes, linear_policy_from_genome, scorecard_from_positions,
)
from threadforge.optimization.genetic import evolve


@dataclass
class EvolveConfig:
    lookback: int = 252          # bounded memory: bars each re-evolution may look back on
    reevolve_every: int = 63     # how often to re-evolve / consider promotion (~quarterly)
    val_frac: float = 0.33       # tail of the lookback held out to validate promotions (anti-overfit)
    pop: int = 16
    gen: int = 10
    dd_penalty: float = 3.0
    margin: float = 0.05         # challenger must beat champion by this (Sharpe-dd) to promote
    weight_range: float = 3.0
    leverage: float = 1.0
    window_size: int = 30
    z_window: int = 60
    fee: float = 0.0001
    slippage: float = 0.0002
    periods: int = 252
    seed: int = 0


@dataclass
class Promotion:
    bar: int                     # global bar index where the swap took effect
    delta: float                 # recent-window fitness improvement


@dataclass
class EvolveResult:
    start: int                   # first live bar (== lookback)
    live_pnl: np.ndarray         # adaptive champion's realized per-step P&L over the live region
    static_pnl: np.ndarray       # frozen initial champion over the same region
    promotions: list[Promotion] = field(default_factory=list)
    n_reevolutions: int = 0

    def scorecards(self, periods: int = 252) -> dict:
        def sc(p):
            return {"sharpe": sharpe(p, periods), "return": total_return(p),
                    "max_drawdown": max_drawdown(p)}
        return {"adaptive": sc(self.live_pnl), "static": sc(self.static_pnl)}


def _evolve_on(state_win, price_win, cfg: EvolveConfig, rng):
    """Genetic search for the best policy genome on one trailing window."""
    genes = policy_genes(cfg.weight_range, cfg.leverage)

    def fitness(genome):
        pol = linear_policy_from_genome(genome, leverage=cfg.leverage)
        sc = scorecard_from_positions(price_win, pol.target(state_win),
                                      fee=cfg.fee, slippage=cfg.slippage, periods=cfg.periods)
        return fitness_score(sc, cfg.dd_penalty)

    best, _, _ = evolve(genes, fitness, pop_size=cfg.pop, generations=cfg.gen, rng=rng)
    return best


def _window_fitness(genome, state_win, price_win, cfg: EvolveConfig) -> float:
    pol = linear_policy_from_genome(genome, leverage=cfg.leverage)
    sc = scorecard_from_positions(price_win, pol.target(state_win),
                                  fee=cfg.fee, slippage=cfg.slippage, periods=cfg.periods)
    return fitness_score(sc, cfg.dd_penalty)


def evolve_live(stream, cfg: EvolveConfig | None = None) -> EvolveResult:
    """Run the bounded-lookback evolving trader over a price stream."""
    cfg = cfg or EvolveConfig()
    rng = random.Random(cfg.seed)

    state, _ = signal_matrix(stream, cfg.window_size, cfg.z_window)   # causal, computed once
    prices = np.asarray([v for _, v in stream], dtype=float)
    n = len(prices)
    L = cfg.lookback
    if n <= L + 2:
        raise ValueError(f"need more than lookback+2={L + 2} bars, got {n}")

    # initial champion: evolve on the first lookback window
    champ = _evolve_on(state[0:L], prices[0:L], cfg, rng)
    static = dict(champ)                                              # frozen baseline

    live_pos = np.zeros(n)
    static_pos = np.zeros(n)
    promotions: list[Promotion] = []
    n_reev = 0

    champ_pol = linear_policy_from_genome(champ, leverage=cfg.leverage)
    static_pol = linear_policy_from_genome(static, leverage=cfg.leverage)

    t = L
    while t < n:
        b = min(t + cfg.reevolve_every, n)
        live_pos[t:b] = champ_pol.target(state[t:b])                 # champion trades the block LIVE (OOS)
        static_pos[t:b] = static_pol.target(state[t:b])

        # re-evolve on the trailing lookback, but split it: evolve on the older part,
        # PROMOTE only on the held-out recent tail. Judging a challenger on the same
        # window it was evolved on is rolling overfitting -- it always "wins" then
        # fails forward. The validation tail is out-of-sample for the challenger.
        lo = max(0, b - L)
        mid = lo + max(1, int((b - lo) * (1.0 - cfg.val_frac)))
        challenger = _evolve_on(state[lo:mid], prices[lo:mid], cfg, rng)
        n_reev += 1
        champ_fit = _window_fitness(champ, state[mid:b], prices[mid:b], cfg)
        chal_fit = _window_fitness(challenger, state[mid:b], prices[mid:b], cfg)
        if chal_fit - champ_fit >= cfg.margin:
            champ = challenger
            champ_pol = linear_policy_from_genome(champ, leverage=cfg.leverage)
            promotions.append(Promotion(b, chal_fit - champ_fit))
        t = b

    live = pnl_from_positions(prices[L:n], live_pos[L:n], fee=cfg.fee, slippage=cfg.slippage)
    stat = pnl_from_positions(prices[L:n], static_pos[L:n], fee=cfg.fee, slippage=cfg.slippage)
    return EvolveResult(L, live, stat, promotions, n_reev)
