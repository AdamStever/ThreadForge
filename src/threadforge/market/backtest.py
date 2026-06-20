"""Paper-trading backtest — turn anomaly scores into positions and grade by P&L.

Causal by construction: the position held over ``[t, t+1)`` is decided only from
information up to ``t`` (the score and the trailing price move), so there is no
look-ahead. The trade rule is intentionally simple — an agent acts only when its
anomaly score crosses a threshold, betting *with* the recent move (``follow`` /
momentum) or *against* it (``fade`` / mean-reversion), sized by ``size``.
Transaction cost is charged on position changes.

Fitness is risk-adjusted (Sharpe); total return and max drawdown are reported
alongside. SIMULATED ONLY.
"""

from __future__ import annotations

import numpy as np

MODES = ("fade", "follow")


def positions(prices, scores, *, mode="follow", threshold=3.0, size=1.0,
              momentum_window=5) -> np.ndarray:
    """Causal position series: act when score >= threshold, direction by recent move."""
    prices = np.asarray(prices, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = len(prices)
    pos = np.zeros(n)
    for t in range(n):
        if scores[t] < threshold:
            continue
        if t >= momentum_window and prices[t - momentum_window] > 0:
            recent = (prices[t] - prices[t - momentum_window]) / prices[t - momentum_window]
        else:
            recent = 0.0
        d = np.sign(recent)
        if mode == "fade":
            d = -d
        pos[t] = d * size
    return pos


def pnl_from_positions(prices, pos, *, fee=0.0001, slippage=0.0002) -> np.ndarray:
    """Per-step net P&L (returns) for an arbitrary causal position series.

    Costs are charged on **turnover** (the change in position): each unit of
    position bought or sold pays ``fee`` (commission) + ``slippage`` (execution
    worse than the observed price). Defaults are ~3 bps one-way (a conservative
    figure for a liquid ETF like SPY); set both to 0 for gross P&L. ``pos[t]`` is
    decided at ``t`` and held over ``[t, t+1)`` — causal, no look-ahead.
    """
    prices = np.asarray(prices, dtype=float)
    pos = np.asarray(pos, dtype=float)
    if len(prices) < 2:
        return np.zeros(0)
    rets = np.diff(prices) / prices[:-1]            # rets[t] = return t -> t+1
    held = pos[:-1]                                  # position held over [t, t+1)
    prev = np.concatenate([[0.0], held[:-1]])
    turnover = np.abs(held - prev)
    return held * rets - (fee + slippage) * turnover


def backtest(prices, scores, *, mode="follow", threshold=3.0, size=1.0,
             momentum_window=5, fee=0.0001, slippage=0.0002) -> np.ndarray:
    """Per-step net P&L of the anomaly-gated trade rule, after fees and slippage."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        return np.zeros(0)
    pos = positions(prices, scores, mode=mode, threshold=threshold, size=size,
                    momentum_window=momentum_window)
    return pnl_from_positions(prices, pos, fee=fee, slippage=slippage)


def sharpe(pnl, periods: int = 252) -> float:
    """Annualised Sharpe ratio of a per-step P&L series (0 if flat)."""
    pnl = np.asarray(pnl, dtype=float)
    if len(pnl) == 0 or pnl.std() == 0.0:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(periods))


def total_return(pnl) -> float:
    pnl = np.asarray(pnl, dtype=float)
    return float(np.prod(1.0 + pnl) - 1.0) if len(pnl) else 0.0


def max_drawdown(pnl) -> float:
    """Most negative peak-to-trough of the equity curve (<= 0)."""
    pnl = np.asarray(pnl, dtype=float)
    if len(pnl) == 0:
        return 0.0
    equity = np.cumprod(1.0 + pnl)
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min())
