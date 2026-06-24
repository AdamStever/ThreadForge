"""Flip-flop realization for a leveraged long/short pair (e.g. TQQQ / SQQQ).

A directional signal is expressed *long-only* by holding the 3x-long ETF (TQQQ)
when bullish and switching to the 3x-inverse ETF (SQQQ) when bearish — no shorting,
loss capped per leg. Because we settle P&L on the **actual** ETF return series, the
products' volatility decay and the cost of switching legs are captured automatically
(no need to model decay).

    pos[t] > 0  ->  hold the long ETF, size  pos[t]
    pos[t] < 0  ->  hold the short ETF, size |pos[t]|

Causal (position at t decided from data up to t) and SIMULATED ONLY. Leverage
amplifies returns and risk equally — this does not create edge, it only changes the
instrument the signal is expressed through.
"""

from __future__ import annotations

import numpy as np


def align_pair(a: list[tuple[str, float]], b: list[tuple[str, float]]):
    """Intersect two (date, price) streams on common dates; return (dates, pa, pb)."""
    da, db = dict(a), dict(b)
    dates = sorted(set(da) & set(db))
    pa = np.asarray([da[d] for d in dates], dtype=float)
    pb = np.asarray([db[d] for d in dates], dtype=float)
    return dates, pa, pb


def flip_flop_pnl(pos, long_prices, short_prices, *, fee=0.0001, slippage=0.0002) -> np.ndarray:
    """Per-step net P&L of expressing ``pos`` via the long/short ETF pair.

    ``pos[t]`` decided at t is held over ``[t, t+1)``. Cost is charged on the
    turnover of *each* leg, so a full TQQQ->SQQQ flip pays two units of cost.
    """
    pos = np.asarray(pos, dtype=float)
    lp = np.asarray(long_prices, dtype=float)
    sp = np.asarray(short_prices, dtype=float)
    if len(lp) < 2:
        return np.zeros(0)
    lr = np.diff(lp) / lp[:-1]
    sr = np.diff(sp) / sp[:-1]
    a_long = np.clip(pos, 0.0, None)
    a_short = np.clip(-pos, 0.0, None)
    hl, hs = a_long[:-1], a_short[:-1]                  # held over [t, t+1)
    prev_l = np.concatenate([[0.0], hl[:-1]])
    prev_s = np.concatenate([[0.0], hs[:-1]])
    cost = (fee + slippage) * (np.abs(hl - prev_l) + np.abs(hs - prev_s))
    return hl * lr + hs * sr - cost
