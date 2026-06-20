"""Volatility-targeted position sizing — weak/turbulent regimes get smaller bets.

The policy outputs a *direction + conviction* in [-1, 1]; on its own, `tanh`
saturates to full size on weak signals, so a slightly-wrong policy takes a big
persistent bet and compounds into a deep drawdown (we saw -76% on EURUSD). Volatility
targeting fixes the *magnitude*: scale the position so the strategy runs at a fixed
risk budget. When the asset is calm a unit of conviction is worth a full position;
when it is turbulent the same conviction is sized down.

    scale[t] = target_per_bar_vol / trailing_realized_vol[t]
    position[t] = clip(raw[t] * scale[t], -max_leverage, +max_leverage)

The scale depends only on prices (not the policy), is causal (trailing returns
only), and so can be computed once and reused for evolution and live trading alike.
SIMULATED ONLY.
"""

from __future__ import annotations

import numpy as np


def realized_vol(prices, window: int = 20) -> np.ndarray:
    """Causal trailing std of simple returns, per bar (0 until enough history)."""
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    rets = np.zeros(n)
    rets[1:] = np.diff(prices) / prices[:-1]
    vol = np.zeros(n)
    min_count = max(5, window // 2)
    for t in range(n):
        lo = max(1, t - window + 1)           # rets[0] is a synthetic 0; skip it
        w = rets[lo:t + 1]
        if len(w) >= min_count:
            vol[t] = w.std()
    return vol


def vol_target_scale(prices, target_vol: float, *, vol_window: int = 20,
                     periods: int = 252) -> np.ndarray:
    """Per-bar position multiplier that targets ``target_vol`` (annualized)."""
    rv = realized_vol(prices, vol_window)
    per_bar_target = target_vol / np.sqrt(periods)
    scale = np.zeros(len(rv))
    nz = rv > 1e-9
    scale[nz] = per_bar_target / rv[nz]
    return scale


def apply_sizing(raw, scale, max_leverage: float = 1.0) -> np.ndarray:
    """Combine a raw [-1,1] target with a vol-target scale, capped at max_leverage."""
    return np.clip(np.asarray(raw, dtype=float) * np.asarray(scale, dtype=float),
                   -max_leverage, max_leverage)
