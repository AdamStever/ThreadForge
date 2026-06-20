"""Trend overlay — use the *signed* momentum sensor as a regime, continuously.

The anomaly path collapses every signal to ``abs(z)``, discarding direction. But
momentum is a signed sensor: negative = downtrend. This overlay uses that sign
directly to decide the regime every bar (not only on rare anomaly spikes):

  - momentum convincingly **up**   -> long
  - momentum convincingly **down** -> flat (de-risk) or short
  - inside a deadband             -> hold the current position (limits whipsaw)

"Convincingly" is measured in sigma: momentum is standardised by its own trailing
std (causal), so the deadband is scale-free across instruments and timeframes.
This is the "sell while it's tanking, re-enter when it turns up" idea, built from a
sensor we already have. Causal by construction; SIMULATED ONLY.
"""

from __future__ import annotations

import numpy as np

from threadforge.market.backtest import pnl_from_positions, sharpe, total_return, max_drawdown
from threadforge.market.perception import rolling_z, signal_series


def trend_positions(momentum, *, deadband: float = 0.25, allow_short: bool = False,
                    size: float = 1.0, z_window: int = 60) -> np.ndarray:
    """Continuous regime position from a signed trend sensor. Causal: pos[t] uses
    only information up to t. Holds through the deadband to cut whipsaw."""
    z = rolling_z(momentum, z_window)
    pos = np.zeros(len(z))
    cur = 0.0
    for t in range(len(z)):
        if z[t] > deadband:
            cur = size
        elif z[t] < -deadband:
            cur = -size if allow_short else 0.0
        pos[t] = cur                      # else: hold previous regime
    return pos


def evaluate_trend(stream, *, name="momentum", window_size=30, deadband=0.25,
                   allow_short=False, size=1.0, z_window=60,
                   fee=0.0001, slippage=0.0002, periods=252) -> dict:
    """Scorecard for the trend overlay over a price stream."""
    prices = np.asarray([v for _, v in stream], dtype=float)
    mom = signal_series(stream, name, window_size)
    pos = trend_positions(mom, deadband=deadband, allow_short=allow_short,
                          size=size, z_window=z_window)
    pnl = pnl_from_positions(prices, pos, fee=fee, slippage=slippage)
    trades = int(np.count_nonzero(np.diff(np.concatenate([[0.0], pos]))))
    exposure = float(np.mean(np.abs(pos) > 0)) if len(pos) else 0.0
    return {"sharpe": sharpe(pnl, periods), "return": total_return(pnl),
            "max_drawdown": max_drawdown(pnl), "trades": trades, "exposure": exposure}
