"""Synthetic market price series — random walk with volatility regimes + jumps.

A stand-in to build and validate the agent/backtest/evolution pipeline before
plugging in real data. It has the features that make markets interesting (and
that the 10 signals should pick up): drifting random-walk prices, **volatility
clustering** (calm and turbulent regimes), and occasional **jumps** (the
anomalies). Returns the standard ``[(timestamp, value)]`` stream.
"""

from __future__ import annotations

import numpy as np


def generate_prices(
    n: int = 4000,
    *,
    seed: int = 0,
    mu: float = 0.0002,
    base_vol: float = 0.01,
    start: float = 100.0,
    regimes: bool = True,
    jumps: bool = True,
) -> list[tuple[str, float]]:
    """Generate a synthetic price series as ``[(timestamp, price)]``."""
    rng = np.random.RandomState(seed)

    vol = np.full(n, base_vol)
    if regimes:
        t = 0
        while t < n:
            dur = rng.randint(120, 400)
            vol[t:t + dur] = base_vol * rng.choice([0.5, 1.0, 1.0, 2.0, 3.0])
            t += dur

    rets = rng.normal(mu, 1.0, n) * vol
    if jumps:
        for _ in range(max(1, n // 500)):
            i = rng.randint(n)
            rets[i] += rng.choice([-1.0, 1.0]) * base_vol * rng.uniform(8.0, 16.0)

    price = start * np.exp(np.cumsum(rets))
    return [(str(i), float(p)) for i, p in enumerate(price)]
