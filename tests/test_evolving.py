"""Tests for the online evolving trader (bounded-lookback, live promotion)."""

import numpy as np
import pytest

from threadforge.market.synthetic import generate_prices
from threadforge.market.evolving import EvolveConfig, evolve_live


def _cfg(**kw):
    base = dict(lookback=200, reevolve_every=100, pop=8, gen=4, seed=0)
    base.update(kw)
    return EvolveConfig(**base)


def test_live_region_and_alignment():
    stream = generate_prices(900, seed=1)
    res = evolve_live(stream, _cfg())
    assert res.start == 200                              # live region starts after the first lookback
    # one P&L step per bar in the live region (minus the final return boundary)
    assert len(res.live_pnl) == len(stream) - res.start - 1
    assert len(res.static_pnl) == len(res.live_pnl)
    assert res.n_reevolutions >= 1


def test_is_deterministic_with_seed():
    stream = generate_prices(800, seed=2)
    a = evolve_live(stream, _cfg())
    b = evolve_live(stream, _cfg())
    assert np.allclose(a.live_pnl, b.live_pnl)
    assert len(a.promotions) == len(b.promotions)


def test_high_margin_blocks_all_promotions():
    # an unreachable promotion margin means the champion is never replaced -> adaptive == static
    stream = generate_prices(800, seed=3)
    res = evolve_live(stream, _cfg(margin=1e9))
    assert res.promotions == []
    assert np.allclose(res.live_pnl, res.static_pnl)


def test_too_short_stream_raises():
    with pytest.raises(ValueError):
        evolve_live(generate_prices(150, seed=0), _cfg(lookback=200))
