"""Tests for the leveraged flip-flop realization."""

import numpy as np

from threadforge.market.leveraged import align_pair, flip_flop_pnl


def test_align_pair_intersects_dates():
    a = [("d1", 1.0), ("d2", 2.0), ("d3", 3.0)]
    b = [("d2", 20.0), ("d3", 30.0), ("d4", 40.0)]
    dates, pa, pb = align_pair(a, b)
    assert dates == ["d2", "d3"]
    assert list(pa) == [2.0, 3.0] and list(pb) == [20.0, 30.0]


def test_constant_long_earns_long_returns():
    lp = np.array([100.0, 110.0, 121.0])      # +10% each step
    sp = np.array([100.0, 97.0, 94.0])
    pos = np.array([1.0, 1.0, 1.0])           # always long the long ETF
    pnl = flip_flop_pnl(pos, lp, sp, fee=0.0, slippage=0.0)
    assert np.allclose(pnl, [0.10, 0.10])     # earns the long ETF's returns


def test_constant_short_earns_short_etf_returns():
    lp = np.array([100.0, 110.0, 121.0])
    sp = np.array([100.0, 90.0, 81.0])        # short ETF -10% each step
    pos = np.array([-1.0, -1.0, -1.0])        # always hold the inverse ETF
    pnl = flip_flop_pnl(pos, lp, sp, fee=0.0, slippage=0.0)
    assert np.allclose(pnl, [-0.10, -0.10])   # earns the inverse ETF's (negative) returns


def test_flipping_legs_costs_two_units():
    lp = np.array([100.0, 100.0, 100.0])      # flat -> isolate cost
    sp = np.array([100.0, 100.0, 100.0])
    pos = np.array([1.0, -1.0, -1.0])         # full flip long->short between bar 0 and 1
    pnl = flip_flop_pnl(pos, lp, sp, fee=0.01, slippage=0.0)
    # held over [1,2): long leg 1->0 (1 unit) + short leg 0->1 (1 unit) = 2 units * 0.01
    assert np.isclose(pnl[1], -0.02)
