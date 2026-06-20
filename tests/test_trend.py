"""Tests for the signed-momentum trend overlay."""

import numpy as np

from threadforge.market.synthetic import generate_prices
from threadforge.market.trend import signal_series, rolling_z, trend_positions, evaluate_trend


def test_rolling_z_is_causal_and_standardised():
    vals = list(range(100))
    z = rolling_z(vals, window=20)
    assert len(z) == 100
    assert z[0] == 0.0                       # not enough history -> 0
    assert z[-1] > 0                          # rising series -> last point above its trailing mean


def test_trend_goes_flat_in_downtrend_long_flat():
    # rise then fall; long/flat overlay should be long early, flat (0) once it's falling
    up = [100 + i for i in range(60)]
    down = [up[-1] - i for i in range(1, 61)]
    mom = signal_series([(str(i), p) for i, p in enumerate(up + down)])
    pos = trend_positions(mom, deadband=0.25, allow_short=False, size=1.0, z_window=20)
    assert pos.max() > 0                       # took a long in the uptrend
    assert pos.min() >= 0.0                     # never short in long/flat mode
    assert pos[-1] == 0.0                       # de-risked by the end of the downtrend


def test_trend_can_short_when_allowed():
    down = [200 - i for i in range(120)]
    mom = signal_series([(str(i), p) for i, p in enumerate(down)])
    pos = trend_positions(mom, deadband=0.25, allow_short=True, size=1.0, z_window=20)
    assert pos.min() < 0.0                      # shorts the sustained downtrend


def test_evaluate_trend_scorecard():
    stream = generate_prices(800, seed=3)
    res = evaluate_trend(stream, deadband=0.25)
    assert set(res) >= {"sharpe", "return", "max_drawdown", "trades", "exposure"}
    assert 0.0 <= res["exposure"] <= 1.0
    assert all(np.isfinite(v) for v in res.values())
