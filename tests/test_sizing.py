"""Tests for volatility-targeted position sizing."""

import numpy as np

from threadforge.market.sizing import realized_vol, vol_target_scale, apply_sizing


def test_realized_vol_causal_and_nonneg():
    rng = np.random.default_rng(0)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, size=200))
    rv = realized_vol(prices, window=20)
    assert len(rv) == 200
    assert rv[0] == 0.0                 # no history yet
    assert np.all(rv >= 0.0)
    assert rv[-1] > 0.0


def test_scale_is_inverse_to_volatility():
    # a calm (low-vol) stretch then a turbulent (high-vol) stretch -> calm sized larger
    rng = np.random.default_rng(0)
    calm = 100 * np.cumprod(1 + rng.normal(0, 0.002, size=80))
    turbulent = calm[-1] * np.cumprod(1 + rng.normal(0, 0.03, size=80))
    prices = np.concatenate([calm, turbulent])
    scale = vol_target_scale(prices, target_vol=0.10, vol_window=20)
    assert scale[70] > scale[-1]        # calm regime sized larger than turbulent regime


def test_apply_sizing_respects_leverage():
    raw = np.array([1.0, -1.0, 0.5])
    scale = np.array([5.0, 5.0, 0.1])   # would blow past leverage without the clip
    pos = apply_sizing(raw, scale, max_leverage=1.0)
    assert np.all(np.abs(pos) <= 1.0 + 1e-9)
    assert pos[2] == 0.05               # within cap -> passes through (0.5 * 0.1)
