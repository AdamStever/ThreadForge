"""Tests for the LSTM (recurrent) trading policy. Skipped if torch is unavailable."""

import numpy as np
import pytest

pytest.importorskip("torch")

from threadforge.market.synthetic import generate_prices
from threadforge.market.recurrent_policy import train_recurrent_policy, recurrent_trader
from threadforge.presets import default_signal_names


def test_target_bounded():
    stream = generate_prices(400, seed=1)
    policy = train_recurrent_policy(stream, epochs=10, hidden=8, device="cpu", seed=0)
    state = np.random.default_rng(0).normal(size=(40, len(default_signal_names())))
    pos = policy.target(state)
    assert pos.shape == (40,)
    assert np.all(np.abs(pos) <= 1.0 + 1e-5)


def test_target_is_causal():
    # an LSTM is causal: output at t must not depend on inputs after t
    stream = generate_prices(300, seed=2)
    policy = train_recurrent_policy(stream, epochs=5, hidden=8, device="cpu", seed=0)
    state = np.random.default_rng(1).normal(size=(60, len(default_signal_names())))
    full = policy.target(state)
    prefix = policy.target(state[:30])
    assert np.allclose(full[:30], prefix, atol=1e-5)   # truncating the future leaves the past unchanged


def test_recurrent_trader_scorecard():
    stream = generate_prices(500, seed=3)
    trader = recurrent_trader(stream, epochs=15, hidden=8, device="cpu", seed=0)
    res = trader.evaluate(stream)
    assert set(res) >= {"sharpe", "return", "max_drawdown", "exposure", "turnover"}
    assert all(np.isfinite(v) for v in res.values())


def test_training_improves_in_sample():
    stream = generate_prices(600, seed=4)
    cold = recurrent_trader(stream, epochs=2, hidden=8, device="cpu", seed=0).evaluate(stream)["sharpe"]
    warm = recurrent_trader(stream, epochs=250, hidden=8, device="cpu", seed=0).evaluate(stream)["sharpe"]
    assert warm >= cold
