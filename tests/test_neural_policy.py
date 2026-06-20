"""Tests for the neural surfing policy (torch). Skipped if torch is unavailable."""

import numpy as np
import pytest

pytest.importorskip("torch")

from threadforge.market.synthetic import generate_prices
from threadforge.market.neural_policy import train_neural_policy, neural_trader
from threadforge.presets import default_signal_names


def test_neural_policy_target_bounded_and_causal():
    stream = generate_prices(400, seed=1)
    policy = train_neural_policy(stream, epochs=20, device="cpu", seed=0)
    state = np.random.default_rng(0).normal(size=(30, len(default_signal_names())))
    pos = policy.target(state)
    assert pos.shape == (30,)
    assert np.all(np.abs(pos) <= 1.0 + 1e-5)        # leverage 1 via tanh


def test_neural_trader_scorecard():
    stream = generate_prices(600, seed=2)
    trader = neural_trader(stream, epochs=30, device="cpu", seed=0)
    res = trader.evaluate(stream)
    assert set(res) >= {"sharpe", "return", "max_drawdown", "exposure", "turnover"}
    assert all(np.isfinite(v) for v in res.values())


def test_training_reduces_loss_proxy():
    # a longer-trained policy should reach a better in-sample Sharpe than a barely-trained one
    stream = generate_prices(700, seed=3)
    cold = neural_trader(stream, epochs=2, device="cpu", seed=0).evaluate(stream)["sharpe"]
    warm = neural_trader(stream, epochs=300, device="cpu", seed=0).evaluate(stream)["sharpe"]
    assert warm >= cold
