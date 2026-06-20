"""Tests for the linear surfing policy."""

import numpy as np

from threadforge.market.synthetic import generate_prices
from threadforge.market.policy import (
    LinearPolicy, PolicyTrader, fitness_score, policy_genes, linear_from_genome,
)
from threadforge.presets import default_signal_names


def test_linear_policy_bounded_by_leverage():
    rng = np.random.default_rng(0)
    state = rng.normal(size=(50, len(default_signal_names())))
    pol = LinearPolicy(rng.normal(size=len(default_signal_names())), bias=0.5, leverage=2.0)
    pos = pol.target(state)
    assert pos.shape == (50,)
    assert np.all(np.abs(pos) <= 2.0 + 1e-9)


def test_policy_trader_scorecard():
    stream = generate_prices(800, seed=1)
    pol = LinearPolicy([1.0] + [0.0] * (len(default_signal_names()) - 1))  # momentum only
    res = PolicyTrader(pol).evaluate(stream)
    assert set(res) >= {"sharpe", "return", "max_drawdown", "exposure", "turnover"}
    assert all(np.isfinite(v) for v in res.values())
    assert 0.0 <= res["exposure"] <= 1.0


def test_fitness_penalizes_drawdown():
    shallow = {"sharpe": 1.0, "max_drawdown": -0.05}
    deep = {"sharpe": 1.0, "max_drawdown": -0.40}
    assert fitness_score(shallow) > fitness_score(deep)   # same Sharpe, shallower dd wins


def test_genome_roundtrip():
    genes = policy_genes()
    assert len(genes) == len(default_signal_names()) + 1   # weights + bias
    genome = {g.name: 0.0 for g in genes}
    genome["w_momentum"] = 1.0
    trader = linear_from_genome(genome)
    stream = generate_prices(400, seed=2)
    assert len(trader.positions(stream)) == len(stream)    # causal, one position per bar
