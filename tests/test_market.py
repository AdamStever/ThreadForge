"""Tests for the market layer (synthetic data, backtest, trading agent)."""

import numpy as np
import pytest

from threadforge.market.synthetic import generate_prices
from threadforge.market.backtest import positions, backtest, sharpe, total_return, max_drawdown
from threadforge.market.agent import TradingAgent, trader_genes, agent_from_genome
from threadforge.presets import default_signal_names


# --- synthetic data ---------------------------------------------------------

def test_generate_prices_shape_and_positive():
    stream = generate_prices(500, seed=0)
    assert len(stream) == 500
    assert all(p > 0 for _, p in stream)
    assert generate_prices(100, seed=1) == generate_prices(100, seed=1)   # deterministic


# --- backtest ---------------------------------------------------------------

def _rising(n=50):
    return [100.0 * 1.001 ** i for i in range(n)]


def test_follow_profits_on_rising_fade_loses():
    prices = _rising()
    scores = [10.0] * len(prices)                       # always act
    pnl_follow = backtest(prices, scores, mode="follow", threshold=3.0, momentum_window=1, cost=0.0)
    pnl_fade = backtest(prices, scores, mode="fade", threshold=3.0, momentum_window=1, cost=0.0)
    assert total_return(pnl_follow) > 0
    assert total_return(pnl_fade) < 0


def test_positions_only_fire_above_threshold():
    prices = _rising(20)
    scores = [0.0] * 10 + [5.0] * 10
    pos = positions(prices, scores, mode="follow", threshold=3.0, size=1.0, momentum_window=1)
    assert all(p == 0.0 for p in pos[:10])              # below threshold -> flat
    assert pos[15] == 1.0                                # firing + rising -> long, size 1


def test_sharpe_and_drawdown():
    assert sharpe([0.0, 0.0, 0.0]) == 0.0
    assert sharpe([0.01, 0.02, 0.01, 0.02]) > 0
    assert max_drawdown([0.1, -0.3, 0.05]) < 0


# --- agent ------------------------------------------------------------------

def test_agent_evaluate_returns_scorecard():
    stream = generate_prices(800, seed=1)
    agent = TradingAgent({n: 1.0 for n in default_signal_names()}, "follow", 3.0, 1.0)
    res = agent.evaluate(stream)
    assert set(res) >= {"sharpe", "return", "max_drawdown"}
    assert all(np.isfinite(v) for v in res.values())


def test_trader_genes_and_genome_mapping():
    assert len(trader_genes()) == len(default_signal_names()) + 3
    g = {n: 1.0 for n in default_signal_names()}
    g.update({"mode": 0.3, "threshold": 4.0, "size": 0.5})
    a = agent_from_genome(g)
    assert a.mode == "fade" and a.threshold == 4.0 and a.size == 0.5
    g["mode"] = 0.7
    assert agent_from_genome(g).mode == "follow"
