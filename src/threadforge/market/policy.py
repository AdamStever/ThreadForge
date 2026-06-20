"""Policy — map the signed signal state vector to a continuous position, every bar.

This is the "surfer": instead of an anomaly score that occasionally trips a trade,
the policy holds a continuous target position in ``[-leverage, +leverage]`` each
bar, scaled by conviction. ``LinearPolicy`` is the interpretable core — a weighted
sum of the standardized signals squashed through ``tanh`` — with weights learned by
the genetic search (no hand-tuning). The neural challenger (``neural_policy.py``)
implements the same ``target`` interface, so both are graded identically.

Fitness is **Sharpe minus a drawdown penalty**: the honest edge over buy-and-hold
is risk control, so an agent that earns its Sharpe with shallow drawdowns is
preferred to one that gets there through deep ones. Causal, SIMULATED ONLY.
"""

from __future__ import annotations

import numpy as np

from threadforge.market.backtest import pnl_from_positions, sharpe, total_return, max_drawdown
from threadforge.market.perception import signal_matrix
from threadforge.market.sizing import vol_target_scale, apply_sizing
from threadforge.optimization.genetic import Gene
from threadforge.presets import default_signal_names


class LinearPolicy:
    """position = leverage * tanh(state . weights + bias). Continuous, signed."""

    def __init__(self, weights, bias: float = 0.0, leverage: float = 1.0):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)
        self.leverage = float(leverage)

    def target(self, state: np.ndarray) -> np.ndarray:
        raw = state @ self.weights + self.bias
        return self.leverage * np.tanh(raw)


class PolicyTrader:
    """Perception + policy + costed backtest -> a risk-adjusted scorecard."""

    def __init__(self, policy, *, window_size: int = 30, z_window: int = 60,
                 fee: float = 0.0001, slippage: float = 0.0002, periods: int = 252,
                 target_vol: float | None = None, vol_window: int = 20,
                 max_leverage: float = 1.0):
        self.policy = policy
        self.window_size = window_size
        self.z_window = z_window
        self.fee = fee
        self.slippage = slippage
        self.periods = periods
        self.target_vol = target_vol          # None -> raw policy output (no vol targeting)
        self.vol_window = vol_window
        self.max_leverage = max_leverage

    def positions(self, stream) -> np.ndarray:
        state, _ = signal_matrix(stream, self.window_size, self.z_window)
        raw = self.policy.target(state)
        if self.target_vol is None:
            return np.clip(raw, -self.max_leverage, self.max_leverage)
        prices = np.asarray([v for _, v in stream], dtype=float)
        scale = vol_target_scale(prices, self.target_vol, vol_window=self.vol_window,
                                 periods=self.periods)
        return apply_sizing(raw, scale, self.max_leverage)

    def evaluate(self, stream) -> dict:
        prices = np.asarray([v for _, v in stream], dtype=float)
        pos = self.positions(stream)
        return scorecard_from_positions(prices, pos, fee=self.fee, slippage=self.slippage,
                                        periods=self.periods)


def scorecard_from_positions(prices, pos, *, fee=0.0001, slippage=0.0002, periods=252) -> dict:
    """Risk-adjusted scorecard for a position series over precomputed prices.

    Lets callers (e.g. the online evolving loop) grade a policy on a slice of
    already-computed causal features without rebuilding the stream.
    """
    prices = np.asarray(prices, dtype=float)
    pos = np.asarray(pos, dtype=float)
    pnl = pnl_from_positions(prices, pos, fee=fee, slippage=slippage)
    turnover = float(np.mean(np.abs(np.diff(np.concatenate([[0.0], pos]))))) if len(pos) else 0.0
    return {
        "sharpe": sharpe(pnl, periods),
        "return": total_return(pnl),
        "max_drawdown": max_drawdown(pnl),
        "exposure": float(np.mean(np.abs(pos))) if len(pos) else 0.0,
        "turnover": turnover,
    }


def fitness_score(scorecard: dict, dd_penalty: float = 3.0) -> float:
    """Sharpe minus a penalty on drawdown (max_drawdown is <= 0, so this subtracts)."""
    return scorecard["sharpe"] + dd_penalty * scorecard["max_drawdown"]


def policy_genes(weight_range: float = 3.0, leverage: float = 1.0) -> list[Gene]:
    """GA genome: one signed weight per signal + a bias term."""
    genes = [Gene(f"w_{n}", -weight_range, weight_range) for n in default_signal_names()]
    genes.append(Gene("bias", -1.0, 1.0))
    return genes


def linear_policy_from_genome(genome: dict, *, leverage: float = 1.0) -> LinearPolicy:
    """Build just the LinearPolicy from a GA genome (no trader / feature rebuild)."""
    names = default_signal_names()
    weights = [genome[f"w_{n}"] for n in names]
    return LinearPolicy(weights, bias=genome.get("bias", 0.0), leverage=leverage)


def linear_from_genome(genome: dict, *, leverage: float = 1.0, **trader_kw) -> PolicyTrader:
    return PolicyTrader(linear_policy_from_genome(genome, leverage=leverage), **trader_kw)
