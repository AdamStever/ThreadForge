"""Neural surfing policy — the challenger, trained by gradient descent on P&L.

Same contract as ``LinearPolicy``: map the signed signal state vector to a
continuous position in ``[-leverage, +leverage]``. The difference is capacity and
how it learns. A small MLP can represent *interactions* the linear policy can't —
e.g. "follow momentum only when autocorrelation is high, fade it when negative" —
and it is trained by backpropagating a differentiable Sharpe-minus-downside loss
*through the costed backtest itself* (the differentiable-simulator idea). The
position map is pointwise per bar, so causality is preserved automatically.

GPU-optional via :func:`threadforge.models.torch_util.get_device` — trains on the
NVIDIA card when present, CPU otherwise. Torch is imported on demand so the core
package stays torch-free. SIMULATED ONLY.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from threadforge.market.perception import signal_matrix
from threadforge.market.policy import PolicyTrader
from threadforge.models.torch_util import get_device


class _PolicyNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):                       # -> raw score per row
        return self.net(x).squeeze(-1)


class NeuralPolicy:
    """Trained MLP wrapped to expose the same ``target(state)`` interface."""

    def __init__(self, net: _PolicyNet, leverage: float = 1.0, device=None):
        self.net = net
        self.leverage = float(leverage)
        self.device = device or torch.device("cpu")

    def target(self, state: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            pos = self.leverage * torch.tanh(self.net(x))
        return pos.cpu().numpy()


def _pnl_torch(prices, pos, fee, slippage):
    """Differentiable per-step P&L (mirrors backtest.pnl_from_positions)."""
    rets = (prices[1:] - prices[:-1]) / prices[:-1]
    held = pos[:-1]
    prev = torch.cat([pos.new_zeros(1), held[:-1]])
    turnover = torch.abs(held - prev)
    return held * rets - (fee + slippage) * turnover


def train_neural_policy(stream, *, window_size: int = 30, z_window: int = 60,
                        hidden: int = 16, epochs: int = 400, lr: float = 0.01,
                        risk_penalty: float = 1.0, leverage: float = 1.0,
                        fee: float = 0.0001, slippage: float = 0.0002, periods: int = 252,
                        device=None, seed: int = 0) -> NeuralPolicy:
    """Fit a neural policy on one price stream; returns a ready-to-evaluate policy.

    Loss = -(annualized Sharpe) + ``risk_penalty`` * (annualized downside deviation),
    a smooth, differentiable stand-in for "high Sharpe, shallow drawdowns".
    """
    device = device or get_device()
    torch.manual_seed(seed)

    state_np, names = signal_matrix(stream, window_size, z_window)
    prices_np = np.asarray([v for _, v in stream], dtype=float)
    state = torch.as_tensor(state_np, dtype=torch.float32, device=device)
    prices = torch.as_tensor(prices_np, dtype=torch.float32, device=device)
    ann = float(np.sqrt(periods))

    net = _PolicyNet(state.shape[1], hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        pos = leverage * torch.tanh(net(state))
        pnl = _pnl_torch(prices, pos, fee, slippage)
        mean, std = pnl.mean(), pnl.std()
        sharpe = mean / (std + 1e-8) * ann
        downside = torch.sqrt(torch.mean(torch.clamp(-pnl, min=0.0) ** 2) + 1e-12) * ann
        loss = -sharpe + risk_penalty * downside
        loss.backward()
        opt.step()

    return NeuralPolicy(net, leverage=leverage, device=device)


def neural_trader(stream, *, window_size: int = 30, z_window: int = 60,
                  fee: float = 0.0001, slippage: float = 0.0002, periods: int = 252,
                  **train_kw) -> PolicyTrader:
    """Train a neural policy and wrap it in a PolicyTrader for identical grading."""
    policy = train_neural_policy(stream, window_size=window_size, z_window=z_window,
                                 fee=fee, slippage=slippage, periods=periods, **train_kw)
    return PolicyTrader(policy, window_size=window_size, z_window=z_window,
                        fee=fee, slippage=slippage, periods=periods)
