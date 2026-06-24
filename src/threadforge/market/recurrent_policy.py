"""Recurrent (LSTM) trading policy — the neural brain, now with memory.

The MLP policy (`neural_policy.py`) is a *reflex*: it decides from this bar's signal
snapshot alone, with no sense of the trajectory that led here. This policy is an LSTM
that consumes the **sequence** of recent state vectors and carries a hidden memory,
so its position at bar t can depend on the order and shape of what came before (e.g.
"calm *after* a crash" vs "calm *after* a long grind up") — not just the instant.

Causal by construction: an LSTM reads left-to-right, so its output at t depends only
on inputs up to t. The features are already causal, so the whole policy is. Trained
the same way as the MLP — gradient descent through the costed backtest (differentiable
simulator) on a Sharpe-minus-downside loss — plus weight decay and dropout, because a
larger net overfits short, noisy market data fast. GPU-optional. SIMULATED ONLY.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from threadforge.market.perception import signal_matrix
from threadforge.market.policy import PolicyTrader
from threadforge.market.neural_policy import _pnl_torch
from threadforge.models.torch_util import get_device


class _RecurrentPolicyNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, seq):                       # seq: (T, D) one sequence
        out, _ = self.lstm(seq.unsqueeze(0))      # (1, T, H)
        return self.head(self.drop(out)).squeeze(0).squeeze(-1)   # (T,) raw score per bar


class RecurrentPolicy:
    """Trained LSTM wrapped to expose the same ``target(state)`` interface."""

    def __init__(self, net: _RecurrentPolicyNet, leverage: float = 1.0, device=None):
        self.net = net
        self.leverage = float(leverage)
        self.device = device or torch.device("cpu")

    def target(self, state: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            pos = self.leverage * torch.tanh(self.net(x))
        return pos.cpu().numpy()


def train_recurrent_policy(stream, *, window_size: int = 30, z_window: int = 60,
                           hidden: int = 32, layers: int = 1, dropout: float = 0.1,
                           epochs: int = 300, lr: float = 0.01, weight_decay: float = 1e-4,
                           risk_penalty: float = 1.0, leverage: float = 1.0,
                           fee: float = 0.0001, slippage: float = 0.0002, periods: int = 252,
                           device=None, seed: int = 0) -> RecurrentPolicy:
    """Fit an LSTM policy on one price stream; returns a ready-to-evaluate policy."""
    device = device or get_device()
    torch.manual_seed(seed)

    state_np, _ = signal_matrix(stream, window_size, z_window)
    prices_np = np.asarray([v for _, v in stream], dtype=float)
    state = torch.as_tensor(state_np, dtype=torch.float32, device=device)
    prices = torch.as_tensor(prices_np, dtype=torch.float32, device=device)
    ann = float(np.sqrt(periods))

    net = _RecurrentPolicyNet(state.shape[1], hidden, layers, dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

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

    return RecurrentPolicy(net, leverage=leverage, device=device)


def recurrent_trader(stream, *, window_size: int = 30, z_window: int = 60,
                     fee: float = 0.0001, slippage: float = 0.0002, periods: int = 252,
                     target_vol: float | None = None, vol_window: int = 20,
                     max_leverage: float = 1.0, **train_kw) -> PolicyTrader:
    """Train an LSTM policy and wrap it in a PolicyTrader for identical grading."""
    policy = train_recurrent_policy(stream, window_size=window_size, z_window=z_window,
                                    fee=fee, slippage=slippage, periods=periods, **train_kw)
    return PolicyTrader(policy, window_size=window_size, z_window=z_window,
                        fee=fee, slippage=slippage, periods=periods,
                        target_vol=target_vol, vol_window=vol_window, max_leverage=max_leverage)
