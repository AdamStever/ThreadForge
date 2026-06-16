"""PyTorch encoder + score network — the deep model over raw value windows.

This is the capacity increase the metrics have been asking for: rather than a
linear model on 10 hand-crafted signals, a small neural net learns its own
features from the raw window.

  encoder:  raw window (window_size) -> hidden -> latent state
  score:    latent state -> one anomaly logit

The latent state is the learned counterpart of the hand-built StateVector, and is
the representation a temporal model (LSTM / transformer) will later consume.

Imported on demand (not from `models/__init__`), so the rest of the package
stays free of the torch dependency.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class EncoderScorer(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )
        self.score = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(self.encoder(x)).squeeze(-1)  # raw logits

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    latent_dim: int = 16,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
) -> EncoderScorer:
    """Train the encoder/score net with class-weighted logistic loss."""
    torch.manual_seed(seed)
    model = EncoderScorer(X.shape[1], latent_dim)

    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    # weight the rare positive class by its inverse frequency
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(yt)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def predict_proba(model: EncoderScorer, X: np.ndarray) -> np.ndarray:
    """Return the positive-class probability for each row (shape (n,))."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()
