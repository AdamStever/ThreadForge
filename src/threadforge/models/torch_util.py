"""Device selection for the torch models — GPU when present, CPU otherwise.

The whole point of this module: the same model code trains and runs on an NVIDIA
GPU when one is visible and falls back to CPU transparently. GPU-optional, never
GPU-required. Kept separate (imported on demand) so the core package stays
torch-free.
"""

from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the torch device to use.

    ``prefer`` forces a device ("cuda" / "cpu") when given; otherwise picks CUDA
    if it is available and falls back to CPU.
    """
    if prefer is not None:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device | None = None) -> str:
    """Human-readable device label, e.g. 'cuda (NVIDIA GeForce RTX 4060)' or 'cpu'."""
    device = device or get_device()
    if device.type == "cuda" and torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu"
