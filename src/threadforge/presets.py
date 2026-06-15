"""Preset pipeline configurations.

A single source of truth for the canonical signal bank, so the detection
scripts and the modeling layer extract exactly the same features. (The detection
scripts still build their own engine for now; new code should use this.)
"""

from threadforge.engine import SignalEngine
from threadforge.signals import (
    Momentum,
    Volatility,
    Entropy,
    EntropyFine,
    EntropyCoarse,
    ZScore,
    Acceleration,
    Autocorrelation,
    HilbertEnvelope,
    SpectralFlatness,
)


def default_signal_engine(window_size: int) -> SignalEngine:
    """Build the standard 10-signal engine used across the project."""
    engine = SignalEngine()
    engine.register("momentum", Momentum(window_size))
    engine.register("volatility", Volatility(window_size))
    engine.register("entropy", Entropy(window_size))
    engine.register("entropy_fine", EntropyFine(window_size))
    engine.register("entropy_coarse", EntropyCoarse(window_size))
    engine.register("zscore", ZScore(window_size))
    engine.register("acceleration", Acceleration(window_size))
    engine.register("autocorrelation", Autocorrelation(window_size))
    engine.register("hilbert", HilbertEnvelope(window_size))
    engine.register("spectral_flatness", SpectralFlatness(window_size))
    return engine


def default_signal_names(window_size: int = 30) -> list[str]:
    """The signal names in registration order — the feature-vector schema."""
    return list(default_signal_engine(window_size)._signals.keys())
