"""Signal layer: causal rolling-window features computed from a data stream."""

from threadforge.signals.base import Signal
from threadforge.signals.momentum import Momentum
from threadforge.signals.volatility import Volatility
from threadforge.signals.entropy import Entropy
from threadforge.signals.entropy_variants import EntropyFine, EntropyCoarse
from threadforge.signals.sharpness import Sharpness
from threadforge.signals.acceleration import Acceleration
from threadforge.signals.zscore import ZScore
from threadforge.signals.autocorrelation import Autocorrelation

__all__ = [
    "Signal",
    "Momentum", "Volatility", "Entropy", "EntropyFine", "EntropyCoarse",
    "Sharpness", "Acceleration", "ZScore", "Autocorrelation",
]
