"""Entropy variants: coarse and fine binning for different disorder scales.

The base Entropy signal uses 8 bins. These variants expose finer and coarser
granularity as separate signals so the engine can run both simultaneously.
A coarse-grained entropy catches large structural shifts; a fine-grained one
catches subtle distributional changes within an otherwise stable range.
"""

from threadforge.signals.entropy import Entropy


class EntropyFine(Entropy):
    """Entropy with 16 bins — sensitive to subtle distributional shifts."""
    def __init__(self, window_size: int):
        super().__init__(window_size, bins=16)


class EntropyCoarse(Entropy):
    """Entropy with 4 bins — sensitive to large structural regime changes."""
    def __init__(self, window_size: int):
        super().__init__(window_size, bins=4)
