"""Latent-state representation: the stream's state at one instant as a vector.

So far each signal is treated as its own independent detector. But the signals
computed at a single time step can also be read together as the *coordinates of
one point* in a d-dimensional feature space — d being the number of signals.
That point is the system's "state" at that instant.

  state(t) = [ volatility(t), zscore(t), entropy(t), ... ]  ∈  ℝ^d

This is the raw, hand-built state. The Term-7 encoder will later *learn* a
compressed latent state from these same inputs; this module builds the explicit
feature-space vector that such a model consumes, and that the feature store
already records one signal-column at a time.

WHY A FIXED ORDER MATTERS
  A vector is only meaningful if dimension i always means the same signal.
  `StateVector` is constructed with an ordered list of signal names — that order
  is the schema. Every vector it produces uses exactly that layout, so vectors
  from different steps (or different runs) are directly comparable and stackable
  into a matrix for analysis or model training.

THE LINEAR ALGEBRA
  - `vector()` assembles a point in ℝ^d.
  - `standardize()` is an affine map x -> (x - center) / scale applied per axis,
    putting signals of wildly different magnitudes (volatility ~5, hilbert ~25,
    autocorrelation ~0) onto a common scale so distances are meaningful.
  - `euclidean_distance()` is the L2 norm of the difference of two states — how
    far apart two instants are in feature space.
"""

import math

import numpy as np


class StateVector:
    def __init__(self, signal_names: list[str], fill: float = math.nan):
        """
        Args:
            signal_names: ordered signal names; this fixed order is the schema
                that every produced vector follows.
            fill: value used for a signal that is None (still in warm-up).
                Defaults to NaN so missing data is explicit rather than silently
                read as zero.
        """
        if not signal_names:
            raise ValueError("signal_names must be non-empty")
        self.signal_names = list(signal_names)
        self.fill = fill

    @property
    def dim(self) -> int:
        """Dimensionality d of the feature space."""
        return len(self.signal_names)

    def vector(self, outputs: dict[str, float | None]) -> np.ndarray:
        """Assemble one state vector from a SignalEngine output dict.

        Missing or None signals become `fill`. Order follows the schema, not the
        dict's insertion order.
        """
        return np.array(
            [self._coord(outputs.get(name)) for name in self.signal_names],
            dtype=float,
        )

    def _coord(self, value: float | None) -> float:
        return self.fill if value is None else float(value)

    def is_ready(self, outputs: dict[str, float | None]) -> bool:
        """True once every signal in the schema has a real (non-None) value.

        Lets callers skip warm-up steps, where the state is only partially
        defined, before treating a vector as a complete feature point.
        """
        return all(outputs.get(name) is not None for name in self.signal_names)


def standardize(
    vector: np.ndarray, centers: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """Per-axis affine map (vector - centers) / scales.

    Axes with zero scale (a signal that never varied during calibration) are
    left at 0 rather than dividing by zero — a constant feature carries no
    information about how far an instant is from normal.
    """
    centers = np.asarray(centers, dtype=float)
    scales = np.asarray(scales, dtype=float)
    out = vector - centers
    nonzero = scales != 0.0
    out[nonzero] = out[nonzero] / scales[nonzero]
    out[~nonzero] = 0.0
    return out


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between two state vectors — how far apart two instants are."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
