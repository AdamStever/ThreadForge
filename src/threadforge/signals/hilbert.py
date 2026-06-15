"""HilbertEnvelope: instantaneous oscillation amplitude via the Hilbert transform.

This is the project's first signal built on numpy. Where Volatility measures the
*spread* of a window and Autocorrelation measures its *structure*, the Hilbert
envelope measures the *amplitude of oscillation* — how strong the up-and-down
swings are around the local average, right now.

THE ANALYTIC SIGNAL, INTUITIVELY
  Any real oscillation x(t) can be paired with a 90-degrees-shifted copy of
  itself, H[x](t), to form a complex "analytic signal" z(t) = x(t) + i*H[x](t).
  Picture x(t) as the shadow of a point spinning around a circle: the radius of
  that circle is the *envelope* |z(t)|, the instantaneous amplitude. For a pure
  sine of amplitude A the envelope is flat at A, even as the sine itself swings
  between -A and +A. So the envelope strips away the wiggle and leaves the
  "loudness" of the wiggle — which is exactly what an amplitude-burst anomaly
  looks like.

HOW IT IS COMPUTED (the standard FFT method, same as scipy.signal.hilbert)
  1. demean the window so the result reflects oscillation, not the DC level
  2. FFT the window into the frequency domain
  3. zero the negative frequencies and double the positive ones (this is what
     "shift by 90 degrees and add" becomes in frequency space)
  4. inverse-FFT to get the complex analytic signal; |.| is the envelope
  We report the envelope at the most recent sample — the amplitude "now".

CAUSALITY
  The transform runs over the rolling window, which by construction holds only
  past and current values (see signals/base.py). No future data enters, so the
  causal guarantee is preserved. The transform is non-causal *within* the window
  (it uses the whole window at once, like Volatility and Entropy do) and has
  edge effects at the boundary — which is one reason this is wired as a logged
  feature for the future ML layer rather than a primary detector.
"""

import numpy as np

from threadforge.signals.base import Signal


def _analytic(x: np.ndarray) -> np.ndarray:
    """Return the analytic signal of a real 1-D array (scipy.signal.hilbert)."""
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(X * h)


class HilbertEnvelope(Signal):
    def compute(self, window: list[float]) -> float:
        arr = np.asarray(window, dtype=float)
        centered = arr - arr.mean()  # remove DC so envelope reflects oscillation
        envelope = np.abs(_analytic(centered))
        return float(envelope[-1])  # instantaneous amplitude at the latest sample
