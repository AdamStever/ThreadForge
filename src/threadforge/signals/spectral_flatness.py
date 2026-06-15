"""SpectralFlatness: is this window tone-like or noise-like, in the frequency domain?

The previous signals work in the time domain (values, spreads, slopes) or on the
analytic envelope. Spectral flatness looks at the *shape of the frequency
spectrum* — how the window's energy is distributed across frequencies.

  near 1   energy spread evenly across all frequencies  -> white-noise-like
  near 0   energy concentrated in a few frequencies      -> tonal / periodic

WHY THIS IS A GENUINELY NEW VIEW
  A steady hum, a daily cycle, a polling loop — these put almost all their
  energy at one frequency, so their spectrum is "peaky" and flatness is low.
  Broadband noise spreads energy everywhere, so flatness is high. An anomaly
  that turns a clean periodic signal into noise (or vice-versa) moves flatness
  even when the amplitude and variance look unchanged. None of the other
  signals measure this frequency-domain property.

HOW IT IS COMPUTED (the standard Wiener-entropy / spectral-flatness measure)
  1. demean the window (drop the DC level, which carries no oscillation info)
  2. take the power spectrum via the real FFT
  3. flatness = geometric_mean(power) / arithmetic_mean(power)
  By the AM-GM inequality this ratio is always in (0, 1]: it is 1 only when all
  frequency bins carry equal power (perfectly flat), and approaches 0 as the
  energy concentrates into fewer bins.

A constant window has no spectral content at all, so we define its flatness as
0.0 (degenerate, not "flat noise").

CAUSALITY
  Computed over the rolling window, which holds only past/current values, so the
  causal guarantee holds (same as every other signal). Wired as a logged feature
  for the future ML layer with no Scorer weight.
"""

import numpy as np

from threadforge.signals.base import Signal


class SpectralFlatness(Signal):
    def compute(self, window: list[float]) -> float:
        arr = np.asarray(window, dtype=float)
        centered = arr - arr.mean()  # remove DC so it reflects oscillation

        # power spectrum, excluding the DC bin (index 0)
        power = np.abs(np.fft.rfft(centered)) ** 2
        power = power[1:]

        total = power.sum()
        if total <= 0.0:
            return 0.0  # constant / silent window: no spectral content

        # floor tiny bins relative to total energy so log() is well-defined;
        # this keeps the geometric mean from collapsing to 0 on near-zero bins
        power = np.maximum(power, total * 1e-12)

        geometric_mean = np.exp(np.mean(np.log(power)))
        arithmetic_mean = power.mean()
        return float(geometric_mean / arithmetic_mean)
