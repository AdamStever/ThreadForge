"""Tests for the HilbertEnvelope signal."""

import math

from threadforge.signals import HilbertEnvelope


def _sine(n: int, amplitude: float, cycles: float = 3.0) -> list[float]:
    """A pure sine of the given amplitude sampled at n points."""
    return [amplitude * math.sin(2 * math.pi * cycles * i / n) for i in range(n)]


def test_returns_none_until_window_fills():
    sig = HilbertEnvelope(window_size=4)
    assert sig.update(1.0) is None
    assert sig.update(2.0) is None
    assert sig.update(3.0) is None
    assert sig.update(4.0) is not None


def test_zero_for_constant_window():
    # demeaning a flat window leaves all zeros => zero envelope
    sig = HilbertEnvelope(window_size=8)
    out = None
    for _ in range(8):
        out = sig.update(5.0)
    assert out == 0.0


def test_envelope_is_non_negative():
    sig = HilbertEnvelope(window_size=16)
    out = None
    for v in _sine(16, amplitude=3.0):
        out = sig.update(v)
    assert out >= 0.0


def test_envelope_tracks_sine_amplitude():
    # for a pure sine the envelope should sit on the order of the amplitude
    n = 32
    sig = HilbertEnvelope(window_size=n)
    out = None
    for v in _sine(n, amplitude=4.0):
        out = sig.update(v)
    assert 2.0 <= out <= 6.0  # ~4.0, loose band for edge effects


def test_envelope_scales_linearly_with_amplitude():
    # the transform is linear, so doubling the amplitude doubles the envelope
    n = 32
    sig1 = HilbertEnvelope(window_size=n)
    sig2 = HilbertEnvelope(window_size=n)
    out1 = out2 = None
    for v in _sine(n, amplitude=1.0):
        out1 = sig1.update(v)
    for v in _sine(n, amplitude=2.0):
        out2 = sig2.update(v)
    assert math.isclose(out2, 2.0 * out1, rel_tol=1e-9)
