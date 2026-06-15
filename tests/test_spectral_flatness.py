"""Tests for the SpectralFlatness signal."""

import math
import random

from threadforge.signals import SpectralFlatness


def _sine(n: int, amplitude: float = 1.0, cycles: float = 4.0) -> list[float]:
    return [amplitude * math.sin(2 * math.pi * cycles * i / n) for i in range(n)]


def test_returns_none_until_window_fills():
    sig = SpectralFlatness(window_size=4)
    assert sig.update(1.0) is None
    assert sig.update(2.0) is None
    assert sig.update(3.0) is None
    assert sig.update(4.0) is not None


def test_zero_for_constant_window():
    # constant window has no spectral content => flatness defined as 0.0
    sig = SpectralFlatness(window_size=16)
    out = None
    for _ in range(16):
        out = sig.update(5.0)
    assert out == 0.0


def test_result_within_unit_interval():
    sig = SpectralFlatness(window_size=32)
    out = None
    for v in _sine(32):
        out = sig.update(v)
    assert 0.0 <= out <= 1.0


def test_pure_tone_has_low_flatness():
    # a single dominant frequency => peaky spectrum => flatness near 0
    n = 64
    sig = SpectralFlatness(window_size=n)
    out = None
    for v in _sine(n, cycles=8.0):
        out = sig.update(v)
    assert out < 0.2


def test_noise_is_flatter_than_a_pure_tone():
    n = 64

    tone_sig = SpectralFlatness(window_size=n)
    tone_out = None
    for v in _sine(n, cycles=8.0):
        tone_out = tone_sig.update(v)

    rng = random.Random(42)  # fixed seed => deterministic test
    noise_sig = SpectralFlatness(window_size=n)
    noise_out = None
    for _ in range(n):
        noise_out = noise_sig.update(rng.uniform(-1.0, 1.0))

    assert noise_out > tone_out
