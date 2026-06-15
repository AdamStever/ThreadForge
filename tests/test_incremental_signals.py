"""Equivalence tests for the incremental (O(1) per step) signals.

Volatility and ZScore maintain running sums in update() instead of recomputing
over the whole window. These tests prove the fast path agrees with the plain
O(W) reference defined in compute(), and that reset() clears the running state.
"""

import random

import pytest

from threadforge.signals import Volatility, ZScore


@pytest.mark.parametrize("cls", [Volatility, ZScore])
@pytest.mark.parametrize("amplitude", [100.0, 1000.0])
def test_incremental_update_matches_reference(cls, amplitude):
    rng = random.Random(123)
    window_size = 30
    sig = cls(window_size)
    buf: list[float] = []

    for _ in range(500):
        v = rng.uniform(-amplitude, amplitude)
        buf.append(v)
        out = sig.update(v)
        if len(buf) < window_size:
            assert out is None
        else:
            # compute() is stateless given the window — use it as the oracle
            reference = sig.compute(buf[-window_size:])
            assert out == pytest.approx(reference, rel=1e-7, abs=1e-9)


@pytest.mark.parametrize("cls", [Volatility, ZScore])
def test_reset_clears_running_state(cls):
    sig = cls(3)
    for v in (10.0, 20.0, 30.0):
        sig.update(v)
    sig.reset()
    # window emptied — needs to refill
    assert sig.update(5.0) is None
    assert sig.update(5.0) is None
    out = sig.update(5.0)  # window now [5, 5, 5]
    assert out == 0.0  # zero variance => zero volatility / zero z-score


def test_volatility_incremental_handles_constant_then_spike():
    sig = Volatility(4)
    for _ in range(4):
        assert sig.update(50.0) in (None, 0.0)
    out = sig.update(1000.0)  # window [50, 50, 50, 1000]
    assert out == pytest.approx(sig.compute([50.0, 50.0, 50.0, 1000.0]), rel=1e-9)
