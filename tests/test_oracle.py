"""Tests for the reverse-lens (matrix-profile) oracle."""

import numpy as np
import pytest

from threadforge.oracle import (
    matrix_profile, matrix_profile_scores, noncausal_residual_scores, oracle_scores,
    left_matrix_profile_scores, causal_residual_scores, causal_oracle_scores,
)


def _brute_matrix_profile(series, m, excl):
    """Reference: z-normalised Euclidean nearest-neighbour distance per window."""
    series = np.asarray(series, float)
    n = len(series)
    length = n - m + 1

    def znorm(w):
        mu, sd = w.mean(), w.std()
        return (w - mu) / sd if sd > 1e-10 else np.zeros_like(w)

    zwin = [znorm(series[i:i + m]) for i in range(length)]
    profile = np.full(length, np.inf)
    for i in range(length):
        for j in range(length):
            if abs(i - j) <= excl:
                continue
            d = float(np.sqrt(np.sum((zwin[i] - zwin[j]) ** 2)))
            profile[i] = min(profile[i], d)
    return profile


def test_matches_brute_force():
    rng = np.random.RandomState(0)
    series = np.sin(np.linspace(0, 12, 80)) + rng.normal(0, 0.3, 80)
    m, excl = 8, 4
    got = matrix_profile(series, m, exclusion=excl)
    expected = _brute_matrix_profile(series, m, excl)
    assert got == pytest.approx(expected, abs=1e-6)


def test_finds_planted_discord():
    rng = np.random.RandomState(1)
    base = np.tile(np.sin(np.linspace(0, 2 * np.pi, 25)), 12)   # repeating motif
    base += rng.normal(0, 0.02, len(base))
    base[150:175] = rng.normal(0, 1.0, 25)                       # a discord (different shape)
    profile = matrix_profile(base, m=25, exclusion=12)
    peak = int(np.argmax(profile))
    assert 130 <= peak <= 175      # the matrix profile peaks at/around the discord


def test_scores_length_and_highlights_discord():
    rng = np.random.RandomState(2)
    base = np.tile(np.sin(np.linspace(0, 2 * np.pi, 25)), 12)
    base += rng.normal(0, 0.02, len(base))
    base[150:175] += 5.0
    scores = matrix_profile_scores(base, m=25)
    assert len(scores) == len(base)
    assert all(np.isfinite(s) for s in scores)
    assert max(scores[150:175]) > max(scores[0:120])


def test_short_series():
    with pytest.raises(ValueError):
        matrix_profile(np.arange(10.0), m=8)
    assert matrix_profile_scores(list(range(10)), m=8) == [0.0] * 10


def test_noncausal_residual_flags_spike():
    rng = np.random.RandomState(3)
    x = rng.normal(0, 1, 400)
    x[200] += 12.0
    scores = noncausal_residual_scores(x, w=20)
    assert scores[200] == max(scores)        # the spike is the top non-causal residual


def test_combined_oracle_catches_shape_and_spike():
    rng = np.random.RandomState(4)
    base = np.tile(np.sin(np.linspace(0, 2 * np.pi, 25)), 16) + rng.normal(0, 0.02, 400)
    base[100:125] = rng.normal(0, 1.0, 25)   # shape discord
    base[300] += 10.0                         # point spike
    scores = oracle_scores(base, m=25, w=10)
    assert len(scores) == len(base)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert max(scores[100:125]) > 0.8        # shape anomaly flagged (matrix profile)
    assert scores[300] > 0.8                  # spike flagged (non-causal residual)


# --- causal (backward-only) oracle ------------------------------------------

def test_causal_residual_is_causal():
    rng = np.random.RandomState(5)
    x = rng.normal(0, 1, 400)
    full = causal_residual_scores(x, w=20)
    trunc = causal_residual_scores(x[:200], w=20)
    assert list(full[:200]) == pytest.approx(list(trunc), abs=1e-9)   # past-only


def test_left_mp_no_past_scores_zero():
    x = np.tile(np.sin(np.linspace(0, 2 * np.pi, 25)), 8)
    scores = left_matrix_profile_scores(x, m=25, exclusion=12)
    assert scores[0] == 0.0                    # first window has no past to compare


def test_left_mp_flags_later_discord():
    rng = np.random.RandomState(6)
    base = np.tile(np.sin(np.linspace(0, 2 * np.pi, 25)), 16) + rng.normal(0, 0.02, 400)
    base[300:325] = rng.normal(0, 1.0, 25)     # discord after lots of normal history
    scores = left_matrix_profile_scores(base, m=25)
    assert max(scores[300:325]) > max(scores[50:280])   # unlike its past -> stands out


def test_causal_oracle_catches_shape_and_spike():
    rng = np.random.RandomState(7)
    base = np.tile(np.sin(np.linspace(0, 2 * np.pi, 25)), 16) + rng.normal(0, 0.02, 400)
    base[300:325] = rng.normal(0, 1.0, 25)     # shape discord (late)
    base[350] += 10.0                           # spike (late)
    scores = causal_oracle_scores(base, m=25, w=10)
    assert len(scores) == len(base)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert max(scores[300:325]) > 0.7
    assert scores[350] > 0.7
