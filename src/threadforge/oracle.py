"""Reverse-lens oracle — a NON-CAUSAL retrospective anomaly scorer (a teacher).

Everything else in ThreadForge is causal by construction. This is the deliberate
exception: anomalies are often far clearer in hindsight (a level shift is only
obvious once you see the level *stayed* shifted; shape anomalies are defined by
what surrounds them). So this looks at the whole series at once to score how
anomalous each region is — a "reverse lens."

It is NOT a live detector and must never be used as one (it sees the future). Its
job is to produce *retrospective pseudo-labels*: if it recovers the true anomalies
well, it can serve as a label-free fitness/teacher signal for tuning and pruning
the live causal detectors (and is the basis for observation-based cloning).

Method: the **matrix profile** (discord discovery, Keogh et al.). For each
length-``m`` subsequence, find the z-normalised Euclidean distance to its nearest
neighbour elsewhere in the series — a subsequence unlike anything else is a
"discord", i.e. anomalous. Training-free and shape-aware, which is exactly the
regime (pattern anomalies) where the causal forecaster is weak.

This is a faithful from-scratch matrix profile via the MASS distance profile
(FFT sliding dot products); ``tests/test_oracle.py`` pins it against a brute-force
z-normalised distance.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-10


def _sliding_stats(series: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of every length-m window (length n-m+1)."""
    csum = np.concatenate([[0.0], np.cumsum(series)])
    csum2 = np.concatenate([[0.0], np.cumsum(series * series)])
    seg = csum[m:] - csum[:-m]
    seg2 = csum2[m:] - csum2[:-m]
    mean = seg / m
    var = seg2 / m - mean * mean
    std = np.sqrt(np.clip(var, 0.0, None))
    return mean, std


def _sliding_dot(query: np.ndarray, series: np.ndarray) -> np.ndarray:
    """Dot product of ``query`` (len m) with every length-m window of ``series``.

    Via FFT convolution: O(n log n) instead of O(n*m).
    """
    n = len(series)
    m = len(query)
    fft_len = 1 << (n + m).bit_length()
    prod = np.fft.irfft(np.fft.rfft(series, fft_len) * np.fft.rfft(query[::-1], fft_len), fft_len)
    return prod[m - 1:n]


def _mass(query, series, series_mean, series_std):
    """Z-normalised Euclidean distance profile of ``query`` against all windows."""
    m = len(query)
    qt = _sliding_dot(query, series)
    mu_q = query.mean()
    sig_q = query.std()
    if sig_q < _EPS:                                   # constant query: undefined corr
        return np.full(len(series_mean), np.sqrt(2.0 * m))
    denom = m * sig_q * np.maximum(series_std, _EPS)
    corr = (qt - m * mu_q * series_mean) / denom
    corr = np.clip(corr, -1.0, 1.0)
    return np.sqrt(np.clip(2.0 * m * (1.0 - corr), 0.0, None))


def matrix_profile(series, m: int, exclusion: int | None = None) -> np.ndarray:
    """Matrix profile of ``series`` for subsequence length ``m``.

    Returns an array of length ``n-m+1``: entry ``i`` is the distance from window
    ``i`` to its nearest non-trivial neighbour. High values = discords = anomalies.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 2 * m:
        raise ValueError(f"series too short ({n}) for subsequence length {m}")
    excl = m // 2 if exclusion is None else exclusion
    mean, std = _sliding_stats(series, m)
    length = n - m + 1
    profile = np.full(length, np.inf)
    for i in range(length):
        dist = _mass(series[i:i + m], series, mean, std)
        lo, hi = max(0, i - excl), min(length, i + excl + 1)
        dist[lo:hi] = np.inf                           # ignore trivial self-matches
        profile[i] = dist.min()
    return profile


def matrix_profile_scores(values, m: int = 100, exclusion: int | None = None) -> list[float]:
    """Per-point anomaly scores from the matrix profile (non-causal).

    The discord distance of each window is spread over the points it covers (max),
    giving a score per point aligned with the anomalous region.
    """
    series = np.asarray(values, dtype=float)
    n = len(series)
    if n < 2 * m:
        return [0.0] * n
    profile = matrix_profile(series, m, exclusion)
    scores = np.zeros(n)
    for i, p in enumerate(profile):
        np.maximum.at(scores, np.arange(i, i + m), p)
    return scores.tolist()


def noncausal_residual_scores(values, w: int = 20) -> np.ndarray:
    """Non-causal residual: deviation from a *centered* moving average, robust-z'd.

    The centered window uses future as well as past, so it catches spikes and
    level deviations more cleanly than a causal forecaster — the complement to the
    shape-sensitive matrix profile.
    """
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return np.zeros(len(x))
    cma = np.convolve(x, np.ones(w) / w, mode="same")
    resid = np.abs(x - cma)
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) or 1.0
    return (resid - med) / (1.4826 * mad)


def _rank01(a) -> np.ndarray:
    """Rank-normalise to [0, 1] (ties broken by order; VUS-PR only uses ordering)."""
    a = np.asarray(a, dtype=float)
    if len(a) <= 1:
        return np.zeros(len(a))
    return a.argsort().argsort() / (len(a) - 1)


def oracle_scores(values, m: int = 64, w: int = 20) -> list[float]:
    """Combined reverse-lens score: max of the matrix-profile and non-causal-residual lenses.

    Matrix profile covers shape/pattern anomalies; the non-causal residual covers
    spikes/level deviations. Rank-normalising each and taking the max flags a point
    that is anomalous under *either* lens — a stronger, more universal teacher than
    either alone. NON-CAUSAL (the upper-bound reference); use the causal variants
    below for a live teacher.
    """
    mp = matrix_profile_scores(values, m)
    res = noncausal_residual_scores(values, w)
    return np.maximum(_rank01(mp), _rank01(res)).tolist()


# --- causal (backward-only) oracle — runs on a live feed --------------------
#
# These use only PAST data, so they can teach a causal student online with no
# look-ahead. The only inherent latency is the subsequence length m: a *shape*
# anomaly can't be confirmed until its whole window has arrived — you judge the
# window once it completes, never using anything beyond it.


def left_matrix_profile(series, m: int, exclusion: int | None = None,
                        past_window: int | None = None) -> np.ndarray:
    """Causal matrix profile: each window's nearest neighbour among **past** windows only.

    "Is this pattern unlike anything I've seen so far?" Windows with no eligible
    past get ``inf``. ``past_window`` bounds how far back to compare (None = all
    history) — smaller is cheaper and more local.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < m + 1:
        raise ValueError(f"series too short ({n}) for subsequence length {m}")
    excl = m // 2 if exclusion is None else exclusion
    mean, std = _sliding_stats(series, m)
    length = n - m + 1
    profile = np.full(length, np.inf)
    for i in range(length):
        hi = i - excl                                   # past windows: indices [lo, hi)
        lo = 0 if past_window is None else max(0, hi - past_window)
        if hi <= lo:
            continue                                    # not enough history yet
        dist = _mass(series[i:i + m], series, mean, std)
        profile[i] = dist[lo:hi].min()
    return profile


def left_matrix_profile_scores(values, m: int = 100, exclusion: int | None = None,
                               past_window: int | None = None) -> list[float]:
    """Per-point causal discord scores from the left matrix profile."""
    series = np.asarray(values, dtype=float)
    n = len(series)
    if n < m + 1:
        return [0.0] * n
    profile = left_matrix_profile(series, m, exclusion, past_window)
    scores = np.zeros(n)
    for i, p in enumerate(profile):
        if np.isfinite(p):
            np.maximum.at(scores, np.arange(i, i + m), p)
    return scores.tolist()


def causal_residual_scores(values, w: int = 20) -> np.ndarray:
    """Causal residual: deviation from a *trailing* moving average (past w incl. current)."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return np.zeros(0)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - w + 1)
    cnt = idx - lo + 1
    trailing_mean = (csum[idx + 1] - csum[lo]) / cnt
    return np.abs(x - trailing_mean)


def causal_oracle_scores(values, m: int = 64, w: int = 20,
                         past_window: int | None = None) -> list[float]:
    """Combined CAUSAL oracle: max of the left-matrix-profile and trailing-residual lenses.

    Backward-only, so it scores a live feed and can teach a causal student online.
    """
    mp = left_matrix_profile_scores(values, m, past_window=past_window)
    res = causal_residual_scores(values, w)
    return np.maximum(_rank01(mp), _rank01(res)).tolist()
