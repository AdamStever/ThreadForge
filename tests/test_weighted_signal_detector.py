"""Tests for the WeightedSignalDetector (10 signals, weighted combine)."""

import numpy as np
import pytest

from threadforge.detection import WeightedSignalDetector
from threadforge.presets import default_signal_names


def _stream(n=600, spike=(400, 412), seed=0):
    rng = np.random.RandomState(seed)
    v = rng.normal(0.0, 1.0, n)
    for i in range(*spike):
        v[i] += 8.0
    return [(str(i), float(x)) for i, x in enumerate(v)]


def _det(weights=None):
    return WeightedSignalDetector(weights, window_size=30, feature_window=100,
                                  resid_window=100, min_history=10, min_window=20)


def test_scores_shape_and_finite():
    scores = _det().scores(_stream())
    assert len(scores) == 600
    assert all(np.isfinite(s) for s in scores)


def test_probation_region_scores_zero():
    det = _det()
    stream = _stream()
    scores = det.scores(stream)
    assert all(s == 0.0 for s in scores[: det.probation(len(stream))])


def test_detects_planted_anomaly():
    stream = _stream(n=700, spike=(450, 470))
    scores = _det().scores(stream)
    assert max(scores[450:470]) > max(scores[200:400])


def test_deterministic():
    stream = _stream()
    assert _det().scores(stream) == pytest.approx(_det().scores(stream), abs=1e-9)


def test_weights_change_the_output():
    stream = _stream()
    equal = _det().scores(stream)
    names = default_signal_names(30)
    only_zscore = {n: (1.0 if n == "zscore" else 0.0) for n in names}
    weighted = _det(only_zscore).scores(stream)
    assert equal != pytest.approx(weighted, abs=1e-9)   # weighting actually matters


def test_accepts_dict_and_list_weights():
    names = default_signal_names(30)
    as_list = _det([1.0] * len(names))
    as_dict = _det({n: 1.0 for n in names})
    s = _stream()
    assert as_list.scores(s) == pytest.approx(as_dict.scores(s), abs=1e-9)
