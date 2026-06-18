"""Tests for the affiliation (Aff-F1) metric wrapper.

The numbers come from the vendored canonical affiliation implementation
(`threadforge._vendor.affiliation`, Huet et al., KDD 2022), which is kept
verbatim. These tests pin our thin wrapper (event conversion + F1 combination +
edge handling) on top of it.
"""

import numpy as np
import pytest

from threadforge.tab_scoring import aff_f1, aff_f1_at


def _labels(n, segs):
    y = np.zeros(n, dtype=int)
    for s, e in segs:
        y[s:e] = 1
    return y


def test_perfect_prediction_scores_one():
    labels = _labels(100, [(40, 50), (70, 75)])
    out = aff_f1(labels, labels)
    assert out["Aff_P"] == pytest.approx(1.0)
    assert out["Aff_R"] == pytest.approx(1.0)
    assert out["Aff_F1"] == pytest.approx(1.0)


def test_near_miss_is_tolerated():
    """A prediction shifted a few steps still scores high — affiliation's point."""
    labels = _labels(100, [(40, 50), (70, 75)])
    pred = _labels(100, [(42, 52), (71, 76)])
    out = aff_f1(labels, pred)
    assert 0.9 < out["Aff_F1"] < 1.0


def test_golden_values():
    """Pinned to the vendored reference on a fixed input."""
    labels = _labels(60, [(10, 15), (40, 45)])
    pred = _labels(60, [(11, 16), (30, 33)])
    out = aff_f1(labels, pred)
    assert out["Aff_P"] == pytest.approx(0.6397202797202798, abs=1e-9)
    assert out["Aff_R"] == pytest.approx(0.7040559440559441, abs=1e-9)
    assert out["Aff_F1"] == pytest.approx(0.6703480200066378, abs=1e-9)


def test_empty_prediction_gives_zero_f1():
    labels = _labels(100, [(40, 50)])
    out = aff_f1(labels, np.zeros(100, dtype=int))
    assert out["Aff_F1"] == 0.0
    assert out["Aff_R"] == 0.0
    assert np.isnan(out["Aff_P"])  # precision undefined with no predictions


def test_closer_prediction_scores_higher():
    labels = _labels(200, [(90, 110)])
    near = _labels(200, [(92, 112)])     # off by 2
    far = _labels(200, [(150, 170)])     # well away
    assert aff_f1(labels, near)["Aff_F1"] > aff_f1(labels, far)["Aff_F1"]


def test_aff_f1_at_thresholds_scores():
    labels = _labels(80, [(30, 40)])
    scores = np.zeros(80, dtype=float)
    scores[30:40] = 5.0  # clearly above threshold inside the anomaly
    scores[10] = 0.1     # noise below threshold
    out = aff_f1_at(labels, scores, threshold=1.0)
    # equivalent to thresholding by hand
    assert out == aff_f1(labels, (scores >= 1.0).astype(int))
    assert out["Aff_F1"] > 0.9


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="align"):
        aff_f1(np.array([0, 1, 0]), np.array([0, 1]))


def test_no_anomalies_raises():
    with pytest.raises(ValueError, match="no anomalies"):
        aff_f1(np.zeros(50, dtype=int), np.ones(50, dtype=int))
