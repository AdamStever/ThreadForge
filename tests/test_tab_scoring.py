"""Tests for TAB-style VUS / range-AUC scoring.

The golden values below were produced by the reference VUS implementation
(`TheDatumOrg/VUS`, `metricor.RangeAUC` / `RangeAUC_volume_opt`) on the exact
inputs reconstructed in ``_make_case``. They pin our reimplementation to the
reference to the floating-point tolerance.
"""

import numpy as np
import pytest

from threadforge.tab_scoring import vus, range_auc, _segments


def _make_case(seed, n, seg_list):
    """Rebuild the fixed (labels, scores) inputs used to generate the golden values."""
    rng = np.random.RandomState(seed)
    labels = np.zeros(n, dtype=int)
    for s, e in seg_list:
        labels[s:e + 1] = 1
    scores = rng.rand(n)
    scores[labels == 1] += 0.5
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    return labels, scores


# (seed, n, seg_list, window, thre, VUS_ROC, VUS_PR, R_AUC_ROC, R_AUC_PR)
GOLDEN = [
    (0, 200, [(40, 55), (120, 130)], 10, 50,
     0.883237268470, 0.657318866508, 0.836546629567, 0.604647987998),
    (1, 300, [(50, 60), (150, 180), (250, 255)], 20, 100,
     0.875831252771, 0.720301443269, 0.816655976650, 0.653359686795),
]


@pytest.mark.parametrize("seed,n,segs,window,thre,vroc,vpr,rroc,rpr", GOLDEN)
def test_vus_matches_reference(seed, n, segs, window, thre, vroc, vpr, rroc, rpr):
    labels, scores = _make_case(seed, n, segs)
    out = vus(labels, scores, window=window, thre=thre)
    assert out["VUS_ROC"] == pytest.approx(vroc, abs=1e-6)
    assert out["VUS_PR"] == pytest.approx(vpr, abs=1e-6)


@pytest.mark.parametrize("seed,n,segs,window,thre,vroc,vpr,rroc,rpr", GOLDEN)
def test_range_auc_matches_reference(seed, n, segs, window, thre, vroc, vpr, rroc, rpr):
    labels, scores = _make_case(seed, n, segs)
    out = range_auc(labels, scores, window=window)  # reference RangeAUC uses 250 thresholds
    assert out["R_AUC_ROC"] == pytest.approx(rroc, abs=1e-6)
    assert out["R_AUC_PR"] == pytest.approx(rpr, abs=1e-6)


def test_outputs_in_unit_interval():
    labels, scores = _make_case(0, 200, [(40, 55), (120, 130)])
    out = vus(labels, scores, window=10, thre=50)
    for v in out.values():
        assert 0.0 <= v <= 1.0


def test_better_separation_scores_higher():
    """A detector that ranks anomalies above normal beats an uncorrelated one."""
    n = 300
    labels = np.zeros(n, dtype=int)
    labels[100:120] = 1
    labels[200:210] = 1

    rng = np.random.RandomState(7)
    good = rng.rand(n)
    good[labels == 1] += 1.5  # clear separation
    noise = rng.rand(n)       # uncorrelated with labels

    good_pr = vus(labels, good, window=10)["VUS_PR"]
    noise_pr = vus(labels, noise, window=10)["VUS_PR"]
    assert good_pr > noise_pr
    assert good_pr > 0.5  # well-separated detector clears the random baseline


def test_window_zero_is_pointwise_and_finite():
    labels, scores = _make_case(1, 300, [(50, 60), (150, 180), (250, 255)])
    out = vus(labels, scores, window=0, thre=50)
    assert 0.0 <= out["VUS_PR"] <= 1.0
    assert 0.0 <= out["VUS_ROC"] <= 1.0


def test_segments_finds_contiguous_runs():
    label = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    assert _segments(label) == [(1, 2), (5, 5), (7, 9)]


def test_segment_at_array_end():
    label = np.array([0, 0, 1, 1])
    assert _segments(label) == [(2, 3)]


def test_no_anomalies_raises():
    with pytest.raises(ValueError, match="no anomalies"):
        vus(np.zeros(50, dtype=int), np.random.rand(50), window=5)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="align"):
        vus(np.array([0, 1, 0]), np.array([0.1, 0.2]), window=1)
