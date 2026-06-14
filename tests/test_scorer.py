"""Tests for the composite anomaly Scorer."""

import pytest
from threadforge.detection import Scorer


WEIGHTS = {
    "volatility":          0.5,
    "zscore":              0.5,
    "momentum":            0.3,
    "volatility+zscore":   0.8,
    "volatility+momentum": 0.6,
    "zscore+momentum":     0.6,
}


def test_no_signals_active_scores_zero():
    s = Scorer(WEIGHTS)
    assert s.score({"volatility": False, "zscore": False, "momentum": False}) == 0.0


def test_solo_signal_adds_its_weight():
    s = Scorer(WEIGHTS)
    score = s.score({"volatility": True, "zscore": False, "momentum": False})
    assert score == pytest.approx(0.5)


def test_pair_adds_both_solos_and_pair_weight():
    s = Scorer(WEIGHTS)
    score = s.score({"volatility": True, "zscore": True, "momentum": False})
    assert score == pytest.approx(0.5 + 0.5 + 0.8)


def test_pair_weight_is_order_insensitive():
    s = Scorer({"a+b": 1.0, "a": 0.2, "b": 0.2})
    score_ab = s.score({"a": True, "b": True})
    s2 = Scorer({"b+a": 1.0, "a": 0.2, "b": 0.2})
    score_ba = s2.score({"a": True, "b": True})
    assert score_ab == pytest.approx(score_ba)


def test_is_anomalous_below_threshold():
    s = Scorer(WEIGHTS, score_threshold=2.0)
    assert not s.is_anomalous({"volatility": True, "zscore": False, "momentum": False})


def test_is_anomalous_meets_threshold():
    s = Scorer(WEIGHTS, score_threshold=0.5)
    assert s.is_anomalous({"volatility": True, "zscore": False, "momentum": False})


def test_unknown_signal_contributes_zero():
    s = Scorer(WEIGHTS)
    score = s.score({"unknown_signal": True})
    assert score == 0.0


def test_three_signals_sums_all_pairs_and_solos():
    s = Scorer(WEIGHTS)
    score = s.score({"volatility": True, "zscore": True, "momentum": True})
    expected = 0.5 + 0.5 + 0.3 + 0.8 + 0.6 + 0.6
    assert score == pytest.approx(expected)
