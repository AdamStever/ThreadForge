"""Tests for shadow detection."""

import numpy as np
import pytest

from threadforge.detection.online_forecast import OnlineForecastResidualDetector
from threadforge.registry import DetectorRecord, DetectorRegistry
from threadforge.shadow import (
    ShadowRuntime, best_challenger, build_detector, shadow_from_registry, CHAMPION,
)


def _labeled_stream(n=400, anom=(200, 215), seed=0):
    rng = np.random.RandomState(seed)
    values = rng.rand(n)
    labels = np.zeros(n, dtype=int)
    s, e = anom
    values[s:e] += 6.0          # a clear spike
    labels[s:e] = 1
    stream = [(str(i), float(v)) for i, v in enumerate(values)]
    return stream, labels


def _det(**kw):
    return OnlineForecastResidualDetector(probation=50, min_history=10, resid_window=100, **kw)


def test_collects_scores_for_every_detector():
    stream, _ = _labeled_stream()
    rt = ShadowRuntime(_det(), {"alt": _det(ewma_alpha=0.5)}, threshold=5.0)
    rt.run(stream)
    assert len(rt.scores(CHAMPION)) == len(stream)
    assert len(rt.scores("alt")) == len(stream)


def test_identical_challenger_matches_champion():
    """A shadow with the champion's config produces the identical score series."""
    stream, _ = _labeled_stream()
    rt = ShadowRuntime(_det(ewma_alpha=0.2), {"twin": _det(ewma_alpha=0.2)}, threshold=5.0)
    rt.run(stream)
    assert rt.scores("twin") == pytest.approx(rt.scores(CHAMPION), abs=1e-12)


def test_only_champion_alerts():
    stream, _ = _labeled_stream()
    events = []
    rt = ShadowRuntime(_det(), {"alt": _det(ewma_alpha=0.5)}, threshold=4.0,
                       on_event=events.append)
    returned = rt.run(stream)
    # on_event fires only for champion events; challengers never alert
    assert returned == rt.champion_events
    assert len(events) == len(rt.champion_events)
    assert len(events) >= 1  # the planted spike should produce at least one champion event


def test_evaluate_scores_all_detectors():
    stream, labels = _labeled_stream()
    rt = ShadowRuntime(_det(), {"alt": _det(ewma_alpha=0.5)}, threshold=5.0)
    rt.run(stream)
    comp = rt.evaluate(labels, window=20, aff_threshold=2.5)
    assert set(comp) == {CHAMPION, "alt"}
    for metrics in comp.values():
        assert 0.0 <= metrics["VUS_PR"] <= 1.0
        assert 0.0 <= metrics["Aff_F1"] <= 1.0


def test_best_challenger_picks_top_and_flags_beating():
    comp = {
        CHAMPION: {"VUS_PR": 0.20},
        "a": {"VUS_PR": 0.15},
        "b": {"VUS_PR": 0.31},
    }
    out = best_challenger(comp, "VUS_PR")
    assert out["name"] == "b"
    assert out["beats_champion"] is True
    assert out["delta"] == pytest.approx(0.11)


def test_best_challenger_none_without_challengers():
    assert best_challenger({CHAMPION: {"VUS_PR": 0.2}}, "VUS_PR") is None


def test_build_detector_from_record():
    rec = DetectorRecord(id=1, name="ewma_forecast",
                         params={"ewma_alpha": 0.3, "resid_window": 50,
                                 "probation": 10, "min_history": 5})
    det = build_detector(rec)
    assert (det.ewma_alpha, det.resid_window, det.probation, det.min_history) == (0.3, 50, 10, 5)


def test_shadow_from_registry(tmp_path):
    reg = DetectorRegistry(tmp_path / "reg.json")
    champ = reg.register("ewma_forecast", params={"ewma_alpha": 0.2})
    reg.register("ewma_tuned", params={"ewma_alpha": 0.4})
    reg.promote(champ.id)

    rt, champ_rec = shadow_from_registry(reg, threshold=5.0)
    assert champ_rec.id == champ.id
    stream, labels = _labeled_stream()
    rt.run(stream)
    comp = rt.evaluate(labels, window=20)
    assert CHAMPION in comp
    assert any(name != CHAMPION for name in comp)  # the other record runs as a shadow


def test_shadow_from_registry_requires_champion(tmp_path):
    reg = DetectorRegistry(tmp_path / "reg.json")
    reg.register("ewma_forecast")
    with pytest.raises(ValueError, match="no champion"):
        shadow_from_registry(reg, threshold=5.0)
