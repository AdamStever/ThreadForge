"""Tests for champion-challenger promotion."""

import numpy as np
import pytest

from threadforge.registry import DetectorRegistry
from threadforge.promotion import (
    sign_test_pvalue, decide_promotion, run_promotion, collect_per_file_scores,
)


# --- sign test --------------------------------------------------------------

def test_sign_test_known_values():
    assert sign_test_pvalue(10, 10) == pytest.approx(1 / 1024)
    assert sign_test_pvalue(9, 10) == pytest.approx(11 / 1024)
    assert sign_test_pvalue(0, 10) == pytest.approx(1.0)
    assert sign_test_pvalue(5, 0) == 1.0          # nothing to test (all ties)


# --- decision logic ---------------------------------------------------------

def test_promotes_clear_consistent_winner():
    champ = [0.2] * 10
    d = decide_promotion(champ, {"c": [0.3] * 10})
    assert d.promote is True and d.challenger == "c"
    assert d.delta == pytest.approx(0.1) and d.wins == 10 and d.losses == 0


def test_no_promotion_when_below_margin():
    d = decide_promotion([0.2] * 10, {"c": [0.205] * 10}, min_delta=0.01)
    assert d.promote is False and "below margin" in d.reason


def test_no_promotion_when_not_significant():
    # 6 wins, 4 losses -> sign-test p ~ 0.377, not significant despite positive delta
    challenger = [0.3] * 6 + [0.1] * 4
    d = decide_promotion([0.2] * 10, {"c": challenger})
    assert d.promote is False
    assert d.delta > 0 and "not significant" in d.reason


def test_picks_highest_delta_among_qualifying():
    champ = [0.2] * 8
    d = decide_promotion(champ, {"small": [0.25] * 8, "big": [0.30] * 8})
    assert d.promote is True and d.challenger == "big"


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="files"):
        decide_promotion([0.2] * 5, {"c": [0.3] * 4})


def test_no_challengers():
    d = decide_promotion([0.2] * 5, {})
    assert d.promote is False and d.challenger is None


# --- registry-wired runner --------------------------------------------------

def _spiky_stream(n=400, spike=(200, 210), seed=0):
    rng = np.random.RandomState(seed)
    values = rng.rand(n) * 0.1            # low-variance baseline
    labels = np.zeros(n, dtype=int)
    s, e = spike
    values[s:e] += 5.0                    # sharp isolated spike
    labels[s:e] = 1
    return [(str(i), float(v)) for i, v in enumerate(values)], labels


def _registry(tmp_path, champ_alpha, chal_alpha):
    reg = DetectorRegistry(tmp_path / "reg.json")
    champ = reg.register("ewma_forecast", params={"ewma_alpha": champ_alpha,
                                                  "resid_window": 100, "min_history": 10})
    reg.register("ewma_challenger", params={"ewma_alpha": chal_alpha,
                                            "resid_window": 100, "min_history": 10})
    reg.promote(champ.id)
    return reg, champ.id


def test_collect_scores_aligns_per_file(tmp_path):
    reg, _ = _registry(tmp_path, 0.1, 0.5)
    streams = [_spiky_stream(seed=s) for s in range(4)]
    champ, challengers, keys = collect_per_file_scores(reg, streams, window=20)
    assert len(champ) == 4
    assert all(len(v) == 4 for v in challengers.values())
    assert len(keys) == 1


def test_run_promotion_applies_and_can_dry_run(tmp_path):
    reg, champ_id = _registry(tmp_path, 0.1, 0.5)
    streams = [_spiky_stream(seed=s) for s in range(6)]   # responsive challenger should win

    # dry run never changes the champion
    decision = run_promotion(reg, streams, window=20, apply=False)
    assert reg.champion().id == champ_id
    assert decision.n_files == 6

    # applying a positive decision promotes the winner; a negative one leaves it
    applied = run_promotion(reg, streams, window=20, apply=True)
    if applied.promote:
        assert reg.champion().id != champ_id
        assert reg.champion().name == "ewma_challenger"
    else:
        assert reg.champion().id == champ_id


def test_run_promotion_requires_champion(tmp_path):
    reg = DetectorRegistry(tmp_path / "reg.json")
    reg.register("ewma_forecast")
    with pytest.raises(ValueError, match="no champion"):
        run_promotion(reg, [_spiky_stream()], apply=False)
