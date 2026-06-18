"""Tests for drift monitoring and automated challenger spawning."""

import numpy as np
import pytest

from threadforge.registry import DetectorRegistry
from threadforge.drift import (
    psi, drift_level, DriftMonitor, DriftRetrainer, DriftStatus,
    propose_challengers, register_challengers,
    severity_from_psi, severity_from_score, spawn_count,
)


class _ScriptedMonitor:
    """A DriftMonitor stand-in that emits a fixed PSI sequence (for deterministic tests)."""
    threshold = 0.25

    def __init__(self, psis):
        self._psis = list(psis)
        self._i = -1

    def update(self, value):
        self._i += 1
        p = self._psis[min(self._i, len(self._psis) - 1)]
        return DriftStatus(self._i, p, p > self.threshold, drift_level(p))


# --- PSI --------------------------------------------------------------------

def test_psi_same_distribution_is_low():
    rng = np.random.RandomState(0)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(0, 1, 2000)
    assert psi(ref, cur) < 0.1


def test_psi_shifted_distribution_is_high():
    rng = np.random.RandomState(0)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(3, 1, 2000)
    assert psi(ref, cur) > 0.25


def test_psi_constant_reference_is_zero():
    assert psi([5.0] * 100, [5.0] * 100) == 0.0
    assert psi([], [1.0]) == 0.0


def test_drift_level_thresholds():
    assert drift_level(0.05) == "stable"
    assert drift_level(0.2) == "moderate"
    assert drift_level(0.4) == "significant"


# --- monitor ----------------------------------------------------------------

def _feed(monitor, values):
    return [monitor.update(v) for v in values]


def test_monitor_stable_stream_no_drift():
    rng = np.random.RandomState(1)
    mon = DriftMonitor(reference_size=200, window=200, recompute_every=20)
    statuses = _feed(mon, rng.normal(0, 1, 600).tolist())
    assert not any(s.drift for s in statuses)


def test_monitor_detects_a_shift():
    rng = np.random.RandomState(2)
    mon = DriftMonitor(reference_size=200, window=200, recompute_every=20)
    stable = rng.normal(0, 1, 400).tolist()
    shifted = rng.normal(6, 1, 400).tolist()
    statuses = _feed(mon, stable + shifted)
    # no drift while the current window is still the reference distribution
    assert not any(s.drift for s in statuses[:400])
    # drift once the window has filled with the shifted distribution
    assert statuses[-1].drift and statuses[-1].level == "significant"


# --- challenger proposal (count-parameterised) ------------------------------

def test_propose_returns_n_distinct_excluding_champion():
    champ = {"ewma_alpha": 0.2, "resid_window": 200}
    cands = propose_challengers(champ, n=6)
    assert len(cands) == 6
    assert champ not in cands
    assert all("ewma_alpha" in c and "resid_window" in c for c in cands)
    assert len({(c["ewma_alpha"], c["resid_window"]) for c in cands}) == 6  # distinct


def test_propose_larger_n_is_superset_nearest_first():
    champ = {"ewma_alpha": 0.2, "resid_window": 200}
    small = propose_challengers(champ, n=3)
    big = propose_challengers(champ, n=8)
    assert big[:3] == small        # nearest-first, stable ordering


def test_severity_from_psi():
    assert severity_from_psi(0.1, threshold=0.25) == 0.0       # below threshold
    assert severity_from_psi(0.25, threshold=0.25) == 0.0      # at threshold
    assert severity_from_psi(0.50, threshold=0.25) == pytest.approx(1.0)  # 2x
    assert severity_from_psi(0.375, threshold=0.25) == pytest.approx(0.5) # 1.5x


def test_severity_from_score():
    assert severity_from_score(0.4, ceil=0.3) == 0.0           # doing well
    assert severity_from_score(0.0, floor=0.0, ceil=0.3) == 1.0  # floored
    assert severity_from_score(0.15, floor=0.0, ceil=0.3) == pytest.approx(0.5)


def test_spawn_count_scales_and_caps():
    assert spawn_count(0.0) == 2          # base
    assert spawn_count(1.0) == 12         # cap
    assert spawn_count(0.5) > spawn_count(0.0)
    assert spawn_count(1.0) >= spawn_count(0.5)


def test_register_challengers(tmp_path):
    reg = DetectorRegistry(tmp_path / "reg.json")
    recs = register_challengers(reg, [{"ewma_alpha": 0.3, "resid_window": 200}])
    assert len(recs) == 1 and reg.get(recs[0].id).params["ewma_alpha"] == 0.3


# --- retrainer --------------------------------------------------------------

def test_retrainer_escalates_with_severity_and_resets(tmp_path):
    reg = DetectorRegistry(tmp_path / "reg.json")
    champ = reg.register("ewma_forecast", params={"ewma_alpha": 0.2, "resid_window": 200})
    reg.promote(champ.id)

    rng = np.random.RandomState(3)
    mon = DriftMonitor(reference_size=150, window=150, recompute_every=10)
    retrainer = DriftRetrainer(reg, monitor=mon)

    # stable: nothing spawned
    for v in rng.normal(0, 1, 300):
        retrainer.update(float(v))
    assert len(reg.all()) == 1
    assert retrainer.last_spawn_count == 0

    # severe sustained shift: PSI ramps up, so the pool escalates to the cap
    for v in rng.normal(6, 1, 400):
        retrainer.update(float(v))
    assert len(reg.all()) - 1 == retrainer.last_spawn_count == 12   # cumulative, saturated
    assert retrainer.last_severity == pytest.approx(1.0)

    # still in the same episode at the cap: no further spawning
    before = len(reg.all())
    for v in rng.normal(6, 1, 200):
        retrainer.update(float(v))
    assert len(reg.all()) == before

    # drift clears (back to the reference distribution) -> episode resets
    for v in rng.normal(0, 1, 400):
        retrainer.update(float(v))
    # a fresh shift starts a new episode and spawns again
    n_before = len(reg.all())
    for v in rng.normal(-6, 1, 400):
        retrainer.update(float(v))
    assert len(reg.all()) > n_before


def _retrainer_with_champion(tmp_path, name, monitor):
    reg = DetectorRegistry(tmp_path / name)
    reg.promote(reg.register("ewma_forecast", params={"ewma_alpha": 0.2, "resid_window": 200}).id)
    return reg, DriftRetrainer(reg, monitor=monitor)


def test_retrainer_escalation_is_monotonic(tmp_path):
    # PSI ramps 0.3 -> 0.375 -> 0.5  =>  severity 0.2, 0.5, 1.0  =>  spawn 4, 7, 12
    reg, rt = _retrainer_with_champion(tmp_path, "esc.json", _ScriptedMonitor([0.3, 0.375, 0.5]))
    counts = []
    for _ in range(3):
        rt.update(0.0)
        counts.append(len(reg.all()) - 1)
    assert counts == [4, 7, 12]
    assert rt.last_spawn_count == 12


def test_mild_drift_spawns_fewer_than_severe(tmp_path):
    def run(psi_value, name):
        reg, rt = _retrainer_with_champion(tmp_path, name, _ScriptedMonitor([psi_value] * 3))
        for _ in range(3):
            rt.update(0.0)
        return rt.last_spawn_count
    assert run(0.3, "mild.json") < run(2.0, "severe.json")    # 4 < 12
