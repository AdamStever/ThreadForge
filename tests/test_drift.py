"""Tests for drift monitoring and automated challenger spawning."""

import numpy as np
import pytest

from threadforge.registry import DetectorRegistry
from threadforge.drift import (
    psi, drift_level, DriftMonitor, DriftRetrainer,
    propose_challengers, register_challengers,
)


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


# --- challenger proposal ----------------------------------------------------

def test_propose_excludes_champion_config():
    cands = propose_challengers({"ewma_alpha": 0.2, "resid_window": 200})
    assert {"ewma_alpha": 0.2, "resid_window": 200} not in cands
    assert {"ewma_alpha": 0.5, "resid_window": 200} in cands
    assert {"ewma_alpha": 0.2, "resid_window": 100} in cands


def test_register_challengers(tmp_path):
    reg = DetectorRegistry(tmp_path / "reg.json")
    recs = register_challengers(reg, [{"ewma_alpha": 0.3, "resid_window": 200}])
    assert len(recs) == 1 and reg.get(recs[0].id).params["ewma_alpha"] == 0.3


# --- retrainer --------------------------------------------------------------

def test_retrainer_spawns_challengers_once_on_drift(tmp_path):
    reg = DetectorRegistry(tmp_path / "reg.json")
    champ = reg.register("ewma_forecast", params={"ewma_alpha": 0.2, "resid_window": 200})
    reg.promote(champ.id)

    rng = np.random.RandomState(3)
    mon = DriftMonitor(reference_size=150, window=150, recompute_every=10)
    retrainer = DriftRetrainer(reg, monitor=mon)

    stable = rng.normal(0, 1, 300).tolist()
    for v in stable:
        retrainer.update(v)
    assert len(reg.all()) == 1            # nothing spawned while stable

    shifted = rng.normal(6, 1, 300).tolist()
    for v in shifted:
        retrainer.update(v)
    spawned = len(reg.all()) - 1
    assert spawned == len(retrainer.last_registered) > 0   # challengers spawned on drift

    # disarmed: further drift does not keep spawning
    before = len(reg.all())
    for v in rng.normal(6, 1, 200).tolist():
        retrainer.update(v)
    assert len(reg.all()) == before

    retrainer.rearm()                     # can fire again after a promotion pass
    for v in rng.normal(-6, 1, 300).tolist():
        retrainer.update(v)
    assert len(reg.all()) > before
