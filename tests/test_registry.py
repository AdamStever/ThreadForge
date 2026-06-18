"""Tests for the detector registry."""

import pytest

from threadforge.registry import DetectorRegistry, DetectorRecord


def test_register_assigns_incrementing_ids(tmp_path):
    reg = DetectorRegistry(tmp_path / "reg.json")
    a = reg.register("ewma_forecast", params={"alpha": 0.2}, metrics={"VUS_PR": 0.196})
    b = reg.register("lstm_forecast", metrics={"VUS_PR": 0.21})
    assert (a.id, b.id) == (1, 2)
    assert a.created_at  # timestamp populated
    assert reg.latest().id == 2


def test_persists_across_instances(tmp_path):
    path = tmp_path / "reg.json"
    reg = DetectorRegistry(path)
    rec = reg.register("ewma_forecast", metrics={"VUS_PR": 0.196, "Aff_F1": 0.70})
    reg.promote(rec.id)

    reloaded = DetectorRegistry(path)
    assert [r.id for r in reloaded.all()] == [1]
    assert reloaded.get(1).metrics == {"VUS_PR": 0.196, "Aff_F1": 0.70}
    assert reloaded.champion().id == 1


def test_best_picks_highest_and_ignores_missing_metric(tmp_path):
    reg = DetectorRegistry(tmp_path / "reg.json")
    reg.register("a", metrics={"VUS_PR": 0.10})
    reg.register("b", metrics={"VUS_PR": 0.30})
    reg.register("c", metrics={"Aff_F1": 0.99})  # no VUS_PR -> ignored for that metric
    assert reg.best("VUS_PR").name == "b"
    assert reg.best("Aff_F1").name == "c"


def test_best_and_latest_empty(tmp_path):
    reg = DetectorRegistry(tmp_path / "reg.json")
    assert reg.best("VUS_PR") is None
    assert reg.latest() is None
    assert reg.champion() is None


def test_promote_and_rollback(tmp_path):
    reg = DetectorRegistry(tmp_path / "reg.json")
    v1 = reg.register("ewma_forecast", metrics={"VUS_PR": 0.196})
    v2 = reg.register("ewma_tuned", metrics={"VUS_PR": 0.22})
    reg.promote(v2.id)
    assert reg.champion().id == v2.id
    reg.promote(v1.id)                 # rollback to the previous champion
    assert reg.champion().id == v1.id


def test_get_unknown_raises(tmp_path):
    reg = DetectorRegistry(tmp_path / "reg.json")
    with pytest.raises(KeyError):
        reg.get(99)


def test_promote_unknown_raises(tmp_path):
    reg = DetectorRegistry(tmp_path / "reg.json")
    with pytest.raises(KeyError):
        reg.promote(99)


def test_missing_file_starts_empty(tmp_path):
    reg = DetectorRegistry(tmp_path / "does_not_exist.json")
    assert reg.all() == []
