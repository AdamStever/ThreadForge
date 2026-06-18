"""Drift monitoring + automated retraining — the trigger that feeds the loop.

Champion-challenger promotion only matters if fresh challengers keep arriving.
This is what produces them: watch the live input distribution, and when it shifts
away from what the champion was tuned on, spawn new candidate detectors into the
registry. The promotion loop then evaluates and (if one wins) promotes them — so
the system adapts on its own when the world changes.

  - `psi(reference, current)` — Population Stability Index, the MLOps-standard
    data-drift measure. Bins by the reference's quantiles; PSI < 0.1 is stable,
    0.1–0.25 moderate, > 0.25 significant drift.
  - `DriftMonitor` — online: freezes a reference distribution from the opening
    window, then reports PSI of a rolling current window against it, one point at
    a time.
  - `propose_challengers` — perturbs the champion's hyperparameters into a small
    candidate set.
  - `DriftRetrainer` — wires it together: on the first drift it registers the
    proposed challengers (then disarms until `rearm()`), so a drift event leaves
    the registry stocked for the next promotion pass.

Causal by construction: the reference is frozen from past data; current windows
only ever look backward.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from threadforge.registry import DetectorRecord, DetectorRegistry

_EPS = 1e-6


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index between a reference and current sample.

    Bin edges are the reference's quantiles (tails opened to ±inf so out-of-range
    current values are captured). Returns 0.0 when the reference has no spread
    (drift is undefined on a constant signal).
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        return 0.0

    edges = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return 0.0  # constant reference — nothing to compare against
    edges[0], edges[-1] = -np.inf, np.inf

    ref_hist = np.histogram(reference, bins=edges)[0] / reference.size
    cur_hist = np.histogram(current, bins=edges)[0] / current.size
    ref_hist = np.clip(ref_hist, _EPS, None)
    cur_hist = np.clip(cur_hist, _EPS, None)
    return float(np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist)))


def drift_level(psi_value: float) -> str:
    if psi_value > 0.25:
        return "significant"
    if psi_value > 0.1:
        return "moderate"
    return "stable"


@dataclass
class DriftStatus:
    index: int
    psi: float
    drift: bool
    level: str   # "warmup" | "stable" | "moderate" | "significant"


class DriftMonitor:
    def __init__(
        self,
        reference_size: int = 300,
        window: int = 300,
        bins: int = 10,
        threshold: float = 0.25,
        recompute_every: int = 50,
    ):
        self.reference_size = reference_size
        self.window = window
        self.bins = bins
        self.threshold = threshold
        self.recompute_every = recompute_every
        self._reference: list[float] = []
        self._ref_array: np.ndarray | None = None
        self._current: deque[float] = deque(maxlen=window)
        self._i = 0
        self._last_psi = 0.0

    def update(self, value: float) -> DriftStatus:
        """Ingest one value; return the current drift status."""
        self._i += 1

        if self._ref_array is None:
            self._reference.append(value)
            if len(self._reference) >= self.reference_size:
                self._ref_array = np.asarray(self._reference, dtype=float)
            return DriftStatus(self._i, 0.0, False, "warmup")

        self._current.append(value)
        if len(self._current) < self.window:
            return DriftStatus(self._i, 0.0, False, "warmup")

        # PSI over the whole current window is O(window); recompute periodically.
        if self._i % self.recompute_every == 0 or self._last_psi == 0.0:
            self._last_psi = psi(self._ref_array, list(self._current), bins=self.bins)

        drift = self._last_psi > self.threshold
        return DriftStatus(self._i, self._last_psi, drift, drift_level(self._last_psi))


def propose_challengers(
    champion_params: dict,
    alphas: tuple[float, ...] = (0.1, 0.3, 0.5),
    resid_windows: tuple[int, ...] = (100, 400),
) -> list[dict]:
    """Perturb the champion's hyperparameters into a set of candidate configs.

    Varies the EWMA smoothing and the residual-window size around the champion's
    values, skipping the champion's own config. (The genetic search could plug in
    here for a richer search; this is the deterministic baseline.)
    """
    base_alpha = champion_params.get("ewma_alpha", 0.2)
    base_rw = champion_params.get("resid_window", 200)
    seen = {(base_alpha, base_rw)}
    candidates: list[dict] = []
    for a in alphas:
        key = (a, base_rw)
        if key not in seen:
            seen.add(key)
            candidates.append({"ewma_alpha": a, "resid_window": base_rw})
    for rw in resid_windows:
        key = (base_alpha, rw)
        if key not in seen:
            seen.add(key)
            candidates.append({"ewma_alpha": base_alpha, "resid_window": rw})
    return candidates


def register_challengers(
    registry: DetectorRegistry,
    candidates: list[dict],
    name_prefix: str = "retrain",
    notes: str = "spawned on drift",
) -> list[DetectorRecord]:
    """Register candidate configs as challengers; returns the created records."""
    out = []
    for params in candidates:
        name = f"{name_prefix}_a{params['ewma_alpha']}_w{params['resid_window']}"
        out.append(registry.register(name, params=params, notes=notes))
    return out


class DriftRetrainer:
    """Drift monitor that stocks the registry with challengers when drift hits.

    On the first drift event it registers `propose_challengers(champion.params)`,
    then disarms so it does not re-spawn every step. Call `rearm()` (e.g. after a
    promotion pass) to let it fire again.
    """

    def __init__(self, registry: DetectorRegistry, monitor: DriftMonitor | None = None):
        self.registry = registry
        self.monitor = monitor or DriftMonitor()
        self._armed = True
        self.last_registered: list[DetectorRecord] = []

    def update(self, value: float) -> DriftStatus:
        status = self.monitor.update(value)
        if status.drift and self._armed:
            champ = self.registry.champion()
            params = champ.params if champ else {}
            self.last_registered = register_challengers(
                self.registry, propose_challengers(params)
            )
            self._armed = False
        return status

    def rearm(self) -> None:
        self._armed = True
