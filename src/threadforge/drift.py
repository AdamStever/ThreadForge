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


_ALPHA_GRID = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6)
_WINDOW_GRID = (50, 100, 200, 400, 800)


def _candidate_grid(champion_params: dict) -> list[dict]:
    """All hyperparameter mutations around the champion, nearest-first."""
    base_a = champion_params.get("ewma_alpha", 0.2)
    base_w = champion_params.get("resid_window", 200)
    cands = [
        {"ewma_alpha": a, "resid_window": w}
        for a in _ALPHA_GRID for w in _WINDOW_GRID
        if (a, w) != (base_a, base_w)
    ]
    cands.sort(key=lambda c: abs(c["ewma_alpha"] - base_a)
               + abs(c["resid_window"] - base_w) / max(base_w, 1))
    return cands


def propose_challengers(champion_params: dict, n: int = 5) -> list[dict]:
    """Up to ``n`` hyperparameter mutations around the champion, nearest first.

    ``n`` controls *breadth*: a small n perturbs lightly (stay near a champion that
    is doing fine), a large n casts a wider net. The retrainer drives ``n`` from
    drift severity (via :func:`spawn_count`), so the system mutates *harder* when
    it is struggling and stays lean when it is not.
    """
    return _candidate_grid(champion_params)[:max(0, n)]


def severity_from_psi(psi: float, threshold: float = 0.25) -> float:
    """Map a drift PSI to a severity in [0, 1] (0 at the threshold, 1 at 2x it)."""
    if psi <= threshold:
        return 0.0
    return min((psi - threshold) / threshold, 1.0)


def severity_from_score(score: float, *, floor: float = 0.0, ceil: float = 0.3) -> float:
    """Map a champion performance metric to severity (low score -> high severity).

    The label-based counterpart of :func:`severity_from_psi`: use it when a
    (possibly delayed) ground-truth score is available, to mutate harder when the
    live detector is actually scoring poorly.
    """
    if score >= ceil:
        return 0.0
    if score <= floor:
        return 1.0
    return (ceil - score) / (ceil - floor)


def spawn_count(severity: float, *, base: int = 2, extra: int = 10, cap: int = 12) -> int:
    """How many challengers to spawn for a given severity in [0, 1] (capped)."""
    severity = max(0.0, min(1.0, severity))
    return min(base + int(round(extra * severity)), cap)


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
    """Drift monitor that stocks the registry with challengers — *more as it worsens*.

    Adaptive exploration tied to drift severity. Drift rarely arrives at full force;
    PSI ramps up as the shifted data fills the window. So rather than spawn a fixed
    batch at the first flicker, this **escalates within a drift episode**: it spawns
    `spawn_count(severity)` challengers, and as severity climbs to new highs it tops
    the pool up to the larger count — so a worsening champion gets progressively more
    (and more diverse) mutations to be replaced by. It only adds on a new severity
    high (never every step), and **resets when drift clears**, so each episode starts
    lean again. `rearm()` force-resets the episode (e.g. after a promotion pass).
    """

    def __init__(self, registry: DetectorRegistry, monitor: DriftMonitor | None = None):
        self.registry = registry
        self.monitor = monitor or DriftMonitor()
        self.last_registered: list[DetectorRecord] = []   # challengers added on the last escalation
        self.last_severity = 0.0
        self.last_spawn_count = 0                          # cumulative spawned this episode
        self._episode_spawned = 0
        self._episode_peak = 0.0

    def _reset_episode(self) -> None:
        self._episode_spawned = 0
        self._episode_peak = 0.0

    def update(self, value: float) -> DriftStatus:
        status = self.monitor.update(value)
        if not status.drift:
            self._reset_episode()                          # episode over; next drift starts fresh
            return status

        severity = severity_from_psi(status.psi, self.monitor.threshold)
        target = spawn_count(severity)
        if target > self._episode_spawned:                 # worse than before -> top up the pool
            champ = self.registry.champion()
            params = champ.params if champ else {}
            wanted = propose_challengers(params, n=target)
            new_configs = wanted[self._episode_spawned:]   # only the additional mutations
            self.last_registered = register_challengers(self.registry, new_configs)
            self._episode_spawned = target
            self._episode_peak = severity
            self.last_severity = severity
            self.last_spawn_count = target
        return status

    def rearm(self) -> None:
        self._reset_episode()
