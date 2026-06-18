"""Shadow detection — score challenger detectors on the live stream, silently.

A *shadow* detector runs against the same feed as the live (champion) detector but
**never alerts**. It is scored exactly as the champion is, so a candidate can be
vetted on real traffic before it is ever trusted to fire. This is the safe
evaluation step that champion-challenger promotion (the next task) builds on:
shadows here produce the comparison; promotion there acts on it.

  - `ShadowRuntime` drives a champion plus any number of named challengers over one
    stream. The champion goes through the normal `StreamRuntime` (events +
    `on_event` alerts); each challenger just gets `update(value)` and its score is
    recorded. No challenger can alert.
  - `evaluate(labels, ...)` scores every detector (champion + challengers) on the
    trusted metrics (VUS-PR, Aff-F1) against the same labels — an apples-to-apples
    comparison on identical traffic.
  - `best_challenger(...)` names the strongest challenger and whether it beats the
    champion.
  - `build_detector` / `shadow_from_registry` wire the registry in: the champion
    and challengers come straight from recorded detector versions.

Scoring needs labels, so on a labeled replay (the simulator case) comparison is
immediate; on a live unlabeled feed the scores are collected now and evaluated
once labels arrive.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from threadforge.detection.event import AnomalyEvent
from threadforge.detection.online_forecast import OnlineForecastResidualDetector
from threadforge.registry import DetectorRecord, DetectorRegistry
from threadforge.streaming import StreamRuntime
from threadforge.tab_scoring import aff_f1_at, vus

CHAMPION = "champion"


def build_detector(record: DetectorRecord, probation: int | None = None):
    """Instantiate an online detector from a registry record.

    The EWMA forecast-residual family is the only online detector today; its
    hyperparameters are read from ``record.params`` (with the usual defaults).
    ``probation`` overrides the recorded value — useful for matching a finite
    replay's warm-up (``min(0.15 * n, 750)``) when scoring per file.
    """
    p = record.params
    return OnlineForecastResidualDetector(
        ewma_alpha=p.get("ewma_alpha", 0.2),
        resid_window=p.get("resid_window", 200),
        probation=p.get("probation", 750) if probation is None else probation,
        min_history=p.get("min_history", 20),
    )


class ShadowRuntime:
    """Run a champion detector live and any number of challengers in the shadows."""

    def __init__(
        self,
        champion,
        challengers: dict[str, object],
        threshold: float,
        gap_steps: int = 20,
        on_event: Callable[[AnomalyEvent], None] | None = None,
    ):
        """
        Args:
            champion: the live online detector (its events alert via ``on_event``).
            challengers: ``{name: online_detector}`` scored silently.
            threshold: flag when ``score >= threshold`` (applies to all, but only
                the champion's flags become alerts).
            gap_steps: event-grouping gap for the champion.
            on_event: alert sink — fired only for champion events.
        """
        self._champion_rt = StreamRuntime(
            champion, threshold=threshold, gap_steps=gap_steps, on_event=on_event,
        )
        self._challengers = challengers
        self._scores: dict[str, list[float]] = {CHAMPION: []}
        for name in challengers:
            self._scores[name] = []

    def feed(self, timestamp: str, value: float) -> None:
        """Push one point to the champion (alerts) and every challenger (silent)."""
        result = self._champion_rt.feed(timestamp, value)
        self._scores[CHAMPION].append(result.score)
        for name, det in self._challengers.items():
            self._scores[name].append(det.update(value))

    def run(self, source: Iterable[tuple[str, float]]) -> list[AnomalyEvent]:
        """Drive a `(timestamp, value)` source to exhaustion. Returns champion events."""
        for timestamp, value in source:
            self.feed(timestamp, value)
        self._champion_rt.close()
        return self._champion_rt.events

    @property
    def champion_events(self) -> list[AnomalyEvent]:
        return self._champion_rt.events

    def scores(self, name: str = CHAMPION) -> list[float]:
        """The recorded per-step score series for one detector."""
        return self._scores[name]

    def evaluate(self, labels, window: int = 100, aff_threshold: float = 2.5) -> dict[str, dict]:
        """Score every detector against ``labels`` on the trusted metrics.

        Returns ``{name: {"VUS_PR": .., "Aff_F1": ..}}`` for the champion and each
        challenger — an apples-to-apples comparison on identical traffic.
        """
        out: dict[str, dict] = {}
        for name, series in self._scores.items():
            out[name] = {
                "VUS_PR": vus(labels, series, window=window)["VUS_PR"],
                "Aff_F1": aff_f1_at(labels, series, aff_threshold)["Aff_F1"],
            }
        return out


def best_challenger(comparison: dict[str, dict], metric: str = "VUS_PR") -> dict | None:
    """The strongest challenger in an `evaluate` result, and whether it beats the champion.

    Returns ``{"name", metric, "delta", "beats_champion"}`` or None if there are no
    challengers / the champion lacks the metric.
    """
    champ = comparison.get(CHAMPION, {}).get(metric)
    challengers = {n: m for n, m in comparison.items() if n != CHAMPION and metric in m}
    if champ is None or not challengers:
        return None
    name = max(challengers, key=lambda n: challengers[n][metric])
    score = challengers[name][metric]
    return {
        "name": name,
        metric: score,
        "delta": score - champ,
        "beats_champion": score > champ,
    }


def shadow_from_registry(
    registry: DetectorRegistry,
    threshold: float,
    gap_steps: int = 20,
    on_event: Callable[[AnomalyEvent], None] | None = None,
) -> tuple[ShadowRuntime, DetectorRecord]:
    """Build a `ShadowRuntime` from a registry: champion live, every other record a shadow.

    Returns the runtime and the champion record. Raises if no champion is set.
    """
    champ_rec = registry.champion()
    if champ_rec is None:
        raise ValueError("registry has no champion to run live (promote one first)")
    champion = build_detector(champ_rec)
    challengers = {
        f"#{r.id} {r.name}": build_detector(r)
        for r in registry.all()
        if r.id != champ_rec.id
    }
    return ShadowRuntime(champion, challengers, threshold, gap_steps, on_event=on_event), champ_rec
