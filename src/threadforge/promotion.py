"""Champion-challenger promotion — decide whether a shadow should go live.

Shadow detection produces the comparison; this acts on it. A challenger replaces
the champion only when its win is **both**:

  - **large enough** — mean improvement on the metric clears a margin
    (``min_delta``), so we don't churn the live detector for noise-level gains; and
  - **consistent** — it beats the champion on enough files that a one-sided
    **sign test** rejects "it's a coin-flip" at level ``alpha``. The sign test is
    non-parametric and stdlib-only: no assumption that per-file scores are normal,
    and one lucky outlier file can't carry the decision.

`decide_promotion` is the pure rule (paired per-file metric values in, a
`PromotionDecision` out). `run_promotion` wires it to the registry and a labeled
corpus: it shadow-scores every registered detector per file, decides, and (unless
``apply=False``) promotes the winner. Rollback needs no special path — promoting
any earlier record reinstates it, which the registry already supports.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from threadforge.registry import DetectorRegistry
from threadforge.shadow import CHAMPION, ShadowRuntime, build_detector


@dataclass
class PromotionDecision:
    promote: bool
    challenger: str | None       # key of the chosen challenger (or best-by-delta if none qualify)
    metric: str
    champion_mean: float
    challenger_mean: float | None
    delta: float | None          # mean(challenger - champion) over files
    n_files: int
    wins: int                    # files where challenger > champion
    losses: int
    p_value: float | None        # one-sided sign-test p
    reason: str


def sign_test_pvalue(wins: int, n_eff: int) -> float:
    """One-sided sign-test p-value: P(X >= wins) for X ~ Binomial(n_eff, 0.5).

    ``n_eff`` excludes ties. Returns 1.0 when there is nothing to test.
    """
    if n_eff <= 0:
        return 1.0
    wins = max(0, min(wins, n_eff))
    tail = sum(math.comb(n_eff, k) for k in range(wins, n_eff + 1))
    return tail / (2 ** n_eff)


def decide_promotion(
    champion: list[float],
    challengers: dict[str, list[float]],
    *,
    metric: str = "VUS_PR",
    min_delta: float = 0.0,
    alpha: float = 0.05,
) -> PromotionDecision:
    """Decide whether any challenger should replace the champion.

    Args:
        champion: the champion's per-file metric values.
        challengers: ``{name: per-file metric values}``, each paired to ``champion``.
        metric: label for reporting (the values are already that metric).
        min_delta: minimum mean improvement required to consider promotion.
        alpha: significance level for the one-sided sign test.
    """
    n = len(champion)
    champ_mean = statistics.mean(champion) if champion else 0.0

    summaries = []  # (name, delta, mean, wins, losses, p)
    for name, vals in challengers.items():
        if len(vals) != n:
            raise ValueError(f"challenger {name!r} has {len(vals)} files, champion has {n}")
        diffs = [c - h for c, h in zip(vals, champion)]
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        delta = statistics.mean(diffs) if diffs else 0.0
        p = sign_test_pvalue(wins, wins + losses)
        summaries.append((name, delta, statistics.mean(vals), wins, losses, p))

    if not summaries:
        return PromotionDecision(False, None, metric, champ_mean, None, None, n, 0, 0, None,
                                 "no challengers to compare")

    qualifying = [s for s in summaries if s[1] >= min_delta and s[5] < alpha]
    if qualifying:
        name, delta, ch_mean, wins, losses, p = max(qualifying, key=lambda s: s[1])
        reason = (f"{name} beats champion by {delta:+.4f} {metric} "
                  f"({wins}-{losses} files, sign-test p={p:.4f} < {alpha})")
        return PromotionDecision(True, name, metric, champ_mean, ch_mean, delta, n, wins, losses, p, reason)

    # nothing qualifies — report the strongest candidate and why it fell short
    name, delta, ch_mean, wins, losses, p = max(summaries, key=lambda s: s[1])
    if delta < min_delta:
        reason = f"best challenger {name} delta {delta:+.4f} below margin {min_delta}"
    else:
        reason = (f"best challenger {name} not significant "
                  f"({wins}-{losses} files, sign-test p={p:.4f} >= {alpha})")
    return PromotionDecision(False, name, metric, champ_mean, ch_mean, delta, n, wins, losses, p, reason)


def collect_per_file_scores(
    registry: DetectorRegistry,
    streams: list[tuple[list[tuple[str, float]], list[int]]],
    *,
    metric: str = "VUS_PR",
    window: int = 100,
    aff_threshold: float = 2.5,
) -> tuple[list[float], dict[str, list[float]], dict[str, int]]:
    """Shadow-score the champion + every other registered detector over each stream.

    Returns ``(champion_per_file, {key: per_file}, {key: record_id})``. Detectors
    are rebuilt fresh per file with that file's warm-up so state never leaks across
    files. Keys are ``"#<id> <name>"`` so the winner maps back to a record.
    """
    champ_rec = registry.champion()
    if champ_rec is None:
        raise ValueError("registry has no champion (promote one first)")
    others = [r for r in registry.all() if r.id != champ_rec.id]
    keys = {f"#{r.id} {r.name}": r.id for r in others}

    champion: list[float] = []
    challengers: dict[str, list[float]] = {k: [] for k in keys}
    for stream, labels in streams:
        probation = min(int(0.15 * len(stream)), 750)
        champ_det = build_detector(champ_rec, probation=probation)
        challenger_dets = {k: build_detector(registry.get(keys[k]), probation=probation) for k in keys}
        rt = ShadowRuntime(champ_det, challenger_dets, threshold=10.0)
        rt.run(stream)
        comp = rt.evaluate(labels, window=window, aff_threshold=aff_threshold)
        champion.append(comp[CHAMPION][metric])
        for k in keys:
            challengers[k].append(comp[k][metric])
    return champion, challengers, keys


def run_promotion(
    registry: DetectorRegistry,
    streams: list[tuple[list[tuple[str, float]], list[int]]],
    *,
    metric: str = "VUS_PR",
    window: int = 100,
    aff_threshold: float = 2.5,
    min_delta: float = 0.0,
    alpha: float = 0.05,
    apply: bool = True,
) -> PromotionDecision:
    """Shadow-score the registry over a labeled corpus, decide, and (if ``apply``) promote.

    Returns the `PromotionDecision`. When it promotes, the registry's champion
    pointer is moved to the winning record.
    """
    champion, challengers, keys = collect_per_file_scores(
        registry, streams, metric=metric, window=window, aff_threshold=aff_threshold,
    )
    decision = decide_promotion(
        champion, challengers, metric=metric, min_delta=min_delta, alpha=alpha,
    )
    if decision.promote and apply and decision.challenger is not None:
        registry.promote(keys[decision.challenger])
    return decision


# --- online / sequential promotion -----------------------------------------

@dataclass
class PromotionEvent:
    file_index: int        # the unit after which the switch happened (0-based)
    from_champion: str
    to_champion: str
    delta: float
    wins: int
    losses: int
    p_value: float


@dataclass
class SequentialResult:
    final_champion: str
    events: list[PromotionEvent]
    live_champion: list[str]      # who was live for each unit (champion *before* its decision)
    adaptive_scores: list[float]  # the live champion's metric on each unit
    static_scores: list[float]    # the initial champion's metric on each unit


def sequential_promotion(
    per_file: dict[str, list[float]],
    champion: str,
    *,
    min_delta: float = 0.01,
    alpha: float = 0.05,
    min_files: int = 20,
    cooldown: int = 0,
) -> SequentialResult:
    """Promote *as the run goes*, not after a full batch.

    Walks the per-unit (per-file) scores in order. After each unit, it re-runs the
    same two-gate decision (`decide_promotion`) on everything seen *so far*; the
    moment a challenger qualifies, the champion switches and the new one carries
    forward. A unit is credited to whoever was champion *before* that unit's
    decision, so a promotion only helps subsequent units — exactly how a live swap
    would behave.

    ``min_files`` is the evidence required before any promotion; ``cooldown`` is the
    minimum units between promotions (anti-thrash). Note: re-deciding every unit is
    repeated peeking — the ``min_delta`` / ``min_files`` / sign-test guards make it
    conservative, but it is not a formal alpha-spending procedure.
    """
    names = list(per_file)
    if champion not in per_file:
        raise ValueError(f"champion {champion!r} not in per_file")
    n = len(per_file[champion])

    current = champion
    events: list[PromotionEvent] = []
    live: list[str] = []
    adaptive: list[float] = []
    last_promo = -1

    for k in range(n):
        live.append(current)
        adaptive.append(per_file[current][k])

        can_promote = (k + 1) >= min_files and (k - last_promo) > cooldown
        if can_promote:
            champ_so_far = per_file[current][:k + 1]
            challengers = {nm: per_file[nm][:k + 1] for nm in names if nm != current}
            decision = decide_promotion(
                champ_so_far, challengers, min_delta=min_delta, alpha=alpha,
            )
            if decision.promote and decision.challenger is not None:
                events.append(PromotionEvent(
                    k, current, decision.challenger, decision.delta,
                    decision.wins, decision.losses, decision.p_value,
                ))
                current = decision.challenger
                last_promo = k

    return SequentialResult(current, events, live, adaptive, list(per_file[champion]))
