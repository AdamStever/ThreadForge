"""NAB-style scoring — the standardized, hard-to-game benchmark.

Our own overlap-F1 turned out to be reward-hackable (flag everything -> one
file-spanning event -> a fake perfect score). NAB's scoring methodology is
designed to resist exactly that, so we adopt it as the trusted headline metric.

How it works (Lavin & Ahmad, the Numenta Anomaly Benchmark):

  - A detection is scored by a **scaled sigmoid** of its position relative to the
    anomaly window it falls in: detecting at the window *start* earns almost full
    credit, detecting at the *end* earns ~0 — earlier is better.
  - Only the **earliest** detection inside a window scores; extra detections in
    the same window are ignored. So you can't farm credit by spamming a window.
  - A detection **outside** every window is a false positive, penalized by a
    sigmoid that decays from ~0 (just after a window) to the full penalty far
    away. Flagging everything therefore accrues a huge negative FP total.
  - A window with no detection is a false negative (flat penalty).
  - Per-class weights come from an **application profile** (standard /
    reward_low_fp / reward_low_fn).

The raw score is normalized to 0–100 across the corpus:

    score = 100 * (raw - null) / (perfect - null)

where a do-nothing detector scores 0 and a detector that catches every window at
its first point with no false positives scores 100.

Note: this is a faithful implementation of NAB's scoring methodology (scaled
sigmoid, single-credit windows, FP decay, profile weights, 0–100 normalization),
not a byte-for-byte port of the reference repo.
"""

from __future__ import annotations

import math

from threadforge.data import parse_timestamp


# Application profiles: weights for true positives, false positives, false negatives.
PROFILES = {
    "standard":       {"tp": 1.0, "fp": 0.11, "fn": 1.0},
    "reward_low_fp":  {"tp": 1.0, "fp": 0.22, "fn": 1.0},
    "reward_low_fn":  {"tp": 1.0, "fp": 0.11, "fn": 2.0},
}

# Credit for catching a window at its very first point (relative position -1).
_MAX_TP = None  # filled in below


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def scaled_sigmoid(relative_position: float) -> float:
    """NAB's scoring curve.

    relative_position is in units of window-lengths measured from the window end:
      -1 = window start (earliest in-window),  0 = window end,  >0 = after.
    Returns ~+1 at the start, 0 at the end, decaying to -1 well after.
    """
    if relative_position > 3.0:
        return -1.0
    return 2.0 * sigmoid(-5.0 * relative_position) - 1.0


_MAX_TP = scaled_sigmoid(-1.0)  # ≈ 0.9866


def _window_ranges(timestamps: list[str], windows: list[tuple[str, str]]) -> list[tuple[int, int]]:
    """Map (start, end) timestamp windows to inclusive (start_idx, end_idx) ranges."""
    parsed = [parse_timestamp(t) for t in timestamps]
    ranges: list[tuple[int, int]] = []
    for a, b in windows:
        ta, tb = parse_timestamp(a), parse_timestamp(b)
        idxs = [i for i, t in enumerate(parsed) if ta <= t <= tb]
        if idxs:
            ranges.append((idxs[0], idxs[-1]))
    return ranges


def score_file(
    timestamps: list[str],
    flags: list[bool],
    windows: list[tuple[str, str]],
    profile: str = "standard",
    probation: int = 0,
) -> dict:
    """Score one file. Returns raw plus the null/perfect references for normalizing.

    `probation` excludes the first N rows (and windows that end inside them) — the
    detector's warm-up/calibration region, which it can't fairly be judged on.
    """
    w = PROFILES[profile]
    n = len(timestamps)

    ranges = [(s, e) for (s, e) in _window_ranges(timestamps, windows) if e >= probation]
    detections = [i for i in range(n) if flags[i] and i >= probation]

    raw = 0.0
    caught = 0
    for (s, e) in ranges:
        in_window = [d for d in detections if s <= d <= e]
        if in_window:
            length = max(e - s, 1)
            rel = (min(in_window) - e) / length  # start -> -1, end -> 0
            raw += w["tp"] * scaled_sigmoid(rel)
            caught += 1

    missed = len(ranges) - caught
    raw -= w["fn"] * missed

    for d in detections:
        if any(s <= d <= e for (s, e) in ranges):
            continue  # inside a window — not a false positive
        preceding = [(s, e) for (s, e) in ranges if e < d]
        if preceding:
            s_p, e_p = max(preceding, key=lambda r: r[1])
            length = max(e_p - s_p, 1)
            rel = (d - e_p) / length
        else:
            rel = 4.0  # before any window: full penalty
        raw += w["fp"] * scaled_sigmoid(rel)

    null = -w["fn"] * len(ranges)            # detect nothing -> every window missed
    perfect = w["tp"] * _MAX_TP * len(ranges)  # catch every window at its start, no FPs
    return {"raw": raw, "null": null, "perfect": perfect,
            "windows": len(ranges), "caught": caught}


def normalized_score(file_results: list[dict]) -> float:
    """Aggregate per-file results into the 0–100 NAB score."""
    raw = sum(r["raw"] for r in file_results)
    null = sum(r["null"] for r in file_results)
    perfect = sum(r["perfect"] for r in file_results)
    if perfect == null:
        return 0.0
    return 100.0 * (raw - null) / (perfect - null)
