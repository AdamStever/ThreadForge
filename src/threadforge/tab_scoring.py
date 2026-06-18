"""TAB-style scoring — VUS-PR and range-AUC, the modern TSAD accuracy measures.

ThreadForge's headline benchmark is TAB (*Unified Benchmarking of Time Series
Anomaly Detection Methods*, PVLDB 2025), which adopts **VUS-PR** as its primary
score-based metric. VUS-PR is robust where point-wise PR and the discredited
point-adjusted F1 are not: it tolerates small localisation errors and needs no
hand-picked threshold.

The construction (Paparrizos et al., *Volume Under the Surface*, VLDB 2022):

  1. **Range extension.** Each labelled anomaly segment is padded by a buffer of
     half-width ``window//2`` on both sides, with a square-root-decaying weight
     (1 at the segment edge → 0 at the buffer edge). A detection landing just
     outside a segment still earns partial credit.
  2. **Range-AUC.** Sweep the detection threshold; at each one compute a
     range-aware TPR (recall × the fraction of segments touched), FPR, and
     precision, then integrate → R-AUC-ROC and R-AUC-PR for that buffer size.
  3. **Volume Under the Surface.** Repeat for every buffer size ``0..window`` and
     average the range-AUCs. Averaging over buffer sizes removes the arbitrary
     choice of one buffer width → VUS-ROC and VUS-PR.

This is a faithful reimplementation of the reference VUS algorithm
(`TheDatumOrg/VUS`, `metricor.RangeAUC` / `RangeAUC_volume_opt`), not a port of
the package — the same discipline used for `nab_scoring.py`. It is validated
against golden values produced by the reference in `tests/test_tab_scoring.py`.

Inputs are a per-point anomaly ``score`` array and a binary ``labels`` array of
the same length; both are domain-agnostic, so any detector that emits a per-step
score (e.g. `ForecastResidualDetector.scores`) can be measured here.
"""

from __future__ import annotations

import math

import numpy as np

from threadforge._vendor.affiliation.generics import convert_vector_to_events
from threadforge._vendor.affiliation.metrics import pr_from_events


def _segments(label: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` index pairs of each maximal run of nonzero values."""
    segs: list[tuple[int, int]] = []
    n = len(label)
    i = 0
    while i < n:
        if label[i] != 0:
            j = i
            while j + 1 < n and label[j + 1] != 0:
                j += 1
            segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs


def _extend_labels(labels: np.ndarray, seq: list[tuple[int, int]], window: int) -> np.ndarray:
    """Pad each anomaly segment with a sqrt-decaying buffer of half-width ``window//2``.

    Mirrors the reference ``sequencing`` / ``extend_postive_range``: weights decay
    from 1 at a segment edge toward 0 at the buffer edge, clamped to [0, 1]. With
    ``window == 0`` the buffers are empty, so the labels stay binary.
    """
    out = labels.astype(float).copy()
    n = len(out)
    half = window // 2
    for s, e in seq:
        right = np.arange(e + 1, min(e + half + 1, n))
        if right.size:
            out[right] += np.sqrt(1.0 - (right - e) / window)
        left = np.arange(max(s - half, 0), s)
        if left.size:
            out[left] += np.sqrt(1.0 - (s - left) / window)
    return np.minimum(1.0, out)


def _merge_sequence(n: int, seq: list[tuple[int, int]], window: int) -> list[tuple[int, int]]:
    """Merge buffer-padded segments into non-overlapping ``(start, end)`` ranges.

    Faithful to the reference ``new_sequence``: two segments stay separate only if
    a gap remains between their padded edges, otherwise they merge.
    """
    half = window // 2
    a = max(seq[0][0] - half, 0)
    merged: list[tuple[int, int]] = []
    for i in range(len(seq) - 1):
        if seq[i][1] + half < seq[i + 1][0] - half:
            merged.append((a, seq[i][1] + half))
            a = seq[i + 1][0] - half
    merged.append((a, min(seq[-1][1] + half, n - 1)))
    return merged


def _threshold_indices(n: int, thre: int) -> np.ndarray:
    """The ``thre`` evenly spaced ranks into the descending-sorted score array."""
    return np.linspace(0, n - 1, thre).astype(int)


def _as_arrays(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError(f"labels and scores must align: {labels.shape} vs {scores.shape}")
    if labels.ndim != 1:
        raise ValueError("labels and scores must be 1-D")
    if np.sum(labels) == 0:
        raise ValueError("labels contain no anomalies — range/VUS metrics are undefined")
    return labels, scores


def range_auc(labels, scores, window: int, thre: int = 250) -> dict:
    """Range-AUC-ROC and Range-AUC-PR for a single buffer ``window``.

    Faithful port of the reference ``metricor.RangeAUC`` (trapezoidal PR).
    """
    labels, scores = _as_arrays(labels, scores)
    n = len(scores)
    P = float(np.sum(labels))
    score_sorted = -np.sort(-scores)

    extended = _extend_labels(labels, _segments(labels), window)
    L = _segments((extended > 0).astype(int))

    tf = np.zeros((thre + 2, 2))
    precision = np.ones(thre + 1)
    j = 0
    for i in _threshold_indices(n, thre):
        pred = scores >= score_sorted[i]
        tpr, fpr, prec = _range_point(extended, pred, P, L, n)
        j += 1
        tf[j] = [tpr, fpr]
        precision[j] = prec
    tf[j + 1] = [1.0, 1.0]

    width = tf[1:, 1] - tf[:-1, 1]
    height = (tf[1:, 0] + tf[:-1, 0]) / 2
    r_auc_roc = float(np.dot(width, height))

    width_pr = tf[1:-1, 0] - tf[:-2, 0]
    height_pr = (precision[1:] + precision[:-1]) / 2
    r_auc_pr = float(np.dot(width_pr, height_pr))

    return {"R_AUC_ROC": r_auc_roc, "R_AUC_PR": r_auc_pr}


def _range_point(extended, pred, P, L, n):
    """One threshold's range-aware TPR / FPR / precision for the single-window AUC."""
    product = extended * pred
    TP = float(np.sum(product))
    P_new = (P + float(np.sum(extended))) / 2
    recall = min(TP / P_new, 1.0)

    existence = sum(1 for s, e in L if np.any(product[s:e + 1] > 0))
    existence_ratio = existence / len(L)
    tpr = recall * existence_ratio

    FP = float(np.sum(pred)) - TP
    N_new = n - P_new
    fpr = FP / N_new

    prec = TP / float(np.sum(pred))
    return tpr, fpr, prec


def vus(labels, scores, window: int, thre: int = 250) -> dict:
    """Volume Under the Surface: VUS-ROC and VUS-PR over buffer sizes ``0..window``.

    Vectorised reimplementation of the reference ``metricor.RangeAUC_volume_opt``.
    The reference's per-threshold inner loop collapses to two matrix-vector
    products per buffer size: the per-point label weight (``extended`` restricted
    to the integration region — cores already weigh 1) dotted with the threshold
    predictions gives the true positives, and the same with cores excluded gives
    the variable part of the label mass. Mathematically identical to the reference
    (the golden-value tests pin it), with the row-by-row Python loop removed.

    ``window`` is the maximum buffer half-extent integrated over; ``thre`` is the
    number of thresholds swept. VUS-PR is ThreadForge's headline TAB metric.
    """
    labels, scores = _as_arrays(labels, scores)
    n = len(scores)
    P = float(np.sum(labels))
    seq = _segments(labels)

    full = _merge_sequence(n, seq, window)  # integration region at the max buffer
    full_mask = np.zeros(n, dtype=bool)
    for s, e in full:
        full_mask[s:e + 1] = True
    noncore = labels == 0  # cores add a fixed weight of 1 to the label mass

    # all threshold predictions at once: preds[i, k] = (score_i >= t_k)
    score_sorted = -np.sort(-scores)
    thresholds = score_sorted[_threshold_indices(n, thre)]
    preds = (scores[:, None] >= thresholds[None, :]).astype(np.float64)  # (n, thre)
    n_pred = preds.sum(axis=0)  # (thre,)

    auc_per_window = np.zeros(window + 1)
    ap_per_window = np.zeros(window + 1)

    for w in range(window + 1):
        extended = _extend_labels(labels, seq, w)
        L = _merge_sequence(n, seq, w)

        weight = extended * full_mask              # label weight per point (cores = 1)
        tp = weight @ preds                         # (thre,) true positives
        n_labels = P + (weight * noncore) @ preds   # cores contribute a fixed P
        fp = n_pred - tp

        existence = np.zeros(thre)
        for s, e in L:
            existence += preds[s:e + 1].any(axis=0)
        existence_ratio = existence / len(L)

        p_new = (P + n_labels) / 2
        recall = np.minimum(tp / p_new, 1.0)
        tpr = recall * existence_ratio
        fpr = fp / (n - p_new)
        prec = tp / n_pred

        tf = np.zeros((thre + 2, 2))
        tf[1:thre + 1, 0] = tpr
        tf[1:thre + 1, 1] = fpr
        tf[thre + 1] = [1.0, 1.0]
        precision = np.ones(thre + 1)
        precision[1:thre + 1] = prec

        width = tf[1:, 1] - tf[:-1, 1]
        height = (tf[1:, 0] + tf[:-1, 0]) / 2
        auc_per_window[w] = np.dot(width, height)

        width_pr = tf[1:-1, 0] - tf[:-2, 0]
        height_pr = precision[1:]  # left-precision, matching the reference volume
        ap_per_window[w] = np.dot(width_pr, height_pr)

    return {
        "VUS_ROC": float(np.mean(auc_per_window)),
        "VUS_PR": float(np.mean(ap_per_window)),
    }


def aff_f1(labels, pred) -> dict:
    """Affiliation precision / recall / F1 — TAB's other primary metric.

    Affiliation (Huet et al., KDD 2022) scores how *close* predictions land to
    each ground-truth event rather than counting exact overlaps, so it tolerates
    small localisation errors without the inflation of point-adjusted F1. Unlike
    VUS-PR it is defined on a **binary** prediction, so threshold the score first
    (see :func:`aff_f1_at`).

    Computed via the vendored canonical implementation
    (``threadforge._vendor.affiliation``) — kept verbatim so the numbers match the
    published method exactly. ``labels`` and ``pred`` are 0/1 arrays of equal
    length. Returns ``{Aff_P, Aff_R, Aff_F1}``.
    """
    labels = np.asarray(labels)
    pred = np.asarray(pred)
    if labels.shape != pred.shape:
        raise ValueError(f"labels and pred must align: {labels.shape} vs {pred.shape}")
    if labels.ndim != 1:
        raise ValueError("labels and pred must be 1-D")

    events_gt = convert_vector_to_events((labels > 0).astype(int).tolist())
    if not events_gt:
        raise ValueError("labels contain no anomalies — affiliation metrics are undefined")
    events_pred = convert_vector_to_events((pred > 0).astype(int).tolist())

    res = pr_from_events(events_pred, events_gt, (0, len(labels)))
    p = res["Affiliation_Precision"]
    r = float(res["Affiliation_Recall"])

    # Precision is NaN when there are no predictions; treat that as zero F1.
    p_valid = p is not None and not math.isnan(p)
    f1 = 2 * p * r / (p + r) if (p_valid and (p + r) > 0) else 0.0
    return {
        "Aff_P": float(p) if p_valid else float("nan"),
        "Aff_R": r,
        "Aff_F1": float(f1),
    }


def aff_f1_at(labels, scores, threshold: float) -> dict:
    """Affiliation F1 for a continuous ``scores`` series thresholded at ``threshold``.

    Convenience over :func:`aff_f1`: flags ``score >= threshold`` and scores the
    resulting binary prediction. Since Aff-F1 depends on the threshold (VUS-PR
    does not), callers typically sweep ``threshold`` and report the best.
    """
    scores = np.asarray(scores, dtype=float)
    return aff_f1(labels, (scores >= threshold).astype(int))
