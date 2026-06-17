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

import numpy as np


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

    Faithful port of the reference ``metricor.RangeAUC_volume_opt``. ``window`` is
    the maximum buffer half-extent the volume integrates over; ``thre`` is the
    number of thresholds swept. VUS-PR is ThreadForge's headline TAB metric.
    """
    labels, scores = _as_arrays(labels, scores)
    n = len(scores)
    P = float(np.sum(labels))
    seq = _segments(labels)
    full = _merge_sequence(n, seq, window)  # merged regions at the max buffer size
    score_sorted = -np.sort(-scores)

    idx = _threshold_indices(n, thre)
    n_pred = np.array([float(np.sum(scores >= score_sorted[i])) for i in idx])

    auc_per_window = np.zeros(window + 1)
    ap_per_window = np.zeros(window + 1)

    for w in range(window + 1):
        extended = _extend_labels(labels, seq, w)
        L = _merge_sequence(n, seq, w)

        tf = np.zeros((thre + 2, 2))
        precision = np.ones(thre + 1)
        j = 0
        for k, i in enumerate(idx):
            pred = scores >= score_sorted[i]

            lab = extended.copy()
            existence = 0
            for s, e in L:
                lab[s:e + 1] = extended[s:e + 1] * pred[s:e + 1]
                if np.any(pred[s:e + 1] > 0):
                    existence += 1
            for s, e in seq:
                lab[s:e + 1] = 1.0

            TP = 0.0
            n_labels = 0.0
            for s, e in full:
                TP += float(np.dot(lab[s:e + 1], pred[s:e + 1]))
                n_labels += float(np.sum(lab[s:e + 1]))

            FP = n_pred[k] - TP
            existence_ratio = existence / len(L)
            P_new = (P + n_labels) / 2
            recall = min(TP / P_new, 1.0)
            tpr = recall * existence_ratio
            fpr = FP / (n - P_new)
            prec = TP / n_pred[k]

            j += 1
            tf[j] = [tpr, fpr]
            precision[j] = prec
        tf[j + 1] = [1.0, 1.0]

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
