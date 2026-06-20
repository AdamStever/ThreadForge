"""Walk-forward validation splits — train on the past, test on the next chunk.

A single train/test split can be lucky. Walk-forward re-evaluates across several
expanding windows: for each fold, train on everything *before* the test chunk and
test on the chunk. Aggregating out-of-sample results across folds is the honest
way to judge a strategy (and to catch in-sample overfitting).
"""

from __future__ import annotations


def walk_forward_splits(n: int, n_folds: int = 3, min_train_frac: float = 0.5
                        ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Expanding-window splits as ``[((tr_lo, tr_hi), (te_lo, te_hi)), ...]``.

    The tail after ``min_train_frac`` is divided into ``n_folds`` test chunks; each
    fold trains on all data before its test chunk.
    """
    start = int(n * min_train_frac)
    if n_folds < 1 or start >= n - 1:
        return []
    fold = (n - start) // n_folds
    if fold < 2:
        return []
    splits = []
    for k in range(n_folds):
        te_lo = start + k * fold
        te_hi = n if k == n_folds - 1 else start + (k + 1) * fold
        splits.append(((0, te_lo), (te_lo, te_hi)))
    return splits
