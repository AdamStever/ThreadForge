"""Tests for walk-forward validation splits."""

from threadforge.market.walkforward import walk_forward_splits


def test_splits_are_expanding_and_causal():
    splits = walk_forward_splits(100, n_folds=3, min_train_frac=0.5)
    assert len(splits) == 3
    for (tr_lo, tr_hi), (te_lo, te_hi) in splits:
        assert tr_lo == 0                  # expanding window always starts at 0
        assert tr_hi == te_lo              # train ends exactly where test begins (no look-ahead/overlap)
        assert te_lo < te_hi               # non-empty test


def test_splits_cover_the_tail_contiguously():
    splits = walk_forward_splits(100, n_folds=3, min_train_frac=0.5)
    assert splits[0][1][0] == 50           # first test starts at min_train_frac
    assert splits[-1][1][1] == 100         # last test runs to the end
    for a, b in zip(splits, splits[1:]):
        assert a[1][1] == b[1][0]          # test chunks are contiguous


def test_too_few_points_returns_empty():
    assert walk_forward_splits(3, n_folds=3) == []
    assert walk_forward_splits(0, n_folds=3) == []
    assert walk_forward_splits(100, n_folds=0) == []
