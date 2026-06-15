"""Tests for the NAB-style scorer — focused on its non-gameable properties."""

from threadforge.nab_scoring import (
    score_file,
    normalized_score,
    scaled_sigmoid,
)


def _ts(n: int) -> list[str]:
    return [f"2024-01-01 00:{i:02d}:00" for i in range(n)]


def _score(timestamps, flags, windows, profile="standard"):
    return normalized_score([score_file(timestamps, flags, windows, profile=profile)])


WINDOW = [("2024-01-01 00:05:00", "2024-01-01 00:08:00")]


def test_scaled_sigmoid_shape():
    assert scaled_sigmoid(-1.0) > 0.95     # window start: near-full credit
    assert abs(scaled_sigmoid(0.0)) < 1e-9  # window end: zero
    assert scaled_sigmoid(5.0) == -1.0      # far after: full penalty


def test_null_detector_scores_zero():
    ts = _ts(20)
    flags = [False] * 20
    assert _score(ts, flags, WINDOW) == 0.0


def test_perfect_detector_scores_100():
    ts = _ts(20)
    flags = [i == 5 for i in range(20)]  # detect exactly at the window start
    assert _score(ts, flags, WINDOW) == 100.0


def test_flag_everything_scores_negative():
    # with many non-window rows, the false-positive penalties dominate
    ts = _ts(60)
    flags = [True] * 60
    assert _score(ts, flags, WINDOW) < 0.0


def test_earlier_detection_beats_later():
    ts = _ts(20)
    early = [i == 5 for i in range(20)]   # window start
    late = [i == 8 for i in range(20)]    # window end
    assert _score(ts, early, WINDOW) > _score(ts, late, WINDOW)


def test_false_positive_reduces_score():
    ts = _ts(20)
    clean = [i == 5 for i in range(20)]
    with_fp = [i in (5, 19) for i in range(20)]
    assert _score(ts, with_fp, WINDOW) < _score(ts, clean, WINDOW)


def test_reward_low_fp_profile_penalizes_fp_more():
    ts = _ts(20)
    flags = [i in (5, 19) for i in range(20)]  # one TP + one far FP
    standard = _score(ts, flags, WINDOW, profile="standard")
    low_fp = _score(ts, flags, WINDOW, profile="reward_low_fp")
    assert low_fp < standard


def test_probation_excludes_early_window():
    ts = _ts(20)
    flags = [False] * 20
    # the only window ends at idx 8; with probation past it, it's not scored,
    # so a no-op detector is neither penalized nor credited -> 0
    r = score_file(ts, flags, WINDOW, probation=10)
    assert r["windows"] == 0
    assert normalized_score([r]) == 0.0


def test_corpus_normalization_combines_files():
    ts = _ts(20)
    perfect = score_file(ts, [i == 5 for i in range(20)], WINDOW)
    nothing = score_file(ts, [False] * 20, WINDOW)
    # one perfect file + one empty file -> between 0 and 100
    combined = normalized_score([perfect, nothing])
    assert 0.0 < combined < 100.0
