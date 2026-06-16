"""Tests for the forecasting-based residual detector."""

from threadforge.detection import ForecastResidualDetector


def _stream(values: list[float]) -> list[tuple[str, float]]:
    return [(f"2024-01-01 00:{i:02d}:00", v) for i, v in enumerate(values)]


def test_scores_length_matches_stream():
    det = ForecastResidualDetector(probation_frac=0.0, min_history=5)
    stream = _stream([float(i % 3) for i in range(40)])
    assert len(det.scores(stream)) == len(stream)


def test_flat_stream_never_flags():
    det = ForecastResidualDetector(probation_frac=0.0, min_history=5)
    stream = _stream([5.0] * 50)            # perfectly flat: zero residuals
    assert not any(det.flags(stream, threshold=3.0))


def test_spike_after_calm_is_flagged():
    det = ForecastResidualDetector(probation_frac=0.1, min_history=5)
    # calm noise, then a single large spike near the end
    values = [5.0 + (0.1 if i % 2 else -0.1) for i in range(50)]
    values[45] = 100.0
    flags = det.flags(_stream(values), threshold=4.0)
    assert flags[45] is True
    assert sum(flags) <= 3                  # sparse, not a flood


def test_probation_suppresses_early_flags():
    det = ForecastResidualDetector(probation_frac=0.5, min_history=5)
    values = [5.0] * 50
    values[5] = 100.0                       # spike inside the probation region
    flags = det.flags(_stream(values), threshold=3.0)
    assert flags[5] is False                # not flagged during probation


def test_higher_threshold_flags_no_more():
    det = ForecastResidualDetector(probation_frac=0.1, min_history=5)
    values = [5.0 + (1.0 if i % 7 == 0 else 0.0) for i in range(60)]
    low = sum(det.flags(_stream(values), threshold=3.0))
    high = sum(det.flags(_stream(values), threshold=6.0))
    assert high <= low                      # raising the bar can only reduce flags


def test_probation_scales_with_length():
    det = ForecastResidualDetector(probation_frac=0.15, probation_max=750)
    assert det.probation(1000) == 150
    assert det.probation(100000) == 750     # capped
