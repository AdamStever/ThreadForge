"""Tests for the data layer."""

from threadforge.data.stream import check_timestamps


def _make_stream(intervals_seconds: list[int]) -> list[tuple[str, float]]:
    """Build a synthetic stream with the given inter-row gaps (in seconds)."""
    from datetime import datetime, timedelta
    t = datetime(2024, 1, 1, 0, 0, 0)
    rows = [(t.strftime("%Y-%m-%d %H:%M:%S"), 1.0)]
    for secs in intervals_seconds:
        t += timedelta(seconds=secs)
        rows.append((t.strftime("%Y-%m-%d %H:%M:%S"), 1.0))
    return rows


def test_no_warnings_on_uniform_stream():
    stream = _make_stream([300] * 20)
    assert check_timestamps(stream) == []


def test_detects_large_gap():
    # 20 uniform 5-min gaps, then one 2-hour gap
    stream = _make_stream([300] * 20 + [7200])
    warnings = check_timestamps(stream)
    assert len(warnings) == 1
    assert warnings[0]["type"] == "gap"
    assert warnings[0]["gap_seconds"] == 7200


def test_no_warnings_for_short_stream():
    stream = _make_stream([300])  # only 2 points, 1 gap
    assert check_timestamps(stream) == []


def test_empty_stream_returns_no_warnings():
    assert check_timestamps([]) == []
