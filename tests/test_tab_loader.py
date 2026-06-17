"""Tests for the TAB long-format loader."""

import pytest

from threadforge.data.tab import (
    load_tab_meta, load_tab_csv, load_tab_univariate,
)


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_load_univariate_series(tmp_path):
    csv = _write(tmp_path / "uni.csv", (
        "date,data,cols\n"
        "1,10.0,channel_1\n"
        "2,11.0,channel_1\n"
        "3,12.0,channel_1\n"
        "1,0.0,label\n"
        "2,1.0,label\n"
        "3,0.0,label\n"
    ))
    stream, labels = load_tab_univariate(csv)
    assert stream == [("1", 10.0), ("2", 11.0), ("3", 12.0)]
    assert labels == [0, 1, 0]


def test_rows_sorted_by_date(tmp_path):
    """Channel values are ordered by their date index even if rows are shuffled."""
    csv = _write(tmp_path / "shuffled.csv", (
        "date,data,cols\n"
        "3,12.0,channel_1\n"
        "1,10.0,channel_1\n"
        "2,11.0,channel_1\n"
        "2,1.0,label\n"
        "3,0.0,label\n"
        "1,0.0,label\n"
    ))
    stream, labels = load_tab_univariate(csv)
    assert [v for _, v in stream] == [10.0, 11.0, 12.0]
    assert labels == [0, 1, 0]


def test_load_csv_returns_channels_and_labels(tmp_path):
    csv = _write(tmp_path / "mv.csv", (
        "date,data,cols\n"
        "1,1.0,channel1\n2,2.0,channel1\n"
        "1,3.0,channel2\n2,4.0,channel2\n"
        "1,0.0,label\n2,1.0,label\n"
    ))
    channels, labels = load_tab_csv(csv)
    assert channels == {"channel1": [1.0, 2.0], "channel2": [3.0, 4.0]}
    assert labels == [0, 1]


def test_multivariate_rejected_by_univariate_loader(tmp_path):
    csv = _write(tmp_path / "mv.csv", (
        "date,data,cols\n"
        "1,1.0,channel1\n1,3.0,channel2\n1,0.0,label\n"
    ))
    with pytest.raises(ValueError, match="one data channel"):
        load_tab_univariate(csv)


def test_missing_label_channel_raises(tmp_path):
    csv = _write(tmp_path / "nolabel.csv", (
        "date,data,cols\n1,1.0,channel_1\n2,2.0,channel_1\n"
    ))
    with pytest.raises(ValueError, match="label"):
        load_tab_csv(csv)


def test_load_meta(tmp_path):
    meta = _write(tmp_path / "DETECT_META.csv", (
        "file_name,trend,seasonal,stationary,pattern,shifting,dataset_name,"
        "train_lens,test_lens,time_steps,if_univariate,size,type_value,anomaly_rate\n"
        "a.csv,F,T,T,T,F,NAB,563,563,1126,TRUE,large,Shapelet,0.099\n"
        "b.csv,,,,,,MITDB,37500,112500,150000,FALSE,large,,0.124\n"
    ))
    records = load_tab_meta(meta)
    assert len(records) == 2
    a, b = records
    assert a.dataset_name == "NAB" and a.if_univariate is True
    assert a.train_lens == 563 and a.time_steps == 1126
    assert a.anomaly_rate == pytest.approx(0.099)
    assert b.dataset_name == "MITDB" and b.if_univariate is False
