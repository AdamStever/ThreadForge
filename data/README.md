# Data

Input data files go in `data/raw/`. They are **not** committed to the repository
(see `.gitignore`) because they are external and can be large.

## Getting the NAB dataset

The Numenta Anomaly Benchmark (NAB) provides labeled streaming time-series for
anomaly detection.

1. Clone or download it from: https://github.com/numenta/NAB
2. Copy the CSV files you want into `data/raw/`. A good starter file is:
   `data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv`
3. The official anomaly labels live in `labels/combined_labels.json` and
   `labels/combined_windows.json` in that repo.

## Format

Each CSV has two columns:

```
timestamp,value
2014-02-14 14:27:00,51.846
...
```

## Getting the TAB dataset

TAB is the current headline benchmark (1,635 univariate + multivariate series).

1. Download the pre-processed bundle from the link in the TAB repo
   (https://github.com/decisionintelligence/TAB) and unpack it under `data/TAB/`,
   so the metadata index lands at
   `data/TAB/TAB_dataset/dataset/anomaly_detect/DETECT_META.csv` and the series at
   `.../anomaly_detect/data/*.csv`. The whole `data/TAB/` tree is gitignored.
2. Score the forecasting detector on the univariate corpus by VUS-PR:
   `python scripts/run_tab.py` (subset) or `python scripts/run_tab.py --limit 0`
   (full, slow).

### TAB format

TAB CSVs are **long format**: one or more data channels followed by a `label`
channel (0/1), distinguished by the `cols` column; each channel's `date` index
restarts at 1.

```
date,data,cols
1,-142.9,channel_1
2,-164.9,channel_1
...
1,0.0,label
2,1.0,label
```

`load_tab_univariate()` in `src/threadforge/data/tab.py` turns one univariate
series into the `[(timestamp, value)]` stream plus an aligned 0/1 label list.

