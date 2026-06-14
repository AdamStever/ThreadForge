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
