# ThreadForge

A streaming anomaly-detection system: real-time ingestion → signal extraction →
threshold calibration → event detection → evaluation.

The current scope is the foundation: causal rolling-window signals, calibration,
event grouping, and by-hand evaluation against labeled anomaly windows — all in
plain Python.

## Layout

```
config/        run settings (window size, threshold, calibration fraction)
data/raw/      input CSVs (gitignored — see data/README.md)
src/threadforge/
  signals/     rolling-window features (momentum, volatility, entropy)
  detection/   calibrator, detector, anomaly events
  data/        CSV stream reader
  engine.py    fans one stream out to many signals
  evaluation.py  precision/recall vs labeled windows
scripts/       runnable entry points
tests/         pytest suite
```

## Setup

```bash
# from the project root
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

pip install -e .          # makes `threadforge` importable
pip install pytest        # for running tests
```

## Run

1. Download a NAB CSV into `data/raw/` (see `data/README.md`).
2. Run detection:

```bash
python scripts/run_detection.py data/raw/ec2_cpu_utilization_5f5533.csv
```

## Test

```bash
pytest
```

## Design notes

- **Causal by construction.** Signals only ever see past/current values, and the
  detection threshold is learned from an early calibration window — never from
  future data. This avoids lookahead bias.
- **One responsibility per module.** Signals measure, the calibrator learns a
  threshold, the detector flags and groups, evaluation scores. New capabilities
  are added as new modules/subpackages rather than by editing existing ones.
