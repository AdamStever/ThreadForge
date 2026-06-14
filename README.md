# ThreadForge

A streaming anomaly-detection system built in pure Python. Data arrives as a
time series; a bank of causal rolling-window signals extracts features in
real time; a calibrator learns what "normal" looks like from the opening
portion of the stream; the detector flags and groups deviations into anomaly
events; an evaluator scores the results against labeled windows.

The architecture is domain-general — server telemetry, market data, IoT
sensors, or any other univariate time series plugs straight in.

## How it works

```
CSV / stream
    │
    ▼
SignalEngine  ──► Momentum       ─┐
              ──► Volatility      │
              ──► Entropy         ├─► Detector ──► AnomalyEvent[]
              ──► Sharpness       │       │
              ──► Acceleration   ─┘       ▼
                                      Evaluator
```

1. **Signals** — each maintains a rolling window of the last N values and
   emits one number per step. All are causal: no future data is ever seen.
2. **Calibrator** — observes the first `calibration_steps` signal outputs and
   freezes a threshold at `mean + k·std`. Never updated after calibration.
3. **Detector** — streams the remainder of the data, flags any step where at
   least one signal exceeds its threshold, and groups consecutive flags into
   `AnomalyEvent` objects.
4. **Evaluator** — compares detected events against labeled anomaly windows
   and reports precision and recall.

## Signals

| Signal | What it measures |
|---|---|
| `Momentum` | Net change per step — direction and speed of trend |
| `Volatility` | Sample standard deviation — turbulence |
| `Entropy` | Shannon entropy over binned window — choppiness |
| `Sharpness` | Current value vs window mean, in units of spread |
| `Acceleration` | Second difference — rate of change of rate of change |

## Layout

```
config/          run settings (window size, threshold multiplier, etc.)
data/raw/        input CSVs — gitignored, see data/README.md
labels/          anomaly window registry (windows.json)
scripts/         CLI entry points
src/threadforge/
  signals/       rolling-window feature extractors
  detection/     calibrator, detector, anomaly events
  data/          stream reader and timestamp utilities
  engine.py      fans one stream out to all signals simultaneously
  evaluation.py  precision/recall against labeled windows
tests/           pytest suite
```

## Setup

```bash
python -m venv venv
# Windows:       venv\Scripts\activate
# macOS/Linux:   source venv/bin/activate

pip install -e ".[dev]"
```

## Run

1. Download a NAB CSV into `data/raw/` (see `data/README.md`).
2. Optionally add anomaly windows to `labels/windows.json`.
3. Run detection:

```bash
python scripts/run_detection.py data/raw/ec2_cpu_utilization_5f5533.csv
```

## Test

```bash
pytest
```

## Design principles

- **Causal by construction.** Signals see only past and current values.
  The threshold is learned from an early calibration window — never from
  future data. No lookahead bias anywhere in the pipeline.
- **One responsibility per module.** Signals measure; the calibrator learns;
  the detector flags and groups; evaluation scores. New capabilities are added
  as new modules rather than by editing existing ones.
- **Config-driven.** All tunable values live in `config/default.json`.
- **Domain-agnostic.** The core pipeline has no domain-specific logic.
  Any univariate time series with timestamps plugs in at the data layer.
