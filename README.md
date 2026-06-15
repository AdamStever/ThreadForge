# ThreadForge

A streaming anomaly-detection system. Data arrives as a time series; a bank of
causal rolling-window signals extracts features in real time; a robust
calibrator learns what "normal" looks like from the opening portion of the
stream; a weighted scorer combines the signals and the detector groups
deviations into anomaly events; an evaluator scores the results against labeled
windows; and every value can be persisted to a SQLite feature store.

The architecture is domain-general — server telemetry, market data, IoT
sensors, or any other univariate `(timestamp, value)` stream plugs straight in.

## How it works

```
CSV / stream
    │
    ▼
SignalEngine ──► 10 causal signals (Volatility, ZScore, Entropy ×3,
    │            Momentum, Acceleration, Autocorrelation, Hilbert,
    │            SpectralFlatness)
    │                 │
    ▼                 ▼
RobustCalibrator  per-signal two-tailed bands (median ± k·IQR)
    │                 │
    ▼                 ▼
   Scorer  ──► weighted solo + pair voting ──► Detector ──► AnomalyEvent[]
    │                                              │
 (optional)                                        ▼
 FeatureStore (SQLite)                         Evaluator  (peak | overlap)
```

1. **Signals** — each maintains a rolling window of the last N values and emits
   one number per step. All are causal: no future data is ever seen.
2. **Calibrator** (`RobustCalibrator`) — observes the first `calibration_steps`
   signal outputs and freezes a two-tailed band at `median ± k·IQR`. Robust to
   anomalies that occur during calibration; never updated afterward.
3. **Scorer** — turns the per-signal anomaly flags into a composite score using
   configurable weights for individual signals and signal pairs. A step is
   flagged when the score crosses `score_threshold`.
4. **Detector** — streams the data, scores each step, and groups consecutive
   flags into `AnomalyEvent`s — by row count (`gap_steps`) or by elapsed time
   (`gap_seconds`) for irregularly-sampled streams.
5. **Evaluator** — compares detected events against labeled windows in either
   `peak` mode (peak timestamp inside a window) or `overlap` mode (event span
   intersects a window), reporting precision and recall.
6. **Feature store** (optional) — persists raw values and every signal score,
   keyed by run and channel, for later query, replay, or model training. Two
   interchangeable backends: `FeatureStore` (row-oriented SQLite, streaming
   writes) and `ParquetFeatureStore` (columnar Apache Arrow / Parquet, batch
   writes — better for large scans and compression).

## Signals

| Signal | Domain | What it measures |
|---|---|---|
| `Momentum` | time | Net change per step — direction and speed of trend |
| `Volatility` | time | Sample standard deviation — turbulence |
| `Acceleration` | time | Second difference — change of the rate of change |
| `ZScore` | time | Standard deviations from the rolling mean — unbounded outlier score |
| `Entropy` | amplitude | Shannon entropy over an 8-bin window — choppiness |
| `EntropyFine` | amplitude | Shannon entropy, 16 bins — fine distributional detail |
| `EntropyCoarse` | amplitude | Shannon entropy, 4 bins — coarse distributional view |
| `Autocorrelation` | time | Lag-k self-similarity — structure / predictability |
| `HilbertEnvelope` | time | Instantaneous oscillation amplitude (analytic signal) |
| `SpectralFlatness` | frequency | Tonal vs noise-like (Wiener entropy of the spectrum) |

`Sharpness` also exists in the codebase but is superseded by the unbounded
`ZScore` and is not wired into the live engine.

## Layout

```
config/          run settings (window, multiplier, calibration, gaps, scorer weights)
data/raw/        input CSVs — gitignored, see data/README.md
labels/          anomaly window registry (windows.json)
scripts/         run_detection.py · benchmark.py · inspect_store.py
src/threadforge/
  signals/       causal rolling-window feature extractors (+ base.Signal ABC)
  detection/     robust_calibrator, scorer, detector, anomaly events
  data/          stream.py (CSV reader, timestamp utils) · store.py (SQLite) · parquet_store.py (columnar)
  engine.py      fans one stream out to all signals simultaneously
  evaluation.py  precision/recall (peak or overlap matching)
tests/           pytest suite
```

## Setup

```bash
python -m venv venv
# Windows:       venv\Scripts\activate
# macOS/Linux:   source venv/bin/activate

pip install -e ".[dev]"   # installs numpy, pyarrow + pytest
```

## Run

1. Download a NAB CSV into `data/raw/` (see `data/README.md`).
2. Optionally add anomaly windows to `labels/windows.json`.

```bash
# detect on one file (add --store to persist features)
python scripts/run_detection.py data/raw/ec2_cpu_utilization_5f5533.csv
python scripts/run_detection.py data/raw/ec2_cpu_utilization_5f5533.csv --store threadforge.db
python scripts/run_detection.py data/raw/ec2_cpu_utilization_5f5533.csv --store runs/ --store-format parquet

# inspect a feature store (backend auto-detected: file => SQLite, dir => Parquet)
python scripts/inspect_store.py threadforge.db          # list runs
python scripts/inspect_store.py threadforge.db 1        # summarize a run
python scripts/inspect_store.py runs/ 1                 # same, Parquet store

# score the whole labeled corpus (peak or overlap matching)
python scripts/benchmark.py overlap
```

Current benchmark across 52 labeled NAB files: **F1 ≈ 0.499** (overlap matching),
**0.444** (peak matching).

## Test

```bash
pytest
```

## Complexity

Per-step cost of each signal over a window of size `W` (the whole stream of `n`
points is `n ×` the per-step cost):

| Signal | Per step | Notes |
|---|---|---|
| `Momentum` | **O(1)** | first/last endpoints only |
| `Acceleration` | **O(1)** | second differences telescope to a 4-endpoint closed form |
| `Volatility` | **O(1)** | incremental running sums (sum, sum of squares) |
| `ZScore` | **O(1)** | incremental running sums |
| `Entropy` / `EntropyFine` / `EntropyCoarse` | O(W) | re-bins the window each step |
| `Autocorrelation` | O(W) | sum of lagged products |
| `HilbertEnvelope` | O(W log W) | FFT |
| `SpectralFlatness` | O(W log W) | FFT |

The base `Signal.update()` passes a fresh `list(window)` to `compute()` each
step — an unavoidable O(W) copy if a signal needs random access to the window.
The O(1) signals override `update()` to compute from running state or window
endpoints and skip that copy entirely; their `compute()` is retained as the
plain O(W) reference that the fast path is tested against. The remaining signals
are O(W) or O(W·log W) by nature (they must read the whole window).

## Documentation

Formal definitions for every signal, the calibrator, the scorer, event
grouping, evaluation, and the latent-state representation are in
[docs/signal-detection-math.md](docs/signal-detection-math.md).

## Design principles

- **Causal by construction.** Signals see only past and current values. The
  threshold is learned from an early calibration window — never from future
  data. No lookahead bias anywhere in the pipeline.
- **One responsibility per module.** Signals measure; the calibrator learns; the
  scorer combines; the detector flags and groups; evaluation scores; the store
  persists. New capabilities are added as new modules rather than by editing
  existing ones.
- **Config-driven.** All tunable values live in `config/default.json`.
- **Domain-agnostic.** The core pipeline has no domain-specific logic. Any
  univariate time series with timestamps plugs in at the data layer; the storage
  schema is already channel-aware for future multi-stream data.
