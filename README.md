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
api/             C#/.NET REST API — read layer over the feature store (see api/README.md)
scripts/         run_detection.py · benchmark.py · inspect_store.py · train_baseline.py · tune_hyperparams.py · train_encoder.py · train_temporal.py · run_forecast.py
src/threadforge/
  signals/       causal rolling-window feature extractors (+ base.Signal ABC)
  detection/     robust_calibrator, scorer, detector, forecast_detector, anomaly events
  data/          stream.py (CSV reader, timestamp utils) · store.py (SQLite) · parquet_store.py (columnar)
  models/        dataset + baseline (dataset.py, baseline.py) · raw-window + torch encoder (window_dataset.py, torch_model.py)
  optimization/  genetic.py (stdlib GA) + tuning.py (hyperparameter search)
  engine.py      fans one stream out to all signals simultaneously
  presets.py     the canonical 10-signal engine (shared feature schema)
  state.py       latent-state vector (signals at one instant as a point in R^d)
  evaluation.py  precision/recall (peak or overlap matching)
  nab_scoring.py standardized NAB score (0-100), the trusted headline metric
tests/           pytest suite
```

## Setup

```bash
python -m venv venv
# Windows:       venv\Scripts\activate
# macOS/Linux:   source venv/bin/activate

pip install -e ".[dev]"   # core: numpy, pyarrow, scikit-learn + pytest

# optional deep-learning extra (PyTorch encoder). Install the CPU build:
pip install -e ".[dev,dl]" --extra-index-url https://download.pytorch.org/whl/cpu
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

# train the baseline ML model and compare it to the heuristic (cross-file split)
python scripts/train_baseline.py

# tune model hyperparameters with a genetic algorithm (train/val/test split)
python scripts/tune_hyperparams.py

# train the PyTorch encoder (raw windows) and compare to the linear baseline (needs the dl extra)
python scripts/train_encoder.py

# compare the temporal LSTM to the flat-window encoder (needs the dl extra)
python scripts/train_temporal.py
```

Current heuristic benchmark across 52 labeled NAB files: **F1 ≈ 0.499** (overlap),
**0.444** (peak). On a held-out cross-file split, the baseline learned model
(logistic regression over the signals) improves on the heuristic — e.g. **F1
≈ 0.59 vs 0.46** on the same test files — without any lookahead.

The **standardized NAB score** (0–100, where a do-nothing detector scores 0 and
a perfect one 100) is the trusted, non-gameable headline metric. It exposes what
F1 hides: the detectors **over-flag** and score far below 0, because NAB heavily
penalizes every false-positive row.

Retuning the baseline model *against the NAB score* (via the genetic search) cuts
the false-positive rate sharply — raising the decision threshold to ~0.7 and
dropping the alert rate from ~49% to ~0.3% improves held-out NAB from ≈ −6650 to
≈ −32. But the linear model can only **approach the do-nothing baseline (0)**, not
beat it: catching anomalies requires flagging, and its flags carry too many false
positives to come out ahead. A *positive* NAB score needs a higher-capacity model
— the motivation for the deep-learning phase.

The PyTorch **encoder** (learning features from raw value windows rather than the
10 hand-crafted signals) bears this out: it reaches a **positive validation NAB
(~+12)** — the first model to beat the do-nothing baseline — and ~7× better
held-out test NAB than the linear baseline (≈ −31 vs ≈ −215).

Adding **temporal memory** (an LSTM that reads the window as a sequence) lifts
validation further (~+19) but not a single held-out split (≈ −31). However,
**5-fold cross-file CV** (`scripts/cross_validate.py`) shows that −31 was an
unlucky split: across folds the encoder averages **+2.3 NAB (std 4.8, range
−0.3…+11.9)** — i.e. right around break-even with the do-nothing baseline,
occasionally beating it. The high fold-to-fold variance points at the real
limiter: a small, heterogeneous corpus. **More and more-varied data** (the later
datasets) should both stabilize and lift the score — the next lever, rather than
a fancier architecture on the same windows.

### Forecasting detector — the breakthrough

The supervised classifier track topped out near break-even because it fights
NAB's grain. NAB's strongest detectors are **unsupervised and online**:
`ForecastResidualDetector` predicts the next value (EWMA), flags when the
residual is unusual *relative to recent residuals* (a residual z-score), and runs
per-file with a probationary period — exactly NAB's protocol. Scored over the
full corpus it reaches **NAB ≈ 34** (threshold 10) — competitive with the
published Windowed Gaussian (~39), far above random (~11) and null (0), and a
world apart from the supervised models' ~+2. Run it with
`python scripts/run_forecast.py`.

Swapping the EWMA predictor for a per-file **LSTM forecaster** (the literature's
approach; `scripts/run_lstm_forecast.py`) barely moves the needle — **NAB 34.9 vs
34.3**. A learned forecaster shrinks normal residuals, but the residual z-score
already adapts to each stream, so the *relative* anomaly signal is nearly
unchanged. The takeaway: here the simple predictor is the sweet spot, so EWMA
stays the default — a strong simple baseline beats the heavier model.

### TAB benchmark (VUS-PR) — the modern standard

NAB (2015) is now widely regarded as a flawed benchmark (trivial / mislabeled
anomalies), so the headline benchmark is migrating to **TAB** (*Unified
Benchmarking of Time Series Anomaly Detection Methods*, PVLDB 2025) and its
primary metric **VUS-PR** (Volume Under the Surface) — a range-aware,
threshold-free, non-gameable measure. VUS-PR is faithfully reimplemented in
`tab_scoring.py` (validated against the reference); score the univariate corpus
with `python scripts/run_tab.py --limit 0 --max-steps 0` (parallel across cores,
with live progress + ETA).

Across the **full TAB univariate corpus — 1,635 series, 15 datasets** — the EWMA
`ForecastResidualDetector` reaches **macro VUS-PR ≈ 0.196** (window 100). The TAB
leaderboard's strongest methods sit around ~0.3, so the simple unsupervised
baseline is competitive but not state of the art.

The per-dataset breakdown is the real signal: the one-step forecaster **excels at
spiky point anomalies** but is **near-blind to subtle pattern/shape anomalies**
(KDD21, OPPORTUNITY — together 705 of the 1,635 files — drag the macro average
down). That pinpoints where a *shape-aware* detector, not a faster one, would earn
its keep.

| Dataset | Files | VUS-PR | | Dataset | Files | VUS-PR |
|---|--:|--:|---|---|--:|--:|
| YAHOO | 346 | **0.625** | | NASA-SMAP | 35 | 0.055 |
| GAIA | 184 | 0.247 | | Daphnet | 21 | 0.046 |
| NAB | 45 | 0.154 | | OPPORTUNITY | 462 | 0.044 |
| IOPS | 11 | 0.141 | | Genesis | 1 | 0.041 |
| NASA-MSL | 22 | 0.125 | | KDD21 | 243 | 0.028 |
| ECG | 22 | 0.103 | | MGAB | 6 | 0.007 |
| SVDB | 52 | 0.067 | | GHL | 1 | 0.005 |
| SMD | 184 | 0.064 | | **CORPUS (macro)** | **1635** | **0.196** |

TAB's **second** primary metric, **Aff-F1** (affiliation-based; Huet et al., KDD
2022), reaches a best **macro Aff-F1 ≈ 0.70** (residual-z threshold 2.5, swept
corpus-wide — Aff-F1 needs a binary prediction, unlike threshold-free VUS-PR).
The two are **not directly comparable**: affiliation is far more lenient by
construction — it credits a prediction for landing *near* an anomaly rather than
overlapping it — so ~0.70 reflects that leniency, not a stronger detector than the
VUS-PR ~0.196 suggests. TAB reports both because they capture complementary
things (separation vs. localisation). Both come from one pass of
`python scripts/run_tab.py --limit 0 --max-steps 0`.

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
