# Signal & Detection Mathematics

A formal reference for the math behind ThreadForge's detection pipeline: the
signals, the calibrator, the scorer, event grouping, evaluation, and the
latent-state representation. Everything here is **causal** — at time *t* only
values up to and including *t* are ever used.

## Notation

A signal sees a rolling window of the `n = window_size` most recent values:

```
W = [x_1, x_2, ..., x_n]        x_n is the current (most recent) value
x̄ = (1/n) Σ x_i                 the window mean
```

Each signal maps `W` to a single number. A signal returns nothing (warm-up)
until the window is full, i.e. for the first `n - 1` steps.

---

## Signals

### Momentum — discrete first derivative

```
momentum = (x_n - x_1) / (n - 1)
```

Average change per step across the window. Positive = trending up. O(1) per step
(endpoints only).

### Acceleration — discrete second derivative

The mean of the second differences. The sum of second differences telescopes,
so it reduces to a four-endpoint closed form:

```
acceleration = ( (x_n - x_{n-1}) - (x_2 - x_1) ) / (n - 2)
```

Positive = speeding up, negative = slowing down. O(1) per step.

### Volatility — sample standard deviation

```
volatility = sqrt( (1/(n-1)) Σ (x_i - x̄)^2 )
```

Uses `n - 1` (Bessel's correction) because the window is a sample, not the whole
population. Computed in O(1) per step from running sums:

```
var = ( Σ x_i^2  -  (Σ x_i)^2 / n ) / (n - 1)
```

(The one-pass form differs from the two-pass form only by floating-point
rounding.)

### ZScore — standardized deviation of the current value

```
zscore = (x_n - x̄) / s          s = sample standard deviation (as above)
                                 0 if s = 0
```

Unbounded, so an extreme value can always exceed a calibrated threshold (unlike
the bounded `Sharpness` signal it replaced, which is retained in the codebase
but no longer wired into the engine). O(1) per step.

### Entropy / EntropyFine / EntropyCoarse — Shannon entropy of the window

Split the value range `[lo, hi] = [min W, max W]` into `b` equal-width buckets,
count how many values fall in each, and form proportions `p_k`:

```
bucket(x) = min( floor( (x - lo)/(hi - lo) · b ), b - 1 )
p_k       = count_k / n
H         = - Σ_k  p_k · log2(p_k)          (sum over non-empty buckets)
H         = 0   if hi = lo
```

`H` ranges from 0 (all values in one bucket — orderly) to `log2(b)` (values
spread evenly — choppy). Three resolutions share this formula: `Entropy` (b=8),
`EntropyFine` (b=16), `EntropyCoarse` (b=4).

### Autocorrelation — self-similarity at a lag

Standard sample autocorrelation at lag `k` (default `k = 1`):

```
                Σ_{i=1}^{n-k} (x_i - x̄)(x_{i+k} - x̄)
autocorr_k  =  ---------------------------------------
                     Σ_{i=1}^{n} (x_i - x̄)^2
```

Bounded to roughly `[-1, +1]`: near +1 = smooth/predictable, near 0 = noise,
near −1 = alternating. Returns 0 for a constant window. Measures temporal
*structure*, which magnitude-based signals miss.

### HilbertEnvelope — instantaneous oscillation amplitude

Demean the window (`x_c = x - x̄`) and form the analytic signal via the FFT
(the standard `scipy.signal.hilbert` construction):

```
X = FFT(x_c)
multiply X by h:  h_0 = 1;  h_{n/2} = 1 (n even);  h = 2 on positive freqs;  h = 0 on negative freqs
z = IFFT(h · X)              the complex analytic signal
envelope = |z|
HilbertEnvelope = |z_n|      the instantaneous amplitude at the latest sample
```

For a pure sine of amplitude *A* the envelope is ≈ *A*, independent of phase — it
recovers the "loudness" of the oscillation. O(n log n) per step (FFT).

### SpectralFlatness — Wiener entropy of the spectrum

Demean, take the real-FFT power spectrum `P` (excluding the DC bin), and form
the ratio of geometric to arithmetic mean:

```
P_j        = |rFFT(x_c)_j|^2          j = 1 .. n/2  (DC bin dropped)
flatness   = geomean(P) / mean(P)
           = exp( mean(ln P) ) / mean(P)
           = 0   for a constant window
```

By the AM–GM inequality this lies in `(0, 1]`: near 0 = tonal/periodic (peaky
spectrum), near 1 = noise-like (flat spectrum). O(n log n) per step.

---

## Calibration

The calibrator learns each signal's notion of "normal" from the first
`calibration_steps` of the stream and then **freezes** — it is never updated
during detection, which keeps detection causal and prevents the system from
slowly accepting anomalies as normal.

### RobustCalibrator (current) — median ± k·IQR, two-tailed

From the calibration values (sorted ascending, count `m`):

```
median = middle value (or mean of the two middle values)
Q1     = v[ m // 4 ]                  Q3 = v[ (3m) // 4 ]      (nearest-rank quartiles)
IQR    = Q3 - Q1
upper  = median + k · IQR
lower  = median - k · IQR
anomalous(x)  ⇔  x > upper  OR  x < lower
```

`k = threshold_multiplier` (default 6.0). Two properties matter:

- **Robust**: median and IQR barely move if a few anomalies leak into the
  calibration window, unlike mean and standard deviation.
- **Two-tailed**: catches both unusually high *and* unusually low values (e.g. a
  signal flatlining to zero), which a one-sided threshold misses.

### Calibrator (legacy) — mean + k·σ, one-tailed

Retained and tested but not wired into the live pipeline:

```
σ          = sqrt( (1/m) Σ (v_i - v̄)^2 )    population variance (÷ m)
threshold  = v̄ + k · σ
anomalous(x)  ⇔  x > threshold
```

Note the population variance (÷ m) here vs. the sample variance (÷ n−1) in
`Volatility` — a known inconsistency slated to be unified.

### Effective calibration size (warm-up)

Each signal emits nothing for its first `n − 1` steps, so the calibrator sees
fewer real points than requested:

```
effective_samples ≈ calibration_steps − warm_up    (≈ 600 − 30 = 570 by default)
```

The detector records this per signal and warns when it falls below
`min_calibration_samples`.

---

## Scoring

Rather than "flag if any single signal exceeds its threshold," the scorer turns
the per-signal boolean flags into a weighted composite. Let `A` be the set of
signals flagged anomalous at a step:

```
score = Σ_{s ∈ A} w_s        (solo weights)
      + Σ_{ {a,b} ⊆ A} w_{ab}  (pair weights, order-independent)

anomalous_step  ⇔  score ≥ score_threshold
```

Weights live in `config/default.json`. Pair weights let high-confidence
combinations (e.g. volatility *and* z-score together) carry more evidence than a
single noisy signal. The weights are hand-set placeholders; the structure is the
slot where learned weights will later plug in.

---

## Detection & event grouping

At each post-calibration step the detector builds the flag set, scores it, and —
when the step is anomalous — emits a flagged point. Consecutive flagged points
are grouped into one `AnomalyEvent` to avoid one sustained anomaly producing many
alerts. Two flags belong to the same event when they are close enough:

```
index mode (gap_steps):    same event  ⇔  Δrows ≤ gap_steps
time  mode (gap_seconds):  same event  ⇔  Δseconds ≤ gap_seconds
```

Time mode is correct for irregularly-sampled streams (market calendars, sparse
telemetry) where a fixed row count spans wildly different durations. Each event's
**peak** is the flagged point with the largest `|signal value|`.

---

## Evaluation

Detected events are compared against labeled anomaly windows `[a, b]` in one of
two matching modes:

```
peak mode:     event matches window  ⇔  peak_timestamp ∈ [a, b]
overlap mode:  event matches window  ⇔  start ≤ b  AND  a ≤ end   (spans intersect)
```

One event counts as at most one true positive, but may cover several windows for
recall:

```
precision = TP / (TP + FP)            TP = events matching ≥ 1 window
recall    = |matched windows| / |windows|
F1        = 2 · precision · recall / (precision + recall)
```

Overlap mode is closer to NAB's window-based philosophy: it credits a detection
that caught part of an anomaly even if its peak landed just outside the label.

---

## Latent-state representation

The signals at one instant are the coordinates of a single point in feature
space — the stream's **state**:

```
state(t) = [ signal_1(t), signal_2(t), ..., signal_d(t) ] ∈ ℝ^d
```

Because signals live on very different scales, a per-axis affine standardization
makes the geometry meaningful:

```
standardize(x)_i = (x_i - center_i) / scale_i        (0 if scale_i = 0)
distance(a, b)   = || a - b ||_2 = sqrt( Σ (a_i - b_i)^2 )
```

This is the raw, hand-built state vector. A later learned encoder will compress
these same inputs into a lower-dimensional latent state.

---

## Complexity

Per-step cost over a window of size `W` (see the README for the table). The
statistical and difference signals are O(1) per step via running sums or
closed forms; the entropy and autocorrelation signals are O(W); the FFT-based
signals (`HilbertEnvelope`, `SpectralFlatness`) are O(W log W).
