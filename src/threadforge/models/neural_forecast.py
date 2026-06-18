"""Neural forecasting-residual detector — an LSTM predictor in place of EWMA.

Same recipe as `ForecastResidualDetector` (one-step prediction -> residual ->
residual z-score -> threshold), but the predictor is a small LSTM trained per file
on the probationary prefix (unsupervised — anomaly labels are never used). It runs
on the GPU when one is available, CPU otherwise.

The motivation is the corpus diagnosis: the EWMA one-step predictor is strong on
spiky point anomalies but near-blind to the *pattern/shape* anomalies in datasets
like KDD21 and OPPORTUNITY. An LSTM forecaster models the run-up's dynamics, so it
should fire where EWMA can't.

It exposes the same ``scores`` / ``flags`` / ``probation`` interface as the EWMA
detector, so it drops straight into the shadow / promotion comparison as a
challenger. Imported on demand so the core package stays torch-free.
"""

from __future__ import annotations

from threadforge.detection.forecast_detector import residual_zscores
from threadforge.models.torch_forecaster import lstm_residuals


class NeuralForecastResidualDetector:
    def __init__(
        self,
        *,
        window: int = 20,
        hidden_dim: int = 32,
        epochs: int = 15,
        lr: float = 1e-2,
        resid_window: int = 200,
        probation_frac: float = 0.15,
        probation_max: int = 750,
        min_history: int = 20,
        seed: int = 0,
        device=None,
    ):
        self.window = window
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.resid_window = resid_window
        self.probation_frac = probation_frac
        self.probation_max = probation_max
        self.min_history = min_history
        self.seed = seed
        self.device = device

    def probation(self, n: int) -> int:
        return min(int(self.probation_frac * n), self.probation_max)

    def scores(self, stream: list[tuple[str, float]]) -> list[float]:
        """Per-step anomaly score (residual z-score). 0.0 during warm-up/probation."""
        values = [v for _, v in stream]
        probation = self.probation(len(values))
        residuals = lstm_residuals(
            values, probation,
            window=self.window, hidden_dim=self.hidden_dim,
            epochs=self.epochs, lr=self.lr, seed=self.seed, device=self.device,
        )
        return residual_zscores(residuals, probation, self.resid_window, self.min_history)

    def flags(self, stream: list[tuple[str, float]], threshold: float) -> list[bool]:
        return [s >= threshold for s in self.scores(stream)]
