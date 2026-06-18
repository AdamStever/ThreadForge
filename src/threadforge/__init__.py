"""ThreadForge: a streaming anomaly-detection system.

Real-time ingestion through to detection: causal rolling-window signals feed a
robust calibrator and weighted scorer (the heuristic path) and an EWMA
forecasting-residual detector (the unsupervised path). An online streaming
runtime scores live feeds and groups anomaly events as they arrive. Results are
graded with the trusted TAB metrics (VUS-PR, Aff-F1), and detector versions are
tracked in a registry for champion/challenger comparison.

Domain-agnostic: any sequential ``(timestamp, value)`` stream plugs in.
"""

__version__ = "0.1.0"
