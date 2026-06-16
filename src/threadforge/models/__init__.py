"""Modeling layer: turn stored signal features into trained anomaly models.

This is where the heuristic pipeline graduates to a learned one. The signals
already computed per step become a feature matrix; labeled anomaly windows
become targets; a model learns to separate them. A logistic regression over the
signals is, in effect, the learned version of the hand-weighted Scorer.
"""

from threadforge.models.dataset import FileExamples, build_file_examples, cross_file_split
from threadforge.models.window_dataset import WindowExamples, build_window_examples
from threadforge.models.baseline import (
    train,
    predict_events,
    group_predictions,
    evaluate_model,
)

# NOTE: torch_model is intentionally NOT imported here so the package stays
# usable without the optional torch dependency. Import it directly:
#   from threadforge.models.torch_model import EncoderScorer, train_model

__all__ = [
    "FileExamples",
    "build_file_examples",
    "cross_file_split",
    "WindowExamples",
    "build_window_examples",
    "train",
    "predict_events",
    "group_predictions",
    "evaluate_model",
]
