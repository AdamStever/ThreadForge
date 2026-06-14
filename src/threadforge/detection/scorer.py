"""Composite anomaly scorer.

Converts a dict of {signal_name: is_anomalous} flags into a single numeric
score by summing weights for each active solo signal and each active signal
pair. A step is flagged when the score meets or exceeds ``score_threshold``.

Weights are loaded from config and are intentionally naive starting values.
They are designed to be replaced by learned values (regression, neural net,
genetic search) when the modeling layer arrives — the structure stays the same,
only the numbers change.

Weight key format (in config ``scorer_weights``):
  "signal_a"          -> solo weight for signal_a
  "signal_a+signal_b" -> pair weight when both are anomalous (order-insensitive)
"""

from __future__ import annotations


class Scorer:
    def __init__(self, weights: dict[str, float], score_threshold: float = 1.0):
        """
        Args:
            weights: mapping of solo/pair keys to float weights.
            score_threshold: minimum score to call a step anomalous.
        """
        self._solo: dict[str, float] = {}
        self._pairs: dict[frozenset, float] = {}
        self.score_threshold = score_threshold

        for key, weight in weights.items():
            parts = [p.strip() for p in key.split("+")]
            if len(parts) == 1:
                self._solo[parts[0]] = weight
            elif len(parts) == 2:
                self._pairs[frozenset(parts)] = weight
            # keys with 3+ signals are ignored for now; reserved for future use

    def score(self, flags: dict[str, bool]) -> float:
        """Return composite score for one time step.

        Args:
            flags: {signal_name: True/False} — True means that signal fired.
        """
        active = {name for name, fired in flags.items() if fired}
        total = 0.0

        for name in active:
            total += self._solo.get(name, 0.0)

        active_list = sorted(active)
        for i in range(len(active_list)):
            for j in range(i + 1, len(active_list)):
                pair = frozenset([active_list[i], active_list[j]])
                total += self._pairs.get(pair, 0.0)

        return total

    def is_anomalous(self, flags: dict[str, bool]) -> bool:
        return self.score(flags) >= self.score_threshold
