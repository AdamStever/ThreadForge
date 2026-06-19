"""TradingAgent — a 10-signal detector plus an evolvable trade rule, graded by P&L.

An agent is the WeightedSignalDetector (all 10 signals, weighted) wired to a trade
rule (fade vs follow, threshold, size). Its genome is therefore the 10 signal
weights *plus* the three rule parameters — all searched together by the genetic
algorithm, with **Sharpe** as fitness. The market grades the agent; no oracle.

    genome = {<signal>: weight ..., "mode": 0..1, "threshold": .., "size": ..}
"""

from __future__ import annotations

from threadforge.detection import WeightedSignalDetector
from threadforge.market.backtest import backtest, max_drawdown, sharpe, total_return
from threadforge.optimization.genetic import Gene
from threadforge.presets import default_signal_names


class TradingAgent:
    def __init__(self, weights, mode: str, threshold: float, size: float,
                 *, window_size: int = 30, momentum_window: int = 5, cost: float = 0.0005):
        self.detector = WeightedSignalDetector(weights, window_size=window_size)
        self.mode = mode
        self.threshold = threshold
        self.size = size
        self.momentum_window = momentum_window
        self.cost = cost

    def pnl(self, stream):
        prices = [v for _, v in stream]
        scores = self.detector.scores(stream)
        return backtest(prices, scores, mode=self.mode, threshold=self.threshold,
                        size=self.size, momentum_window=self.momentum_window, cost=self.cost)

    def evaluate(self, stream) -> dict:
        """Risk-adjusted scorecard over a price stream (the agent's live fitness)."""
        p = self.pnl(stream)
        return {"sharpe": sharpe(p), "return": total_return(p), "max_drawdown": max_drawdown(p)}


def trader_genes() -> list[Gene]:
    """GA genome: one weight per signal + the trade-rule parameters."""
    genes = [Gene(name, 0.0, 1.0) for name in default_signal_names()]
    genes += [
        Gene("mode", 0.0, 1.0),        # < 0.5 -> fade, >= 0.5 -> follow
        Gene("threshold", 1.0, 8.0),   # anomaly-score threshold to act
        Gene("size", 0.1, 1.0),        # position size
    ]
    return genes


def agent_from_genome(genome: dict, **kwargs) -> TradingAgent:
    names = default_signal_names()
    weights = {n: genome[n] for n in names}
    mode = "follow" if genome.get("mode", 1.0) >= 0.5 else "fade"
    return TradingAgent(weights, mode, genome["threshold"], genome["size"], **kwargs)
