"""Market layer — synthetic price data, paper-trading backtest, and trading agents.

This is a caller-side / data-layer concern, kept OUT of the domain-agnostic core
(signals/, detection/, engine.py). It turns a detector's anomaly scores into
simulated positions and grades them by risk-adjusted P&L, so the existing genetic
search and champion-challenger machinery can evolve agents against the market
itself — the market is the fitness function, replacing the oracle.

PAPER-TRADED / SIMULATED ONLY. No real money, no live brokerage or exchange
execution — this is a backtest/learning simulator, by project policy.
"""
