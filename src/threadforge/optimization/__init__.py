"""Optimization layer: search over hyperparameters.

A small, dependency-free genetic algorithm (`genetic.py`) plus the search-space
and fitness wiring for tuning the detector/model (`tuning.py`).
"""

from threadforge.optimization.genetic import Gene, evolve, random_genome
from threadforge.optimization.tuning import (
    SEARCH_SPACE,
    decode,
    point_scores,
    make_fitness,
    run_search,
)

__all__ = [
    "Gene",
    "evolve",
    "random_genome",
    "SEARCH_SPACE",
    "decode",
    "point_scores",
    "make_fitness",
    "run_search",
]
