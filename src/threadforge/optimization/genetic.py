"""A small, dependency-free genetic algorithm for real/integer hyperparameters.

A genome is a dict mapping gene name -> value. The GA maintains a population of
genomes, scores each with a fitness function (higher is better), and evolves the
population over generations with:

  - elitism      the best few genomes carry over unchanged
  - tournament   parents are the fittest of a small random sample
  - crossover    a child takes each gene from one parent or the other
  - mutation     genes are nudged by gaussian noise (then clamped to bounds)

It is intentionally simple and stdlib-only — a transparent optimizer rather than
a black-box library.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable


@dataclass
class Gene:
    """One searchable parameter with inclusive bounds."""
    name: str
    low: float
    high: float
    integer: bool = False


def _coerce(gene: Gene, value: float) -> float:
    value = max(gene.low, min(gene.high, value))  # clamp to bounds
    return int(round(value)) if gene.integer else value


def _random_value(gene: Gene, rng: random.Random) -> float:
    return _coerce(gene, rng.uniform(gene.low, gene.high))


def random_genome(genes: list[Gene], rng: random.Random) -> dict:
    return {g.name: _random_value(g, rng) for g in genes}


def _crossover(a: dict, b: dict, genes: list[Gene], rng: random.Random) -> dict:
    return {g.name: (a if rng.random() < 0.5 else b)[g.name] for g in genes}


def _mutate(genome: dict, genes: list[Gene], rng: random.Random,
            rate: float, scale: float) -> dict:
    out = dict(genome)
    for g in genes:
        if rng.random() < rate:
            span = g.high - g.low
            out[g.name] = _coerce(g, out[g.name] + rng.gauss(0.0, scale * span))
    return out


def _tournament(scored: list[tuple[dict, float]], rng: random.Random, k: int = 3) -> dict:
    contenders = rng.sample(scored, min(k, len(scored)))
    return max(contenders, key=lambda pair: pair[1])[0]


def evolve(
    genes: list[Gene],
    fitness: Callable[[dict], float],
    *,
    pop_size: int = 20,
    generations: int = 15,
    rng: random.Random | None = None,
    mutation_rate: float = 0.3,
    mutation_scale: float = 0.1,
    elitism: int = 2,
) -> tuple[dict, float, list[float]]:
    """Run the GA and return (best_genome, best_fitness, best-per-generation history)."""
    rng = rng or random.Random()
    population = [random_genome(genes, rng) for _ in range(pop_size)]

    best_genome: dict | None = None
    best_fitness = float("-inf")
    history: list[float] = []

    for _ in range(generations):
        scored = [(g, fitness(g)) for g in population]
        scored.sort(key=lambda pair: pair[1], reverse=True)

        if scored[0][1] > best_fitness:
            best_genome, best_fitness = dict(scored[0][0]), scored[0][1]
        history.append(best_fitness)

        # carry the elite over unchanged, then fill with offspring
        nxt = [dict(scored[i][0]) for i in range(min(elitism, len(scored)))]
        while len(nxt) < pop_size:
            parent_a = _tournament(scored, rng)
            parent_b = _tournament(scored, rng)
            child = _crossover(parent_a, parent_b, genes, rng)
            child = _mutate(child, genes, rng, mutation_rate, mutation_scale)
            nxt.append(child)
        population = nxt

    return best_genome, best_fitness, history
