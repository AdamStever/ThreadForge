"""Tests for the genetic algorithm and the tuning decode."""

import random

from threadforge.optimization.genetic import Gene, evolve, random_genome
from threadforge.optimization.tuning import decode, SEARCH_SPACE


def test_random_genome_within_bounds_and_integer():
    genes = [Gene("a", -5, 5), Gene("k", 0, 10, integer=True)]
    rng = random.Random(0)
    for _ in range(50):
        g = random_genome(genes, rng)
        assert -5 <= g["a"] <= 5
        assert 0 <= g["k"] <= 10
        assert isinstance(g["k"], int)


def test_evolve_finds_known_optimum():
    # maximize -(x-3)^2 - (y+1)^2, optimum at (3, -1) with fitness 0
    genes = [Gene("x", -10, 10), Gene("y", -10, 10)]

    def fitness(g):
        return -((g["x"] - 3) ** 2) - ((g["y"] + 1) ** 2)

    rng = random.Random(42)
    best, best_fit, history = evolve(genes, fitness, pop_size=30, generations=40, rng=rng)

    assert abs(best["x"] - 3) < 0.5
    assert abs(best["y"] + 1) < 0.5
    assert best_fit > -0.5
    # best-so-far fitness is monotonically non-decreasing across generations
    assert history == sorted(history)


def test_evolve_is_deterministic_with_seed():
    genes = [Gene("x", -10, 10)]
    fitness = lambda g: -((g["x"] - 2) ** 2)
    a = evolve(genes, fitness, pop_size=10, generations=5, rng=random.Random(1))
    b = evolve(genes, fitness, pop_size=10, generations=5, rng=random.Random(1))
    assert a[0] == b[0] and a[1] == b[1]


def test_integer_gene_stays_integer_through_evolution():
    genes = [Gene("k", 0, 20, integer=True)]
    fitness = lambda g: -abs(g["k"] - 7)
    best, _, _ = evolve(genes, fitness, pop_size=12, generations=15, rng=random.Random(3))
    assert isinstance(best["k"], int)
    assert abs(best["k"] - 7) <= 1


def test_decode_maps_genome_to_hyperparams():
    hp = decode({"log_C": 2.0, "threshold": 0.4})
    assert abs(hp["C"] - 100.0) < 1e-9
    assert hp["threshold"] == 0.4


def test_search_space_shape():
    names = {g.name for g in SEARCH_SPACE}
    assert names == {"log_C", "threshold"}
