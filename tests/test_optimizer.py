"""Edge-case and invariant tests for the knapsack solvers.

Run with ``pytest tests/`` (pytest is optional; the module also runs standalone
via ``python tests/test_optimizer.py``). Nothing here needs torch or a model
download: the matrices are constructed by hand.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedder import (Embedder, HashingBackend, build_corpus,
                      cosine_similarity_matrix, l2_normalize)
from optimizer import ContextKnapsack

SEED = 42


def _knapsack(budget: int = 100, lambda_: float = 0.1, n: int = 5,
              seed: int = SEED) -> ContextKnapsack:
    rng = np.random.default_rng(seed)
    relevance = rng.uniform(0.1, 0.9, size=n)
    sim = rng.uniform(0.0, 0.8, size=(n, n))
    sim = 0.5 * (sim + sim.T)
    np.fill_diagonal(sim, 0.0)
    tokens = rng.integers(10, 40, size=n)
    return ContextKnapsack(relevance, sim, tokens, budget=budget, lambda_=lambda_)


# --- boundary conditions --------------------------------------------------- #
def test_empty_input():
    k = ContextKnapsack(np.zeros(0), np.zeros((0, 0)), np.zeros(0, dtype=int), budget=100)
    for result in k.solve_all(include_ilp=True).values():
        assert result.indices == []
        assert result.tokens_used == 0
        assert result.score == 0.0
        assert result.density == 0.0


def test_single_document_fits_and_overflows():
    fits = ContextKnapsack(np.array([0.7]), np.zeros((1, 1)), np.array([50]), budget=100)
    for result in fits.solve_all().values():
        assert result.indices == [0]

    too_big = ContextKnapsack(np.array([0.7]), np.zeros((1, 1)), np.array([500]), budget=100)
    assert too_big.infeasible_indices == [0]
    for result in too_big.solve_all().values():
        assert result.indices == []


def test_oversized_documents_are_never_selected():
    relevance = np.array([0.95, 0.4, 0.35])        # the best doc does not fit
    sim = np.zeros((3, 3))
    tokens = np.array([9_999, 30, 30])
    k = ContextKnapsack(relevance, sim, tokens, budget=100, lambda_=0.1)
    for method, result in k.solve_all().items():
        assert 0 not in result.indices, method
        assert result.tokens_used <= 100, method


def test_zero_budget():
    k = _knapsack(budget=0)
    for result in k.solve_all().values():
        assert result.indices == []
        assert result.budget_utilization == 0.0


def test_negative_similarity_is_clipped():
    relevance = np.array([0.6, 0.6])
    sim = np.array([[0.0, -0.9], [-0.9, 0.0]])
    k = ContextKnapsack(relevance, sim, np.array([10, 10]), budget=100, lambda_=1.0)
    assert (k.similarity >= 0).all()
    # With the penalty clipped to zero, both documents are worth selecting.
    assert sorted(k.solve_greedy().indices) == [0, 1]


def test_lambda_zero_matches_top_k_value():
    k = _knapsack(lambda_=0.0, budget=60)
    top_k = k.solve_top_k()
    greedy = k.solve_greedy()
    assert greedy.score >= top_k.score - 1e-9


# --- invariants ------------------------------------------------------------ #
def test_budget_is_always_respected():
    for budget in (20, 50, 100, 250):
        k = _knapsack(budget=budget, n=12)
        for method, result in k.solve_all().items():
            assert result.tokens_used <= budget, f"{method} overspent at {budget}"


def test_no_duplicate_selection():
    k = _knapsack(budget=200, n=12)
    for method, result in k.solve_all().items():
        assert len(set(result.indices)) == len(result.indices), method


def test_greedy_beats_top_k_on_redundant_corpus():
    # Docs 0-2 are near-duplicates; doc 3 is slightly less relevant but novel.
    relevance = np.array([0.90, 0.89, 0.88, 0.70])
    sim = np.array([
        [0.0, 0.95, 0.95, 0.05],
        [0.95, 0.0, 0.95, 0.05],
        [0.95, 0.95, 0.0, 0.05],
        [0.05, 0.05, 0.05, 0.0],
    ])
    tokens = np.array([25, 25, 25, 25])
    k = ContextKnapsack(relevance, sim, tokens, budget=75, lambda_=0.5)
    assert 3 in k.solve_greedy().indices
    assert k.solve_greedy().score > k.solve_top_k().score


def test_ilp_is_an_upper_bound_on_the_heuristics():
    k = _knapsack(budget=80, n=10)
    try:
        exact = k.solve_ilp(time_limit=60.0)
    except RuntimeError:
        return  # PuLP not installed; nothing to compare against
    if not exact.optimal:
        return
    for method, result in k.solve_all(include_ilp=False).items():
        assert result.score <= exact.score + 1e-6, method


def test_score_matches_manual_computation():
    relevance = np.array([0.5, 0.4])
    sim = np.array([[0.0, 0.3], [0.3, 0.0]])
    k = ContextKnapsack(relevance, sim, np.array([10, 10]), budget=100, lambda_=0.5)
    score, relevance_sum, redundancy = k.score([0, 1])
    assert np.isclose(relevance_sum, 0.9)
    assert np.isclose(redundancy, 0.3)
    assert np.isclose(score, 0.9 - 0.5 * 0.3)


def test_determinism():
    a = _knapsack(budget=90, n=15).solve_greedy().indices
    b = _knapsack(budget=90, n=15).solve_greedy().indices
    assert a == b


# --- embedder -------------------------------------------------------------- #
def test_cosine_matrix_properties():
    rng = np.random.default_rng(SEED)
    vectors = l2_normalize(rng.normal(size=(6, 16)))
    sim = cosine_similarity_matrix(vectors)
    assert sim.shape == (6, 6)
    assert np.allclose(sim, sim.T, atol=1e-6)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-5)
    assert (sim <= 1.0 + 1e-6).all() and (sim >= -1.0 - 1e-6).all()


def test_zero_vector_normalization_has_no_nan():
    assert not np.isnan(l2_normalize(np.zeros((2, 4)))).any()


def test_build_corpus_offline_backend():
    embedder = Embedder(backend=HashingBackend(dim=64, n_buckets=1024))
    corpus = build_corpus("attention in retrieval",
                          ["attention is all you need",
                           "attention is all you need",
                           "sourdough needs feeding"],
                          embedder)
    assert len(corpus) == 3
    assert corpus.similarity.shape == (3, 3)
    assert np.allclose(np.diag(corpus.similarity), 0.0)
    assert (corpus.tokens > 0).all()
    # Identical documents must be more similar than unrelated ones.
    assert corpus.similarity[0, 1] > corpus.similarity[0, 2]


def test_build_corpus_empty():
    corpus = build_corpus("q", [], Embedder(backend=HashingBackend(dim=32, n_buckets=256)))
    assert len(corpus) == 0
    assert corpus.similarity.shape == (0, 0)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
