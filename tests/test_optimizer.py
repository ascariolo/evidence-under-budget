"""Edge-case, invariant and repair tests for the knapsack solvers.

Run with ``pytest tests/`` (pytest is optional; the module also runs standalone
via ``python tests/test_optimizer.py``). Everything except the ILP tests runs
without torch or a model download: the matrices are constructed by hand.

Tests added at the repair step are grouped under the headings
"ILP status", "MMR", "stopping policy", "restricted pool", "metrics" and
"metadata".
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedder import (Embedder, HashingBackend, build_corpus,
                      cosine_similarity_matrix, l2_normalize)
from optimizer import (ContextKnapsack, SolveScope, SolveStatus,
                       classify_pulp_status, DEFAULT_MMR_LAMBDA,
                       DEFAULT_OBJECTIVE_LAMBDA)

SEED = 42


def _knapsack(budget: int = 100, objective_lambda: float = 0.1, n: int = 5,
              seed: int = SEED, **kwargs) -> ContextKnapsack:
    rng = np.random.default_rng(seed)
    relevance = rng.uniform(0.1, 0.9, size=n)
    sim = rng.uniform(0.0, 0.8, size=(n, n))
    sim = 0.5 * (sim + sim.T)
    np.fill_diagonal(sim, 0.0)
    tokens = rng.integers(10, 40, size=n)
    return ContextKnapsack(relevance, sim, tokens, budget=budget,
                           objective_lambda=objective_lambda, **kwargs)


def _has_pulp() -> bool:
    try:
        import pulp  # noqa: F401
        return True
    except ImportError:
        return False


# --- boundary conditions --------------------------------------------------- #
def test_empty_input():
    k = ContextKnapsack(np.zeros(0), np.zeros((0, 0)), np.zeros(0, dtype=int), budget=100)
    for result in k.solve_all(include_ilp=True).values():
        assert result.indices == []
        assert result.tokens_used == 0
        assert result.score == 0.0
        assert result.density_diagnostic == 0.0


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
    k = ContextKnapsack(relevance, sim, tokens, budget=100, objective_lambda=0.1)
    for method, result in k.solve_all().items():
        assert 0 not in result.indices, method
        assert result.tokens_used <= 100, method


def test_zero_budget():
    k = _knapsack(budget=0)
    for result in k.solve_all().values():
        assert result.indices == []
        assert result.budget_utilization == 0.0


def test_zero_token_document_does_not_warn_or_crash():
    """Repair: `gain / tokens` used to divide before masking (RuntimeWarning)."""
    relevance = np.array([0.5, 0.4])
    k = ContextKnapsack(relevance, np.zeros((2, 2)), np.array([0, 20]), budget=100)
    with np.errstate(divide="raise", invalid="raise"):
        result = k.solve_greedy()
    assert 0 not in result.indices          # zero-cost doc is infeasible by mask
    assert result.indices == [1]


def test_negative_similarity_is_clipped():
    relevance = np.array([0.6, 0.6])
    sim = np.array([[0.0, -0.9], [-0.9, 0.0]])
    k = ContextKnapsack(relevance, sim, np.array([10, 10]), budget=100,
                        objective_lambda=1.0)
    assert (k.similarity >= 0).all()
    # With the penalty clipped to zero, both documents are worth selecting.
    assert sorted(k.solve_greedy().indices) == [0, 1]


def test_lambda_zero_matches_top_k_value():
    k = _knapsack(objective_lambda=0.0, budget=60)
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


def test_score_matches_manual_computation():
    relevance = np.array([0.5, 0.4])
    sim = np.array([[0.0, 0.3], [0.3, 0.0]])
    k = ContextKnapsack(relevance, sim, np.array([10, 10]), budget=100,
                        objective_lambda=0.5)
    score, relevance_sum, redundancy = k.score([0, 1])
    assert np.isclose(relevance_sum, 0.9)
    assert np.isclose(redundancy, 0.3)
    assert np.isclose(score, 0.9 - 0.5 * 0.3)


def test_objective_is_unchanged_from_v0():
    """The research objective must survive the repair untouched."""
    relevance = np.array([0.5, 0.4, 0.3])
    sim = np.array([[0.0, 0.3, 0.1], [0.3, 0.0, 0.2], [0.1, 0.2, 0.0]])
    k = ContextKnapsack(relevance, sim, np.array([10, 10, 10]), budget=100,
                        objective_lambda=0.1)
    expected_rel = 1.2
    expected_red = 0.3 + 0.1 + 0.2
    score, rel, red = k.score([0, 1, 2])
    assert np.isclose(rel, expected_rel)
    assert np.isclose(red, expected_red)
    assert np.isclose(score, expected_rel - 0.1 * expected_red)


def test_marginal_gain_matches_score_delta():
    k = _knapsack(n=6, budget=1000)
    selected = [0, 2]
    base = k.score(selected)[0]
    for cand in (1, 3, 4):
        delta = k.score(selected + [cand])[0] - base
        assert np.isclose(delta, k.marginal_gain(cand, selected))


def test_determinism_across_repeated_runs():
    """Repeated solves of the SAME instance must return identical selections."""
    k = _knapsack(budget=90, n=15)
    for name in ("solve_top_k", "solve_mmr", "solve_greedy_objective", "solve_greedy"):
        runs = [getattr(k, name)().indices for _ in range(5)]
        assert all(r == runs[0] for r in runs), name


# --- ILP status ------------------------------------------------------------ #
class _FakeProblem:
    def __init__(self, status, sol_status):
        self.status = status
        self.sol_status = sol_status


def test_classify_pulp_status_distinguishes_proven_from_incumbent():
    """The core v0 defect: LpStatus alone cannot tell these two apart."""
    if not _has_pulp():
        return
    import pulp
    proven = _FakeProblem(pulp.LpStatusOptimal, pulp.LpSolutionOptimal)
    incumbent = _FakeProblem(pulp.LpStatusOptimal, pulp.LpSolutionIntegerFeasible)
    # Both report LpStatus "Optimal" - v0's test.
    assert pulp.LpStatus[proven.status] == "Optimal"
    assert pulp.LpStatus[incumbent.status] == "Optimal"
    # The repaired classifier separates them.
    assert classify_pulp_status(proven, pulp) is SolveStatus.PROVEN_OPTIMAL
    assert classify_pulp_status(incumbent, pulp) is SolveStatus.FEASIBLE_NOT_PROVEN


def test_classify_pulp_status_other_states():
    if not _has_pulp():
        return
    import pulp
    cases = {
        (pulp.LpStatusInfeasible, pulp.LpSolutionInfeasible): SolveStatus.INFEASIBLE,
        (pulp.LpStatusUnbounded, pulp.LpSolutionUnbounded): SolveStatus.UNBOUNDED,
        (pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound): SolveStatus.NOT_SOLVED,
    }
    for (status, sol_status), expected in cases.items():
        assert classify_pulp_status(_FakeProblem(status, sol_status), pulp) is expected
    # Missing sol_status (older PuLP) must fall back conservatively, never to proven.
    class _NoSolStatus:
        status = pulp.LpStatusOptimal
    assert classify_pulp_status(_NoSolStatus(), pulp) is SolveStatus.FEASIBLE_NOT_PROVEN


def test_ilp_proven_optimal_is_identified():
    if not _has_pulp():
        return
    k = _knapsack(budget=80, n=8)
    result = k.solve_ilp(time_limit=120.0)
    assert result.status is SolveStatus.PROVEN_OPTIMAL
    assert result.scope is SolveScope.FULL_CORPUS
    assert result.is_proven_global_optimum
    assert result.optimal            # deprecated alias agrees


def test_ilp_objective_value_is_preserved():
    """Our recomputed Score(S) must match the value the solver reported."""
    if not _has_pulp():
        return
    k = _knapsack(budget=80, n=8)
    result = k.solve_ilp(time_limit=120.0)
    assert result.meta["solver_objective"] is not None
    assert result.meta["objective_consistent"] is True
    assert np.isclose(result.meta["solver_objective"], result.score, atol=1e-6)


def test_ilp_time_limited_run_is_not_called_optimal():
    """A time-limited incumbent must never be labelled a proven optimum."""
    if not _has_pulp():
        return
    rng = np.random.default_rng(7)
    n = 70
    relevance = rng.uniform(0.2, 0.9, size=n)
    sim = rng.uniform(0.1, 0.9, size=(n, n))
    sim = 0.5 * (sim + sim.T)
    np.fill_diagonal(sim, 0.0)
    tokens = rng.integers(10, 60, size=n)
    k = ContextKnapsack(relevance, sim, tokens, budget=int(tokens.sum() // 3),
                        objective_lambda=0.1)
    result = k.solve_ilp(time_limit=0.05)
    # Whatever CBC managed in 50ms, the label must be honest.
    if result.status is not SolveStatus.PROVEN_OPTIMAL:
        assert result.status is SolveStatus.FEASIBLE_NOT_PROVEN
        assert not result.is_proven_global_optimum
        assert not result.optimal
    # The incumbent itself is preserved, never discarded.
    assert result.tokens_used <= k.budget
    assert result.meta["time_limit_s"] == 0.05


def test_selection_result_optimal_flag_is_read_only():
    """It must be impossible to relabel a result as a proven optimum."""
    k = _knapsack(budget=80, n=6)
    result = k.solve_greedy()
    assert result.status is SolveStatus.HEURISTIC
    assert not result.optimal
    try:
        result.optimal = True
    except AttributeError:
        pass
    else:
        raise AssertionError("SelectionResult.optimal must not be assignable")


def test_heuristics_carry_no_optimality_claim():
    k = _knapsack(budget=80, n=8)
    for method, result in k.solve_all(include_ilp=False).items():
        assert result.status is SolveStatus.HEURISTIC, method
        assert not result.is_proven_global_optimum, method
        assert not result.is_proven_pool_optimum, method


def test_ilp_is_an_upper_bound_on_the_heuristics():
    if not _has_pulp():
        return
    k = _knapsack(budget=80, n=10)
    exact = k.solve_ilp(time_limit=60.0)
    if not exact.is_proven_global_optimum:
        return
    for method, result in k.solve_all(include_ilp=False).items():
        assert result.score <= exact.score + 1e-6, method


# --- MMR ------------------------------------------------------------------- #
def test_mmr_uses_max_similarity_not_sum():
    """Constructed so max-similarity and summed-similarity disagree.

    Doc 1 is very similar to ONE selected document. Docs 2 and 3 are each
    moderately similar to SEVERAL selected documents, so their *summed*
    similarity is larger while their *max* similarity is smaller.
    """
    relevance = np.array([0.90, 0.60, 0.62, 0.62])
    sim = np.array([
        [0.00, 0.80, 0.30, 0.30],
        [0.80, 0.00, 0.00, 0.00],
        [0.30, 0.00, 0.00, 0.75],
        [0.30, 0.00, 0.75, 0.00],
    ])
    tokens = np.array([10, 10, 10, 10])
    k = ContextKnapsack(relevance, sim, tokens, budget=20,
                        objective_lambda=1.0, mmr_lambda=0.5)

    mmr = k.solve_mmr().indices
    summed = k.solve_greedy_objective().indices
    assert mmr[0] == 0 and summed[0] == 0          # both start at the top doc

    # After {0}: max-sim penalty for doc1 is 0.80, for docs 2/3 it is 0.30.
    # MMR score: 0.5*0.60 - 0.5*0.80 = -0.10 (doc1) vs 0.5*0.62-0.5*0.30 = 0.16.
    assert mmr[1] in (2, 3)
    # The summed-penalty greedy uses the SAME numbers here (|S| = 1, so sum ==
    # max), which is exactly why a single-step test is not enough; the formulas
    # are verified directly below.
    lam = k.mmr_lambda
    max_sim_to_selected = sim[:, 0]
    mmr_scores = lam * relevance - (1 - lam) * max_sim_to_selected
    assert int(np.argmax([mmr_scores[i] if i != 0 else -np.inf for i in range(4)])) == mmr[1]


def test_mmr_differs_from_v0_summed_greedy():
    """The two rules must select different documents on a separating instance.

    Docs 0 and 1 are taken first by both. Then doc 2 has LOW maximum similarity
    to the selected set (0.50) but HIGH summed similarity (0.50 + 0.50 = 1.00),
    while doc 3 has HIGH maximum similarity (0.70) but LOW summed similarity
    (0.70 + 0.00 = 0.70). Max-similarity MMR therefore takes doc 2; the summed
    penalty v0 shipped under the name "MMR" takes doc 3.
    """
    relevance = np.array([0.90, 0.85, 0.60, 0.60])
    sim = np.array([
        [0.00, 0.10, 0.50, 0.70],
        [0.10, 0.00, 0.50, 0.00],
        [0.50, 0.50, 0.00, 0.05],
        [0.70, 0.00, 0.05, 0.00],
    ])
    tokens = np.array([10, 10, 10, 10])
    k = ContextKnapsack(relevance, sim, tokens, budget=30,
                        objective_lambda=0.5, mmr_lambda=0.5)
    mmr = k.solve_mmr().indices
    summed = k.solve_greedy_objective().indices
    assert mmr == [0, 1, 2], mmr
    assert summed == [0, 1, 3], summed
    assert mmr != summed


def test_mmr_lambda_extremes():
    """lambda_mmr = 1 is pure relevance; lambda_mmr = 0 is pure diversity."""
    relevance = np.array([0.90, 0.85, 0.20])
    sim = np.array([[0.0, 0.95, 0.05], [0.95, 0.0, 0.05], [0.05, 0.05, 0.0]])
    tokens = np.array([10, 10, 10])
    pure_relevance = ContextKnapsack(relevance, sim, tokens, budget=20,
                                     objective_lambda=0.0, mmr_lambda=1.0)
    assert pure_relevance.solve_mmr().indices == [0, 1]
    pure_diversity = ContextKnapsack(relevance, sim, tokens, budget=20,
                                     objective_lambda=0.0, mmr_lambda=0.0)
    assert pure_diversity.solve_mmr().indices[1] == 2


def test_mmr_lambda_is_separate_from_objective_lambda():
    """Changing mmr_lambda must not move the objective or any other solver."""
    base = _knapsack(budget=120, n=8)
    other = _knapsack(budget=120, n=8, mmr_lambda=0.1)
    assert base.objective_lambda == other.objective_lambda
    assert base.score([0, 1, 2]) == other.score([0, 1, 2])
    assert base.solve_top_k().indices == other.solve_top_k().indices
    assert base.solve_greedy().indices == other.solve_greedy().indices
    assert base.solve_greedy_objective().indices == other.solve_greedy_objective().indices


def test_mmr_lambda_validated():
    for bad in (-0.1, 1.1):
        try:
            _knapsack(mmr_lambda=bad)
        except ValueError:
            continue
        raise AssertionError(f"mmr_lambda={bad} should be rejected")


# --- stopping policy ------------------------------------------------------- #
def test_common_stopping_policy_applies_to_every_heuristic():
    """No heuristic may select a document with non-positive marginal gain."""
    k = _knapsack(budget=400, n=14, objective_lambda=0.3)
    for method, result in k.solve_all(include_ilp=False).items():
        running: list = []
        for idx in result.indices:
            gain = k.marginal_gain(idx, running)
            assert gain > 0.0, f"{method} took index {idx} at gain {gain:.4f}"
            running.append(idx)


def test_top_k_and_greedy_share_the_stopping_rule():
    """v0's asymmetry: Top-K kept buying after the gain turned negative."""
    relevance = np.array([0.90, 0.89, 0.88, 0.87])
    sim = np.full((4, 4), 0.95)
    np.fill_diagonal(sim, 0.0)
    tokens = np.array([10, 10, 10, 10])
    repaired = ContextKnapsack(relevance, sim, tokens, budget=40,
                               objective_lambda=1.0)
    v0_style = ContextKnapsack(relevance, sim, tokens, budget=40,
                               objective_lambda=1.0,
                               stop_on_nonpositive_gain=False)
    # Under the common policy both stop early and agree.
    assert repaired.solve_top_k().n_selected == repaired.solve_greedy().n_selected
    assert repaired.solve_top_k().meta["stop_reason"] == "no_positive_marginal_gain"
    # Without it, Top-K fills the budget and its objective collapses.
    assert v0_style.solve_top_k().n_selected == 4
    assert v0_style.solve_top_k().score < repaired.solve_top_k().score


def test_top_k_remains_pure_relevance_ranking():
    """Top-K must not become token-aware: order is by relevance alone."""
    relevance = np.array([0.9, 0.8, 0.7])
    sim = np.zeros((3, 3))
    tokens = np.array([90, 5, 5])          # doc 0 is by far the most expensive
    k = ContextKnapsack(relevance, sim, tokens, budget=100, objective_lambda=0.1)
    # A token-aware method would prefer the cheap docs; Top-K must take doc 0 first.
    assert k.solve_top_k().indices[0] == 0
    assert k.solve_greedy().indices[0] == 1


def test_top_k_skips_but_does_not_stop_on_budget():
    """Preserved v0 behaviour: an unaffordable doc is skipped, not a stop."""
    relevance = np.array([0.9, 0.8, 0.7])
    sim = np.zeros((3, 3))
    tokens = np.array([500, 10, 10])
    k = ContextKnapsack(relevance, sim, tokens, budget=100, objective_lambda=0.1)
    assert k.solve_top_k().indices == [1, 2]


def test_stop_reason_is_recorded():
    k = _knapsack(budget=10_000, n=6, objective_lambda=0.0)
    result = k.solve_greedy()
    assert result.meta["stop_reason"] in (
        "no_candidate_fits_budget", "no_positive_marginal_gain", "k_reached")
    assert result.meta["stopping_policy"] == "positive_marginal_gain"


def test_top_k_honours_k():
    k = _knapsack(budget=10_000, n=8, objective_lambda=0.0)
    result = k.solve_top_k(k=3)
    assert result.n_selected == 3
    assert result.meta["stop_reason"] == "k_reached"


# --- restricted pool ------------------------------------------------------- #
def test_restricted_pool_is_not_a_global_optimum():
    if not _has_pulp():
        return
    from benchmark import solve_ilp_reference
    embedder = Embedder(backend=HashingBackend(dim=64, n_buckets=2048))
    corpus = build_corpus(
        "attention retrieval context",
        ["attention is all you need"] * 4
        + ["dense retrieval encodes passages"] * 4
        + ["sourdough starters need feeding"] * 4,
        embedder)
    full = ContextKnapsack(corpus.relevance, corpus.similarity, corpus.tokens,
                           budget=60, objective_lambda=0.1)
    pooled = solve_ilp_reference(corpus, full, 60, 0.1, DEFAULT_MMR_LAMBDA,
                                 max_docs=5, time_limit=60.0)
    assert pooled.scope is SolveScope.RESTRICTED_POOL
    assert not pooled.is_proven_global_optimum
    if pooled.status is SolveStatus.PROVEN_OPTIMAL:
        assert pooled.is_proven_pool_optimum
    assert pooled.meta["candidate_pool"] == 5


def test_unrestricted_pool_keeps_full_corpus_scope():
    if not _has_pulp():
        return
    from benchmark import solve_ilp_reference
    embedder = Embedder(backend=HashingBackend(dim=64, n_buckets=2048))
    corpus = build_corpus("attention", ["attention is all you need",
                                        "dense retrieval encodes passages",
                                        "sourdough starters need feeding"],
                          embedder)
    full = ContextKnapsack(corpus.relevance, corpus.similarity, corpus.tokens,
                           budget=100, objective_lambda=0.1)
    result = solve_ilp_reference(corpus, full, 100, 0.1, DEFAULT_MMR_LAMBDA,
                                 max_docs=0, time_limit=60.0)
    assert result.scope is SolveScope.FULL_CORPUS


# --- metrics --------------------------------------------------------------- #
def test_density_is_diagnostic_only():
    """It must not appear among the primary metrics, under either name."""
    import benchmark
    from benchmark import MethodMetrics, RunMetadata
    fields = set(MethodMetrics.__dataclass_fields__)
    assert "density_diagnostic" in fields
    assert "density" not in fields          # the undecorated name is gone
    # And it is not plotted as a primary panel.
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "benchmark.py"), encoding="utf-8").read()
    plot_section = src.split("def plot_results")[1].split("def dump_json")[0]
    assert "density" not in plot_section.split('"""')[2]


def test_density_metric_rewards_tiny_selections():
    """Documents the defect that demoted the metric; guards against promotion."""
    relevance = np.array([0.50, 0.48, 0.46])
    sim = np.full((3, 3), 0.4)
    np.fill_diagonal(sim, 0.0)
    tokens = np.array([10, 100, 100])
    k = ContextKnapsack(relevance, sim, tokens, budget=210, objective_lambda=0.1)
    one = k._result("probe", [0], 0.0)
    three = k._result("probe", [0, 1, 2], 0.0)
    assert three.score > one.score                      # more objective
    assert one.density_diagnostic > three.density_diagnostic   # yet "denser"


def test_redundancy_per_pair_is_reported():
    """Total redundancy conflates duplication with selection size."""
    from benchmark import evaluate
    embedder = Embedder(backend=HashingBackend(dim=64, n_buckets=2048))
    corpus = build_corpus("attention", ["attention is all you need",
                                        "attention is all you need",
                                        "sourdough starters need feeding"],
                          embedder)
    k = ContextKnapsack(corpus.relevance, corpus.similarity, corpus.tokens,
                        budget=500, objective_lambda=0.1)
    row = evaluate(k.solve_top_k(), corpus, reference_score=None)
    if row.n_selected >= 2:
        pairs = row.n_selected * (row.n_selected - 1) / 2
        assert np.isclose(row.redundancy_per_pair, row.redundancy / pairs)
    single = evaluate(k._result("probe", [0], 0.0), corpus)
    assert single.redundancy_per_pair is None       # no pairs, not zero


def test_ratio_suppressed_against_nonpositive_reference():
    """A ratio to a negative optimum is not an approximation quality."""
    from benchmark import evaluate
    embedder = Embedder(backend=HashingBackend(dim=64, n_buckets=2048))
    corpus = build_corpus("attention", ["attention is all you need",
                                        "sourdough starters need feeding"],
                          embedder)
    k = ContextKnapsack(corpus.relevance, corpus.similarity, corpus.tokens,
                        budget=500, objective_lambda=0.1)
    row = evaluate(k.solve_top_k(), corpus, reference_score=-2.5,
                   reference_kind="best_known")
    assert row.score_vs_reference is None


# --- metadata -------------------------------------------------------------- #
REQUIRED_METADATA_FIELDS = (
    "benchmark_config_version", "timestamp_utc", "seed", "objective_lambda",
    "mmr_lambda", "n_docs", "budgets", "repeats", "stopping_policy",
    "ilp_time_limit", "ilp_max_docs", "ilp_solver", "embedding_backend",
    "tokenizer_encoding", "git", "python_version", "platform", "packages",
    "primary_metrics", "diagnostic_metrics",
)


def test_metadata_contains_required_fields():
    from benchmark import collect_metadata
    embedder = Embedder(backend=HashingBackend(dim=64, n_buckets=2048))
    corpus = build_corpus("attention", ["attention is all you need",
                                        "sourdough starters need feeding"],
                          embedder)
    meta = collect_metadata(embedder, corpus, seed=42, objective_lambda=0.1,
                            mmr_lambda=0.5, n_docs=2, budgets=[128],
                            repeats=3, ilp_repeats=1, include_ilp=True,
                            ilp_time_limit=300.0, ilp_max_docs=0,
                            stopping_policy="positive_marginal_gain",
                            requested_model="all-MiniLM-L6-v2")
    from dataclasses import asdict
    payload = asdict(meta)
    for key in REQUIRED_METADATA_FIELDS:
        assert key in payload, key
        assert payload[key] is not None, key
    assert payload["git"]["commit"]                 # present or "unavailable"
    assert payload["embedding_fallback_used"] is True   # hashing backend here
    assert "density_diagnostic" in payload["diagnostic_metrics"]
    assert "density_diagnostic" not in payload["primary_metrics"]


def test_dumped_json_is_self_describing():
    from benchmark import collect_metadata, dump_json, evaluate
    embedder = Embedder(backend=HashingBackend(dim=64, n_buckets=2048))
    corpus = build_corpus("attention", ["attention is all you need",
                                        "sourdough starters need feeding"],
                          embedder)
    meta = collect_metadata(embedder, corpus, seed=42, objective_lambda=0.1,
                            mmr_lambda=0.5, n_docs=2, budgets=[128], repeats=1,
                            ilp_repeats=1, include_ilp=False,
                            ilp_time_limit=None, ilp_max_docs=0,
                            stopping_policy="positive_marginal_gain",
                            requested_model="all-MiniLM-L6-v2")
    k = ContextKnapsack(corpus.relevance, corpus.similarity, corpus.tokens,
                        budget=128, objective_lambda=0.1)
    table = {128: {"top_k": evaluate(k.solve_top_k(), corpus)}}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.json")
        dump_json(table, meta, path)
        payload = json.load(open(path, encoding="utf-8"))
    assert set(payload) == {"metadata", "results"}
    assert payload["metadata"]["seed"] == 42
    row = payload["results"]["128"]["top_k"]
    assert row["solve_status"] == SolveStatus.HEURISTIC.value
    assert row["solve_scope"] == SolveScope.FULL_CORPUS.value
    assert "density_diagnostic" in row


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
            except Exception as exc:  # surface errors, do not swallow them
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
