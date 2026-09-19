"""Invariant tests for the LOCKED v7 final-validation protocol.

Each test pins one locked decision. A failure here means the implementation
drifted from ``docs/final_validation_protocol.md`` - the protocol is never the
thing that gets adjusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimizer import ContextKnapsack  # noqa: E402

from experiments.final_validation import analyze as A  # noqa: E402
from experiments.final_validation import data as D  # noqa: E402
from experiments.final_validation import protocol as P  # noqa: E402
from experiments.final_validation import run_experiment as R  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _FakeCounter:
    """Token counter stand-in: one token per whitespace-separated word."""

    def count_batch(self, texts):
        return np.array([len(t.split()) for t in texts], dtype=np.int64)


def _row(title, sentences):
    return {"title": title, "sentences": sentences}


def _raw_row(qid, titles, paragraphs, gold_titles, gold_ids):
    return {
        "id": qid, "question": "q?", "answer": "a", "type": "comparison", "level": "hard",
        "supporting_facts": {"title": list(gold_titles), "sent_id": list(gold_ids)},
        "context": {"title": list(titles), "sentences": [list(p) for p in paragraphs]},
    }


def _question(tokens, gold_indices, n_gold=None):
    tokens = np.asarray(tokens, dtype=np.int64)
    return D.Question(
        qid="q", question="query", sentences=tuple(f"s{i}" for i in range(tokens.size)),
        sentence_keys=tuple(("t", i) for i in range(tokens.size)), tokens=tokens,
        gold_indices=tuple(gold_indices),
        n_gold_annotated=len(gold_indices) if n_gold is None else n_gold,
        n_gold_unreachable=0)


def _instance(n=12, seed=0, budget=40):
    rng = np.random.default_rng(seed)
    relevance = rng.normal(0.3, 0.2, size=n)
    vectors = rng.normal(size=(n, 8))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = vectors @ vectors.T
    tokens = rng.integers(3, 20, size=n).astype(np.int64)
    return ContextKnapsack(relevance, similarity, tokens, budget,
                           objective_lambda=P.LAMBDA_OBJ, mmr_lambda=P.LAMBDA_MMR)


# --------------------------------------------------------------------------- #
# 1. Protocol hash lock
# --------------------------------------------------------------------------- #
def test_protocol_hash_matches_lock_record():
    assert P.assert_locked() == P.PROTOCOL_SHA256
    assert P.protocol_hash() == P.PROTOCOL_SHA256


def test_altered_protocol_is_refused(tmp_path, monkeypatch):
    tampered = tmp_path / "final_validation_protocol.md"
    tampered.write_text(P.PROTOCOL_DOC.read_text() + "\nsneaky amendment\n")
    monkeypatch.setattr(P, "PROTOCOL_DOC", tampered)
    with pytest.raises(P.ProtocolViolation):
        P.assert_locked()


def test_frozen_constants_match_the_locked_document():
    constants = P.frozen_constants()
    assert constants["n_questions"] == 500
    assert constants["rho_levels"] == [0.10, 0.25, 0.50]
    assert constants["sample_seed"] == 7000
    assert constants["bootstrap_seed"] == 7001
    assert constants["permutation_seed"] == 7002
    assert constants["lambda_obj"] == 0.1 and constants["lambda_mmr"] == 0.5
    assert constants["sesoi"] is None
    assert constants["sidedness"] == "two-sided"
    assert len(constants["arms"]) == 6


# --------------------------------------------------------------------------- #
# 2. Supporting-fact coverage
# --------------------------------------------------------------------------- #
def test_coverage_is_gold_hits_over_annotated_gold():
    question = _question(tokens=[5, 5, 5, 5], gold_indices=[1, 3])
    assert R.coverage([1, 3], question) == (1.0, 2)
    assert R.coverage([1], question) == (0.5, 1)
    assert R.coverage([0, 2], question) == (0.0, 0)
    assert R.coverage([], question) == (0.0, 0)


def test_coverage_denominator_counts_unreachable_gold():
    """An unreachable gold sentence lowers every arm equally, never Delta_C."""
    question = _question(tokens=[5, 5], gold_indices=[0], n_gold=2)
    value, hit = R.coverage([0], question)
    assert hit == 1 and value == 0.5


def test_coverage_ignores_duplicate_indices():
    question = _question(tokens=[5, 5], gold_indices=[0])
    assert R.coverage([0, 0], question) == (1.0, 1)


# --------------------------------------------------------------------------- #
# 3. Identical candidate pool across arms
# --------------------------------------------------------------------------- #
def test_every_arm_sees_the_same_untouched_pool():
    knapsack = _instance()
    relevance = knapsack.relevance.copy()
    similarity = knapsack.similarity.copy()
    tokens = knapsack.tokens.copy()
    budget = knapsack.budget
    for arm in P.ARMS:
        R.run_arm(arm, knapsack)
        assert np.array_equal(knapsack.relevance, relevance)
        assert np.array_equal(knapsack.similarity, similarity)
        assert np.array_equal(knapsack.tokens, tokens)
        assert knapsack.budget == budget


# --------------------------------------------------------------------------- #
# 4. Budget is never exceeded
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(6))
def test_no_arm_exceeds_the_budget(seed):
    knapsack = _instance(seed=seed, budget=35)
    for arm in P.ARMS:
        record = R.run_arm(arm, knapsack)
        assert record["tokens_used"] <= knapsack.budget


def test_zero_budget_selects_nothing():
    knapsack = _instance(budget=0)
    for arm in P.ARMS:
        assert R.run_arm(arm, knapsack)["indices"] == []


# --------------------------------------------------------------------------- #
# 5. Deterministic tie-breaking
# --------------------------------------------------------------------------- #
def test_ties_go_to_the_lowest_index():
    relevance = np.array([0.5, 0.5, 0.5])
    similarity = np.zeros((3, 3))
    tokens = np.array([10, 10, 10], dtype=np.int64)
    knapsack = ContextKnapsack(relevance, similarity, tokens, budget=10,
                               objective_lambda=P.LAMBDA_OBJ)
    for arm in (P.ARM_TA, P.ARM_GO, P.ARM_TOP_K, P.ARM_MMR):
        assert R.run_arm(arm, knapsack)["indices"] == [0]


def test_reruns_are_bit_identical():
    knapsack = _instance(seed=3)
    for arm in P.ARMS:
        first = R.run_arm(arm, knapsack)
        second = R.run_arm(arm, knapsack)
        assert first["indices"] == second["indices"]
        assert first["score"] == second["score"]


# --------------------------------------------------------------------------- #
# 6. Common stopping rule
# --------------------------------------------------------------------------- #
def test_ta_and_go_stop_on_nonpositive_marginal_gain():
    """Ample budget, but the third sentence can only lower the objective."""
    relevance = np.array([1.0, 1.0, 0.01])
    similarity = np.array([[0.0, 0.0, 1.0],
                           [0.0, 0.0, 1.0],
                           [1.0, 1.0, 0.0]])
    tokens = np.array([5, 5, 1], dtype=np.int64)
    knapsack = ContextKnapsack(relevance, similarity, tokens, budget=100,
                               objective_lambda=1.0)
    for arm in (P.ARM_TA, P.ARM_GO):
        indices = R.run_arm(arm, knapsack)["indices"]
        assert 2 not in indices
        assert set(indices) == {0, 1}


def test_nonpositive_relevance_is_never_bought():
    relevance = np.array([0.5, -0.2])
    similarity = np.zeros((2, 2))
    tokens = np.array([5, 1], dtype=np.int64)
    knapsack = ContextKnapsack(relevance, similarity, tokens, budget=100,
                               objective_lambda=P.LAMBDA_OBJ)
    for arm in (P.ARM_TA, P.ARM_GO):
        assert R.run_arm(arm, knapsack)["indices"] == [0]


# --------------------------------------------------------------------------- #
# 7. No gold-label leakage
# --------------------------------------------------------------------------- #
def test_selection_is_invariant_to_the_gold_labels():
    knapsack = _instance(seed=5, budget=30)
    baseline = {arm: R.run_arm(arm, knapsack)["indices"] for arm in P.ARMS}
    # Relabelling gold cannot reach the solvers: they never receive a Question.
    for gold in ([], [0], [1, 2, 3], list(range(knapsack.n))):
        question = _question(tokens=knapsack.tokens, gold_indices=gold,
                             n_gold=max(len(gold), 1))
        for arm in P.ARMS:
            record = R.run_arm(arm, knapsack)
            assert record["indices"] == baseline[arm]
            R.coverage(record["indices"], question)       # scoring happens after


def test_run_arm_signature_excludes_gold():
    import inspect

    parameters = set(inspect.signature(R.run_arm).parameters)
    assert parameters == {"arm", "knapsack"}
    assert "gold" not in inspect.getsource(R.run_arm)


# --------------------------------------------------------------------------- #
# 8. W_pool definition
# --------------------------------------------------------------------------- #
def test_w_pool_is_the_pool_token_sum_and_budget_is_its_floor():
    question = _question(tokens=[7, 11, 2], gold_indices=[0])
    assert question.w_pool == 20
    assert R.budget_for(question, 0.10) == 2      # floor(2.0)
    assert R.budget_for(question, 0.25) == 5      # floor(5.0)
    assert R.budget_for(question, 0.50) == 10


def test_w_pool_does_not_depend_on_arm_or_budget():
    question = _question(tokens=[9, 9, 9], gold_indices=[2])
    before = question.w_pool
    knapsack = ContextKnapsack(np.array([0.4, 0.3, 0.2]), np.zeros((3, 3)),
                               question.tokens, budget=R.budget_for(question, 0.50),
                               objective_lambda=P.LAMBDA_OBJ)
    for arm in P.ARMS:
        R.run_arm(arm, knapsack)
    assert question.w_pool == before


# --------------------------------------------------------------------------- #
# 9. Holm family of exactly three
# --------------------------------------------------------------------------- #
def test_holm_family_has_exactly_three_members():
    assert P.CONFIRMATORY_FAMILY_SIZE == 3 == len(P.RHO_LEVELS)
    adjusted = A.holm([0.01, 0.02, 0.04])
    assert len(adjusted) == 3
    assert [round(a["p_holm"], 10) for a in adjusted] == [0.03, 0.04, 0.04]


def test_holm_is_monotone_and_capped():
    adjusted = A.holm([0.5, 0.6, 0.9])
    values = [a["p_holm"] for a in adjusted]
    assert values == sorted(values)
    assert max(values) <= 1.0


def test_signflip_is_two_sided_and_retains_zeros():
    rng = np.random.default_rng(0)
    differences = np.concatenate([rng.normal(0.0, 0.1, 40), np.zeros(60)])
    observed, p_value = A.signflip_pvalue(differences, flips=500, seed=P.PERMUTATION_SEED)
    assert 0.0 < p_value <= 1.0
    mirrored, p_mirrored = A.signflip_pvalue(-differences, flips=500,
                                             seed=P.PERMUTATION_SEED)
    assert mirrored == pytest.approx(-observed)
    assert p_mirrored == pytest.approx(p_value)


def test_signflip_is_deterministic_under_the_frozen_seed():
    values = np.array([0.1, -0.2, 0.0, 0.3, 0.0])
    first = A.signflip_pvalue(values, flips=200, seed=P.PERMUTATION_SEED)
    second = A.signflip_pvalue(values, flips=200, seed=P.PERMUTATION_SEED)
    assert first == second


def test_effect_statistics_behave():
    differences = np.array([0.5, 0.5, 0.0, -0.5])
    assert A.rank_biserial(differences) == pytest.approx(1.0 / 3.0)
    assert A.rank_biserial(np.zeros(5)) == 0.0
    assert np.isnan(A.cohen_dz(np.array([0.2])))
    summary = A.distribution(differences)
    assert summary["wins_ta"] == 2 and summary["losses_ta"] == 1 and summary["ties"] == 1


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    values = np.random.default_rng(1).normal(0.05, 0.2, 200)
    low, high = A.bootstrap_ci(values, resamples=500, seed=P.BOOTSTRAP_SEED)
    assert low < values.mean() < high
    assert (low, high) == A.bootstrap_ci(values, resamples=500, seed=P.BOOTSTRAP_SEED)


# --------------------------------------------------------------------------- #
# 10. Deterministic sampling and exclusions
# --------------------------------------------------------------------------- #
def test_frozen_sample_is_reproducible_and_seed_dependent():
    pool = [_question(tokens=[3], gold_indices=[0]) for _ in range(50)]
    pool = [D.Question(qid=str(i), question="q", sentences=q.sentences,
                       sentence_keys=q.sentence_keys, tokens=q.tokens,
                       gold_indices=q.gold_indices, n_gold_annotated=1,
                       n_gold_unreachable=0)
            for i, q in enumerate(pool)]
    first = [q.qid for q in D.frozen_sample(pool, n=10, seed=P.SAMPLE_SEED)]
    second = [q.qid for q in D.frozen_sample(pool, n=10, seed=P.SAMPLE_SEED)]
    other = [q.qid for q in D.frozen_sample(pool, n=10, seed=P.SAMPLE_SEED + 1)]
    assert first == second
    assert first != other
    assert len(set(first)) == 10


def test_exclusions_are_applied_counted_and_never_resampled():
    rows = [
        _raw_row("keep", ["A", "B"], [["one two", "three four"], ["five six"]],
                 ["A", "B"], [0, 0]),
        _raw_row("missing_fact", ["A"], [["one two"]], ["A"], [7]),
        _raw_row("unknown_title", ["A"], [["one two"]], ["ZZ"], [0]),
        _raw_row("empty_pool", ["A"], [[""]], ["A"], [0]),
    ]
    questions, ledger = D.build_questions(rows, _FakeCounter())
    assert [q.qid for q in questions] == ["keep"]
    assert ledger["rows_total"] == 4
    assert ledger["excluded_missing_supporting_fact"] == 2
    assert ledger["excluded_empty_pool"] == 1
    assert ledger["questions_eligible"] == 1
    # Fewer eligible questions than requested: we stop, we do not resample.
    assert len(D.frozen_sample(questions, n=10, seed=P.SAMPLE_SEED)) == 1


def test_zero_token_sentences_leave_the_pool_and_are_counted():
    rows = [_raw_row("q", ["A"], [["alpha beta", "", "gamma"]], ["A"], [0])]
    questions, ledger = D.build_questions(rows, _FakeCounter())
    question = questions[0]
    assert question.n == 2                                   # the empty sentence is gone
    assert ledger["sentences_dropped_zero_tokens"] == 1
    assert all(t > 0 for t in question.tokens)


def test_gold_parsing_maps_to_pool_positions():
    rows = [_raw_row("q", ["A", "B"], [["a b", "c d"], ["e f"]], ["B", "A"], [0, 1])]
    questions, _ = D.build_questions(rows, _FakeCounter())
    question = questions[0]
    gold_keys = {question.sentence_keys[i] for i in question.gold_indices}
    assert gold_keys == {("B", 0), ("A", 1)}
    assert question.n_gold_annotated == 2


# --------------------------------------------------------------------------- #
# 11. ILP scope
# --------------------------------------------------------------------------- #
def test_ilp_scope_is_rho_025_and_the_first_hundred_questions():
    assert P.ILP_RHO == 0.25 and P.ILP_N_QUESTIONS == 100
    assert P.ILP_TIME_LIMIT_S == 60.0 and P.ILP_THREADS == 1
    assert R.ilp_in_scope(0, 0.25) and R.ilp_in_scope(99, 0.25)
    assert not R.ilp_in_scope(100, 0.25)
    assert not R.ilp_in_scope(0, 0.10)
    assert not R.ilp_in_scope(0, 0.50)


def test_only_proven_optimal_ilp_rows_enter_the_ilp_summary():
    rows = [
        {"arm": P.ARM_ILP, "rho": 0.25, "coverage": 1.0, "tokens_used": 10, "w_max": 10,
         "n_selected": 2, "latency_ms": 5.0, "score": 1.0, "status": "proven_optimal",
         "qid": "a", "position": 0},
        {"arm": P.ARM_ILP, "rho": 0.25, "coverage": 0.0, "tokens_used": 10, "w_max": 10,
         "n_selected": 2, "latency_ms": 5.0, "score": 0.5, "status": "feasible_not_proven",
         "qid": "b", "position": 1},
    ]
    summary = [s for s in A.arm_summary(rows) if s["arm"] == P.ARM_ILP][0]
    assert summary["n"] == 2                       # incumbents are kept in the raw record
    assert summary["n_proven_optimal"] == 1        # but only proven optima are summarized
    assert summary["coverage_mean_proven_only"] == 1.0


def test_ilp_result_is_marked_and_never_relabelled():
    knapsack = _instance(seed=2, budget=30)
    record = R.run_arm(P.ARM_ILP, knapsack)
    assert record["status"] in {"proven_optimal", "feasible_not_proven", "infeasible"}
    heuristic = R.run_arm(P.ARM_TA, knapsack)
    assert heuristic["status"] == "heuristic_no_guarantee"
    assert record["score"] >= heuristic["score"] - 1e-9


# --------------------------------------------------------------------------- #
# Best-singleton guard (arm 5)
# --------------------------------------------------------------------------- #
def test_best_singleton_guard_takes_the_better_of_the_two():
    """The classic density trap: cheap high-density crumbs crowd out the big item.

    Crumbs have density 0.6 against the fat sentence's 0.5, so token-aware buys
    them first; five of them leave only 15 tokens, and the fat sentence no
    longer fits. The guard must fall back to it.
    """
    relevance = np.array([10.0, 0.6, 0.6, 0.6, 0.6, 0.6])
    similarity = np.zeros((6, 6))
    tokens = np.array([20, 1, 1, 1, 1, 1], dtype=np.int64)
    knapsack = ContextKnapsack(relevance, similarity, tokens, budget=20,
                               objective_lambda=P.LAMBDA_OBJ)
    token_aware = R.run_arm(P.ARM_TA, knapsack)
    guarded = R.run_arm(P.ARM_TA_SINGLETON, knapsack)
    assert guarded["indices"] == [0]
    assert guarded["score"] > token_aware["score"]
    assert guarded["meta"]["guard_choice"] == "best_singleton"


def test_best_singleton_guard_defers_to_token_aware_on_ties():
    knapsack = _instance(seed=7, budget=40)
    token_aware = R.run_arm(P.ARM_TA, knapsack)
    guarded = R.run_arm(P.ARM_TA_SINGLETON, knapsack)
    assert guarded["score"] >= token_aware["score"] - 1e-12


# --------------------------------------------------------------------------- #
# Analysis wiring
# --------------------------------------------------------------------------- #
def test_paired_deltas_are_ta_minus_go_in_sample_order():
    rows = []
    for position, (qid, ta, go) in enumerate([("a", 1.0, 0.5), ("b", 0.0, 0.5)]):
        rows.append({"qid": qid, "position": position, "rho": 0.25,
                     "arm": P.ARM_TA, "coverage": ta})
        rows.append({"qid": qid, "position": position, "rho": 0.25,
                     "arm": P.ARM_GO, "coverage": go})
    deltas, qids = A.paired_deltas(rows, 0.25)
    assert qids == ["a", "b"]
    assert deltas.tolist() == [0.5, -0.5]


def test_unpaired_rows_are_refused():
    rows = [{"qid": "a", "position": 0, "rho": 0.25, "arm": P.ARM_TA, "coverage": 1.0}]
    with pytest.raises(P.ProtocolViolation):
        A.paired_deltas(rows, 0.25)
