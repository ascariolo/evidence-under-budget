"""Validation suite for the controlled benchmark generator.

These are NOT optional sanity checks. The causal-isolation tests in particular
are what prevents a repeat of the v0 benchmark-design problem, where the
generator silently determined the relationship it was supposed to control.

Run standalone (``python tests/test_controlled_generator.py``) or under pytest.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.controlled import protocol as P
from experiments.controlled.generator import (ControlledInstance,
                                              DegenerateInstanceError,
                                              GeneratorConfig, InstanceSpec,
                                              apply_adversarial, build_budgets,
                                              compute_saturation,
                                              generate_instance,
                                              prepare_instance,
                                              validate_instance)
from optimizer import ContextKnapsack, SolveStatus

SEEDS = list(range(12))
MANY_SEEDS = list(range(40))


def _spec(seed=0, rl="A", red="low", **cfg) -> InstanceSpec:
    return InstanceSpec(seed=seed, rl_condition=rl, redundancy_level=red,
                        config=GeneratorConfig(**cfg))


def _knap(inst, budget):
    return ContextKnapsack(inst.relevance, inst.similarity, inst.tokens,
                           budget=budget, objective_lambda=P.OBJECTIVE_LAMBDA,
                           mmr_lambda=P.MMR_LAMBDA)


def _pool(rl, seeds=MANY_SEEDS, red="low"):
    rel, tok = [], []
    for s in seeds:
        i = generate_instance(_spec(seed=s, rl=rl, red=red))
        rel.append(i.relevance); tok.append(i.tokens.astype(float))
    return np.concatenate(rel), np.concatenate(tok)


# --- 9. determinism -------------------------------------------------------- #
def test_generation_is_deterministic():
    for rl in P.RL_CONDITIONS:
        for red in P.REDUNDANCY_LEVELS:
            a = generate_instance(_spec(seed=7, rl=rl, red=red))
            b = generate_instance(_spec(seed=7, rl=rl, red=red))
            assert np.array_equal(a.relevance, b.relevance), rl
            assert np.array_equal(a.tokens, b.tokens), rl
            assert np.array_equal(a.similarity, b.similarity), rl
            assert np.array_equal(a.topics, b.topics), rl


def test_different_seeds_give_different_instances():
    a = generate_instance(_spec(seed=1))
    b = generate_instance(_spec(seed=2))
    assert not np.allclose(a.relevance, b.relevance)
    assert not np.array_equal(a.tokens, b.tokens)
    assert not np.allclose(a.similarity, b.similarity)


# --- 4. embedding construction --------------------------------------------- #
def test_cosine_to_query_equals_relevance_exactly():
    for rl in P.RL_CONDITIONS:
        for red in P.REDUNDANCY_LEVELS:
            inst = generate_instance(_spec(seed=3, rl=rl, red=red))
            cos = (inst.embeddings @ inst.query_vector) / np.linalg.norm(
                inst.embeddings, axis=1)
            assert np.abs(cos - inst.relevance).max() < P.RELEVANCE_EXACTNESS_TOLERANCE


def test_similarity_is_a_valid_gram_matrix():
    for seed in SEEDS:
        for red in P.REDUNDANCY_LEVELS:
            s = generate_instance(_spec(seed=seed, red=red)).similarity
            assert np.abs(s - s.T).max() < 1e-10
            assert np.abs(np.diag(s) - 1.0).max() < 1e-8
            assert np.linalg.eigvalsh(0.5 * (s + s.T)).min() >= P.GRAM_EIGENVALUE_TOLERANCE
            assert s.min() >= -1 - 1e-9 and s.max() <= 1 + 1e-9


def test_embeddings_are_unit_norm():
    inst = generate_instance(_spec(seed=5))
    assert np.abs(np.linalg.norm(inst.embeddings, axis=1) - 1.0).max() < 1e-10


# --- 11. causal isolation: redundancy intervention -------------------------- #
def test_gamma_intervention_leaves_relevance_and_tokens_untouched():
    """Changing gamma must move similarity and NOTHING else."""
    for seed in SEEDS:
        lo = generate_instance(_spec(seed=seed, red="low"))
        hi = generate_instance(_spec(seed=seed, red="high"))
        assert np.array_equal(lo.relevance, hi.relevance), seed
        assert np.array_equal(lo.tokens, hi.tokens), seed
        assert lo.diagnostics["beta_hat"] == hi.diagnostics["beta_hat"], seed
        assert np.array_equal(lo.topics, hi.topics), seed
        # ... while similarity genuinely moves.
        assert not np.allclose(lo.similarity, hi.similarity), seed


def test_gamma_raises_inter_document_similarity():
    iu = np.triu_indices(P.N_DOCS, 1)
    lows, highs = [], []
    for seed in SEEDS:
        lows.append(generate_instance(_spec(seed=seed, red="low")).similarity[iu].mean())
        highs.append(generate_instance(_spec(seed=seed, red="high")).similarity[iu].mean())
    assert np.mean(highs) > np.mean(lows)
    # A real contrast, not a rounding difference.
    assert np.mean(highs) - np.mean(lows) > 0.05


def test_gamma_does_not_change_query_relevance_distribution():
    rel_lo, tok_lo = _pool("A", red="low")
    rel_hi, tok_hi = _pool("A", red="high")
    assert np.array_equal(rel_lo, rel_hi)
    assert np.array_equal(tok_lo, tok_hi)


# --- 11. causal isolation: relevance-length intervention -------------------- #
def test_copula_preserves_both_marginals_across_conditions():
    """A, B and C must differ in DEPENDENCE only, not in either marginal."""
    rel_a, tok_a = _pool("A")
    for rl in ("B", "C"):
        rel, tok = _pool(rl)
        assert ks_2samp(rel_a, rel).pvalue > 0.01, f"relevance marginal moved in {rl}"
        assert ks_2samp(tok_a, tok).pvalue > 0.01, f"token marginal moved in {rl}"


def test_conditions_realize_their_target_elasticity():
    """beta_hat averaged over seeds must match the declared target."""
    for rl in ("A", "B", "C"):
        target = P.RL_CONDITIONS[rl]["target_beta"]
        betas = [generate_instance(_spec(seed=s, rl=rl)).diagnostics["beta_hat"]
                 for s in MANY_SEEDS]
        assert abs(np.mean(betas) - target) <= P.BETA_TOLERANCE, \
            f"{rl}: mean beta_hat {np.mean(betas):+.3f} vs target {target:+.2f}"


def test_condition_sign_is_not_inferred_from_realized_correlation():
    """The condition is declared, not discovered: the spec carries it."""
    spec = _spec(seed=0, rl="B")
    assert spec.rho_c == 0.5 and spec.target_beta == 0.5
    inst = generate_instance(spec)
    assert inst.condition_metadata()["rl_condition"] == "B"
    assert inst.condition_metadata()["target_beta"] == 0.5
    # Realized values are recorded alongside, never used to define the condition.
    for key in ("beta_hat", "pearson_r_relevance_tokens",
                "spearman_rho_relevance_tokens"):
        assert key in inst.diagnostics


def test_positive_and_negative_conditions_have_opposite_correlation():
    def mean_pearson(rl):
        return np.mean([generate_instance(_spec(seed=s, rl=rl)).diagnostics
                        ["pearson_r_relevance_tokens"] for s in MANY_SEEDS])
    assert mean_pearson("B") > 0.25
    assert mean_pearson("C") < -0.25
    assert abs(mean_pearson("A")) < 0.15


# --- 11. causal isolation: budget and seed --------------------------------- #
def test_budget_intervention_does_not_touch_the_documents():
    inst = generate_instance(_spec(seed=4))
    before = (inst.relevance.copy(), inst.tokens.copy(), inst.similarity.copy())
    budgets = build_budgets(w_sat=500)
    for b in budgets.values():
        _knap(inst, b).solve_greedy()
    assert np.array_equal(inst.relevance, before[0])
    assert np.array_equal(inst.tokens, before[1])
    assert np.array_equal(inst.similarity, before[2])


def test_seed_intervention_preserves_the_condition():
    for seed in SEEDS:
        inst = generate_instance(_spec(seed=seed, rl="B", red="high"))
        md = inst.condition_metadata()
        assert md["rl_condition"] == "B" and md["rho_c"] == 0.5
        assert md["redundancy_level"] == "high" and md["gamma"] == 0.70
        assert md["seed"] == seed


# --- 10. non-degeneracy ---------------------------------------------------- #
def test_token_variance_guard_is_enforced():
    """The critical guard: near-constant w makes the two greedies identical."""
    for seed in SEEDS:
        inst = generate_instance(_spec(seed=seed))
        assert inst.diagnostics["tokens_cv"] >= P.MIN_TOKEN_CV


def test_degenerate_token_lengths_fail_loudly():
    """A near-constant length distribution must be rejected, not returned."""
    spec = _spec(seed=0, log_token_sigma=0.001)
    try:
        generate_instance(spec, strict=True)
    except DegenerateInstanceError as exc:
        assert "token CV" in str(exc)
        return
    raise AssertionError("a degenerate length distribution was accepted")


def test_relevance_and_similarity_are_not_degenerate():
    for seed in SEEDS:
        for red in P.REDUNDANCY_LEVELS:
            d = generate_instance(_spec(seed=seed, red=red)).diagnostics
            assert d["relevance_std"] >= P.MIN_RELEVANCE_STD
            assert d["similarity_std"] >= P.MIN_SIMILARITY_STD


def test_tokens_are_exact_positive_integers():
    inst = generate_instance(_spec(seed=6))
    assert np.issubdtype(inst.tokens.dtype, np.integer)
    assert (inst.tokens > 0).all()
    assert (inst.tokens >= P.TOKEN_MIN).all() and (inst.tokens <= P.TOKEN_MAX).all()


# --- condition D ----------------------------------------------------------- #
def test_adversarial_condition_inflates_the_most_relevant_documents():
    base = generate_instance(_spec(seed=8, rl="D"))
    before = base.tokens.copy()
    top = np.argsort(-base.relevance, kind="stable")[:P.ADVERSARIAL_TOP_K]
    inst = apply_adversarial(base, w_sat_pre=800)
    assert inst.diagnostics["adversarial_applied"] is True
    assert (inst.tokens[top] >= before[top]).all()
    assert (inst.tokens[top] > before[top]).any()
    untouched = np.setdiff1d(np.arange(len(before)), top)
    assert np.array_equal(inst.tokens[untouched], before[untouched])


def test_adversarial_documents_stay_individually_feasible_at_medium_budget():
    out = prepare_instance(_spec(seed=9, rl="D"))
    medium = out["budgets"]["medium"]
    top = out["diagnostics"]["adversarial_indices"]
    assert all(out["instance"].tokens[i] <= medium for i in top), \
        "an adversarial document cannot be bought at the medium budget"


def test_non_adversarial_conditions_are_not_modified():
    for rl in ("A", "B", "C"):
        out = prepare_instance(_spec(seed=10, rl=rl))
        assert out["diagnostics"]["adversarial_applied"] is False


# --- 7. budget construction ------------------------------------------------ #
def test_budgets_are_instance_relative_and_separated():
    out = prepare_instance(_spec(seed=11, rl="A", red="high"))
    b, w_sat = out["budgets"], out["saturation"]["w_sat"]
    assert b["tight"] < b["medium"] < b["loose"]
    for name, rho in P.BUDGET_LEVELS.items():
        assert abs(b[name] / w_sat - rho) < 0.02, name
    assert out["saturation"]["w_sat_proven"] is True
    for key in ("w_sat", "corpus_token_mass", "w_sat_n_selected"):
        assert key in out["saturation"]


def test_loose_budget_does_not_bind_and_tight_budget_does():
    out = prepare_instance(_spec(seed=2, rl="A", red="high"))
    inst = out["instance"]
    loose = _knap(inst, out["budgets"]["loose"]).solve_ilp(time_limit=120)
    tight = _knap(inst, out["budgets"]["tight"]).solve_ilp(time_limit=120)
    assert loose.tokens_used <= out["saturation"]["w_sat"] + 1
    assert tight.tokens_used <= out["budgets"]["tight"]
    assert tight.score < loose.score


# --- 10. ILP tractability -------------------------------------------------- #
def test_ilp_proves_optimality_on_representative_instances():
    for rl in P.RL_CONDITIONS:
        for red in P.REDUNDANCY_LEVELS:
            out = prepare_instance(_spec(seed=1, rl=rl, red=red))
            k = _knap(out["instance"], out["budgets"]["medium"])
            res = k.solve_ilp(time_limit=P.ILP_TIME_LIMIT_S)
            assert res.status is SolveStatus.PROVEN_OPTIMAL, f"{rl}/{red}: {res.status}"
            assert res.is_proven_global_optimum


# --- 14. critical: algorithm divergence ------------------------------------ #
def test_greedy_objective_and_token_aware_actually_diverge():
    """If these never differ, the benchmark cannot test token-awareness at all."""
    diverged = []
    for seed in SEEDS:
        for rl in P.RL_CONDITIONS:
            for red in P.REDUNDANCY_LEVELS:
                inst = generate_instance(_spec(seed=seed, rl=rl, red=red))
                for budget in (150, 400, 900):
                    k = _knap(inst, budget)
                    if k.solve_greedy_objective().indices != k.solve_greedy().indices:
                        diverged.append((seed, rl, red, budget))
    assert diverged, ("greedy_objective and greedy_token_aware made identical "
                      "choices everywhere: the benchmark does not expose the "
                      "token-cost mechanism")
    # Divergence must not be a single freak instance.
    assert len({d[1] for d in diverged}) >= 3, "divergence in too few conditions"


def test_divergence_is_absent_when_lengths_are_constant():
    """Control for the test above: equal w makes the two provably identical."""
    inst = generate_instance(_spec(seed=0), strict=False)
    inst.tokens = np.full(P.N_DOCS, 50, dtype=np.int64)
    for budget in (150, 400, 900):
        k = _knap(inst, budget)
        assert k.solve_greedy_objective().indices == k.solve_greedy().indices


# --- metadata -------------------------------------------------------------- #
def test_metadata_reconstructs_the_condition():
    out = prepare_instance(_spec(seed=3, rl="C", red="high"))
    md = out["metadata"]
    for key in ("protocol_version", "seed", "rl_condition", "rho_c",
                "target_beta", "redundancy_level", "gamma", "is_adversarial",
                "n_docs", "objective_lambda", "mmr_lambda", "config"):
        assert key in md, key
    rebuilt = InstanceSpec(seed=md["seed"], rl_condition=md["rl_condition"],
                           redundancy_level=md["redundancy_level"])
    again = generate_instance(rebuilt)
    assert np.array_equal(again.relevance, out["instance"].relevance)


def test_frozen_parameters_match_the_repaired_baseline():
    assert P.OBJECTIVE_LAMBDA == 0.1
    assert P.MMR_LAMBDA == 0.5


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
            except Exception as exc:
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
