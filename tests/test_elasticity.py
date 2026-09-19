"""Tests for the Step 6 elasticity experiment: generator invariants, the T1-T10
confirmatory-family structure, and the TOST margin rule.

Analysis-structure tests use SYNTHETIC rows with known values and deliberate
contamination (other arms, other methods), never pilot outcomes.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.elasticity import protocol as EP
import experiments.elasticity.analysis as AN
from experiments.elasticity.analysis import (CONFIRMATORY_FAMILY, AnalysisStructureError,
                                             RetiredArtifactError, assert_artifact_usable,
                                             build_inputs, holm, run_confirmatory,
                                             seed_level_table, signflip_pvalue, t8_cluster_scores)
from experiments.elasticity.generator import (ElasticitySpec, generate,
                                              ranking_disagreement)

OFF_PLAN = list(range(7000, 7040))   # never confirmatory (0-59), never pilot (1000-1004)


# --- protocol / A1, F1, C1 ------------------------------------------------- #
def test_locked_levels():
    assert EP.BETA_LEVELS == (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
    assert EP.RHO_C == {-0.5: -0.2, 0.0: 0.0, 0.5: 0.2, 1.0: 0.4, 1.5: 0.6, 2.0: 0.8}
    assert EP.BETA_PER_RHO == 2.5
    assert EP.REDUNDANCY_LEVELS == {"low": 0.15, "high": 0.70}
    assert EP.BUDGET_LEVELS == {"tight": 0.25, "medium": 0.60, "loose": 1.50}
    assert EP.MIN_TOKEN_CV == 0.30 and EP.BETA_TOLERANCE == 0.15
    assert EP.ELASTICITY_PROTOCOL_VERSION in ("6.2-lock-candidate", "6.2-locked")
    assert EP.OBJECTIVE_LAMBDA == 0.1 and EP.MMR_LAMBDA == 0.5
    assert set(EP.CONFIRMATORY_SEEDS) == set(range(60))
    assert not set(EP.PILOT_SEEDS) & set(EP.CONFIRMATORY_SEEDS)


# --- generator ------------------------------------------------------------- #
def test_determinism_bit_identical():
    for b in EP.BETA_LEVELS:
        x = generate(ElasticitySpec(7001, b, "high"))
        y = generate(ElasticitySpec(7001, b, "high"))
        for a in ("relevance", "tokens", "similarity", "embeddings", "topics"):
            assert np.array_equal(getattr(x, a), getattr(y, a)), (b, a)
        assert x.diagnostics["D"] == y.diagnostics["D"]


def test_different_seeds_differ():
    x, y = generate(ElasticitySpec(7001, 1.0, "low")), generate(ElasticitySpec(7002, 1.0, "low"))
    assert not np.allclose(x.relevance, y.relevance)


def test_geometry_exact_and_gram_valid():
    for b in EP.BETA_LEVELS:
        g = generate(ElasticitySpec(7003, b, "low")).diagnostics["geometry"]
        assert g["ok"], g


def test_tokens_are_the_own_latent_identical_across_beta():
    """IMPL-1: w must be bit-identical across every beta level for a seed."""
    base = generate(ElasticitySpec(7004, -0.5, "low")).tokens
    for b in EP.BETA_LEVELS:
        assert np.array_equal(generate(ElasticitySpec(7004, b, "low")).tokens, base), b


def test_gamma_moves_only_similarity():
    for b in EP.BETA_LEVELS:
        lo, hi = generate(ElasticitySpec(7005, b, "low")), generate(ElasticitySpec(7005, b, "high"))
        assert np.array_equal(lo.relevance, hi.relevance)
        assert np.array_equal(lo.tokens, hi.tokens)
        assert np.array_equal(lo.topics, hi.topics)
        assert lo.diagnostics["beta_hat"] == hi.diagnostics["beta_hat"]
        assert lo.diagnostics["D"] == hi.diagnostics["D"]
        assert not np.allclose(lo.similarity, hi.similarity)


def test_beta_mapping_on_large_sample():
    """Population check on off-plan seeds: the construction realises beta = 2.5*rho_c."""
    for b in EP.BETA_LEVELS:
        m = np.mean([generate(ElasticitySpec(s, b, "low")).diagnostics["beta_hat"]
                     for s in range(9000, 9400)])
        assert abs(m - b) <= EP.BETA_TOLERANCE, (b, m)


def test_relevance_marginal_invariant_across_beta():
    pools = {b: np.concatenate([generate(ElasticitySpec(s, b, "low")).relevance for s in OFF_PLAN])
             for b in EP.BETA_LEVELS}
    for b in EP.BETA_LEVELS[1:]:
        assert ks_2samp(pools[-0.5], pools[b]).pvalue > EP.KS_MIN_P, b


def test_disagreement_definition():
    r = np.array([0.30, 0.10, 0.20, 0.05])
    assert ranking_disagreement(r, np.array([10, 10, 10, 10])) == 0.0
    # identical ordering of r and r/w -> D = 0
    assert ranking_disagreement(r, np.array([10, 11, 12, 13])) == 0.0
    # r ranks reversed by r/w -> D = 2
    assert abs(ranking_disagreement(np.array([1.0, 2.0, 3.0]), np.array([1, 10, 100])) - 2.0) < 1e-12


# --- synthetic rows for analysis-structure tests --------------------------- #
BUDGET_OFF = {"tight": 0.0, "medium": 1.0, "loose": 2.0}
RED_OFF = {"low": 0.0, "high": 0.02}
ZERO_CELLS = {(0, 0.0, "low"), (0, 0.0, "high"), (2, 1.5, "low")}   # exact ties: identical sets


def _true_delta(seed, beta, red, budget):
    if (seed, beta, red) in ZERO_CELLS:
        return 0.0
    return seed * 1e-3 + beta * 0.1 + RED_OFF[red] + BUDGET_OFF[budget]


def synthetic_rows(seeds=(0, 1, 2, 3), contaminate=True, drop=None):
    rows = []
    for s in seeds:
        for b in EP.BETA_LEVELS:
            for red in EP.REDUNDANCY_LEVELS:
                iid = f"EL-s{s}-b{b:+.1f}-{red}"
                for bud in EP.BUDGET_LEVELS:
                    if drop == (s, b, red, bud):
                        continue
                    d = _true_delta(s, b, red, bud)
                    base = {"arm": EP.ARM, "instance_id": iid, "seed": s, "beta_target": b,
                            "redundancy_level": red, "budget_level": bud, "D": 0.1 + 0.01 * s + b / 100}
                    rows.append({**base, "method": "greedy_objective", "objective_score": 1.0,
                                 "selected_indices": [0, 1]})
                    rows.append({**base, "method": "greedy_token_aware", "objective_score": 1.0 + d,
                                 "selected_indices": [1, 0] if d == 0.0 else [0, 2]})
                    rows.append({**base, "method": "top_k", "objective_score": 999.0, "selected_indices": [5]})
                    rows.append({**base, "method": "mmr", "objective_score": -999.0, "selected_indices": [6]})
                    rows.append({**base, "method": "ilp", "objective_score": 5.0, "selected_indices": [0],
                                 "ilp_proven_optimal": True})
                    if contaminate:
                        for arm in ("D_separate", "E_realism_anchor"):
                            rows.append({**base, "arm": arm, "method": "greedy_token_aware",
                                         "objective_score": 12345.0, "selected_indices": [9]})
    return rows


# --- amendment 6.1: family ------------------------------------------------- #
def test_family_has_exactly_nine_tests_and_no_t5():
    ids = [t.test_id for t in CONFIRMATORY_FAMILY]
    assert ids == ["T1", "T2", "T3", "T4", "T6", "T7", "T8", "T9", "T10"]
    assert "T5" not in ids
    assert len(ids) == EP.CONFIRMATORY_FAMILY_SIZE == 9
    assert all(t.family == "F1" and t.status == "confirmatory" for t in CONFIRMATORY_FAMILY)
    assert abs(EP.HOLM_ALPHA_FOR_POWER - 0.05 / 9) < 1e-15


def test_no_pairwise_beta_comparisons():
    for t in CONFIRMATORY_FAMILY:
        assert len(t.beta_levels) in (1, 6), t.test_id
        if len(t.beta_levels) == 6:
            assert t.beta_levels == EP.BETA_LEVELS


def test_declared_subsets_and_statistics():
    spec = {t.test_id: (t.budget_level, t.beta_levels, t.statistic, t.sidedness) for t in CONFIRMATORY_FAMILY}
    assert spec["T1"] == ("tight", EP.BETA_LEVELS, "friedman", "two")
    assert spec["T2"] == ("tight", (1.0,), "signflip_mean", "two")
    assert spec["T3"] == ("tight", (1.5,), "signflip_mean", "two")
    assert spec["T4"] == ("tight", (2.0,), "signflip_mean", "two")
    assert spec["T6"] == ("tight", EP.BETA_LEVELS, "signflip_contrast_linear", "two")
    assert spec["T7"] == ("tight", EP.BETA_LEVELS, "signflip_contrast_quadratic", "two")
    assert spec["T8"] == ("tight", EP.BETA_LEVELS, "cluster_signflip_score_slope", "one_greater")
    assert spec["T9"] == ("loose", EP.BETA_LEVELS, "friedman", "two")
    assert spec["T10"] == ("medium", EP.BETA_LEVELS, "signflip_contrast_quadratic", "two")
    assert not any("wilcoxon" in t.statistic or "tost" in t.statistic for t in CONFIRMATORY_FAMILY)


def test_exploratory_items_never_enter_the_family():
    stats = " ".join(t.statistic + t.estimand for t in CONFIRMATORY_FAMILY).lower()
    for word in ("one-document", "hurdle", "0.30", "tost", "equivalence"):
        assert word not in stats, word
    assert all(t.budget_level in EP.BUDGET_LEVELS for t in CONFIRMATORY_FAMILY)


# --- structure ------------------------------------------------------------- #
def test_redundancy_pooled_as_mean_zeros_kept_and_contamination_excluded():
    inp = build_inputs(synthetic_rows())
    for tid, bud, levels in (("T2", "tight", (1.0,)), ("T9", "loose", EP.BETA_LEVELS),
                             ("T10", "medium", EP.BETA_LEVELS)):
        e = inp[tid]
        assert e["seeds"] == [0, 1, 2, 3] and e["n_seeds"] == 4
        for b in levels:
            expected = [np.mean([_true_delta(s, b, r, bud) for r in EP.REDUNDANCY_LEVELS]) for s in e["seeds"]]
            assert np.allclose(e["values"][b], expected), (tid, b)
    assert inp["T1"]["values"][0.0][0] == 0.0          # a fully-tied cell is kept as an exact zero


def test_exact_raw_rows_enter_each_test():
    inp = build_inputs(synthetic_rows())
    for t in CONFIRMATORY_FAMILY:
        keys = inp[t.test_id]["raw_row_keys"]
        assert len(keys) == 4 * len(t.beta_levels) * 2 * 2, t.test_id
        assert {k[1] for k in keys} == {t.budget_level}
        assert {k[2] for k in keys} == set(("greedy_objective", "greedy_token_aware"))


def test_t8_instance_level_binary_outcome_by_set_inequality():
    e = build_inputs(synthetic_rows())["T8"]
    assert len(e["instances"]) == 4 * 6 * 2                       # seeds x beta x redundancy
    for r in e["instances"]:
        assert r["Y"] in (0, 1)
        assert r["Y"] == int(_true_delta(r["seed"], r["beta"], r["redundancy"], "tight") != 0.0)
    # [1, 0] vs [0, 1] is the SAME set -> Y = 0
    assert all(r["Y"] == 0 for r in e["instances"] if (r["seed"], r["beta"], r["redundancy"]) in ZERO_CELLS)


def test_t8_score_slope_is_on_probability_scale():
    rng = np.random.default_rng(3)
    inst = []
    for s in range(40):
        for b in EP.BETA_LEVELS:
            D = 0.25 + 0.05 * rng.standard_normal()
            for red in ("high", "low"):
                inst.append({"seed": s, "beta": b, "redundancy": red, "D": D,
                             "Y": int(rng.random() < 0.3 + 0.8 * (D - 0.25))})
    out = t8_cluster_scores(inst)
    assert abs(out["b_hat"] - 0.8) < 0.35                           # recovers slope in probability points
    assert len(out["scores"]) == 40
    none = [{**r, "Y": 1} for r in inst]
    assert t8_cluster_scores(none)["b_hat"] == 0.0                   # constant Y -> zero slope


def test_t8_protocol_frozen_and_rejects_mixed_models():
    assert "identity link" in EP.T8_SPEC["estimand"] and EP.T8_SPEC["H1"] == "b > 0"
    assert any("logistic" in x for x in EP.T8_SPEC["rejected"])
    assert EP.DECLARATIONS_6_1["T8_linear_mixed_model_carryover"].startswith("NOT CONFIRMED")
    import inspect
    assert "mixedlm" not in inspect.getsource(AN)


def test_signflip_one_sided():
    x = np.full(30, 0.1)
    assert signflip_pvalue(x, n_flips=4000, rng_seed=1, alternative="greater") < 0.01
    assert signflip_pvalue(-x, n_flips=4000, rng_seed=1, alternative="greater") > 0.99


def test_seed_count_rule_has_no_unresolved_items():
    R = EP.SEED_COUNT_RULE
    assert R["regime"] == "R2" and R["status"] in ("LOCK_CANDIDATE", "LOCKED") and R["unresolved"] == []
    assert set(R["resolved_in_lock_candidate"]) == {"U-SC2a-Q", "U-SC-H"}
    assert "max(60" in R["formulas"]["n_final"] and EP.N_MIN_CONFIRMATORY_SEEDS == 60
    assert EP.DECLARATIONS_6_1["permutation_rng_seed_61000"] == "CONFIRMED"
    assert {l["id"] for l in EP.REPRODUCIBILITY_LIMITATIONS} >= {"RL-1", "RL-2", "RL-3", "RL-4"}


def test_missing_redundancy_level_raises():
    try:
        build_inputs(synthetic_rows(drop=(1, 1.0, "high", "tight")))
    except AnalysisStructureError:
        return
    raise AssertionError("an incomplete redundancy pair was silently pooled")


def test_unproven_ilp_raises():
    rows = synthetic_rows(contaminate=False)
    for r in rows:
        if r["method"] == "ilp" and r["seed"] == 2:
            r["ilp_proven_optimal"] = False
    try:
        build_inputs(rows)
    except AnalysisStructureError:
        return
    raise AssertionError("an unproven ILP reference was accepted")


# --- statistics ------------------------------------------------------------ #
def test_signflip_keeps_zeros_in_the_denominator():
    x_with_zeros = np.array([0.0] * 8 + [0.3, 0.4])
    x_nonzero = np.array([0.3, 0.4])
    p_zeros = signflip_pvalue(x_with_zeros, n_flips=5000, rng_seed=1)
    p_nonzero = signflip_pvalue(x_nonzero, n_flips=5000, rng_seed=1)
    assert np.isclose(np.mean(x_with_zeros), 0.07)
    assert p_zeros > 0.0 and p_nonzero > 0.0
    # zeros are not discarded: an all-zero vector gives p = 1 with n preserved
    assert signflip_pvalue(np.zeros(12), n_flips=2000, rng_seed=1) == 1.0


def test_signflip_deterministic_under_declared_seed():
    x = np.linspace(-0.1, 0.3, 40)
    assert signflip_pvalue(x) == signflip_pvalue(x)
    assert EP.PERMUTATION_N_FLIPS == 10_000 and isinstance(EP.PERMUTATION_RNG_SEED, int)


def test_holm_over_nine():
    p = {f"T{i}": v for i, v in zip((1, 2, 3, 4, 6, 7, 8, 9, 10), (.001, .01, .02, .03, .04, .05, .2, .5, .9))}
    adj = holm(p)
    assert np.isclose(adj["T1"], 0.009) and len(adj) == 9


def test_run_confirmatory_has_no_margin_argument():
    import inspect
    assert list(inspect.signature(run_confirmatory).parameters) == ["rows"]
    src = inspect.getsource(AN)
    assert "retired_tost" not in src.split('"""', 2)[2] and "tost_margin.json" not in src.split('"""', 2)[2]


# --- retirement ------------------------------------------------------------ #
def test_retired_tost_unusable_by_analysis():
    for name in ("compute_margin", "tost_power", "n_required", "write_margin_once"):
        assert not hasattr(AN, name), name
    try:
        assert_artifact_usable("results/elasticity/tost_margin.json")
    except RetiredArtifactError:
        pass
    else:
        raise AssertionError("retired artifact accepted")
    assert not any(k.startswith("TOST") for k in dir(EP))


def test_retired_artifact_preserved_with_sidecar():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import hashlib
    art = os.path.join(root, "results/elasticity/tost_margin.json")
    side = os.path.join(root, EP.RETIRED_ARTIFACTS["results/elasticity/tost_margin.json"]["sidecar"])
    if not os.path.exists(art):
        return
    meta = json.load(open(side))
    assert meta["RETIRED"] is True
    assert hashlib.sha256(open(art, "rb").read()).hexdigest() == meta["sha256_at_retirement"]


# --- gates / invariants ----------------------------------------------------- #
def test_variance_pilot_gate_follows_protocol_status():
    locked = EP.SEED_COUNT_RULE["status"] == "LOCKED"
    try:
        EP.assert_seed_count_rule_locked()
        opened = True
    except EP.ProtocolNotLockedError:
        opened = False
    assert opened == locked
    assert EP.VARIANCE_PILOT_SEEDS == tuple(range(2000, 2040))
    assert not set(EP.VARIANCE_PILOT_SEEDS) & (set(EP.CONFIRMATORY_SEEDS) | set(EP.PILOT_SEEDS))


def test_invariant_status_after_amendment():
    st = {i["id"]: i["status"] for i in EP.INVARIANTS}
    assert st["G4"] == "descriptive"
    assert st["K1"] == st["K2"] == st["K3"] == "retired"
    assert st["G4P"] == st["L1"] == st["M1"] == "locked"
    assert EP.SESOI_FRACTION_OF_OPTIMUM == 0.03 and EP.MAX_CONFIRMATORY_SEEDS == 300


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                fails += 1; print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{fails} failure(s)"); raise SystemExit(1 if fails else 0)
