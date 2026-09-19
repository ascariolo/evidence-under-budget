r"""Frozen protocol for the Step 6 mechanistic elasticity experiment.

Specification: docs/elasticity_experiment_design.md rev. 2.

This module is written BEFORE any measurement. Every locked design requirement is
converted into a machine-checkable invariant in ``INVARIANTS``; every point where the
design text left an implementation choice open is resolved in
``IMPLEMENTATION_DECISIONS`` and flagged for sign-off rather than chosen silently.

The ONLY quantity the pilot may determine is the TOST margin ``m`` (design sec. F),
written once to results/elasticity/tost_margin.json.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

ELASTICITY_PROTOCOL_VERSION = "6.2-locked"
PREVIOUS_PROTOCOL_VERSION = "6.2-lock-candidate"
ARM = "elasticity_factorial"

# --------------------------------------------------------------------------- #
# Factor 1: beta via the Gaussian-copula correlation   (design A4, D)
# --------------------------------------------------------------------------- #
SIGMA_R = 1.25            # a CHOICE, not the minimum (min is 1.00, degenerate) - design A3
SIGMA_W = 0.50            # UNCHANGED from Step 4
BETA_PER_RHO = SIGMA_R / SIGMA_W          # = 2.5, exact under joint log-normality

BETA_LEVELS: Tuple[float, ...] = (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
RHO_C: Dict[float, float] = {b: round(b / BETA_PER_RHO, 10) for b in BETA_LEVELS}
# -> {-0.5: -0.2, 0.0: 0.0, 0.5: 0.2, 1.0: 0.4, 1.5: 0.6, 2.0: 0.8}

BETA_TOLERANCE = 0.15     # on the per-level MEAN of beta_hat; declared before measuring
BETA_GT1_LEVEL = 2.0
BETA_GT1_MIN_FRACTION = 0.95

# --------------------------------------------------------------------------- #
# Marginals   (design A4)
# --------------------------------------------------------------------------- #
MU_R = -2.995732273553991        # = log(0.05)
R_CLIP = (1e-3, 0.95)
MU_W = 4.343805421853684         # = log(77), UNCHANGED from Step 4
TOKEN_CLIP = (8, 400)            # UNCHANGED from Step 4

# --------------------------------------------------------------------------- #
# Held constant - UNCHANGED from the locked Step 4 protocol
# --------------------------------------------------------------------------- #
N_DOCS = 25
EMBED_DIM = 64
N_TOPICS = 5
REDUNDANCY_LEVELS: Dict[str, float] = {"low": 0.15, "high": 0.70}   # low vs MODERATE
BUDGET_LEVELS: Dict[str, float] = {"tight": 0.25, "medium": 0.60, "loose": 1.50}
OBJECTIVE_LAMBDA = 0.1
MMR_LAMBDA = 0.5
ILP_TIME_LIMIT_S = 600.0
MIN_TOKEN_CV = 0.30              # locked guard; exclusion policy below

# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #
CONFIRMATORY_SEEDS: Tuple[int, ...] = tuple(range(60))
PILOT_SEEDS: Tuple[int, ...] = (1000, 1001, 1002, 1003, 1004)     # disjoint (IMPL-5)

# --------------------------------------------------------------------------- #
# Degeneracy / diagnostic thresholds   (design H)
# --------------------------------------------------------------------------- #
USABLE_RELEVANCE_THRESHOLD = 0.05
MIN_USABLE_DOCS = 4
MAX_USABLE_COUNT_LEVEL_DIFF = 1.5          # IMPL-8, declared before measuring
MIN_SELECTED_AT_TIGHT = 2
MIN_WITHIN_LEVEL_SD_D = 0.02
D_EXPECTED_RANGE = (0.1, 0.3)              # qualitative expectation; diagnostic only
KS_MIN_P = 0.01
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
QUANTILE_TOL_FRACTION_OF_SIGMA = 0.25      # IMPL-8, declared before measuring
MAX_TOKEN_CV_REJECTION_RATE = 0.01
RHO_BUDGET_TOLERANCE = 0.02
ILP_RUNTIME_Q95_MAX_S = 5.0

# --------------------------------------------------------------------------- #
# Confirmatory analysis   (design E)
# --------------------------------------------------------------------------- #
CONFIRMATORY_FAMILY_SIZE = 9          # amended 6.1 (T5 retired)
POLY_LINEAR = (-5, -3, -1, 1, 3, 5)
POLY_QUADRATIC = (5, -1, -4, -4, -1, 5)

# --------------------------------------------------------------------------- #
# TOST - RETIRED at 6.1. Values kept ONLY for provenance; no top-level names remain,
# so no code can pick them up by accident.
# --------------------------------------------------------------------------- #
RETIRED_CONSTANTS = {
    "TOST_MARGIN_FALLBACK": 0.03,
    "TOST_MARGIN_UNSTABLE_IQR_MULTIPLE": 2.0,
    "TOST_MARGIN_DECIMALS": 2,
    "TOST_ALPHA": 0.05,
    "TOST_TARGET_POWER": 0.80,
    "PLANNED_N_SEEDS": 60,
}

# --------------------------------------------------------------------------- #
# Upstream integrity - modules this experiment reuses read-only
# --------------------------------------------------------------------------- #
UPSTREAM_SHA256_PREFIX = {
    "optimizer.py": "facfb186ddcc",
    "benchmark.py": "5ce79b512ae3",
    "experiments/controlled/generator.py": "d7ccf7907794",
    "experiments/controlled/protocol.py": "566481725b48",
}

# --------------------------------------------------------------------------- #
# INVARIANTS - every locked requirement, machine-checkable
# --------------------------------------------------------------------------- #
INVARIANTS: List[Dict[str, str]] = [
    {"id": "Z1", "src": "6B", "check": "upstream module hashes equal UPSTREAM_SHA256_PREFIX"},
    {"id": "Z2", "src": "6B", "check": "results/controlled/* byte-identical before and after preflight"},
    {"id": "A1", "src": "D", "check": "beta levels == {-0.5,0,0.5,1,1.5,2} and rho_c == beta/2.5"},
    {"id": "A2", "src": "H1", "check": "per-level mean beta_hat within 0.15 of target (planned ensemble)"},
    {"id": "A3", "src": "H2", "check": ">=95% of instances at beta=2.0 have beta_hat > 1"},
    {"id": "B1", "src": "H3", "check": "pairwise KS p > 0.01 for r across all 15 beta pairs"},
    {"id": "B2", "src": "H4", "check": "pairwise KS p > 0.01 for w across all 15 beta pairs"},
    {"id": "B3", "src": "IMPL-8", "check": "quantiles of log r / log w differ by <= 0.25*sigma between any two levels"},
    {"id": "C1", "src": "H5", "check": "token-CV threshold == 0.30 exactly"},
    {"id": "C2", "src": "H5", "check": "token-CV rejection rate <= 1%"},
    {"id": "C3", "src": "H5", "check": "rejections exactly balanced across beta levels"},
    {"id": "C4", "src": "H5", "check": "rejected instances never resampled; rejected seeds absent from every cell"},
    {"id": "D1", "src": "H9", "check": "every instance has >= 4 docs with r > 0.05"},
    {"id": "D2", "src": "IMPL-8", "check": "mean usable-doc count differs by <= 1.5 between any two beta levels"},
    {"id": "E1", "src": "C1", "check": "D = 1 - tau_b(r, r/w) finite and in [0, 2]"},
    {"id": "E2", "src": "H10", "check": "within-level sd(D) > 0.02 at every beta level"},
    {"id": "E3", "src": "6B-I", "check": "D recomputes bit-identically"},
    {"id": "F1", "src": "D", "check": "gamma levels == {low: 0.15, high: 0.70}"},
    {"id": "F2", "src": "H6", "check": "changing gamma leaves r, w, beta_hat, D, topics bit-identical"},
    {"id": "F3", "src": "H6", "check": "mean similarity strictly higher at high gamma in every paired instance"},
    {"id": "G1", "src": "H7", "check": "W_sat from a proven-optimal unconstrained ILP for every instance"},
    {"id": "G2", "src": "H7", "check": "tight < medium < loose and realized rho within 0.02, every instance"},
    {"id": "G3", "src": "6B-G", "check": "tight budget >= cheapest document, every instance"},
    {"id": "G4", "src": "H9", "check": "both greedy methods select >= 2 docs at tight (pilot)"},
    {"id": "G5", "src": "6B-G", "check": "no budget level at which all four heuristics coincide on every instance (pilot)"},
    {"id": "H1", "src": "H8", "check": "100% of pilot ILPs proven globally optimal"},
    {"id": "H2", "src": "H8", "check": "pilot ILP runtime q95 < 5 s"},
    {"id": "H3", "src": "6B-H", "check": "no heuristic ever flagged proven optimal"},
    {"id": "H4", "src": "6B-H", "check": "solver objective == recomputed; gaps >= -1e-6; Delta_gap identity exact"},
    {"id": "I1", "src": "6B-I", "check": "full pipeline repeat identical: corpus, r, similarity, budgets, objectives, selections, ILP"},
    {"id": "J1", "src": "E", "check": "confirmatory family has exactly 10 tests T1-T10 in family F1"},
    {"id": "J2", "src": "E", "check": "no pairwise beta comparison: every test uses 1 level or all 6"},
    {"id": "J3", "src": "E", "check": "only arm == elasticity_factorial and the two greedy methods enter"},
    {"id": "J4", "src": "E/IMPL-2", "check": "redundancy pooled as the mean of both levels per (seed, beta)"},
    {"id": "J5", "src": "E", "check": "each test draws exactly its declared budget and beta subset"},
    {"id": "K1", "src": "F", "check": "margin formula deterministic; margin file written once, never overwritten"},
    {"id": "K2", "src": "F", "check": "m > 0"},
    {"id": "K3", "src": "F", "check": "if planned-n TOST power < 0.80 the decision is PENDING, never silently continued"},
]

# --------------------------------------------------------------------------- #
# IMPLEMENTATION DECISIONS - ambiguities in rev. 2, resolved BEFORE measuring
# --------------------------------------------------------------------------- #
IMPLEMENTATION_DECISIONS: List[Dict[str, str]] = [
    {"id": "IMPL-1",
     "ambiguity": "A4 names two latents (zeta, eta) with corr rho_c but not which is held fixed.",
     "decision": "eta (tokens) is the OWN latent: eta = z1, zeta = rho_c*z1 + sqrt(1-rho_c^2)*z0.",
     "basis": "B2 states 'w is drawn from its own latent, independent of rho_c'. Taken literally, "
              "w is then bit-identical across all beta and gamma levels for a seed, so a token-CV "
              "rejection removes the whole seed and every Friedman block stays complete.",
     "status": "needs sign-off"},
    {"id": "IMPL-2",
     "ambiguity": "E says 'unit = base instance' and 'pooled over redundancy; paired by seed'.",
     "decision": "Analysis value per (seed, beta, budget) = mean over the two redundancy levels. "
                 "n = 60 independent seeds per test.",
     "basis": "The design's power analysis (G) uses n = 60. The two redundancy levels of a seed share "
              "r, w and topics (IMPL-1), so treating them as independent blocks would pseudo-replicate.",
     "status": "needs sign-off"},
    {"id": "IMPL-3",
     "ambiguity": "E gives no test statistic for T5-T8 and T10.",
     "decision": "T6/T7/T10: one-sample two-sided Wilcoxon on per-seed orthogonal-polynomial contrast "
                 "scores. T5: TOST as two one-sided Wilcoxon tests, p = max. T8: one-sided Wald z from "
                 "mixedlm(mean|Delta_gap| ~ C(beta) + D_centred, groups=seed). T1/T9: Friedman.",
     "basis": "Keeps the family non-parametric, consistent with T2-T4.",
     "status": "needs sign-off"},
    {"id": "IMPL-4",
     "ambiguity": "H lists the beta, marginal, CV, usable-doc, D and redundancy checks under a 5-seed pilot.",
     "decision": "Those generator-only checks run on the planned confirmatory ensemble (seeds 0-59). "
                 "Pilot 5-seed beta means are reported descriptively only.",
     "basis": "Step 3C.1 locked beta as a population-level parameter over the planned ensemble. "
              "Pre-measurement arithmetic: sd(beta_hat) <= 0.57, so SE over 5 seeds ~ 0.25 > 0.15 "
              "tolerance - a 5-seed check would test estimator noise, not the generator.",
     "status": "needs sign-off"},
    {"id": "IMPL-5",
     "ambiguity": "H does not say which seeds the pilot uses.",
     "decision": "Pilot seeds 1000-1004, disjoint from confirmatory seeds 0-59. No compared-method "
                 "selection is computed on confirmatory seeds during preflight (outcome-blind): only "
                 "generator properties and the unconstrained saturation ILP.",
     "basis": "m must not be estimated from data that later enter T5.",
     "status": "needs sign-off"},
    {"id": "IMPL-6",
     "ambiguity": "F does not fix the pooling or the order of stability check and rounding.",
     "decision": "m uses instance-level greedy_token_aware absolute gaps at tight over ALL pilot "
                 "instances (all beta, both gamma). Stability check on the UNROUNDED median and IQR; "
                 "then round to 2 decimals.",
     "basis": "'at the tight budget' carries no beta restriction; checking stability before rounding "
              "avoids rounding hiding instability.",
     "status": "needs sign-off"},
    {"id": "IMPL-7",
     "ambiguity": "F says 'the pilot's sd' without naming the quantity.",
     "decision": "sd of the seed-level Delta_gap at tight, beta = 2.0 - the T5 analysis unit. "
                 "Power by the design's normal approximation: 2*Phi(m*sqrt(n)/sd - z_0.95) - 1.",
     "basis": "Power must refer to the unit T5 actually tests.",
     "status": "needs sign-off"},
    {"id": "IMPL-8",
     "ambiguity": "The design gives only KS for marginal invariance and no tolerance for usable-doc balance.",
     "decision": "Added B3 (quantile tolerance 0.25*sigma) and D2 (usable-count level difference <= 1.5), "
                 "declared here before measuring.",
     "basis": "6B requires distributional summaries beyond KS and a check for systematic differences.",
     "status": "needs sign-off"},
]


# =========================================================================== #
# AMENDMENT 6.1 - signed off by the study owner after the 6B.1 repair analysis
# =========================================================================== #
AMENDMENT_6_1 = {
    "version": "6.1-amended",
    "source": "docs/elasticity_design_repair.md sec. 8 + owner decisions 1-12",
    "decisions": {
        "1_tight_budget": "rho = 0.25 unchanged",
        "2_budget_formulation": "W_max = rho * W_sat unchanged",
        "3_G4": "descriptive diagnostic only; no longer a locked guard",
        "4_G4_prime": "locked guard: the two cheapest documents fit within W_max at every budget",
        "5_TOST": "T5, K1-K3 and results/elasticity/tost_margin.json RETIRED; artifact preserved",
        "6_family": "9 tests T1-T4, T6-T10; Holm over 9",
        "7_test_statistic": "sign-flip permutation test on seed-level mean Delta_gap, zeros retained",
        "8_T8": "divergence probability on centred D, mixed model, one-sided > 0",
        "9_exploratory": "one-document optima by beta; divergence/effect decomposition; rho = 0.30",
        "10_SESOI": "3% of the mean ILP optimum",
        "11_seed_cap": "required n > 300 -> STOP, redesign",
        "12_frozen": "all other factors unchanged",
    },
}

SESOI_FRACTION_OF_OPTIMUM = 0.03
MAX_CONFIRMATORY_SEEDS = 300
VARIANCE_PILOT_SEEDS: Tuple[int, ...] = tuple(range(2000, 2040))
POWER_TARGET = 0.80
FAMILY_ALPHA = 0.05
HOLM_ALPHA_FOR_POWER = FAMILY_ALPHA / CONFIRMATORY_FAMILY_SIZE      # = .05/9

# Permutation test (decision 7). The repair doc requires "10,000 flips, fixed RNG seed
# declared". The value below is declared here, before any confirmatory or pilot data.
PERMUTATION_N_FLIPS = 10_000
PERMUTATION_RNG_SEED = 61_000

# G4' semantics: a LOCKED PREFLIGHT INVARIANT (failure -> design STOP), exactly like
# every other locked invariant in this protocol. It is NOT an exclusion criterion, so it
# can never drop individual instances (which could correlate with beta via W_sat).
G4_PRIME_SEMANTICS = "locked preflight invariant; failure stops the design; never excludes instances"

# Divergence (decision 8): two selections diverge iff their index SETS differ.
DIVERGENCE_DEFINITION = "sorted(selected_indices_objective) != sorted(selected_indices_token_aware)"

RETIRED_ARTIFACTS = {
    "results/elasticity/tost_margin.json": {
        "retired_at": "6.1-amended",
        "reason": "margin rule degenerate (median gap exactly 0; fallback triggered by IQR = 2.8e-17 float noise); "
                  "T5 answers no hypothesis's distinctive prediction",
        "sidecar": "results/elasticity/tost_margin.json.RETIRED.json",
    },
}

EXPLORATORY_ONLY = (
    "one-document optima and ILP cardinality by beta",
    "hurdle decomposition: P(divergence) and conditional Delta_gap given divergence, per beta and budget",
    "tight rho = 0.30 sensitivity level",
    "95% interval for mean Delta_gap per beta level with SESOI band (replaces T5)",
    "G4 statistics (n_selected distribution per method at tight)",
)

# =========================================================================== #
# AMENDMENT 6.2 (PROPOSED) - resolution of U-SC1..U-SC5 and T8 re-specification
# =========================================================================== #
DECLARATIONS_6_1 = {
    "permutation_rng_seed_61000": "CONFIRMED",
    "G4_prime_failure_stops_design": "CONFIRMED",
    "divergence_is_set_inequality": "CONFIRMED",
    "T8_linear_mixed_model_carryover": "NOT CONFIRMED -> re-specified in T8_SPEC below",
}

N_MIN_CONFIRMATORY_SEEDS = 60          # U-SC4: n may increase above 60, never decrease below

# T8 - primary analysis frozen for a binary outcome, before any pilot data.
T8_SPEC = {
    "outcome": "Y_{s,beta,gamma} in {0,1}: 1 iff greedy_objective and greedy_token_aware select "
               "different document SETS, at the tight budget; instance level (both redundancy levels)",
    "predictor": "Dc_{s,beta} = D_{s,beta} - mean_s D_{s,beta}  (centred within beta level)",
    "estimand": "b = within-level slope of P(Y=1) on D, in PROBABILITY POINTS per unit D "
                "(identity link, level fixed effects)",
    "estimator": "b_hat = sum_s S_s / sum_{s,beta,gamma} Dc^2, where "
                 "S_s = sum_{beta,gamma} Dc_{s,beta} * (Y_{s,beta,gamma} - Ybar_beta)",
    "test": "one-sided seed-cluster sign-flip test on the per-seed scores S_s: "
            "p = (1 + #{mean(sign*S) >= mean(S)}) / (10000 + 1), RNG seed 61000",
    "H0": "b = 0", "H1": "b > 0",
    "why_identity_link": "The SESOI is declared on the absolute probability scale (U-SC2). A logistic "
                         "model estimates a log-odds slope; converting it to probability points depends "
                         "on the fitted outcome probabilities, i.e. on the data. The identity-link "
                         "estimand is on the SESOI scale by construction.",
    "why_cluster_signflip": "No Gaussian random-effect assumption for a bounded binary outcome; seeds are "
                            "the independent unit; consistent with the family's sign-flip machinery.",
    "rejected": ["Gaussian linear mixed model on the seed-level share (6.1; not confirmed)",
                 "logistic GLMM (estimand not on the SESOI scale)"],
}

# --------------------------------------------------------------------------- #
# Seed-count rule - 6.2-proposed-r2. Implemented in experiments/elasticity/seed_count.py.
# --------------------------------------------------------------------------- #
GENERAL_EDGE_INVARIANT = ("No edge-case rule may replace an undefined or mathematically invalid statistical "
                          "quantity with an arbitrary numerical value merely to permit continuation. Degenerate "
                          "or invalid inputs cause the seed-count calculation to stop.")

SEED_COUNT_RULE = {
    "version": "6.2-locked",
    "regime": "R2",
    "status": "LOCKED",
    "locked_on_basis_of": {"audit": "results/elasticity/amendment_6_2_lock_candidate_audit.json", "sha256": "c62b9ac5b50c1077e2fd70559795d7c6271eb21852df1d0fd18f4061a9500b75",
                           "result": "all owner checks 1-15 true; protocol_lockable true; no unresolved item"},
    "role_of_pilot": "PARAMETER ESTIMATION ONLY",
    "before_pilot": "The complete deterministic function F: (pilot quantities) -> n_final is frozen, together "
                    "with every FIXED quantity it uses (delta_b via OPT_b on B, Delta_D and E_Q on B), the "
                    "SESOI, the tests, the effect-size mappings, the power target, alpha/Holm, the formulas, "
                    "the minimum-60 rule and all edge-case rules.",
    "after_pilot": "Observed pilot quantities are substituted into the already-frozen F. The pilot cannot "
                   "determine or modify any of the items listed under before_pilot.",
    "populations": {
        "P": "P = the ACCEPTED seeds among 2000-2039 under the locked token-CV acceptance policy; rejected "
             "seeds are excluded from P and reported",
        "B": "fixed baseline population, confirmatory seeds 0-59 (all 60 accepted in 6B); independent of n",
    },
    "constants": {"k": 6, "alpha": "0.05/9", "power": 0.80, "nonzero_tol": 1e-12,
                  "rho_tol": 1e-10, "n_search": "integers 1..300", "N_min": 60, "N_max": 300,
                  "ARE": "0.955*k/(k+1)", "T8_probability_sesoi": 0.03},
    "fixed_quantities_from_B": {
        "OPT_b": "mean of the PROVEN ILP optimum at budget b over s in B, 6 beta, 2 gamma (720 instances), "
                 "W_max = build_budgets(rho_b * W_sat); reference solver only; any unproven solve -> STOP",
        "delta_b": "0.03 * OPT_b",
        "Delta_D": "mean over the 6 beta levels j of [q90_j - q10_j], q = quantile of D_{s,j} over s in B "
                   "(one D per (s, beta); D is independent of gamma and budget). Generator-only, outcome-blind. "
                   "Chosen BEFORE the pilot; no alternative D contrast may be searched after the pilot. "
                   "Percentile convention: PERCENTILE_CONVENTION (locked).",
        "E_Q": "mean over s in B of Q_s, Q_s = sum_j sum_gamma Dc_{s,j}^2, Dc centred within beta level over B",
    },
    "pilot_quantities_from_P": {
        "pi_b": "share of instances (s in P, beta, gamma) with |x| > 1e-12 at budget b",
        "M2_b": "mean of x^2 over instances with |x| > 1e-12 at budget b (none -> STOP)",
        "rho_red_b": "Pearson over pairs (x_{s,j,low,b}, x_{s,j,high,b}), s in P, all j (not finite -> STOP)",
        "rho_lvl_b": "mean of the 15 pairwise Pearson correlations between beta columns of the seed-level "
                     "redundancy-mean x at budget b over s in P (any not finite -> STOP)",
        "sigma_S": "sample SD (ddof=1) over s in P of T8 scores S_s at tight, Dc and Y centred within beta "
                   "level over P",
    },
    "formulas": {
        "sigma2_b": "(pi_b*M2_b - delta_b^2) * (1 + rho_red_b) / 2",
        "T2_T3_T4": "n = max(1, ceil((z2 * sqrt(sigma2_tight) / delta_tight)^2)),  z2 = z_{1-alpha/2} + z_{0.80}",
        "T6_T7_T10": "L = delta_b * sum c^2 / (max c - min c); sd = sqrt(sigma2_b * sum c^2 * (1 - rho_lvl_b)); "
                     "n = max(1, ceil((z2 * sd / L)^2)); b = tight for T6,T7 and medium for T10",
        "T1_T9_friedman": "f = (delta_b / sqrt(sigma2_b)) * sqrt(1/(2k)); lambda(n) = ARE*n*k*f^2/(1 - rho_lvl_b); "
                          "power(n) = P[ncchi2(k-1, lambda(n)) > chi2_{1-alpha; k-1}]; "
                          "n = min{n in 1..300 : power(n) >= 0.80}; W_min = ARE*k*f^2/((k-1)(1 - rho_lvl_b)); "
                          "b = tight for T1 and loose for T9",
        "T8": "b_min = 0.03 / Delta_D  (0.03 is a PROBABILITY-POINT DIFFERENCE across the Delta_D contrast, "
              "NOT a slope of 0.03); n = max(1, ceil((z1 * sigma_S / (b_min * E_Q))^2)), z1 = z_{1-alpha} + z_{0.80}",
        "n_final": "max(60, max over the 9 tests of n_t); any n_t > 300 -> STOP, redesign",
    },
    "edge_case_rules": {
        "a": "pi_b*M2_b - delta_b^2 <= 0 -> STOP (degenerate). The -delta^2 term is never dropped.",
        "b": "any required Pearson correlation not finite (zero-variance vector) -> STOP. No substitution.",
        "c": "rho_lvl must satisfy -1/(k-1) <= rho_lvl < 1 within tolerance 1e-10 at the boundaries; materially "
             "outside -> STOP; no clamping; no n_t = 1 shortcut. A value within tolerance of or above 1 makes "
             "1 - rho_lvl <= 0, which is invalid -> STOP under GENERAL_EDGE_INVARIANT.",
        "d": "search n over 1..300; power < 0.80 at 300 -> STOP, redesign; n_final = max(60, max_t n_t).",
        "e": "nonzero means |x| > 1e-12.",
        "f": "sigma_S is the sample SD with ddof = 1.",
        "g": "all centring is within beta level; sigma_S over P; E_Q over B.",
        "h": "RULE_H (locked): The pilot population is insufficient only when a required statistic is "
             "undefined or fails an explicitly declared mathematical/data-quality condition. Token-CV rejections "
             "are applied using the existing locked acceptance policy; rejected seeds are excluded from P and "
             "reported. A rejection does not by itself stop the pilot. The calculation stops only if, after "
             "applying the acceptance policy, any required statistic cannot be computed or its declared "
             "mathematical/data-quality requirements are not satisfied. Declared conditions: RULE_H_CONDITIONS.",
        "i": "any unproven reference ILP while constructing OPT_b -> STOP; never substituted.",
        "general": "sigma2_b <= 0, sigma_S <= 0, Delta_D <= 0, E_Q <= 0, OPT_b <= 0, or any non-finite quantity "
                   "-> STOP under GENERAL_EDGE_INVARIANT.",
    },
    "resolved_in_r2": {"U-SC0": "R2", "U-SC1": "frozen Friedman formula", "U-SC2a": "Delta_D on B, 10-90 range",
                       "U-SC5a": "option (i): OPT_b over B = seeds 0-59, independent of n",
                       "circularity": "n does not determine the population used to define delta"},
    "resolved_in_lock_candidate": {"U-SC2a-Q": "PERCENTILE_CONVENTION", "U-SC-H": "RULE_H, option (i)"},
    "unresolved": [],
}

PERCENTILE_CONVENTION = {
    "name": "linear interpolation between adjacent order statistics (Hyndman-Fan type 7), stated explicitly; "
            "NO library default is relied upon",
    "definition": "For beta level j, sort the N values D_{s,j}, s in B, ascending into x_(0) <= ... <= x_(N-1) "
                  "(0-based). For p in (0,1): h = (N-1)*p; lo = floor(h); frac = h - lo; "
                  "P_p = x_(lo) + frac * (x_(lo+1) - x_(lo)). If h is an integer, P_p = x_(h).",
    "N": 60,
    "P10": "h = 59 * 0.10 = 5.9  -> x_(5) + 0.9 * (x_(6) - x_(5))",
    "P90": "h = 59 * 0.90 = 53.1 -> x_(53) + 0.1 * (x_(54) - x_(53))",
    "arithmetic": "h computed as an exact rational (N-1)*num/den, so no floating-point floor error",
    "Delta_D": "mean over the 6 beta levels of [P90(D_j,B) - P10(D_j,B)]",
    "data_requirements": "exactly N = 60 finite D values per beta level (one per seed in B); otherwise STOP",
    "status": "LOCKED; NOT computed",
}

RULE_H_CONDITIONS = [
    {"id": "H1", "condition": "every seed in 2000-2039 receives exactly one disposition (accepted / rejected) under "
                              "the locked token-CV policy; rejected seeds are reported",
     "basis": "rule h wording"},
    {"id": "H2", "condition": "every accepted seed has the complete grid: 6 beta x 2 gamma x 3 budgets, with both "
                              "greedy_objective and greedy_token_aware scores and selections",
     "basis": "mathematically required by the definitions of rho_red (gamma pairs), rho_lvl (beta columns) and "
              "the T8 scores (sum over beta and gamma)"},
    {"id": "H3", "condition": "|P| >= 2",
     "basis": "mathematically required: sigma_S is a sample SD with ddof = 1 and rho_lvl is a between-seed "
              "Pearson correlation; neither is defined for fewer than 2 seeds. No other minimum count is imposed."},
    {"id": "H4", "condition": "rules a, b, c, e (M2_b defined), f and the general invariant",
     "basis": "already-locked edge-case rules"},
]


REPRODUCIBILITY_LIMITATIONS = [
    {"id": "RL-1", "issue": "git commit hash unobtainable: git is blocked on this machine by an unaccepted "
                            "Xcode licence. Artifacts record git as 'unavailable'; file-level sha256 hashes "
                            "are recorded instead. Not a reason to run the pilot.",
     "remedy": "accept the licence (sudo xcodebuild -license) and commit before the confirmatory run"},
    {"id": "RL-2", "issue": "no byte-level hash baseline for results/controlled/ existed before 6.1"},
    {"id": "RL-4", "issue": "embedder.py was found modified on disk (2026-09-16 15:53 local, while open in the IDE; "
                            "not by any command in this work): one added line of four spaces in build_corpus; "
                            "AST-identical to the committed version. RESTORED byte-exact from git object blob "
                            "f76e9826183a (commit 8fe4d34b2106); sha256 again c22a456f2f46. The changed hash was "
                            "NOT re-recorded."},
    {"id": "RL-3", "issue": "INCIDENT 2026-09-16T10:19:43Z: a failed patch left the old audit script in place, "
                            "and running it OVERWROTE results/elasticity/amendment_6_1_audit.json with a run "
                            "under 6.2-proposed. The original 6.1 audit bytes are lost. The overwritten file "
                            "was renamed '...OVERWRITTEN_20260916T101943Z_old_script_run_under_6_2_proposed.json'. "
                            "The 6.1 record survives only as transcribed in docs/elasticity_amendment_6_1.md "
                            "(protocol.py sha 10a638f3580bb6e5, analysis.py sha a05662c0eff73d0c). No "
                            "experimental data were involved. The audit script now refuses to overwrite."},
]


class ProtocolNotLockedError(RuntimeError):
    """Raised when a step requires a protocol component that is not yet locked."""


def assert_seed_count_rule_locked() -> None:
    if SEED_COUNT_RULE["status"] != "LOCKED":
        raise ProtocolNotLockedError(
            "seed-count rule is " + SEED_COUNT_RULE["status"] + " ("
            + ", ".join(u["id"] for u in SEED_COUNT_RULE["unresolved"]) + "); the variance pilot may not run")


# Machine-readable invariant status after the amendment.
_AMENDED_STATUS = {"G4": "descriptive", "K1": "retired", "K2": "retired", "K3": "retired"}
for _inv in INVARIANTS:
    _inv["status"] = _AMENDED_STATUS.get(_inv["id"], "locked")
    if _inv["id"] == "J1":
        _inv["check"] = "confirmatory family has exactly 9 tests T1-T4,T6-T10 in family F1 (amended 6.1)"
INVARIANTS.extend([
    {"id": "G4P", "src": "6.1-dec4", "status": "locked",
     "check": "two cheapest documents fit within W_max at every budget, every instance"},
    {"id": "L1", "src": "6.1-dec5", "status": "locked",
     "check": "retired TOST artifact byte-identical to its recorded hash and unreadable by the analysis"},
    {"id": "M1", "src": "6.1-dec11", "status": "locked",
     "check": "seed-count rule LOCKED before any variance-pilot data are generated"},
])
