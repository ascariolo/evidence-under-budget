r"""Frozen protocol constants for the controlled benchmark.

Single source of truth for the experimental design, imported by the generator,
the pilot and (later) the full runner and the analysis, so the four cannot drift
apart. Specification: ``docs/controlled_benchmark_design.md``.

NOTHING HERE MAY BE CHANGED IN RESPONSE TO A RESULT. Factor levels are fixed a
priori. Two documented deviations from the design text are recorded in
``DESIGN_DEVIATIONS`` below and in docs/controlled_benchmark_implementation.md.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

#: Bumped whenever the controlled protocol changes materially.
CONTROLLED_PROTOCOL_VERSION = "0.1-pilot"

# --------------------------------------------------------------------------- #
# Factor 1: relevance-length relationship
# --------------------------------------------------------------------------- #
#: level -> (Gaussian-copula correlation, target log-log elasticity beta)
#: Level D shares condition A's copula (rho_c = 0) and adds the adversarial
#: token intervention on top; its elasticity is mixed by construction.
RL_CONDITIONS: Dict[str, Dict[str, object]] = {
    "A": {"name": "independent", "rho_c": 0.0, "target_beta": 0.0,
          "adversarial": False},
    "B": {"name": "positive_sublinear", "rho_c": 0.5, "target_beta": 0.5,
          "adversarial": False},
    "C": {"name": "negative", "rho_c": -0.5, "target_beta": -0.5,
          "adversarial": False},
    "D": {"name": "adversarial_mismatch", "rho_c": 0.0, "target_beta": None,
          "adversarial": True},
}

# --------------------------------------------------------------------------- #
# Beta: four distinct quantities, never to be conflated
# --------------------------------------------------------------------------- #
# 1. beta_target      - the GENERATOR PARAMETER. A population-level property of
#                       the instance-generating distribution, realised exactly
#                       through the copula correlation rho_c. Declared in
#                       RL_CONDITIONS above; never estimated, never inferred.
# 2. beta_hat(seed)   - the PER-INSTANCE ESTIMATOR. An OLS log-log slope fitted
#                       to the N=25 documents of one realisation. A finite-sample
#                       MEASUREMENT of (1), carrying sampling error.
# 3. mean(beta_hat)   - the AGGREGATE ESTIMATE over the fixed seed ensemble
#                       (seeds 0-39). Its standard error is sd(beta_hat)/sqrt(40).
# 4. BETA_TOLERANCE   - the LOCKED VALIDATION CRITERION, applied to (3) only.
#
# LOCKED VALIDATION CRITERION (step 3C.1):
#
#     |mean(beta_hat over seeds 0-39) - beta_target| <= BETA_TOLERANCE
#
# applied per beta condition (A, B, C). Condition D has no beta target.
#
# The per-seed criterion |beta_hat(seed) - beta_target| <= BETA_TOLERANCE is
# EXPLICITLY NOT REQUIRED and must not be used to accept or reject a seed.
# Rationale: beta_hat is estimated from only N=25 documents, and the 40-seed
# pre-flight measured per-instance sd(beta_hat) = 0.15-0.19, i.e. 1.5-1.9x the
# tolerance itself. A per-seed acceptance rule would therefore select
# realisations by ESTIMATOR NOISE rather than validate the generator parameter,
# which is a form of seed selection. The generator uses the exact target
# parameter; beta_hat merely measures it.
#
# Per-seed beta_hat values ARE still recorded and reported for every instance,
# because they quantify finite-sample estimator variability and serve as a
# covariate in the continuous companion model. Reporting them is required;
# filtering on them is forbidden.
BETA_TOLERANCE = 0.10   # LOCKED at step 3C, criterion fixed at 3C.1. Never widened.

#: The criterion above, as a string, for embedding in result artifacts.
BETA_CRITERION = ("|mean(beta_hat over the fixed seed ensemble) - beta_target| "
                  "<= BETA_TOLERANCE, applied per condition. Per-seed deviations "
                  "are reported but are NOT an acceptance criterion.")

# --------------------------------------------------------------------------- #
# Factor 2: budget tightness, instance-relative
# --------------------------------------------------------------------------- #
#: rho = W_max / W_sat, where W_sat is the token mass of the UNCONSTRAINED
#: optimum (proven-optimal ILP at budget = total token mass).
BUDGET_LEVELS: Dict[str, float] = {"tight": 0.25, "medium": 0.60, "loose": 1.50}

# --------------------------------------------------------------------------- #
# Factor 3: redundancy
# --------------------------------------------------------------------------- #
#: Within-topic alignment of the query-orthogonal component.
#:
#: LOCKED at step 3C. This is a LOW vs MODERATE redundancy contrast. It is NOT
#: calibrated to the v0 retrieved-document similarity and must never be
#: described as such: measured mean pairwise similarity is 0.093 (low) and
#: 0.178 (moderate) against 0.330-0.405 for the real corpus. See DEV-2.
REDUNDANCY_LEVELS: Dict[str, float] = {"low": 0.15, "high": 0.70}

# --------------------------------------------------------------------------- #
# Held constant
# --------------------------------------------------------------------------- #
N_DOCS = 25            # matches the original benchmark; keeps the ILP tractable
EMBED_DIM = 64
N_TOPICS = 5

# Relevance marginal: Beta(a, b) mapped onto [RELEVANCE_MIN, RELEVANCE_MAX].
# Provenance: v0 corpus measured rel in [0.186, 0.525], mean 0.363; the support
# here is deliberately slightly wider.
RELEVANCE_BETA_A = 2.0
RELEVANCE_BETA_B = 3.0
RELEVANCE_MIN = 0.05
RELEVANCE_MAX = 0.60

# Token marginal: log-normal. Provenance: v0 corpus measured median 77,
# mean 84, max 205, right-skewed. exp(LOG_TOKEN_MU) = 77 by construction.
LOG_TOKEN_MU = 4.343805421853684     # = log(77)
LOG_TOKEN_SIGMA = 0.50
TOKEN_MIN = 8
TOKEN_MAX = 400

# Condition D adversarial intervention.
ADVERSARIAL_TOP_K = 4
ADVERSARIAL_MULTIPLIER = 6.0
ADVERSARIAL_CAP_FRACTION = 0.45      # of the pre-intervention W_sat

# Frozen solver parameters - NOT tuned here, carried from the repaired baseline.
OBJECTIVE_LAMBDA = 0.1
MMR_LAMBDA = 0.5
ILP_TIME_LIMIT_S = 600.0

# --------------------------------------------------------------------------- #
# Non-degeneracy thresholds (design section 13)
# --------------------------------------------------------------------------- #
#: If token cost is near-constant then Delta_i / w_i is proportional to Delta_i
#: and the two greedy methods are IDENTICAL BY CONSTRUCTION. This is the single
#: most important guard in the suite.
MIN_TOKEN_CV = 0.30
MIN_RELEVANCE_STD = 0.05
MIN_SIMILARITY_STD = 0.01
MIN_GAP_STD_FOR_RESOLUTION = 0.01    # across seeds, on greedy_objective's gap
GRAM_EIGENVALUE_TOLERANCE = -1e-8
RELEVANCE_EXACTNESS_TOLERANCE = 1e-10

# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #
def primary_cells() -> List[Tuple[str, str]]:
    """The 4 x 2 (relevance-length x redundancy) base cells of the design."""
    return [(rl, red) for rl in RL_CONDITIONS for red in REDUNDANCY_LEVELS]


#: Arm E (realism anchor) uses the UNCHANGED original text generator and is
#: never pooled with the synthetic factorial.
REALISM_ANCHOR_ARM = "E_realism_anchor"

DESIGN_DEVIATIONS = [
    {
        "id": "DEV-1",
        "topic": "Condition D token cap is circular in the design text",
        "design_text": "token cost multiplied by m_adv = 6, capped at 0.45 * W_sat",
        "problem": ("W_sat is computed from the tokens, so a cap expressed in "
                    "terms of W_sat cannot be applied while the tokens are "
                    "being assigned."),
        "resolution": ("Two-pass. Pass 1 computes W_sat on the pre-intervention "
                       "tokens (w_sat_pre). Pass 2 applies the multiplier capped "
                       "at 0.45 * w_sat_pre, then recomputes the final W_sat on "
                       "the modified instance. Both values are recorded. This "
                       "preserves the stated intent (adversarial documents stay "
                       "individually feasible at the medium budget) without "
                       "circularity, at the cost of one extra ILP solve for "
                       "condition D only."),
        "status": "implemented; needs sign-off",
    },
    {
        "id": "DEV-2",
        "topic": "Redundancy calibration target is unreachable by the stated construction",
        "design_text": ("gamma = 0.15 -> mean off-diagonal similarity ~0.2; "
                        "gamma = 0.70 -> ~0.45, 'calibrated to the measured v0 "
                        "corpus (mean pairwise cosine 0.405)'"),
        "problem": ("Measured: gamma = 0.15 -> 0.094, gamma = 0.70 -> 0.194. "
                    "With K = 5 topics, cross-topic pairs contribute ~0 and are "
                    "80% of all pairs, so mean similarity ceilings at about "
                    "r_bar^2 + 0.93 * gamma / K ~ 0.26 even at gamma = 1. The "
                    "v0 level of 0.405 is NOT reachable at K = 5."),
        "resolution": ("LOCKED at step 3C by explicit instruction: K, gamma and "
                       "the construction are UNCHANGED. The factor is redefined "
                       "in wording only - it is a LOW vs MODERATE redundancy "
                       "contrast, not a calibration to the v0 corpus, and the "
                       "calibration claim has been retracted from the design "
                       "document. gamma is implemented exactly "
                       "as specified (0.15 / 0.70), which still gives a genuine "
                       "~2x redundancy contrast and is non-degenerate. Only the "
                       "design's claim that 'high' equals the v0 redundancy "
                       "level is false. Changing K, gamma, or the construction "
                       "would redefine an experimental condition and was NOT "
                       "done."),
        "status": "LOCKED at step 3C - wording retracted, parameters unchanged",
    },
    {
        "id": "DEV-3",
        "topic": "Beta tolerance was not specified in the design",
        "design_text": "'within the tolerance specified by the design'",
        "problem": "No tolerance is stated anywhere in the design document.",
        "resolution": ("BETA_TOLERANCE = 0.10 on |mean(beta_hat) - target| "
                       "across seeds, chosen before measuring and recorded here."),
        "status": "LOCKED at step 3C; criterion disambiguated at 3C.1 (see DEV-4)",
    },
    {
        "id": "DEV-4",
        "topic": "Beta criterion was ambiguous between per-condition mean and per-seed",
        "design_text": ("step 3C text asked to verify |beta_hat - beta_target| "
                        "<= 0.10 'for each condition and seed', while protocol.py "
                        "implemented the criterion on the per-condition mean."),
        "problem": ("The two readings disagree. The aggregate criterion passes for "
                    "A, B and C with 3-4x margin; the per-seed reading fails 40-60% "
                    "of instances. Per-instance sd(beta_hat) is 0.15-0.19, larger "
                    "than the 0.10 tolerance, so no generator at N=25 could satisfy "
                    "the per-seed reading."),
        "resolution": ("RESOLVED at step 3C.1 by explicit decision: beta is a "
                       "POPULATION-LEVEL parameter of the generator, so the locked "
                       "criterion is |mean(beta_hat across seeds) - beta_target| "
                       "<= 0.10 per condition. The per-seed criterion is explicitly "
                       "NOT required. Per-seed values remain reported as estimator "
                       "variability. No tolerance was widened, no seed resampled, "
                       "no generator parameter changed."),
        "status": "RESOLVED and LOCKED at step 3C.1",
    },
]

#: Recorded at step 3C.1 so every downstream artifact can assert provenance.
PROTOCOL_LOCK = {
    "locked_at": "step 3C.1",
    "seeds": "0-39 fixed; no seed resampled, excluded or reordered",
    "beta_criterion": BETA_CRITERION,
    "beta_tolerance": BETA_TOLERANCE,
    "beta_tolerance_widened": False,
    "beta_targets_changed": False,
    "n_docs": N_DOCS,
    "generator_parameters_changed": False,
    "optimizer_code_changed": False,
    "benchmark_code_changed": False,
    "experimental_factors_changed": False,
    "redundancy_interpretation": "LOW vs MODERATE; not calibrated to the v0 corpus",
    "step_3C_pass_remains_valid": True,
}
