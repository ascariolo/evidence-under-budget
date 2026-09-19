r"""Frozen seed-count function (protocol SEED_COUNT_RULE, regime R2). PURE: no file or data access.

Three separated layers, so that no pilot quantity can reach a protocol definition:

  1. FIXED quantities, computed from the baseline population B = seeds 0-59 only:
       fixed_opt_b(...)      -> OPT_b, delta_b
       fixed_E_Q(...)        -> E_Q
       fixed_delta_D(...)    -> Delta_D            (protocol.PERCENTILE_CONVENTION, explicit type 7)
  2. PILOT quantities, estimated from the variance-pilot population P:
       pilot_quantities(...) -> pi_b, M2_b, rho_red_b, rho_lvl_b, sigma_S
       assert_pilot_sufficient(...)                (protocol rule h, RULE_H_CONDITIONS H1-H3)
  3. The frozen map (fixed, pilot) -> n_final:
       n_final(fixed, pilot)

Every edge-case rule raises SeedCountStop. Nothing is clamped, substituted or defaulted
(protocol.GENERAL_EDGE_INVARIANT).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from experiments.elasticity import protocol as EP

K = len(EP.BETA_LEVELS)                       # 6
ALPHA = EP.FAMILY_ALPHA / EP.CONFIRMATORY_FAMILY_SIZE
POWER = EP.POWER_TARGET
N_MIN, N_MAX = EP.N_MIN_CONFIRMATORY_SEEDS, EP.MAX_CONFIRMATORY_SEEDS
NONZERO_TOL = 1e-12
RHO_TOL = 1e-10
ARE = 0.955 * K / (K + 1)
T8_PROB_SESOI = 0.03
BUDGET_OF = {"T1": "tight", "T2": "tight", "T3": "tight", "T4": "tight", "T6": "tight",
             "T7": "tight", "T8": "tight", "T9": "loose", "T10": "medium"}


class SeedCountStop(RuntimeError):
    """A degenerate or invalid input: the seed-count calculation stops."""


class SeedCountNotLocked(RuntimeError):
    """A component of the rule is not yet locked."""


def _finite_positive(name: str, v: float) -> float:
    if not (isinstance(v, (int, float)) and math.isfinite(v)):
        raise SeedCountStop(f"{name} is not finite ({v})")
    if v <= 0:
        raise SeedCountStop(f"{name} = {v} <= 0 (degenerate)")
    return float(v)


# --------------------------------------------------------------------------- #
# 1. FIXED quantities (baseline population B)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FixedQuantities:
    delta: Mapping[str, float]        # by budget
    delta_D: float
    E_Q: float


def fixed_opt_b(ilp_records: Sequence[Mapping[str, Any]], baseline_seeds: Sequence[int]) -> Dict[str, Dict[str, float]]:
    """ilp_records: reference-solver results with seed, beta_target, redundancy_level, budget_level,
    objective_score, ilp_proven_optimal. Rule i: any unproven -> STOP. Requires the complete B grid."""
    B = set(baseline_seeds)
    out = {}
    for b in EP.BUDGET_LEVELS:
        recs = [r for r in ilp_records if r["budget_level"] == b and r["seed"] in B]
        expected = len(B) * K * len(EP.REDUNDANCY_LEVELS)
        if len(recs) != expected:
            raise SeedCountStop(f"OPT_{b}: {len(recs)} records, expected {expected}")
        if not all(bool(r["ilp_proven_optimal"]) for r in recs):
            raise SeedCountStop(f"OPT_{b}: an unproven reference ILP (rule i)")
        opt = _finite_positive(f"OPT_{b}", float(np.mean([r["objective_score"] for r in recs])))
        out[b] = {"OPT": opt, "delta": EP.SESOI_FRACTION_OF_OPTIMUM * opt}
    return out


def fixed_E_Q(D_by_seed_beta: Mapping[Tuple[int, float], float], baseline_seeds: Sequence[int]) -> float:
    """E_Q = mean_s sum_j sum_gamma Dc^2, Dc centred within beta over B (rule g). D is gamma-independent,
    so the gamma sum contributes a factor equal to the number of redundancy levels."""
    B = list(baseline_seeds)
    n_red = len(EP.REDUNDANCY_LEVELS)
    Q = {s: 0.0 for s in B}
    for b in EP.BETA_LEVELS:
        vals = np.array([D_by_seed_beta[(s, b)] for s in B], float)
        dc = vals - vals.mean()
        for s, d in zip(B, dc):
            Q[s] += n_red * d * d
    return _finite_positive("E_Q", float(np.mean(list(Q.values()))))


def percentile_type7(values: Sequence[float], num: int, den: int) -> float:
    """protocol.PERCENTILE_CONVENTION, written out - no library percentile is used.
    Sorted 0-based x_(0..N-1); h = (N-1)*num/den as an EXACT rational; lo = floor(h);
    P = x_(lo) + frac*(x_(lo+1) - x_(lo)); if h is an integer, P = x_(h)."""
    x = sorted(float(v) for v in values)
    N = len(x)
    if N < 2 or not all(math.isfinite(v) for v in x):
        raise SeedCountStop(f"percentile needs >= 2 finite values, got {N}")
    h_num = (N - 1) * num
    lo, rem = divmod(h_num, den)
    if rem == 0:
        return x[lo]
    return x[lo] + (rem / den) * (x[lo + 1] - x[lo])


def fixed_delta_D(D_by_seed_beta: Mapping[Tuple[int, float], float], baseline_seeds: Sequence[int]) -> float:
    """Delta_D = mean over the 6 beta levels of [P90 - P10] of D over B (N = 60 per level)."""
    B = list(baseline_seeds)
    N = EP.PERCENTILE_CONVENTION["N"]
    if len(B) != N or len(set(B)) != N:
        raise SeedCountStop(f"Delta_D requires exactly {N} distinct baseline seeds, got {len(set(B))}")
    ranges = []
    for b in EP.BETA_LEVELS:
        vals = [D_by_seed_beta[(s, b)] for s in B]
        ranges.append(percentile_type7(vals, 9, 10) - percentile_type7(vals, 1, 10))
    return _finite_positive("Delta_D", float(np.mean(ranges)))


# --------------------------------------------------------------------------- #
# 2. PILOT quantities (variance-pilot population P)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PilotQuantities:
    pi: Mapping[str, float]
    M2: Mapping[str, float]
    rho_red: Mapping[str, float]
    rho_lvl: Mapping[str, float]
    sigma_S: float


def _pearson(x: Sequence[float], y: Sequence[float], what: str) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0.0 or y.std() == 0.0:
        raise SeedCountStop(f"{what}: undefined Pearson (zero-variance vector), rule b")
    r = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(r):
        raise SeedCountStop(f"{what}: Pearson not finite, rule b")
    return r


def assert_pilot_sufficient(accepted: Sequence[int], rejected: Sequence[int],
                            grid_keys: Sequence[Tuple[int, float, str, str, str]]) -> Dict[str, Any]:
    """Rule h with RULE_H_CONDITIONS H1-H3 (H4 is enforced inside pilot_quantities / n_final).
    A token-CV rejection does NOT by itself stop the pilot. grid_keys: (seed, beta, redundancy, budget, method)."""
    planned = set(EP.VARIANCE_PILOT_SEEDS)
    acc, rej = set(accepted), set(rejected)
    if acc & rej or (acc | rej) != planned:                                                   # H1
        raise SeedCountStop("H1: every pilot seed needs exactly one disposition (accepted/rejected)")
    have = set(grid_keys)
    missing = [(s, b, g, bud, m) for s in sorted(acc) for b in EP.BETA_LEVELS for g in EP.REDUNDANCY_LEVELS
               for bud in EP.BUDGET_LEVELS for m in ("greedy_objective", "greedy_token_aware")
               if (s, b, g, bud, m) not in have]
    if missing:                                                                               # H2
        raise SeedCountStop(f"H2: incomplete grid for accepted seeds ({len(missing)} cells missing)")
    if len(acc) < 2:                                                                          # H3
        raise SeedCountStop(f"H3: |P| = {len(acc)} < 2 (ddof=1 SD and between-seed Pearson undefined)")
    return {"P": sorted(acc), "rejected_reported": sorted(rej), "n_P": len(acc)}


def pilot_quantities(x: Mapping[Tuple[int, float, str, str], float],
                     t8_instances: Sequence[Mapping[str, Any]], pilot_seeds: Sequence[int]) -> PilotQuantities:
    """x[(seed, beta, redundancy, budget)] = score(token_aware) - score(objective) for s in P."""
    from experiments.elasticity.analysis import t8_cluster_scores
    P = list(pilot_seeds)
    reds = sorted(EP.REDUNDANCY_LEVELS)
    pi, M2, rr, rl = {}, {}, {}, {}
    for b in EP.BUDGET_LEVELS:
        vals = np.array([x[(s, j, g, b)] for s in P for j in EP.BETA_LEVELS for g in reds], float)
        nz = np.abs(vals) > NONZERO_TOL                                       # rule e
        pi[b] = float(nz.mean())
        if not nz.any():
            raise SeedCountStop(f"M2_{b}: no nonzero instance (undefined)")
        M2[b] = float(np.mean(vals[nz] ** 2))
        rr[b] = _pearson([x[(s, j, reds[0], b)] for s in P for j in EP.BETA_LEVELS],
                         [x[(s, j, reds[1], b)] for s in P for j in EP.BETA_LEVELS], f"rho_red_{b}")
        seed_mean = {j: [np.mean([x[(s, j, g, b)] for g in reds]) for s in P] for j in EP.BETA_LEVELS}
        pairs = [(a, c) for i, a in enumerate(EP.BETA_LEVELS) for c in EP.BETA_LEVELS[i + 1:]]
        rl[b] = float(np.mean([_pearson(seed_mean[a], seed_mean[c], f"rho_lvl_{b}[{a},{c}]") for a, c in pairs]))
    scores = t8_cluster_scores(t8_instances)["scores"]                        # centred within beta over P (rule g)
    if len(scores) < 2:
        raise SeedCountStop("sigma_S: fewer than 2 seeds (ddof=1 undefined)")
    sigma_S = float(np.std(scores, ddof=1))                                   # rule f
    return PilotQuantities(pi, M2, rr, rl, sigma_S)


# --------------------------------------------------------------------------- #
# 3. The frozen map (fixed, pilot) -> n_final
# --------------------------------------------------------------------------- #
def _z(p: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(p))


def _check_rho_lvl(r: float, b: str) -> float:
    if not math.isfinite(r):
        raise SeedCountStop(f"rho_lvl_{b} not finite")
    lower = -1.0 / (K - 1)
    if r < lower - RHO_TOL or r > 1.0 + RHO_TOL:                               # rule c: materially outside
        raise SeedCountStop(f"rho_lvl_{b} = {r} outside [-1/(k-1), 1) beyond tolerance")
    if 1.0 - r <= 0.0:                                                         # rule c + general invariant
        raise SeedCountStop(f"rho_lvl_{b} = {r}: 1 - rho_lvl <= 0 (invalid)")
    return r


def sigma2(fixed: FixedQuantities, pilot: PilotQuantities, b: str) -> float:
    core = pilot.pi[b] * pilot.M2[b] - fixed.delta[b] ** 2
    if core <= 0:
        raise SeedCountStop(f"pi*M2 - delta^2 = {core} <= 0 at {b} (rule a)")
    if not math.isfinite(pilot.rho_red[b]):
        raise SeedCountStop(f"rho_red_{b} not finite")
    return _finite_positive(f"sigma2_{b}", core * (1.0 + pilot.rho_red[b]) / 2.0)


def _cap(n: int, test: str) -> int:
    if n > N_MAX:
        raise SeedCountStop(f"{test}: required n = {n} > {N_MAX} (rule d) -> REDESIGN")
    return n


def n_single_level(fixed, pilot, test) -> int:
    b = BUDGET_OF[test]
    z2 = _z(1 - ALPHA / 2) + _z(POWER)
    return _cap(max(1, math.ceil((z2 * math.sqrt(sigma2(fixed, pilot, b)) / fixed.delta[b]) ** 2)), test)


def n_contrast(fixed, pilot, test) -> int:
    b = BUDGET_OF[test]
    c = np.asarray(EP.POLY_LINEAR if test == "T6" else EP.POLY_QUADRATIC, float)
    rho = _check_rho_lvl(pilot.rho_lvl[b], b)
    L = fixed.delta[b] * float(np.sum(c ** 2)) / float(c.max() - c.min())
    sd = math.sqrt(sigma2(fixed, pilot, b) * float(np.sum(c ** 2)) * (1.0 - rho))
    sd = _finite_positive(f"{test} score sd", sd)
    z2 = _z(1 - ALPHA / 2) + _z(POWER)
    return _cap(max(1, math.ceil((z2 * sd / L) ** 2)), test)


def friedman_power(n: int, f: float, rho: float) -> float:
    from scipy.stats import chi2, ncx2
    lam = ARE * n * K * f ** 2 / (1.0 - rho)
    return float(ncx2.sf(chi2.ppf(1 - ALPHA, K - 1), K - 1, lam))


def n_friedman(fixed, pilot, test) -> Dict[str, float]:
    b = BUDGET_OF[test]
    rho = _check_rho_lvl(pilot.rho_lvl[b], b)
    f = (fixed.delta[b] / math.sqrt(sigma2(fixed, pilot, b))) * math.sqrt(1.0 / (2 * K))
    for n in range(1, N_MAX + 1):                                             # rule d
        if friedman_power(n, f, rho) >= POWER:
            return {"n": n, "f": f, "W_min": ARE * K * f ** 2 / ((K - 1) * (1.0 - rho))}
    raise SeedCountStop(f"{test}: power < {POWER} at n = {N_MAX} (rule d) -> REDESIGN")


def n_t8(fixed, pilot) -> int:
    dD = _finite_positive("Delta_D", fixed.delta_D)
    EQ = _finite_positive("E_Q", fixed.E_Q)
    sS = _finite_positive("sigma_S", pilot.sigma_S)
    b_min = T8_PROB_SESOI / dD          # 0.03 probability points ACROSS Delta_D, not a slope of 0.03
    z1 = _z(1 - ALPHA) + _z(POWER)
    return _cap(max(1, math.ceil((z1 * sS / (b_min * EQ)) ** 2)), "T8")


def n_final(fixed: FixedQuantities, pilot: PilotQuantities) -> Dict[str, Any]:
    """The frozen map. Deterministic given its two arguments; no other input exists."""
    for b in EP.BUDGET_LEVELS:
        _finite_positive(f"delta_{b}", fixed.delta[b])
    per = {"T2": n_single_level(fixed, pilot, "T2"), "T3": n_single_level(fixed, pilot, "T3"),
           "T4": n_single_level(fixed, pilot, "T4"),
           "T6": n_contrast(fixed, pilot, "T6"), "T7": n_contrast(fixed, pilot, "T7"),
           "T10": n_contrast(fixed, pilot, "T10"),
           "T1": n_friedman(fixed, pilot, "T1")["n"], "T9": n_friedman(fixed, pilot, "T9")["n"],
           "T8": n_t8(fixed, pilot)}
    return {"n_by_test": per, "n_required": max(per.values()), "n_final": max(N_MIN, max(per.values()))}
