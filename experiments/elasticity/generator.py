r"""Step 6 elasticity generator.

Implements docs/elasticity_experiment_design.md rev. 2, section A4:

    z0, z1 ~ N(0, 1) i.i.d.
    eta  = z1                                   tokens' OWN latent (IMPL-1)
    zeta = rho_c * z1 + sqrt(1 - rho_c^2) * z0  so corr(zeta, eta) = rho_c
    r    = clip(exp(MU_R + SIGMA_R * zeta), 1e-3, 0.95)
    w    = clip(round(exp(MU_W + SIGMA_W * eta)), 8, 400)

Jointly Gaussian logs make the log-log regression exactly linear, so
beta = rho_c * SIGMA_R / SIGMA_W = 2.5 * rho_c.

Everything else is the LOCKED Step 4 construction, reused read-only from
experiments/controlled/generator.py (RNG streams, embeddings, topics, redundancy,
budget rule) so that "unchanged from Step 4" is true by construction, not by
re-implementation.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from experiments.controlled.generator import (GeneratorConfig as _Step4Config,
                                              _build_embeddings, _streams,
                                              build_budgets)
from experiments.elasticity import protocol as EP
from optimizer import ContextKnapsack, SolveScope

_STEP4_CFG = _Step4Config(n_docs=EP.N_DOCS, dim=EP.EMBED_DIM, n_topics=EP.N_TOPICS)


@dataclass(frozen=True)
class ElasticitySpec:
    seed: int
    beta_target: float
    redundancy_level: str

    def __post_init__(self) -> None:
        if self.beta_target not in EP.RHO_C:
            raise ValueError(f"beta_target {self.beta_target} not a locked level")
        if self.redundancy_level not in EP.REDUNDANCY_LEVELS:
            raise ValueError(f"unknown redundancy_level {self.redundancy_level!r}")

    @property
    def rho_c(self) -> float:
        return EP.RHO_C[self.beta_target]

    @property
    def gamma(self) -> float:
        return EP.REDUNDANCY_LEVELS[self.redundancy_level]

    @property
    def instance_id(self) -> str:
        return f"EL-s{self.seed}-b{self.beta_target:+.1f}-{self.redundancy_level}"


@dataclass
class ElasticityInstance:
    spec: ElasticitySpec
    relevance: np.ndarray
    tokens: np.ndarray
    similarity: np.ndarray
    embeddings: np.ndarray
    query_vector: np.ndarray
    topics: np.ndarray
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_cv_ok(self) -> bool:
        return bool(self.diagnostics["tokens_cv"] >= EP.MIN_TOKEN_CV)


def draw_latents(copula_rng: np.random.Generator, n: int, rho_c: float):
    """IMPL-1: eta (tokens) is the own latent, fixed across every rho_c."""
    z = copula_rng.standard_normal((n, 2))
    eta = z[:, 1]
    zeta = rho_c * z[:, 1] + np.sqrt(max(0.0, 1.0 - rho_c ** 2)) * z[:, 0]
    return zeta, eta


def relevance_from_latent(zeta: np.ndarray) -> np.ndarray:
    return np.clip(np.exp(EP.MU_R + EP.SIGMA_R * zeta), *EP.R_CLIP)


def tokens_from_latent(eta: np.ndarray) -> np.ndarray:
    w = np.exp(EP.MU_W + EP.SIGMA_W * eta)
    return np.clip(np.round(w), *EP.TOKEN_CLIP).astype(np.int64)


def ranking_disagreement(relevance: np.ndarray, tokens: np.ndarray) -> float:
    """Pre-registered D = 1 - Kendall tau_b(r, r/w)  (design C1)."""
    tau = kendalltau(relevance, relevance / tokens.astype(float), variant="b").statistic
    return float(1.0 - tau)


def elasticity_hat(relevance: np.ndarray, tokens: np.ndarray) -> float:
    """OLS slope of log r on log w."""
    return float(np.polyfit(np.log(tokens.astype(float)), np.log(relevance), 1)[0])


def _diagnostics(r: np.ndarray, w: np.ndarray, sim: np.ndarray) -> Dict[str, Any]:
    n = len(r)
    iu = np.triu_indices(n, 1)
    off = sim[iu]
    wf = w.astype(float)
    return {
        "beta_hat": elasticity_hat(r, w),
        "D": ranking_disagreement(r, w),
        "pearson_r_relevance_tokens": float(pearsonr(r, wf)[0]),
        "spearman_rho_relevance_tokens": float(spearmanr(r, wf)[0]),
        "n_usable": int((r > EP.USABLE_RELEVANCE_THRESHOLD).sum()),
        "n_below_threshold": int((r <= EP.USABLE_RELEVANCE_THRESHOLD).sum()),
        "n_relevance_clipped_low": int((r <= EP.R_CLIP[0]).sum()),
        "n_relevance_clipped_high": int((r >= EP.R_CLIP[1]).sum()),
        "relevance_mean": float(r.mean()), "relevance_median": float(np.median(r)),
        "relevance_std": float(r.std(ddof=1)),
        "relevance_min": float(r.min()), "relevance_max": float(r.max()),
        "tokens_mean": float(wf.mean()), "tokens_std": float(wf.std(ddof=1)),
        "tokens_min": int(w.min()), "tokens_max": int(w.max()),
        "tokens_cv": float(wf.std(ddof=1) / wf.mean()),
        "tokens_total": int(w.sum()),
        "similarity_mean": float(off.mean()), "similarity_std": float(off.std(ddof=1)),
        "similarity_min": float(off.min()), "similarity_max": float(off.max()),
    }


def validate_geometry(inst: ElasticityInstance) -> Dict[str, Any]:
    emb, q, sim, r = inst.embeddings, inst.query_vector, inst.similarity, inst.relevance
    cos = (emb @ q) / np.linalg.norm(emb, axis=1)
    out = {
        "relevance_exact_err": float(np.abs(cos - r).max()),
        "gram_symmetry_err": float(np.abs(sim - sim.T).max()),
        "gram_diag_err": float(np.abs(np.diag(sim) - 1.0).max()),
        "gram_min_eig": float(np.linalg.eigvalsh(0.5 * (sim + sim.T)).min()),
    }
    out["ok"] = bool(out["relevance_exact_err"] < 1e-10 and out["gram_symmetry_err"] < 1e-10
                     and out["gram_diag_err"] < 1e-8 and out["gram_min_eig"] >= -1e-8)
    return out


def generate(spec: ElasticitySpec) -> ElasticityInstance:
    """Deterministic in spec. Never rejects: the token-CV exclusion is recorded by the
    caller and applied at seed level (policy C4), so rejection cannot trigger a resample."""
    rng = _streams(spec.seed)
    zeta, eta = draw_latents(rng["copula"], EP.N_DOCS, spec.rho_c)
    r = relevance_from_latent(zeta)
    w = tokens_from_latent(eta)
    emb, q, topics = _build_embeddings(r, spec.gamma, _STEP4_CFG, rng["topic"], rng["noise"])
    sim = emb @ emb.T
    inst = ElasticityInstance(spec, r, w, sim, emb, q, topics)
    inst.diagnostics = _diagnostics(r, w, sim)
    inst.diagnostics["geometry"] = validate_geometry(inst)
    return inst


def knapsack(inst: ElasticityInstance, budget: int) -> ContextKnapsack:
    return ContextKnapsack(inst.relevance, inst.similarity, inst.tokens, budget=budget,
                           objective_lambda=EP.OBJECTIVE_LAMBDA, mmr_lambda=EP.MMR_LAMBDA)


def saturation(inst: ElasticityInstance) -> Dict[str, Any]:
    total = int(inst.tokens.sum())
    res = knapsack(inst, total).solve_ilp(time_limit=EP.ILP_TIME_LIMIT_S,
                                          scope=SolveScope.FULL_CORPUS)
    return {"w_sat": int(res.tokens_used), "w_sat_proven": bool(res.is_proven_global_optimum),
            "w_sat_status": res.status.value, "w_sat_n_selected": int(res.n_selected),
            "corpus_token_mass": total}


def budgets_for(w_sat: int) -> Dict[str, int]:
    return build_budgets(w_sat, EP.BUDGET_LEVELS)


METHODS = ("top_k", "mmr", "greedy_objective", "greedy_token_aware")


def evaluate_instance(inst: ElasticityInstance) -> List[Dict[str, Any]]:
    """Full pipeline for ONE instance: saturation, budgets, 5 methods x 3 budgets.
    Returns long-format rows in the analysis schema. Used on PILOT seeds only (IMPL-5)."""
    sat = saturation(inst)
    budgets = budgets_for(sat["w_sat"])
    s = inst.spec
    common = {"arm": EP.ARM, "instance_id": s.instance_id, "seed": s.seed,
              "beta_target": s.beta_target, "rho_c": s.rho_c,
              "redundancy_level": s.redundancy_level, "gamma": s.gamma,
              "beta_hat": inst.diagnostics["beta_hat"], "D": inst.diagnostics["D"],
              "n_usable": inst.diagnostics["n_usable"], "w_sat": sat["w_sat"],
              "w_sat_proven": sat["w_sat_proven"],
              "protocol_version": EP.ELASTICITY_PROTOCOL_VERSION,
              "objective_lambda": EP.OBJECTIVE_LAMBDA, "mmr_lambda": EP.MMR_LAMBDA}
    rows: List[Dict[str, Any]] = []
    import time
    for level, budget in budgets.items():
        k = knapsack(inst, budget)
        results = {}
        for m in METHODS:
            t = time.perf_counter()
            res = getattr(k, {"top_k": "solve_top_k", "mmr": "solve_mmr",
                              "greedy_objective": "solve_greedy_objective",
                              "greedy_token_aware": "solve_greedy"}[m])()
            results[m] = (res, (time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        ilp = k.solve_ilp(time_limit=EP.ILP_TIME_LIMIT_S, scope=SolveScope.FULL_CORPUS)
        results["ilp"] = (ilp, (time.perf_counter() - t) * 1000)
        for m, (res, ms) in results.items():
            rows.append({**common, "budget_level": level, "rho_budget": EP.BUDGET_LEVELS[level],
                         "w_max": int(budget), "method": m,
                         "objective_score": float(res.score),
                         "ilp_optimum": float(ilp.score),
                         "optimality_gap_absolute": float(ilp.score - res.score),
                         "selected_indices": [int(i) for i in res.indices],
                         "n_selected": int(res.n_selected), "tokens_used": int(res.tokens_used),
                         "solve_status": res.status.value,
                         "is_proven_global_optimum": bool(res.is_proven_global_optimum),
                         "ilp_proven_optimal": bool(ilp.is_proven_global_optimum),
                         "ilp_solver_objective": ilp.meta.get("solver_objective"),
                         "ilp_objective_consistent": ilp.meta.get("objective_consistent"),
                         "runtime_ms": float(ms)})
    return rows
