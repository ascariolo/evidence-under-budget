r"""Controlled synthetic instance generator.

Specification: ``docs/controlled_benchmark_design.md`` section 5.
Frozen constants: ``experiments/controlled/protocol.py``.

Construction
------------
An instance is fully specified by ``(relevance, similarity, tokens)`` - the only
three arrays any solver consumes - so no natural-language text is generated and
no encoder is involved. The controlled latent variables ARE the ground truth.

1. **Copula.** Draw ``(zeta_i, eta_i)`` from a bivariate normal with correlation
   ``rho_c``. ``zeta`` drives relevance, ``eta`` drives length. Because a
   Gaussian copula changes only the dependence, the marginals of relevance and
   of token count are identical across conditions A-C.
2. **Relevance.** ``r_i = r_min + (r_max - r_min) * BetaPPF(Phi(zeta_i))``.
3. **Tokens.** ``w_i = clip(round(exp(mu_w + sigma_w * eta_i)))`` - integers,
   exact by construction.
4. **Embeddings.** ``e_i = r_i * q + sqrt(1 - r_i^2) * u_i`` with ``u_i``
   orthogonal to ``q`` and unit norm, so ``cos(q, e_i) == r_i`` exactly.
5. **Redundancy.** ``u_i = normalize(sqrt(gamma) * t_k(i) + sqrt(1-gamma) * eps_i)``
   inside the subspace orthogonal to ``q``. ``gamma`` therefore moves
   inter-document similarity without touching relevance.

Independent RNG streams
-----------------------
The copula stream, the topic stream and the noise stream are spawned from
separate ``SeedSequence`` children. This is what makes the causal-isolation
tests pass: changing ``gamma`` cannot perturb the relevance or token draws,
because it does not consume from their stream.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import beta as beta_dist, norm, pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from optimizer import ContextKnapsack, SolveScope, SolveStatus
from experiments.controlled import protocol as P


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GeneratorConfig:
    """All generator parameters. Defaults come from the frozen protocol."""

    n_docs: int = P.N_DOCS
    dim: int = P.EMBED_DIM
    n_topics: int = P.N_TOPICS
    relevance_beta_a: float = P.RELEVANCE_BETA_A
    relevance_beta_b: float = P.RELEVANCE_BETA_B
    relevance_min: float = P.RELEVANCE_MIN
    relevance_max: float = P.RELEVANCE_MAX
    log_token_mu: float = P.LOG_TOKEN_MU
    log_token_sigma: float = P.LOG_TOKEN_SIGMA
    token_min: int = P.TOKEN_MIN
    token_max: int = P.TOKEN_MAX
    adversarial_top_k: int = P.ADVERSARIAL_TOP_K
    adversarial_multiplier: float = P.ADVERSARIAL_MULTIPLIER
    adversarial_cap_fraction: float = P.ADVERSARIAL_CAP_FRACTION


@dataclass(frozen=True)
class InstanceSpec:
    """Identifies one instance completely: seed + condition + configuration."""

    seed: int
    rl_condition: str                 # "A" | "B" | "C" | "D"
    redundancy_level: str             # "low" | "high"
    config: GeneratorConfig = field(default_factory=GeneratorConfig)

    def __post_init__(self) -> None:
        if self.rl_condition not in P.RL_CONDITIONS:
            raise ValueError(f"unknown rl_condition {self.rl_condition!r}")
        if self.redundancy_level not in P.REDUNDANCY_LEVELS:
            raise ValueError(f"unknown redundancy_level {self.redundancy_level!r}")

    @property
    def rho_c(self) -> float:
        return float(P.RL_CONDITIONS[self.rl_condition]["rho_c"])

    @property
    def gamma(self) -> float:
        return float(P.REDUNDANCY_LEVELS[self.redundancy_level])

    @property
    def is_adversarial(self) -> bool:
        return bool(P.RL_CONDITIONS[self.rl_condition]["adversarial"])

    @property
    def target_beta(self) -> Optional[float]:
        return P.RL_CONDITIONS[self.rl_condition]["target_beta"]


class DegenerateInstanceError(RuntimeError):
    """Raised when a generated instance fails a non-degeneracy guard."""


# --------------------------------------------------------------------------- #
# Instance
# --------------------------------------------------------------------------- #
@dataclass
class ControlledInstance:
    spec: InstanceSpec
    relevance: np.ndarray             # (N,)  == cos(q, e_i) exactly
    tokens: np.ndarray                # (N,)  int64, exact by construction
    similarity: np.ndarray            # (N,N) valid Gram matrix
    embeddings: np.ndarray            # (N,d)
    query_vector: np.ndarray          # (d,)
    topics: np.ndarray                # (N,)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def condition_metadata(self) -> Dict[str, Any]:
        """Everything needed to reconstruct the experimental condition."""
        return {
            "protocol_version": P.CONTROLLED_PROTOCOL_VERSION,
            "seed": self.spec.seed,
            "rl_condition": self.spec.rl_condition,
            "rl_condition_name": P.RL_CONDITIONS[self.spec.rl_condition]["name"],
            "rho_c": self.spec.rho_c,
            "target_beta": self.spec.target_beta,
            "redundancy_level": self.spec.redundancy_level,
            "gamma": self.spec.gamma,
            "is_adversarial": self.spec.is_adversarial,
            "n_docs": self.spec.config.n_docs,
            "embed_dim": self.spec.config.dim,
            "n_topics": self.spec.config.n_topics,
            "objective_lambda": P.OBJECTIVE_LAMBDA,
            "mmr_lambda": P.MMR_LAMBDA,
            "config": asdict(self.spec.config),
        }


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def _streams(seed: int) -> Dict[str, np.random.Generator]:
    """Independent RNG streams so one factor cannot perturb another's draws."""
    copula_ss, topic_ss, noise_ss = np.random.SeedSequence(seed).spawn(3)
    return {"copula": np.random.default_rng(copula_ss),
            "topic": np.random.default_rng(topic_ss),
            "noise": np.random.default_rng(noise_ss)}


def _draw_latents(rng: np.random.Generator, n: int, rho_c: float) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian copula: standard-normal pair with correlation ``rho_c``."""
    z = rng.standard_normal((n, 2))
    # Cholesky of [[1, rho], [rho, 1]] - exact, and leaves marginals standard
    # normal for every rho_c, which is what preserves the marginals.
    zeta = z[:, 0]
    eta = rho_c * z[:, 0] + np.sqrt(max(0.0, 1.0 - rho_c ** 2)) * z[:, 1]
    return zeta, eta


def _relevance_from_latent(zeta: np.ndarray, cfg: GeneratorConfig) -> np.ndarray:
    u = norm.cdf(zeta)
    q = beta_dist.ppf(u, cfg.relevance_beta_a, cfg.relevance_beta_b)
    return cfg.relevance_min + (cfg.relevance_max - cfg.relevance_min) * q


def _tokens_from_latent(eta: np.ndarray, cfg: GeneratorConfig) -> np.ndarray:
    w = np.exp(cfg.log_token_mu + cfg.log_token_sigma * eta)
    return np.clip(np.round(w), cfg.token_min, cfg.token_max).astype(np.int64)


def _build_embeddings(relevance: np.ndarray, gamma: float, cfg: GeneratorConfig,
                      topic_rng: np.random.Generator,
                      noise_rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``e_i = r_i q + sqrt(1-r_i^2) u_i`` with ``u_i`` orthogonal to ``q``.

    ``q`` is the first basis vector, so the orthogonal complement is simply
    coordinates ``1..d-1``. Any other choice of ``q`` is a rotation of this one
    and changes nothing measurable.
    """
    n, d, k = cfg.n_docs, cfg.dim, cfg.n_topics
    sub = d - 1

    # Orthonormal topic directions inside the subspace orthogonal to q.
    topic_dirs = np.linalg.qr(topic_rng.standard_normal((sub, k)))[0][:, :k].T

    # Balanced topic membership, assigned INDEPENDENTLY of relevance so that
    # topic (and therefore redundancy) is not confounded with relevance.
    topics = np.tile(np.arange(k), int(np.ceil(n / k)))[:n].copy()
    topic_rng.shuffle(topics)

    eps = noise_rng.standard_normal((n, sub))
    eps /= np.linalg.norm(eps, axis=1, keepdims=True)

    u = np.sqrt(gamma) * topic_dirs[topics] + np.sqrt(1.0 - gamma) * eps
    u /= np.linalg.norm(u, axis=1, keepdims=True)

    embeddings = np.zeros((n, d), dtype=np.float64)
    embeddings[:, 0] = relevance
    embeddings[:, 1:] = np.sqrt(1.0 - relevance ** 2)[:, None] * u

    query = np.zeros(d, dtype=np.float64)
    query[0] = 1.0
    return embeddings, query, topics


def _elasticity(relevance: np.ndarray, tokens: np.ndarray) -> float:
    """Realized log-log slope ``beta_hat = d log r / d log w`` (OLS)."""
    return float(np.polyfit(np.log(tokens.astype(float)), np.log(relevance), 1)[0])


def _diagnostics(relevance, tokens, similarity, spec) -> Dict[str, Any]:
    n = len(relevance)
    iu = np.triu_indices(n, 1)
    off = similarity[iu]
    pear_r, pear_p = pearsonr(relevance, tokens.astype(float))
    spear_r, spear_p = spearmanr(relevance, tokens.astype(float))
    tok = tokens.astype(float)
    return {
        "beta_target": spec.target_beta,
        "beta_hat": _elasticity(relevance, tokens),
        "pearson_r_relevance_tokens": float(pear_r),
        "pearson_p": float(pear_p),
        "spearman_rho_relevance_tokens": float(spear_r),
        "spearman_p": float(spear_p),
        "relevance_mean": float(relevance.mean()),
        "relevance_std": float(relevance.std(ddof=1)),
        "relevance_min": float(relevance.min()),
        "relevance_max": float(relevance.max()),
        "tokens_mean": float(tok.mean()),
        "tokens_std": float(tok.std(ddof=1)),
        "tokens_min": int(tok.min()),
        "tokens_max": int(tok.max()),
        "tokens_cv": float(tok.std(ddof=1) / tok.mean()),
        "tokens_total": int(tok.sum()),
        "similarity_mean": float(off.mean()),
        "similarity_std": float(off.std(ddof=1)),
        "similarity_min": float(off.min()),
        "similarity_max": float(off.max()),
        "n_docs": int(n),
    }


def generate_instance(spec: InstanceSpec, strict: bool = True) -> ControlledInstance:
    """Generate one controlled instance. Deterministic in ``spec``.

    ``strict`` raises :class:`DegenerateInstanceError` when a non-degeneracy
    guard fails, rather than returning an instance that cannot test what the
    experiment is for.
    """
    cfg = spec.config
    rng = _streams(spec.seed)

    zeta, eta = _draw_latents(rng["copula"], cfg.n_docs, spec.rho_c)
    relevance = _relevance_from_latent(zeta, cfg)
    tokens = _tokens_from_latent(eta, cfg)

    embeddings, query, topics = _build_embeddings(
        relevance, spec.gamma, cfg, rng["topic"], rng["noise"])
    similarity = embeddings @ embeddings.T

    instance = ControlledInstance(
        spec=spec, relevance=relevance, tokens=tokens, similarity=similarity,
        embeddings=embeddings, query_vector=query, topics=topics)
    instance.diagnostics = _diagnostics(relevance, tokens, similarity, spec)
    instance.diagnostics["adversarial_applied"] = False

    report = validate_instance(instance)
    instance.diagnostics["validation"] = report
    if strict and not report["ok"]:
        raise DegenerateInstanceError(
            f"instance {spec.seed}/{spec.rl_condition}/{spec.redundancy_level} "
            f"failed: {report['failures']}")
    return instance


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_instance(instance: ControlledInstance) -> Dict[str, Any]:
    """Every guard from design section 13, as a machine-readable report."""
    cfg = instance.spec.config
    rel, tok, sim = instance.relevance, instance.tokens, instance.similarity
    emb, q = instance.embeddings, instance.query_vector
    n = cfg.n_docs
    checks: Dict[str, Any] = {}
    failures: List[str] = []

    # -- exact relevance ---------------------------------------------------- #
    norms = np.linalg.norm(emb, axis=1)
    cos_q = (emb @ q) / norms
    err = float(np.abs(cos_q - rel).max())
    checks["relevance_exact_max_abs_error"] = err
    if err > P.RELEVANCE_EXACTNESS_TOLERANCE:
        failures.append(f"cos(q, e_i) != r_i (max err {err:.2e})")

    # -- valid Gram matrix -------------------------------------------------- #
    checks["gram_symmetry_max_abs_error"] = float(np.abs(sim - sim.T).max())
    checks["gram_diagonal_max_abs_error"] = float(np.abs(np.diag(sim) - 1.0).max())
    eig_min = float(np.linalg.eigvalsh(0.5 * (sim + sim.T)).min())
    checks["gram_min_eigenvalue"] = eig_min
    checks["similarity_in_range"] = bool(sim.min() >= -1 - 1e-9 and sim.max() <= 1 + 1e-9)
    if checks["gram_symmetry_max_abs_error"] > 1e-10:
        failures.append("similarity is not symmetric")
    if checks["gram_diagonal_max_abs_error"] > 1e-8:
        failures.append("similarity diagonal is not 1")
    if eig_min < P.GRAM_EIGENVALUE_TOLERANCE:
        failures.append(f"similarity is not PSD (min eig {eig_min:.2e})")
    if not checks["similarity_in_range"]:
        failures.append("similarity outside [-1, 1]")

    # -- non-degeneracy ----------------------------------------------------- #
    tokens_cv = float(tok.std(ddof=1) / tok.mean())
    checks["tokens_cv"] = tokens_cv
    if tokens_cv < P.MIN_TOKEN_CV:
        failures.append(
            f"token CV {tokens_cv:.3f} < {P.MIN_TOKEN_CV}: Delta_i/w_i would be "
            f"proportional to Delta_i and the two greedy methods would be "
            f"identical by construction")
    rel_std = float(rel.std(ddof=1))
    checks["relevance_std"] = rel_std
    if rel_std < P.MIN_RELEVANCE_STD:
        failures.append(f"relevance std {rel_std:.4f} < {P.MIN_RELEVANCE_STD}")
    iu = np.triu_indices(n, 1)
    sim_std = float(sim[iu].std(ddof=1))
    checks["similarity_std"] = sim_std
    if sim_std < P.MIN_SIMILARITY_STD:
        failures.append(f"similarity std {sim_std:.4f} < {P.MIN_SIMILARITY_STD}")

    # -- token integrality -------------------------------------------------- #
    checks["tokens_are_positive_integers"] = bool(
        np.issubdtype(tok.dtype, np.integer) and (tok > 0).all())
    if not checks["tokens_are_positive_integers"]:
        failures.append("token counts are not positive integers")

    return {"ok": not failures, "failures": failures, "checks": checks}


# --------------------------------------------------------------------------- #
# Budget construction (design section 7)
# --------------------------------------------------------------------------- #
def _knapsack(instance: ControlledInstance, budget: int) -> ContextKnapsack:
    return ContextKnapsack(instance.relevance, instance.similarity,
                           instance.tokens, budget=budget,
                           objective_lambda=P.OBJECTIVE_LAMBDA,
                           mmr_lambda=P.MMR_LAMBDA)


def compute_saturation(instance: ControlledInstance,
                       time_limit: float = P.ILP_TIME_LIMIT_S) -> Dict[str, Any]:
    """Token mass of the UNCONSTRAINED optimum, via a proven-optimal ILP.

    ``W_sat`` is the denominator of the instance-relative budget ``rho``. It is
    method-independent by construction. If the unconstrained solve does not
    prove optimality the result says so and the caller must not use it.
    """
    total = int(instance.tokens.sum())
    result = _knapsack(instance, total).solve_ilp(time_limit=time_limit,
                                                  scope=SolveScope.FULL_CORPUS)
    return {
        "w_sat": int(result.tokens_used),
        "w_sat_proven": bool(result.is_proven_global_optimum),
        "w_sat_status": result.status.value,
        "w_sat_n_selected": int(result.n_selected),
        "w_sat_score": float(result.score),
        "corpus_token_mass": total,
        "saturation_indices": list(result.indices),
    }


def apply_adversarial(instance: ControlledInstance, w_sat_pre: int) -> ControlledInstance:
    """Condition D: inflate the token cost of the most relevant documents.

    DEV-1. The design caps the inflated cost at ``0.45 * W_sat``, but ``W_sat``
    is a function of the tokens, so the cap cannot be applied while the tokens
    are being assigned. Resolved as a two-pass procedure: ``w_sat_pre`` is the
    saturation mass of the PRE-intervention instance, the cap is taken against
    it, and the caller recomputes the final ``W_sat`` afterwards. Both values
    are recorded. See protocol.DESIGN_DEVIATIONS.
    """
    cfg = instance.spec.config
    cap = int(round(cfg.adversarial_cap_fraction * w_sat_pre))
    top = np.argsort(-instance.relevance, kind="stable")[:cfg.adversarial_top_k]

    tokens = instance.tokens.copy()
    inflated = np.minimum(
        np.round(tokens[top] * cfg.adversarial_multiplier).astype(np.int64),
        max(cap, cfg.token_min))
    # Never make a document cheaper than it was.
    tokens[top] = np.maximum(tokens[top], inflated)

    instance.tokens = tokens
    instance.diagnostics.update(_diagnostics(instance.relevance, tokens,
                                             instance.similarity, instance.spec))
    instance.diagnostics.update({
        "adversarial_applied": True,
        "adversarial_indices": [int(i) for i in top],
        "adversarial_cap": cap,
        "adversarial_w_sat_pre": int(w_sat_pre),
        "adversarial_tokens_after": [int(t) for t in tokens[top]],
    })
    instance.diagnostics["validation"] = validate_instance(instance)
    return instance


def build_budgets(w_sat: int, levels: Optional[Dict[str, float]] = None) -> Dict[str, int]:
    """``W_max = round(rho * W_sat)`` for each named tightness level."""
    levels = levels or P.BUDGET_LEVELS
    return {name: max(1, int(round(rho * w_sat))) for name, rho in levels.items()}


def prepare_instance(spec: InstanceSpec,
                     time_limit: float = P.ILP_TIME_LIMIT_S) -> Dict[str, Any]:
    """Generate, apply condition D if required, and derive the budget schedule.

    Returns the instance plus everything needed to reconstruct its condition and
    its budgets, with no aggregate statistic computed.
    """
    instance = generate_instance(spec)
    saturation = compute_saturation(instance, time_limit)

    if spec.is_adversarial:
        w_sat_pre = saturation["w_sat"]
        instance = apply_adversarial(instance, w_sat_pre)
        saturation_post = compute_saturation(instance, time_limit)
        saturation = {**saturation_post,
                      "w_sat_pre_intervention": w_sat_pre,
                      "w_sat_pre_proven": saturation["w_sat_proven"]}

    budgets = build_budgets(saturation["w_sat"])
    return {
        "instance": instance,
        "saturation": saturation,
        "budgets": budgets,
        "realized_rho": {name: budgets[name] / saturation["w_sat"]
                         for name in budgets},
        "metadata": instance.condition_metadata(),
        "diagnostics": instance.diagnostics,
    }
