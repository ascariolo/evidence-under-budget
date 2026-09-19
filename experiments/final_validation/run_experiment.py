"""Run the six locked arms at the three locked budgets (protocol sections 3-6).

Produces raw per-question / per-arm / per-budget rows. No inferential analysis
happens here: that is ``analyze.py``. Gold supporting facts are used *only*
after a selection exists, to score supporting-fact coverage.

Usage::

    python -m experiments.final_validation.run_experiment --dry-run
    python -m experiments.final_validation.run_experiment --full
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from embedder import Embedder, cosine_similarity_matrix  # noqa: E402
from optimizer import ContextKnapsack, SolveScope, SolveStatus  # noqa: E402

from . import protocol as P  # noqa: E402
from .data import Question, load_sample  # noqa: E402

RESULTS_DIR = P.REPO_ROOT / "results" / "final_validation"


# --------------------------------------------------------------------------- #
# Per-question quantities (protocol section 3)
# --------------------------------------------------------------------------- #
def encode_question(question: Question, embedder: Embedder) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(relevance, similarity)``: raw cosines, no clipping, no rescaling."""
    vectors = embedder.encode([question.question, *question.sentences])
    query_vec, sentence_vecs = vectors[:1], vectors[1:]
    relevance = cosine_similarity_matrix(
        sentence_vecs, query_vec, assume_normalized=True).ravel().astype(np.float64)
    similarity = cosine_similarity_matrix(
        sentence_vecs, assume_normalized=True).astype(np.float64)
    return relevance, similarity


def rank_disagreement(relevance: np.ndarray, tokens: np.ndarray) -> float:
    """D = 1 - Kendall tau_b(r, r/w), computed before any selection (section 8)."""
    from scipy.stats import kendalltau

    if relevance.size < 2:
        return float("nan")
    per_token = relevance / tokens.astype(np.float64)
    tau = kendalltau(relevance, per_token, variant="b").statistic
    return float("nan") if np.isnan(tau) else 1.0 - float(tau)


def budget_for(question: Question, rho: float) -> int:
    """W_max = floor(rho * W_pool), with W_pool from the candidate pool alone."""
    return int(np.floor(rho * question.w_pool))


def ilp_in_scope(position: int, rho: float) -> bool:
    """The locked ILP scope: rho = 0.25, first 100 sampled questions only."""
    return rho == P.ILP_RHO and position < P.ILP_N_QUESTIONS


# --------------------------------------------------------------------------- #
# Arms (protocol section 5)
# --------------------------------------------------------------------------- #
def _best_singleton(knapsack: ContextKnapsack) -> Tuple[List[int], float]:
    """Highest-scoring single affordable sentence; ties go to the lowest index."""
    feasible = np.flatnonzero(knapsack.feasible_mask)
    if feasible.size == 0:
        return [], 0.0
    scores = knapsack.relevance[feasible]          # a singleton has no redundancy term
    best = int(feasible[int(np.argmax(scores))])
    return [best], float(knapsack.score([best])[0])


def run_arm(arm: str, knapsack: ContextKnapsack) -> Dict[str, object]:
    """Execute one arm and return a raw record. Gold labels are not in scope."""
    start = time.perf_counter()
    if arm == P.ARM_TOP_K:
        result = knapsack.solve_top_k()
    elif arm == P.ARM_GO:
        result = knapsack.solve_greedy_objective()
    elif arm == P.ARM_TA:
        result = knapsack.solve_greedy()
    elif arm == P.ARM_MMR:
        result = knapsack.solve_mmr()
    elif arm == P.ARM_TA_SINGLETON:
        token_aware = knapsack.solve_greedy()
        singleton_indices, singleton_score = _best_singleton(knapsack)
        if singleton_score > token_aware.score:          # strict: ties keep token-aware
            indices = singleton_indices
            chosen = "best_singleton"
        else:
            indices = list(token_aware.indices)
            chosen = "token_aware"
        latency = (time.perf_counter() - start) * 1000.0
        score, relevance_sum, redundancy = knapsack.score(indices)
        return {
            "arm": arm, "indices": [int(i) for i in indices],
            "tokens_used": int(knapsack.tokens[indices].sum()) if indices else 0,
            "score": score, "relevance_sum": relevance_sum, "redundancy": redundancy,
            "latency_ms": latency, "status": SolveStatus.HEURISTIC.value,
            "scope": SolveScope.FULL_CORPUS.value,
            "meta": {"guard_choice": chosen},
        }
    elif arm == P.ARM_ILP:
        result = knapsack.solve_ilp(time_limit=P.ILP_TIME_LIMIT_S, msg=False,
                                    scope=SolveScope.FULL_CORPUS)
    else:  # pragma: no cover - guarded by ARMS
        raise P.ProtocolViolation(f"unknown arm {arm!r}")

    return {
        "arm": arm,
        "indices": [int(i) for i in result.indices],
        "tokens_used": int(result.tokens_used),
        "score": float(result.score),
        "relevance_sum": float(result.relevance_sum),
        "redundancy": float(result.redundancy),
        "latency_ms": float(result.latency_ms),
        "status": result.status.value,
        "scope": result.scope.value,
        "meta": {k: v for k, v in result.meta.items()
                 if isinstance(v, (int, float, str, bool, type(None)))},
    }


def coverage(indices: Sequence[int], question: Question) -> Tuple[float, int]:
    """C(S) = |S ∩ G(q)| / |G(q)|, scored after selection (protocol section 6)."""
    gold = set(question.gold_indices)
    hit = len(gold.intersection(int(i) for i in indices))
    denominator = question.n_gold_annotated
    return (hit / denominator if denominator else float("nan")), hit


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(n_questions: int, out_path: Path, label: str) -> Dict[str, object]:
    verified_sha = P.assert_locked()
    if out_path.exists():
        raise P.ProtocolViolation(f"REFUSING to overwrite existing artifact: {out_path}")

    started = time.time()
    sample, ledger = load_sample(n=n_questions)
    embedder = Embedder(model_name=P.EMBEDDER_MODEL, allow_fallback=False,
                        encoding_name=P.TOKENIZER_ENCODING)
    if embedder.backend_name != P.EMBEDDER_MODEL:
        raise P.ProtocolViolation(
            f"embedding backend is {embedder.backend_name!r}, protocol requires {P.EMBEDDER_MODEL!r}")

    question_records: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []

    for position, question in enumerate(sample):
        relevance, similarity = encode_question(question, embedder)
        question_records.append({
            "position": position,
            "qid": question.qid,
            "n_pool": question.n,
            "w_pool": question.w_pool,
            "n_gold_annotated": question.n_gold_annotated,
            "n_gold_unreachable": question.n_gold_unreachable,
            "D": rank_disagreement(relevance, question.tokens),
            "level": question.meta.get("level"),
            "type": question.meta.get("type"),
        })

        for rho in P.RHO_LEVELS:
            w_max = budget_for(question, rho)
            knapsack = ContextKnapsack(
                relevance=relevance, similarity=similarity, tokens=question.tokens,
                budget=w_max, objective_lambda=P.LAMBDA_OBJ, mmr_lambda=P.LAMBDA_MMR,
                stop_on_nonpositive_gain=True)

            for arm in P.ARMS:
                if arm == P.ARM_ILP and not ilp_in_scope(position, rho):
                    continue                      # locked ILP scope (protocol section 5)
                record = run_arm(arm, knapsack)
                cov, hit = coverage(record["indices"], question)
                if record["tokens_used"] > w_max:
                    raise P.ProtocolViolation(
                        f"budget violated: qid={question.qid} rho={rho} arm={arm} "
                        f"tokens={record['tokens_used']} > W_max={w_max}")
                rows.append({
                    "position": position, "qid": question.qid, "rho": rho,
                    "w_pool": question.w_pool, "w_max": w_max,
                    "n_pool": question.n, "n_gold_annotated": question.n_gold_annotated,
                    "coverage": cov, "n_gold_retrieved": hit,
                    "n_selected": len(record["indices"]),
                    **{k: record[k] for k in ("arm", "tokens_used", "score",
                                              "relevance_sum", "redundancy",
                                              "latency_ms", "status", "scope", "meta")},
                    "indices": record["indices"],
                })

    payload = {
        "label": label,
        "protocol_sha256_verified_at_runtime": verified_sha,
        "protocol": P.frozen_constants(),
        "data_ledger": ledger,
        "environment": _environment(),
        "wall_clock_s": time.time() - started,
        "n_questions": len(sample),
        "questions": question_records,
        "rows": rows,
        "implementation_notes": [
            "ContextKnapsack clips negative pairwise similarity to zero inside the "
            "frozen core; relevance is used raw, as the protocol requires.",
            "CBC is invoked through PULP_CBC_CMD without a threads argument, which "
            "is CBC's single-threaded default; OMP_NUM_THREADS is pinned to 1.",
            "W_pool is computed on the post-exclusion pool; zero-token sentences "
            "contribute nothing to the sum, so the value is identical either way.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1) + "\n")
    return payload


def _environment() -> Dict[str, object]:
    packages = {}
    for name in ("numpy", "scipy", "sentence_transformers", "tiktoken", "pulp",
                 "pyarrow", "sklearn", "statsmodels"):
        try:
            packages[name] = getattr(importlib.import_module(name), "__version__", "?")
        except Exception:
            packages[name] = "MISSING"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         cwd=P.REPO_ROOT).strip()
    except Exception:
        commit = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": commit,
        "code_sha256": {
            name: P.sha256_of(Path(__file__).with_name(name))
            for name in ("protocol.py", "data.py", "run_experiment.py")
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true",
                       help=f"software/data sanity check on {P.DRY_RUN_N} questions")
    group.add_argument("--full", action="store_true",
                       help=f"the locked experiment on {P.N_QUESTIONS} questions")
    args = parser.parse_args(argv)

    if args.dry_run:
        out = RESULTS_DIR / "dry_run_raw.json"
        payload = run(P.DRY_RUN_N, out, label="dry_run")
    else:
        out = RESULTS_DIR / "experiment_raw.json"
        payload = run(P.N_QUESTIONS, out, label="full")

    print(f"wrote {out}")
    print(f"questions={payload['n_questions']} rows={len(payload['rows'])} "
          f"wall_clock_s={payload['wall_clock_s']:.1f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
