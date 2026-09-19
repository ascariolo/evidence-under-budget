r"""Statistical evaluation of the REPAIRED solvers on the ORIGINAL benchmark.

PRE-REGISTERED PROTOCOL (fixed before any result was inspected)
---------------------------------------------------------------
Corpora   : ``benchmark.generate_corpus`` UNCHANGED, ``n_docs = 25``
            (the original reported size).
Seeds     : 0..23 inclusive - 24 contiguous seeds. The range is fixed a priori;
            no seed is added, dropped or reordered after seeing results.
Budgets   : 96, 128, 192, 256, 384, 512, 768, 1024, 2048.
            Includes all four originally reported budgets (256/512/1024/2048)
            plus five intermediate points, so conclusions do not rest on the
            original four.
Methods   : top_k, mmr (true max-similarity), greedy_objective,
            greedy_token_aware (proposed), ilp (exact).
Parameters: objective_lambda = 0.1, mmr_lambda = 0.5. BOTH FROZEN at their
            documented values and NOT tuned on these results.
ILP       : full-corpus scope, time limit 300 s. Every solve records its status;
            non-proven solves are reported, never dropped.
Objective : unchanged - Score(S) = sum r_i - objective_lambda * sum_{i<j} s_ij.

Writes ``results/repaired_original/raw_results.json`` and ``raw_results.csv``.
Analysis lives in ``analyze_repaired_original.py``; this script computes no
aggregate statistic, so the run cannot be steered by its own output.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import (BENCHMARK_CONFIG_VERSION, UNAVAILABLE, _git_info,
                       _package_version, generate_corpus, unique_coverage)
from embedder import Embedder, build_corpus
from optimizer import (ContextKnapsack, SolveScope, SolveStatus,
                       DEFAULT_MMR_LAMBDA, DEFAULT_OBJECTIVE_LAMBDA)

SEEDS = list(range(24))
BUDGETS = [96, 128, 192, 256, 384, 512, 768, 1024, 2048]
ORIGINAL_BUDGETS = [256, 512, 1024, 2048]
N_DOCS = 25
OBJECTIVE_LAMBDA = DEFAULT_OBJECTIVE_LAMBDA      # 0.1, frozen
MMR_LAMBDA = DEFAULT_MMR_LAMBDA                  # 0.5, frozen
ILP_TIME_LIMIT = 300.0
REQUESTED_MODEL = "all-MiniLM-L6-v2"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "repaired_original")

METHOD_CALLS = {
    "top_k": lambda k: k.solve_top_k(),
    "mmr": lambda k: k.solve_mmr(),
    "greedy_objective": lambda k: k.solve_greedy_objective(),
    "greedy_token_aware": lambda k: k.solve_greedy(),
}


def corpus_descriptives(corpus) -> dict:
    """Descriptive statistics of the corpus. Descriptive only - no causal claim."""
    rel = np.asarray(corpus.relevance, dtype=float)
    tok = np.asarray(corpus.tokens, dtype=float)
    iu = np.triu_indices(len(corpus), 1)
    pear_r, pear_p = pearsonr(rel, tok)
    spear_r, spear_p = spearmanr(rel, tok)
    return {
        "pearson_r_relevance_tokens": float(pear_r),
        "pearson_p": float(pear_p),
        "spearman_rho_relevance_tokens": float(spear_r),
        "spearman_p": float(spear_p),
        "relevance_mean": float(rel.mean()),
        "relevance_min": float(rel.min()),
        "relevance_max": float(rel.max()),
        "tokens_mean": float(tok.mean()),
        "tokens_min": int(tok.min()),
        "tokens_max": int(tok.max()),
        "tokens_total": int(tok.sum()),
        "pairwise_sim_mean": float(corpus.similarity[iu].mean()),
        "n_unique_documents": len(set(corpus.documents)),
    }


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    embedder = Embedder(model_name=REQUESTED_MODEL)
    started = time.time()

    rows: list[dict] = []
    per_seed: dict[str, dict] = {}
    ilp_status_counts: dict[str, int] = {}

    for si, seed in enumerate(SEEDS):
        query, documents = generate_corpus(n_docs=N_DOCS, seed=seed)
        corpus = build_corpus(query, documents, embedder)
        if corpus.backend_name != REQUESTED_MODEL:
            raise RuntimeError(
                f"encoder fallback to {corpus.backend_name!r}: results would "
                f"not be comparable; aborting rather than silently continuing")
        per_seed[str(seed)] = corpus_descriptives(corpus)

        for budget in BUDGETS:
            knap = ContextKnapsack(corpus.relevance, corpus.similarity,
                                   corpus.tokens, budget=budget,
                                   objective_lambda=OBJECTIVE_LAMBDA,
                                   mmr_lambda=MMR_LAMBDA)
            results = {name: call(knap) for name, call in METHOD_CALLS.items()}
            results["ilp"] = knap.solve_ilp(time_limit=ILP_TIME_LIMIT,
                                            scope=SolveScope.FULL_CORPUS)
            ilp = results["ilp"]
            ilp_status_counts[ilp.status.value] = \
                ilp_status_counts.get(ilp.status.value, 0) + 1

            # An optimality gap is only defined against a PROVEN global optimum
            # with a usable denominator. Otherwise it is recorded as None and
            # the instance is excluded from gap statistics downstream.
            gap_valid = (ilp.is_proven_global_optimum and abs(ilp.score) > 1e-9)

            for name, res in results.items():
                pairs = res.n_selected * (res.n_selected - 1) / 2.0
                rows.append({
                    "seed": seed,
                    "budget": budget,
                    "n_docs": N_DOCS,
                    "method": name,
                    "score": res.score,
                    "relevance_sum": res.relevance_sum,
                    "redundancy_total": res.redundancy,
                    "redundancy_per_pair": (res.redundancy / pairs) if pairs > 0 else None,
                    "n_selected": res.n_selected,
                    "tokens_used": res.tokens_used,
                    "budget_utilization": res.budget_utilization,
                    "unique_coverage": unique_coverage(res, corpus),
                    "optimality_gap": ((ilp.score - res.score) / abs(ilp.score)
                                       if gap_valid else None),
                    "gap_is_valid": bool(gap_valid),
                    "ilp_proven_global_optimum": bool(ilp.is_proven_global_optimum),
                    "ilp_status": ilp.status.value,
                    "solve_status": res.status.value,
                    "solve_scope": res.scope.value,
                    "stop_reason": str(res.meta.get("stop_reason", "n/a")),
                    "latency_ms": res.latency_ms,
                    "density_diagnostic": res.density_diagnostic,  # diagnostic only
                    "indices": list(res.indices),
                })
        elapsed = time.time() - started
        print(f"[{si + 1:2d}/{len(SEEDS)}] seed={seed:2d} done "
              f"({elapsed / 60:.1f} min elapsed)", flush=True)

    metadata = {
        "experiment": "repaired_original_benchmark",
        "benchmark_config_version": BENCHMARK_CONFIG_VERSION,
        "protocol": "pre-registered; see module docstring",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "budgets": BUDGETS,
        "original_budgets": ORIGINAL_BUDGETS,
        "n_docs": N_DOCS,
        "objective_lambda": OBJECTIVE_LAMBDA,
        "mmr_lambda": MMR_LAMBDA,
        "lambda_provenance": ("objective_lambda was selected on the evaluation "
                              "corpus in v0 and is frozen here, NOT tuned on "
                              "these results; mmr_lambda is the untuned "
                              "textbook default"),
        "methods": list(METHOD_CALLS) + ["ilp"],
        "generator": "benchmark.generate_corpus (UNCHANGED from v0)",
        "objective": "Score(S) = sum r_i - objective_lambda * sum_{i<j} s_ij "
                     "s.t. sum w_i <= W_max (UNCHANGED from v0)",
        "stopping_policy": "positive_marginal_gain (common to all heuristics)",
        "embedding_backend": REQUESTED_MODEL,
        "tokenizer_encoding": getattr(embedder.token_counter, "encoding_name", UNAVAILABLE),
        "tokenizer_exact": bool(getattr(embedder.token_counter, "exact", False)),
        "ilp_solver": "PULP_CBC_CMD",
        "ilp_time_limit_s": ILP_TIME_LIMIT,
        "ilp_scope": SolveScope.FULL_CORPUS.value,
        "ilp_status_counts": ilp_status_counts,
        "git": _git_info(),
        "python_version": sys.version.split()[0],
        "packages": {n: _package_version(n) for n in
                     ("numpy", "scipy", "pulp", "sentence-transformers",
                      "tiktoken", "matplotlib")},
        "n_instances": len(SEEDS) * len(BUDGETS),
        "n_rows": len(rows),
        "runtime_minutes": (time.time() - started) / 60.0,
        "notes": [
            "density_diagnostic is recorded for continuity only and is NOT a "
            "primary metric.",
            "optimality_gap is null unless the ILP proved a global optimum for "
            "that instance.",
        ],
    }

    with open(os.path.join(OUT_DIR, "raw_results.json"), "w", encoding="utf-8") as fh:
        json.dump({"metadata": metadata,
                   "corpus_descriptives_per_seed": per_seed,
                   "rows": rows}, fh, indent=2, default=str)

    csv_fields = [f for f in rows[0] if f != "indices"]
    with open(os.path.join(OUT_DIR, "raw_results.csv"), "w", encoding="utf-8",
              newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} rows over {metadata['n_instances']} instances "
          f"in {metadata['runtime_minutes']:.1f} min")
    print(f"ILP status counts: {ilp_status_counts}")
    print(f"written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
