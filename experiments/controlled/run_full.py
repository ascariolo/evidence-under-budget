r"""STEP 4 - the full controlled experiment.

Executes the LOCKED protocol in ``experiments/controlled/protocol.py``:
4 relevance-length conditions x 2 redundancy levels x 3 budgets x 40 seeds,
five methods per instance-budget, plus arm E (realism anchor) kept separate.

This script performs DATA COLLECTION ONLY. It computes no aggregate statistic
and no test, so the run cannot be steered by its own output. Analysis lives in
``analyze_full.py``.

Condition D is generated through the locked two-pass W_sat procedure and is
tagged so it can never be pooled with A-C. Arm E is tagged ``E_realism_anchor``.

Writes results/controlled/full_experiment.{json,csv} and
results/controlled/full_experiment_manifest.json. Pre-flight artifacts are not
touched.
"""

from __future__ import annotations

import argparse, csv, json, os, platform, sys, time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from benchmark import (UNAVAILABLE, _git_info, _package_version, build_corpus,
                       generate_corpus, unique_coverage)
from embedder import Embedder
from experiments.controlled import protocol as P
from experiments.controlled.generator import (InstanceSpec, build_budgets,
                                              generate_instance, prepare_instance)
from optimizer import ContextKnapsack, SolveScope, SolveStatus

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "controlled")
SEEDS = list(range(40))

METHOD_CALLS = {
    "top_k": lambda k: k.solve_top_k(),
    "mmr": lambda k: k.solve_mmr(),
    "greedy_objective": lambda k: k.solve_greedy_objective(),
    "greedy_token_aware": lambda k: k.solve_greedy(),
}


def _knap(relevance, similarity, tokens, budget) -> ContextKnapsack:
    return ContextKnapsack(relevance, similarity, tokens, budget=budget,
                           objective_lambda=P.OBJECTIVE_LAMBDA,
                           mmr_lambda=P.MMR_LAMBDA)


def _selection_stats(indices, relevance, tokens, similarity) -> Dict[str, Any]:
    """Mechanism diagnostics for one selection. Descriptive only."""
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return {"tokens_of_selected_mean": None, "tokens_of_selected_median": None,
                "tokens_of_selected_min": None, "tokens_of_selected_max": None,
                "relevance_of_selected_mean": None,
                "marginal_utility_per_token_mean": None,
                "redundancy_per_pair": None, "n_pairs": 0}
    w = tokens[idx].astype(float)
    r = relevance[idx]
    pairs = idx.size * (idx.size - 1) / 2.0
    sub = similarity[np.ix_(idx, idx)]
    redundancy = float(sub.sum() / 2.0)
    return {"tokens_of_selected_mean": float(w.mean()),
            "tokens_of_selected_median": float(np.median(w)),
            "tokens_of_selected_min": float(w.min()),
            "tokens_of_selected_max": float(w.max()),
            "tokens_of_selected_std": float(w.std(ddof=1)) if idx.size > 1 else 0.0,
            "relevance_of_selected_mean": float(r.mean()),
            "marginal_utility_per_token_mean": float((r / w).mean()),
            "redundancy_per_pair": (redundancy / pairs) if pairs > 0 else None,
            "n_pairs": int(pairs)}


def _evaluate_budget(inst_rel, inst_sim, inst_tok, budget, time_limit,
                     coverage_corpus=None) -> Dict[str, Any]:
    """Run all five methods at one budget. Returns per-method records."""
    knap = _knap(inst_rel, inst_sim, inst_tok, budget)
    out: Dict[str, Any] = {}
    for name, call in METHOD_CALLS.items():
        t = time.perf_counter()
        res = call(knap)
        out[name] = (res, (time.perf_counter() - t) * 1000.0)
    t = time.perf_counter()
    ilp = knap.solve_ilp(time_limit=time_limit, scope=SolveScope.FULL_CORPUS)
    out["ilp"] = (ilp, (time.perf_counter() - t) * 1000.0)
    return out


def _rows(results, budget_level, budget, rho_target, common, inst_rel, inst_sim,
          inst_tok, coverage_corpus=None) -> List[Dict[str, Any]]:
    ilp, _ = results["ilp"]
    proven = ilp.is_proven_global_optimum
    rows = []
    for method, (res, ms) in results.items():
        stats = _selection_stats(res.indices, inst_rel, inst_tok, inst_sim)
        row = {
            **common,
            "budget_level": budget_level, "rho_target": rho_target,
            "w_max": int(budget), "method": method,
            "objective_score": float(res.score),
            "relevance_sum": float(res.relevance_sum),
            "redundancy_total": float(res.redundancy),
            "ilp_optimum": float(ilp.score),
            # Absolute gap: the reference objective can be non-positive, so no
            # ratio is used as the primary quantity (protocol section 4).
            "optimality_gap_absolute": float(ilp.score - res.score),
            "optimality_gap_relative": (float((ilp.score - res.score) / abs(ilp.score))
                                        if proven and abs(ilp.score) > 1e-9 else None),
            "gap_reference_is_proven_optimal": bool(proven),
            "selected_indices": [int(i) for i in res.indices],
            "tokens_used": int(res.tokens_used),
            "n_selected": int(res.n_selected),
            "budget_utilization": float(res.budget_utilization),
            "solve_status": res.status.value, "solve_scope": res.scope.value,
            "is_proven_global_optimum": bool(res.is_proven_global_optimum),
            "stop_reason": str(res.meta.get("stop_reason", "n/a")),
            "runtime_ms": float(ms),
            **stats,
        }
        if coverage_corpus is not None:
            row["unique_coverage"] = float(unique_coverage(res, coverage_corpus))
        if method == "ilp":
            row.update({
                "ilp_solver_status_raw": res.meta.get("pulp_status"),
                "ilp_solution_status_raw": res.meta.get("pulp_sol_status"),
                "ilp_proven_optimal": bool(res.is_proven_global_optimum),
                "ilp_solver_objective": res.meta.get("solver_objective"),
                "ilp_objective_consistent": res.meta.get("objective_consistent"),
                "ilp_model_build_ms": res.meta.get("model_build_ms"),
                "ilp_solver_ms": res.meta.get("solver_ms"),
                "ilp_n_pairs": res.meta.get("n_pairs"),
                "ilp_time_limit_s": res.meta.get("time_limit_s"),
            })
        rows.append(row)
    return rows


def run_synthetic(time_limit: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for seed in SEEDS:
        for rl in P.RL_CONDITIONS:
            for red in P.REDUNDANCY_LEVELS:
                spec = InstanceSpec(seed=seed, rl_condition=rl, redundancy_level=red)
                prep = prepare_instance(spec, time_limit=time_limit)
                inst, sat, budgets = prep["instance"], prep["saturation"], prep["budgets"]
                d = prep["diagnostics"]
                common = {
                    "arm": "synthetic_factorial",
                    "analysis_group": "ABC" if rl != "D" else "D_separate",
                    "instance_id": f"S{seed:02d}-{rl}-{red}",
                    "seed": seed, "rl_condition": rl,
                    "rl_condition_name": P.RL_CONDITIONS[rl]["name"],
                    "redundancy_level": red, "gamma": P.REDUNDANCY_LEVELS[red],
                    "rho_c": P.RL_CONDITIONS[rl]["rho_c"],
                    "beta_target": P.RL_CONDITIONS[rl]["target_beta"],
                    "beta_hat": d["beta_hat"],
                    "pearson_r_relevance_tokens": d["pearson_r_relevance_tokens"],
                    "spearman_rho_relevance_tokens": d["spearman_rho_relevance_tokens"],
                    "n_docs": P.N_DOCS, "n_topics": P.N_TOPICS,
                    "objective_lambda": P.OBJECTIVE_LAMBDA,
                    "mmr_lambda": P.MMR_LAMBDA,
                    "protocol_version": P.CONTROLLED_PROTOCOL_VERSION,
                    "w_sat": sat["w_sat"], "w_sat_proven": bool(sat["w_sat_proven"]),
                    "w_sat_status": sat["w_sat_status"],
                    "corpus_token_mass": sat["corpus_token_mass"],
                    "instance_similarity_mean": d["similarity_mean"],
                    "instance_tokens_cv": d["tokens_cv"],
                    "instance_relevance_mean": d["relevance_mean"],
                    "adversarial_applied": bool(d.get("adversarial_applied", False)),
                    "w_sat_pre_intervention": sat.get("w_sat_pre_intervention"),
                }
                for level, budget in budgets.items():
                    res = _evaluate_budget(inst.relevance, inst.similarity,
                                           inst.tokens, budget, time_limit)
                    rows.extend(_rows(res, level, budget, P.BUDGET_LEVELS[level],
                                      common, inst.relevance, inst.similarity,
                                      inst.tokens))
        print(f"  seed {seed:2d} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    return rows


def run_arm_e(time_limit: float) -> List[Dict[str, Any]]:
    """Realism anchor on the UNCHANGED original text generator. Never pooled."""
    from scipy.stats import pearsonr, spearmanr
    emb = Embedder(model_name="all-MiniLM-L6-v2")
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        query, docs = generate_corpus(n_docs=P.N_DOCS, seed=seed)
        corpus = build_corpus(query, docs, emb)
        rel = np.asarray(corpus.relevance, float)
        tok = np.asarray(corpus.tokens)
        pos = rel > 0
        n_pos = int(pos.sum())
        beta_hat = (float(np.polyfit(np.log(tok[pos].astype(float)),
                                     np.log(rel[pos]), 1)[0]) if n_pos >= 3 else None)
        sat = _knap(corpus.relevance, corpus.similarity, corpus.tokens,
                    int(tok.sum())).solve_ilp(time_limit=time_limit,
                                              scope=SolveScope.FULL_CORPUS)
        budgets = build_budgets(int(sat.tokens_used))
        iu = np.triu_indices(len(corpus), 1)
        common = {
            "arm": P.REALISM_ANCHOR_ARM, "analysis_group": "E_separate",
            "instance_id": f"E{seed:02d}", "seed": seed,
            "rl_condition": "E", "rl_condition_name": "realism_anchor",
            "redundancy_level": "n/a", "gamma": None, "rho_c": None,
            "beta_target": None, "beta_hat": beta_hat,
            "beta_hat_n_used": n_pos,
            "beta_hat_n_dropped_nonpositive": int(len(rel) - n_pos),
            "relevance_min": float(rel.min()),
            "pearson_r_relevance_tokens": float(pearsonr(rel, tok.astype(float))[0]),
            "spearman_rho_relevance_tokens": float(spearmanr(rel, tok.astype(float))[0]),
            "n_docs": len(corpus), "n_topics": None,
            "objective_lambda": P.OBJECTIVE_LAMBDA, "mmr_lambda": P.MMR_LAMBDA,
            "protocol_version": P.CONTROLLED_PROTOCOL_VERSION,
            "w_sat": int(sat.tokens_used),
            "w_sat_proven": bool(sat.is_proven_global_optimum),
            "w_sat_status": sat.status.value,
            "corpus_token_mass": int(tok.sum()),
            "instance_similarity_mean": float(corpus.similarity[iu].mean()),
            "instance_tokens_cv": float(tok.std(ddof=1) / tok.mean()),
            "instance_relevance_mean": float(rel.mean()),
            "adversarial_applied": False, "w_sat_pre_intervention": None,
            "embedding_backend": corpus.backend_name,
        }
        for level, budget in budgets.items():
            res = _evaluate_budget(corpus.relevance, corpus.similarity,
                                   corpus.tokens, budget, time_limit)
            rows.extend(_rows(res, level, budget, P.BUDGET_LEVELS[level], common,
                              corpus.relevance, corpus.similarity, corpus.tokens,
                              coverage_corpus=corpus))
        print(f"  arm E seed {seed:2d} done", flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-anchor", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)
    started = time.time()

    print("STEP 4 full controlled experiment - LOCKED protocol")
    print(f"  {len(SEEDS)} seeds x {len(P.RL_CONDITIONS)} RL x "
          f"{len(P.REDUNDANCY_LEVELS)} redundancy x {len(P.BUDGET_LEVELS)} budgets "
          f"x {len(METHOD_CALLS)+1} methods")
    rows = run_synthetic(P.ILP_TIME_LIMIT_S)
    if not args.no_anchor:
        print("arm E (realism anchor, analysed separately) ...")
        rows += run_arm_e(P.ILP_TIME_LIMIT_S)

    manifest = {
        "experiment": "controlled_full_experiment_step4",
        "protocol_version": P.CONTROLLED_PROTOCOL_VERSION,
        "protocol_lock": P.PROTOCOL_LOCK,
        "design_deviations": P.DESIGN_DEVIATIONS,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": SEEDS, "n_seeds": len(SEEDS),
        "rl_conditions": {k: dict(v) for k, v in P.RL_CONDITIONS.items()},
        "redundancy_levels": dict(P.REDUNDANCY_LEVELS),
        "budget_levels": dict(P.BUDGET_LEVELS),
        "generator_parameters": {
            "n_docs": P.N_DOCS, "embed_dim": P.EMBED_DIM, "n_topics": P.N_TOPICS,
            "relevance_beta_a": P.RELEVANCE_BETA_A, "relevance_beta_b": P.RELEVANCE_BETA_B,
            "relevance_min": P.RELEVANCE_MIN, "relevance_max": P.RELEVANCE_MAX,
            "log_token_mu": P.LOG_TOKEN_MU, "log_token_sigma": P.LOG_TOKEN_SIGMA,
            "token_min": P.TOKEN_MIN, "token_max": P.TOKEN_MAX,
            "adversarial_top_k": P.ADVERSARIAL_TOP_K,
            "adversarial_multiplier": P.ADVERSARIAL_MULTIPLIER,
            "adversarial_cap_fraction": P.ADVERSARIAL_CAP_FRACTION},
        "optimizer_parameters": {"objective_lambda": P.OBJECTIVE_LAMBDA,
                                 "mmr_lambda": P.MMR_LAMBDA,
                                 "stopping_policy": "positive_marginal_gain"},
        "solver_configuration": {"solver": "PULP_CBC_CMD",
                                 "time_limit_s": P.ILP_TIME_LIMIT_S,
                                 "scope": SolveScope.FULL_CORPUS.value,
                                 "linearization": "y_ij >= x_i + x_j - 1, y continuous"},
        "beta_criterion": P.BETA_CRITERION,
        "methods": list(METHOD_CALLS) + ["ilp"],
        "primary_estimand": ("Delta_gap = gap(greedy_objective) - gap(greedy_token_aware) "
                             "= score(greedy_token_aware) - score(greedy_objective); "
                             "gap measured in ABSOLUTE objective units"),
        "analysis_groups": {"ABC": "synthetic factorial, pooled",
                            "D_separate": "adversarial condition, never pooled with ABC",
                            "E_separate": "realism anchor, never pooled with synthetic"},
        "git": _git_info(), "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {n: _package_version(n) for n in
                     ("numpy", "scipy", "pulp", "statsmodels", "pandas",
                      "sentence-transformers", "tiktoken", "matplotlib")},
        "n_rows": len(rows),
        "runtime_minutes": (time.time() - started) / 60.0,
        "notes": ["Data collection only; no aggregate statistic computed here.",
                  "A heuristic is never described as optimal: solve_status carries "
                  "heuristic_no_guarantee for all four heuristics.",
                  "Condition D and arm E are tagged via analysis_group and must not "
                  "be pooled with ABC."],
    }

    with open(os.path.join(OUT, "full_experiment.json"), "w", encoding="utf-8") as fh:
        json.dump({"manifest": manifest, "rows": rows}, fh, indent=2, default=str)
    with open(os.path.join(OUT, "full_experiment_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    fields = sorted({k for r in rows for k in r if k != "selected_indices"})
    with open(os.path.join(OUT, "full_experiment.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    print(f"\n{len(rows)} rows in {manifest['runtime_minutes']:.1f} min -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
