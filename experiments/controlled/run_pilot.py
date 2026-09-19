r"""Controlled-benchmark PILOT: validation only, not statistical inference.

Purpose (design section 14 / step 3B item 12): confirm that the generator
produces non-degenerate instances, that the ILP proves optimality in every cell,
that budget levels separate, and that ``greedy_objective`` and
``greedy_token_aware`` actually diverge. A small number of seeds, deliberately
too few for inference.

THIS SCRIPT COMPUTES NO AGGREGATE STATISTIC AND DRAWS NO CONCLUSION. The paired
difference between the two greedy methods is recorded per instance purely as
evidence that the mechanism can produce a non-zero difference at all.

Writes ``results/controlled/pilot_report.json`` (+ ``.csv``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from benchmark import (UNAVAILABLE, _git_info, _package_version, build_corpus,
                       generate_corpus, unique_coverage)
from embedder import Embedder
from experiments.controlled import protocol as P
from experiments.controlled.generator import (GeneratorConfig, InstanceSpec,
                                              build_budgets, compute_saturation,
                                              generate_instance,
                                              prepare_instance)
from optimizer import ContextKnapsack, SolveScope, SolveStatus

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "results", "controlled")

METHODS = ("top_k", "mmr", "greedy_objective", "greedy_token_aware", "ilp")


def _knap(relevance, similarity, tokens, budget) -> ContextKnapsack:
    return ContextKnapsack(relevance, similarity, tokens, budget=budget,
                           objective_lambda=P.OBJECTIVE_LAMBDA,
                           mmr_lambda=P.MMR_LAMBDA)


def _solve_all(knap, time_limit: float) -> Dict[str, Any]:
    res = {
        "top_k": knap.solve_top_k(),
        "mmr": knap.solve_mmr(),
        "greedy_objective": knap.solve_greedy_objective(),
        "greedy_token_aware": knap.solve_greedy(),
    }
    res["ilp"] = knap.solve_ilp(time_limit=time_limit,
                                scope=SolveScope.FULL_CORPUS)
    return res


def _rows_for(prepared, results, budget_name, budget, corpus_for_coverage=None) -> List[Dict[str, Any]]:
    ilp = results["ilp"]
    gap_valid = ilp.is_proven_global_optimum and abs(ilp.score) > 1e-9
    rows = []
    for method, r in results.items():
        pairs = r.n_selected * (r.n_selected - 1) / 2.0
        rows.append({
            "method": method,
            "budget_level": budget_name,
            "w_max": budget,
            "score": r.score,
            "relevance_sum": r.relevance_sum,
            "redundancy_total": r.redundancy,
            "redundancy_per_pair": (r.redundancy / pairs) if pairs > 0 else None,
            "n_selected": r.n_selected,
            "tokens_used": r.tokens_used,
            "budget_utilization": r.budget_utilization,
            "optimality_gap": ((ilp.score - r.score) / abs(ilp.score))
                              if gap_valid else None,
            "gap_is_valid": bool(gap_valid),
            "ilp_status": ilp.status.value,
            "ilp_proven_global_optimum": bool(ilp.is_proven_global_optimum),
            "solve_status": r.status.value,
            "solve_scope": r.scope.value,
            "stop_reason": str(r.meta.get("stop_reason", "n/a")),
            "indices": list(r.indices),
        })
    return rows


def run_synthetic(seeds: List[int], time_limit: float) -> List[Dict[str, Any]]:
    instances: List[Dict[str, Any]] = []
    for seed in seeds:
        for rl in P.RL_CONDITIONS:
            for red in P.REDUNDANCY_LEVELS:
                spec = InstanceSpec(seed=seed, rl_condition=rl, redundancy_level=red)
                prepared = prepare_instance(spec, time_limit=time_limit)
                inst, sat, budgets = (prepared["instance"], prepared["saturation"],
                                      prepared["budgets"])
                rows: List[Dict[str, Any]] = []
                for name, budget in budgets.items():
                    results = _solve_all(
                        _knap(inst.relevance, inst.similarity, inst.tokens, budget),
                        time_limit)
                    rows.extend(_rows_for(prepared, results, name, budget))
                by = {(r["budget_level"], r["method"]): r for r in rows}
                instances.append({
                    "arm": "synthetic_factorial",
                    "metadata": prepared["metadata"],
                    "diagnostics": {k: v for k, v in prepared["diagnostics"].items()
                                    if k != "validation"},
                    "validation": prepared["diagnostics"]["validation"],
                    "saturation": sat,
                    "budgets": budgets,
                    "realized_rho": prepared["realized_rho"],
                    "rows": rows,
                    # Recorded as evidence the mechanism can move at all.
                    # NOT a statistical result and NOT to be interpreted.
                    "paired_diff_tokenaware_minus_objective": {
                        name: by[(name, "greedy_token_aware")]["score"]
                              - by[(name, "greedy_objective")]["score"]
                        for name in budgets},
                    "selections_diverged": {
                        name: by[(name, "greedy_token_aware")]["indices"]
                              != by[(name, "greedy_objective")]["indices"]
                        for name in budgets},
                })
        print(f"  synthetic seed {seed} done", flush=True)
    return instances


def run_realism_anchor(seeds: List[int], time_limit: float) -> List[Dict[str, Any]]:
    """Arm E: the UNCHANGED original text generator, never pooled with the
    synthetic factorial. Budgets come from the same instance-relative rho rule."""
    from scipy.stats import pearsonr, spearmanr
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    out: List[Dict[str, Any]] = []
    for seed in seeds:
        query, documents = generate_corpus(n_docs=P.N_DOCS, seed=seed)
        corpus = build_corpus(query, documents, embedder)
        rel = np.asarray(corpus.relevance, dtype=float)
        tok = np.asarray(corpus.tokens, dtype=float)
        # The real encoder can return a NEGATIVE cosine, for which the log-log
        # elasticity is undefined. The synthetic generator cannot (its relevance
        # floor is > 0), so this is an anchor-arm-only concern. Compute the slope
        # on the positive subset and record how many documents it used, rather
        # than silently emitting nan.
        positive = rel > 0
        n_pos = int(positive.sum())
        beta_hat = (float(np.polyfit(np.log(tok[positive]),
                                     np.log(rel[positive]), 1)[0])
                    if n_pos >= 3 else None)
        total = int(tok.sum())
        sat_res = _knap(corpus.relevance, corpus.similarity, corpus.tokens,
                        total).solve_ilp(time_limit=time_limit,
                                         scope=SolveScope.FULL_CORPUS)
        w_sat = int(sat_res.tokens_used)
        budgets = build_budgets(w_sat)
        rows: List[Dict[str, Any]] = []
        for name, budget in budgets.items():
            results = _solve_all(_knap(corpus.relevance, corpus.similarity,
                                       corpus.tokens, budget), time_limit)
            rows.extend(_rows_for(None, results, name, budget))
        by = {(r["budget_level"], r["method"]): r for r in rows}
        iu = np.triu_indices(len(corpus), 1)
        out.append({
            "arm": P.REALISM_ANCHOR_ARM,
            "metadata": {"protocol_version": P.CONTROLLED_PROTOCOL_VERSION,
                         "seed": seed, "rl_condition": "E", "redundancy_level": "n/a",
                         "generator": "benchmark.generate_corpus (UNCHANGED)",
                         "n_docs": P.N_DOCS,
                         "objective_lambda": P.OBJECTIVE_LAMBDA,
                         "mmr_lambda": P.MMR_LAMBDA,
                         "backend": corpus.backend_name},
            "diagnostics": {
                "beta_target": None,
                "beta_hat": beta_hat,
                "beta_hat_n_used": n_pos,
                "beta_hat_n_dropped_nonpositive": int(len(rel) - n_pos),
                "pearson_r_relevance_tokens": float(pearsonr(rel, tok)[0]),
                "spearman_rho_relevance_tokens": float(spearmanr(rel, tok)[0]),
                "relevance_mean": float(rel.mean()), "relevance_std": float(rel.std(ddof=1)),
                "relevance_min": float(rel.min()), "relevance_max": float(rel.max()),
                "tokens_mean": float(tok.mean()), "tokens_std": float(tok.std(ddof=1)),
                "tokens_min": int(tok.min()), "tokens_max": int(tok.max()),
                "tokens_cv": float(tok.std(ddof=1) / tok.mean()),
                "tokens_total": total,
                "similarity_mean": float(corpus.similarity[iu].mean()),
                "similarity_std": float(corpus.similarity[iu].std(ddof=1)),
                "similarity_min": float(corpus.similarity[iu].min()),
                "similarity_max": float(corpus.similarity[iu].max()),
                "n_docs": len(corpus), "gamma": None, "adversarial_applied": False,
            },
            "saturation": {"w_sat": w_sat, "corpus_token_mass": total,
                           "w_sat_proven": bool(sat_res.is_proven_global_optimum),
                           "w_sat_status": sat_res.status.value,
                           "w_sat_n_selected": sat_res.n_selected},
            "budgets": budgets,
            "realized_rho": {n: budgets[n] / w_sat for n in budgets},
            "rows": rows,
            "paired_diff_tokenaware_minus_objective": {
                n: by[(n, "greedy_token_aware")]["score"]
                   - by[(n, "greedy_objective")]["score"] for n in budgets},
            "selections_diverged": {
                n: by[(n, "greedy_token_aware")]["indices"]
                   != by[(n, "greedy_objective")]["indices"] for n in budgets},
        })
        print(f"  anchor seed {seed} done", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--anchor-seeds", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=P.ILP_TIME_LIMIT_S)
    ap.add_argument("--no-anchor", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)

    started = time.time()
    print(f"pilot: {args.seeds} seeds x {len(P.RL_CONDITIONS)} RL x "
          f"{len(P.REDUNDANCY_LEVELS)} redundancy x {len(P.BUDGET_LEVELS)} budgets")
    instances = run_synthetic(list(range(args.seeds)), args.time_limit)
    if not args.no_anchor:
        instances += run_realism_anchor(list(range(args.anchor_seeds)), args.time_limit)

    payload = {
        "pilot": True,
        "purpose": ("generator validation only - not statistical inference; "
                    "no conclusion may be drawn from these numbers"),
        "protocol_version": P.CONTROLLED_PROTOCOL_VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_seeds": args.seeds, "anchor_seeds": args.anchor_seeds,
        "rl_conditions": {k: dict(v) for k, v in P.RL_CONDITIONS.items()},
        "budget_levels": P.BUDGET_LEVELS,
        "redundancy_levels": P.REDUNDANCY_LEVELS,
        "objective_lambda": P.OBJECTIVE_LAMBDA, "mmr_lambda": P.MMR_LAMBDA,
        "ilp_time_limit_s": args.time_limit,
        "design_deviations": P.DESIGN_DEVIATIONS,
        "git": _git_info(),
        "python_version": sys.version.split()[0],
        "packages": {n: _package_version(n) for n in
                     ("numpy", "scipy", "pulp", "sentence-transformers",
                      "tiktoken", "matplotlib")},
        "runtime_minutes": None,
        "instances": instances,
    }
    payload["runtime_minutes"] = (time.time() - started) / 60.0

    with open(os.path.join(OUT_DIR, "pilot_report.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    flat: List[Dict[str, Any]] = []
    for inst in instances:
        base = {**{f"meta_{k}": v for k, v in inst["metadata"].items()
                   if k not in ("config",)},
                **{f"diag_{k}": v for k, v in inst["diagnostics"].items()},
                "arm": inst["arm"], "w_sat": inst["saturation"]["w_sat"],
                "w_sat_proven": inst["saturation"]["w_sat_proven"],
                "corpus_token_mass": inst["saturation"]["corpus_token_mass"]}
        for r in inst["rows"]:
            flat.append({**base, **{k: v for k, v in r.items() if k != "indices"},
                         "rho_realized": inst["realized_rho"][r["budget_level"]]})
    with open(os.path.join(OUT_DIR, "pilot_report.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in flat for k in r}),
                           extrasaction="ignore")
        w.writeheader(); w.writerows(flat)

    print(f"\n{len(instances)} instances, {len(flat)} rows in "
          f"{payload['runtime_minutes']:.1f} min -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
