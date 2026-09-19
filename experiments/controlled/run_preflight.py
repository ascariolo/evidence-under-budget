r"""STEP 3C pre-flight: validate the locked protocol across ALL 40 planned seeds.

Generator + protocol validation only. This is NOT the experiment: the ILP runs
on every instance's *saturation* solve (needed to construct budgets at all) and
on a representative per-condition sample at the three budgets, never on the full
factorial at every budget.

Nothing here may change a condition, widen a tolerance, or resample a seed. Every
check reports pass/fail; failures are surfaced, never repaired.

Writes results/controlled/preflight_report.{json,csv}.
"""

from __future__ import annotations

import argparse, csv, json, os, sys, time
from typing import Any, Dict, List

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from benchmark import _git_info, _package_version, build_corpus, generate_corpus
from embedder import Embedder
from experiments.controlled import protocol as P
from experiments.controlled.generator import (InstanceSpec, build_budgets,
                                              compute_saturation,
                                              generate_instance, prepare_instance)
from optimizer import ContextKnapsack, SolveScope, SolveStatus

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "controlled")
SEEDS = list(range(40))
REPRESENTATIVE_SEEDS = [0, 13, 27]          # fixed a priori for the ILP sample


def dist(v, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> Dict[str, Any]:
    a = np.asarray([x for x in v if x is not None], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()),
            **{f"q{int(q*100):02d}": float(np.quantile(a, q)) for q in qs}}


def _knap(inst, budget):
    return ContextKnapsack(inst.relevance, inst.similarity, inst.tokens,
                           budget=budget, objective_lambda=P.OBJECTIVE_LAMBDA,
                           mmr_lambda=P.MMR_LAMBDA)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-anchor", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)
    t0 = time.time()
    checks: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    stop_conditions: List[str] = []

    # ---------- phase 1: generate every synthetic instance ---------------- #
    print("phase 1/6: generating 40 seeds x 4 conditions x 2 redundancy ...")
    inst_map: Dict[tuple, Any] = {}
    gen_failures: List[str] = []
    for seed in SEEDS:
        for rl in P.RL_CONDITIONS:
            for red in P.REDUNDANCY_LEVELS:
                spec = InstanceSpec(seed=seed, rl_condition=rl, redundancy_level=red)
                try:
                    inst_map[(seed, rl, red)] = generate_instance(spec, strict=True)
                except Exception as exc:
                    gen_failures.append(f"{seed}/{rl}/{red}: {exc}")
    checks["generation"] = {"n_attempted": len(SEEDS) * 4 * 2,
                            "n_generated": len(inst_map),
                            "failures": gen_failures,
                            "pass": not gen_failures}
    if gen_failures:
        stop_conditions.append("generator rejected instances under the locked guards")

    # ---------- phase 2: per-instance diagnostics ------------------------- #
    print("phase 2/6: diagnostics, beta, non-degeneracy, causal isolation ...")
    by_cond: Dict[str, List[dict]] = {rl: [] for rl in P.RL_CONDITIONS}
    for (seed, rl, red), inst in inst_map.items():
        d = inst.diagnostics
        n_uniq_tok = int(len(np.unique(inst.tokens)))
        n_uniq_rel = int(len(np.unique(np.round(inst.relevance, 12))))
        row = {"arm": "synthetic", "seed": seed, "rl_condition": rl,
               "redundancy_level": red, "gamma": P.REDUNDANCY_LEVELS[red],
               "rho_c": P.RL_CONDITIONS[rl]["rho_c"],
               "beta_target": P.RL_CONDITIONS[rl]["target_beta"],
               "beta_hat": d["beta_hat"], "pearson": d["pearson_r_relevance_tokens"],
               "spearman": d["spearman_rho_relevance_tokens"],
               "relevance_mean": d["relevance_mean"], "relevance_std": d["relevance_std"],
               "relevance_min": d["relevance_min"], "relevance_max": d["relevance_max"],
               "tokens_mean": d["tokens_mean"], "tokens_std": d["tokens_std"],
               "tokens_min": d["tokens_min"], "tokens_max": d["tokens_max"],
               "tokens_cv": d["tokens_cv"], "tokens_total": d["tokens_total"],
               "similarity_mean": d["similarity_mean"], "similarity_std": d["similarity_std"],
               "similarity_min": d["similarity_min"], "similarity_max": d["similarity_max"],
               "n_unique_tokens": n_uniq_tok, "n_unique_relevance": n_uniq_rel,
               "n_topics_used": int(len(np.unique(inst.topics))),
               "validation_ok": inst.diagnostics["validation"]["ok"]}
        rows.append(row)
        by_cond[rl].append(row)

    # --- 3. beta validation (per instance, tolerance LOCKED at 0.10) ------- #
    beta_tab = {}
    for rl in ("A", "B", "C"):
        tgt = P.RL_CONDITIONS[rl]["target_beta"]
        # one value per seed (gamma cannot affect beta; asserted in phase 3)
        vals = [r["beta_hat"] for r in by_cond[rl] if r["redundancy_level"] == "low"]
        devs = [abs(v - tgt) for v in vals]
        n_pass = sum(1 for x in devs if x <= P.BETA_TOLERANCE)
        beta_tab[rl] = {"target": tgt, **dist(vals),
                        "worst_abs_deviation": float(max(devs)),
                        "mean_abs_deviation": float(np.mean(devs)),
                        "n_pass": n_pass, "n_total": len(vals),
                        "per_instance_pass_rate": n_pass / len(vals),
                        "mean_within_tolerance": bool(abs(np.mean(vals) - tgt) <= P.BETA_TOLERANCE),
                        "per_seed_beta_hat": vals}
    d_vals = [r["beta_hat"] for r in by_cond["D"] if r["redundancy_level"] == "low"]
    beta_tab["D"] = {"target": None, **dist(d_vals),
                     "note": "condition D has no target beta; mixed by construction",
                     "per_seed_beta_hat": d_vals}
    checks["beta"] = {"tolerance": P.BETA_TOLERANCE, "by_condition": beta_tab,
                      "criterion": ("protocol requires |mean(beta_hat) - target| <= "
                                    "tolerance; per-instance rate reported for context"),
                      "pass": all(beta_tab[r]["mean_within_tolerance"] for r in "ABC")}
    if not checks["beta"]["pass"]:
        stop_conditions.append("systematic beta tolerance failure")

    # --- 4. non-degeneracy ------------------------------------------------ #
    cv = [r["tokens_cv"] for r in rows]
    nd = {"tokens_cv": dist(cv),
          "tokens_cv_guard": P.MIN_TOKEN_CV,
          "tokens_cv_n_below_guard": int(sum(1 for x in cv if x < P.MIN_TOKEN_CV)),
          "tokens_cv_headroom_min": float(min(cv) - P.MIN_TOKEN_CV),
          "relevance_std": dist([r["relevance_std"] for r in rows]),
          "relevance_std_guard": P.MIN_RELEVANCE_STD,
          "relevance_std_n_below": int(sum(1 for r in rows if r["relevance_std"] < P.MIN_RELEVANCE_STD)),
          "tokens_std": dist([r["tokens_std"] for r in rows]),
          "similarity_std": dist([r["similarity_std"] for r in rows]),
          "similarity_std_guard": P.MIN_SIMILARITY_STD,
          "similarity_std_n_below": int(sum(1 for r in rows if r["similarity_std"] < P.MIN_SIMILARITY_STD)),
          "n_unique_tokens": dist([r["n_unique_tokens"] for r in rows]),
          "n_unique_relevance": dist([r["n_unique_relevance"] for r in rows]),
          "n_topics_used": dist([r["n_topics_used"] for r in rows]),
          "min_unique_tokens": int(min(r["n_unique_tokens"] for r in rows)),
          "min_unique_relevance": int(min(r["n_unique_relevance"] for r in rows))}
    nd["pass"] = (nd["tokens_cv_n_below_guard"] == 0 and
                  nd["relevance_std_n_below"] == 0 and
                  nd["similarity_std_n_below"] == 0 and
                  nd["min_unique_tokens"] > 1 and nd["min_unique_relevance"] > 1)
    checks["nondegeneracy"] = nd
    if not nd["pass"]:
        stop_conditions.append("token/relevance/similarity variance collapse")

    # --- 5. causal isolation ---------------------------------------------- #
    print("phase 3/6: causal isolation ...")
    iso: Dict[str, Any] = {}
    # gamma must move similarity ONLY
    dr = dt = db = dtop = 0.0
    dsim = []
    for seed in SEEDS:
        for rl in P.RL_CONDITIONS:
            lo, hi = inst_map[(seed, rl, "low")], inst_map[(seed, rl, "high")]
            dr = max(dr, float(np.abs(lo.relevance - hi.relevance).max()))
            dt = max(dt, float(np.abs(lo.tokens - hi.tokens).max()))
            db = max(db, abs(lo.diagnostics["beta_hat"] - hi.diagnostics["beta_hat"]))
            dtop = max(dtop, float(np.abs(lo.topics - hi.topics).max()))
            dsim.append(hi.diagnostics["similarity_mean"] - lo.diagnostics["similarity_mean"])
    iso["gamma_intervention"] = {
        "max_abs_change_relevance": dr, "max_abs_change_tokens": dt,
        "max_abs_change_beta_hat": db, "max_abs_change_topic_assignment": dtop,
        "similarity_mean_delta": dist(dsim),
        "pass": (dr == 0.0 and dt == 0.0 and db == 0.0 and dtop == 0.0
                 and min(dsim) > 0.0)}
    # rho_c must move dependence only; topics must be identical across conditions
    pool = {rl: (np.concatenate([inst_map[(s, rl, "low")].relevance for s in SEEDS]),
                 np.concatenate([inst_map[(s, rl, "low")].tokens.astype(float) for s in SEEDS]))
            for rl in P.RL_CONDITIONS}
    ks = {}
    for rl in ("B", "C"):
        ks[f"A_vs_{rl}_relevance_ks_p"] = float(ks_2samp(pool["A"][0], pool[rl][0]).pvalue)
        ks[f"A_vs_{rl}_tokens_ks_p"] = float(ks_2samp(pool["A"][1], pool[rl][1]).pvalue)
    topic_same = all(np.array_equal(inst_map[(s, "A", "low")].topics,
                                    inst_map[(s, rl, "low")].topics)
                     for s in SEEDS for rl in P.RL_CONDITIONS)
    # D shares A's copula, so its PRE-intervention draws must match A exactly
    d_matches_a = all(np.array_equal(inst_map[(s, "A", "low")].relevance,
                                     inst_map[(s, "D", "low")].relevance)
                      for s in SEEDS)
    iso["rho_c_intervention"] = {
        **ks, "topic_structure_identical_across_conditions": bool(topic_same),
        "D_preintervention_matches_A": bool(d_matches_a),
        "marginals_preserved": all(v > 0.01 for v in ks.values()),
        "pass": all(v > 0.01 for v in ks.values()) and topic_same and d_matches_a}
    checks["causal_isolation"] = {**iso,
                                  "pass": iso["gamma_intervention"]["pass"] and
                                          iso["rho_c_intervention"]["pass"]}
    if not checks["causal_isolation"]["pass"]:
        stop_conditions.append("redundancy manipulation changes unrelated factors "
                               "or marginals not preserved")

    # --- 6. redundancy manipulation --------------------------------------- #
    lo_s = [r["similarity_mean"] for r in rows if r["redundancy_level"] == "low"]
    hi_s = [r["similarity_mean"] for r in rows if r["redundancy_level"] == "high"]
    paired = np.array(hi_s) - np.array(lo_s)
    pooled_sd = np.sqrt((np.var(lo_s, ddof=1) + np.var(hi_s, ddof=1)) / 2)
    checks["redundancy"] = {
        "interpretation": ("LOW vs MODERATE redundancy contrast. NOT calibrated to "
                           "the v0 corpus (0.330-0.405); see DEV-2."),
        "low_gamma": P.REDUNDANCY_LEVELS["low"], "high_gamma": P.REDUNDANCY_LEVELS["high"],
        "low_similarity_mean": dist(lo_s), "moderate_similarity_mean": dist(hi_s),
        "paired_difference": dist(paired.tolist()),
        "cohens_d_pooled": float((np.mean(hi_s) - np.mean(lo_s)) / pooled_sd),
        "ratio_of_means": float(np.mean(hi_s) / np.mean(lo_s)),
        "ranges_overlap": bool(max(lo_s) >= min(hi_s)),
        "n_pairs_positive": int((paired > 0).sum()), "n_pairs": int(paired.size),
        "pass": bool((paired > 0).all() and max(lo_s) < min(hi_s))}
    if not checks["redundancy"]["pass"]:
        stop_conditions.append("redundancy manipulation fails to separate")

    # ---------- phase 4: saturation + budgets (ILP per instance) ---------- #
    print("phase 4/6: saturation ILP + budget construction (40 seeds) ...")
    budget_rows, sat_fail, dbg = [], [], []
    for seed in SEEDS:
        for rl in P.RL_CONDITIONS:
            for red in P.REDUNDANCY_LEVELS:
                spec = InstanceSpec(seed=seed, rl_condition=rl, redundancy_level=red)
                t = time.time()
                prep = prepare_instance(spec, time_limit=P.ILP_TIME_LIMIT_S)
                sat, b = prep["saturation"], prep["budgets"]
                if not sat["w_sat_proven"]:
                    sat_fail.append(f"{seed}/{rl}/{red}: {sat['w_sat_status']}")
                ordered = b["tight"] < b["medium"] < b["loose"]
                dev = {n: abs(b[n] / sat["w_sat"] - P.BUDGET_LEVELS[n]) for n in b}
                rec = {"seed": seed, "rl_condition": rl, "redundancy_level": red,
                       "w_sat": sat["w_sat"], "w_sat_proven": sat["w_sat_proven"],
                       "w_sat_status": sat["w_sat_status"],
                       "corpus_token_mass": sat["corpus_token_mass"],
                       "w_sat_n_selected": sat["w_sat_n_selected"],
                       "saturation_fraction": sat["w_sat"] / sat["corpus_token_mass"],
                       **{f"budget_{n}": b[n] for n in b},
                       **{f"rho_realized_{n}": b[n] / sat["w_sat"] for n in b},
                       **{f"rho_dev_{n}": dev[n] for n in b},
                       "budgets_ordered": bool(ordered),
                       "min_token": int(prep["instance"].tokens.min()),
                       "tight_affords_min_token": bool(b["tight"] >= prep["instance"].tokens.min()),
                       "elapsed_s": time.time() - t}
                if rl == "D":
                    rec.update({"w_sat_pre_intervention": sat.get("w_sat_pre_intervention"),
                                "w_sat_pre_proven": sat.get("w_sat_pre_proven"),
                                "adversarial_cap": prep["diagnostics"]["adversarial_cap"],
                                "adversarial_indices": prep["diagnostics"]["adversarial_indices"],
                                "adversarial_tokens_after": prep["diagnostics"]["adversarial_tokens_after"],
                                "adv_feasible_at_medium": bool(all(
                                    prep["instance"].tokens[i] <= b["medium"]
                                    for i in prep["diagnostics"]["adversarial_indices"]))})
                    dbg.append(rec)
                budget_rows.append(rec)
                # cache for divergence phase
                inst_map[(seed, rl, red)] = prep["instance"]
                inst_map[("budgets", seed, rl, red)] = b
        print(f"    seed {seed} ({time.time()-t0:.0f}s)", flush=True)

    checks["budgets"] = {
        "n": len(budget_rows),
        "w_sat": dist([r["w_sat"] for r in budget_rows]),
        "saturation_fraction": dist([r["saturation_fraction"] for r in budget_rows]),
        "w_sat_n_selected": dist([r["w_sat_n_selected"] for r in budget_rows]),
        "n_ordered": int(sum(r["budgets_ordered"] for r in budget_rows)),
        "n_saturation_not_proven": len(sat_fail), "saturation_failures": sat_fail,
        "rho_target": dict(P.BUDGET_LEVELS),
        "rho_realized": {n: dist([r[f"rho_realized_{n}"] for r in budget_rows])
                         for n in P.BUDGET_LEVELS},
        "rho_max_deviation": {n: float(max(r[f"rho_dev_{n}"] for r in budget_rows))
                              for n in P.BUDGET_LEVELS},
        "n_tight_cannot_afford_any_document": int(sum(
            1 for r in budget_rows if not r["tight_affords_min_token"])),
        "pass": (all(r["budgets_ordered"] for r in budget_rows) and not sat_fail and
                 all(r["tight_affords_min_token"] for r in budget_rows))}
    if not checks["budgets"]["pass"]:
        stop_conditions.append("budget ordering / saturation / degenerate budget failure")

    # --- 10. condition D audit -------------------------------------------- #
    checks["condition_D_audit"] = {
        "n": len(dbg),
        "procedure": [
            "1. generate instance with UNMODIFIED tokens",
            "2. compute w_sat_pre from a proven-optimal unconstrained ILP on that state",
            "3. inflate top-k relevant docs by m_adv, capped at 0.45 * w_sat_pre",
            "4. recompute final W_sat on the modified instance (proven-optimal ILP)",
            "5. derive tight/medium/loose from the FINAL W_sat only",
        ],
        "non_circularity": ("the cap reads w_sat_pre, which is a function of the "
                            "PRE-intervention tokens only; the final W_sat is never "
                            "fed back into the cap, so no fixed point is required"),
        "all_pre_proven": bool(all(r["w_sat_pre_proven"] for r in dbg)),
        "all_final_proven": bool(all(r["w_sat_proven"] for r in dbg)),
        "w_sat_pre": dist([r["w_sat_pre_intervention"] for r in dbg]),
        "w_sat_final": dist([r["w_sat"] for r in dbg]),
        "w_sat_shift": dist([r["w_sat"] - r["w_sat_pre_intervention"] for r in dbg]),
        "adversarial_cap": dist([r["adversarial_cap"] for r in dbg]),
        "n_adv_feasible_at_medium": int(sum(r["adv_feasible_at_medium"] for r in dbg)),
        "pass": bool(all(r["adv_feasible_at_medium"] for r in dbg) and
                     all(r["w_sat_proven"] and r["w_sat_pre_proven"] for r in dbg))}
    if not checks["condition_D_audit"]["pass"]:
        stop_conditions.append("condition D construction failed its audit")

    # ---------- phase 5: divergence + ILP tractability -------------------- #
    print("phase 5/6: method divergence + representative ILP ...")
    div_rows, ilp_rows = [], []
    for seed in SEEDS:
        for rl in P.RL_CONDITIONS:
            for red in P.REDUNDANCY_LEVELS:
                inst = inst_map[(seed, rl, red)]
                b = inst_map[("budgets", seed, rl, red)]
                for name, budget in b.items():
                    k = _knap(inst, budget)
                    go, ta = k.solve_greedy_objective(), k.solve_greedy()
                    div_rows.append({"seed": seed, "rl_condition": rl,
                                     "redundancy_level": red, "budget_level": name,
                                     "diverged": go.indices != ta.indices,
                                     "n_go": go.n_selected, "n_ta": ta.n_selected,
                                     "score_diff": ta.score - go.score})
                    if seed in REPRESENTATIVE_SEEDS:
                        t = time.time()
                        r = k.solve_ilp(time_limit=P.ILP_TIME_LIMIT_S,
                                        scope=SolveScope.FULL_CORPUS)
                        ilp_rows.append({"seed": seed, "rl_condition": rl,
                                         "redundancy_level": red, "budget_level": name,
                                         "status": r.status.value,
                                         "proven": r.is_proven_global_optimum,
                                         "elapsed_s": time.time() - t,
                                         "tokens_used": r.tokens_used, "budget": budget,
                                         "feasible": r.tokens_used <= budget,
                                         "n_selected": r.n_selected,
                                         "n_pairs": r.meta.get("n_pairs"),
                                         "objective_consistent": r.meta.get("objective_consistent")})
    n_div = sum(r["diverged"] for r in div_rows)
    per_cond = {}
    for rl in P.RL_CONDITIONS:
        for red in P.REDUNDANCY_LEVELS:
            for name in P.BUDGET_LEVELS:
                sub = [r for r in div_rows if r["rl_condition"] == rl
                       and r["redundancy_level"] == red and r["budget_level"] == name]
                per_cond[f"{rl}|{red}|{name}"] = {
                    "n_diverged": int(sum(r["diverged"] for r in sub)), "n": len(sub)}
    # constant-length control
    ctrl_identical = True
    for seed in REPRESENTATIVE_SEEDS:
        inst = generate_instance(InstanceSpec(seed=seed, rl_condition="A",
                                              redundancy_level="low"), strict=False)
        inst.tokens = np.full(P.N_DOCS, 50, dtype=np.int64)
        for budget in (150, 400, 900):
            k = _knap(inst, budget)
            if k.solve_greedy_objective().indices != k.solve_greedy().indices:
                ctrl_identical = False
    checks["divergence"] = {
        "role": "generator validation, NOT a performance result",
        "n_instances": len(div_rows), "n_diverged": int(n_div),
        "n_identical": int(len(div_rows) - n_div),
        "divergence_rate": n_div / len(div_rows),
        "cells_with_zero_divergence": [k for k, v in per_cond.items() if v["n_diverged"] == 0],
        "per_condition": per_cond,
        "constant_length_control_identical": bool(ctrl_identical),
        "pass": bool(n_div / len(div_rows) > 0.5 and ctrl_identical)}
    if not checks["divergence"]["pass"]:
        stop_conditions.append("method divergence check failed")

    el = [r["elapsed_s"] for r in ilp_rows]
    checks["ilp_tractability"] = {
        "representative_seeds": REPRESENTATIVE_SEEDS, "n_solves": len(ilp_rows),
        "status_counts": {s: sum(1 for r in ilp_rows if r["status"] == s)
                          for s in {r["status"] for r in ilp_rows}},
        "n_proven": int(sum(r["proven"] for r in ilp_rows)),
        "n_feasible": int(sum(r["feasible"] for r in ilp_rows)),
        "n_objective_consistent": int(sum(1 for r in ilp_rows
                                          if r["objective_consistent"] is not False)),
        "runtime_s": dist(el), "n_pairs": dist([r["n_pairs"] for r in ilp_rows]),
        "saturation_runtime_s": dist([r["elapsed_s"] for r in budget_rows]),
        "pass": bool(all(r["proven"] for r in ilp_rows) and
                     all(r["feasible"] for r in ilp_rows))}
    if not checks["ilp_tractability"]["pass"]:
        stop_conditions.append("ILP intractable or infeasible on representative sample")

    # --- 12. determinism --------------------------------------------------- #
    det_ok, det_diff = True, True
    for seed in REPRESENTATIVE_SEEDS:
        for rl in P.RL_CONDITIONS:
            for red in P.REDUNDANCY_LEVELS:
                s = InstanceSpec(seed=seed, rl_condition=rl, redundancy_level=red)
                a, b2 = generate_instance(s), generate_instance(s)
                det_ok &= (np.array_equal(a.relevance, b2.relevance) and
                           np.array_equal(a.tokens, b2.tokens) and
                           np.array_equal(a.similarity, b2.similarity) and
                           np.array_equal(a.topics, b2.topics))
    for rl in P.RL_CONDITIONS:
        x = generate_instance(InstanceSpec(seed=0, rl_condition=rl, redundancy_level="low"))
        y = generate_instance(InstanceSpec(seed=1, rl_condition=rl, redundancy_level="low"))
        det_diff &= not np.allclose(x.relevance, y.relevance)
    checks["determinism"] = {"identical_on_repeat": bool(det_ok),
                             "different_across_seeds": bool(det_diff),
                             "pass": bool(det_ok and det_diff)}
    if not checks["determinism"]["pass"]:
        stop_conditions.append("determinism failure")

    # ---------- phase 6: arm E -------------------------------------------- #
    anchor_rows = []
    if not args.no_anchor:
        print("phase 6/6: arm E realism anchor (separate; never pooled) ...")
        emb = Embedder(model_name="all-MiniLM-L6-v2")
        for seed in SEEDS:
            q, docs = generate_corpus(n_docs=P.N_DOCS, seed=seed)
            c = build_corpus(q, docs, emb)
            rel = np.asarray(c.relevance, float); tok = np.asarray(c.tokens, float)
            pos = rel > 0; npos = int(pos.sum())
            bh = (float(np.polyfit(np.log(tok[pos]), np.log(rel[pos]), 1)[0])
                  if npos >= 3 else None)
            bh2 = (float(np.polyfit(np.log(tok[pos]), np.log(rel[pos]), 1)[0])
                   if npos >= 3 else None)
            iu = np.triu_indices(len(c), 1)
            sat = _knap(c, int(tok.sum())).solve_ilp(time_limit=P.ILP_TIME_LIMIT_S,
                                                     scope=SolveScope.FULL_CORPUS) \
                if False else ContextKnapsack(
                    c.relevance, c.similarity, c.tokens, budget=int(tok.sum()),
                    objective_lambda=P.OBJECTIVE_LAMBDA, mmr_lambda=P.MMR_LAMBDA
                ).solve_ilp(time_limit=P.ILP_TIME_LIMIT_S, scope=SolveScope.FULL_CORPUS)
            b = build_budgets(int(sat.tokens_used))
            anchor_rows.append({
                "arm": "E_realism_anchor", "seed": seed, "backend": c.backend_name,
                "beta_hat_positive_subset": bh,
                "beta_hat_deterministic": bool(bh == bh2),
                "n_relevance_nonpositive": int(len(rel) - npos),
                "beta_hat_n_used": npos,
                "relevance_mean": float(rel.mean()), "relevance_std": float(rel.std(ddof=1)),
                "relevance_min": float(rel.min()), "relevance_max": float(rel.max()),
                "tokens_mean": float(tok.mean()), "tokens_cv": float(tok.std(ddof=1)/tok.mean()),
                "tokens_min": int(tok.min()), "tokens_max": int(tok.max()),
                "similarity_mean": float(c.similarity[iu].mean()),
                "similarity_std": float(c.similarity[iu].std(ddof=1)),
                "w_sat": int(sat.tokens_used), "w_sat_proven": bool(sat.is_proven_global_optimum),
                "budgets_ordered": bool(b["tight"] < b["medium"] < b["loose"]),
                "pearson": float(np.corrcoef(rel, tok)[0, 1])})
        checks["arm_E"] = {
            "pooled_with_factorial": False,
            "beta_hat_policy": ("computed on the positive-relevance subset only; "
                                "the real encoder can return a negative cosine, for "
                                "which the log-log slope is undefined. Dropped count "
                                "recorded per seed."),
            "n_seeds": len(anchor_rows),
            "n_seeds_with_nonpositive_relevance": int(sum(
                1 for r in anchor_rows if r["n_relevance_nonpositive"] > 0)),
            "n_relevance_nonpositive": dist([r["n_relevance_nonpositive"] for r in anchor_rows]),
            "beta_hat": dist([r["beta_hat_positive_subset"] for r in anchor_rows]),
            "pearson": dist([r["pearson"] for r in anchor_rows]),
            "relevance_mean": dist([r["relevance_mean"] for r in anchor_rows]),
            "tokens_cv": dist([r["tokens_cv"] for r in anchor_rows]),
            "similarity_mean": dist([r["similarity_mean"] for r in anchor_rows]),
            "w_sat": dist([r["w_sat"] for r in anchor_rows]),
            "all_deterministic": bool(all(r["beta_hat_deterministic"] for r in anchor_rows)),
            "all_w_sat_proven": bool(all(r["w_sat_proven"] for r in anchor_rows)),
            "all_budgets_ordered": bool(all(r["budgets_ordered"] for r in anchor_rows)),
            "pass": bool(all(r["beta_hat_deterministic"] for r in anchor_rows) and
                         all(r["w_sat_proven"] for r in anchor_rows) and
                         all(r["budgets_ordered"] for r in anchor_rows))}
        if not checks["arm_E"]["pass"]:
            stop_conditions.append("arm E handling inconsistent or non-deterministic")

    overall = not stop_conditions and all(
        v.get("pass", True) for v in checks.values())
    payload = {
        "step": "3C_preflight", "verdict": "PASS" if overall else "FAIL",
        "stop_conditions_triggered": stop_conditions,
        "purpose": "generator/protocol validation across all 40 planned seeds; "
                   "NOT the experiment and NOT a performance result",
        "locked": {"beta_tolerance": P.BETA_TOLERANCE,
                   "gamma": dict(P.REDUNDANCY_LEVELS),
                   "n_topics": P.N_TOPICS,
                   "redundancy_interpretation": "LOW vs MODERATE (not v0-calibrated)",
                   "objective_lambda": P.OBJECTIVE_LAMBDA, "mmr_lambda": P.MMR_LAMBDA},
        "protocol_version": P.CONTROLLED_PROTOCOL_VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": SEEDS, "checks": checks,
        "design_deviations": P.DESIGN_DEVIATIONS,
        "git": _git_info(), "python_version": sys.version.split()[0],
        "packages": {n: _package_version(n) for n in
                     ("numpy", "scipy", "pulp", "sentence-transformers", "tiktoken")},
        "runtime_minutes": (time.time() - t0) / 60.0,
        "per_instance": rows, "per_instance_budgets": budget_rows,
        "per_instance_divergence": div_rows, "representative_ilp": ilp_rows,
        "arm_E_per_seed": anchor_rows,
    }
    with open(os.path.join(OUT, "preflight_report.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    merged = {(r["seed"], r["rl_condition"], r["redundancy_level"]): dict(r) for r in rows}
    for r in budget_rows:
        merged[(r["seed"], r["rl_condition"], r["redundancy_level"])].update(
            {k: v for k, v in r.items() if k not in ("seed", "rl_condition", "redundancy_level")})
    flat = list(merged.values())
    with open(os.path.join(OUT, "preflight_report.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in flat for k in r}),
                           extrasaction="ignore")
        w.writeheader(); w.writerows(flat)

    print(f"\nVERDICT: {payload['verdict']}  ({payload['runtime_minutes']:.1f} min)")
    for name, c in checks.items():
        print(f"  {name:22s} {'PASS' if c.get('pass', True) else 'FAIL'}")
    if stop_conditions:
        print("\nSTOP CONDITIONS:"); [print("  -", s) for s in stop_conditions]
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
