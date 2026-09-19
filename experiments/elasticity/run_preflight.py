r"""Step 6B preflight for the elasticity experiment. NOT the experiment.

Phase 1  confirmatory seeds 0-59, GENERATOR ONLY (outcome-blind, IMPL-4/5):
         A beta, B marginals, C token-CV, D usable docs, E disagreement, F redundancy
Phase 2  confirmatory seeds, unconstrained SATURATION ILP only: G1-G3
Phase 3  pilot seeds 1000-1004, full pipeline: G4-G5, H1-H4
Phase 4  determinism re-run: I1
Phase 5  analysis STRUCTURE dry run on pilot rows (no inference): J
Phase 6  TOST margin m (written once) and power: K
Z        upstream hashes and results/controlled/* integrity

Every invariant reports pass/fail. Any failure -> verdict FAIL, STOP.
"""

from __future__ import annotations

import csv, glob, hashlib, itertools, json, os, sys, time
from typing import Any, Dict, List

import numpy as np
from scipy.stats import ks_2samp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from benchmark import _git_info, _package_version
from experiments.elasticity import protocol as EP
from experiments.elasticity.analysis import CONFIRMATORY_FAMILY, build_inputs
# Historical 6.0 runner only: the TOST helpers were retired at 6.1 and live in retired_tost.
from experiments.elasticity.retired_tost import (compute_margin, n_required, tost_power,
                                                 write_margin_once)
from experiments.elasticity.generator import (ElasticitySpec, budgets_for, evaluate_instance,
                                              generate, ranking_disagreement, saturation)

OUT = os.path.join(ROOT, "results", "elasticity")


def sha(path): return hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest()


def dist(v):
    a = np.asarray(v, float)
    return {"n": int(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()),
            **{f"q{int(q*100):02d}": float(np.quantile(a, q)) for q in EP.QUANTILES}}


def main() -> int:
    # HISTORICAL (6.0-preflight). Its checks (T5, K1-K3, 10-test family) are retired at
    # 6.1-amended; results/elasticity/preflight_report.json is the 6.0 record.
    if EP.ELASTICITY_PROTOCOL_VERSION != "6.0-preflight":
        raise SystemExit("run_preflight.py is the historical 6.0 preflight; protocol is "
                         f"{EP.ELASTICITY_PROTOCOL_VERSION}. Not re-runnable by design.")
    os.makedirs(OUT, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)
    t0 = time.time()
    inv: Dict[str, Dict[str, Any]] = {}

    def record(iid, ok, **detail):
        inv[iid] = {"pass": bool(ok), **detail}
        print(f"  [{'PASS' if ok else 'FAIL'}] {iid}", flush=True)

    # ------------------------------------------------------------- Z1 / Z2 pre
    up = {p: sha(p)[:12] for p in EP.UPSTREAM_SHA256_PREFIX}
    record("Z1", up == EP.UPSTREAM_SHA256_PREFIX, observed=up, expected=EP.UPSTREAM_SHA256_PREFIX)
    ctrl_files = sorted(glob.glob(os.path.join(ROOT, "results/controlled/*.*")))
    ctrl_before = {os.path.relpath(f, ROOT): hashlib.sha256(open(f, "rb").read()).hexdigest()
                   for f in ctrl_files}

    # ------------------------------------------------------------- A1 / F1 / C1
    record("A1", EP.BETA_LEVELS == (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
           and all(abs(EP.RHO_C[b] - b / 2.5) < 1e-12 for b in EP.BETA_LEVELS), rho_c=EP.RHO_C)
    record("F1", EP.REDUNDANCY_LEVELS == {"low": 0.15, "high": 0.70}, gamma=EP.REDUNDANCY_LEVELS)
    record("C1", EP.MIN_TOKEN_CV == 0.30, threshold=EP.MIN_TOKEN_CV)

    # ======================================================= PHASE 1: generator
    print("phase 1: generator-only, confirmatory seeds 0-59 (outcome-blind)")
    gen: Dict[tuple, Any] = {}
    for s in EP.CONFIRMATORY_SEEDS:
        for b in EP.BETA_LEVELS:
            for red in EP.REDUNDANCY_LEVELS:
                gen[(s, b, red)] = generate(ElasticitySpec(s, b, red))

    geom_ok = all(i.diagnostics["geometry"]["ok"] for i in gen.values())

    # --- C: token-CV exclusion at SEED level (w is identical across beta/gamma, IMPL-1)
    seed_cv = {s: gen[(s, EP.BETA_LEVELS[0], "low")].diagnostics["tokens_cv"] for s in EP.CONFIRMATORY_SEEDS}
    w_shared = all(np.array_equal(gen[(s, b, red)].tokens, gen[(s, -0.5, "low")].tokens)
                   for s in EP.CONFIRMATORY_SEEDS for b in EP.BETA_LEVELS for red in EP.REDUNDANCY_LEVELS)
    rejected = sorted(s for s, cv in seed_cv.items() if cv < EP.MIN_TOKEN_CV)
    accepted = [s for s in EP.CONFIRMATORY_SEEDS if s not in rejected]
    n_inst, n_rej_inst = len(gen), len(rejected) * len(EP.BETA_LEVELS) * len(EP.REDUNDANCY_LEVELS)
    per_level_rej = {b: sum(1 for s in rejected for red in EP.REDUNDANCY_LEVELS) for b in EP.BETA_LEVELS}
    record("C2", n_rej_inst / n_inst <= EP.MAX_TOKEN_CV_REJECTION_RATE,
           rejected_seeds=rejected, rejected_instances=n_rej_inst, total_instances=n_inst,
           rate=n_rej_inst / n_inst, token_cv=dist(list(seed_cv.values())))
    record("C3", w_shared and len(set(per_level_rej.values())) == 1,
           per_level_rejections=per_level_rej, tokens_identical_across_beta_and_gamma=w_shared)
    record("C4", set(accepted) | set(rejected) == set(EP.CONFIRMATORY_SEEDS)
           and not set(accepted) & set(rejected) and len(accepted) == 60 - len(rejected),
           accepted_n=len(accepted), policy="record and exclude at seed level, never resample")

    acc = {k: v for k, v in gen.items() if k[0] in accepted}
    low = lambda b: [acc[(s, b, "low")] for s in accepted]

    # --- A2 / A3
    beta_tab = {}
    for b in EP.BETA_LEVELS:
        bh = [i.diagnostics["beta_hat"] for i in low(b)]
        beta_tab[str(b)] = {**dist(bh), "target": b, "abs_mean_dev": abs(np.mean(bh) - b),
                            "n_rejected": per_level_rej[b] // 2, "values": bh}
    record("A2", all(v["abs_mean_dev"] <= EP.BETA_TOLERANCE for v in beta_tab.values()),
           tolerance=EP.BETA_TOLERANCE, by_level={k: {x: y for x, y in v.items() if x != "values"}
                                                  for k, v in beta_tab.items()})
    frac_gt1 = float(np.mean([i.diagnostics["beta_hat"] > 1 for i in low(2.0)]))
    record("A3", frac_gt1 >= EP.BETA_GT1_MIN_FRACTION, fraction_beta_hat_gt_1_at_beta_2=frac_gt1,
           fraction_gt1_by_level={str(b): float(np.mean([i.diagnostics["beta_hat"] > 1 for i in low(b)]))
                                  for b in EP.BETA_LEVELS})

    # --- B marginals
    pool_r = {b: np.concatenate([i.relevance for i in low(b)]) for b in EP.BETA_LEVELS}
    pool_w = {b: np.concatenate([i.tokens.astype(float) for i in low(b)]) for b in EP.BETA_LEVELS}
    pairs = list(itertools.combinations(EP.BETA_LEVELS, 2))
    ks_r = {f"{a}|{c}": float(ks_2samp(pool_r[a], pool_r[c]).pvalue) for a, c in pairs}
    ks_w = {f"{a}|{c}": float(ks_2samp(pool_w[a], pool_w[c]).pvalue) for a, c in pairs}
    record("B1", min(ks_r.values()) > EP.KS_MIN_P, min_p=min(ks_r.values()), ks_p=ks_r)
    record("B2", min(ks_w.values()) > EP.KS_MIN_P, min_p=min(ks_w.values()), ks_p=ks_w)
    qr = {b: np.quantile(np.log(pool_r[b]), EP.QUANTILES) for b in EP.BETA_LEVELS}
    qw = {b: np.quantile(np.log(pool_w[b]), EP.QUANTILES) for b in EP.BETA_LEVELS}
    maxdr = max(float(np.abs(qr[a] - qr[c]).max()) for a, c in pairs)
    maxdw = max(float(np.abs(qw[a] - qw[c]).max()) for a, c in pairs)
    record("B3", maxdr <= EP.QUANTILE_TOL_FRACTION_OF_SIGMA * EP.SIGMA_R
           and maxdw <= EP.QUANTILE_TOL_FRACTION_OF_SIGMA * EP.SIGMA_W,
           max_logr_quantile_diff=maxdr, tol_logr=0.25 * EP.SIGMA_R,
           max_logw_quantile_diff=maxdw, tol_logw=0.25 * EP.SIGMA_W,
           summaries={"r": {str(b): dist(pool_r[b]) for b in EP.BETA_LEVELS},
                      "log_r": {str(b): dist(np.log(pool_r[b])) for b in EP.BETA_LEVELS},
                      "w": {str(b): dist(pool_w[b]) for b in EP.BETA_LEVELS},
                      "log_w": {str(b): dist(np.log(pool_w[b])) for b in EP.BETA_LEVELS}})

    # --- D usable docs
    usable = {b: [i.diagnostics["n_usable"] for i in low(b)] for b in EP.BETA_LEVELS}
    all_usable = [x for v in usable.values() for x in v]
    lvl_means = {str(b): float(np.mean(v)) for b, v in usable.items()}
    record("D1", min(all_usable) >= EP.MIN_USABLE_DOCS, min_usable=min(all_usable),
           n_instances_below_min=int(sum(1 for x in all_usable if x < EP.MIN_USABLE_DOCS)),
           usable=dist(all_usable),
           below_threshold=dist([i.diagnostics["n_below_threshold"] for b in EP.BETA_LEVELS for i in low(b)]))
    record("D2", max(lvl_means.values()) - min(lvl_means.values()) <= EP.MAX_USABLE_COUNT_LEVEL_DIFF,
           level_means=lvl_means, max_diff=max(lvl_means.values()) - min(lvl_means.values()))

    # --- E disagreement
    Dv = {b: [i.diagnostics["D"] for i in low(b)] for b in EP.BETA_LEVELS}
    allD = [x for v in Dv.values() for x in v]
    record("E1", all(np.isfinite(allD)) and min(allD) >= 0 and max(allD) <= 2, D=dist(allD),
           expected_qualitative_range=EP.D_EXPECTED_RANGE,
           fraction_in_expected_range=float(np.mean([EP.D_EXPECTED_RANGE[0] <= x <= EP.D_EXPECTED_RANGE[1] for x in allD])))
    record("E2", all(np.std(v, ddof=1) > EP.MIN_WITHIN_LEVEL_SD_D for v in Dv.values()),
           by_level={str(b): dist(v) for b, v in Dv.items()})
    record("E3", all(ranking_disagreement(i.relevance, i.tokens) == i.diagnostics["D"] for i in acc.values()),
           n_checked=len(acc))

    # --- F redundancy
    f2 = all(np.array_equal(acc[(s, b, "low")].relevance, acc[(s, b, "high")].relevance)
             and np.array_equal(acc[(s, b, "low")].tokens, acc[(s, b, "high")].tokens)
             and np.array_equal(acc[(s, b, "low")].topics, acc[(s, b, "high")].topics)
             and acc[(s, b, "low")].diagnostics["beta_hat"] == acc[(s, b, "high")].diagnostics["beta_hat"]
             and acc[(s, b, "low")].diagnostics["D"] == acc[(s, b, "high")].diagnostics["D"]
             for s in accepted for b in EP.BETA_LEVELS)
    record("F2", f2, n_pairs=len(accepted) * 6)
    sim_lo = [acc[(s, b, "low")].diagnostics["similarity_mean"] for s in accepted for b in EP.BETA_LEVELS]
    sim_hi = [acc[(s, b, "high")].diagnostics["similarity_mean"] for s in accepted for b in EP.BETA_LEVELS]
    diff = np.asarray(sim_hi) - np.asarray(sim_lo)
    record("F3", bool((diff > 0).all()), low=dist(sim_lo), moderate=dist(sim_hi), paired_diff=dist(diff),
           n_positive=int((diff > 0).sum()), n=int(diff.size),
           ranges_overlap=bool(max(sim_lo) >= min(sim_hi)),
           by_level={str(b): {"low": float(np.mean([acc[(s, b, 'low')].diagnostics['similarity_mean'] for s in accepted])),
                              "moderate": float(np.mean([acc[(s, b, 'high')].diagnostics['similarity_mean'] for s in accepted]))}
                     for b in EP.BETA_LEVELS})

    # ======================================================= PHASE 2: saturation
    print(f"phase 2: saturation ILP on {len(acc)} accepted confirmatory instances (outcome-blind)")
    sat_rows = []
    for k, i in acc.items():
        t = time.perf_counter()
        sat = saturation(i)
        bud = budgets_for(sat["w_sat"])
        sat_rows.append({"seed": k[0], "beta": k[1], "redundancy": k[2], **sat, **{f"budget_{n}": v for n, v in bud.items()},
                         "ordered": bud["tight"] < bud["medium"] < bud["loose"],
                         "max_rho_dev": max(abs(bud[n] / sat["w_sat"] - EP.BUDGET_LEVELS[n]) for n in bud),
                         "min_token": int(i.tokens.min()), "tight_affords_one": bud["tight"] >= int(i.tokens.min()),
                         "runtime_s": time.perf_counter() - t})
    record("G1", all(r["w_sat_proven"] for r in sat_rows), n=len(sat_rows),
           n_not_proven=sum(not r["w_sat_proven"] for r in sat_rows), w_sat=dist([r["w_sat"] for r in sat_rows]),
           w_sat_n_selected=dist([r["w_sat_n_selected"] for r in sat_rows]),
           w_sat_by_beta={str(b): float(np.mean([r["w_sat"] for r in sat_rows if r["beta"] == b])) for b in EP.BETA_LEVELS},
           runtime_s=dist([r["runtime_s"] for r in sat_rows]))
    record("G2", all(r["ordered"] and r["max_rho_dev"] <= EP.RHO_BUDGET_TOLERANCE for r in sat_rows),
           n_ordered=sum(r["ordered"] for r in sat_rows), max_rho_dev=max(r["max_rho_dev"] for r in sat_rows))
    record("G3", all(r["tight_affords_one"] for r in sat_rows),
           n_tight_cannot_afford=sum(not r["tight_affords_one"] for r in sat_rows),
           tight_budget=dist([r["budget_tight"] for r in sat_rows]))

    # ======================================================= PHASE 3: pilot
    print("phase 3: pilot seeds 1000-1004, full pipeline")
    pilot_cv = {s: generate(ElasticitySpec(s, 0.0, "low")).diagnostics["tokens_cv"] for s in EP.PILOT_SEEDS}
    pilot_seeds = [s for s in EP.PILOT_SEEDS if pilot_cv[s] >= EP.MIN_TOKEN_CV]
    pilot_rows: List[Dict[str, Any]] = []
    for s in pilot_seeds:
        for b in EP.BETA_LEVELS:
            for red in EP.REDUNDANCY_LEVELS:
                pilot_rows.extend(evaluate_instance(generate(ElasticitySpec(s, b, red))))
    by = {(r["instance_id"], r["budget_level"], r["method"]): r for r in pilot_rows}
    inst_ids = sorted({r["instance_id"] for r in pilot_rows})

    g4_fail = []
    for iid in inst_ids:
        for m in ("greedy_objective", "greedy_token_aware"):
            if by[(iid, "tight", m)]["n_selected"] < EP.MIN_SELECTED_AT_TIGHT:
                g4_fail.append((iid, m, by[(iid, "tight", m)]["n_selected"]))
    n_sel_tight = {m: dist([by[(i, "tight", m)]["n_selected"] for i in inst_ids])
                   for m in ("greedy_objective", "greedy_token_aware", "ilp")}
    g4_by_beta = {str(b): {m: int(sum(1 for (iid, mm, _) in g4_fail if mm == m and f"b{b:+.1f}" in iid))
                           for m in ("greedy_objective", "greedy_token_aware")} for b in EP.BETA_LEVELS}
    record("G4", not g4_fail, n_violations=len(g4_fail), violations=g4_fail[:40],
           violations_by_beta=g4_by_beta, n_selected_at_tight=n_sel_tight)

    g5 = {}
    for lvl in EP.BUDGET_LEVELS:
        same = [len({tuple(sorted(by[(i, lvl, m)]["selected_indices"]))
                     for m in ("top_k", "mmr", "greedy_objective", "greedy_token_aware")}) == 1 for i in inst_ids]
        div = [sorted(by[(i, lvl, "greedy_objective")]["selected_indices"])
               != sorted(by[(i, lvl, "greedy_token_aware")]["selected_indices"]) for i in inst_ids]
        g5[lvl] = {"fraction_all_four_identical": float(np.mean(same)),
                   "greedy_pair_divergence_rate": float(np.mean(div)),
                   "divergence_by_beta": {str(b): float(np.mean([d for i, d in zip(inst_ids, div) if f"b{b:+.1f}" in i]))
                                          for b in EP.BETA_LEVELS}}
    record("G5", all(v["fraction_all_four_identical"] < 1.0 for v in g5.values()), by_budget=g5)

    ilp = [r for r in pilot_rows if r["method"] == "ilp"]
    heur = [r for r in pilot_rows if r["method"] != "ilp"]
    rt = [r["runtime_ms"] / 1000 for r in ilp]
    record("H1", all(r["ilp_proven_optimal"] for r in ilp), n=len(ilp),
           n_proven=sum(r["ilp_proven_optimal"] for r in ilp))
    record("H2", float(np.quantile(rt, 0.95)) < EP.ILP_RUNTIME_Q95_MAX_S, runtime_s=dist(rt))
    record("H3", not any(r["is_proven_global_optimum"] for r in heur), n_heuristic_rows=len(heur))
    ident = max(abs((by[(i, l, "greedy_token_aware")]["objective_score"] - by[(i, l, "greedy_objective")]["objective_score"])
                    - (by[(i, l, "greedy_objective")]["optimality_gap_absolute"] - by[(i, l, "greedy_token_aware")]["optimality_gap_absolute"]))
                for i in inst_ids for l in EP.BUDGET_LEVELS)
    min_gap = min(r["optimality_gap_absolute"] for r in heur)
    record("H4", all(r["ilp_objective_consistent"] is not False for r in ilp) and min_gap >= -1e-6 and ident <= 1e-9,
           n_objective_inconsistent=sum(r["ilp_objective_consistent"] is False for r in ilp),
           min_heuristic_gap=min_gap, delta_gap_identity_max_err=ident)

    pilot_beta = {str(b): float(np.mean([generate(ElasticitySpec(s, b, "low")).diagnostics["beta_hat"]
                                         for s in pilot_seeds])) for b in EP.BETA_LEVELS}

    # ======================================================= PHASE 4: determinism
    print("phase 4: determinism re-run")
    probes = [ElasticitySpec(pilot_seeds[0], -0.5, "low"), ElasticitySpec(pilot_seeds[1], 1.0, "high"),
              ElasticitySpec(pilot_seeds[-1], 2.0, "low")]
    strip = lambda rows: [{k: v for k, v in r.items() if k != "runtime_ms"} for r in rows]
    det = []
    for sp in probes:
        a, b2 = generate(sp), generate(sp)
        arrays = all(np.array_equal(getattr(a, x), getattr(b2, x))
                     for x in ("relevance", "tokens", "similarity", "embeddings", "topics"))
        ra = strip(evaluate_instance(a)); rb = strip(evaluate_instance(b2))
        det.append({"instance": sp.instance_id, "arrays_identical": arrays,
                    "rows_identical": ra == rb,
                    "matches_phase3": ra == strip([r for r in pilot_rows if r["instance_id"] == sp.instance_id])})
    record("I1", all(d["arrays_identical"] and d["rows_identical"] and d["matches_phase3"] for d in det),
           probes=det, compared="corpus, relevance, embeddings, similarity, topics, budgets, "
                                "objective values, selections, ILP result")

    # ======================================================= PHASE 5: structure
    print("phase 5: T1-T10 structure dry run on pilot rows (NO inference)")
    inputs = build_inputs(pilot_rows)
    struct = {tid: {"budget": e["test"]["budget_level"], "beta_levels": e["test"]["beta_levels"],
                    "statistic": e["test"]["statistic"], "n_seeds": e["n_seeds"],
                    "n_raw_rows": len(e["raw_row_keys"]),
                    "expected_raw_rows": len(pilot_seeds) * len(e["test"]["beta_levels"]) * 2 * 2,
                    "methods": sorted({k[2] for k in e["raw_row_keys"]}),
                    "budgets": sorted({k[1] for k in e["raw_row_keys"]})}
              for tid, e in inputs.items()}
    record("J1", [t.test_id for t in CONFIRMATORY_FAMILY] == [f"T{i}" for i in range(1, 11)]
           and all(t.family == "F1" for t in CONFIRMATORY_FAMILY))
    record("J2", all(len(t.beta_levels) in (1, 6) for t in CONFIRMATORY_FAMILY))
    record("J3", all(v["methods"] == ["greedy_objective", "greedy_token_aware"] for v in struct.values()))
    record("J4", all(v["n_seeds"] == len(pilot_seeds) for v in struct.values()))
    record("J5", all(v["n_raw_rows"] == v["expected_raw_rows"] and v["budgets"] == [v["budget"]]
                     for v in struct.values()), structure=struct)

    # ======================================================= PHASE 6: margin
    print("phase 6: TOST margin (written once) and power")
    ta_gaps = [by[(i, "tight", "greedy_token_aware")]["optimality_gap_absolute"] for i in inst_ids]
    mrec = compute_margin(ta_gaps)
    wrote = write_margin_once(os.path.join(OUT, "tost_margin.json"),
                              {"margin": mrec["m"], "rule": mrec, "pilot_seeds": pilot_seeds,
                               "protocol_version": EP.ELASTICITY_PROTOCOL_VERSION})
    record("K1", True, margin_record=mrec, write=wrote)
    record("K2", mrec["m"] > 0, m=mrec["m"])
    t5 = inputs["T5"]["values"][2.0]
    sd_t5 = float(np.std(t5, ddof=1))
    n_eff = len(accepted)
    power = tost_power(mrec["m"], sd_t5, n_eff)
    need = n_required(mrec["m"], sd_t5)
    decision = ("T5_ADEQUATELY_POWERED" if power >= EP.TOST_TARGET_POWER else "PENDING_USER_DECISION")
    record("K3", True, sd_t5_unit=sd_t5, n_pilot_seeds=len(pilot_seeds), n_planned=n_eff,
           power_at_planned_n=power, n_required_for_0_80=need, decision=decision,
           options=None if decision.startswith("T5_ADEQ") else
           [f"raise confirmatory seeds from {n_eff} to {need} before the main run",
            f"retain {n_eff} seeds and pre-declare T5 as underpowered / potentially inconclusive"],
           sensitivity={f"sd={x}": {"power": tost_power(mrec["m"], x, n_eff), "n_required": n_required(mrec["m"], x)}
                        for x in (0.5 * sd_t5, sd_t5, 2 * sd_t5)})

    # ------------------------------------------------------------- Z2 post
    ctrl_after = {os.path.relpath(f, ROOT): hashlib.sha256(open(f, "rb").read()).hexdigest()
                  for f in sorted(glob.glob(os.path.join(ROOT, "results/controlled/*.*")))}
    record("Z2", ctrl_before == ctrl_after, n_files=len(ctrl_after))
    inv["geometry_all_instances"] = {"pass": geom_ok}

    locked_ids = [x["id"] for x in EP.INVARIANTS]
    missing = [x for x in locked_ids if x not in inv]
    failed = [k for k, v in inv.items() if not v["pass"]]
    verdict = "PASS" if not failed and not missing else "FAIL"

    report = {"step": "6B_preflight", "verdict": verdict, "failed_invariants": failed,
              "unevaluated_invariants": missing,
              "protocol_version": EP.ELASTICITY_PROTOCOL_VERSION,
              "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "runtime_minutes": (time.time() - t0) / 60, "invariants": inv,
              "pilot_seeds_used": pilot_seeds, "pilot_token_cv": pilot_cv,
              "pilot_beta_hat_means_descriptive_only": pilot_beta,
              "implementation_decisions": EP.IMPLEMENTATION_DECISIONS,
              "git": _git_info(), "python_version": sys.version.split()[0],
              "packages": {n: _package_version(n) for n in ("numpy", "scipy", "pulp", "statsmodels")}}
    json.dump(report, open(os.path.join(OUT, "preflight_report.json"), "w"), indent=2, default=str)
    with open(os.path.join(OUT, "preflight_generator_instances.csv"), "w", newline="") as fh:
        rows = [{"seed": k[0], "beta_target": k[1], "redundancy": k[2], "accepted": k[0] in accepted,
                 **{x: y for x, y in v.diagnostics.items() if x != "geometry"}} for k, v in gen.items()]
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "preflight_saturation.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sat_rows[0])); w.writeheader(); w.writerows(sat_rows)
    json.dump(pilot_rows, open(os.path.join(OUT, "pilot_rows.json"), "w"), indent=1, default=str)
    json.dump({"protocol_version": EP.ELASTICITY_PROTOCOL_VERSION,
               "status": "PREFLIGHT_PASS_PENDING_SIGNOFF" if verdict == "PASS" else "PREFLIGHT_FAIL_STOP",
               "constants": {k: getattr(EP, k) for k in dir(EP) if k.isupper() and k not in ("INVARIANTS", "IMPLEMENTATION_DECISIONS")},
               "invariants": EP.INVARIANTS, "implementation_decisions": EP.IMPLEMENTATION_DECISIONS,
               "upstream_sha256_prefix": up, "tost_margin": mrec["m"], "tost_decision": decision,
               "rejected_confirmatory_seeds": rejected},
              open(os.path.join(OUT, "protocol_lock.json"), "w"), indent=2, default=str)

    print(f"\nVERDICT: {verdict}   ({report['runtime_minutes']:.1f} min)")
    if failed: print("FAILED:", failed)
    if missing: print("UNEVALUATED:", missing)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
