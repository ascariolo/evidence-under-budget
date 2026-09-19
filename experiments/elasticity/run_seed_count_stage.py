r"""6.2-locked: fixed baseline quantities -> blinded variance pilot -> frozen seed-count -> n_final. HARD STOP.

Uses experiments/elasticity/{protocol,generator,seed_count}.py UNMODIFIED. Every artifact is write-once.
Prints only non-directional quantities. Never runs heuristics on seeds 0-59, never runs top_k/mmr, never
computes a confirmatory statistic, never runs the confirmatory experiment.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.elasticity import protocol as EP
from experiments.elasticity import seed_count as SC
from experiments.elasticity.generator import ElasticitySpec, budgets_for, generate, knapsack, saturation
from optimizer import SolveScope

OUT = os.path.join(ROOT, "results/elasticity")
EXPECTED = {"experiments/elasticity/protocol.py": "0fe3412c890697bddcb8009620387fbd58d3d8c433e4fbd72e2dbd02fee6c22e",
            "experiments/elasticity/seed_count.py": "2bbc392f6b4ec4d6d1c0e9c37a3b3c4062e43ed9a9d6bab5af704b605a01e8cd",
            "experiments/elasticity/generator.py": "351949f310ba73e4f5e9d07b97abd7788cc65347ab614149600f723f7f1b5019",
            "optimizer.py": "facfb186ddccc831786a25463177755db6e06ad801d4a01d4140df9441c97b07"}
B = list(range(60))
P_PLANNED = list(EP.VARIANCE_PILOT_SEEDS)
BUD = list(EP.BUDGET_LEVELS)
REDS = sorted(EP.REDUNDANCY_LEVELS)


def sha(p): return hashlib.sha256(open(os.path.join(ROOT, p) if not os.path.isabs(p) else p, "rb").read()).hexdigest()


def write_once(name, obj):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        raise SystemExit(f"{name} exists; refusing to overwrite")
    json.dump(obj, open(path, "w"), indent=2, default=str)
    return sha(path)


def stop(stage, reason):
    write_once("stage_STOP.json", {"stage": stage, "reason": str(reason),
                                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"\n*** STOP at {stage}: {reason}")
    raise SystemExit(2)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    t0 = time.time()
    for f, h in EXPECTED.items():
        if sha(f) != h:
            stop("pre-check", f"{f} hash changed: locked protocol not intact")
    if EP.ELASTICITY_PROTOCOL_VERSION != "6.2-locked" or EP.SEED_COUNT_RULE["status"] != "LOCKED":
        stop("pre-check", "protocol not 6.2-locked")
    EP.assert_seed_count_rule_locked()

    # ============================================================ STAGE 1
    print("STAGE 1: baseline reference ILPs (B = seeds 0-59)")
    sat_rows = {}
    with open(os.path.join(OUT, "preflight_saturation.csv")) as fh:
        for r in csv.DictReader(fh):
            sat_rows[(int(r["seed"]), float(r["beta"]), r["redundancy"])] = r
    if len(sat_rows) != 720:
        stop("stage1", f"6B saturation rows {len(sat_rows)} != 720")
    reuse_checks = {"w_sat_proven_all": True, "token_mass_matches": True, "budgets_reproduce": True}
    D_B, cv_B, recs = {}, {}, []
    for s in B:
        for b in EP.BETA_LEVELS:
            for g in REDS:
                inst = generate(ElasticitySpec(s, b, g))
                row = sat_rows[(s, b, g)]
                if row["w_sat_proven"] != "True":
                    reuse_checks["w_sat_proven_all"] = False
                if int(inst.tokens.sum()) != int(row["corpus_token_mass"]):
                    reuse_checks["token_mass_matches"] = False
                bud = budgets_for(int(row["w_sat"]))
                if any(bud[k] != int(row[f"budget_{k}"]) for k in BUD):
                    reuse_checks["budgets_reproduce"] = False
                if g == REDS[0]:
                    D_B[(s, b)] = inst.diagnostics["D"]
                    cv_B[s] = inst.diagnostics["tokens_cv"]
                elif inst.diagnostics["D"] != D_B[(s, b)]:
                    stop("stage1", f"D differs across redundancy for {(s, b)}")
                for k in BUD:
                    t = time.perf_counter()
                    res = knapsack(inst, bud[k]).solve_ilp(time_limit=EP.ILP_TIME_LIMIT_S, scope=SolveScope.FULL_CORPUS)
                    recs.append({"seed": s, "beta_target": b, "redundancy_level": g, "budget_level": k,
                                 "w_sat_reused": int(row["w_sat"]), "w_max": bud[k],
                                 "objective_score": float(res.score), "ilp_proven_optimal": bool(res.is_proven_global_optimum),
                                 "status": res.status.value, "objective_consistent": res.meta.get("objective_consistent"),
                                 "runtime_s": time.perf_counter() - t})
        if s % 10 == 9:
            print(f"  seeds 0..{s} done ({(time.time()-t0)/60:.1f} min)")
    if not all(reuse_checks.values()):
        stop("stage1", f"6B reuse verification failed: {reuse_checks}")
    if any(r["objective_consistent"] is False for r in recs):
        stop("stage1", "reference ILP objective inconsistent (general invariant)")
    rejected_B = [s for s in B if cv_B[s] < EP.MIN_TOKEN_CV]
    if rejected_B:
        stop("stage1", f"baseline seeds rejected by token-CV: {rejected_B} (B must be the fixed 60)")
    try:
        opt = SC.fixed_opt_b(recs, B)
    except SC.SeedCountStop as e:
        stop("stage1", e)
    s1 = write_once("stage1_reference_ilp_B.json", {
        "protocol": "6.2-locked", "population": "B = seeds 0-59", "n_instances_per_budget": 720,
        "reuse": {"artifact": "results/elasticity/preflight_saturation.csv",
                  "sha256": sha("results/elasticity/preflight_saturation.csv"),
                  "reused": "W_sat and budgets", "verification": reuse_checks},
        "reference_solver_only": True, "heuristics_run": False,
        "n_solves": len(recs), "n_proven": sum(r["ilp_proven_optimal"] for r in recs),
        "runtime_s": {"mean": float(np.mean([r["runtime_s"] for r in recs])), "max": float(max(r["runtime_s"] for r in recs))},
        "OPT_b": {k: opt[k]["OPT"] for k in BUD}, "delta_b": {k: opt[k]["delta"] for k in BUD},
        "records": recs})
    print(f"  reuse verification: {reuse_checks}")
    print(f"  {len(recs)} reference ILPs, proven {sum(r['ilp_proven_optimal'] for r in recs)}/{len(recs)}")

    # ============================================================ STAGE 2
    print("STAGE 2: generator-only fixed quantities on B")
    csv_D = {}
    with open(os.path.join(OUT, "preflight_generator_instances.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["redundancy"] == REDS[0] and int(r["seed"]) in B:
                csv_D[(int(r["seed"]), float(r["beta_target"]))] = float(r["D"])
    d_crosscheck = all(csv_D[k] == v for k, v in D_B.items()) and len(csv_D) == len(D_B) == 360
    if not d_crosscheck:
        stop("stage2", "recomputed D differs from 6B generator artifact")
    per_level = {}
    for b in EP.BETA_LEVELS:
        vals = [D_B[(s, b)] for s in B]
        p10, p90 = SC.percentile_type7(vals, 1, 10), SC.percentile_type7(vals, 9, 10)
        per_level[str(b)] = {"P10": p10, "P90": p90, "range": p90 - p10}
    try:
        delta_D = SC.fixed_delta_D(D_B, B)
        E_Q = SC.fixed_E_Q(D_B, B)
    except SC.SeedCountStop as e:
        stop("stage2", e)
    if abs(delta_D - float(np.mean([v["range"] for v in per_level.values()]))) > 1e-15:
        stop("stage2", "Delta_D internal consistency failure")
    s2 = write_once("stage2_generator_fixed_B.json", {
        "protocol": "6.2-locked", "population": "B = seeds 0-59", "generator_only": True, "outcome_blind": True,
        "percentile_convention": EP.PERCENTILE_CONVENTION["definition"],
        "D_crosscheck_vs_6B_csv_bit_equal": d_crosscheck,
        "per_beta_level": per_level, "Delta_D": delta_D, "E_Q": E_Q,
        "b_min_T8": SC.T8_PROB_SESOI / delta_D})

    # ============================================================ STAGE 3
    fixed = SC.FixedQuantities(delta={k: opt[k]["delta"] for k in BUD}, delta_D=delta_D, E_Q=E_Q)
    s3 = write_once("stage3_fixed_quantities_FROZEN.json", {
        "protocol": "6.2-locked", "frozen_before_pilot_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "OPT_b": {k: opt[k]["OPT"] for k in BUD}, "delta_b": dict(fixed.delta),
        "Delta_D": fixed.delta_D, "E_Q": fixed.E_Q,
        "source_hashes": {"stage1_reference_ilp_B.json": s1, "stage2_generator_fixed_B.json": s2},
        "immutable": "these values are inputs to the frozen n_final and are not changed by any pilot result"})
    print(f"STAGE 3: fixed quantities frozen (sha256 {s3[:16]}) BEFORE any pilot seed is generated")

    # ============================================================ STAGE 4
    print("STAGE 4: blinded variance pilot, seeds 2000-2039")
    cv_P = {s: generate(ElasticitySpec(s, 0.0, REDS[0])).diagnostics["tokens_cv"] for s in P_PLANNED}
    accepted = [s for s in P_PLANNED if cv_P[s] >= EP.MIN_TOKEN_CV]
    rejected = [s for s in P_PLANNED if cv_P[s] < EP.MIN_TOKEN_CV]
    x, grid, t8, raw, sat_qc = {}, [], [], [], {"n": 0, "proven": 0}
    for s in accepted:
        w0 = None
        for b in EP.BETA_LEVELS:
            for g in REDS:
                inst = generate(ElasticitySpec(s, b, g))
                if w0 is None:
                    w0 = inst.tokens.copy()
                elif not np.array_equal(w0, inst.tokens):
                    stop("stage4", "token vector differs across beta/gamma within a seed (IMPL-1)")
                sat = saturation(inst)
                sat_qc["n"] += 1; sat_qc["proven"] += int(sat["w_sat_proven"])
                if not sat["w_sat_proven"]:
                    stop("stage4", f"unproven saturation ILP for pilot instance {(s, b, g)}")
                bud = budgets_for(sat["w_sat"])
                for k in BUD:
                    kn = knapsack(inst, bud[k])
                    go, ta = kn.solve_greedy_objective(), kn.solve_greedy()
                    x[(s, b, g, k)] = float(ta.score - go.score)
                    grid += [(s, b, g, k, "greedy_objective"), (s, b, g, k, "greedy_token_aware")]
                    Y = int(sorted(go.indices) != sorted(ta.indices))
                    if k == "tight":
                        t8.append({"seed": s, "beta": b, "redundancy": g, "Y": Y, "D": inst.diagnostics["D"]})
                    raw.append({"seed": s, "beta": b, "redundancy": g, "budget": k, "w_sat": sat["w_sat"],
                                "w_max": bud[k], "x": x[(s, b, g, k)], "Y": Y, "D": inst.diagnostics["D"]})
    try:
        suff = SC.assert_pilot_sufficient(accepted, rejected, grid)
        pilot = SC.pilot_quantities(x, t8, accepted)
    except SC.SeedCountStop as e:
        stop("stage4", e)
    s4raw = write_once("stage4_variance_pilot_SEALED_raw.json", {
        "SEALED": "raw per-instance pilot values for audit reproduction only; NOT to be summarised by beta, "
                  "signed, tested or inspected for direction", "records": raw})
    s4 = write_once("stage4_variance_pilot_quantities.json", {
        "protocol": "6.2-locked", "planned_seeds": [2000, 2039],
        "accepted": accepted, "rejected": rejected,
        "rejected_token_cv": {str(s): cv_P[s] for s in rejected},
        "token_cv_threshold": EP.MIN_TOKEN_CV, "rule_h": suff,
        "qc": {"instances": len(accepted) * 12, "saturation_ilp": sat_qc, "grid_cells": len(grid),
               "grid_expected": len(accepted) * 6 * 2 * 3 * 2, "methods_run": ["greedy_objective", "greedy_token_aware"],
               "top_k_or_mmr_run": False, "budget_level_ilp_run": False},
        "pilot_quantities": {"pi_b": dict(pilot.pi), "M2_b": dict(pilot.M2), "rho_red_b": dict(pilot.rho_red),
                             "rho_lvl_b": dict(pilot.rho_lvl), "sigma_S": pilot.sigma_S},
        "sealed_raw_sha256": s4raw,
        "fixed_quantities_sha256_used": s3})
    print(f"  accepted {len(accepted)}, rejected {rejected}; saturation proven {sat_qc['proven']}/{sat_qc['n']}")

    # ============================================================ STAGE 5
    print("STAGE 5: frozen seed-count function")
    try:
        out = SC.n_final(fixed, pilot)
    except SC.SeedCountStop as e:
        stop("stage5", e)
    s5 = write_once("stage5_seed_count.json", {
        "protocol": "6.2-locked", "n_by_test": out["n_by_test"], "n_required": out["n_required"],
        "n_final": out["n_final"], "max_allowed": EP.MAX_CONFIRMATORY_SEEDS,
        "rule_le_300_passes": out["n_required"] <= EP.MAX_CONFIRMATORY_SEEDS,
        "inputs": {"fixed_sha256": s3, "pilot_sha256": s4},
        "HARD_STOP": "confirmatory experiment NOT run; seeds 0..n_final-1 NOT run as a comparison experiment",
        "runtime_minutes": (time.time() - t0) / 60})
    print(f"  n_by_test {out['n_by_test']}  n_final {out['n_final']}")
    print(f"HARD STOP. total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
