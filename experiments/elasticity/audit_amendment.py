r"""Audit of the elasticity protocol amendments. Generates NO experimental data.
Writes results/elasticity/amendment_<version>_audit.json and REFUSES to overwrite an existing record."""
from __future__ import annotations
import ast, csv, glob, hashlib, inspect, json, os, re, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.elasticity import protocol as EP
import experiments.elasticity.analysis as AN

def sha(p): return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()
A = {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

# 1. protocol identity
A["protocol"] = {"version": EP.ELASTICITY_PROTOCOL_VERSION, "previous": EP.PREVIOUS_PROTOCOL_VERSION,
                 "protocol_py_sha256": sha("experiments/elasticity/protocol.py"),
                 "analysis_py_sha256": sha("experiments/elasticity/analysis.py"),
                 "amendment_6_1": EP.AMENDMENT_6_1, "declarations_6_1": EP.DECLARATIONS_6_1,
                 "t8_spec": EP.T8_SPEC, "seed_count_rule": EP.SEED_COUNT_RULE,
                 "invariant_status": {i["id"]: i["status"] for i in EP.INVARIANTS}}

# 2. frozen factors vs the recorded 6.0 lock
old = json.load(open(os.path.join(ROOT, "results/elasticity/protocol_lock.json")))["constants"]
cur = json.loads(json.dumps({k: getattr(EP, k) for k in dir(EP) if k.isupper()
                             and k not in ("INVARIANTS", "IMPLEMENTATION_DECISIONS")}, default=str))
changed = sorted(k for k in old if k in cur and old[k] != cur[k])
removed = sorted(k for k in old if k not in cur)
added = sorted(k for k in cur if k not in old)
EXPECTED_CHANGED = ["CONFIRMATORY_FAMILY_SIZE", "ELASTICITY_PROTOCOL_VERSION"]
EXPECTED_REMOVED = sorted(EP.RETIRED_CONSTANTS)
retired_values_preserved = all(old[k] == EP.RETIRED_CONSTANTS[k] for k in EP.RETIRED_CONSTANTS)
A["frozen_factors"] = {"changed": changed, "removed": removed, "added": added,
                       "unchanged_count": sum(1 for k in old if k in cur and old[k] == cur[k]),
                       "only_expected_changes": changed == EXPECTED_CHANGED and removed == EXPECTED_REMOVED,
                       "retired_values_preserved_exactly": retired_values_preserved,
                       "key_frozen_values": {k: cur[k] for k in (
                           "BETA_LEVELS", "RHO_C", "SIGMA_R", "SIGMA_W", "MU_R", "MU_W", "R_CLIP", "TOKEN_CLIP",
                           "N_DOCS", "EMBED_DIM", "N_TOPICS", "REDUNDANCY_LEVELS", "BUDGET_LEVELS",
                           "OBJECTIVE_LAMBDA", "MMR_LAMBDA", "MIN_TOKEN_CV", "BETA_TOLERANCE",
                           "CONFIRMATORY_SEEDS", "PILOT_SEEDS", "POLY_LINEAR", "POLY_QUADRATIC")}}

# 3. code integrity
code = {p: sha(p)[:12] for p in ("optimizer.py", "benchmark.py", "embedder.py",
        "experiments/controlled/generator.py", "experiments/controlled/protocol.py",
        "experiments/elasticity/generator.py")}
expected = {**EP.UPSTREAM_SHA256_PREFIX, "embedder.py": "c22a456f2f46",
            "experiments/elasticity/generator.py": "351949f310ba"}
A["code_integrity"] = {"observed": code, "expected": expected,
                       "all_match": all(code[k] == expected[k] for k in code)}

# 4. Step 4 / Step 5 artifacts
ctrl = sorted(glob.glob(os.path.join(ROOT, "results/controlled/*.*")))
latest = max(os.path.getmtime(f) for f in ctrl)
fe = json.load(open(os.path.join(ROOT, "results/controlled/full_experiment.json")))
rows = fe["rows"]; ilp = [r for r in rows if r["method"] == "ilp"]
A["step4_step5_artifacts"] = {
    "n_files": len(ctrl),
    "latest_mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(latest)),
    "no_file_modified_after_step5": latest < time.mktime((2026, 9, 15, 23, 59, 59, 0, 0, -1)),
    "full_experiment_rows": len(rows), "ilp_proven": sum(bool(r["ilp_proven_optimal"]) for r in ilp),
    "ilp_rows": len(ilp),
    "sha256_baseline_recorded_now": {os.path.relpath(f, ROOT): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in ctrl},
    "note": "No byte-level hash baseline for results/controlled was persisted before 6.1 (the 6.0 preflight "
            "compared hashes in memory and stored only the file count). Verification is by mtime plus "
            "content QC; hashes are recorded now as the baseline for all later steps."}

# 5. confirmatory seeds 0-59 untouched (no compared-method outcome for any of them)
outcome_hits, scanned = [], []
for f in sorted(glob.glob(os.path.join(ROOT, "results/elasticity/*"))):
    rel = os.path.relpath(f, ROOT); scanned.append(rel)
    text = open(f, encoding="utf-8", errors="ignore").read()
    for m in re.finditer(r"EL-s(\d+)-", text):
        if int(m.group(1)) < 1000:
            outcome_hits.append((rel, "instance_id", int(m.group(1))))
    if f.endswith(".csv"):
        with open(f) as fh:
            hdr = next(csv.reader(fh))
        bad = [h for h in hdr if h in ("objective_score", "selected_indices", "method", "optimality_gap_absolute")]
        if bad:
            outcome_hits.append((rel, "outcome_columns", bad))
pilot_rows = json.load(open(os.path.join(ROOT, "results/elasticity/pilot_rows.json")))
A["confirmatory_seeds_untouched"] = {
    "files_scanned": scanned, "outcome_hits_for_seeds_below_1000": outcome_hits,
    "pilot_row_seeds": sorted({r["seed"] for r in pilot_rows}),
    "variance_pilot_seeds_already_used": any(2000 <= r["seed"] < 2040 for r in pilot_rows),
    "pass": not outcome_hits and set(r["seed"] for r in pilot_rows) <= set(EP.PILOT_SEEDS)}

# 6. retired artifact
art = "results/elasticity/tost_margin.json"
side = json.load(open(os.path.join(ROOT, EP.RETIRED_ARTIFACTS[art]["sidecar"])))
try:
    AN.assert_artifact_usable(art); blocked = False
except AN.RetiredArtifactError:
    blocked = True
# Static imports vs textual references vs runtime dependencies (6.2 audit fix).
static_imports, textual_refs = [], []
for f in sorted(glob.glob(os.path.join(ROOT, "**/*.py"), recursive=True)):
    if "venv" in f or f.endswith("retired_tost.py") or f.endswith("audit_amendment.py"):
        continue
    src = open(f, errors="ignore").read()
    if "retired_tost" not in src:
        continue
    imported = any((isinstance(n, ast.ImportFrom) and n.module and "retired_tost" in n.module)
                   or (isinstance(n, ast.Import) and any("retired_tost" in x.name for x in n.names))
                   for n in ast.walk(ast.parse(src)))
    (static_imports if imported else textual_refs).append(os.path.relpath(f, ROOT))
probe = subprocess.run([sys.executable, "-c",
    "import sys; sys.path.insert(0, '.'); import experiments.elasticity.protocol, "
    "experiments.elasticity.generator, experiments.elasticity.analysis; "
    "print('experiments.elasticity.retired_tost' in sys.modules)"], cwd=ROOT, capture_output=True, text=True)
runtime_loaded = probe.stdout.strip() == "True"
legacy = subprocess.run([sys.executable, "experiments/elasticity/run_preflight.py"], cwd=ROOT,
                        capture_output=True, text=True)
legacy_refuses = legacy.returncode != 0 and "historical 6.0 preflight" in (legacy.stdout + legacy.stderr)
LEGACY = {"experiments/elasticity/run_preflight.py"}
active_deps = [m for m in static_imports if m not in LEGACY]
A["retired_tost"] = {"artifact_sha256": sha(art), "sidecar_sha256": side["sha256_at_retirement"],
                     "preserved_byte_identical": sha(art) == side["sha256_at_retirement"],
                     "analysis_refuses_artifact": blocked,
                     "analysis_has_no_tost_functions": not any(hasattr(AN, n) for n in ("compute_margin", "tost_power", "n_required", "write_margin_once")),
                     "no_top_level_TOST_constants": not any(k.startswith("TOST") for k in dir(EP)),
                     "static_imports_of_retired_tost": static_imports,
                     "textual_references_only": textual_refs,
                     "legacy_6_0_executables": sorted(LEGACY & set(static_imports)),
                     "legacy_runner_refuses_under_current_protocol": legacy_refuses,
                     "active_analysis_runtime_loads_retired_tost": runtime_loaded,
                     "active_analysis_dependencies_on_retired_tost": active_deps}

# 7. confirmatory definitions
A["confirmatory_family"] = [t.__dict__ for t in AN.CONFIRMATORY_FAMILY]
A["exploratory_only"] = list(EP.EXPLORATORY_ONLY)

# 8. test suites
def run(t):
    out = subprocess.run([sys.executable, t], cwd=ROOT, capture_output=True, text=True)
    return {"exit": out.returncode, "pass": out.stdout.count("\nPASS ") + out.stdout.startswith("PASS "),
            "tail": out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-200:]}
A["tests"] = {t: run(t) for t in ("tests/test_optimizer.py", "tests/test_controlled_generator.py",
                                   "tests/test_elasticity.py", "tests/test_seed_count.py")}

# --- 6.2 seed-count rule checks (owner items 1-9) --------------------------- #
import experiments.elasticity.seed_count as SC
R = EP.SEED_COUNT_RULE
FRIEDMAN_FROZEN = ("f = (delta_b / sqrt(sigma2_b)) * sqrt(1/(2k)); lambda(n) = ARE*n*k*f^2/(1 - rho_lvl_b); "
                   "power(n) = P[ncchi2(k-1, lambda(n)) > chi2_{1-alpha; k-1}]; "
                   "n = min{n in 1..300 : power(n) >= 0.80}; W_min = ARE*k*f^2/((k-1)(1 - rho_lvl_b)); "
                   "b = tight for T1 and loose for T9")
fr_src = inspect.getsource(SC.n_friedman) + inspect.getsource(SC.friedman_power)
sc_tree = ast.parse(inspect.getsource(SC))
def names_in(fn_name):
    fn = next(n for n in sc_tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    return ({n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
            | {x.arg for x in fn.args.args})
pilot_names = {"PilotQuantities", "pilot", "pilot_quantities", "VARIANCE_PILOT_SEEDS", "sigma_S", "rho_lvl", "rho_red"}
fixed_fns = [n.name for n in sc_tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("fixed_")]
syn_f = SC.FixedQuantities(delta={b: 0.05 for b in EP.BUDGET_LEVELS}, delta_D=0.2, E_Q=1.5)
syn_p = SC.PilotQuantities(pi={b: 0.5 for b in EP.BUDGET_LEVELS}, M2={b: 0.04 for b in EP.BUDGET_LEVELS},
                           rho_red={b: 0.5 for b in EP.BUDGET_LEVELS}, rho_lvl={b: 0.3 for b in EP.BUDGET_LEVELS}, sigma_S=0.05)
det = [SC.n_final(syn_f, syn_p) for _ in range(3)]
seed_hits_2000 = []
for f in sorted(glob.glob(os.path.join(ROOT, "results/**/*.*"), recursive=True)):
    if os.path.isdir(f) or f.endswith(".png"):
        continue
    txt = open(f, encoding="utf-8", errors="ignore").read()
    if re.search(r"EL-s20[0-3]\d-", txt) or re.search(r'"seed": 20[0-3]\d[,\n}]', txt):
        seed_hits_2000.append(os.path.relpath(f, ROOT))
outcome_files = []
for f in sorted(glob.glob(os.path.join(ROOT, "results/elasticity/*"))):
    txt = open(f, encoding="utf-8", errors="ignore").read()
    if "objective_score" in txt:
        outcome_files.append({"file": os.path.relpath(f, ROOT),
                              "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(f))),
                              "instance_seeds": sorted({int(m) for m in re.findall(r"EL-s(\d+)-", txt)})})
A["v6_2_checks"] = {
    "1_U_SC0_is_R2": R.get("regime") == "R2" and R.get("role_of_pilot") == "PARAMETER ESTIMATION ONLY",
    "2_U_SC1_frozen_friedman": R["formulas"]["T1_T9_friedman"] == FRIEDMAN_FROZEN
                               and "range(1, N_MAX + 1)" in fr_src and "ncx2.sf" in fr_src and "ARE * n * K * f ** 2 / (1.0 - rho)" in fr_src,
    "3_U_SC2a_fixed_generator_only_DeltaD": ("q90_j - q10_j" in R["fixed_quantities_from_B"]["Delta_D"]
                                             and "Generator-only" in R["fixed_quantities_from_B"]["Delta_D"]
                                             and not (names_in("fixed_delta_D") & pilot_names)),
    "3b_DeltaD_quantile_method_locked": not any(u["id"] == "U-SC2a-Q" for u in R["unresolved"]),
    "4_U_SC5a_fixed_seeds_0_59": ("0-59" in R["populations"]["B"] and "independent of n" in R["populations"]["B"]
                                  and "baseline_seeds" in names_in("fixed_opt_b")),
    "5_edge_rules_explicit": set(R["edge_case_rules"]) == set("abcdefghi") | {"general"}
                             and EP.GENERAL_EDGE_INVARIANT.startswith("No edge-case rule"),
    "5b_rule_h_operational": not any(u["id"] == "U-SC-H" for u in R["unresolved"]),
    "6_map_deterministic_given_pilot_quantities": (all(d == det[0] for d in det)
                                                   and list(inspect.signature(SC.n_final).parameters) == ["fixed", "pilot"]
                                                   and "random" not in inspect.getsource(SC)),
    "7_no_pilot_output_can_alter_definitions": (all(not (names_in(fn) & pilot_names) for fn in fixed_fns)
                                                and not ({"delta", "SESOI_FRACTION_OF_OPTIMUM", "T8_PROB_SESOI", "ALPHA", "POWER", "FixedQuantities"}
                                                         & names_in("pilot_quantities"))
                                                and "2000" not in str(R["formulas"]) and "2000" not in str(R["fixed_quantities_from_B"])),
    "8_seeds_2000_2039_untouched": not seed_hits_2000,
    "9_no_new_comparison_outcomes": (all(o["instance_seeds"] and set(o["instance_seeds"]) <= set(EP.PILOT_SEEDS) for o in outcome_files)
                                     and not A["confirmatory_seeds_untouched"]["outcome_hits_for_seeds_below_1000"]),
    "detail": {"fixed_functions": fixed_fns, "files_with_seed_2000_2039": seed_hits_2000,
               "outcome_bearing_files": outcome_files, "synthetic_determinism_probe_note":
               "n_final evaluated 3x on SYNTHETIC inputs only, to verify determinism; no pilot or experimental value"},
}
# --- lock-candidate checks ----------------------------------------------------- #
syn = [float(i) ** 1.5 for i in range(60)]; xs = sorted(syn)
pct_exact = (abs(SC.percentile_type7(syn, 1, 10) - (xs[5] + 0.9 * (xs[6] - xs[5]))) < 1e-12
             and abs(SC.percentile_type7(syn, 9, 10) - (xs[53] + 0.1 * (xs[54] - xs[53]))) < 1e-12)
sc_src = inspect.getsource(SC)
PC = EP.PERCENTILE_CONVENTION
A["v6_2_checks"]["10_percentile_method_explicit"] = (PC["N"] == 60 and "h = (N-1)*p" in PC["definition"]
    and "0-based" in PC["definition"] and "5.9" in PC["P10"] and "53.1" in PC["P90"]
    and "np.percentile" not in sc_src and "np.quantile" not in sc_src and pct_exact)
try:
    SC.assert_pilot_sufficient([], [], [])
    h_impl = False
except SC.SeedCountStop:
    h_impl = True
except Exception:
    h_impl = False
A["v6_2_checks"]["11_rule_h_operational"] = (h_impl and "A rejection does not by itself stop the pilot" in R["edge_case_rules"]["h"]
    and [c["id"] for c in EP.RULE_H_CONDITIONS] == ["H1", "H2", "H3", "H4"]
    and "accepted seeds among 2000-2039".lower() in R["populations"]["P"].lower().replace("the accepted", "accepted"))
A["v6_2_checks"]["12_all_formulas_and_edge_rules_frozen"] = (R["status"] in ("LOCK_CANDIDATE", "LOCKED") and R["unresolved"] == []
    and set(R["edge_case_rules"]) == set("abcdefghi") | {"general"}
    and set(R["formulas"]) == {"sigma2_b", "T2_T3_T4", "T6_T7_T10", "T1_T9_friedman", "T8", "n_final"})
num_key = re.compile(r'"(Delta_D|delta_D|DeltaD|OPT|OPT_b|OPT_tight|OPT_medium|OPT_loose|E_Q|n_final|sigma_S|sigma2_b|'
                     r'rho_lvl_b|rho_red_b|pi_b|M2_b|delta_tight|delta_medium|delta_loose)"\s*:\s*-?\d')
computed_hits = []
for f in sorted(glob.glob(os.path.join(ROOT, "results/**/*.*"), recursive=True)):
    if f.endswith((".json", ".csv", ".txt")):
        txt = open(f, encoding="utf-8", errors="ignore").read()
        if num_key.search(txt):
            computed_hits.append(os.path.relpath(f, ROOT))
        if f.endswith(".csv"):
            hdr = txt.splitlines()[0].split(",") if txt else []
            if set(hdr) & {"Delta_D", "OPT_b", "E_Q", "n_final", "sigma_S"}:
                computed_hits.append(os.path.relpath(f, ROOT) + " [header]")
sat_csv = os.path.join(ROOT, "results/elasticity/preflight_saturation.csv")
sat_rows = sum(1 for _ in open(sat_csv)) - 1 if os.path.exists(sat_csv) else 0
A["v6_2_checks"]["13_no_DeltaD_computed"] = not computed_hits
A["v6_2_checks"]["14_no_budget_level_reference_ILP_on_B"] = (not computed_hits
    and not A["confirmatory_seeds_untouched"]["outcome_hits_for_seeds_below_1000"])
A["v6_2_checks"]["15_no_pilot_variance_power_n_via_frozen_F"] = (not computed_hits and not seed_hits_2000)
A["v6_2_checks"]["detail"]["numeric_value_scan_hits"] = computed_hits
A["v6_2_checks"]["detail"]["historical_disclosures"] = [
    {"what": "unconstrained SATURATION ILP (W_sat) on seeds 0-59", "when": "6B preflight, 2026-09-16",
     "artifact": "results/elasticity/preflight_saturation.csv", "rows": sat_rows,
     "why_not_a_violation": "outcome-blind budget construction, a different quantity from the budget-level "
                            "reference optimum OPT_b; no OPT_b, delta_b or Delta_D was computed"},
    {"what": "per-instance D values and beta_hat on seeds 0-59 (generator diagnostics)", "when": "6B preflight",
     "artifact": "results/elasticity/preflight_generator_instances.csv",
     "why_not_a_violation": "the Delta_D STATISTIC was never computed from them"},
    {"what": "variance components and illustrative required-n on the EARLIER pilot seeds 1000-1004",
     "when": "6B.1 design-repair analysis", "artifact": "results/elasticity/design_repair_analysis.json",
     "why_not_a_violation": "not the variance-pilot population P (2000-2039), not the frozen function F, and "
                            "prior to R2; no quantity from it enters F"},
]
A["v6_2_checks"]["remaining_unresolved"] = [u["id"] for u in R["unresolved"]]
A["v6_2_checks"]["protocol_lockable"] = (all(v for k, v in A["v6_2_checks"].items() if k[0].isdigit())
    and not R["unresolved"] and A["code_integrity"]["all_match"] and A["frozen_factors"]["only_expected_changes"]
    and A["retired_tost"]["preserved_byte_identical"] and all(v["exit"] == 0 for v in A["tests"].values()))

A["verdict"] = {
    "frozen_factors_ok": A["frozen_factors"]["only_expected_changes"] and retired_values_preserved,
    "code_integrity_ok": A["code_integrity"]["all_match"],
    "step4_artifacts_ok": A["step4_step5_artifacts"]["no_file_modified_after_step5"]
                          and A["step4_step5_artifacts"]["full_experiment_rows"] == 5400
                          and A["step4_step5_artifacts"]["ilp_proven"] == 1080,
    "confirmatory_seeds_untouched": A["confirmatory_seeds_untouched"]["pass"],
    "retired_artifact_ok": A["retired_tost"]["preserved_byte_identical"] and blocked
                           and not runtime_loaded and not active_deps and legacy_refuses,
    "tests_ok": all(v["exit"] == 0 for v in A["tests"].values()),
    "seed_count_rule_locked": EP.SEED_COUNT_RULE["status"] == "LOCKED",
    "variance_pilot_permitted": EP.SEED_COUNT_RULE["status"] == "LOCKED",
}
A["verdict"]["seed_count_rule_unresolved_items"] = [u["id"] for u in EP.SEED_COUNT_RULE.get("unresolved", [])]
A["reproducibility_limitations"] = EP.REPRODUCIBILITY_LIMITATIONS
A["git_status"] = {"available": False, "reason": "Xcode licence not accepted (RL-1)"}
out = os.path.join(ROOT, "results/elasticity/amendment_" + EP.ELASTICITY_PROTOCOL_VERSION.replace(".", "_").replace("-", "_") + "_audit.json")
if os.path.exists(out):
    raise SystemExit(f"{os.path.relpath(out, ROOT)} exists; refusing to overwrite an audit record")
json.dump(A, open(out, "w"), indent=2, default=str)
print("written", os.path.relpath(out, ROOT))
print(json.dumps({k: A[k] for k in ("verdict",)}, indent=1))
