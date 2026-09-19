r"""STEP 4 analysis of results/controlled/full_experiment.json.

Pre-registered plan (docs/controlled_benchmark_design.md section 10):
 - statistical unit = the base instance (seed x condition x redundancy);
   budget is a WITHIN-instance factor, so budget contrasts are paired
 - primary: linear mixed model on Delta_gap with (1 | seed)
 - per-cell: paired Wilcoxon + Cohen's d_z + matched-pairs rank-biserial,
   BCa bootstrap CIs over seeds, Holm (confirmatory) and BH (exploratory)
 - no p-value without its effect size and paired counts

Primary estimand (protocol section 4):
    Delta_gap = gap(greedy_objective) - gap(greedy_token_aware)
              = score(greedy_token_aware) - score(greedy_objective)
in ABSOLUTE objective units (the reference can be non-positive, so no ratio).

Condition D and arm E are analysed separately and never pooled with A-C.
Quality control runs FIRST and aborts before any statistic if it fails.
"""

from __future__ import annotations

import json, os, sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from experiments.controlled import protocol as P

IN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "controlled")
FIG_DIR = os.path.join(IN_DIR, "figures_full")
TOL = 1e-9
BUDGET_ORDER = ["tight", "medium", "loose"]
METHODS = ["top_k", "mmr", "greedy_objective", "greedy_token_aware", "ilp"]


# --------------------------------------------------------------------------- #
def load():
    with open(os.path.join(IN_DIR, "full_experiment.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["manifest"], payload["rows"]


def quality_control(manifest, rows) -> Dict[str, Any]:
    """Runs BEFORE any statistic. Any failure aborts the analysis."""
    failures: List[str] = []
    syn = [r for r in rows if r["arm"] == "synthetic_factorial"]
    anchor = [r for r in rows if r["arm"] != "synthetic_factorial"]

    expected_syn = 40 * 4 * 2 * 3 * 5
    checks: Dict[str, Any] = {
        "n_rows_total": len(rows), "n_rows_synthetic": len(syn),
        "n_rows_arm_E": len(anchor),
        "expected_rows_synthetic": expected_syn,
        "expected_rows_arm_E": 40 * 3 * 5,
    }
    if len(syn) != expected_syn:
        failures.append(f"synthetic rows {len(syn)} != expected {expected_syn}")
    if len(anchor) != 40 * 3 * 5:
        failures.append(f"arm E rows {len(anchor)} != expected {40*3*5}")

    ids = {(r["instance_id"], r["budget_level"], r["method"]) for r in rows}
    checks["n_unique_instance_budget_method"] = len(ids)
    if len(ids) != len(rows):
        failures.append("duplicate (instance, budget, method) keys")

    seeds = sorted({r["seed"] for r in syn})
    checks["seeds_present"] = seeds
    if seeds != list(range(40)):
        failures.append(f"seeds present {seeds[:5]}... != 0-39")
    for key, want in (("rl_condition", {"A", "B", "C", "D"}),
                      ("redundancy_level", {"low", "high"}),
                      ("budget_level", set(BUDGET_ORDER)),
                      ("method", set(METHODS))):
        got = {r[key] for r in syn}
        checks[f"{key}_present"] = sorted(got)
        if got != want:
            failures.append(f"{key}: {sorted(got)} != {sorted(want)}")

    ilp_rows = [r for r in rows if r["method"] == "ilp"]
    n_proven = sum(bool(r["ilp_proven_optimal"]) for r in ilp_rows)
    checks["n_ilp_solves"] = len(ilp_rows)
    checks["n_ilp_proven_optimal"] = n_proven
    checks["ilp_proven_rate"] = n_proven / len(ilp_rows)
    if n_proven != len(ilp_rows):
        failures.append(f"ILP proven optimal only {n_proven}/{len(ilp_rows)}")
    inconsistent = [r["instance_id"] for r in ilp_rows
                    if r.get("ilp_objective_consistent") is False]
    checks["n_ilp_objective_inconsistent"] = len(inconsistent)
    if inconsistent:
        failures.append(f"ILP objective inconsistent on {inconsistent[:3]}")

    checks["n_missing_objective"] = sum(1 for r in rows if r["objective_score"] is None)
    if checks["n_missing_objective"]:
        failures.append("missing primary outcome")

    heur_claiming_opt = [r["instance_id"] for r in rows
                         if r["method"] != "ilp" and r["is_proven_global_optimum"]]
    checks["n_heuristics_claiming_optimal"] = len(heur_claiming_opt)
    if heur_claiming_opt:
        failures.append("a heuristic was flagged proven-optimal")

    # protocol lock vs execution metadata
    lock_ok = {
        "objective_lambda": all(r["objective_lambda"] == P.OBJECTIVE_LAMBDA for r in rows),
        "mmr_lambda": all(r["mmr_lambda"] == P.MMR_LAMBDA for r in rows),
        "n_docs": all(r["n_docs"] == P.N_DOCS for r in syn),
        "gamma": all(r["gamma"] == P.REDUNDANCY_LEVELS[r["redundancy_level"]] for r in syn),
        "rho_target": all(abs(r["rho_target"] - P.BUDGET_LEVELS[r["budget_level"]]) < 1e-12
                          for r in syn),
        "beta_target": all(r["beta_target"] == P.RL_CONDITIONS[r["rl_condition"]]["target_beta"]
                           for r in syn),
        "protocol_version": all(r["protocol_version"] == P.CONTROLLED_PROTOCOL_VERSION
                                for r in rows),
    }
    checks["protocol_lock_matches_execution"] = lock_ok
    for k, v in lock_ok.items():
        if not v:
            failures.append(f"protocol lock mismatch: {k}")

    checks["w_sat_all_proven"] = all(bool(r["w_sat_proven"]) for r in rows)
    if not checks["w_sat_all_proven"]:
        failures.append("a saturation solve was not proven optimal")

    checks["ok"] = not failures
    checks["failures"] = failures
    return checks


# --------------------------------------------------------------------------- #
def describe(v: Sequence[float]) -> Dict[str, Any]:
    a = np.asarray([x for x in v if x is not None], float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()),
            "median": float(np.median(a)),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def bca_ci(x: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
           seed: int = 12345) -> Dict[str, Optional[float]]:
    """BCa bootstrap CI of the mean, over the seed ensemble."""
    x = np.asarray(x, float)
    if x.size < 2 or np.allclose(x, x[0]):
        return {"lo": float(x.mean()) if x.size else None,
                "hi": float(x.mean()) if x.size else None, "method": "degenerate"}
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n_boot)])
    theta = x.mean()
    prop = np.mean(boots < theta)
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    from scipy.stats import norm
    z0 = norm.ppf(prop)
    jack = np.array([np.delete(x, i).mean() for i in range(x.size)])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    acc = num / den if den != 0 else 0.0
    out = {}
    for name, q in (("lo", alpha / 2), ("hi", 1 - alpha / 2)):
        zq = norm.ppf(q)
        adj = z0 + (z0 + zq) / (1 - acc * (z0 + zq))
        out[name] = float(np.quantile(boots, norm.cdf(adj)))
    out["method"] = "BCa"
    return out


def paired(a: Sequence[float], b: Sequence[float], ci: bool = True) -> Dict[str, Any]:
    """Paired comparison a - b with effect sizes and counts. Never p alone."""
    d = np.asarray(a, float) - np.asarray(b, float)
    wins = int((d > TOL).sum()); losses = int((d < -TOL).sum())
    ties = int((np.abs(d) <= TOL).sum())
    out = {"n_pairs": int(d.size), "wins": wins, "losses": losses, "ties": ties,
           "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
           "std_diff": float(d.std(ddof=1)) if d.size > 1 else 0.0,
           "min_diff": float(d.min()), "max_diff": float(d.max()),
           "cohens_dz": None, "rank_biserial": None, "wilcoxon_p": None}
    if out["std_diff"] > 0:
        out["cohens_dz"] = float(d.mean() / out["std_diff"])
    nz = d[np.abs(d) > TOL]
    if nz.size:
        ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
        wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
        if wp + wm > 0:
            out["rank_biserial"] = float((wp - wm) / (wp + wm))
    if nz.size >= 3:
        try:
            out["wilcoxon_p"] = float(wilcoxon(d, zero_method="wilcox").pvalue)
        except ValueError:
            pass
    if ci:
        out["ci95_mean_diff"] = bca_ci(d)
    return out


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(((k, v) for k, v in pvals.items() if v is not None),
                   key=lambda kv: kv[1])
    m = len(items); out = {}; run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); out[k] = run
    return out


def bh(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(((k, v) for k, v in pvals.items() if v is not None),
                   key=lambda kv: kv[1])
    m = len(items); out = {}; prev = 1.0
    for i, (k, p) in reversed(list(enumerate(items))):
        prev = min(prev, m * p / (i + 1)); out[k] = min(1.0, prev)
    return out


# --------------------------------------------------------------------------- #
def build_frame(rows):
    """Wide per (instance, budget): one row with every method's score/gap."""
    import pandas as pd
    recs: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        key = (r["instance_id"], r["budget_level"])
        rec = recs.setdefault(key, {
            k: r[k] for k in ("arm", "analysis_group", "instance_id", "seed",
                              "rl_condition", "redundancy_level", "gamma",
                              "beta_target", "beta_hat", "budget_level",
                              "rho_target", "w_max", "w_sat",
                              "pearson_r_relevance_tokens",
                              "instance_similarity_mean")})
        m = r["method"]
        rec[f"score_{m}"] = r["objective_score"]
        rec[f"gap_{m}"] = r["optimality_gap_absolute"]
        rec[f"n_selected_{m}"] = r["n_selected"]
        rec[f"tokens_used_{m}"] = r["tokens_used"]
        rec[f"redundancy_per_pair_{m}"] = r["redundancy_per_pair"]
        rec[f"tokens_of_selected_mean_{m}"] = r["tokens_of_selected_mean"]
        rec[f"mupt_{m}"] = r["marginal_utility_per_token_mean"]
        rec[f"relsum_{m}"] = r["relevance_sum"]
        if m == "ilp":
            rec["ilp_optimum"] = r["objective_score"]
            rec["ilp_proven"] = r["ilp_proven_optimal"]
            rec["ilp_runtime_ms"] = r["runtime_ms"]
    df = pd.DataFrame(list(recs.values()))
    # Primary estimand, protocol section 4.
    df["delta_gap"] = df["gap_greedy_objective"] - df["gap_greedy_token_aware"]
    df["delta_n_selected"] = df["n_selected_greedy_token_aware"] - df["n_selected_greedy_objective"]
    df["delta_tokens_used"] = df["tokens_used_greedy_token_aware"] - df["tokens_used_greedy_objective"]
    df["delta_relsum"] = df["relsum_greedy_token_aware"] - df["relsum_greedy_objective"]
    df["delta_redundancy_per_pair"] = (df["redundancy_per_pair_greedy_token_aware"]
                                       - df["redundancy_per_pair_greedy_objective"])
    df["delta_selected_len"] = (df["tokens_of_selected_mean_greedy_token_aware"]
                                - df["tokens_of_selected_mean_greedy_objective"])
    return df


def cell_table(df, groups=("rl_condition", "redundancy_level", "budget_level")):
    cells = []
    for key, sub in df.groupby(list(groups), sort=False):
        sub = sub.sort_values("seed")
        st = paired(sub["score_greedy_token_aware"].to_numpy(),
                    sub["score_greedy_objective"].to_numpy())
        cells.append({
            **dict(zip(groups, key if isinstance(key, tuple) else (key,))),
            "n": int(len(sub)),
            "mean_delta_gap": float(sub["delta_gap"].mean()),
            "median_delta_gap": float(sub["delta_gap"].median()),
            "sd_delta_gap": float(sub["delta_gap"].std(ddof=1)),
            "score_greedy_objective_mean": float(sub["score_greedy_objective"].mean()),
            "score_greedy_token_aware_mean": float(sub["score_greedy_token_aware"].mean()),
            "ilp_optimum_mean": float(sub["ilp_optimum"].mean()),
            "gap_greedy_objective_mean": float(sub["gap_greedy_objective"].mean()),
            "gap_greedy_token_aware_mean": float(sub["gap_greedy_token_aware"].mean()),
            "beta_hat_mean": float(sub["beta_hat"].mean()),
            **{k: v for k, v in st.items() if k != "ci95_mean_diff"},
            "ci95_lo": st["ci95_mean_diff"]["lo"], "ci95_hi": st["ci95_mean_diff"]["hi"],
        })
    return cells


def mixed_models(df, label: str) -> Dict[str, Any]:
    """Pre-registered mixed models with (1 | seed)."""
    import statsmodels.formula.api as smf
    out: Dict[str, Any] = {}
    d = df.copy()
    d["budget_level"] = d["budget_level"].astype(str)
    d["log_rho"] = np.log(d["rho_target"].astype(float))
    try:
        m1 = smf.mixedlm("delta_gap ~ C(budget_level, Treatment('tight'))"
                         " * C(rl_condition) * C(redundancy_level)",
                         d, groups=d["seed"]).fit()
        out["factorial_model"] = {
            "formula": "delta_gap ~ budget * rl_condition * redundancy + (1|seed)",
            "n_obs": int(m1.nobs),
            "params": {k: float(v) for k, v in m1.params.items()},
            "pvalues": {k: float(v) for k, v in m1.pvalues.items()},
            "conf_int": {k: [float(a), float(b)]
                         for k, (a, b) in m1.conf_int().iterrows()},
        }
    except Exception as exc:
        out["factorial_model"] = {"error": str(exc)}
    try:
        m2 = smf.mixedlm("delta_gap ~ log_rho * beta_hat + instance_similarity_mean",
                         d, groups=d["seed"]).fit()
        out["continuous_model"] = {
            "formula": "delta_gap ~ log(rho) * beta_hat + mean_similarity + (1|seed)",
            "n_obs": int(m2.nobs),
            "params": {k: float(v) for k, v in m2.params.items()},
            "pvalues": {k: float(v) for k, v in m2.pvalues.items()},
            "conf_int": {k: [float(a), float(b)]
                         for k, (a, b) in m2.conf_int().iterrows()},
        }
    except Exception as exc:
        out["continuous_model"] = {"error": str(exc)}
    return out


def secondary_comparisons(df) -> Dict[str, Any]:
    out = {}
    for a, b in (("greedy_token_aware", "top_k"), ("greedy_token_aware", "mmr"),
                 ("greedy_objective", "top_k"), ("greedy_objective", "mmr")):
        per_budget = {}
        for bl in BUDGET_ORDER:
            sub = df[df["budget_level"] == bl].sort_values("instance_id")
            per_budget[bl] = paired(sub[f"score_{a}"].to_numpy(),
                                    sub[f"score_{b}"].to_numpy())
        out[f"{a}_vs_{b}"] = per_budget
    gaps = {}
    for m in ("top_k", "mmr", "greedy_objective", "greedy_token_aware"):
        gaps[m] = {bl: describe(df[df["budget_level"] == bl][f"gap_{m}"].tolist())
                   for bl in BUDGET_ORDER}
    out["gap_by_method_and_budget"] = gaps
    return out


def mechanism(df) -> Dict[str, Any]:
    """Descriptive diagnostics. NOT causal mediation - the design randomizes the
    factors, not these quantities."""
    out = {"disclaimer": ("mechanism diagnostics, descriptive only; the design "
                          "randomizes the experimental factors, not these "
                          "quantities, so these are not causal mediators")}
    for bl in BUDGET_ORDER:
        sub = df[df["budget_level"] == bl]
        cors = {}
        for col in ("delta_n_selected", "delta_tokens_used", "delta_relsum",
                    "delta_redundancy_per_pair", "delta_selected_len"):
            v = sub[["delta_gap", col]].dropna()
            cors[col] = {"pearson_with_delta_gap":
                         float(np.corrcoef(v["delta_gap"], v[col])[0, 1])
                         if len(v) > 2 else None,
                         **describe(sub[col].tolist())}
        out[bl] = cors
    return out


def make_figures(df_abc, df_d, cells_abc) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    colors = {"A": "#4c8dd9", "B": "#e0742a", "C": "#3f9a54", "D": "#9b59b6"}

    # 1. Delta_gap by budget x condition, 95% CI over seeds
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, red in zip(axes, ("low", "high")):
        for cond in ("A", "B", "C"):
            xs, ys, lo, hi = [], [], [], []
            for i, bl in enumerate(BUDGET_ORDER):
                v = df_abc[(df_abc.rl_condition == cond) &
                           (df_abc.redundancy_level == red) &
                           (df_abc.budget_level == bl)]["delta_gap"].to_numpy()
                ci = bca_ci(v)
                xs.append(i); ys.append(v.mean()); lo.append(ci["lo"]); hi.append(ci["hi"])
            ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo),
                                      np.array(hi) - np.array(ys)],
                        marker="o", capsize=3, label=f"{cond}", color=colors[cond])
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(range(3)); ax.set_xticklabels(BUDGET_ORDER)
        ax.set_title(f"redundancy = {red} (gamma={P.REDUNDANCY_LEVELS[red]})", fontsize=10)
        ax.set_xlabel("budget tightness"); ax.grid(alpha=.25, lw=.6)
    axes[0].set_ylabel(r"$\Delta_{gap}$  (token-aware $-$ objective)")
    axes[0].legend(fontsize=8, frameon=False, title="condition")
    fig.suptitle("Effect of token-awareness by budget and relevance-length condition\n"
                 "mean over 40 seeds, bars = 95% BCa CI", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig1_delta_gap.png"), dpi=160)
    plt.close(fig)

    # 2. Delta_gap vs beta_hat per budget
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, bl in zip(axes, BUDGET_ORDER):
        sub = df_abc[df_abc.budget_level == bl]
        for cond in ("A", "B", "C"):
            s = sub[sub.rl_condition == cond]
            ax.scatter(s["beta_hat"], s["delta_gap"], s=10, alpha=.55,
                       color=colors[cond], label=cond)
        if len(sub) > 2:
            z = np.polyfit(sub["beta_hat"], sub["delta_gap"], 1)
            xs = np.linspace(sub["beta_hat"].min(), sub["beta_hat"].max(), 50)
            ax.plot(xs, np.polyval(z, xs), "k-", lw=1.3)
        ax.axhline(0, color="k", lw=.8, ls="--")
        ax.set_title(f"{bl} (rho={P.BUDGET_LEVELS[bl]})", fontsize=10)
        ax.set_xlabel(r"realized $\hat{\beta}$"); ax.grid(alpha=.25, lw=.6)
    axes[0].set_ylabel(r"$\Delta_{gap}$"); axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(r"$\Delta_{gap}$ against realized elasticity $\hat{\beta}$ (A/B/C only)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig2_delta_gap_vs_beta.png"), dpi=160)
    plt.close(fig)

    # 3. absolute gaps by method
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (dd, name) in zip(axes, ((df_abc, "A/B/C pooled"), (df_d, "condition D"))):
        for m, c in (("top_k", "#9aa0a6"), ("mmr", "#4c8dd9"),
                     ("greedy_objective", "#8e6fbf"), ("greedy_token_aware", "#e0742a")):
            ys = [dd[dd.budget_level == bl][f"gap_{m}"].mean() for bl in BUDGET_ORDER]
            ax.plot(range(3), ys, marker="o", label=m, color=c)
        ax.set_xticks(range(3)); ax.set_xticklabels(BUDGET_ORDER)
        ax.set_title(name, fontsize=10); ax.grid(alpha=.25, lw=.6); ax.set_xlabel("budget")
    axes[0].set_ylabel("mean absolute optimality gap"); axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Distance from the proven ILP optimum (lower is better)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig3_gap_by_method.png"), dpi=160)
    plt.close(fig)


def main() -> int:
    import pandas as pd
    manifest, rows = load()
    report: List[str] = []
    def emit(s=""):
        print(s); report.append(s)

    emit("=" * 78); emit("STEP 4 - CONTROLLED FULL EXPERIMENT: FIRST-PASS ANALYSIS")
    emit("=" * 78)
    qc = quality_control(manifest, rows)
    emit(f"QUALITY CONTROL: {'PASS' if qc['ok'] else 'FAIL'}")
    for k in ("n_rows_total", "n_rows_synthetic", "n_rows_arm_E",
              "n_ilp_solves", "n_ilp_proven_optimal", "ilp_proven_rate",
              "n_heuristics_claiming_optimal", "n_missing_objective"):
        emit(f"  {k}: {qc[k]}")
    emit(f"  protocol lock matches execution: {all(qc['protocol_lock_matches_execution'].values())}")
    if not qc["ok"]:
        emit("FAILURES: " + "; ".join(qc["failures"]))
        emit("ABORTING before any statistic, per protocol section 12.")
        json.dump({"quality_control": qc}, open(os.path.join(
            IN_DIR, "full_analysis.json"), "w"), indent=2, default=str)
        return 1

    df = build_frame(rows)
    df_abc = df[df.analysis_group == "ABC"].copy()
    df_d = df[df.analysis_group == "D_separate"].copy()
    df_e = df[df.analysis_group == "E_separate"].copy()

    emit(f"\nframe: {len(df)} instance-budget rows "
         f"(A/B/C {len(df_abc)}, D {len(df_d)}, arm E {len(df_e)})")
    emit(f"primary estimand: {manifest['primary_estimand']}")

    # ---- primary ---- #
    emit("\n" + "-" * 78)
    emit("PRIMARY: Delta_gap by budget (A/B/C pooled; D and E excluded)")
    emit("-" * 78)
    emit(f"  {'budget':<8}{'n':>5}{'mean':>10}{'median':>10}{'sd':>9}"
         f"{'W/L/T':>12}{'d_z':>8}{'r_rb':>8}{'wilcoxon p':>12}{'95% CI':>22}")
    primary = {}
    praw = {}
    for bl in BUDGET_ORDER:
        sub = df_abc[df_abc.budget_level == bl].sort_values("instance_id")
        st = paired(sub["score_greedy_token_aware"].to_numpy(),
                    sub["score_greedy_objective"].to_numpy())
        primary[bl] = st; praw[f"primary_{bl}"] = st["wilcoxon_p"]
        ci = st["ci95_mean_diff"]
        emit(f"  {bl:<8}{st['n_pairs']:>5}{st['mean_diff']:>+10.4f}"
             f"{st['median_diff']:>+10.4f}{st['std_diff']:>9.4f}"
             f"{st['wins']:>4}/{st['losses']:>3}/{st['ties']:>3}"
             f"{st['cohens_dz']:>+8.3f}{st['rank_biserial']:>+8.3f}"
             f"{st['wilcoxon_p']:>12.2e}"
             f"   [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")

    emit("\nPRIMARY: Delta_gap by beta condition x budget (A/B/C)")
    emit(f"  {'cond':<5}{'beta_t':>8}{'budget':<9}{'n':>4}{'mean':>10}{'d_z':>8}"
         f"{'W/L/T':>12}{'wilcoxon p':>12}{'95% CI':>22}")
    by_cb = {}
    for cond in ("A", "B", "C"):
        for bl in BUDGET_ORDER:
            sub = df_abc[(df_abc.rl_condition == cond) &
                         (df_abc.budget_level == bl)].sort_values("instance_id")
            st = paired(sub["score_greedy_token_aware"].to_numpy(),
                        sub["score_greedy_objective"].to_numpy())
            by_cb[f"{cond}|{bl}"] = st; praw[f"{cond}|{bl}"] = st["wilcoxon_p"]
            ci = st["ci95_mean_diff"]
            emit(f"  {cond:<5}{P.RL_CONDITIONS[cond]['target_beta']:>+8.2f}{bl:<9}"
                 f"{st['n_pairs']:>4}{st['mean_diff']:>+10.4f}{st['cohens_dz']:>+8.3f}"
                 f"{st['wins']:>4}/{st['losses']:>3}/{st['ties']:>3}"
                 f"{st['wilcoxon_p']:>12.2e}   [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")

    emit("\nPRIMARY: Delta_gap by redundancy x budget (A/B/C)")
    by_rb = {}
    for red in ("low", "high"):
        for bl in BUDGET_ORDER:
            sub = df_abc[(df_abc.redundancy_level == red) &
                         (df_abc.budget_level == bl)].sort_values("instance_id")
            st = paired(sub["score_greedy_token_aware"].to_numpy(),
                        sub["score_greedy_objective"].to_numpy())
            by_rb[f"{red}|{bl}"] = st
            emit(f"  redundancy={red:<5} {bl:<7} n={st['n_pairs']:>3} "
                 f"mean={st['mean_diff']:+.4f} d_z={st['cohens_dz']:+.3f} "
                 f"W/L/T={st['wins']}/{st['losses']}/{st['ties']} "
                 f"p={st['wilcoxon_p']:.2e}")

    # ---- multiplicity ---- #
    holm_p, bh_p = holm(praw), bh(praw)
    emit(f"\nmultiplicity over {len(praw)} confirmatory cells: "
         f"Holm-significant at 0.05: {sum(1 for v in holm_p.values() if v < .05)}; "
         f"BH-significant: {sum(1 for v in bh_p.values() if v < .05)}")

    # ---- cells ---- #
    cells_abc = cell_table(df_abc)
    cells_d = cell_table(df_d)

    # ---- models ---- #
    emit("\n" + "-" * 78); emit("MIXED MODELS (pre-registered)"); emit("-" * 78)
    models = mixed_models(df_abc, "ABC")
    for name in ("factorial_model", "continuous_model"):
        m = models[name]
        if "error" in m:
            emit(f"  {name}: ERROR {m['error']}"); continue
        emit(f"  {name}: {m['formula']}  (n={m['n_obs']})")
        for k, v in m["params"].items():
            if k in ("Group Var",):
                continue
            p = m["pvalues"].get(k)
            ci = m["conf_int"].get(k, [None, None])
            star = "" if p is None or p >= .05 else (" *" if p >= .001 else " **")
            emit(f"     {k:<58}{v:>+9.4f}  p={p:<9.2e} "
                 f"[{ci[0]:+.4f},{ci[1]:+.4f}]{star}")

    sec = secondary_comparisons(df_abc)
    mech = mechanism(df_abc)
    emit("\n" + "-" * 78); emit("MECHANISM DIAGNOSTICS (descriptive, not mediation)")
    emit("-" * 78)
    for bl in BUDGET_ORDER:
        c = mech[bl]
        emit(f"  {bl}: d n_selected={c['delta_n_selected']['mean']:+.2f} "
             f"d tokens={c['delta_tokens_used']['mean']:+.1f} "
             f"d relevance={c['delta_relsum']['mean']:+.3f} "
             f"d red/pair={c['delta_redundancy_per_pair']['mean']:+.4f} "
             f"d sel.len={c['delta_selected_len']['mean']:+.1f}")
        emit(f"       corr(Delta_gap, d n_selected)="
             f"{c['delta_n_selected']['pearson_with_delta_gap']:+.3f}  "
             f"corr(.., d relevance)={c['delta_relsum']['pearson_with_delta_gap']:+.3f}")

    # ---- D and E ---- #
    emit("\n" + "-" * 78); emit("CONDITION D (adversarial; analysed separately)")
    emit("-" * 78)
    d_res = {}
    for bl in BUDGET_ORDER:
        sub = df_d[df_d.budget_level == bl].sort_values("instance_id")
        st = paired(sub["score_greedy_token_aware"].to_numpy(),
                    sub["score_greedy_objective"].to_numpy())
        d_res[bl] = st; ci = st["ci95_mean_diff"]
        emit(f"  {bl:<8}n={st['n_pairs']:>3} mean={st['mean_diff']:+.4f} "
             f"d_z={st['cohens_dz']:+.3f} W/L/T={st['wins']}/{st['losses']}/{st['ties']} "
             f"p={st['wilcoxon_p']:.2e} CI=[{ci['lo']:+.4f},{ci['hi']:+.4f}]")

    emit("\n" + "-" * 78); emit("ARM E (realism anchor; separate, not external validity)")
    emit("-" * 78)
    e_res = {}
    for bl in BUDGET_ORDER:
        sub = df_e[df_e.budget_level == bl].sort_values("instance_id")
        st = paired(sub["score_greedy_token_aware"].to_numpy(),
                    sub["score_greedy_objective"].to_numpy())
        e_res[bl] = st; ci = st["ci95_mean_diff"]
        emit(f"  {bl:<8}n={st['n_pairs']:>3} mean={st['mean_diff']:+.4f} "
             f"d_z={st['cohens_dz']:+.3f} W/L/T={st['wins']}/{st['losses']}/{st['ties']} "
             f"p={st['wilcoxon_p']:.2e} CI=[{ci['lo']:+.4f},{ci['hi']:+.4f}]")
    e_rows = [r for r in rows if r["arm"] != "synthetic_factorial"]
    e_meta = {"n_seeds": len({r["seed"] for r in e_rows}),
              "beta_hat": describe([r["beta_hat"] for r in e_rows
                                    if r["method"] == "ilp"]),
              "n_dropped_nonpositive_total": int(sum(
                  r.get("beta_hat_n_dropped_nonpositive", 0) or 0
                  for r in e_rows if r["method"] == "ilp" and r["budget_level"] == "tight")),
              "similarity_mean": describe([r["instance_similarity_mean"] for r in e_rows
                                           if r["method"] == "ilp" and r["budget_level"] == "tight"])}
    emit(f"  beta_hat (positive subset): mean={e_meta['beta_hat']['mean']:+.4f} "
         f"sd={e_meta['beta_hat']['std']:.4f}; non-positive relevance values dropped: "
         f"{e_meta['n_dropped_nonpositive_total']}")

    make_figures(df_abc, df_d, cells_abc)

    out = {"quality_control": qc, "manifest_digest": {
               k: manifest[k] for k in ("protocol_version", "timestamp_utc",
                                        "runtime_minutes", "n_rows", "git",
                                        "primary_estimand", "protocol_lock")},
           "primary_by_budget": primary, "primary_by_condition_budget": by_cb,
           "primary_by_redundancy_budget": by_rb,
           "multiplicity": {"raw": praw, "holm": holm_p, "bh": bh_p},
           "cells_ABC": cells_abc, "cells_D": cells_d,
           "mixed_models": models, "secondary": sec, "mechanism": mech,
           "condition_D": d_res, "arm_E": e_res, "arm_E_metadata": e_meta}
    json.dump(out, open(os.path.join(IN_DIR, "full_analysis.json"), "w"),
              indent=2, default=str)
    import csv as _csv
    with open(os.path.join(IN_DIR, "full_analysis_cells.csv"), "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=sorted({k for c in cells_abc + cells_d for k in c}))
        w.writeheader(); w.writerows(cells_abc + cells_d)
    open(os.path.join(IN_DIR, "full_analysis_report.txt"), "w").write("\n".join(report))
    emit(f"\nartifacts -> {IN_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
