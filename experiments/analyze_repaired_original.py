r"""Analysis of ``results/repaired_original/raw_results.json``.

Statistical unit
----------------
The independent unit is the **corpus (seed)**. Budgets evaluated on the same
corpus are NOT independent of one another, so every paired test is run
**within a budget across the 24 seeds** (24 independent pairs per test).
A pooled test over all seed x budget cells is also printed, explicitly flagged
as non-independent, and is never used to support a conclusion on its own.

Reported per comparison: n, mean/median/std/min/max of the paired difference,
win/loss/tie counts, Wilcoxon signed-rank p (two-sided), and two effect sizes
(Cohen's d_z and the matched-pairs rank-biserial correlation). A p-value is
never printed without its effect size and the underlying paired counts.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR = os.path.join(ROOT, "results", "repaired_original")
FIG_DIR = os.path.join(IN_DIR, "figures")

METHODS = ["top_k", "mmr", "greedy_objective", "greedy_token_aware", "ilp"]
LABELS = {
    "top_k": "Top-K (relevance only)",
    "mmr": "True MMR (max-sim)",
    "greedy_objective": "Greedy on objective",
    "greedy_token_aware": "Token-aware greedy",
    "ilp": "ILP (exact)",
}
COLORS = {"top_k": "#9aa0a6", "mmr": "#4c8dd9", "greedy_objective": "#8e6fbf",
          "greedy_token_aware": "#e0742a", "ilp": "#3f9a54"}
TOL = 1e-9

# The comparisons named in the step-2 protocol.
COMPARISONS = [
    ("greedy_token_aware", "top_k", "C1: token-aware greedy vs relevance-only Top-K"),
    ("greedy_token_aware", "mmr", "C2: token-aware greedy vs true MMR"),
    ("greedy_token_aware", "greedy_objective",
     "C2b: token-aware greedy vs same greedy WITHOUT token normalization "
     "(isolates token-awareness)"),
    ("greedy_token_aware", "ilp", "C3: token-aware greedy vs proven ILP optimum"),
    ("mmr", "ilp", "C3: true MMR vs proven ILP optimum"),
    ("greedy_objective", "ilp", "C3: greedy on objective vs proven ILP optimum"),
    ("top_k", "ilp", "C3: Top-K vs proven ILP optimum"),
]


def load() -> tuple[dict, dict, List[dict]]:
    with open(os.path.join(IN_DIR, "raw_results.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["metadata"], payload["corpus_descriptives_per_seed"], payload["rows"]


def index_rows(rows: Sequence[dict]) -> Dict[tuple, dict]:
    return {(r["seed"], r["budget"], r["method"]): r for r in rows}


def describe(values: Sequence[float]) -> dict:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {k: None for k in ("n", "mean", "median", "std", "min", "max")}
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max())}


def paired_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired comparison of ``a`` against ``b`` (difference = a - b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    wins = int((d > TOL).sum())
    losses = int((d < -TOL).sum())
    ties = int((np.abs(d) <= TOL).sum())
    out = {
        "n_pairs": int(d.size), "wins": wins, "losses": losses, "ties": ties,
        "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
        "std_diff": float(d.std(ddof=1)) if d.size > 1 else 0.0,
        "min_diff": float(d.min()), "max_diff": float(d.max()),
        "cohens_dz": None, "rank_biserial": None, "wilcoxon_p": None,
        "n_nonzero": int(wins + losses),
    }
    if out["std_diff"] > 0:
        out["cohens_dz"] = float(d.mean() / out["std_diff"])
    nonzero = d[np.abs(d) > TOL]
    if nonzero.size >= 1:
        ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1.0
        w_plus = float(ranks[nonzero > 0].sum())
        w_minus = float(ranks[nonzero < 0].sum())
        total = w_plus + w_minus
        if total > 0:
            out["rank_biserial"] = (w_plus - w_minus) / total
    if nonzero.size >= 3:
        try:
            out["wilcoxon_p"] = float(wilcoxon(d, zero_method="wilcox").pvalue)
        except ValueError:
            out["wilcoxon_p"] = None
    return out


def main() -> int:
    metadata, per_seed, rows = load()
    os.makedirs(FIG_DIR, exist_ok=True)
    idx = index_rows(rows)
    seeds = metadata["seeds"]
    budgets = metadata["budgets"]

    report: List[str] = []

    def emit(line: str = "") -> None:
        print(line)
        report.append(line)

    emit("=" * 78)
    emit("REPAIRED SOLVERS ON THE ORIGINAL BENCHMARK")
    emit("=" * 78)
    emit(f"seeds={len(seeds)}  budgets={budgets}  n_docs={metadata['n_docs']}")
    emit(f"objective_lambda={metadata['objective_lambda']} (frozen)  "
         f"mmr_lambda={metadata['mmr_lambda']} (frozen)")
    emit(f"ILP status counts across {metadata['n_instances']} instances: "
         f"{metadata['ilp_status_counts']}")
    emit(f"git={metadata['git']['describe']}  config={metadata['benchmark_config_version']}")
    emit()

    # ---- section 8: relevance vs token length (descriptive only) ---------- #
    emit("-" * 78)
    emit("RELEVANCE vs TOKEN COUNT  (descriptive; no causal claim)")
    emit("-" * 78)
    pear = [per_seed[str(s)]["pearson_r_relevance_tokens"] for s in seeds]
    spear = [per_seed[str(s)]["spearman_rho_relevance_tokens"] for s in seeds]
    pear_p = [per_seed[str(s)]["pearson_p"] for s in seeds]
    dp, ds = describe(pear), describe(spear)
    emit(f"  Pearson  r : mean={dp['mean']:+.4f}  median={dp['median']:+.4f}  "
         f"sd={dp['std']:.4f}  min={dp['min']:+.4f}  max={dp['max']:+.4f}")
    emit(f"  Spearman rho: mean={ds['mean']:+.4f}  median={ds['median']:+.4f}  "
         f"sd={ds['std']:.4f}  min={ds['min']:+.4f}  max={ds['max']:+.4f}")
    emit(f"  seeds with Pearson p < 0.05: "
         f"{sum(1 for p in pear_p if p < 0.05)}/{len(seeds)}")
    emit()

    # ---- per-budget summary ---------------------------------------------- #
    emit("-" * 78)
    emit("OBJECTIVE Score(S) BY BUDGET  (mean +/- sd over seeds)")
    emit("-" * 78)
    emit(f"  {'W_max':>6} " + " ".join(f"{LABELS[m][:18]:>20}" for m in METHODS))
    summary_rows: List[dict] = []
    for b in budgets:
        cells = []
        for m in METHODS:
            vals = [idx[(s, b, m)]["score"] for s in seeds]
            d = describe(vals)
            cells.append(f"{d['mean']:>13.3f}+-{d['std']:.3f}")
            for metric in ("score", "relevance_sum", "redundancy_total",
                           "redundancy_per_pair", "n_selected", "tokens_used",
                           "budget_utilization", "unique_coverage",
                           "optimality_gap"):
                mv = [idx[(s, b, m)][metric] for s in seeds]
                st = describe(mv)
                summary_rows.append({"budget": b, "method": m, "metric": metric,
                                     **st})
        emit(f"  {b:>6} " + " ".join(cells))
    emit()

    # ---- optimality gap --------------------------------------------------- #
    emit("-" * 78)
    emit("OPTIMALITY GAP vs PROVEN GLOBAL ILP OPTIMUM   gap=(ILP-method)/|ILP|")
    emit("-" * 78)
    valid = [(s, b) for s in seeds for b in budgets
             if idx[(s, b, "ilp")]["ilp_proven_global_optimum"]]
    emit(f"  instances with a PROVEN global optimum: {len(valid)}/"
         f"{len(seeds) * len(budgets)}")
    emit(f"  {'W_max':>6} " + " ".join(f"{LABELS[m][:18]:>20}" for m in METHODS[:-1]))
    for b in budgets:
        cells = []
        for m in METHODS[:-1]:
            g = [idx[(s, b, m)]["optimality_gap"] for s in seeds
                 if idx[(s, b, m)]["gap_is_valid"]]
            d = describe(g)
            cells.append(f"{100 * d['mean']:>12.2f}% (max {100 * d['max']:.1f}%)"
                         if d["n"] else f"{'n/a':>20}")
        emit(f"  {b:>6} " + " ".join(cells))
    emit()

    # ---- paired comparisons ----------------------------------------------- #
    emit("-" * 78)
    emit("PAIRED COMPARISONS  (unit = corpus/seed; 24 independent pairs each)")
    emit("-" * 78)
    comparison_rows: List[dict] = []
    for a, b_m, title in COMPARISONS:
        emit(f"\n{title}")
        emit(f"  {'W_max':>6} {'n':>3} {'W/L/T':>10} {'mean diff':>11} "
             f"{'median':>9} {'sd':>8} {'d_z':>7} {'r_rb':>7} {'wilcoxon p':>11}")
        for budget in budgets:
            av = [idx[(s, budget, a)]["score"] for s in seeds]
            bv = [idx[(s, budget, b_m)]["score"] for s in seeds]
            st = paired_test(av, bv)
            comparison_rows.append({"comparison": f"{a}_vs_{b_m}",
                                    "budget": budget, **st})
            dz = f"{st['cohens_dz']:+.3f}" if st["cohens_dz"] is not None else "-"
            rb = f"{st['rank_biserial']:+.3f}" if st["rank_biserial"] is not None else "-"
            pv = f"{st['wilcoxon_p']:.2e}" if st["wilcoxon_p"] is not None else "-"
            emit(f"  {budget:>6} {st['n_pairs']:>3} "
                 f"{st['wins']:>3}/{st['losses']:>2}/{st['ties']:>2} "
                 f"{st['mean_diff']:>+11.4f} {st['median_diff']:>+9.4f} "
                 f"{st['std_diff']:>8.4f} {dz:>7} {rb:>7} {pv:>11}")
        pooled_a = [idx[(s, bu, a)]["score"] for s in seeds for bu in budgets]
        pooled_b = [idx[(s, bu, b_m)]["score"] for s in seeds for bu in budgets]
        ps = paired_test(pooled_a, pooled_b)
        comparison_rows.append({"comparison": f"{a}_vs_{b_m}", "budget": "POOLED",
                                **ps})
        emit(f"  POOLED (NOT independent - budgets share a corpus; context only) "
             f"n={ps['n_pairs']} W/L/T={ps['wins']}/{ps['losses']}/{ps['ties']} "
             f"mean={ps['mean_diff']:+.4f}")

    # ---- secondary metrics ------------------------------------------------ #
    emit()
    emit("-" * 78)
    emit("SECONDARY / DESCRIPTIVE METRICS  (mean over seeds)")
    emit("-" * 78)
    for metric, fmt in (("n_selected", "{:.2f}"), ("tokens_used", "{:.0f}"),
                        ("budget_utilization", "{:.3f}"),
                        ("redundancy_total", "{:.3f}"),
                        ("redundancy_per_pair", "{:.3f}"),
                        ("unique_coverage", "{:.3f}")):
        emit(f"\n  {metric}")
        emit(f"  {'W_max':>6} " + " ".join(f"{LABELS[m][:16]:>18}" for m in METHODS))
        for b in budgets:
            cells = []
            for m in METHODS:
                d = describe([idx[(s, b, m)][metric] for s in seeds])
                cells.append(f"{fmt.format(d['mean']):>18}" if d["n"] else f"{'n/a':>18}")
            emit(f"  {b:>6} " + " ".join(cells))

    # ---- write artifacts -------------------------------------------------- #
    with open(os.path.join(IN_DIR, "summary_by_budget.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["budget", "method", "metric", "n",
                                           "mean", "median", "std", "min", "max"])
        w.writeheader()
        w.writerows(summary_rows)
    with open(os.path.join(IN_DIR, "paired_comparisons.csv"), "w", newline="",
              encoding="utf-8") as fh:
        keys = ["comparison", "budget", "n_pairs", "wins", "losses", "ties",
                "mean_diff", "median_diff", "std_diff", "min_diff", "max_diff",
                "cohens_dz", "rank_biserial", "wilcoxon_p", "n_nonzero"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(comparison_rows)
    with open(os.path.join(IN_DIR, "analysis_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")

    make_figures(idx, seeds, budgets)
    emit(f"\nartifacts written to {IN_DIR}")
    return 0


def make_figures(idx, seeds, budgets) -> None:
    """Five figures, each showing seed-to-seed variation, never a single seed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing - skipping figures")
        return

    def series(metric, method):
        mean, lo, hi = [], [], []
        for b in budgets:
            v = np.asarray([x for x in (idx[(s, b, method)][metric] for s in seeds)
                            if x is not None], dtype=float)
            if v.size == 0:
                mean.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            m = v.mean()
            # 95% CI of the mean over seeds (t approx with 1.96 for n=24).
            half = 1.96 * v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
            mean.append(m); lo.append(m - half); hi.append(m + half)
        return np.array(mean), np.array(lo), np.array(hi)

    panels = [
        ("score", "Objective $Score(S)$ vs token budget", METHODS, "fig1_objective_vs_budget.png"),
        ("optimality_gap", "Optimality gap vs proven ILP optimum", METHODS[:-1], "fig2_optimality_gap.png"),
        ("budget_utilization", "Token utilization (tokens used / $W_{max}$)", METHODS, "fig3_token_utilization.png"),
        ("redundancy_per_pair", "Redundancy per selected pair", METHODS, "fig4_redundancy_per_pair.png"),
        ("n_selected", "Number of documents selected", METHODS, "fig5_n_selected.png"),
    ]
    for metric, title, methods, fname in panels:
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in methods:
            mean, lo, hi = series(metric, m)
            ax.plot(budgets, mean, marker="o", label=LABELS[m], color=COLORS[m],
                    linewidth=1.8, markersize=4)
            ax.fill_between(budgets, lo, hi, color=COLORS[m], alpha=0.15, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_xlabel("Token budget $W_{max}$")
        ax.set_title(f"{title}\nmean over {len(seeds)} seeds, shaded = 95% CI of the mean",
                     fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, fname), dpi=160)
        plt.close(fig)
    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    raise SystemExit(main())
