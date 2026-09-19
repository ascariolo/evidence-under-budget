r"""Confirmatory family for the elasticity experiment - protocol 6.1-amended.

Nine tests, Holm over 9. Structure is separated from inference:
  * build_inputs(rows)      - which rows enter each test, aggregated to the analysis unit.
                              No statistics. Safe on non-confirmatory data.
  * run_confirmatory(rows)  - statistics + Holm. Confirmatory data only.

Analysis unit (IMPL-2): per (seed, beta, budget), Delta_gap = score(token_aware) -
score(greedy_objective) per redundancy level, then the MEAN over both redundancy levels.
Zeros are retained everywhere (amendment decision 7).

TOST (former T5) is RETIRED. This module has no import path to experiments/elasticity/
retired_tost.py and refuses to read any artifact listed in protocol.RETIRED_ARTIFACTS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from experiments.elasticity import protocol as EP

PRIMARY_METHODS = ("greedy_objective", "greedy_token_aware")
ALL_LEVELS = tuple(EP.BETA_LEVELS)


class AnalysisStructureError(RuntimeError):
    """Data cannot be aggregated exactly as preregistered."""


class RetiredArtifactError(RuntimeError):
    """An analysis attempted to use a retired artifact."""


def assert_artifact_usable(path: str) -> None:
    rel = os.path.normpath(path).replace(os.sep, "/")
    for retired in EP.RETIRED_ARTIFACTS:
        if rel.endswith(retired) or rel.endswith(os.path.basename(retired)):
            raise RetiredArtifactError(f"{retired} is RETIRED at {EP.RETIRED_ARTIFACTS[retired]['retired_at']}")


@dataclass(frozen=True)
class ConfirmatoryTest:
    test_id: str
    estimand: str
    budget_level: str
    beta_levels: Tuple[float, ...]
    unit: str
    statistic: str
    h0: str
    h1: str
    sidedness: str
    status: str = "confirmatory"
    family: str = "F1"


CONFIRMATORY_FAMILY: Tuple[ConfirmatoryTest, ...] = (
    ConfirmatoryTest("T1", "profile of mean Delta_gap across the 6 beta levels", "tight", ALL_LEVELS,
                     "seed x beta, redundancy-mean Delta_gap", "friedman",
                     "the 6 level distributions are equal", "at least one differs", "two"),
    ConfirmatoryTest("T2", "mean Delta_gap at beta = +1.0", "tight", (1.0,),
                     "seed, redundancy-mean Delta_gap (zeros retained)", "signflip_mean",
                     "mean Delta_gap = 0", "mean Delta_gap != 0", "two"),
    ConfirmatoryTest("T3", "mean Delta_gap at beta = +1.5", "tight", (1.5,),
                     "seed, redundancy-mean Delta_gap (zeros retained)", "signflip_mean",
                     "mean Delta_gap = 0", "mean Delta_gap != 0", "two"),
    ConfirmatoryTest("T4", "mean Delta_gap at beta = +2.0 (E hinge)", "tight", (2.0,),
                     "seed, redundancy-mean Delta_gap (zeros retained)", "signflip_mean",
                     "mean Delta_gap = 0", "mean Delta_gap != 0", "two"),
    ConfirmatoryTest("T6", "linear orthogonal contrast over the 6 levels", "tight", ALL_LEVELS,
                     "seed contrast score over redundancy-mean Delta_gap", "signflip_contrast_linear",
                     "mean contrast score = 0", "!= 0", "two"),
    ConfirmatoryTest("T7", "quadratic orthogonal contrast over the 6 levels", "tight", ALL_LEVELS,
                     "seed contrast score over redundancy-mean Delta_gap", "signflip_contrast_quadratic",
                     "mean contrast score = 0", "!= 0", "two"),
    ConfirmatoryTest("T8", "within-level slope of P(selection SETS differ) on centred D, probability "
                     "points per unit D (R-primary; protocol.T8_SPEC)", "tight", ALL_LEVELS,
                     "instance-level binary divergence, seed as cluster",
                     "cluster_signflip_score_slope", "b = 0", "b > 0", "one_greater"),
    ConfirmatoryTest("T9", "profile of mean Delta_gap across the 6 beta levels", "loose", ALL_LEVELS,
                     "seed x beta, redundancy-mean Delta_gap", "friedman",
                     "the 6 level distributions are equal", "at least one differs", "two"),
    ConfirmatoryTest("T10", "quadratic orthogonal contrast over the 6 levels", "medium", ALL_LEVELS,
                     "seed contrast score over redundancy-mean Delta_gap", "signflip_contrast_quadratic",
                     "mean contrast score = 0", "!= 0", "two"),
)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def seed_level_table(rows: Sequence[Dict[str, Any]], budget_level: str) -> Dict[Tuple[int, float], Dict[str, Any]]:
    inst: Dict[Tuple[int, float, str], Dict[str, Any]] = {}
    for r in rows:
        if r.get("arm") != EP.ARM or r["budget_level"] != budget_level:
            continue
        if r["method"] not in PRIMARY_METHODS and r["method"] != "ilp":
            continue
        key = (int(r["seed"]), float(r["beta_target"]), r["redundancy_level"])
        rec = inst.setdefault(key, {"keys": []})
        if r["method"] == "ilp":
            rec["ilp_proven"] = bool(r["ilp_proven_optimal"])
            continue
        rec[r["method"]] = float(r["objective_score"])
        rec[r["method"] + "_set"] = tuple(sorted(int(i) for i in r["selected_indices"]))
        rec["D"] = float(r["D"])
        rec["keys"].append((r["instance_id"], r["budget_level"], r["method"]))

    table: Dict[Tuple[int, float], Dict[str, Any]] = {}
    for (seed, beta, red), rec in inst.items():
        if not all(m in rec for m in PRIMARY_METHODS):
            raise AnalysisStructureError(f"missing primary method for {seed}/{beta}/{red}")
        if not rec.get("ilp_proven", False):
            raise AnalysisStructureError(f"ILP not proven optimal for {seed}/{beta}/{red}")
        cell = table.setdefault((seed, beta), {"dgap": {}, "div": {}, "D": set(), "keys": []})
        cell["dgap"][red] = rec["greedy_token_aware"] - rec["greedy_objective"]
        cell["div"][red] = int(rec["greedy_objective_set"] != rec["greedy_token_aware_set"])
        cell["D"].add(rec["D"])
        cell["keys"].extend(rec["keys"])

    out: Dict[Tuple[int, float], Dict[str, Any]] = {}
    for (seed, beta), cell in table.items():
        if set(cell["dgap"]) != set(EP.REDUNDANCY_LEVELS):
            raise AnalysisStructureError(f"seed {seed} beta {beta}: redundancy levels {sorted(cell['dgap'])}")
        if len(cell["D"]) != 1:
            raise AnalysisStructureError(f"D differs across redundancy for {seed}/{beta}")
        reds = sorted(EP.REDUNDANCY_LEVELS)
        out[(seed, beta)] = {"delta_gap": float(np.mean([cell["dgap"][k] for k in reds])),
                             "divergence_share": float(np.mean([cell["div"][k] for k in reds])),
                             "div_by_red": dict(cell["div"]),
                             "D": next(iter(cell["D"])), "keys": sorted(cell["keys"])}
    return out


def _complete_seeds(table, levels) -> List[int]:
    return [s for s in sorted({s for s, _ in table}) if all((s, b) in table for b in levels)]


def build_inputs(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    tables = {b: seed_level_table(rows, b) for b in EP.BUDGET_LEVELS}
    out: Dict[str, Dict[str, Any]] = {}
    for t in CONFIRMATORY_FAMILY:
        tab = tables[t.budget_level]
        seeds = _complete_seeds(tab, t.beta_levels)
        entry = {"test": asdict(t), "seeds": seeds, "n_seeds": len(seeds),
                 "values": {b: [tab[(s, b)]["delta_gap"] for s in seeds] for b in t.beta_levels},
                 "raw_row_keys": sorted({k for s in seeds for b in t.beta_levels for k in tab[(s, b)]["keys"]})}
        if t.statistic == "cluster_signflip_score_slope":
            entry["instances"] = [{"seed": s, "beta": b, "redundancy": red, "Y": tab[(s, b)]["div_by_red"][red],
                                   "D": tab[(s, b)]["D"]}
                                  for s in seeds for b in t.beta_levels for red in sorted(EP.REDUNDANCY_LEVELS)]
        out[t.test_id] = entry
    return out


# --------------------------------------------------------------------------- #
# Inference - confirmatory data only
# --------------------------------------------------------------------------- #
def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, adj = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


def signflip_pvalue(x: Sequence[float], n_flips: int = EP.PERMUTATION_N_FLIPS,
                    rng_seed: int = EP.PERMUTATION_RNG_SEED, alternative: str = "two-sided") -> float:
    """Sign-flip permutation test of mean(x) = 0. Zeros are RETAINED: a zero flips to zero
    but still counts in the denominator of every permuted mean."""
    x = np.asarray(x, float)
    signs = np.random.default_rng(rng_seed).choice((-1.0, 1.0), size=(n_flips, x.size))
    perm = (signs * x).mean(axis=1)
    if alternative == "two-sided":
        hits = np.sum(np.abs(perm) >= abs(x.mean()) - 1e-15)
    elif alternative == "greater":
        hits = np.sum(perm >= x.mean() - 1e-15)
    else:
        raise ValueError(alternative)
    return float((1 + hits) / (n_flips + 1))


def t8_cluster_scores(instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """protocol.T8_SPEC: Dc centred within beta level; Y centred by its beta-level mean;
    S_s = sum Dc*(Y - Ybar_beta) per seed; b_hat = sum S / sum Dc^2 (probability points per unit D)."""
    betas = sorted({r["beta"] for r in instances})
    mD = {b: np.mean([r["D"] for r in instances if r["beta"] == b]) for b in betas}
    mY = {b: np.mean([r["Y"] for r in instances if r["beta"] == b]) for b in betas}
    S: Dict[int, float] = {}
    sdc2 = 0.0
    for r in instances:
        dc = r["D"] - mD[r["beta"]]
        S[r["seed"]] = S.get(r["seed"], 0.0) + dc * (r["Y"] - mY[r["beta"]])
        sdc2 += dc * dc
    seeds = sorted(S)
    return {"seeds": seeds, "scores": [S[s] for s in seeds],
            "b_hat": float(sum(S.values()) / sdc2) if sdc2 > 0 else float("nan")}


def run_confirmatory(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from scipy.stats import friedmanchisquare
    inputs = build_inputs(rows)
    results: Dict[str, Dict[str, Any]] = {}
    for t in CONFIRMATORY_FAMILY:
        inp = inputs[t.test_id]
        v = inp["values"]
        if t.statistic == "friedman":
            p = float(friedmanchisquare(*[v[b] for b in t.beta_levels]).pvalue)
        elif t.statistic == "signflip_mean":
            p = signflip_pvalue(v[t.beta_levels[0]])
        elif t.statistic.startswith("signflip_contrast_"):
            c = np.asarray(EP.POLY_LINEAR if t.statistic.endswith("linear") else EP.POLY_QUADRATIC, float)
            p = signflip_pvalue(np.column_stack([v[b] for b in t.beta_levels]) @ c)
        elif t.statistic == "cluster_signflip_score_slope":
            p = signflip_pvalue(t8_cluster_scores(inp["instances"])["scores"], alternative="greater")
        else:
            raise AnalysisStructureError(t.statistic)
        results[t.test_id] = {"p": p, "n_seeds": inp["n_seeds"]}
    if len(results) != EP.CONFIRMATORY_FAMILY_SIZE:
        raise AnalysisStructureError("family size mismatch")
    adj = holm({k: r["p"] for k, r in results.items()})
    for k in results:
        results[k]["p_holm"] = adj[k]
    return {"protocol_version": EP.ELASTICITY_PROTOCOL_VERSION, "family_size": len(results), "results": results}
