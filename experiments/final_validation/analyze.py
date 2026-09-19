"""Pre-specified analysis of the locked v7 experiment (protocol sections 7-8).

Confirmatory content is exactly three two-sided paired sign-flip tests of
Delta_C = C(TA) - C(GO), one per locked budget, Holm-corrected across those
three and nothing else. Everything else in this module is descriptive: arm
summaries, the full Delta_C distribution, and the exploratory D diagnostic.

No SESOI is applied, because none is specified. No threshold is searched for.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import protocol as P

RESULTS_DIR = P.REPO_ROOT / "results" / "final_validation"


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def signflip_pvalue(differences: np.ndarray, flips: int = P.PERMUTATION_FLIPS,
                    seed: int = P.PERMUTATION_SEED) -> Tuple[float, float]:
    """Two-sided paired sign-flip permutation test on the mean difference.

    Zeros are retained: a zero difference contributes nothing under any sign
    assignment, which is exactly the behaviour the protocol asks for. The
    p-value uses the (1 + #extreme) / (1 + B) convention, so it is never 0.
    """
    differences = np.asarray(differences, dtype=np.float64)
    n = differences.size
    if n == 0:
        return float("nan"), float("nan")
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(flips):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        if abs(float(np.mean(signs * differences))) >= abs(observed) - 1e-15:
            extreme += 1
    return observed, (1.0 + extreme) / (1.0 + flips)


def holm(pvalues: Sequence[float], alpha: float = P.ALPHA) -> List[Dict[str, object]]:
    """Holm step-down over exactly the p-values given; no family is implied."""
    pvalues = list(pvalues)
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running = max(running, min(1.0, value))
        adjusted[idx] = running
    return [{"p_raw": pvalues[i], "p_holm": adjusted[i], "reject": adjusted[i] < alpha}
            for i in range(m)]


def bootstrap_ci(values: np.ndarray, resamples: int = P.BOOTSTRAP_RESAMPLES,
                 seed: int = P.BOOTSTRAP_SEED, level: float = 0.95) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[draws].mean(axis=1)
    lower = float(np.percentile(means, 100 * (1 - level) / 2))
    upper = float(np.percentile(means, 100 * (1 + level) / 2))
    return lower, upper


def cohen_dz(differences: np.ndarray) -> float:
    """Paired standardized effect size: mean(d) / sd(d)."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.size < 2:
        return float("nan")
    sd = float(np.std(differences, ddof=1))
    return float("nan") if sd == 0.0 else float(np.mean(differences)) / sd


def rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, (R+ - R-) / (R+ + R-)."""
    from scipy.stats import rankdata

    differences = np.asarray(differences, dtype=np.float64)
    nonzero = differences[differences != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    total = positive + negative
    return 0.0 if total == 0 else (positive - negative) / total


def wilcoxon_p(differences: np.ndarray) -> float:
    """Secondary check only; the primary test is the sign-flip permutation."""
    from scipy.stats import wilcoxon

    differences = np.asarray(differences, dtype=np.float64)
    if differences.size == 0 or np.all(differences == 0.0):
        return float("nan")
    try:
        return float(wilcoxon(differences, alternative="two-sided",
                              zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def distribution(differences: np.ndarray) -> Dict[str, object]:
    """The full Delta_C distribution, reported instead of a SESOI verdict."""
    differences = np.asarray(differences, dtype=np.float64)
    deciles = {f"q{int(q * 100):02d}": float(np.quantile(differences, q))
               for q in np.arange(0.1, 1.0, 0.1)} if differences.size else {}
    values, counts = np.unique(differences, return_counts=True)
    return {
        "min": float(differences.min()) if differences.size else float("nan"),
        "max": float(differences.max()) if differences.size else float("nan"),
        "deciles": deciles,
        "wins_ta": int((differences > 0).sum()),
        "losses_ta": int((differences < 0).sum()),
        "ties": int((differences == 0).sum()),
        "ecdf": [{"value": float(v), "count": int(c),
                  "cumulative": float(np.cumsum(counts)[i] / differences.size)}
                 for i, (v, c) in enumerate(zip(values, counts))],
    }


def spearman_with_ci(x: np.ndarray, y: np.ndarray,
                     resamples: int = P.BOOTSTRAP_RESAMPLES,
                     seed: int = P.BOOTSTRAP_SEED) -> Dict[str, float]:
    """Exploratory association, reported with a percentile bootstrap CI."""
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if x.size < 3:
        return {"rho": float("nan"), "p": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n": int(x.size)}
    result = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, x.size, size=(resamples, x.size))
    boot = np.array([spearmanr(x[d], y[d]).statistic for d in draws])
    boot = boot[~np.isnan(boot)]
    return {
        "rho": float(result.statistic), "p": float(result.pvalue),
        "ci_low": float(np.percentile(boot, 2.5)) if boot.size else float("nan"),
        "ci_high": float(np.percentile(boot, 97.5)) if boot.size else float("nan"),
        "n": int(x.size), "bootstrap_resamples_used": int(boot.size),
    }


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def paired_deltas(rows: Sequence[dict], rho: float) -> Tuple[np.ndarray, List[str]]:
    """Delta_C = C(TA) - C(GO) per question at one budget, in sample order."""
    ta = {r["qid"]: r["coverage"] for r in rows if r["arm"] == P.ARM_TA and r["rho"] == rho}
    go = {r["qid"]: r["coverage"] for r in rows if r["arm"] == P.ARM_GO and r["rho"] == rho}
    qids = [q for q in ta if q in go]
    if len(qids) != len(ta) or len(qids) != len(go):
        raise P.ProtocolViolation(f"unpaired questions at rho={rho}")
    order = {r["qid"]: r["position"] for r in rows}
    qids.sort(key=lambda q: order[q])
    return np.array([ta[q] - go[q] for q in qids], dtype=np.float64), qids


def arm_summary(rows: Sequence[dict]) -> List[Dict[str, object]]:
    """Descriptive summary per (arm, budget). No tests are run on these."""
    out = []
    for rho in P.RHO_LEVELS:
        for arm in P.ARMS:
            subset = [r for r in rows if r["arm"] == arm and r["rho"] == rho]
            if not subset:
                continue
            cov = np.array([r["coverage"] for r in subset], dtype=np.float64)
            entry = {
                "arm": arm, "rho": rho, "n": len(subset),
                "coverage_mean": float(cov.mean()), "coverage_median": float(np.median(cov)),
                "tokens_used_mean": float(np.mean([r["tokens_used"] for r in subset])),
                "budget_utilization_mean": float(np.mean(
                    [r["tokens_used"] / r["w_max"] if r["w_max"] else np.nan for r in subset])),
                "n_selected_mean": float(np.mean([r["n_selected"] for r in subset])),
                "latency_ms_mean": float(np.mean([r["latency_ms"] for r in subset])),
                "score_mean": float(np.mean([r["score"] for r in subset])),
            }
            if arm == P.ARM_ILP:
                proven = [r for r in subset if r["status"] == "proven_optimal"]
                entry["n_proven_optimal"] = len(proven)
                entry["coverage_mean_proven_only"] = (
                    float(np.mean([r["coverage"] for r in proven])) if proven else float("nan"))
            out.append(entry)
    return out


def analyze(raw_path: Path, out_path: Path) -> Dict[str, object]:
    P.assert_locked()
    if out_path.exists():
        raise P.ProtocolViolation(f"REFUSING to overwrite existing artifact: {out_path}")
    payload = json.loads(raw_path.read_text())
    rows = payload["rows"]
    questions = {q["qid"]: q for q in payload["questions"]}

    confirmatory = []
    for rho in P.RHO_LEVELS:
        deltas, qids = paired_deltas(rows, rho)
        observed, p_raw = signflip_pvalue(deltas)
        low, high = bootstrap_ci(deltas)
        confirmatory.append({
            "rho": rho, "n": int(deltas.size),
            "mean_delta_C": observed, "median_delta_C": float(np.median(deltas)),
            "ci95_mean": [low, high],
            "cohen_dz": cohen_dz(deltas),
            "rank_biserial": rank_biserial(deltas),
            "p_signflip_two_sided": p_raw,
            "p_wilcoxon_secondary": wilcoxon_p(deltas),
            "distribution": distribution(deltas),
        })

    corrected = holm([entry["p_signflip_two_sided"] for entry in confirmatory])
    if len(corrected) != P.CONFIRMATORY_FAMILY_SIZE:
        raise P.ProtocolViolation(
            f"Holm family has {len(corrected)} members, protocol fixes it at "
            f"{P.CONFIRMATORY_FAMILY_SIZE}")
    for entry, adjustment in zip(confirmatory, corrected):
        entry["p_holm"] = adjustment["p_holm"]
        entry["reject_at_alpha"] = adjustment["reject"]

    diagnostic = []
    for rho in P.RHO_LEVELS:
        deltas, qids = paired_deltas(rows, rho)
        d_values = np.array([questions[q]["D"] for q in qids], dtype=np.float64)
        entry = {"rho": rho, **spearman_with_ci(d_values, deltas)}
        entry["note"] = ("exploratory and predictive only; not causal, not part of the "
                         "confirmatory family, no threshold estimated")
        diagnostic.append(entry)

    result = {
        "protocol_sha256_verified_at_runtime": P.assert_locked(),
        "protocol": P.frozen_constants(),
        "source_raw": str(raw_path),
        "n_questions": payload["n_questions"],
        "confirmatory": {
            "comparison": "greedy_token_aware vs greedy_objective",
            "hypotheses": "H0: Delta_C = 0 vs H1: Delta_C != 0 (two-sided)",
            "family": [f"rho={rho}" for rho in P.RHO_LEVELS],
            "multiplicity": "holm",
            "alpha": P.ALPHA,
            "results": confirmatory,
        },
        "descriptive_arms": arm_summary(rows),
        "exploratory_diagnostic": diagnostic,
        "sesoi": {
            "specified": False,
            "statement": ("No externally established practical threshold exists for "
                          "supporting-fact coverage. The full Delta_C distribution is "
                          "reported instead of a single practical-significance verdict."),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(RESULTS_DIR / "experiment_raw.json"))
    parser.add_argument("--out", default=str(RESULTS_DIR / "experiment_analysis.json"))
    args = parser.parse_args(argv)
    result = analyze(Path(args.raw), Path(args.out))
    print(json.dumps(result["confirmatory"]["results"], indent=1)[:2000])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
