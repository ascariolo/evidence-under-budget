r"""RETIRED at protocol 6.1-amended. Kept only so the 6.0 preflight record can be audited.

TOST (former T5) and its margin rule were retired: the margin rule degenerated (median
token-aware gap exactly 0; the 'unstable' fallback fired on an IQR of 2.8e-17, float noise),
and equivalence to zero at beta = 2.0 is a prediction of none of the three hypotheses.

The confirmatory analysis (experiments/elasticity/analysis.py) has no import path to this
module and refuses to read results/elasticity/tost_margin.json.
"""

from __future__ import annotations

import json, math, os
from typing import Any, Dict, Sequence

import numpy as np

from experiments.elasticity.protocol import RETIRED_CONSTANTS as RC

RETIRED = True


def compute_margin(token_aware_gaps_at_tight: Sequence[float]) -> Dict[str, Any]:
    g = np.sort(np.asarray(token_aware_gaps_at_tight, float))
    med = float(np.median(g))
    iqr = float(np.quantile(g, 0.75) - np.quantile(g, 0.25))
    unstable = bool(iqr > RC["TOST_MARGIN_UNSTABLE_IQR_MULTIPLE"] * med)
    m = RC["TOST_MARGIN_FALLBACK"] if unstable else round(med, RC["TOST_MARGIN_DECIMALS"])
    return {"n": int(g.size), "median_unrounded": med, "iqr_unrounded": iqr, "unstable": unstable,
            "rule": "fallback 0.03" if unstable else "median rounded to 2dp", "m": float(m)}


def tost_power(m: float, sd: float, n: int, alpha: float = RC["TOST_ALPHA"]) -> float:
    from scipy.stats import norm
    if sd <= 0:
        return 1.0
    return float(max(0.0, 2 * norm.cdf(m * math.sqrt(n) / sd - norm.ppf(1 - alpha)) - 1))


def n_required(m: float, sd: float, power: float = RC["TOST_TARGET_POWER"],
               alpha: float = RC["TOST_ALPHA"]) -> int:
    from scipy.stats import norm
    z = norm.ppf(1 - alpha) + norm.ppf(1 - (1 - power) / 2)
    return int(math.ceil((z * sd / m) ** 2))


def write_margin_once(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if os.path.exists(path):
        existing = json.load(open(path, encoding="utf-8"))
        if existing["margin"] != payload["margin"]:
            raise RuntimeError("TOST margin recomputation differs; refusing to overwrite")
        return {"written": False, "matched_existing": True}
    json.dump(payload, open(path, "w", encoding="utf-8"), indent=2, default=str)
    return {"written": True, "matched_existing": None}
