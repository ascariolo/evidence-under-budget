# Step 4 — Controlled Full Experiment: data collection and first-pass analysis

**This is not a finished research conclusion.** It is data collection plus a
first-pass statistical analysis under the locked protocol. Pre-registered
hypotheses, observed results and exploratory observations are kept separate
throughout, and exploratory findings are not promoted to hypotheses.

---

## 1. Exact protocol

Executed exactly as locked in `experiments/controlled/protocol.py`
(`CONTROLLED_PROTOCOL_VERSION = 0.1-pilot`, lock stamped at step 3C.1).

| Factor | Levels |
|---|---|
| Relevance–length | A (`ρ_c=0`, β target 0), B (`+0.5`, +0.5), C (`−0.5`, −0.5), D (adversarial, two-pass `W_sat`) |
| Redundancy | low `γ=0.15`, moderate `γ=0.70` |
| Budget | tight `ρ=0.25`, medium `ρ=0.60`, loose `ρ=1.50` of `W_sat` |
| Seeds | 0–39, fixed |
| Held constant | `N=25`, `K=5`, `d=64`, `objective_lambda=0.1`, `mmr_lambda=0.5` |
| Methods | `top_k`, `mmr`, `greedy_objective`, `greedy_token_aware`, `ilp` |
| Solver | PuLP/CBC, 600 s limit, full-corpus scope |

**Primary estimand** (protocol §4), in **absolute** objective units — no ratio is
used, because the reference objective can be non-positive:

```
Δ_gap = gap(greedy_objective) − gap(greedy_token_aware)
      = score(greedy_token_aware) − score(greedy_objective)
```

`Δ_gap > 0` ⟹ token-awareness reduces the distance to the proven optimum.

`greedy_objective` and `greedy_token_aware` differ **only** by the `1/w_i` factor,
so their paired difference isolates token-awareness. Top-K and true MMR are
secondary. Condition D and arm E are tagged `analysis_group` at collection time
and are never pooled with A/B/C.

## 2. Execution summary

| | |
|---|---|
| Runtime | **24.3 min** |
| Instances | 320 synthetic (40 seeds × 4 conditions × 2 redundancy) + 40 arm E |
| Instance × budget | 1080 (960 synthetic + 120 arm E) |
| Rows (instance × budget × method) | **5400** (4800 synthetic + 600 arm E) |
| ILP solves | 1080 |
| Failed / missing runs | **none** |

## 3. Data completeness and 4. Quality control

**QC PASS.** Run before any statistic; a failure would have aborted the analysis.

| Check | Result |
|---|---|
| Rows present vs expected | 4800/4800 synthetic, 600/600 arm E |
| Duplicate (instance, budget, method) keys | 0 |
| Seeds 0–39 present | yes |
| All conditions / redundancy / budgets / methods present | yes |
| **ILP proven globally optimal** | **1080/1080 (100%)** |
| ILP objective-consistency failures | 0 |
| Missing primary outcome | 0 |
| **Heuristics flagged proven-optimal** | **0** |
| Saturation solves proven | all |
| Protocol lock matches execution metadata | **True** (λ, N, γ, ρ targets, β targets, version) |

## 5. Primary results

### 5.1 Δ_gap by budget (A/B/C pooled; D and E excluded)

| budget | n | mean | median | sd | W/L/T | `d_z` | `r_rb` | Wilcoxon p | 95% BCa CI |
|---|---|---|---|---|---|---|---|---|---|
| tight | 240 | **+0.3016** | +0.2310 | 0.283 | 205/19/16 | +1.066 | +0.957 | 1.1e-35 | [+0.267, +0.339] |
| medium | 240 | **+0.1018** | +0.0725 | 0.117 | 195/38/7 | +0.871 | +0.859 | 5.5e-30 | [+0.088, +0.118] |
| loose | 240 | **−0.0309** | −0.0167 | 0.043 | 8/166/66 | −0.714 | −0.975 | 6.0e-31 | [−0.037, −0.026] |

**All 12 confirmatory cells survive Holm correction** (and BH).

### 5.2 Δ_gap by β condition × budget

| cond | β target | budget | n | mean | `d_z` | W/L/T | p | 95% CI |
|---|---|---|---|---|---|---|---|---|
| A | 0.00 | tight | 80 | +0.3134 | +1.045 | 68/8/4 | 4.2e-13 | [+0.252, +0.382] |
| A | | medium | 80 | +0.1039 | +1.082 | 70/9/1 | 1.4e-12 | [+0.084, +0.126] |
| A | | loose | 80 | −0.0294 | −0.822 | 3/58/19 | 1.0e-11 | [−0.038, −0.022] |
| B | +0.50 | tight | 80 | **+0.4080** | +1.428 | 76/3/1 | 1.7e-14 | [+0.347, +0.471] |
| B | | medium | 80 | +0.1418 | +0.983 | 68/11/1 | 5.2e-12 | [+0.112, +0.176] |
| B | | loose | 80 | −0.0484 | −0.839 | 2/66/12 | 2.4e-13 | [−0.063, −0.038] |
| C | −0.50 | tight | 80 | **+0.1835** | +0.866 | 61/8/11 | 8.0e-11 | [+0.141, +0.232] |
| C | | medium | 80 | +0.0597 | +0.671 | 57/18/5 | 5.0e-08 | [+0.042, +0.081] |
| C | | loose | 80 | −0.0150 | −0.667 | 3/42/35 | 1.5e-08 | [−0.021, −0.011] |

### 5.3 Δ_gap by redundancy × budget

| redundancy | budget | n | mean | `d_z` | W/L/T | p |
|---|---|---|---|---|---|---|
| low (0.15) | tight | 120 | +0.3424 | +1.181 | 113/5/2 | 2.6e-20 |
| low | medium | 120 | +0.1115 | +0.963 | 98/16/6 | 2.0e-17 |
| low | loose | 120 | −0.0147 | −0.524 | 3/63/54 | 6.2e-12 |
| moderate (0.70) | tight | 120 | +0.2608 | +0.963 | 92/14/14 | 9.3e-17 |
| moderate | medium | 120 | +0.0921 | +0.782 | 97/22/1 | 5.1e-14 |
| moderate | loose | 120 | −0.0472 | −0.953 | 5/103/12 | 9.8e-20 |

### 5.4 Pre-registered mixed models

**Factorial** — `Δ_gap ~ budget × condition × redundancy + (1|seed)`, n=720:

| term | coef | p | 95% CI |
|---|---|---|---|
| Intercept (tight, A, moderate) | +0.2709 | 2.4e-24 | [+0.219, +0.323] |
| budget → loose | **−0.3155** | 1.6e-21 | [−0.380, −0.251] |
| budget → medium | **−0.1634** | 8.0e-07 | [−0.228, −0.099] |
| condition B | +0.0780 | 1.8e-02 | [+0.013, +0.143] |
| condition C | −0.1082 | 1.1e-03 | [−0.173, −0.043] |
| redundancy low | +0.0850 | 1.0e-02 | [+0.020, +0.150] |
| loose × B | −0.1068 | 2.3e-02 | [−0.199, −0.015] |
| loose × C | +0.1291 | 5.8e-03 | [+0.037, +0.221] |
| medium × redundancy low | −0.0922 | 4.9e-02 | [−0.184, −0.000] |

All three-way interactions non-significant (p ≥ 0.45).

**Continuous** — `Δ_gap ~ log(ρ) × β̂ + mean_similarity + (1|seed)`, n=720:

| term | coef | p | 95% CI |
|---|---|---|---|
| `log(ρ)` | **−0.1894** | 2.6e-141 | [−0.204, −0.175] |
| `β̂` (main effect) | +0.0104 | **0.557 (null)** | [−0.024, +0.045] |
| **`log(ρ) × β̂`** | **−0.1839** | 3.2e-21 | [−0.222, −0.146] |
| mean similarity | −0.4757 | 2.0e-04 | [−0.726, −0.225] |

Both models produced a `ConvergenceWarning` ("MLE may be on the boundary"),
consistent with a near-zero seed-level variance component. Coefficients agree with
the non-parametric per-cell results, which are the pre-registered primary readout.

## 6. Secondary results

Mean **absolute** optimality gap (A/B/C; lower is better):

| method | tight | medium | loose |
|---|---|---|---|
| Top-K | 0.3372 | 0.2006 | 0.0234 |
| true MMR | 0.4295 | 0.2077 | 0.0345 |
| greedy_objective | 0.3315 | 0.1311 | **0.0016** |
| **greedy_token_aware** | **0.0299** | **0.0293** | 0.0325 |

Token-aware greedy is remarkably **flat** across budgets (0.030 → 0.033) while every
other heuristic degrades sharply as the budget tightens. At the loose budget
`greedy_objective` is near-exact (0.0016) and token-awareness is the worst of the two.

Paired vs baselines: token-aware beats Top-K at tight (+0.307, `d_z`=1.06,
p=8e-36) and medium (+0.171, p=1e-38) but is **indistinguishable at loose**
(−0.009, p=0.14). It beats true MMR at tight (+0.400) and medium (+0.178); at loose
the difference is negligible (+0.002, p=0.012).

## 7. Mechanism diagnostics

**Descriptive only. The design randomizes the experimental factors, not these
quantities, so they are not causal mediators.**

| budget | Δ n_selected | Δ tokens used | Δ relevance | Δ redundancy/pair | Δ mean selected length | corr(Δ_gap, Δn) | corr(Δ_gap, Δrel) |
|---|---|---|---|---|---|---|---|
| tight | +2.10 | −3.5 | +0.434 | −0.192 | −32.5 | +0.867 | **+0.982** |
| medium | +2.13 | −8.2 | +0.391 | −0.039 | −17.3 | +0.770 | +0.826 |
| loose | +0.42 | −24.0 | +0.075 | −0.002 | −4.1 | −0.343 | −0.101 |

Where token-awareness helps, it buys ~2 more documents that are ~33 tokens shorter,
captures more total relevance while spending *fewer* tokens, and lowers redundancy
per pair. At the loose budget the selection difference nearly vanishes and the
correlations reverse.

## 8. Condition D (adversarial; analysed separately, never pooled)

| budget | n | mean Δ_gap | `d_z` | W/L/T | p | 95% CI |
|---|---|---|---|---|---|---|
| tight | 80 | **+0.9975** | +2.060 | **80/0/0** | 7.9e-15 | [+0.893, +1.105] |
| medium | 80 | +0.6491 | +1.675 | 79/1/0 | 1.1e-14 | [+0.566, +0.737] |
| loose | 80 | −0.0868 | −1.127 | 0/74/6 | 2.0e-14 | [−0.105, −0.071] |

D produces by far the largest benefit at tight budgets — an unbroken 80/0/0 — and
still reverses at loose. Its absolute budgets are ~2× A/B/C's because the
intervention roughly doubles `W_sat`; this is the reason it is never pooled.

## 9. Arm E (realism anchor; **not** an external-validity claim)

40 seeds, the **unchanged** original text generator, same instance-relative `ρ` rule,
no redundancy factor (real corpus structure), `all-MiniLM-L6-v2` + `cl100k_base`.

| budget | n | mean Δ_gap | `d_z` | W/L/T | p | 95% CI |
|---|---|---|---|---|---|---|
| tight | 40 | +0.3876 | +1.449 | 37/2/1 | 1.7e-07 | [+0.307, +0.472] |
| medium | 40 | +0.1309 | +0.470 | 24/13/3 | 9.7e-03 | [+0.065, +0.248] |
| loose | 40 | −0.0432 | −1.142 | 5/34/1 | 2.8e-07 | [−0.056, −0.033] |

`β̂` computed on the **positive-relevance subset only** (the real encoder returns
negative cosines; **8 values dropped in total** across 40 seeds); mean `β̂` = +0.076.
Mean pairwise similarity 0.351, vs 0.095/0.180 in the synthetic low/moderate levels.

Arm E reproduces the synthetic pattern qualitatively. **This does not establish
external validity**: it is one generator, one query, one encoder.

## 10. Limitations

1. **β > 1 was never realized — in 0 of 440 instances.** Per-condition ranges: A
   [−0.40, +0.37], B [−0.07, +0.76], C [−0.72, −0.11], D [+0.07, +0.37], E
   [−0.07, +0.31]. Every observation lies in `β < 1`.
2. The redundancy factor spans 0.095 → 0.180 while the real corpus sits at 0.351;
   it is a **low vs moderate** contrast (DEV-2, locked).
3. `objective_lambda = 0.1` was selected on the original evaluation corpus in v0 and
   is frozen, not held out. Every result is conditional on it.
4. `mmr_lambda = 0.5` is untuned by protocol, so MMR is not shown at its best.
5. One generator family, `N = 25`, one embedding dimension, no downstream task metric.
6. Mixed models hit a boundary in the variance component; the non-parametric
   per-cell results are the pre-registered primary readout.

## 11. Unexpected findings (exploratory — NOT hypotheses)

- **Condition D behaves as an extreme `β < 1` case, not the `β > 1` case the design
  intended.** The design asserted D would create "`β > 1` in the tail". Measured, its
  adversarial documents are **3.25× *less* dense** than the rest and the global
  `β̂` = +0.24. Making highly relevant documents expensive makes them *worse* value
  per token, which is the low-elasticity direction. The design text was wrong about
  the sign, so **H3 is untested**, not refuted.
- `β̂` has **no main effect** (p = 0.557) but a strong **interaction with budget**
  (p = 3.2e-21). Elasticity appears to modulate the budget effect rather than act
  on its own — an exploratory reading requiring confirmation.
- Condition C shows the **smallest** benefit at every budget (+0.18 tight vs +0.41
  for B), consistent with the design's H2a reasoning that the two rankings largely
  agree when short documents are already the most relevant.
- `greedy_token_aware`'s absolute gap is nearly **budget-invariant** (0.030–0.033)
  while every other heuristic degrades 10–20× as the budget tightens.
- Mean similarity carries a significant negative coefficient (−0.476, p = 2.0e-04):
  more redundancy, less benefit — directionally consistent with H4.

## 12. Deviations

| ID | Status |
|---|---|
| DEV-1 (condition D two-pass `W_sat`) | implemented as locked; audited clean; **still awaiting sign-off** |
| DEV-2 (redundancy wording) | locked; low-vs-moderate; parameters unchanged |
| DEV-3 (β tolerance 0.10) | locked |
| DEV-4 (β criterion = aggregate mean) | resolved at 3C.1 |
| **New, technical** | `statsmodels` (+`pandas`, `patsy`, `formulaic`) installed into the venv to run the pre-registered mixed model. Implementation dependency only; no scientific protocol change. |

No protocol parameter, seed, condition, tolerance, generator or optimizer was changed
during or after the run. No run was stopped early. Code hashes unchanged from step 3C.
