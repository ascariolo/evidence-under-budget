# Step 5 — Post-run statistical and methodological audit

Read-only. No source code, generator, optimizer, seed, or experimental artifact was
modified. New artifact written: `results/controlled/step5_audit.json`.

**This is not a paper conclusion.** It is an audit of whether the Step 4 analysis
matches the preregistered design, and of exactly what the completed experiment supports.

---

# A. Statistical audit report

## A1. Primary estimand — verified correct

| Check | Result |
|---|---|
| `Δ_gap == score(token_aware) − score(objective)` | max abs deviation **2.2e-16** (machine epsilon) |
| `gap(m) == ILP − score(m)` | max abs deviation **0.0** (exact) |
| Reference proven optimal on every row | **1080/1080** |
| Sign convention | `Δ_gap > 0` ⟹ token-aware is closer to the optimum ✓ |

Two gaps are marginally negative (`−4.8e-8`, `−2.7e-8`) on instances where a heuristic
*matched* the optimum. Floating-point noise, not a violation of `ILP ≥ heuristic`.

## A2. Analysis families — **preregistration mismatch found**

**This is the audit's main methodological finding.**

The design (§10, §6) preregistered **24 primary cells = 4 RL × 3 budgets × 2 redundancy**,
Holm for confirmatory, BH alongside.

The Step 4 analysis actually corrected a **12-member family** consisting of:
- 3 budget-level tests, pooled over condition **and** redundancy, and
- 9 condition × budget tests, pooled over redundancy.

Two separate problems:

1. **Wrong family.** It matches neither the preregistered 24 cells nor an 18-cell
   A/B/C-only family.
2. **Mixed aggregation levels.** Pooled and per-condition tests were placed in one
   correction family. Those tests are not independent — the 3 pooled tests are
   aggregates of the 9 — so the family is not a coherent multiplicity unit.

The Step 4 report's "12/12 cells survive Holm" therefore describes a family that was
never preregistered.

**Recomputed on the correct families** (this audit):

| Family | n tests | Holm-significant @0.05 | BH-significant | worst Holm-adjusted p |
|---|---|---|---|---|
| **Preregistered, 24 cells (incl. D)** | 24 | **24/24** | 24/24 | 0.0052 |
| **A/B/C only, 18 cells** | 18 | **18/18** | 18/18 | 0.0052 |
| As executed, 12 | 12 | 12/12 | 12/12 | 5.0e-08 |

**The substantive conclusion is unaffected** — every cell survives under either correct
family, with a worst adjusted p of 0.0052. The reporting was wrong; the result was not.

**A second, pre-existing inconsistency:** the preregistered 24-cell family *includes*
condition D's 6 cells, while the same design states D must never be pooled with A–C.
The design contradicts itself. The 18-cell A/B/C family is the defensible confirmatory
unit, and is what this audit uses.

## A3. Effect-size scale — no retrospective normalization in the primary reading

Primary (absolute objective units, as preregistered):

| budget | ILP optimum mean (sd) | Δ_gap mean (sd) | gap `greedy_objective` | gap `token_aware` |
|---|---|---|---|---|
| tight | 2.006 (0.375) | **+0.302** (0.283) | 0.3315 | **0.0299** |
| medium | 3.037 (0.459) | **+0.102** (0.117) | 0.1311 | **0.0293** |
| loose | 3.390 (0.510) | **−0.031** (0.043) | **0.0016** | 0.0325 |

Overall ILP objective: mean 2.811, sd 0.741, range [1.185, 4.341].

**Secondary, clearly labelled as such** (normalized after the fact):

| budget | Δ_gap / ILP mean | Δ_gap / across-instance ILP sd |
|---|---|---|
| tight | **+15.0%** | +0.80 |
| medium | +3.4% | +0.22 |
| loose | −0.9% | −0.06 |

Reading: at tight budgets the effect is **large** — token-awareness closes ~91% of
`greedy_objective`'s distance to the optimum (0.3315 → 0.0299) and is worth 0.80 sd of
the instance-to-instance variation in the optimum itself. At medium it is modest. At
loose it is **statistically certain but practically negligible in absolute terms**
(−0.031 on an optimum of 3.39), even though in *relative* terms `greedy_objective` is
~20× closer (0.0016 vs 0.0325).

## A4. Budget effect — the most robust finding in the experiment

| grouping | tight | medium | loose | monotone ↓ | sign flip |
|---|---|---|---|---|---|
| overall | +0.3016 | +0.1018 | −0.0309 | ✓ | ✓ |
| A | +0.3134 | +0.1039 | −0.0294 | ✓ | ✓ |
| B | +0.4080 | +0.1418 | −0.0484 | ✓ | ✓ |
| C | +0.1835 | +0.0597 | −0.0150 | ✓ | ✓ |
| low redundancy | +0.3424 | +0.1115 | −0.0147 | ✓ | ✓ |
| moderate redundancy | +0.2608 | +0.0921 | −0.0472 | ✓ | ✓ |

**All 6 condition × redundancy subgroups are monotone decreasing and all 6 flip sign.
Zero subgroups reverse the overall pattern.**

Within-instance paired contrast (tight vs loose on the *same* generated corpus):
mean +0.3325, **222/14/4**, `d_z` = +1.146, p = 1.5e-38.

## A5. β effect — interpretable only inside β < 1, and not extrapolable

`β̂` range across the **entire** experiment: **[−0.721, +0.755]**. Instances with
β̂ > 1: **0**. Instances with β̂ > 0.8: **0**.

| | observed |
|---|---|
| `β̂` main effect (mixed model) | +0.0104, **p = 0.557 — null** |
| `log(ρ) × β̂` interaction | −0.1839, p = 3.2e-21 |
| Spearman(Δ_gap, β̂) at tight | **+0.399** (p = 1.3e-10) |
| at medium | +0.338 (p = 8.2e-08) |
| at loose | **−0.325** (p = 2.6e-07) |
| condition means, tight | C +0.18 < A +0.31 < B +0.41 — increasing in β target |
| condition means, loose | B −0.048 < A −0.029 < C −0.015 — decreasing in β target |

What this **does** support: within the tested range `β ∈ [−0.72, +0.76]`, the association
between β and the benefit of token-awareness **itself reverses with budget** — positive
at tight, negative at loose.

What it **does not** support, and must not be described as: evidence for the predicted
β-sign reversal at β = 1. Two reasons:

1. **No β > 1 observation exists.** The neutral point of the theory was never reached.
2. **The observed slope points the wrong way for extrapolation.** Inside β < 1, *higher*
   β is associated with *more* benefit at tight budgets. Naively extrapolating that
   slope past β = 1 would predict token-awareness helps even more — the opposite of the
   theoretical prediction. The observed interaction therefore cannot be used as
   indirect evidence about the β > 1 regime in either direction.

The pattern **is** consistent with the design's separate H2a reasoning (condition C,
where relevance ranking and density ranking already agree, shows the smallest marginal
contribution). That is an agreement-of-rankings explanation, not an elasticity-threshold
one.

## A6. Redundancy effect — **not robust where it matters**

Preregistered status: the redundancy × budget contrast was **not** part of any declared
correction family. These 9 condition × budget subgroup tests are **exploratory**.

Holm-corrected within this exploratory family of 9:

| cell | low − moderate | `d_z` | raw p | Holm p | significant |
|---|---|---|---|---|---|
| B \| loose | +0.0498 | +0.86 | 0.0000 | 0.0000 | **yes** |
| A \| loose | +0.0304 | +0.80 | 0.0000 | 0.0000 | **yes** |
| C \| loose | +0.0173 | +0.63 | 0.0007 | 0.0052 | **yes** |
| B \| medium | +0.0661 | +0.44 | 0.0012 | 0.0070 | **yes** |
| B \| tight | +0.1180 | +0.45 | 0.0232 | 0.1160 | no |
| A \| tight | +0.0850 | +0.29 | 0.0697 | 0.2786 | no |
| C \| tight | +0.0416 | +0.19 | 0.2707 | 0.8121 | no |
| A \| medium | **−0.0072** | −0.06 | 0.9417 | 1.0000 | no |
| C \| medium | **−0.0007** | −0.01 | 0.9735 | 1.0000 | no |

- Correct sign in **7/9**; significant after Holm in **4/9**.
- **3 of the 4 survivors are at the loose budget**, where Δ_gap is *negative*. There the
  correct statement is "moderate redundancy makes token-awareness **more harmful**",
  which is a different claim.
- **At tight budgets — where the benefit actually exists — 0/3 survive Holm.**
- Two subgroups have the wrong sign (both at medium, both ≈ 0).

The marginal pooled contrast is real (low vs moderate at tight: +0.342 vs +0.261), and
the continuous model's `mean_similarity` coefficient is negative (−0.476, p = 2.0e-04),
but that term is the least model-robust (see A7). The claim "higher redundancy reduces
the benefit" is **directionally suggestive, not established**, and applies only to the
tested range γ ∈ {0.15, 0.70} → mean similarity 0.095–0.180. The real corpus sits at
0.351, outside the tested range entirely.

## A7. Mixed models — **Step 4 documentation was wrong about the cause**

Step 4's `docs/controlled_full_experiment.md` §5.4 stated the convergence warning was
"consistent with a near-zero seed-level variance component." **That is incorrect.**

| quantity | value |
|---|---|
| model converged | **True** |
| residual variance (`scale`) | 0.021507 |
| `Group Var` parameter (statsmodels reports this **relative to scale**) | 0.2795 |
| actual random-effect variance | 0.00601 (sd 0.0775) |
| empirical between-seed variance of instance means | 0.00761 (sd 0.0872) |
| **ICC** | **0.218** |

The seed variance component is **substantial, not near zero**, and the model-based
estimate matches the empirical between-seed variance. The `ConvergenceWarning` fired but
the optimizer converged, so the warning is not evidence of a degenerate fit.

**Does any reported inference depend on the mixed models?** Coefficient stability vs OLS:

| term | mixed | OLS | stable? |
|---|---|---|---|
| `log(ρ)` | −0.1894 (p=2.6e-141) | −0.1894 (p=5.5e-85) | identical |
| `log(ρ) × β̂` | −0.1839 (p=3.2e-21) | −0.1839 (p=2.5e-16) | identical |
| `β̂` | +0.0104 (p=0.557) | +0.0282 (p=0.147) | both null |
| `mean_similarity` | −0.4757 (p=2.0e-04) | −0.3049 (p=2.6e-02) | **model-dependent** |

The budget effect and the budget × β interaction are invariant to the random-effect
specification. Only the redundancy/similarity coefficient moves materially, which is a
further reason to treat A6 as suggestive. **The preregistered non-parametric per-cell
analyses remain the primary valid analysis**; no model was substituted for them.

## A8. Condition D — correctly excluded from the executed primary family

| | |
|---|---|
| D rows in the **executed** confirmatory family | **0** ✓ |
| D cells in the **preregistered** 24-cell family | 6 ✗ (design self-contradiction, A2) |
| D `β̂` range | **[+0.067, +0.374]**, 0 instances > 1 |
| D `W_sat` vs A/B/C | **2885 vs 1423** (~2×) |
| D Δ_gap | tight +0.998, medium +0.649, loose −0.087 |

**Why D cannot be read as a β > 1 test.** The design asserted D would produce "β > 1 in
the tail". Measured, the opposite holds: inflating the token cost of the most relevant
documents makes them **3.25× less dense** (value per token), which is the *low*-elasticity
direction. D's global β̂ is +0.24 and never exceeds 1. D is an **extreme β < 1** condition,
which is exactly why it yields the largest benefit observed anywhere (80/0/0 at tight).
Its absolute budgets are also ~2× those of A/B/C, so pooling would confound the
intervention with budget scale.

## A9. Arm E — anchor only; quantitative comparison would be inappropriate

| | tight | medium | loose |
|---|---|---|---|
| arm E Δ_gap | +0.3876 | +0.1309 | −0.0432 |
| A/B/C Δ_gap | +0.3016 | +0.1018 | −0.0309 |

Qualitatively similar: same sign pattern, monotone decreasing, same reversal point.

**Quantitative comparison is not licensed.** Arm E is a different instance population:
mean pairwise similarity **0.351 vs 0.137**, mean ILP optimum **2.675 vs 2.811**, no
redundancy factor, a different relevance distribution (it can go negative; the synthetic
floor is 0.05), and `β̂` computed on a positive-relevance subset with 8 values dropped.
The two are not matched on any covariate, so the numeric closeness is not evidence of
calibration. Arm E is one generator, one query, one encoder — **not external validity**.

## A10. Data integrity — full QC re-run from raw, all pass

| check | result |
|---|---|
| rows | 5400 (4800/4800 synthetic, 600/600 arm E) |
| duplicate (instance, budget, method) | **0** |
| seeds 0–39 present | ✓ |
| conditions / redundancy / budgets / methods complete | ✓ |
| ILP proven globally optimal | **1080/1080** |
| heuristics flagged proven-optimal | **0** |
| missing primary outcome | **0** |
| protocol lock vs execution (λ_obj, λ_mmr, γ, ρ, β targets, version) | **all True** |
| saturation solves proven | ✓ |

---

# B. What we can safely claim

**Supported by the preregistered experiment:**

1. **Budget tightness controls the sign and magnitude of the token-awareness effect.**
   Monotone decreasing in ρ, sign-flipping, in **6/6** condition × redundancy subgroups
   with zero reversals; within-instance 222/14/4, p = 1.5e-38. Survives Holm on both
   correct families (18 and 24 cells), worst adjusted p 0.0052.
2. **At tight budgets the effect is large and practically meaningful**: +0.302 absolute,
   closing ~91% of `greedy_objective`'s gap, ≈0.80 sd of the between-instance spread of
   the optimum.
3. **At loose budgets token-awareness is reliably but negligibly harmful**: −0.031 on an
   optimum of ~3.39 (−0.9%), statistically certain, practically near-zero.
4. **`greedy_token_aware` has a near budget-invariant absolute gap** (0.030 / 0.029 /
   0.033) while every other heuristic degrades 10–20× as the budget tightens.
5. **Condition C shows the smallest benefit at every budget**, consistent with the
   ranking-agreement explanation (H2a), not with an elasticity-threshold explanation.
6. **The ILP is a sound ground truth here**: 1080/1080 proven optimal, gap identity exact.

**Not supported:**

1. **Any claim about β > 1.** Zero observations exist; the regime was never built.
2. **The β-sign-reversal hypothesis (H3).** Untested, not refuted. The observed within-
   range slope points the wrong way for extrapolation.
3. **A β main effect.** p = 0.557, null under both mixed and OLS.
4. **"Higher redundancy reduces the token-awareness benefit" as an established result.**
   0/3 tight-budget subgroups survive Holm; 3/4 survivors sit at the loose budget where
   the effect is negative; 2 subgroups have the wrong sign; the supporting model
   coefficient is the least model-robust one.
5. **Anything about real-corpus redundancy.** Tested range 0.095–0.180; the real corpus
   is 0.351, outside it.
6. **External validity.** Arm E is an anchor, not validation.
7. **Condition D as evidence about high elasticity.** It is an extreme low-elasticity
   condition.

**Open:**

1. Does token-awareness actually hurt at β > 1, as theory predicts?
2. Where exactly is the budget crossing point ρ*, and does it move with β or redundancy?
3. Is the redundancy interaction real at tight budgets, with adequate power?
4. Does the benefit survive at real-corpus redundancy (~0.35)?
5. Is the mechanism cardinality (buying more, cheaper documents) or elasticity?
6. Does any of this transfer to a downstream task metric?

---

# C. What experiment is still missing

The completed experiment establishes a **budget** law and leaves the **elasticity**
question — the one the controlled benchmark was built to answer — unresolved, because
the generator never produced β > 1.

Missing, in order of importance:

1. **A β > 1 regime.** Required to test H3 at all. The current construction cannot reach
   it: `β̂ ≈ ρ_c · sd(log r)/sd(log w)` with `sd(log r)/sd(log w) ≈ 0.91`, so even
   `ρ_c = 1` caps β̂ below ~0.91. Reaching β > 1 requires changing the marginal spreads
   (e.g. raising relevance dispersion or lowering token dispersion), which is a
   **generator change** and must be a new, separately preregistered design — not a patch.
2. **Budget resolution around the crossing.** Three budget levels locate a sign change
   between ρ = 0.60 and ρ = 1.50 but cannot estimate ρ* or test whether it shifts.
3. **Redundancy at realistic levels** (up to ~0.35) and with enough power at tight
   budgets, where the current design is underpowered for this contrast.
4. **A mechanism-discriminating manipulation.** Cardinality and elasticity are currently
   confounded: token-awareness buys more, cheaper documents *and* exploits low
   elasticity. A cardinality-matched comparison would separate them.

---

# D. Recommended next experiment

**One experiment, narrowly scoped: the elasticity sweep.** Everything else is secondary.

| element | proposal |
|---|---|
| **Question** | Does token-awareness stop helping, and then hurt, as β crosses 1? |
| **Design** | β as a **continuous swept factor** targeting β ∈ {−0.5, 0, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0}, crossed with a denser budget grid ρ ∈ {0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5} |
| **Generator change required** | Yes, and it must be preregistered as a new design: raise `sd(log r)` and/or lower `sd(log w)` so β > 1 is reachable, then **re-validate marginals and non-degeneracy** exactly as in Step 3C. The token-CV guard (currently only +0.015 of headroom) is the binding constraint — lowering token dispersion will violate it, so relevance dispersion is the safer lever. |
| **Primary estimand** | unchanged: `Δ_gap = score(token_aware) − score(objective)`, absolute units |
| **Preregistered test** | sign of `Δ_gap` at each β level at fixed tight ρ; and the location of the β zero-crossing with a bootstrap CI |
| **Family** | declare **one** coherent family before running — per-cell, one aggregation level, D and E excluded |
| **Held fixed** | `λ_obj = 0.1`, `λ_mmr = 0.5`, `N = 25`, `K = 5`, seeds 0–39, redundancy at both current levels |
| **Power** | 40 seeds detects `d_z ≥ 0.5` at ~0.85; the tight-budget effects here were `d_z ≈ 0.9–1.4`, so 40 is adequate for the primary contrast but **not** for the redundancy interaction — raise to ~80 seeds if that contrast is also to be confirmatory |
| **Cost** | ≈ 8 β × 8 ρ × 2 redundancy × 40 seeds ≈ 5120 instance-budgets; at observed ILP timings (~0.26 s) roughly **1.5–2 h** |
| **Falsification** | if `Δ_gap` stays positive at β = 1.5–2.0 under a tight budget, the elasticity theory is wrong and the effect is purely a knapsack-capacity phenomenon |

**Do not** run this as a modification of the current benchmark. It requires a generator
change, so it needs its own design document, its own pre-flight, and its own lock — the
same sequence used for the current one.
