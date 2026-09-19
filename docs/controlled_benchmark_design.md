# Controlled Benchmark — Design Document

**Status: design only.** No code written, no benchmark generated, no experiment run.
Nothing in the existing repository is modified by this document.

---

## 1. Research question

> **When does token-awareness improve context selection, and when does it hurt?**

Operationally: estimate the causal contribution of the `1/w_i` factor — the *only*
difference between `greedy_objective` and `greedy_token_aware` — as a function of
budget tightness, the relevance–length relationship, and redundancy, and identify the
mechanism that produces it.

The target is not "which method wins". It is **when** and **why**.

---

## 2. Motivation from the repaired original benchmark

`docs/repaired_original_benchmark.md`, 24 seeds × 9 budgets, ILP proven optimal on
216/216 instances, found a statistically robust **sign flip** in the isolation test
(`greedy_token_aware` − `greedy_objective`):

| `W_max` | W/L/T | mean diff | `d_z` | Wilcoxon `p` |
|---:|---|---|---|---|
| 96 | 22/0/2 | +0.528 | +1.72 | 4.0e-05 |
| 256 | 12/8/4 | +0.142 | +0.65 | 1.7e-02 |
| 384 | 10/14/0 | +0.015 | +0.17 | 0.989 |
| 512 | 2/21/1 | −0.046 | −1.16 | 6.0e-05 |
| ≥768 | 2/21/1 | −0.050 | −1.18 | 6.0e-05 |

Two facts constrain this design:

1. **The effect is real and budget-dependent.** Large effect sizes at both ends.
2. **The mechanism is unidentified.** That benchmark has essentially no
   relevance–length structure (Pearson `+0.023`, Spearman `+0.076`, 0/24 seeds
   significant), so it cannot distinguish "token-awareness exploits a length/value
   mismatch" from "token-awareness is just the right knapsack heuristic when the
   constraint binds". Those are different claims with different implications.

The original generator cannot settle this, because it determines the
relevance–length relationship *accidentally*: padding is appended to on-topic
sentences and the encoder decides what that does to relevance. This design removes
that accident by manipulating the relationship explicitly.

---

## 3. Causal variables

### 3.1 What the solvers actually consume

`ContextKnapsack` takes exactly three inputs: `relevance` `(N,)`, `similarity`
`(N,N)`, `tokens` `(N,)`. Text is not an input to any solver. An instance is therefore
**fully specified by those three arrays**, which is what makes exact control possible
(section 5.1).

### 3.2 The theoretically correct length variable is elasticity, not correlation

`greedy_objective` ranks by `Δ_i`; `greedy_token_aware` ranks by `Δ_i / w_i`. The two
orderings differ exactly insofar as **value per token varies with token cost**.

Model the relationship as a log-log elasticity `β`:

```
r_i  ∝  w_i^β        ⇔        β = d log r / d log w
```

| `β` | value per token `r/w` | expected consequence |
|---|---|---|
| `β = 0` | falls as `1/w` | short documents are denser → token-awareness should help |
| `0 < β < 1` | falls | still helps, more weakly |
| `β = 1` | **constant** | the two rankings agree on the relevance term → **neutral point** |
| `β > 1` | rises | long documents are denser → token-awareness should **hurt** |

**Pearson correlation is a confounded proxy for this.** A positive relevance–length
correlation with `β < 1` still favours token-awareness; only `β > 1` reverses it. Any
design built on the sign of the correlation alone would be unable to tell those apart.
The factor is therefore parameterized by a latent coupling that *induces* a target
elasticity, and both `β̂` and the realized correlations are recorded per instance.

### 3.3 The redundancy channel is cardinality

`Score(S)` penalizes `Σ_{i<j∈S} s_ij`, which grows as `O(|S|²)` while relevance grows
as `O(|S|)`. Token-awareness systematically selects **more, cheaper** documents — in the
repaired benchmark, 6.83 vs 3.58 documents at `W_max = 96`. More documents means more
pairs, so any benefit it buys in relevance is taxed quadratically by redundancy.

This gives a concrete, falsifiable mechanism for an interaction: **the token-awareness
benefit should shrink as redundancy rises**, and the shrinkage should be *mediated by
the difference in selection cardinality*.

### 3.4 Budget tightness must be instance-relative

Absolute token budgets are not comparable across conditions with different length
distributions: a 256-token budget is tight for long documents and loose for short ones.
Using absolute budgets would confound the budget factor with the length factor.

Define the **saturation mass** `W_sat` per instance as the token cost of the
*unconstrained* optimum:

```
S_sat = argmax_S Score(S)   with no budget constraint
W_sat = Σ_{i ∈ S_sat} w_i
```

and define tightness as `ρ = W_max / W_sat`. `ρ < 1` means the constraint binds;
`ρ ≥ 1` means it does not. This is method-independent and makes budget levels mean the
same thing in every cell. It also matches the empirical observation that the sign flip
in the repaired benchmark occurred exactly where selections saturated.

---

## 4. Experimental factors

### Factor 1 — Relevance–length relationship (4 levels)

Implemented as a Gaussian copula between a relevance latent `ζ` and a length latent
`η`, so that **the marginal distributions of relevance and of token count are identical
in every level** and only the *dependence* changes. This is the central anti-confound.

| Level | Name | Copula `ρ_c` | Target elasticity `β` | Rationale |
|---|---|---|---|---|
| **A** | Independent | `0.0` | ≈ 0 | Reproduces the original benchmark's regime; isolates the pure knapsack effect. |
| **B** | Positive, sublinear | `+0.5` | ≈ 0.5 | Longer documents are more relevant, but *less* relevant per token. The realistic case: more text carries more information with diminishing returns. |
| **C** | Negative | `−0.5` | < 0 | Short documents are more relevant. Relevance ranking and density ranking largely agree. |
| **D** | Adversarial mismatch | see below | mixed, `β > 1` in the tail | A small subset of the most relevant documents is disproportionately expensive. |

**Level D construction.** Levels A–C share marginals; D deliberately does not, so it is
flagged as a *structural* condition rather than a fourth point on the copula scale. The
top `k_adv = 4` documents by relevance have their token cost multiplied by
`m_adv = 6`, capped at `0.45 · W_sat` so they remain individually feasible at the
medium budget. This creates a region with `β > 1` (the most relevant documents are also
the densest in raw relevance terms only if the multiplier is small; at `m = 6` they are
the *least* dense) — the case where a value-greedy ranker buys one expensive document
and a density-greedy ranker buys several cheap ones.

A supplementary continuous arm sweeps `ρ_c ∈ {−0.6, −0.3, 0, +0.3, +0.6}` at the medium
budget only, to trace the effect as a curve rather than four points.

**Validation criterion for `β` — LOCKED at step 3C.1.** Four quantities must be kept
distinct:

| # | Quantity | What it is |
|---|---|---|
| 1 | `β_target` | the **generator parameter** — a population-level property of the instance-generating distribution, realised exactly through `ρ_c`. Declared, never estimated. |
| 2 | `β̂(seed)` | the **per-instance estimator** — an OLS log-log slope fitted to one realisation's `N = 25` documents. A finite-sample *measurement* of (1). |
| 3 | `mean(β̂)` | the **aggregate estimate** over the fixed ensemble, seeds 0–39. |
| 4 | `BETA_TOLERANCE = 0.10` | the **validation criterion**, applied to (3) only. |

```
LOCKED:      | mean(β̂ over seeds 0–39) − β_target |  ≤  0.10      per condition
NOT REQUIRED:| β̂(seed) − β_target |                  ≤  0.10      per seed
```

The per-seed form is explicitly **not** a criterion and must never be used to accept or
reject a seed. The 40-seed pre-flight measured per-instance `sd(β̂) = 0.15–0.19`, i.e.
**1.5–1.9× the tolerance itself**, so no generator at `N = 25` could satisfy it; applying
it would select realisations by *estimator noise* rather than validate the generator
parameter, which is seed selection. Per-seed `β̂` values are still recorded for every
instance — they quantify finite-sample variability and serve as a covariate in the
continuous companion model of §10. **Reporting them is required; filtering on them is
forbidden.**

### Factor 2 — Budget tightness (3 levels, instance-relative)

| Level | `ρ = W_max / W_sat` | Meaning |
|---|---|---|
| **Tight** | `0.25` | Constraint binds hard; roughly a quarter of the useful mass is affordable. |
| **Medium** | `0.60` | Constraint binds. Chosen below `1.0` because the repaired benchmark's zero-crossing sat near saturation; putting a level at the crossing maximizes information about *where* it lies. |
| **Loose** | `1.50` | Constraint does not bind — the unconstrained optimum is affordable. |

`W_max = round(ρ · W_sat)`, computed per instance. The realized `ρ` and `W_sat` are
recorded. Levels are fixed a priori and not adjusted after seeing results.

### Factor 3 — Redundancy (2 levels)

Controlled through the component of similarity orthogonal to the query (section 5.2),
so redundancy is manipulated **without** changing relevance.

| Level | within-cluster `γ` | expected mean off-diagonal similarity |
|---|---|---|
| **Low** | `0.15` | 0.093 (measured, 40 seeds) |
| **Moderate** (key `"high"`) | `0.70` | 0.178 (measured, 40 seeds) |

> **RETRACTED at step 3C.** This paragraph originally claimed the high level was
> "calibrated to the measured v0 corpus (mean pairwise cosine 0.405), so 'high
> redundancy' means 'as redundant as the original benchmark'". **That claim is false.**
> Measured across 40 seeds: `gamma = 0.15` gives mean pairwise similarity 0.093 and
> `gamma = 0.70` gives 0.178, against 0.330-0.405 for the real corpus. At `K = 5`
> cross-topic pairs contribute about zero and are 80% of all pairs, so the construction
> ceilings near 0.26 even at `gamma = 1`. `K`, `gamma` and the construction are
> **unchanged and locked**; only the description is corrected.
>
> **This factor is a LOW vs MODERATE redundancy contrast.** It is a genuine,
> non-degenerate manipulation (about 1.9x, no overlap across seeds) but it does not
> reach real-corpus redundancy, and no analysis may describe it as doing so. See
> DEV-2 in `experiments/controlled/protocol.py`.

### Held constant (explicitly, to avoid multi-factor drift)

`N = 25` documents (identical to the original benchmark, for comparability and ILP
tractability); embedding dimension `d = 64`; `K = 5` latent topics with fixed
membership proportions; relevance marginal `Beta`-shaped and fixed; token marginal
log-normal and fixed; `objective_lambda = 0.1` **frozen at its documented value**;
`mmr_lambda = 0.5` **frozen**; number of high-relevance documents fixed by the shared
relevance marginal.

---

## 5. Benchmark generation procedure

### 5.1 Why generate embeddings/matrices directly rather than text

**Recommended: direct synthesis.** Reasons:

1. **Text cannot decouple length from relevance.** Making a document longer means adding
   tokens, and any real encoder moves the embedding when you do — that is precisely the
   semantic drift the original generator suffers from. With text, `w` and `r` cannot be
   set independently; with vectors they can.
2. **Token counts become exact by construction** rather than an artifact of a tokenizer.
3. **The solvers never see text** (section 3.1), so nothing is lost at the interface.
4. **Redundancy becomes a controlled parameter** instead of an emergent property.

**How interpretability is preserved.** The synthetic instances are *calibrated against
the real corpus*: relevance range and mean, pairwise-similarity mean, and token
distribution are all matched to values measured on the v0 corpus with
`all-MiniLM-L6-v2` and `cl100k_base`. Every generated instance is checked against those
ranges (section 13). In addition, a **realism-anchor arm** (section 7, arm E) runs the
identical analysis on the *unchanged* original text generator, so any synthetic-only
artefact shows up as a discrepancy between arm E and cell A/high-redundancy.

### 5.2 Construction

For each instance, given seed and factor levels:

**Step 1 — latent variables via Gaussian copula.**
Draw `(ζ_i, η_i)` i.i.d. from a bivariate normal with unit variances and correlation
`ρ_c`. `ζ` drives relevance, `η` drives length. Because the copula only changes the
dependence, the marginals below are unchanged across levels A–C.

**Step 2 — relevance.**
Map `Φ(ζ_i)` through the inverse CDF of a fixed `Beta(2, 3)` rescaled to
`[r_min, r_max] = [0.05, 0.60]`, matching the range observed on the v0 corpus
(`[0.186, 0.525]`) with slightly wider support.

**Step 3 — token counts.**
`w_i = round(exp(μ_w + σ_w · η_i))`, clipped to `[8, 400]`, with `μ_w, σ_w` chosen so
the marginal matches the v0 corpus (median ≈ 77, mean ≈ 84, max ≈ 205, right-skewed).
Token counts are integers and **exact by construction**.

**Step 4 — embeddings with exactly the intended relevance.**
Fix a query unit vector `q`. For each document build

```
e_i = r_i · q  +  sqrt(1 − r_i²) · u_i ,      u_i ⊥ q ,  ||u_i|| = 1
```

so `cos(q, e_i) = r_i` **exactly**, by construction. The orthogonal component `u_i`
carries all remaining structure and is free to encode redundancy.

**Step 5 — redundancy through the orthogonal component.**
Assign each document to one of `K` topics. In the subspace orthogonal to `q`, set

```
u_i = normalize( sqrt(γ) · t_{k(i)}  +  sqrt(1 − γ) · ε_i )
```

with `t_k` orthonormal topic directions and `ε_i` isotropic noise. Then

```
sim(e_i, e_j) = r_i r_j + sqrt(1−r_i²) sqrt(1−r_j²) · cos(u_i, u_j)
```

with `cos(u_i, u_j) ≈ γ` within a topic and `≈ 0` across topics. Redundancy is set by
`γ` alone and **relevance is untouched**, which is the decoupling the design needs.

Note the structural floor `sim ≥ r_i r_j > 0`: similarity can never be negative. This is
a property of real cosine similarity between on-topic documents, matches the v0 corpus
(which is dense and positive), and keeps the ILP model size constant at
`C(25,2) = 300` pair variables in every cell — a useful side effect for timing
predictability.

**Step 6 — validity.**
Because the similarity matrix is built from explicit unit vectors, it is a genuine Gram
matrix: symmetric, PSD, unit diagonal, all entries in `[−1, 1]`. No hand-written matrix
can be assumed to have these properties, and an invalid one would make the benchmark
unrealizable by any encoder.

**Step 7 — saturation mass and budgets.**
Solve the unconstrained problem (ILP with `W_max = Σ w_i`) to get `S_sat`, `W_sat`. Set
the three budgets as `round(ρ · W_sat)` for `ρ ∈ {0.25, 0.60, 1.50}`.

**Step 8 — record.** Every instance stores its factor levels, seed, realized `β̂`
(log-log OLS slope), realized Pearson/Spearman `r(relevance, tokens)`, realized mean
off-diagonal similarity, `W_sat`, the three budgets, and full generator parameters.

---

## 6. Factorial design

**Primary design: 4 × 3 × 2 full factorial = 24 cells.**

```
relevance-length  {A independent, B positive-sublinear, C negative, D adversarial}
      ×  budget   {tight ρ=0.25, medium ρ=0.60, loose ρ=1.50}
      ×  redundancy {low γ=0.15, high γ=0.70}
```

- **40 seeds per (relevance-length × redundancy) cell** → `4 × 2 × 40 = 320` base
  instances.
- Budget is a **within-instance** factor: each base instance is evaluated at all three
  budgets, so budget comparisons are paired on the same corpus.
- **960 evaluated instances**, 5 methods each.

Seeds are a fixed contiguous range, declared before generation, shared across cells so
that the same seed index produces matched-noise instances where possible.

**Supplementary arms** (analysed separately, never pooled into the primary design):

- **Arm S1 — correlation curve.** `ρ_c ∈ {−0.6, −0.3, 0, +0.3, +0.6}` × low/high
  redundancy × medium budget × 40 seeds. Traces the effect as a continuous function of
  the relevance–length coupling instead of four discrete points.
- **Arm S2 — budget curve.** Cell A/high redundancy, `ρ ∈ {0.1, 0.25, 0.4, 0.6, 0.8,
  1.0, 1.25, 1.5, 2.0}` × 40 seeds. Locates the zero-crossing `ρ*` by interpolation.
- **Arm S3 — λ sensitivity (robustness, not tuning).** `objective_lambda ∈ {0, 0.05,
  0.1, 0.2}` on cell A × 3 budgets. `λ = 0` reduces the problem to a *linear* knapsack,
  which is the cleanest possible test of whether the effect is knapsack-structural or
  redundancy-mediated. **All values are reported; none is selected.** The primary design
  uses `λ = 0.1` unchanged.
- **Arm E — realism anchor.** The *unmodified* original text generator, 40 seeds, three
  budgets set by the same `ρ` rule. Tests whether synthetic cell A reproduces the
  original benchmark's behaviour.

---

## 7. Ablation matrix

| Method | Rule | Role | Tests which hypothesis |
|---|---|---|---|
| `greedy_objective` | `argmax Δ_i` | **control** | — |
| `greedy_token_aware` | `argmax Δ_i / w_i` | **treatment** | **H1, H2, H3, H4 — the entire design** |
| `top_k` | `argmax r_i` | context only | value of the redundancy term; *not* the main test |
| `mmr` | `argmax λ_m r_i − (1−λ_m) max_{j∈S} s_ij` | different criterion | external reference; scored on an objective it does not target |
| `ilp` | exact | ground truth | denominator of every optimality gap |

**The primary estimand is the paired difference**

```
Δ_gap = gap(greedy_objective) − gap(greedy_token_aware)
```

on instances where the ILP proved a global optimum. `Δ_gap > 0` means token-awareness
closed part of the gap. The gap form is preferred over the raw score difference because
it normalizes by instance difficulty, which varies across cells by construction; the raw
score difference is reported alongside.

These two methods differ **only** by the `1/w_i` factor, so their paired difference *is*
the causal contribution of token-awareness under the generated distribution. Every other
comparison in the table is context, and `top_k` vs `greedy_token_aware` is explicitly
**not** the main test — it confounds token-awareness with the redundancy term.

---

## 8. Ground-truth strategy

1. **The ILP is ground truth only when `status == PROVEN_OPTIMAL` and
   `scope == FULL_CORPUS`.** The repaired result object enforces this; a time-limited
   incumbent cannot be relabelled.
2. **Instances without a proven optimum are excluded from gap analyses and reported
   explicitly** — count, cell, and budget. They are *not* silently dropped, and they are
   still included in raw-score comparisons, which need no ground truth.
3. **Design for tractability.** `N = 25` with a dense 300-pair model solved in ≤ 7 s on
   the original benchmark and proved optimal on 216/216 instances. A pilot (section 14)
   confirms this before the full run; if any cell shows failures, `N` is reduced for
   that cell and the change recorded, rather than accepting incumbents.
4. **Independent verification.** On a random sample of 20 instances with `N = 25`,
   exhaustive enumeration over the `2^25` subsets is infeasible, so verification uses a
   reduced `N = 18` replica (`2^18 = 262k` subsets, brute-forceable) to confirm the ILP
   returns the true maximum. This validates the solver itself, not just its status flag.
5. **Pre-registered time limit**: 600 s, doubled from the original 300 s to reduce
   non-proven instances; recorded per solve.

---

## 9. Metrics and what each one answers

**Primary hierarchy** (in order; ties broken downward):

| # | Metric | Research question it answers |
|---|---|---|
| 1 | **Optimality gap** `(ILP − method)/|ILP|` | *How much of the achievable objective does each method lose?* The main dependent variable. Normalizes instance difficulty, which the design varies deliberately. |
| 2 | **Objective score** `Score(S)` | The raw quantity, for instances without a proven optimum and as an unnormalized check on (1). |
| 3 | **Relevance captured** `Σ r_i` | Separates the two halves of the objective: is token-awareness buying more relevance, or merely paying less redundancy? |
| 4 | **Redundancy per selected pair** | Redundancy controlled for cardinality. The total is reported but never used alone — the repaired benchmark showed the total is dominated by `|S|`. |
| 5 | **Number of documents selected** | The hypothesised **mediator** of the redundancy interaction (section 3.3). `Δ|S|` between the two greedy methods is the mediation variable for H4. |
| 6 | **Budget utilization** `tokens/W_max` | Descriptive. Confirms the budget levels bind as intended; never a quality measure. |

**Secondary:** coverage (v0 definition, retained for continuity, with its known
weaknesses), realized `β̂`, realized correlations, realized mean similarity.

**Excluded from all claims:** Context Density (deprecated — maximized by near-empty
selections) and latency (not comparable across solver classes).

---

## 10. Statistical analysis

**Unit of analysis.** The base instance (seed × relevance-length × redundancy). Budget
is within-instance, so budget contrasts are paired.

**Pre-registration.** Hypotheses, factor levels, metric hierarchy, and the analysis
below are frozen before any instance is generated.

**Primary analysis — the interaction.** A linear mixed model on the paired difference:

```
Δ_gap  ~  budget_tightness * relevance_length * redundancy  +  (1 | seed)
```

with budget as an ordered within-instance factor. The coefficients of interest are the
**interaction terms**, not the intercept — the design exists to estimate *when* the
effect appears, not merely whether it is non-zero on average. A continuous companion
model uses `ρ` and `β̂` as numeric predictors:

```
Δ_gap  ~  log(ρ) * β̂  +  mean_similarity  +  (1 | seed)
```

**Per-cell analysis** (mirrors the previous step, so results stay comparable): for each
of the 24 cells, `n = 40` paired observations, reporting wins/losses/ties, mean, median,
sd, min, max of the paired difference, Wilcoxon signed-rank `p`, Cohen's `d_z`, and
matched-pairs rank-biserial. **No `p`-value is reported without its effect size and
paired counts.**

**Confidence intervals.** BCa bootstrap over seeds (10,000 resamples) for every cell
mean and for the zero-crossing `ρ*` estimated in arm S2.

**Multiple comparisons.** 24 primary cells → Holm–Bonferroni for family-wise control on
the confirmatory hypotheses, Benjamini–Hochberg reported alongside for the exploratory
cells. Declared in advance.

**Power.** With `n = 40` paired observations per cell: `d_z = 0.5` → power ≈ 0.85;
`d_z = 0.6` → ≈ 0.95; `d_z = 0.8` → > 0.99. The effects observed in the repaired
benchmark were `|d_z| ≈ 1.2–1.7`, so 40 seeds is comfortable for effects of that size
and adequate down to `d_z ≈ 0.5`. Effects smaller than `d_z = 0.4` are explicitly
out of reach and will be reported as underpowered rather than as null.

**Mediation (H4).** Report the cell-wise correlation between `Δ_gap` and `Δ|S|`, and
whether conditioning on `Δ|S|` attenuates the redundancy interaction. Treated as
**descriptive evidence for a mechanism, not a causal mediation claim** — the design
randomizes the factors, not the mediator.

---

## 11. Falsifiable hypotheses

Each is stated with its theoretical basis, its prediction, and what would falsify it.
Two of the four suggested hypotheses are **revised** because, on analysis, the naive
forms are not what theory predicts.

### H1 — Token-awareness helps more as the budget tightens. **(kept)**

*Basis.* Classical knapsack theory: density-greedy has an approximation guarantee for
the 0-1 knapsack; value-greedy has none. The guarantee matters only while the capacity
constraint binds. Once `W_max ≥ W_sat` the constraint is inactive, the problem is
unconstrained subset selection, and dividing by `w_i` can only distort the ranking.

*Prediction.* `Δ_gap` is decreasing in `ρ`, positive for small `ρ`, and `≤ 0` for
`ρ ≥ 1`. A zero-crossing `ρ*` exists in `(0, 1.5)`.

*Falsified if.* `Δ_gap` shows no monotone trend in `ρ`, or is positive at `ρ = 1.5`.

### H2 — **REVISED.** The effect is governed by elasticity `β`, not by the sign of the relevance–length correlation.

The suggested form ("token-awareness helps when relevance and token cost are negatively
correlated") is **not** what theory predicts, and this design does not adopt it. Under
negative correlation the relevance ranking and the density ranking largely *coincide* —
the control already picks short, relevant documents — so the *marginal* contribution of
dividing by `w_i` should be **small**, not large. The naive hypothesis confuses "the
method does well" with "the method's distinguishing feature contributes".

*H2a (magnitude).* `|Δ_gap|` is smallest in condition C (negative correlation), because
the two rankings agree there. Falsified if C shows the largest effect.

*H2b (sign).* The sign of `Δ_gap` tracks `β̂ − 1`: positive where `β̂ < 1`, negative
where `β̂ > 1`, at fixed `ρ`. Falsified if the sign tracks the correlation sign but not
`β̂`, or neither.

### H3 — **REFINED.** Token-awareness hurts when relevance rises *superlinearly* with token cost.

The suggested form ("hurts when high relevance corresponds to high token cost") is
under-specified: condition B has exactly that property with `β < 1`, and theory predicts
token-awareness still *helps* there. Only `β > 1` — where long documents deliver more
relevance *per token* — makes the density ranking actively wrong.

*Prediction.* `Δ_gap < 0` in condition D's superlinear region at tight budgets, while
`Δ_gap > 0` in condition B at the same budget. This is a **crossed** prediction: B and D
both have "relevant documents are long", and the design separates them.

*Falsified if.* B and D behave alike at matched `ρ`.

### H4 — The benefit is attenuated by redundancy, mediated by selection cardinality. **(kept, mechanism added)**

*Basis.* Section 3.3: token-awareness selects more documents; the penalty is quadratic
in `|S|`; so the extra documents are taxed more heavily when `γ` is high.

*Prediction.* `Δ_gap` is smaller at `γ = 0.70` than at `γ = 0.15` at matched `ρ` and
condition (a negative budget × redundancy interaction), and cell-wise `Δ_gap` correlates
negatively with `Δ|S| × γ`.

*Falsified if.* Redundancy has no interaction, or the interaction has the opposite sign,
or `Δ|S|` shows no relationship to `Δ_gap`.

### H5 — The original benchmark's sign flip is explained by budget tightness alone.

*Basis.* The original corpus sits at `β ≈ 0` with high redundancy — i.e. cell
A × high-redundancy of this design.

*Prediction.* Cell A/high-redundancy reproduces the observed pattern (positive at tight,
negative at loose, crossing near saturation), and arm E reproduces it on the real text
generator.

*Falsified if.* Cell A/high does not reproduce it — which would indicate the synthetic
generator misses something the text generator has, and would invalidate the
extrapolation from synthetic cells to the original finding.

### H0 — Null hypothesis, genuinely available

Token-awareness has no systematic effect once budget tightness is controlled; the
original finding was driven by an uncontrolled correlate. Retained if no cell shows
`|d_z| > 0.4` with corrected significance.

---

## 12. Possible outcomes and their interpretations

| Observed pattern | Scientific meaning |
|---|---|
| `Δ_gap > 0` only at tight `ρ`, across **all** relevance–length conditions | Token-awareness is a **knapsack-structural** effect, not a length/value-mismatch exploit. The project's stated motivation ("long passages crowd out short ones") would be the *wrong explanation* for a real effect. The method is worth keeping, the narrative is not. |
| `Δ_gap` sign tracks `β̂ − 1`, roughly independent of `ρ` | The effect is a **value-density** effect. The project's motivation is essentially right but must be restated in terms of elasticity, and the method should be gated on a measurable corpus property. |
| Both: effect requires tight `ρ` **and** `β̂ < 1` | The two mechanisms are **conjunctive**. Token-awareness is a narrow-domain technique, and the honest deliverable is a stated applicability condition, not a general method. |
| `Δ_gap ≈ 0` everywhere once redundancy is controlled | The original effect was an artifact of the specific redundancy structure. The hypothesis would be **not supported**, and the repaired-benchmark result would need reinterpretation. |
| Strong `redundancy × ρ` interaction with cardinality mediation | The effect is really about **how many documents you buy**, and `1/w_i` is an indirect way of controlling cardinality. That would suggest a cardinality-aware method as the better formulation — a different research direction. |
| Condition C shows the **smallest** effect | H2a confirmed; the naive "negative correlation helps" intuition is wrong, and correlation is established as the wrong descriptor. |
| Cell A/high fails to reproduce the original pattern (H5 falsified) | The synthetic generator is missing structure present in real text. All synthetic conclusions become suspect and the design needs revision before any claim is made. |
| Effects present but `|d_z| < 0.4` | **Underpowered, not null.** Reported as such; requires more seeds, not a conclusion. |

Every one of these is a publishable result. None requires token-awareness to win.

---

## 13. Risks and failure modes

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Instances too easy** — all heuristics reach the optimum | No resolution; every cell reads as a tie | Pilot check: require the across-seed sd of `gap(greedy_objective)` to exceed 0.01 in every cell before the full run |
| **Degenerate length variance** — if `w` is near-constant, `Δ_i/w_i ∝ Δ_i` and the two methods are *identical by construction* | Silent null | Assert per-instance coefficient of variation of `w` ≥ 0.3; fail generation otherwise |
| **Loose budget selects everything** | All methods identical at `ρ = 1.5` | Expected and informative, but verify `|S| < N` so the instance is not trivially saturated |
| **Similarity matrix not realizable** | "Unrealistic similarity structure" — no encoder could produce it | Built from explicit unit vectors (§5.2 step 6); assert PSD, symmetry, unit diagonal, range |
| **Synthetic ≠ real** | External validity | Calibrate to measured v0 statistics; arm E as anchor; H5 as an explicit falsification test |
| **Marginal drift across conditions** | Would confound the factor with the marginals | Gaussian copula holds marginals fixed across A–C; D is flagged as structurally different and never pooled with A–C |
| **ILP fails to prove optimality** in high-redundancy cells | Ground truth lost | 600 s limit, pilot timing, explicit reporting, `N` reduction as a last resort — never accept incumbents as ground truth |
| **`W_sat` unstable or ill-defined** | Budget levels incomparable | Computed from a proven-optimal unconstrained ILP; instances where that solve is not proven are regenerated with a new seed offset, and the count is reported |
| **Multiple testing across 24 cells** | False positives | Holm for confirmatory, BH reported alongside, both pre-declared |
| **λ = 0.1 interacts with the redundancy factor** | High-`γ` cells might collapse to tiny selections | Arm S3 characterises this; primary design keeps `λ = 0.1` unchanged and reports `|S|` per cell |
| **Post-hoc factor adjustment** | Would destroy the pre-registration | Factor levels frozen in the design module before generation; any change requires a new document and a re-run |
| **Adversarial level D too easy or impossible** | Uninformative cell | `m_adv` capped so the expensive documents stay individually feasible; pilot verifies both methods select non-trivially |

---

## 14. Computational budget estimate

Derived from measured timings: `N = 25` with a dense 300-pair model, 0.11 s–7.3 s per
ILP on the original benchmark, mean ≈ 2.9 s.

| Item | Count | Estimate |
|---|---|---|
| Base instances | `4 × 2 × 40` | 320 |
| ILP solves (1 saturation + 3 budgets each) | `320 × 4` | **1,280** |
| Heuristic solves | `320 × 3 × 4` | 3,840 (negligible, sub-ms) |
| **Primary design runtime** | at 2–5 s/solve | **0.7 – 1.8 h** |
| Arm S1 (correlation curve) | `5 × 2 × 40 × 1` budget + saturation | ≈ 0.4 h |
| Arm S2 (budget curve) | `40 × 9` + saturation | ≈ 0.3 h |
| Arm S3 (λ sensitivity) | `40 × 3 × 4`, `λ=0` is a fast linear knapsack | ≈ 0.3 h |
| Arm E (realism anchor) | `40 × 4` | ≈ 0.2 h |
| Brute-force ILP verification | 20 instances at `N = 18` | minutes |
| **Total** | | **≈ 2–3 h single CPU** |

Instances are independent, so the run parallelises trivially if needed. A **pilot of 5
seeds per cell (≈ 8 min)** runs first and must pass the section-13 non-degeneracy checks
before the full run is launched.

---

## 15. Implementation plan for the next step

Ordered, with the existing repository untouched except by addition.

1. **`experiments/controlled/generator.py`** — instance synthesis.
   - `ControlledInstanceSpec` dataclass: seed, relevance-length level, `ρ_c`, `γ`, `N`,
     `d`, `K`, marginal parameters, adversarial parameters.
   - `generate_instance(spec) -> ControlledInstance` returning `relevance`,
     `similarity`, `tokens`, plus realized diagnostics (`β̂`, Pearson, Spearman, mean
     similarity, CV of `w`).
   - Validity assertions: Gram-matrix properties, `CV(w) ≥ 0.3`, exact
     `cos(q, e_i) == r_i` to floating tolerance.
   - **Calibration constants derived from the measured v0 corpus are hard-coded with a
     comment recording their provenance.**

2. **`tests/test_controlled_generator.py`** — written *before* the runner.
   - Relevance is exactly as specified; changing `γ` does not move relevance; changing
     `ρ_c` does not move either marginal (two-sample KS against level A); similarity is
     PSD/symmetric/unit-diagonal/in range; token counts are integers with the intended
     CV; condition D produces the intended expensive-and-relevant subset; determinism
     under a fixed seed.

3. **`experiments/controlled/protocol.py`** — the frozen design: factor levels, seeds,
   `ρ` values, metric hierarchy, hypothesis list. Imported by both runner and analysis
   so the two cannot drift apart.

4. **Pilot run** — 5 seeds per cell. Check ILP proof rate, timing, and every
   section-13 non-degeneracy condition. **Report the pilot and stop for review before
   the full run.**

5. **`experiments/controlled/run_controlled.py`** — the full run. Computes `W_sat`,
   derives budgets, runs all five methods, writes `results/controlled/raw_results.json`
   + `.csv` with complete per-instance metadata (seed, all factor levels, realized
   diagnostics, ILP status, git commit, package versions). **Computes no aggregate
   statistic**, so the run cannot be steered by its own output.

6. **`experiments/controlled/analyze_controlled.py`** — mixed model, per-cell paired
   tests, bootstrap CIs, multiplicity correction, mediation descriptives, figures
   (interaction plots with CI bands, `Δ_gap` vs `ρ`, `Δ_gap` vs `β̂`, per-cell forest
   plot).

7. **`docs/controlled_benchmark_results.md`** — results, with the same
   supported / not-supported / untestable structure used in
   `docs/repaired_original_benchmark.md`.

**Unchanged throughout:** the v0 tag and artifacts, the original generator
(`benchmark.py`), the objective, `objective_lambda = 0.1`, `mmr_lambda = 0.5`, the
solver implementations, and the README.

**Reuse, do not fork:** the solvers, `SolveStatus`/`SolveScope`, and the metric
definitions come from the repaired `optimizer.py` and `benchmark.py` as-is. If the
controlled benchmark needs a solver change, that is a finding to report, not a patch to
apply quietly.
