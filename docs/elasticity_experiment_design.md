# Step 6A — Mechanistic Elasticity Experiment: DESIGN (rev. 2, post adversarial review)

**Design only. Nothing was run. No Step 4/5 artifact, generator, optimizer, or existing
benchmark was modified.** Separate from the Step 4 controlled benchmark, which remains
the primary experiment and stays immutable.

**Revision note (6A.1).** Three claims in rev. 1 were wrong or overstated and are
corrected here: (i) the experiment was said to causally distinguish elasticity from
ranking disagreement — **it cannot**; (ii) `σ_r = 1.25` was called the minimum marginal
change — **it is not**; (iii) an inverted-U prediction was read off a τ curve without a
derivation — **now formalized before data collection**. Rev. 1 is preserved outside the
repo as superseded.

## Core question

> Does the token-aware advantage depend on the **dependence structure between relevance
> and token cost**, or is the budget-dependent effect explained by knapsack capacity alone?

Note the question is deliberately phrased around *dependence*, not around β. §B1 explains
why the stronger phrasing is not identifiable.

---

# A. Mathematical construction

## A1. Why β > 1 is impossible with the Step 4 marginals — proof *(unchanged, retained)*

`β̂` is the OLS slope of `log r` on `log w`:

```
β̂ = Cov(log w, log r) / Var(log w) = corr(log w, log r) · sd(log r) / sd(log w)
```

By Cauchy–Schwarz `|corr| ≤ 1`, hence

```
|β̂|  ≤  sd(log r) / sd(log w)                                        (★)
```

with equality only if `log r` is an exact affine function of `log w`. Both marginals are
fixed, so both sd's are fixed and **(★) is a hard ceiling no coupling can exceed.**
Step 4: `0.4579 / 0.5000 = 0.9158 < 1`. Verified at the comonotone limit:
`ρ_c = 0.90 → β̂ = 0.809`, `0.99 → 0.890`, `1.00 → 0.901`.

**β > 1 is therefore a distributional impossibility, not a generator limitation.** Some
marginal must change, and (★) names exactly the quantity that must move.

## A2. Which marginal — and why not the obvious one *(unchanged, retained)*

**Lowering `sd(log w)` is rejected.** It breaks the locked token-CV guard (headroom
~0.015) and is scientifically self-defeating: as `w → constant`, `Δ_i/w_i ∝ Δ_i` and the
two methods become identical **by construction**. Shrinking the dispersion under study
destroys the mechanism.

**Raising `sd(log r)` is adopted.** It leaves the token marginal, the token-CV guard and
the redundancy construction untouched.

## A3. `σ_r = 1.25` is a **choice**, not a mathematical minimum — CORRECTED

Rev. 1 described `σ_r = 1.25` as the minimum marginal change. **That was wrong.**

From (★), reaching `β = 2` requires `ρ_c · σ_r / σ_w ≥ 2`, i.e.

```
σ_r  ≥  2 σ_w / ρ_c  =  1.00 / ρ_c
```

so the **true mathematical minimum is `σ_r = 1.00`, attained only at `ρ_c = 1`.** There
is a whole continuum of valid choices, trading marginal change against endpoint
degeneracy:

| `σ_r` | `ρ_c` needed for β = 2 | residual variance `1 − ρ_c²` | note |
|---|---|---|---|
| **1.00** | **1.000** | **0.000** | **mathematical minimum — DEGENERATE: `r` becomes a deterministic function of `w`, zero scatter, no instance-to-instance variation in the r–w relation** |
| 1.11 | 0.901 | 0.188 | very little scatter |
| **1.25** | **0.800** | **0.360** | **chosen** |
| 1.43 | 0.699 | 0.511 | larger marginal change |
| 1.67 | 0.599 | 0.641 | larger still |

**Why 1.25 was chosen:** it is the smallest value in this family that keeps a
*substantial* residual scatter at the β = 2 endpoint (36% of variance unexplained) while
holding the marginal inflation to the smallest workable level. For calibration, Step 4's
most extreme level was `|ρ_c| = 0.5` (residual 0.75), so `ρ_c = 0.8` is already more
extreme than anything previously run — pushing toward `σ_r = 1.00` would make the
endpoint nearly deterministic and would not be a meaningful experimental condition.

**This is a judgement call and is recorded as such**, not presented as forced by the
mathematics. The only forced statement is `σ_r ≥ 1.00`.

## A4. The construction *(unchanged, retained)*

Simply inflating `σ` on Step 4's Beta marginal would inherit its distortion (Step 4
measured `ρ_c = 0.5 → β̂ = 0.405`, a 19% shortfall). **Make both logs Gaussian:** the pair
is then jointly Gaussian, the log-log regression is exactly linear, and
`β̂ = ρ_c · σ_r/σ_w` **exactly**.

```
ζ, η  ~  N(0,1)  with  corr(ζ, η) = ρ_c                 (Gaussian copula)
r_i   =  clip( exp(μ_r + σ_r·ζ_i),  1e-3,  0.95 )       μ_r = log 0.05,  σ_r = 1.25
w_i   =  clip( round(exp(μ_w + σ_w·η_i)), 8, 400 )      μ_w = log 77,    σ_w = 0.50  [UNCHANGED]
```

`σ_r/σ_w = 2.50`, so **β = 2.5 · ρ_c**; β = 1 sits at `ρ_c = 0.4`. Embeddings, topics,
redundancy, budgets, objective and both λ are unchanged from the locked Step 4 build.

**Verified** (200k draws/level, ~1.0% clipping):

| `ρ_c` | target β | realized β̂ |
|---|---|---|
| −0.20 | −0.500 | −0.496 |
| 0.00 | 0.000 | +0.012 |
| +0.20 | +0.500 | +0.493 |
| +0.40 | +1.000 | +0.989 |
| +0.60 | +1.500 | +1.488 |
| +0.80 | +2.000 | **+1.978** |

Both marginals are **identical across every β level**; only `ρ_c` changes.

---

# B. Identification — what this design can and cannot establish

## B1. β and ranking disagreement are NOT independently manipulated — CORRECTED

Rev. 1 framed E (elasticity) and R (ranking disagreement) as competing hypotheses of
which "at most one survives". **That claim is withdrawn.**

The design manipulates exactly **one** variable: the copula correlation `ρ_c`. With
marginals fixed, both candidate mediators are deterministic functions of it:

```
β(ρ_c)          = 2.5 · ρ_c                    monotone, linear
Disagreement(ρ_c) = 1 − τ_b(r, r/w)            non-monotone, peaks near ρ_c ≈ 0.4
```

A one-dimensional manipulation **cannot causally separate two quantities that are both
functions of it.** Any mediator that is a bijection of `ρ_c` is observationally
equivalent to β.

**What IS causally identified:** the effect of `ρ_c` — the r–w dependence structure — on
`Δ_gap`. `ρ_c` is set by design, so this is a genuine randomized manipulation.

**What is NOT identified:** whether that effect "operates through β" versus "through
ranking disagreement" versus through anything else co-varying with `ρ_c`. These are
competing *descriptions of one causal pathway*, not separable causes.

**Partial, associational separation is available**, because disagreement is *non-monotone*
in `ρ_c` while β is monotone — so they are not collinear at the between-level scale, and
their shapes differ. Within a level they also vary across instances. Measured within-level
correlation between realized `β̂` and realized disagreement:

| `ρ_c` | −0.2 | 0.0 | +0.2 | +0.4 | +0.6 | +0.8 |
|---|---|---|---|---|---|---|
| corr(β̂, disagreement) | +0.57 | +0.39 | +0.14 | −0.21 | −0.57 | **−0.79** |

Well below 1 everywhere, so both can be entered jointly — **but the correlation is
substantial and flips sign across levels, so this separation is weak, level-dependent,
and strictly associational.** It is pre-registered as *secondary*, never as identification.

**What would actually identify it:** a two-dimensional manipulation in which β and
disagreement vary independently — e.g. copula families with different rank structure
(tail dependence) chosen to hold β fixed while moving disagreement. That changes the
copula family and introduces its own confound, and is **out of scope here**; it is the
natural follow-up if this experiment shows a strong `ρ_c` effect.

## B2. Confounds introduced by the marginal change *(retained)*

| confound | magnitude | handling |
|---|---|---|
| Relevance dispersion | `sd(log r)`: 0.458 → 1.250 | **Fixed across all β levels**, so it never confounds the within-experiment contrast. Absolute effects are **not comparable to Step 4**. |
| Effective corpus size | docs with `r > 0.05`: mean 12.4/25, min 6 | Constant across levels; recorded; pilot guards a minimum |
| Similarity floor `r_i r_j` | lower (many small `r`) | redundancy construction unchanged; realized similarity recorded |
| Objective scale | will differ from Step 4's ≈2.8 | primary estimand in absolute units; normalizations labelled secondary |
| Clipping | ~1.0%, balanced across levels | recorded |
| **Residual scatter varies with level** | `1−ρ_c²`: 0.96 → 0.36 | **inherent (§B1)**; matched-\|ρ_c\| pairs (±0.2, ±0.8) share scatter exactly |

**Unchanged from the locked Step 4 protocol:** token marginal, token-CV guard, `γ`, `K`,
`N`, `d`, embedding construction, redundancy construction, budget rule, objective,
`λ_obj`, `λ_mmr`, solvers. `w` is drawn from its own latent, independent of `ρ_c`, so any
token-side guard rejection is **balanced across β levels by construction**.

---

# C. Formalizing R *before* the data — NEW

Rev. 1 read an inverted-U off a τ curve and asserted it as a `Δ_gap` prediction. That is
not a derivation. R is formalized here, with its rationale, metric and falsifier fixed in
advance.

## C1. Primary disagreement metric

```
D_i  =  1 − τ_b( r , r/w )        computed per instance over all N = 25 documents
```

`τ_b` is Kendall's tau-b (ties handled). `D ∈ [0, 2]`, in practice ≈ [0.1, 0.3].

## C2. Theoretical rationale — a bound, not a shape

The two methods differ **only** in their priority function: `greedy_objective` orders by
`Δ_i`, `greedy_token_aware` by `Δ_i / w_i`. At the first selection step these reduce to
`r_i` and `r_i/w_i`.

> **If the two priority orders coincide on every comparison the algorithms ever make,
> the two selections are identical and `Δ_gap = 0` exactly.**

`D` is a scalar summary of that ordering agreement. This yields a **one-directional
implication only**:

```
D → 0   ⟹   |Δ_gap| → 0
```

It does **not** imply that large `D` produces large `|Δ_gap|`, and it says **nothing about
the sign** of `Δ_gap`. R is therefore a hypothesis about the *magnitude envelope*, not
about direction.

Caveat stated up front: with `λ_obj > 0` the priorities are recomputed each iteration with
the redundancy penalty, so the static `D` is a **proxy** for the whole selection
trajectory, not an exact invariant. Its adequacy is itself an empirical question.

## C3. Exact preregistered prediction for R

R is tested primarily **within β levels**, where the level manipulation is held constant:

> **R-primary:** at the tight budget, within each β level, `|Δ_gap|` has a **positive**
> association with the instance's realized `D`. Pre-registered as the pooled
> within-level slope of `|Δ_gap|` on centred `D` (level fixed effects, `(1|seed)`).

The between-level shape is a **secondary, weaker** prediction: since `D` peaks near
`ρ_c ≈ 0.4`, the level-mean `|Δ_gap|` profile should be **single-peaked rather than
monotone** in `ρ_c`.

## C4. Why a quadratic term is admissible

The quadratic contrast is **not** chosen because it fits an expected shape. It is
admissible because it is the **lowest-order orthogonal polynomial contrast that can
distinguish the three candidate profiles**, all of which were specified before data
collection:

- **C (capacity only)** ⇒ linear = 0 and quadratic = 0 (flat)
- **E (elasticity)** ⇒ linear < 0, quadratic ≈ 0 (monotone decline, sign change)
- **R (disagreement envelope)** ⇒ quadratic < 0 (single peak), no sign change required

Linear and quadratic contrasts over the 6 equally spaced `ρ_c` levels are pre-specified
as orthogonal polynomial coefficients. No higher-order term is fitted, and no model is
selected after seeing the data.

## C5. What counts against R

1. No within-level association between `|Δ_gap|` and `D` at tight (R-primary null).
2. `Δ_gap` significantly **negative** at β = 2.0 — a sign change R does not predict.
3. A monotone profile with the quadratic contrast indistinguishable from zero.

---

# D. Factorial design *(retained)*

**6 β levels × 3 budgets × 2 redundancy × 60 seeds** = 720 base instances,
2160 instance-budgets, ~2880 ILP solves, est. **30–45 min**.

| factor | levels | status |
|---|---|---|
| **β via `ρ_c`** | −0.5, 0, +0.5, +1.0, +1.5, +2.0 (`ρ_c` = −0.2 … +0.8) | **confirmatory** |
| **Budget** | tight 0.25, medium 0.60, loose 1.50 (within-instance) | **confirmatory** |
| **Redundancy** | low `γ=0.15`, moderate `γ=0.70` | **robustness only — NOT powered for interaction** |
| Held fixed | `N=25`, `K=5`, `d=64`, `λ_obj=0.1`, `λ_mmr=0.5`, budget rule, objective | — |
| Methods | `greedy_objective`, `greedy_token_aware` (primary pair), `top_k`, `mmr`, `ilp` | — |

No condition D (superseded — it was an extreme β<1 case). No arm E (no anchor needed for
a mechanistic question).

---

# E. Exact confirmatory tests — NEW, fully specified

**One family, one aggregation level.** Unit = base instance; budget is within-instance;
pooled over redundancy; paired by seed. **Holm across all 10.** No pairwise β comparisons.

| # | estimand | data subset | aggregation | H₀ | H₁ | family |
|---|---|---|---|---|---|---|
| **T1** | profile of `Δ_gap` across the 6 β levels | tight budget, all redundancy | instance-level, Friedman (paired on seed) | all 6 level distributions equal | at least one differs | F1 |
| **T2** | mean `Δ_gap` | tight, β = +1.0 | one-sample Wilcoxon | `Δ_gap = 0` | `≠ 0` | F1 |
| **T3** | mean `Δ_gap` | tight, β = +1.5 | one-sample Wilcoxon | `Δ_gap = 0` | `≠ 0` | F1 |
| **T4** | mean `Δ_gap` | tight, β = +2.0 | one-sample Wilcoxon | `Δ_gap = 0` | `≠ 0` | F1 |
| **T5** | **equivalence** of `Δ_gap` to 0 | tight, β = +2.0 | TOST, margin ±m (§F) | `\|Δ_gap\| ≥ m` | `\|Δ_gap\| < m` | F1 |
| **T6** | **linear** orthogonal contrast over the 6 levels | tight | contrast on instance values | coefficient = 0 | `≠ 0` | F1 |
| **T7** | **quadratic** orthogonal contrast over the 6 levels | tight | contrast on instance values | coefficient = 0 | `≠ 0` | F1 |
| **T8** | within-level slope of `\|Δ_gap\|` on centred `D` | tight, all levels | mixed model, level fixed effects + `(1\|seed)` | slope = 0 | slope > 0 (R-primary) | F1 |
| **T9** | profile of `Δ_gap` across the 6 β levels | **loose** budget | Friedman | all equal | at least one differs | F1 |
| **T10** | **quadratic** contrast | **medium** budget | contrast | coefficient = 0 | `≠ 0` | F1 |

**Everything else is exploratory** and reported as such: redundancy breakdowns, per-level
tests at medium/loose, `top_k`/`mmr` comparisons, mechanism diagnostics, matched-|ρ_c|
sign contrasts, any joint β̂ + `D` model.

**Mapping tests to hypotheses:**

- **C is falsified** by T1 (or T9) rejecting — variation across β at fixed budget. *This
  is the one clean, identified inference the design supports.*
- **E is falsified** by T4 showing a significantly **positive** `Δ_gap` at β = 2.0.
- **R is weakened** by T8 null, or by T4 significantly **negative**, or by T7 null with
  T6 significant.
- **E vs R is NOT adjudicated causally** (§B1). T6/T7/T8 compare *descriptions* of the
  `ρ_c` effect; they cannot attribute it to one mediator.

---

# F. TOST margin — justified on scientific grounds, not power — CORRECTED

Rev. 1 asserted ±0.05 without a scientific anchor. Replaced by an anchored, pre-registered
rule.

**Anchor:** a difference smaller than the better method's **own residual distance to the
optimum** cannot matter practically — it lies inside "how close to optimal this heuristic
is anyway".

```
m  =  median absolute optimality gap of greedy_token_aware at the tight budget,
      estimated from the PILOT, rounded to 2 decimals
```

Fallback if the pilot estimate is unstable (IQR > 2× median): `m = 0.03`, which is the
magnitude of the loose-budget reversal already characterised in Step 4 as "reliable but
practically negligible" (−0.031). In Step 4 the corresponding token-aware gap was ≈ 0.03,
so both routes land near the same value.

**Honest power consequence — stated, not hidden.** With `m = 0.03` and `sd(Δ_gap) = 0.10`,
60 seeds gives **TOST power ≈ 0.50**, not 0.80; ≈ 95 seeds would be needed for 0.80. With
`m = 0.05` the same design gives ≈ 0.97 — but 0.05 is not scientifically anchored, and
**choosing the margin for its power would invert the logic of the test.**

**Pre-registered rule:** the margin is set by the anchor above. If the pilot's `sd`
implies TOST power < 0.80 at that margin, then **either** seeds are raised to the number
achieving 0.80 (decided before the main run), **or** T5 is reported as
**underpowered/inconclusive**. A non-significant TOST will **never** be reported as
evidence of equivalence.

---

# G. Power *(retained, with the T5 caveat above)*

60 seeds, paired, α = 0.05 two-sided: `d_z = 0.3 → 0.64`; **`0.4 → 0.87`**;
`0.5 → 0.97`; `0.6 → >0.99`. Step 4's tight effects were `d_z ≈ 0.87–1.43`.

**Redundancy interaction is explicitly not powered** (Step 5 effects `d_z ≈ 0.19–0.45`
would need ~80–200 seeds). Carried as robustness, reported descriptively.

---

# H. Pilot *(retained, extended)*

**5 seeds × 6 β × 2 redundancy × 3 budgets = 180 instance-budgets, ≤ 10 min.**
Verification only, never inference.

| # | check | criterion |
|---|---|---|
| 1 | β reaches intended regimes | per-level mean β̂ within **0.15** of target (declared before measuring; `sd(β̂)` is 0.31–0.57 here vs 0.15–0.19 in Step 4) |
| 2 | **β > 1 genuinely reached** | at `ρ_c = 0.8`, ≥ 95% of instances have β̂ > 1 (simulation: 100%) |
| 3 | relevance marginal comparable across levels | pairwise KS `p > 0.01` |
| 4 | token marginal comparable across levels | pairwise KS `p > 0.01` |
| 5 | token CV above the locked guard | ≥ 0.30; rejection rate ≤ 1%, balanced across β |
| 6 | redundancy controlled | `γ` moves similarity; leaves `r`, `w`, β̂, topics bit-identical |
| 7 | budget ordering | tight < medium < loose; realized ρ within 0.02 |
| 8 | ILP tractable | 100% proven optimal; q95 < 5 s |
| 9 | no degeneracy | ≥ 4 docs with `r > 0.05`; ≥ 2 selected by both methods at tight |
| **10** | **`D` is non-degenerate and varies within level** | within-level `sd(D) > 0.02` (simulation: 0.058–0.071) |
| **11** | **TOST margin estimable** | median token-aware gap at tight, with IQR, to fix `m` per §F |

**Token-CV exclusion policy (pre-declared):** a rejected instance is **recorded and
excluded, never resampled**; realized n reported per cell. Simulated rejection rate
**0.255%** (~1.8 of 720), balanced across β because `w ⊥ ρ_c`.

---

# I. Exact changes relative to Step 4 *(retained, corrected)*

| item | Step 4 | Step 6 | why |
|---|---|---|---|
| relevance marginal | Beta(2,3) on [0.05,0.60], `sd(log r)=0.458` | log-normal, `μ_r=log 0.05`, **`σ_r=1.25` (a choice, min is 1.00)**, clipped [1e-3, 0.95] | required by (★) |
| β mapping | `β̂ ≈ 0.91·ρ_c`, 19% shortfall | **`β = 2.5·ρ_c` exact** | joint log-normality |
| β levels | 3 (+D) | **6**, −0.5 … **+2.0** | crosses β = 1 |
| seeds | 40 | **60** (possibly raised for T5, §F) | supports equivalence testing |
| condition D / arm E | present | **removed** | superseded / not needed |
| β tolerance | 0.10 | **0.15**, declared before measuring | `sd(β̂)` rises with `σ_r` |
| correction family | mixed aggregation, 12 | **single level, 10, fully enumerated (§E)** | fixes the Step 5 finding |
| unchanged | token marginal, CV guard, `γ`/`K`/`N`/`d`, embeddings, redundancy, budget rule, objective, both λ, solvers | | |

**Process:** own module, own pre-flight, own lock, mirroring 3B/3C/3C.1. Must **not** be
patched into `experiments/controlled/generator.py`, which is locked and owns Step 4.
