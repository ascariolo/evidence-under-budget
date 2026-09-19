# Step 6B.1 — Design-repair analysis (no protocol change made)

**Analysis only.** No experimental code was edited, the full experiment was not run,
confirmatory seeds 0–59 were not touched, and no seed was added or selected. Alternatives B
and C were evaluated by re-solving the **five pre-declared pilot seeds (1000–1004)** at
candidate budgets, using `experiments/elasticity/generator.py` read-only. Only
identifiability quantities were computed; **no per-β Δ_gap mean or sign was computed or
inspected.** Numbers: `results/elasticity/design_repair_analysis.json`.

Every estimate below rests on 60 instances from 5 seeds and is correspondingly noisy
(bootstrap intervals given).

---

## 1. Methodological diagnosis

Four problems looked coupled. On inspection, they have **different causes**, and only one of
them is caused by the budget definition.

1. **G4 is an invalid guard, not a failed benchmark.** It excludes instances according to
   how many documents the *compared methods* select, which conditions on post-treatment
   behaviour. At high β, a value-greedy ranker buying one long document *is* the mechanism
   under study — and 11 of the 12 violations are at β ≥ 0.5. G4 would be invalid under any
   budget definition. Measured against an outcome-neutral property ("a packing decision
   exists": the two cheapest documents fit), the tight regime is non-degenerate in **100%**
   of instances under every alternative.
2. **Zero inflation is structural, not a tight-budget artefact.** Exact Δ_gap = 0 occurs in
   67% of instances at ρ = 0.25, but still 55% at 0.30, 48% at 0.40, 38% at 0.45 — and
   **47% at the medium budget** (0.60). With heavy-tailed relevance (~12.5 usable documents
   of 25) the two greedy rankings coincide often at every budget. No budget definition
   removes it; it must be handled analytically. At ρ = 0.25, `greedy_objective` is
   *exactly* optimal in **88%** of instances.
3. **The TOST margin is undefined, not merely badly estimated.** The better method is
   usually exactly optimal, so any gap-anchored margin collapses to zero. More
   fundamentally, T5 tests no hypothesis's distinctive prediction (§5).
4. **The seed count is indeterminable from this pilot.** The quantity that sets n is the
   seed-level variance under zero inflation (§6). Its 5-seed bootstrap interval at ρ = 0.25
   is 0.014–0.117, which moves the required n by a factor of ~70.

The only budget-caused issue is **one-document optima** (12% at ρ = 0.25). Their rate may
rise with β (0.0 at β = −0.5 to 0.2 at β = 1.0 and 2.0), but from 10 instances per level
that is within ~1.5 SE and is not established.

---

## 2. Alternatives

- **A** — keep tight ρ = 0.25; treat G4 and zero inflation as structural.
- **B** — raise tight ρ, keeping `ρ = W_max / W_sat` (grid 0.30 / 0.35 / 0.40 / 0.45).
- **C** — define the budget from the token vector alone, `W_max = k · median(w)`
  (k = 2–5). Under IMPL-1, `w` is bit-identical across β and γ, so the **absolute** budget
  is identical across all β levels of a seed.
- *(C2, not computed)* — choose `W_max` so that the ILP optimum holds ≥ k documents.
  Rejected on principle: the budget would be conditioned on the reference solution,
  bindingness would vary with β, and the instance definition becomes circular.

### Comparison (pilot seeds 1000–1004; 60 instances per definition)

| | A 0.25 | B 0.30 | B 0.40 | B 0.45 | C k=3 | C k=4 | medium 0.60 |
|---|---|---|---|---|---|---|---|
| ILP optimum = 1 doc | **12%** | 2% | 2% | 0% | 0% | 0% | 0% |
| G4 violations / 120 | 12 | 3 | 3 | 1 | 0 | 0 | 0 |
| two cheapest docs fit (outcome-neutral) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| exact Δ_gap = 0, instance | **67%** | 55% | 48% | 38% | 58% | 43% | 47% |
| non-zero instances | 20 | 27 | 31 | 37 | 25 | 34 | 32 |
| exact 0, seed-level (redundancy-pooled) | 53% | 37% | 27% | 7% | 50% | 40% | 27% |
| non-zero seed×β cells / 30 | 14 | 19 | 22 | 28 | 15 | 18 | 22 |
| `greedy_objective` exactly optimal | **88%** | 72% | 67% | 55% | 78% | 70% | 58% |
| mean ILP cardinality | 2.4 | 3.0 | 3.8 | 4.0 | 3.2 | 3.7 | 5.1 |
| seed-level sd (point) | 0.070 | 0.077 | 0.049 | 0.046 | 0.027 | 0.095 | 0.042 |
| seed-level sd, bootstrap 95% | 0.014–0.117 | 0.024–0.133 | 0.032–0.069 | 0.025–0.076 | 0.014–0.042 | 0.028–0.150 | 0.019–0.067 |
| π (divergence), bootstrap 95% | 0.18–0.52 | 0.32–0.58 | 0.37–0.67 | 0.52–0.72 | 0.32–0.55 | 0.37–0.77 | 0.43–0.63 |
| mean absolute W across β | 153→244 | — | 245→391 | — | **233 const.** | — | — |
| **bindingness W/W_sat across β** | **0.25 const.** | 0.30 const. | 0.40 const. | 0.45 const. | **0.41→0.25** | 0.54→0.33 | 0.60 const. |
| tight : medium budget ratio | 0.42 | 0.50 | 0.67 | 0.75 | varies | varies | — |

**The sd estimates are not monotone in the budget** (B: 0.070, 0.077, 0.072, 0.049, 0.046;
C: 0.028, 0.027, 0.095, 0.080). That is noise from 5 seeds, and it means none of these
variance estimates can rank the alternatives by power.

### Required seeds — single-level test, power 0.80, SESOI as a fraction of the mean ILP optimum

| def | SESOI | n (α = .05) | n (Holm, α = .05/9) | n over the sd 95% interval (Holm) |
|---|---|---|---|---|
| A 0.25 | 1% | 763 | 1270 | 49 – 3557 |
| A 0.25 | **3%** | 85 | **142** | **6 – 396** |
| A 0.25 | 5% | 31 | 51 | 2 – 143 |
| B 0.30 | 3% | 87 | 144 | 14 – 428 |
| B 0.40 | 3% | 25 | 41 | 18 – 82 |

Linear contrasts need roughly 1.4× these numbers and quadratic contrasts roughly 1.0×,
assuming zero between-level correlation (conservative). If the one-sample test discarded
zeros, as the current Wilcoxon does, n would inflate by a further factor of 1/(1 − share of
exact zeros) — about 2.1× at A.

---

## 3. Adversarial assessment

### A — keep ρ = 0.25

- **For:** changes no locked factor, keeps the Step 4 ρ grid, and involves no post-pilot
  tuning. Its only "failure" (G4) comes from an invalid guard.
- **Against:**
  1. The highest zero inflation of any option, so the signal lives in roughly a third of
     instances.
  2. The tight problem is nearly discrete (ILP holds 2.4 documents; `greedy_objective` is
     exactly optimal 88% of the time), so it may not be the *packing* regime that
     token-awareness theory addresses.
  3. **Strongest objection:** the one-document-optimum rate may rise with β (0.0 → 0.2).
     If real, the *type* of decision at tight changes with β — a confound in the β
     comparison.
  4. Worst expected power per seed.

### B — raise tight ρ

- **For:** keeps bindingness constant across β. Largely removes one-document optima at
  0.30 (12% → 2%).
- **Against:**
  1. **It changes a locked factor after seeing pilot data.** Its main motivation was G4,
     which is invalid as a guard; with that motivation gone, what remains is power, and
     choosing a budget for power means choosing on outcome variance.
  2. At the minimal 0.30, nothing else measurably improves: sd 0.077 vs 0.070 (inside
     noise), zeros 55% vs 67%.
  3. The larger values (0.40–0.45) look better on power, but they compress the
     tight–medium contrast (ratio 0.42 → 0.67–0.75) toward the region where Step 4 found
     the budget effect weakest, and they are exactly the kind of choice this analysis must
     not make on favourability.
  4. It breaks the ρ-grid link to Step 4. That link is already weakened — the relevance
     marginal changed, so ρ = 0.25 does not mean the same number of documents as in
     Step 4 — but it is not gone.
  5. The absolute budget still grows ~60% across β (153 → 244 at A; 245 → 391 at 0.40),
     as in A.

### C — token-only budget

- **For:** outcome-blind, simple, and the absolute budget is identical across β within a
  seed.
- **Against, decisively:** bindingness then **varies systematically with β**
  (k = 3: W/W_sat 0.41 → 0.25; k = 4: 0.54 → 0.33). Step 4 established bindingness as the
  **dominant** driver of Δ_gap, so C would confound the manipulated factor with the
  strongest known cause. **C is rejected on identifiability grounds.**

**The unavoidable trade-off:** because `W_sat` is itself a function of the r–w dependence,
no budget definition can hold both absolute budget and bindingness constant across β. A
and B hold bindingness fixed and let the absolute budget vary; C does the reverse. Holding
the known dominant driver fixed is the right choice, which rules out C.

---

## 4. G4

**Retire G4 as a guard under every alternative** and report its statistics descriptively:
the n_selected distribution for each method at tight, one-document optima by β, and ILP
cardinality by β.

Replace it with an outcome-neutral non-degeneracy guard:

> **G4′:** the two cheapest documents fit within `W_max` at every budget level.

G4′ passed on 100% of pilot instances under every alternative. It guarantees a packing
decision exists without conditioning on what either compared method does.

---

## 5. TOST — retire it

TOST is not an appropriate confirmatory analysis for this experiment. The reasons, in order
of weight:

1. **No hypothesis predicts equivalence to zero at β = 2.0.** E predicts a *negative*
   Δ_gap, C a positive one, and R a positive value decaying toward zero — "small", not
   "equivalent to zero". An equivalence test answers a question none of the three
   hypotheses poses. E remains falsifiable through T4's sign.
2. **The point mass at zero makes equivalence partly trivial.** Δ_gap ≡ 0 whenever the
   two selections coincide, and at tight `greedy_objective` is exactly optimal 88% of the
   time. An "equivalence" finding would largely measure how often the methods pick the
   same set — a property of the benchmark — rather than whether token-awareness has a
   negligible effect when it acts.
3. **No margin exists that is not a free value judgement.** The pilot anchor collapses to
   zero, so the confirmatory conclusion would be determined by whatever margin is chosen.
4. **It is infeasible:** ~293 seeds at the pilot sd, with a margin already shown to be
   arbitrary.

**Replacement:** no confirmatory test. Report the **95% interval for mean Δ_gap at each β
level** as exploratory estimation, alongside the SESOI band from §6 as an interpretive
reference.

Retire `results/elasticity/tost_margin.json` by amendment — mark it retired, **do not
delete or overwrite it**. K1–K3 are retired with it.

---

## 6. Power and seed count under zero inflation

The instance-level Δ_gap is a mixture: a point mass at 0 with probability 1 − π, and a
non-zero component X₁. Its variance is

```
Var_inst = π · E[X₁²] − μ²
```

After pooling the two redundancy levels, which share r, w and topics:

```
Var_seed = Var_inst · (1 + ρ_red) / 2
```

and for contrasts, the between-level correlation ρ_lvl enters as well.

**The seed count must be set by the smallest effect size of interest (SESOI) against
`Var_seed`**, taking the maximum required n over the confirmatory tests at the
Holm-adjusted α. It must **not** come from extrapolating the pilot's observed
standardized effect: that would be both effect-dependent and, at 5 seeds, useless.

Three things follow:

1. **This pilot cannot set n.** At A with a 3% SESOI, the Holm-adjusted n is 142 at the
   point estimate but anywhere from 6 to 396 across the bootstrap interval of the sd.
   ρ_red is equally unstable (0.86 at A, −0.17 at B 0.45).
2. **Zero handling changes the estimand.** `wilcoxon(zero_method="wilcox")` drops zeros,
   so it effectively tests the non-zero component and inflates n by 1/(1 − share of
   zeros). The pre-registered estimand is the **mean** Δ_gap *including* zeros, so the
   tests should use a **sign-flip permutation test on seed-level means**. That keeps zeros
   as data and targets the mean directly.
3. **The SESOI is a value judgement and must be declared, not estimated.** The project
   already fixed a scale in Step 4: about 1% of the optimum was called "practically
   negligible" and 3.4% the smallest effect called real. **Recommended SESOI: 3% of the
   mean ILP optimum at each budget.** The optimum is strictly positive in this generator
   (it is at least the best single document's relevance), so the relative SESOI is well
   defined. This choice needs sign-off; required n scales with 1/SESOI².

---

## 7. Recommended design revision

**Keep the ρ-relative formulation and keep tight ρ = 0.25 (alternative A), repair the
analysis, add a pre-registered exploratory sensitivity level at ρ = 0.30, and determine n
from a blinded variance-component pilot.**

Reasons:

1. **C is ruled out** because it confounds β with bindingness, the dominant known driver.
2. **B's main motivation (G4) is a guard defect, not a benchmark defect.** Its only
   measurable gain at the minimal level is removing one-document optima. Its power gains
   appear only at 0.40–0.45, would be chosen on outcome variance, and compress the budget
   contrast.
3. **The real problems — zero inflation, the missing margin, indeterminate n — are
   analytic** and persist under every budget definition, including medium.
4. **A's strongest weakness** is the possible rise of one-document optima with β. It is
   addressed without replacing a locked factor through an *exploratory* ρ = 0.30
   sensitivity level: same instances, one extra budget, never in the Holm family. If the
   β profile at 0.25 and 0.30 disagrees, that is reported as a limitation.

**Explicitly not recommended:** choosing between 0.30, 0.40 and 0.45 by required n. That is
the favourability selection this analysis must avoid.

---

## 8. Exact protocol changes required (as a pre-registered amendment, before any confirmatory data)

| # | item | current | amended |
|---|---|---|---|
| 1 | **G4** | locked guard on heuristic cardinality | **descriptive diagnostic only** |
| 2 | **G4′** | — | new locked guard: two cheapest docs ≤ `W_max`, every budget |
| 3 | **T5** | TOST at β = 2.0 | **retired**; exploratory 95% interval for mean Δ_gap per β level, with SESOI band |
| 4 | **K1–K3** | margin rule and TOST power | **retired**; `tost_margin.json` marked retired, not deleted |
| 5 | **confirmatory family** | 10 tests, Holm | **9 tests** (T1–T4, T6–T10), Holm over 9 |
| 6 | **T2–T4, T6, T7, T10 statistic** | one-sample Wilcoxon, zeros dropped | **sign-flip permutation test on seed-level mean Δ_gap**, zeros retained (10,000 flips, fixed RNG seed declared) |
| 7 | **T1, T9** | Friedman | unchanged; ties from zero-inflation documented as reducing power |
| 8 | **T8** | slope of mean \|Δ_gap\| on centred D | **divergence probability on centred D** (share of the two redundancy levels whose selections differ, mixed model with level fixed effects + (1\|seed)), one-sided > 0. Rationale: the design's own §C2 bound concerns *coincidence of selections*; \|Δ_gap\| mixes frequency and magnitude. *This is a post-pilot change and must be signed off as such.* |
| 9 | **secondary (exploratory) reporting** | — | hurdle decomposition per β and budget: P(divergence) and conditional Δ_gap given divergence; one-document-optimum rate and ILP cardinality by β |
| 10 | **sensitivity level** | — | ρ = 0.30 at tight, exploratory only, never in the Holm family |
| 11 | **SESOI** | — | 3% of the mean ILP optimum at each budget; declared, needs sign-off |
| 12 | **seed count** | 60 fixed | set by §9 step 2: max over the 9 tests of required n at SESOI, power 0.80, α = .05/9, from `Var_seed`; **pre-declared cap of 300 seeds**, above which the verdict is REDESIGN rather than a smaller n |
| 13 | **confirmatory seeds** | 0–59 | **0 … n−1**; unchanged start, extended only by that rule |

---

## 9. Must remain frozen

- Generator construction: log-normal relevance, `σ_r = 1.25`, `σ_w = 0.50`, `μ_r`, `μ_w`,
  clipping, β = 2.5·ρ_c, **IMPL-1** (tokens as the own latent).
- β levels {−0.5, 0, 0.5, 1.0, 1.5, 2.0}; β tolerance 0.15 on the ensemble mean.
- Redundancy levels γ ∈ {0.15, 0.70}; `N = 25`, `K = 5`, `d = 64`.
- **The ρ-relative budget definition `W_max = ρ · W_sat` and the grid {0.25, 0.60, 1.50}.**
- `W_sat` definition (proven-optimal unconstrained ILP).
- `λ_obj = 0.1`, `λ_mmr = 0.5`, the objective, all solvers, the stopping policy.
- **Primary estimand:** absolute Δ_gap = score(token_aware) − score(objective), mean over
  the two redundancy levels per (seed, β, budget).
- D = 1 − τ_b(r, r/w).
- Token-CV exclusion policy (record and exclude at seed level, never resample).
- Pilot seeds 1000–1004 and all their recorded outputs.
- `optimizer.py`, `benchmark.py`, `experiments/controlled/*`, and every Step 4/5 artifact.

---

## 10. Proposed next step

**Step 6B.2 — protocol amendment plus a blinded variance-component pilot.**

1. Write the amendment (§8) into `experiments/elasticity/protocol.py` as a versioned
   change (`6.1-amended`), with **every item signed off before any new data**, including
   SESOI = 3% and the T8 reformulation.
2. **Variance-component pilot** on pre-declared, disjoint seeds **2000–2039** (40 seeds).
   An automated script outputs **only** π per budget, E[X₁²] pooled over β, ρ_red, ρ_lvl
   and the mean ILP optimum — **no per-β means, no signs, no test statistics** — and from
   them computes n by the §8 rule. The analyst does not see directional quantities.
3. If n ≤ 300: lock n and extend the confirmatory seeds to 0 … n−1. If n > 300: stop with
   REDESIGN — for example raising N, which changes a held-constant factor and needs its
   own design review.
4. **Re-run the preflight** under the amended invariants on the locked confirmatory
   ensemble (generator-only and saturation, outcome-blind).
5. Only then the full experiment.
