# The Repaired Original Benchmark — Results

**What this is.** The repaired solvers (`docs/experiment_repair.md`) evaluated on the
*unmodified* v0 benchmark generator, across enough seeds and budgets to support a
statistical statement. It answers one question: *what does the original experiment
actually show once the methodological problems are fixed?*

**What this is not.** A benchmark redesign, a re-tuning, or a new hypothesis. The
generator, the objective and both lambdas are frozen at their documented values.

---

## 1. Protocol (pre-registered)

Fixed in `experiments/run_repaired_original.py` **before** any result was inspected.

| | |
|---|---|
| Corpora | `benchmark.generate_corpus`, **unchanged from v0**, `n_docs = 25` |
| Seeds | `0 … 23` — 24 contiguous seeds, fixed a priori. None added, dropped or reordered afterwards. |
| Budgets | 96, 128, 192, 256, 384, 512, 768, 1024, 2048 — the four originally reported budgets plus five intermediate points |
| Instances | 24 × 9 = **216**, five methods each = **1080 rows** |
| Methods | `top_k`, `mmr` (true max-similarity), `greedy_objective`, `greedy_token_aware` (proposed), `ilp` (exact) |
| `objective_lambda` | **0.1** — frozen, carried from v0, **not** tuned on these results |
| `mmr_lambda` | **0.5** — untuned textbook default |
| Objective | `Score(S) = Σ r_i − λ_obj · Σ_{i<j} s_ij` s.t. `Σ w_i ≤ W_max` — **unchanged** |
| Stopping | common policy: eligible iff unselected, fits, and `Δ_i > 0` |
| ILP | full-corpus scope, CBC, 300 s limit |
| Encoder / tokenizer | `all-MiniLM-L6-v2` / `cl100k_base` (exact). Run aborts on fallback. |
| Runtime | 26.5 min |

**ILP status: `proven_optimal` on 216/216 instances.** No solve timed out, none was
dropped, and every optimality gap below is measured against a genuinely proven global
optimum.

## 2. Method definitions

| Method | Rule |
|---|---|
| `top_k` | `argmax_i r_i` |
| `mmr` | `argmax_i [ λ_mmr·r_i − (1−λ_mmr)·max_{j∈S} s_ij ]` |
| `greedy_objective` | `argmax_i Δ_i`, `Δ_i = r_i − λ_obj·Σ_{j∈S} s_ij` |
| `greedy_token_aware` | `argmax_i Δ_i / w_i` — **the proposed method** |
| `ilp` | exact maximum of `Score(S)` |

`greedy_objective` differs from `greedy_token_aware` by the `1/w_i` factor **and
nothing else**. Their paired difference therefore *is* the contribution of
token-awareness — this is comparison **C2b** and the only clean test of the hypothesis
available here.

## 3. Statistical procedure

The independent unit is the **corpus (seed)**. Budgets evaluated on the same corpus are
not independent, so every test runs **within a budget across the 24 seeds** (24
independent pairs). A pooled figure over all 216 cells is printed for context, flagged
non-independent, and never used to support a conclusion.

Per comparison: n, wins/losses/ties, mean/median/sd/min/max of the paired difference,
Wilcoxon signed-rank *p* (two-sided), Cohen's `d_z`, and matched-pairs rank-biserial
`r_rb`. No *p*-value appears without its effect size and paired counts.

---

## 4. Relevance vs token length in this benchmark

Descriptive, per corpus, across 24 seeds:

| | mean | median | sd | min | max |
|---|---|---|---|---|---|
| Pearson `r(relevance, tokens)` | **+0.023** | +0.030 | 0.145 | −0.319 | +0.343 |
| Spearman `ρ` | **+0.076** | +0.111 | 0.209 | −0.311 | +0.461 |

Seeds with Pearson `p < 0.05`: **0 / 24**.

> **The current benchmark does not contain a strong relevance–length trade-off, so this
> experiment is not a strong test of the token-awareness hypothesis.**

This is a result about the instrument, not a failure of the run. It constrains what
section 8 may conclude, and it is descriptive only — no causal claim is drawn from it.

## 5. Objective score by budget

Mean ± sd over 24 seeds. Best heuristic per row in bold.

| `W_max` | Top-K | True MMR | Greedy on objective | **Token-aware greedy** | ILP (proven) |
|---:|---|---|---|---|---|
| 96 | 1.630±0.274 | 1.558±0.306 | 1.654±0.293 | **2.182±0.279** | 2.197±0.270 |
| 128 | 1.847±0.382 | 1.930±0.290 | 1.929±0.394 | **2.397±0.289** | 2.433±0.281 |
| 192 | 2.422±0.319 | 2.036±0.417 | 2.550±0.304 | **2.674±0.252** | 2.714±0.248 |
| 256 | 2.686±0.272 | 2.342±0.348 | 2.651±0.319 | **2.794±0.200** | 2.824±0.206 |
| 384 | 2.789±0.222 | 2.630±0.254 | 2.846±0.222 | **2.860±0.184** | 2.903±0.190 |
| 512 | 2.820±0.218 | 2.754±0.196 | **2.924±0.168** | 2.879±0.166 | 2.929±0.168 |
| 768 | 2.852±0.164 | 2.778±0.175 | **2.930±0.163** | 2.880±0.163 | 2.935±0.162 |
| 1024 | 2.852±0.164 | 2.787±0.165 | **2.930±0.163** | 2.880±0.163 | 2.935±0.162 |
| 2048 | 2.852±0.164 | 2.787±0.165 | **2.930±0.163** | 2.880±0.163 | 2.935±0.162 |

**Budget saturation.** Heuristic selections stop changing above `W_max = 768`
(7 distinct regimes out of 9 for every heuristic; the ILP has 9). Rows 1024 and 2048
carry no information the 768 row does not. The v0 sweep's four budgets contained only
two distinct heuristic regimes.

## 6. Optimality gap vs the proven global optimum

`gap = (ILP − method) / |ILP|`, valid on all 216 instances. Mean (worst single instance).

| `W_max` | Top-K | True MMR | Greedy on objective | **Token-aware greedy** |
|---:|---|---|---|---|
| 96 | 25.06% (52.7%) | 28.36% (61.8%) | 24.18% (52.7%) | **0.73% (5.2%)** |
| 128 | 24.17% (42.6%) | 20.19% (52.2%) | 20.83% (41.6%) | **1.49% (6.6%)** |
| 192 | 10.51% (46.7%) | 25.29% (58.7%) | 6.10% (24.5%) | **1.46% (4.4%)** |
| 256 | 4.95% (19.8%) | 17.07% (43.1%) | 6.24% (22.3%) | **1.06% (4.8%)** |
| 384 | 3.91% (14.6%) | 9.44% (21.8%) | 2.00% (13.0%) | **1.45% (3.7%)** |
| 512 | 3.80% (16.0%) | 6.00% (15.7%) | **0.18% (1.0%)** | 1.73% (4.7%) |
| 768–2048 | 2.79% (8.3%) | 5.03–5.35% | **0.17% (1.0%)** | 1.87% (5.6%) |

Token-aware greedy is the only method whose worst-case gap stays under 7% at every
budget. `greedy_objective` is near-exact above 512 but degrades to 24% at tight budgets.
v0's headline "98.1% of optimum" (1.9% gap) holds only in the loose-budget regime; the
tight-budget regime it never tested is where the untokenized heuristics fail worst.

## 7. Paired comparisons

### C1 — token-aware greedy vs relevance-only Top-K (repaired common stopping policy)

| `W_max` | W/L/T | mean diff | `d_z` | Wilcoxon `p` |
|---:|---|---|---|---|
| 96 | 23/0/1 | +0.552 | +1.69 | 2.7e-05 |
| 128 | 23/1/0 | +0.550 | +1.72 | 2.4e-07 |
| 192 | 23/1/0 | +0.252 | +0.86 | 2.3e-06 |
| 256 | 19/3/2 | +0.108 | +0.67 | 9.8e-04 |
| 384 | 17/7/0 | +0.071 | +0.57 | 9.6e-03 |
| 512 | 15/9/0 | +0.059 | +0.56 | 9.6e-03 |
| 768–2048 | 13/11/0 | +0.028 | +0.39 | **0.178 (n.s.)** |

Significant at `W_max ≤ 512`; **not significant** at 768 and above. The advantage over
Top-K is real but shrinks monotonically with budget and vanishes once the constraint
stops binding.

### C2 — token-aware greedy vs true MMR

| `W_max` | W/L/T | mean diff | `d_z` | Wilcoxon `p` |
|---:|---|---|---|---|
| 96 | 24/0/0 | +0.624 | +1.80 | 1.2e-07 |
| 128 | 24/0/0 | +0.467 | +1.58 | 1.2e-07 |
| 192 | 24/0/0 | +0.639 | +2.13 | 1.2e-07 |
| 256 | 23/1/0 | +0.452 | +1.40 | 3.6e-07 |
| 384 | 20/4/0 | +0.231 | +1.25 | 5.1e-06 |
| 512 | 20/4/0 | +0.125 | +0.97 | 9.1e-05 |
| 768–2048 | 20/4/0 | +0.093…+0.102 | +1.02…+1.09 | 3.0e-05 |

Token-aware greedy beats true MMR at **every** budget with large effect sizes.
**This is not evidence that MMR is a weak algorithm.** MMR optimizes a different
criterion (max-similarity diversity, convex weighting) and is being scored on an
objective it does not target, at an untuned `λ_mmr = 0.5`. See section 9.

### C2b — token-aware greedy vs the same greedy without token normalization
**(the isolation test: these two differ only by `1/w_i`)**

| `W_max` | W/L/T | mean diff | `d_z` | `r_rb` | Wilcoxon `p` | verdict |
|---:|---|---|---|---|---|---|
| 96 | 22/0/2 | +0.528 | +1.72 | +1.00 | 4.0e-05 | token-awareness **helps** |
| 128 | 21/2/1 | +0.469 | +1.40 | +0.96 | 5.2e-05 | helps |
| 192 | 18/3/3 | +0.125 | +0.75 | +0.84 | 7.0e-04 | helps |
| 256 | 12/8/4 | +0.142 | +0.65 | +0.61 | 1.7e-02 | helps (weaker) |
| 384 | 10/14/0 | +0.015 | +0.17 | +0.01 | **0.989** | **null** |
| 512 | 2/21/1 | −0.046 | −1.16 | −0.96 | 6.0e-05 | token-awareness **hurts** |
| 768–2048 | 2/21/1 | −0.050 | −1.18 | −0.96 | 6.0e-05 | hurts |

**The effect of token-awareness reverses sign with the token budget.** It is strongly
beneficial when the budget binds hard, null at `W_max ≈ 384`, and reliably harmful once
the budget is loose — all with large effect sizes and `p < 1e-4` at both ends.

### C3 — every heuristic vs the proven ILP optimum

All 216 instances proven optimal. Mean paired difference (method − ILP):

| `W_max` | Top-K | True MMR | Greedy on objective | Token-aware greedy |
|---:|---|---|---|---|
| 96 | −0.567 (0/23/1) | −0.639 (0/24/0) | −0.543 (0/22/2) | **−0.015 (4/8/12)** |
| 256 | −0.139 (0/19/5) | −0.482 (0/24/0) | −0.173 (4/17/3) | **−0.031 (3/17/4)** |
| 512 | −0.110 (0/23/1) | −0.175 (0/24/0) | **−0.005 (7/8/9)** | −0.051 (1/22/1) |
| 2048 | −0.082 (1/21/2) | −0.148 (1/23/0) | **−0.005 (6/11/7)** | −0.055 (1/22/1) |

No heuristic ever exceeds the proven optimum (W column reflects ties in floating point
at instances where a heuristic *matches* it, never beats it — the invariant holds).

## 8. Secondary and descriptive metrics

**Redundancy — why the per-pair figure is mandatory.** At `W_max = 96` token-aware
greedy has the *highest* total redundancy (7.48 vs Top-K's 2.87) and the *second lowest*
per pair (0.372 vs 0.590). It looks worse on the total purely because it selects 6.83
documents to Top-K's 3.58. Any redundancy claim based on the total alone is invalid.

| `W_max` | metric | Top-K | True MMR | Greedy obj. | Token-aware | ILP |
|---:|---|---|---|---|---|---|
| 96 | redundancy total | 2.87 | 1.93 | 2.48 | 7.48 | 7.07 |
| 96 | redundancy per pair | 0.590 | **0.303** | 0.532 | 0.372 | 0.380 |
| 2048 | redundancy per pair | 0.438 | **0.353** | 0.405 | 0.374 | 0.397 |

True MMR achieves the lowest per-pair redundancy at every budget — it is doing exactly
what it is designed to do.

**Documents selected** (mean): at `W_max = 96`, token-aware 6.83 vs Top-K 3.58 vs ILP
6.54. The proposed method tracks the optimum's cardinality closely; the untokenized
heuristics under-select at tight budgets.

**Budget utilization** (descriptive, not quality): token-aware greedy consistently uses
the *fewest* tokens (e.g. 0.776 vs Top-K 0.889 at 512), and the ILP optimum sits near it
(0.822) — so lower utilization here coincides with, but does not by itself demonstrate,
better selection.

**Coverage** (secondary, v0 definition retained): token-aware greedy leads at tight
budgets (0.271 vs 0.174 at 96), but **true MMR leads at loose budgets** (0.640 vs 0.522
at 2048) despite its worse objective. Coverage and the objective disagree above
`W_max ≈ 384`.

---

## What the repaired original benchmark actually supports

### Supported

1. **The ILP is a usable ground truth at N = 25.** 216/216 solves proven globally
   optimal within 300 s.
2. **Redundancy-aware selection beats pure relevance ranking**, even with the stopping
   rule equalized, at every budget where the constraint binds (`W_max ≤ 512`,
   `p ≤ 9.6e-3`). Not significant at 768+ (`p = 0.178`).
3. **The proposed token-aware greedy is the best heuristic in the tight-budget regime**,
   and the only one with a worst-case gap under 7% everywhere. At `W_max ≤ 256` it is
   within 0.7–1.5% of the proven optimum while every other heuristic is 6–28% away.
4. **Token-awareness has a real, budget-dependent effect** (C2b): it helps significantly
   at `W_max ≤ 256`, is null at 384, and hurts significantly at `W_max ≥ 512`. Both
   directions carry `|d_z| > 1.1` and `p < 1e-4`.
5. **Total redundancy is not a valid standalone measure** — demonstrated directly, the
   method with the highest total has among the lowest per-pair values.
6. **v0's headline gap (~1.9%) describes only the loose-budget regime.** It is
   reproducible there, and it understates what happens at tight budgets — where it is
   better for the proposed method, not worse.

### Not supported

1. **"Token-awareness improves selection quality" as an unqualified claim.** It is false
   above `W_max ≈ 384` on this benchmark, where the identical greedy *without* the
   `1/w_i` factor wins 21 of 24 seeds.
2. **The stated mechanism.** The README attributes the benefit to long padded passages
   outranking and crowding out short snippets. Relevance and length are uncorrelated
   here (`r = +0.023`, 0/24 seeds significant), so that mechanism is not present in the
   data and cannot be what produces the tight-budget gain. The gain is consistent with
   ordinary knapsack packing efficiency under a binding constraint — but that is a
   hypothesis this experiment does not test, not a finding.
3. **A general claim over "the budget range".** Heuristic selections saturate above 768;
   the sweep contains ~7 distinct regimes, and the original four budgets contained two.
4. **Any claim resting on Context Density or on total redundancy alone.**

### Untestable with this benchmark

1. **Whether token-awareness solves the length/value trade-off it was designed for.**
   The generator produces no such trade-off (section 4). Testing this requires a corpus
   where document length and relevance are deliberately decoupled — benchmark work that
   is out of scope here.
2. **Whether better packing yields better downstream answers.** No LLM, no task metric.
   `Score(S)` and coverage are proxies.
3. **Whether "lost-in-the-middle" is mitigated.** Never measured anywhere in this
   repository.
4. **Whether MMR is competitive when properly configured.** `λ_mmr = 0.5` is untuned by
   protocol; C2 shows MMR losing on an objective it does not optimize.

### Verdict on the central question

> **Does token-awareness improve selection quality?**
> **Conditionally supported, and only in the regime where the budget binds.**
> Significant benefit at `W_max ≤ 256` (up to `d_z = +1.72`), null at 384, significant
> harm at `W_max ≥ 512` (`d_z ≈ −1.17`). Because the benchmark contains no
> relevance–length trade-off, this experiment cannot attribute the tight-budget benefit
> to the mechanism the project claims, and is not a strong test of the hypothesis as
> originally stated.

### Correction to an earlier finding in this project

An earlier adversarial pass in this repository reported "no evidence for
token-awareness" (15/40 wins, `p = 0.61`). That test used **only budgets 256 and 1024**
— one at the crossover point and one deep in the regime where token-awareness hurts —
so the opposing effects cancelled. With nine budgets the effect is large, significant
and sign-flipping. The earlier null was an artifact of budget selection, and that is a
caution about this kind of sweep generally, not only about that run.

---

## 9. Limitations

1. `objective_lambda = 0.1` was selected on this corpus family in v0. Frozen here, not
   re-tuned, but still not a held-out value. Every result is conditional on it, and the
   C2b crossover budget in particular is a function of λ.
2. `mmr_lambda = 0.5` is untuned; MMR is not shown at its best.
3. One generator, one query string, one corpus size (N = 25), one encoder. Seed
   variation is covered; distribution variation is not.
4. Budgets above 768 add no information for the heuristics.
5. Coverage retains v0's definition, including its known weaknesses (budget-unreachable
   clusters in the denominator, distractors weighted like on-topic content, unswept 0.85
   threshold). Treated as secondary throughout.
6. No downstream task metric.
7. Latency was recorded per row but is excluded from all conclusions: it is not
   comparable across solver classes.

## 10. Artifacts

| Path | Contents |
|---|---|
| `results/repaired_original/raw_results.json` | 1080 rows + per-seed corpus descriptives + full run metadata |
| `results/repaired_original/raw_results.csv` | same rows, flat |
| `results/repaired_original/summary_by_budget.csv` | mean/median/sd/min/max per budget × method × metric |
| `results/repaired_original/paired_comparisons.csv` | every paired test incl. effect sizes |
| `results/repaired_original/analysis_report.txt` | full text analysis |
| `results/repaired_original/figures/fig1…fig5*.png` | objective, optimality gap, utilization, per-pair redundancy, documents selected — each mean over 24 seeds with 95% CI bands |
| `experiments/run_repaired_original.py` | pre-registered runner |
| `experiments/analyze_repaired_original.py` | analysis |

v0 artifacts (`benchmark_results.json/.png`, `docs/v0_results/`) were not touched.
