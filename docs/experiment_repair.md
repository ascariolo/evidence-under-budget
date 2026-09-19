# Experiment Repair — Step 1 (implementation & methodology)

**Baseline preserved at:** git tag `v0-original-experiment` → `8fe4d34`
**This step produces:** a repaired baseline running the *same* research question on
the *same* benchmark data.

> **Scope warning.** This step fixes how the experiment is *measured and labelled*.
> It does **not** establish that the token-awareness hypothesis holds. It does not
> refute it either. The repaired numbers below are validation output, not a
> research conclusion; drawing one requires the benchmark work that has not been
> done yet (see section 9).

---

## 1. What was wrong in v0

Referenced against `docs/experiment_v0.md`, which records each defect at its line.

| ID | Defect | Consequence |
|---|---|---|
| I1 | ILP optimality read from `pulp.LpStatus` (`optimizer.py:320`) | `LpStatus` reports *a solution returned*, not *optimality proven*. A CBC run stopped on `timeLimit` was published as ground truth, so every "% of optimum" derived from it was unverified. |
| I2 | Method named `mmr` used a **summed** similarity penalty with the objective's own λ | The "baseline" was greedy ascent on the very objective being reported. No real MMR existed in the repository. |
| I3 | Top-K had no marginal-gain stopping rule; the greedy solvers did | The reported gap measured the stopping rule as much as the algorithm. |
| I4 | Restricted-pool ILP carried the sub-problem's optimality flag unqualified | A lower bound was labelled a proven optimum; ratios above 1.0 were printed under a `vs opt` header. |
| I5 | Context Density used as a headline comparison metric | `1000·Score(S)/tokens` has an `O(\|S\|²)` numerator penalty over an `O(\|S\|)` denominator, so it is maximized by near-empty selections. |
| I6 | Result JSON carried metric rows only | A result file could not be attributed to a seed, λ, backend, or commit. `np.random.seed` at `benchmark.py:258` had no consumer. |
| I7 | ILP run once while the README claimed "median of 3"; ILP timing included subprocess spawn | Latency figures not comparable, and one README cell disagreed with the shipped JSON. |
| I8 | `best = results[0]` | Named "best", actually the first run; paired one run's selection with another run's timing. |

---

## 2. What was changed

### 2.1 ILP status (I1)

`optimizer.py` gains `SolveStatus`, `SolveScope` and `classify_pulp_status()`.

Classification reads **both** `problem.status` and `problem.sol_status`:

| `status` | `sol_status` | → `SolveStatus` |
|---|---|---|
| `LpStatusOptimal` | `LpSolutionOptimal` | `PROVEN_OPTIMAL` |
| `LpStatusOptimal` | `LpSolutionIntegerFeasible` | `FEASIBLE_NOT_PROVEN` |
| any | `LpSolutionInfeasible` / `LpStatusInfeasible` | `INFEASIBLE` |
| any | `LpSolutionUnbounded` | `UNBOUNDED` |
| any | `LpSolutionNoSolutionFound` / `LpStatusNotSolved` | `NOT_SOLVED` |
| anything else | — | `UNDEFINED` |
| (heuristics) | — | `HEURISTIC` |

When `sol_status` is missing (older PuLP), the conservative branch is taken:
a returned solution is `FEASIBLE_NOT_PROVEN`, never `PROVEN_OPTIMAL`.

`SelectionResult.optimal` is now a **read-only property** derived from
`status` **and** `scope`. There is no assignable optimality flag, so the v0
failure mode — copying a flag onto a result that did not earn it — is
structurally impossible. Assigning to it raises `AttributeError`.

Incumbents are preserved verbatim: a time-limited solution is returned with its
selection intact and labelled `feasible_not_proven`. Only `INFEASIBLE` yields an
empty selection. The solver's own objective value is stored in
`meta["solver_objective"]` and cross-checked against our recomputed `Score(S)`
(`meta["objective_consistent"]`).

### 2.2 A genuine MMR baseline (I2)

Two separately named parameters, because they parameterize different formulas:

| Name | Default | Appears in |
|---|---|---|
| `objective_lambda` | `0.1` | the research objective, as a weight on **summed** pairwise redundancy |
| `mmr_lambda` | `0.5` | the MMR baseline only, as the convex weight on relevance against **max** similarity |

`DEFAULT_LAMBDA` and the `lambda_` keyword survive as deprecated aliases so v0
callers keep working.

`solve_mmr()` now implements

```
i* = argmax_i  λ_mmr · rel(q, d_i) − (1 − λ_mmr) · max_{j∈S} sim(d_i, d_j)
```

The v0 solver that was called `mmr` is kept under its accurate name,
`solve_greedy_objective()`:

```
i* = argmax_i  rel(q, d_i) − λ_obj · Σ_{j∈S} sim(d_i, d_j)
```

Keeping it is deliberate: it differs from the proposed method by the `1/w_i`
factor **and nothing else**, so it is the correct control for isolating token
normalization. `mmr_lambda = 0.5` is the neutral textbook value; it has **not**
been tuned, and tuning it is explicitly out of scope for this step.

### 2.3 Common stopping policy (I3)

Every heuristic now runs through one loop, `ContextKnapsack._select`, and
differs only in a `priority_fn` that decides which eligible candidate to take.

> **Stopping policy.** A candidate `i` is *eligible* iff
> (1) it is unselected, (2) `w_i ≤ remaining budget`, and
> (3) its marginal gain on the objective is strictly positive:
> `Δ_i = r_i − λ_obj · Σ_{j∈S} s_ij > 0`.
> Selection stops when no eligible candidate remains.

Two properties of this definition matter:

- **Condition 3 is stated on the objective's marginal gain for every method**,
  including MMR, whose own ranking score is unrelated to it. Ranking rule and
  termination rule are kept strictly separate, so the comparison isolates the
  ranking rule.
- **Top-K keeps its character.** Its `priority_fn` is `relevance`, full stop —
  no redundancy term, no token normalization. It still skips a document that
  does not fit and continues down the ranking (v0 behaviour, preserved).

For greedy and token-aware greedy this is a no-op: `argmax` over `gain` or
`gain/w` already implied condition 3, and their selections are byte-identical to
v0 (verified, section 8). Only Top-K's behaviour changes, which is the point.

`stop_on_nonpositive_gain=False` (CLI: `--no-gain-stop`) restores v0 semantics
for ablation. Each result records `meta["stopping_policy"]` and
`meta["stop_reason"]` ∈ {`no_positive_marginal_gain`, `no_candidate_fits_budget`,
`k_reached`}.

### 2.4 Pool vs global optimum (I4)

`solve_ilp_on_top_candidates` → **`solve_ilp_reference`**. The pooled solve now
carries `scope = RESTRICTED_POOL`, which makes `is_proven_global_optimum` false
by construction; `is_proven_pool_optimum` is a separate property.

The benchmark's `reference_is_optimal: bool` → **`reference_kind: str`**:

| Value | Meaning | Column header |
|---|---|---|
| `proven_global_optimum` | ILP proved optimal over the full candidate set | `vs opt` |
| `proven_pool_optimum` | ILP proved optimal over a restricted pool — a **lower bound** | `vs pool` |
| `best_known` | nothing proven; best objective any method reached | `vs best` |

Under `vs pool` the table prints that ratios above 1.000 are expected and do not
beat the true optimum. The ILP row is tagged `[pool]` and `[incumbent]`
independently, so the two qualifications can never be confused.

The optimization itself is untouched: same model, same constraints, same pool
construction, same re-scoring against the full corpus.

### 2.5 Context Density demoted (I5)

`SelectionResult.density` → `density_diagnostic` (`density` kept as a deprecated
alias). In `MethodMetrics` the field is named `density_diagnostic`; the bare name
no longer exists.

It is **not** deleted, and it is still computed and reported, because continuity
with v0 is worth keeping. What changed is its standing:

- removed from the plot entirely;
- printed last in the table under `dens*`, with a footnote on every block stating
  it is not a ranking metric;
- listed under `diagnostic_metrics` in the result metadata, never under
  `primary_metrics`;
- never used to choose a reference, order a table, or support a claim.

**Why.** The numerator carries an `O(|S|²)` redundancy penalty while the
denominator grows `O(|S|)` in tokens, so the ratio rises as a selection shrinks.
Measured on the v0 corpus at `W_max = 2048`: the reported winner scored `5.888`,
while a selection consisting of the **single** highest-density document scores
`40.381` at coverage `0.06`, and 8 of the 11 possible single-document removals
from the winning selection *increase* it. A metric whose optimum is a
one-sentence context cannot rank context-packing methods.
`test_density_metric_rewards_tiny_selections` pins this behaviour.

**Primary metrics** are now `score`, `unique_coverage`, `redundancy`,
`redundancy_per_pair`, `tokens_used`, `n_selected`. **Diagnostic** are
`density_diagnostic`, `token_savings_pct` (budget utilization, not quality) and
`latency_ms`.

`redundancy_per_pair` is new and is reported alongside the total because total
redundancy conflates duplication with selection size: on the v0 corpus the
headline "5.9x redundancy reduction" is `110.68/18.90`, but those selections hold
276 and 55 pairs — per pair it is `0.401` vs `0.344`, a 1.17x difference. It is
`None`, not `0.0`, for selections with fewer than two documents.

### 2.6 Reproducibility metadata (I6)

New `RunMetadata` dataclass. The JSON schema is now
`{"metadata": {...}, "results": {...}}` (v0 wrote the rows at top level).

Recorded: `benchmark_config_version`, UTC timestamp, `seed`, `objective_lambda`,
`mmr_lambda`, `n_docs`, `budgets`, `repeats`, `ilp_repeats`, `stopping_policy`,
`include_ilp`, `ilp_time_limit`, `ilp_max_docs`, `ilp_solver`,
`embedding_backend`, `embedding_model_requested`, `embedding_fallback_used`,
`tokenizer_encoding`, `tokenizer_exact`, the query, corpus token statistics,
git commit / describe / dirty, Python version, platform, versions of numpy,
pulp, sentence-transformers, tiktoken, matplotlib, scipy, torch, and the
primary/diagnostic metric lists plus caveat notes.

Anything that cannot be determined is written as the literal string
`"unavailable"`. No field is populated with a guess or a default standing in for
a real value.

The silent encoder fallback is now loud: if the run does not end up on the
requested model, `embedding_fallback_used` is `true` in the artifact **and** a
warning is printed that the results are not comparable to v0.

`np.random.seed(seed)` is kept with a comment recording that it is defensive and
not load-bearing — all randomness in the pipeline is seeded at its source.

### 2.7 Latency (I7)

**The JSON artifact is authoritative.** The README's §4 latency column disagrees
with the shipped `benchmark_results.json` at `W_max = 256` (`721.40` vs `251.68`
ms) while every other field in that row matches, so the README column was not
produced by the run that produced the rest of the table. It was not hand-edited
here; it will be regenerated from a run when the README is rewritten.

The same procedure is applied to every method — median of *N* runs, with *N*
recorded per row as `latency_runs`. `ilp_repeats` defaults to `1` because the
exact solver costs seconds per call; the count is now stated in the output
rather than silently implied.

**Latency is not comparable across solver classes, and is labelled as such.**
Heuristic timings are in-process NumPy loops; the ILP timing additionally covers
model construction, LP file IO and CBC subprocess startup. Each row therefore
carries `latency_breakdown` with `model_build_ms`, `solver_ms` and
`includes_subprocess_spawn`. Latency was removed from the plot, is printed with
a caveat, sits under `diagnostic_metrics`, and must not carry a headline claim.

### 2.8 Small correctness fixes (I8)

- `best = results[0]` → `_merge_repeats()`, which **asserts** that every repeat
  returned the same selection and raises `RuntimeError` naming the method if not.
  Determinism is checked, not assumed.
- `gain / tokens` now uses `np.divide(..., where=tokens > 0)`, so a zero-token
  document no longer raises `RuntimeWarning: divide by zero`
  (`test_zero_token_document_does_not_warn_or_crash` runs under
  `np.errstate(divide="raise")`).
- `score_vs_reference` is suppressed (`None`) when the reference is not strictly
  positive. v0 published `-0.786` as a ratio to an optimum.
- The `ilp` docstring's complexity note was corrected: the model has `N` binaries
  and `O(N²)` **continuous** variables.
- `solve_top_k(k=...)` now actually honours `k`.

---

## 3. What was deliberately NOT changed

- **The objective function.** `ContextKnapsack.score` is byte-for-byte v0.
  `test_objective_is_unchanged_from_v0` pins it.
- **`objective_lambda = 0.1`.** Carried over unchanged, including its provenance
  problem (section 9).
- **The corpus generator.** `generate_corpus`, `TOPIC_BANK` and `PADDING` are
  identical to v0, marked as frozen in the source. The relevance distribution,
  the length distribution and the relationship between them are untouched.
- **The research question.**
- **The proposed method.** `solve_greedy` is mathematically unchanged and returns
  byte-identical selections to v0 at every budget tested (section 8).
- **`unique_coverage`.** Its known weaknesses (unreachable clusters counted in
  the denominator; distractor clusters weighted like on-topic ones; fixed `0.85`
  threshold) are recorded in section 9, not fixed.
- **README claims.** Untouched; they are now known to disagree with the code in
  places and will be rewritten only once repaired results exist.
- **v0 artifacts.** `benchmark_results.json`, `benchmark_results.png` and the
  snapshots under `docs/v0_results/` were not modified.

---

## 4. Exact definition of each method

Notation: `r_i = rel(q, d_i)`, `s_ij = sim(d_i, d_j) ∈ [0,1]`, `w_i` tokens,
`S` the current selection, `Δ_i = r_i − λ_obj · Σ_{j∈S} s_ij`.

**Objective (unchanged):**
```
Score(S) = Σ_{i∈S} r_i − λ_obj · Σ_{i<j∈S} s_ij      s.t.  Σ_{i∈S} w_i ≤ W_max
```

| Method | Selection rule | Notes |
|---|---|---|
| `top_k` | `argmax_i r_i` | pure relevance ranking; no redundancy term, no token term |
| `mmr` | `argmax_i [ λ_mmr·r_i − (1−λ_mmr)·max_{j∈S} s_ij ]` | max similarity; convex weight; `max ∅ = 0` |
| `greedy_objective` | `argmax_i Δ_i` | greedy ascent on `Score(S)`; v0's `mmr` |
| `greedy_token_aware` | `argmax_i Δ_i / w_i` | the proposed method |
| `ilp` | exact max of `Score(S)` | linearized QKP; `N` binaries, `O(N²)` continuous `y_ij ≥ x_i + x_j − 1` |

All four heuristics share the stopping policy in section 5. The ILP has no
stopping policy — the budget constraint is in the model.

## 5. Exact stopping policy

> Candidate `i` is **eligible** iff it is unselected, `w_i ≤ remaining`, and
> `Δ_i > 0`. Selection stops when no eligible candidate remains.

Recorded per result as `meta["stopping_policy"] = "positive_marginal_gain"` and
`meta["stop_reason"]`. The v0 alternative is available as `"fill_budget_v0"` via
`stop_on_nonpositive_gain=False` / `--no-gain-stop`, for ablation only.

## 6. Exact interpretation of ILP status

| Reported | Means | May be called an optimum? |
|---|---|---|
| `proven_optimal` + `full_corpus` | no better feasible selection exists over all candidates | **yes** — the global optimum |
| `proven_optimal` + `restricted_pool` | optimal over the stated pool | **no** — a lower bound on the global optimum |
| `feasible_not_proven` | a valid selection; optimality not established (e.g. time limit) | **no** |
| `infeasible` | no feasible selection; empty result | n/a |
| `unbounded` / `not_solved` / `undefined` | solver failure state | **no** |
| `heuristic_no_guarantee` | any heuristic | **no** |

## 7. Primary vs diagnostic metrics

**Primary** (comparison and claims): `score`, `unique_coverage`, `redundancy`,
`redundancy_per_pair`, `tokens_used`, `n_selected`.

**Diagnostic** (inspection only, never ranking): `density_diagnostic`
(deprecated, section 2.5), `token_savings_pct` (utilization, not quality),
`latency_ms` (not comparable across solver classes, section 2.7).

## 8. Validation of the repair

Small deterministic runs, `seed = 42`, generator unchanged.

**Proposed method unchanged.** Repaired code vs the v0 code checked out from the
tag, N = 25, `λ_obj = 0.1`:

| `W_max` | `greedy_token_aware` v0 → repaired | `greedy_objective` (v0 `mmr`) | `top_k` v0 → repaired |
|---:|---|---|---|
| 256 | 2.584 → 2.584 *(same indices)* | 2.626 → 2.626 *(same indices)* | 2.626 → 2.626 *(same indices)* |
| 512 | 2.768 → 2.768 *(same indices)* | 2.821 → 2.821 *(same indices)* | 2.722 → 2.821 |
| 1024 | 2.768 → 2.768 *(same indices)* | 2.821 → 2.821 *(same indices)* | 2.229 → 2.821 |
| 2048 | 2.768 → 2.768 *(same indices)* | 2.821 → 2.821 *(same indices)* | −2.217 → 2.821 |

The proposed method and the renamed v0 baseline are untouched. Only Top-K moves,
only where the stopping rule bit, exactly as intended.

**Status classification.** With `--ilp-time-limit 0.6` at N = 45, PuLP reported
`LpStatus = "Optimal"` (what v0 trusted) with `sol_status = 2`. The repaired code
labelled it `feasible_not_proven`, tagged the row `[incumbent]`, switched the
comparison column to `vs best`, and preserved the incumbent intact (12 documents,
score 2.9522).

**Pool labelling.** With `--ilp-max-docs 8` at N = 20, the header read `vs pool`,
the row was tagged `[pool]`, ratios above 1.000 appeared (up to 1.176) and the
block printed the explanation. `is_proven_global_optimum` was false throughout.

**Schema and metadata.** The validation artifact carries both top-level keys and
every required metadata field, including the git commit and the note that the
working tree was dirty.

> **Observation, recorded without interpretation.** With the stopping rule
> equalized, Top-K reaches the same objective as `greedy_objective` at
> `W_max ≥ 512` on this corpus, and both exceed the proposed token-aware method
> (2.821 vs 2.768). True MMR scores below all of them on the objective, which is
> expected since it optimizes a different criterion. These are validation
> outputs on one corpus at one seed. **No conclusion about the research
> hypothesis follows from them**, in either direction — see section 9.

## 9. Remaining methodological limitations

Not addressed at this step, and each blocking a research claim:

1. **`objective_lambda` was selected on the evaluation corpus** (v0's "Finding
   2") and is carried over unchanged. Not a held-out value.
2. **The benchmark is one corpus, one query, one seed.** No variance, no
   confidence intervals, no multi-query evaluation. Differences of ~2% cannot be
   distinguished from instance noise at n = 1.
3. **Relevance and token count are uncorrelated in the generator**
   (Pearson `0.028`, Spearman `0.017`, `p = 0.94` on the default corpus). The
   pathology the proposed method targets — long low-value passages crowding out
   short high-value ones — is asserted in the README but is not instantiated in
   the data. This is the single largest obstacle to testing the hypothesis, and
   fixing it requires benchmark work that is explicitly out of scope here.
4. **The budget sweep is not four independent points.** On the v0 corpus the
   knapsack selections at 512, 1024 and 2048 are identical; the solution
   saturates at 480 tokens.
5. **`unique_coverage` remains v0's.** Clusters unreachable within the budget are
   still in the denominator, distractor clusters count like on-topic ones, and
   the `0.85` threshold is unswept.
6. **No downstream task metric.** `Score(S)` and coverage are proxies; the claim
   that better packing yields better answers is untested.
7. **`mmr_lambda = 0.5` is a convention, not a fitted value.** The MMR baseline
   has not been tuned, so it is not being shown at its best.
8. **The README has not been rewritten** and currently contains claims known to
   disagree with the code and with the shipped results.

**Nothing in this document supports or refutes the token-awareness hypothesis.**
The repaired baseline exists so that such a test can be run honestly later.
