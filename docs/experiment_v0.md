# Experiment v0 — Original State (pre-repair)

**Preservation point:** git tag `v0-original-experiment` → commit `8fe4d34b2106b218b6f5ff367b747dfe448d3b15`
**Date of commit:** 2026-09-13
**Status when tagged:** working tree clean, single commit, no prior tags.

This document records **what the code in v0 actually does**, not what it should do and
not how the claims should be interpreted. It is the reference point against which every
later methodological change is measured.

Result artifacts are snapshotted in `docs/v0_results/` because `benchmark_results.json`
is listed in `.gitignore` and is therefore **not** captured by the tag itself.

---

## 1. What the experiment claims to investigate

From `README.md:9-32` and `AGENTS.md`:

Context selection for LLM prompts is framed as constrained subset selection rather than
ranking. The stated hypothesis is that naive Top-K retrieval wastes context through
(a) redundancy between highly-ranked documents, (b) token inefficiency where long
low-density passages crowd out short high-value ones, and (c) "lost-in-the-middle"
degradation from bloated prompts. The study proposes a token-aware greedy heuristic and
compares it against an exact ILP ground truth and two baselines.

---

## 2. Methods compared

All four are methods on one class, `ContextKnapsack` (`optimizer.py:96`). All consume the
pre-computed matrices from `embedder.build_corpus`; none touch raw embeddings.

| Registry key | Label in output | Entry point | What it does in v0 |
|---|---|---|---|
| `top_k` | Naive Top-K | `optimizer.py:171` `solve_top_k` | Sorts by `relevance` descending (stable). Walks the ranking and appends any document whose cost fits the remaining budget, **skipping over** documents that do not fit and continuing down the list. Has no redundancy term and no gain-based stopping rule. The `k` parameter exists but is never passed anywhere in the repository (`solve_all` calls it as `self.solve_top_k()`, `optimizer.py:330`). |
| `mmr` | MMR | `optimizer.py:191` `solve_mmr` | Delegates to `_greedy(token_aware=False)`. Selects `argmax_i [ r_i − λ · Σ_{j∈S} s_ij ]`. |
| `greedy_token_aware` | Token-Aware Knapsack | `optimizer.py:204` `solve_greedy` | Delegates to `_greedy(token_aware=True)`. Selects `argmax_i [ (r_i − λ · Σ_{j∈S} s_ij) / w_i ]`. This is the proposed method. |
| `ilp` | ILP (exact) | `optimizer.py:256` `solve_ilp` | Linearized QKP solved with PuLP + CBC. |

`solve_all` (`optimizer.py:326`) runs the three heuristics and optionally the ILP.

### Shared greedy loop (`optimizer.py:218` `_greedy`)

Maintains a running `penalty` vector (`Σ_{j∈S} sim(i,j)` for every candidate), updated with
one vectorized add per iteration. Per iteration:

1. `fits = available & (tokens <= remaining)`; break if nothing fits.
2. `gain = relevance − λ · penalty`.
3. `objective = gain / tokens` if token-aware, else `gain`.
4. Mask infeasible entries to `−inf`, take `argmax`.
5. **Break if `gain[best] <= 0.0`** (`optimizer.py:243`).
6. Otherwise select, decrement budget, mark unavailable, add `similarity[best]` to `penalty`.

### ILP formulation (`optimizer.py:292-311`)

```
max   Σ_i r_i x_i  −  λ Σ_{i<j} s_ij y_ij
s.t.  Σ_i w_i x_i ≤ W_max
      y_ij ≥ x_i + x_j − 1
      x_i ∈ {0,1},  y_ij ∈ [0,1] continuous
```

`y` is left continuous because its objective coefficient is non-positive; the usual
`y ≤ x_i`, `y ≤ x_j` constraints are omitted as redundant. Pairs with `s_ij <= 0` are
skipped entirely. Solver: `pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)`.

---

## 3. Objective function as implemented

`optimizer.py:143` `ContextKnapsack.score`:

```
Score(S) = Σ_{i∈S} r_i  −  λ · Σ_{i<j∈S} s_ij
```

computed as `relevance[idx].sum() − λ · (similarity[ix_(idx,idx)].sum() / 2)`. The halving
is valid because the stored similarity matrix is symmetric with a zeroed diagonal.

Returns the triple `(score, relevance_sum, redundancy)`.

### Pre-processing applied to the inputs (`optimizer.py:132-140`)

On construction, `ContextKnapsack` takes a defensive copy of `similarity` and then:
symmetrizes (`0.5 * (S + S.T)`), zeroes the diagonal, clips negatives to `0`.
It then computes `feasible_mask = (tokens <= budget) & (tokens > 0)`; everything outside
that mask is pruned once and recorded in `infeasible_indices`.

Note that `embedder.build_corpus` has already zeroed the diagonal and clipped negatives
(`embedder.py:265-267`, `clip_negative_similarity=True` by default), so this is applied twice.

---

## 4. Benchmark structure

### Encoding (`embedder.py`)

| Stage | Implementation |
|---|---|
| Embeddings | `SentenceTransformerBackend`, `all-MiniLM-L6-v2`, device `cpu`, batch 64, `normalize_embeddings=False` (`embedder.py:108`) |
| Fallback | `HashingBackend` — hashed bag-of-words + fixed Gaussian random projection, seed 42 (`embedder.py:75`). Substituted on **any** exception from the sentence-transformers path (`embedder.py:154-159`) |
| Normalization | Row-wise L2 with `eps=1e-12` (`embedder.py:180`) |
| Relevance | `rel(q,d_i) = cos(e_q, e_i)`, clipped to `[−1,1]` (`embedder.py:262`) |
| Similarity | `S = E·Eᵀ`, single BLAS call, diagonal zeroed, negatives clipped to 0 (`embedder.py:264-267`) |
| Token counting | `tiktoken` / `cl100k_base` (`embedder.py:35`). Heuristic fallback `round(1.3 × word-like units)` when tiktoken is absent |

### Corpus generation (`benchmark.py:94` `generate_corpus`)

Fixed query (`benchmark.py:107`):
> "How does attention-based retrieval fit inside a limited LLM context window?"

`TOPIC_BANK` (`benchmark.py:54`) holds 5 topics × 4 sentences. Query topics are
`("transformer attention", "vector retrieval", "context windows")`; the remaining two
topics supply distractors. `PADDING` (`benchmark.py:87`) is a fixed 3-sentence filler block.

Per document, with `rng = random.Random(seed)`:

| Roll | Share | Content |
|---|---|---|
| `< 0.45` | 45% | one on-topic sentence, verbatim (redundancy) |
| `< 0.75` | 30% | on-topic sentence + `PADDING × randint(2,6)` (token inefficiency) |
| `< 0.90` | 15% | 3 distinct on-topic sentences joined (dense multi-fact) |
| else | 10% | off-topic sentence + `PADDING × randint(0,2)` (distractor) |

Followed by `rng.shuffle(documents)`.

Observed corpus at the defaults (N=25, seed 42): 25 documents, 20 unique texts
(5 byte-identical duplicates), tokens min 12 / median 77 / max 205.

---

## 5. Hyperparameters and configuration

| Parameter | Value in v0 | Location |
|---|---|---|
| `λ` (redundancy weight) | **0.1** | `optimizer.py:39` `DEFAULT_LAMBDA` |
| Seed | 42 | `benchmark.py:33`, `embedder.py:27` |
| `n_docs` | 25 | `benchmark.py:398` |
| Budgets `W_max` | `[256, 512, 1024, 2048]` | `benchmark.py:396-397` |
| `repeats` | 3 | `benchmark.py:400` |
| ILP time limit | 300 s | `benchmark.py:402` |
| `ilp_max_docs` | 0 (no restriction) | `benchmark.py:403` |
| Coverage cluster threshold | 0.85 | `benchmark.py:158` |
| Encoder | `all-MiniLM-L6-v2` | `embedder.py:25` |
| Tokenizer encoding | `cl100k_base` | `embedder.py:26` |

The rationale recorded for `λ = 0.1` is the comment block at `optimizer.py:33-38`, which
cites statistics measured on the benchmark corpus, and `README.md:148-170` ("Finding 2").

---

## 6. Evaluation procedure (`benchmark.py:252` `run_benchmark`)

1. `np.random.seed(seed)` (`benchmark.py:258`).
2. Generate the corpus once; encode once via `build_corpus`. Encoding time is measured and
   printed but excluded from all reported latencies.
3. For each budget in `budgets`:
   a. Construct one `ContextKnapsack` over the full corpus.
   b. Run `solve_all(include_ilp=False)` `repeats` times, collecting each run's result.
   c. If ILP is enabled, run `solve_ilp_on_top_candidates` **once**.
   d. Merge: keep `results[0]` (the first run) as the selection, and overwrite its
      `latency_ms` with the median across repeats (`benchmark.py:291-293`).
   e. Choose the comparison reference (`benchmark.py:296-301`): the ILP score if
      `ilp.optimal`, with `reference_is_optimal=True`; otherwise `max(score)` across
      methods with `reference_is_optimal=False`.
   f. Convert each result to a `MethodMetrics` row via `evaluate` and print the block.
4. Dump all rows to JSON; optionally render the plot.

`solve_ilp_on_top_candidates` (`benchmark.py:220`): when `0 < ilp_max_docs < N`, it builds a
sub-problem over the top-`max_docs` documents by relevance, solves it exactly, maps indices
back to the full corpus, and **re-scores** against the full `ContextKnapsack`. The
`optimal` flag is carried over unchanged from the sub-problem, and `candidate_pool` is
recorded in `meta`.

---

## 7. Metrics reported

Defined on `SelectionResult` (`optimizer.py:45`) and collected into `MethodMetrics`
(`benchmark.py:137`).

| Metric | Definition | Location |
|---|---|---|
| `latency_ms` | Wall-clock solve time, median of `repeats` for heuristics, single run for ILP | `optimizer.py:188` etc. |
| `density` — "Context Density Score" | `1000 · Score(S) / Σ_{i∈S} w_i`, `0.0` when nothing is selected | `optimizer.py:88` |
| `token_savings_pct` | `100 · (1 − tokens_used / budget)` | `optimizer.py:83` |
| `score` | `Score(S)` as in §3 | `optimizer.py:143` |
| `relevance_sum` | `Σ_{i∈S} r_i` | `optimizer.py:148` |
| `redundancy` | `Σ_{i<j∈S} s_ij` | `optimizer.py:150` |
| `tokens_used`, `n_selected`, `budget` | Bookkeeping | `optimizer.py:157-167` |
| `unique_coverage` | Documents are greedily clustered over the **whole corpus** at `sim ≥ 0.85` in descending relevance order; the score is `covered_clusters / total_clusters` | `benchmark.py:157` |
| `score_vs_reference` | `result.score / reference_score`, `None` when `|reference| ≤ 1e-9` | `benchmark.py:197-198` |
| `reference_is_optimal` | Whether the reference came from an ILP run flagged `optimal` | `benchmark.py:296-301` |
| `proven_optimal` | `result.optimal` for this row | `benchmark.py:213` |

Plot panels (`benchmark.py:352-356`): density, token savings, redundancy, latency (log y).

---

## 8. How experiments are executed

Documented in `README.md:240-264` and `AGENTS.md`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python benchmark.py                      # default sweep + plots
python benchmark.py --no-ilp --no-plot   # fast smoke run
```

Scale-up form documented at `README.md:257`:

```bash
python benchmark.py --budgets 256 512 1024 2048 4096 --n-docs 60 --ilp-max-docs 30
```

Tests:

```bash
python tests/test_optimizer.py     # standalone runner
pytest tests/
```

Dependencies (`requirements.txt`): `numpy>=1.24`, `scipy>=1.10`, `sentence-transformers>=2.2`,
`tiktoken>=0.5`, `pulp>=2.7`, `matplotlib>=3.7`. `scipy` is not imported anywhere in the
repository.

---

## 9. Result artifacts present in v0

| Artifact | Tracked by git? | Note |
|---|---|---|
| `benchmark_results.json` | **No** — `.gitignore:8` | Raw metric rows for 4 budgets × 4 methods. Snapshotted to `docs/v0_results/benchmark_results.json`. |
| `benchmark_results.png` | Yes | Four-panel plot embedded at `README.md:236`. Also copied to `docs/v0_results/`. |
| `README.md` §4 table | Yes | `README.md:202-219`. |

### Reproducibility check performed at tag time

Ran `python benchmark.py --json-path <scratch>/v0_repro.json --no-plot` (output redirected
so no existing artifact was overwritten) and compared field-by-field against the committed
`benchmark_results.json`:

- **0 mismatches** across every non-latency field (score, relevance_sum, redundancy,
  tokens_used, n_selected, unique_coverage, density, token_savings_pct,
  score_vs_reference, flags), for all 4 budgets × 4 methods.
- Latency varies between runs as expected: ILP `251.7 → 201.4`, `2248.4 → 2115.4`,
  `7232.7 → 7119.0`, `3343.9 → 3471.2` ms.
- All 16 tests pass.

v0 is deterministic and reproducible in its selections, objectives and metrics.

---

## Known Issues To Address After v0

Each entry is verifiable from the repository at tag `v0-original-experiment`.
**None of these are fixed in v0. They are recorded here only so the repair can be
attributed and measured later.**

### I1 — ILP optimality detected with `LpStatus` instead of `sol_status`
- **File:** `optimizer.py:320` — `optimal=(pulp.LpStatus[status] == "Optimal")`
- **Current behaviour:** `LpStatus` reports that CBC returned *a* solution, not that it
  *proved* optimality. `sol_status` does not appear anywhere in the repository. When CBC
  stops on `timeLimit` with an incumbent, this evaluates to `True`.
- **Why it matters:** `optimal` propagates to `reference_is_optimal` (`benchmark.py:297-298`)
  and to the `vs opt` column header (`benchmark.py:310`). A timed-out incumbent is then
  published as ground truth, and every "% of optimum" figure derived from it is
  unverified. The `[t/o]` safeguard at `benchmark.py:322-331` and the claim at
  `README.md:278-280` depend on this flag and therefore never trigger.
- **Observed:** on an N=120 instance, `timeLimit` of 0.2 s / 2 s / 20 s returned objectives
  of −8.65 / 2.90 / 3.02, all three flagged `optimal=True`. Independently,
  `problem.sol_status` returned `2` (`LpSolutionIntegerFeasible`) rather than `1`
  (`LpSolutionOptimal`) for the truncated run.
- **Not affected:** the shipped N=25 table. Its four ILP rows return `sol_status == 1`,
  i.e. they are genuinely optimal.

### I2 — The method named "MMR" uses a summed penalty, not max-similarity MMR
- **File:** `optimizer.py:191-202` (`solve_mmr` → `_greedy(token_aware=False)`), selection
  rule at `optimizer.py:239`.
- **Current behaviour:** `argmax_i [ r_i − λ · Σ_{j∈S} s_ij ]`. Standard MMR is
  `argmax_i [ λ·r_i − (1−λ)·max_{j∈S} s_ij ]` — a **max**, not a sum, and a convex
  weighting.
- **Why it matters:** as implemented, this baseline is exact greedy ascent on the study's
  own objective (§3), so its near-optimality is structural rather than empirical. The
  repository therefore contains no genuine MMR baseline. `README.md:168-170` separately
  argues that sum-vs-max is a scale error, which the shipped baseline then commits.

### I3 — Top-K and the greedy methods use different stopping rules
- **Files:** `optimizer.py:181-187` (Top-K loop) vs `optimizer.py:243-246` (greedy break).
- **Current behaviour:** `_greedy` terminates as soon as the best marginal gain
  `r_i − λ·Σ_{j∈S} s_ij` is `<= 0`. `solve_top_k` has no such rule: it continues down the
  ranking and keeps appending any document that fits the remaining budget.
- **Why it matters:** the comparison bundles three differences at once — the redundancy
  penalty, the `1/w_i` normalization, and the stopping rule — with no ablation isolating
  them. Since the reported objective penalizes pairs, a method without a stopping rule is
  penalized quadratically in `|S|` for spending its budget.
- **Observed:** at `W_max=2048`, granting Top-K the same stopping rule moves it from
  `−2.217` to `+2.531`; removing the rule from the greedy moves it from `+2.768` to
  `−2.063`. Over 40 paired cells (20 seeds × 2 budgets, N=30) the greedy-minus-Top-K mean
  gap falls from `+1.309` to `+0.197` once the rule is matched, while remaining
  significant (37/40 wins).

### I4 — Restricted-pool ILP carries a global-optimality flag
- **File:** `benchmark.py:241-248`; consumed at `benchmark.py:296-299` and `322-324`.
- **Current behaviour:** the sub-problem's `optimal` flag is copied to the re-scored
  full-corpus result without qualification. `run_benchmark` then sets
  `reference_is_optimal=True` and `_print_budget_block` prints the `vs opt` header. The
  `*` marker is used for both a globally proven optimum and a pool-restricted one.
  A note distinguishing the two is printed once at `benchmark.py:269-272`.
- **Why it matters:** a pool-restricted solution is a lower bound on the true optimum, not
  an optimum. Ratios against it can exceed 1.
- **Observed:** N=40 with `--ilp-max-docs 12` produced ratios of 1.026 (greedy) and 1.045
  ("MMR") against a reference flagged optimal. `README.md:257` recommends this form of
  the command.

### I5 — Context Density Score is maximized by very small selections
- **File:** `optimizer.py:88-90` — `1000.0 * self.score / self.tokens_used`.
- **Current behaviour:** the numerator carries an `O(|S|²)` penalty while the denominator
  grows `O(|S|)` in tokens, so the ratio rises as the selection shrinks.
- **Why it matters:** it is used as a headline comparison metric (`README.md:121-122`,
  plot panel 1, and the `density` column) and to declare wins at `README.md:206` and `:210`.
- **Observed:** at `W_max=2048` the reported values are Top-K `−1.096`, MMR `5.876`,
  token-aware `5.888`. A selection of the **single** highest-density document scores
  `40.381` on the same metric (coverage `0.06`). 8 of the 11 possible single-document
  removals from the greedy selection *increase* its density. The margin the README bolds
  is `+0.012`.

### I6 — No reproducibility metadata recorded with results
- **Files:** `benchmark.py:382-388` (`dump_json`); `.gitignore:8`; `benchmark.py:258`;
  `embedder.py:154-159`.
- **Current behaviour:** the JSON payload contains only metric rows. Seed, `λ`, `n_docs`,
  budgets, backend name, encoder id, tokenizer, and library versions are not written.
  Backend name is printed to stdout only (`benchmark.py:265`). `benchmark_results.json`
  is gitignored, so results are not versioned with the code.
  `np.random.seed(seed)` at `benchmark.py:258` has no consumer — `generate_corpus` uses
  `random.Random` and `HashingBackend` uses `np.random.default_rng` — so the reproducibility
  requirement in `AGENTS.md` §3 is satisfied nominally but not mechanically.
- **Why it matters:** a results file cannot be attributed to a configuration, and the
  silent `except Exception` fallback to `HashingBackend` (`embedder.py:157-159`) would
  change every number with no trace in the artifact.

### I7 — Latency numbers are not directly comparable
- **Files:** `optimizer.py:278/317` (ILP timing spans model construction, LP file write,
  CBC subprocess spawn and solve) vs `optimizer.py:225/252` (in-process NumPy loop);
  `benchmark.py:280-293`.
- **Current behaviour:** heuristic latency is the median of `repeats=3` runs; the ILP is
  executed exactly once (`benchmark.py:285`), so its figure has no median despite
  `README.md:114` stating "median of 3 runs". Measured split at N=25/W=2048: ~6.5 ms model
  construction, ~3368 ms for the CBC leg including process startup.
- **Why it matters:** the `1/75,000` ratio at `README.md:172` compares an in-process loop
  against subprocess startup plus branch-and-bound, and is computed against the slowest
  of the four ILP rows.

### I8 — Relevance and token count are uncorrelated in the benchmark corpus
- **File:** `benchmark.py:94-131` (`generate_corpus`), padding applied at `:124` and `:129`.
- **Current behaviour:** padding is appended to on-topic sentences, but the resulting
  cosine relevance barely moves. Measured on the default corpus:
  `Pearson r(relevance, tokens) = 0.028`; `Spearman ρ = 0.017, p = 0.94`. Mean relevance is
  `0.346` for below-median-length documents and `0.384` for above-median-length ones.
- **Why it matters:** the stated motivation at `README.md:13-16` is that long low-density
  passages outrank and crowd out short high-value ones. In this corpus, length carries no
  signal about value, so the `1/w_i` term in the proposed method has no length-vs-value
  mismatch to exploit. This is the mechanism behind the result below.
- **Observed:** over 40 paired cells (20 seeds × 2 budgets, N=30), token-aware greedy minus
  "MMR" gives mean `+0.0386`, 15/40 wins, Wilcoxon `p = 0.61` — indistinguishable from
  zero, with the win count against the proposed method. In the shipped table the
  token-aware method's score is lower than "MMR" at all four budgets.

### I9 — λ selected on the evaluation corpus
- **Files:** `optimizer.py:33-39` (the justification comment cites benchmark-corpus
  statistics); `README.md:148-170`.
- **Current behaviour:** `DEFAULT_LAMBDA = 0.1` is chosen from a sweep run on the same
  generator and seed used to produce the reported results. No held-out corpus exists in
  the repository.
- **Why it matters:** the procedure cannot separate a tuned value from a discovered one,
  and `λ` directly controls both headline quantities (token savings and the magnitude of
  Top-K's negative score).
- **Observed:** across `λ ∈ {0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}` at `W_max=2048`,
  `λ = 0.1` is the global minimum of `greedy − mmr` (`−0.053`) while `greedy − top_k`
  grows monotonically with `λ` (`−0.044` at `λ=0` up to `+102.6` at `λ=1.0`). In the
  `λ` table at `README.md:161-166`, coverage falls from `0.53` to `0.27` between `λ=0.05`
  and the selected `λ=0.10`.

### I10 — Reporting discrepancies against the code and the shipped results
Each verifiable by direct comparison:

1. **ILP latency at `W_max=256`.** `README.md:207` reports `721.40` ms. The committed
   `benchmark_results.json` records `251.68` ms for that cell; a rerun gave `201.4` ms.
   Every other field in that table row matches the JSON exactly, so the latency column
   does not come from the same run as the rest of the table.
2. **Abstract ILP range.** `README.md:26` states "2.3-7.4 s". The range in the shipped
   results is 0.25-7.2 s; the quoted lower bound excludes the `W_max=256` row entirely.
3. **"All methods agree" at `W_max=256`.** `README.md:223-224`. The table one screen
   above (`README.md:204-207`) shows the token-aware method at `2.584` against `2.626`
   for the other three — it is the only method that misses the optimum at that budget.
4. **"Gap consistently ≤ 1.9% across every budget tested"** (`README.md:180`). True for the
   four budgets reported. Sweeping 14 budgets on the same corpus, all ILP-proved, gives a
   worst-case ratio of `0.938` at `W_max=128` (6.2% gap).
5. **"5.9x reduction in wasted context"** (`README.md:191`). `110.683 / 18.895 = 5.86`, but
   the two selections have 276 and 55 pairs respectively. Per-pair mean similarity is
   `0.401` (Top-K) vs `0.344` (greedy) — a 1.17x reduction. Top-K's per-pair figure equals
   the corpus-wide mean of `0.405`.
6. **Budget sweep independence** (`README.md:227`). The greedy and "MMR" selections at
   `W_max` 512, 1024 and 2048 are byte-identical index sets; the solution saturates at 480
   tokens. The four-budget sweep contains two distinct solver outcomes.
7. **"MMR ties the ILP here because the corpus mixes long and short variants of the same
   facts"** (`README.md:232-234`). On a corpus built from the same `TOPIC_BANK` with no
   padding at all (token range 8-16), "MMR" still matches the ILP exactly (ratio 1.000)
   across 4 seeds, so the stated cause is not required to produce the effect.
8. **Scaling notation.** `README.md:178` describes the ILP as `O(N²)` binaries; the
   implementation uses `N` binaries and `O(N²)` *continuous* variables, as the docstring
   at `optimizer.py:268-272` correctly states.

### I11 — Minor items (recorded for completeness, not scientific claims)
- `optimizer.py:240`: `gain / self.tokens` divides before masking, so a zero-token document
  raises `RuntimeWarning: divide by zero`. The result is still correct (masked at `:241`).
- `optimizer.py:117`: float token counts are truncated by the `int64` cast.
- `benchmark.py:291`: the variable named `best` is `results[0]`, the first run, not the
  best; its selection is paired with a median latency drawn from other runs. Harmless while
  the solvers are deterministic.
- Dead code: `embedder.py:67` `count_tokens()` is never called; `Tuple` is imported unused
  at `embedder.py:21`; `EncodedCorpus.embeddings` is stored but never read by any solver or
  metric; `scipy` is required but never imported.
- `tests/test_optimizer.py:140-143`: `test_determinism` compares two knapsacks built from
  the same literal seed and cannot fail.
  `tests/test_optimizer.py:118-127`: the ILP upper-bound test returns silently when PuLP is
  missing or when `exact.optimal` is `False`, and that flag is unreliable per I1.

---

## Attribution of the issues

| Category | Issues |
|---|---|
| Implementation | I1, I2, I11 |
| Experimental design | I3, I7, I10.4, I10.6 |
| Benchmark construction | I8, I10.7 |
| Metric choice | I5, I10.5 |
| Statistical methodology | I9, I10.4 |
| Reproducibility / provenance | I6 |
| Reporting | I10 |

## What v0 establishes that survives scrutiny

Recorded so later work does not discard it:

- The ILP linearization is mathematically correct, and the four shipped ILP rows are
  genuinely proved optimal (`sol_status == 1`).
- The objective arithmetic (`optimizer.py:143-151`) is correct.
- Mean pairwise cosine among retrieved documents is `0.405`, and rises to `0.424` after
  removing the 5 duplicate texts — `README.md` Finding 1 is not a duplication artifact.
- Redundancy-penalized selection beats pure relevance ranking robustly, including against
  a Top-K baseline given a matched stopping rule (37/40 paired cells, Wilcoxon
  `p = 2.6e-11`).
- The experiment is deterministic and bit-reproducible in every non-latency field.
