# Final Validation Protocol (v7) — Real-Task Supporting-Fact Coverage

Status: PROPOSED (awaiting sign-off). Once signed off: LOCKED, hashed, write-once results.
Scope: FINAL empirical phase of the project. No further experiments after this one.

## 1. Question

Does token-aware greedy context selection improve evidence coverage under a fixed token
budget, on a real task with gold evidence annotations?

Evidence coverage is operationalised throughout this protocol as **supporting-fact coverage**,
defined in §6. That term is used consistently; "passage recall" is never used as a synonym for
it, and is reserved for a distinct secondary metric should one ever be reported.

## 2. Data

- HotpotQA dev, distractor setting (gold + 8 distractor paragraphs per question).
- Source: HuggingFace parquet `hotpotqa/hotpot_qa`, `distractor/validation`, downloaded once,
  SHA-256 recorded. No paid API, no generative evaluator.
- Sample: N = 500 questions, drawn by a fixed permutation of the dev set (RNG seed 7000),
  frozen before any outcome is computed.
- Exclusions (documented, counted, applied identically to all methods): questions whose
  supporting-fact annotation references a sentence index absent from the paragraphs; sentences
  with zero tokens. No re-sampling to replace excluded questions.

## 3. Candidate pool and quantities (identical across all methods)

- Candidate unit: sentence. Pool = every sentence of the 10 paragraphs of that question.
- w_i = token count, `tiktoken` `cl100k_base`.
- r_i = cosine(query, sentence), `all-MiniLM-L6-v2`, local and deterministic. Raw cosine,
  no clipping, no rescaling.
- sim(i,j) = cosine between sentence embeddings.
- Objective: Score(S) = Σ_{i∈S} r_i − λ_obj · Σ_{i<j ∈ S} sim(i,j), λ_obj = 0.1 (same as the
  locked core). MMR uses λ_mmr = 0.5.

## 4. Budgets

Let W_pool(q) be the total token count of **all** candidate sentences available for question q:

    W_pool(q) = Σ_{i ∈ pool(q)} w_i

W_pool is computed before any selection takes place, from the candidate pool of §3 alone. It is
identical across all six arms for the same question, and it does not depend on any selection
method, on any solver outcome, on the gold supporting-fact labels, or on previously selected
context.

Three pre-specified budgets, per question:

    ρ ∈ {0.10, 0.25, 0.50},   W_max(q, ρ) = floor(ρ · W_pool(q)),   so   ρ = W / W_pool

ρ measures **relative budget tightness**: the fraction of the available candidate text that may
be kept. It is not an absolute token budget, and W_max varies across questions by construction.

No other budget is analysed. No budget is added after seeing results.

## 5. Arms

Six arms are run: (1) `top_k`, (2) `greedy_objective` (GO), (3) `greedy_token_aware` (TA),
(4) `MMR`, (5) `token_aware + best-singleton guard`, (6) exact `ILP`.

**The only confirmatory comparison is TA vs GO.** Arms 1, 4, 5 and 6 are secondary, descriptive
benchmarks: they are reported as summary statistics only, and no confirmatory hypothesis test is
performed on them. They therefore add no members to any multiplicity family (§7).

Both confirmatory arms use the common stopping policy: a sentence is eligible iff unselected, it
fits the remaining budget, and its marginal gain Δ_i > 0. Ties broken by lowest pool index.
ILP runs at ρ = 0.25 on the first 100 sampled questions only, CBC, 60 s cap per instance, single
thread; only instances with proven optimal status enter the ILP summary.

## 6. Primary outcome and estimand

**Supporting-fact coverage** of a selected set S, for question q:

    C(S) = (number of gold supporting-fact sentences retrieved by S)
           / (total number of gold supporting-fact sentences for q)

so C(S) = |S ∩ G(q)| / |G(q)| ∈ [0,1], where G(q) is the set of gold supporting-fact sentences
annotated for q. The denominator is fixed by the annotation and never depends on the arm, the
budget, or the selection.

Primary estimand, per question q and budget ρ:

    Delta_C(q, ρ) = C(TA; q, ρ) − C(GO; q, ρ)

"Supporting-fact coverage" is the name of this metric everywhere in the protocol, the code and
the final report.

## 7. Statistics (pre-specified)

Hypotheses, at each budget ρ:  H0: Delta_C = 0   vs   H1: Delta_C ≠ 0  (two-sided).

For each of the 3 budgets, on the paired Delta_C:

- Primary test: two-sided paired sign-flip permutation test, 10,000 flips, RNG seed 7002,
  zeros retained (matching the machinery used in v6).
- Multiplicity: Holm across exactly the 3 TA-vs-GO budget tests, α = 0.05, two-sided.
  The family has exactly three members — ρ = 0.10, 0.25, 0.50 — and nothing else. The secondary
  arms of §5 and the diagnostic of §8 are outside it and require no further correction.
- Reported per budget: n, mean difference, median difference, 95% bootstrap CI of the mean
  (10,000 resamples, seed 7001), Cohen's d_z, matched-pairs rank-biserial correlation,
  Wilcoxon signed-rank as a secondary check, and the full Delta_C distribution
  (deciles + ECDF + fraction of ties, wins, losses).

No SESOI is defined. The report states explicitly that no externally established practical
threshold exists for supporting-fact coverage, and presents the full effect distribution
instead of a single significance verdict.

## 8. Diagnostic (exploratory, outside the confirmatory family)

Before selection, per question: D = 1 − Kendall τ_b(r, r/w) over the candidate pool.
Reported: Spearman correlation between D and Delta_C, per budget, with bootstrap CI.

D is exploratory and predictive only, never causal. No threshold search: no D* is estimated,
and no subgroup is defined by D unless a separate, explicitly labelled post-hoc section says so.

## 9. Determinism and governance

Fixed seeds (7000 sample, 7001 bootstrap, 7002 permutation); tie-break by index; single-thread
CBC; embeddings from the local cached model. Recorded: dataset SHA-256, code SHA-256, package
versions, wall-clock, per-arm latency. Results are written once to version-named files under
`results/final_validation/`; nothing is overwritten. The locked v6 / v6.2 artifacts are not
touched, not re-run, and not re-analysed.

## 10. Stop rule

This is the final empirical phase. After the run: final report, repository cleanup, limitations,
stop. A null result is a valid final result and is reported as such.
