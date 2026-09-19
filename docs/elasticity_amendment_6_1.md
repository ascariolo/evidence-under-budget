# Protocol amendment 6.1 — implementation, audit, and pilot STOP

## Status: amendment implemented and audited. **Variance pilot NOT run** — the signed-off seed-count rule is incomplete.

The amendment was implemented exactly as signed off. A repository audit passes on every
item. The blinded variance pilot on seeds 2000–2039 was **not** run: the rule it exists to
evaluate cannot be computed for four of the nine confirmatory tests. Settling that after
seeing the pilot quantities would be a forking path, so the pilot is gated off in code
until the rule is locked.

No experimental data of any kind were generated in this step.

---

## 1. Protocol identity

| | |
|---|---|
| version | **`6.1-amended`** (previous `6.0-preflight`) |
| `experiments/elasticity/protocol.py` sha256 | `10a638f3580bb6e5…` |
| `experiments/elasticity/analysis.py` sha256 | `a05662c0eff73d0c…` |
| machine-readable record | `AMENDMENT_6_1`, `SEED_COUNT_RULE`, `RETIRED_ARTIFACTS`, `RETIRED_CONSTANTS`, `EXPLORATORY_ONLY`, invariant `status` fields |
| full audit | ~~`results/elasticity/amendment_6_1_audit.json`~~ — **overwritten 2026-09-16 10:19 UTC** by an old-script run under 6.2-proposed (incident RL-3); original bytes lost; the values in this document are the surviving 6.1 record |

## 2. What was implemented (owner decisions 1–12)

| # | decision | implementation |
|---|---|---|
| 1–2 | tight ρ = 0.25; ρ·W_sat budgets | unchanged constants |
| 3 | G4 descriptive | invariant `G4.status = "descriptive"` |
| 4 | G4′ locked | invariant `G4P`, locked. Semantics: a preflight invariant whose failure **stops the design**; it **never excludes instances**, because budget-dependent exclusion could correlate with β through W_sat |
| 5 | retire TOST | T5 removed. K1–K3 `status = "retired"`. TOST code moved to `retired_tost.py`, which the analysis never imports. TOST constants removed from the top level, values kept only in `RETIRED_CONSTANTS`. `tost_margin.json` kept byte-identical, with sidecar `tost_margin.json.RETIRED.json` recording its sha256. `analysis.assert_artifact_usable` raises on it. |
| 6 | 9 tests, Holm over 9 | T1–T4, T6–T10; `CONFIRMATORY_FAMILY_SIZE = 9` |
| 7 | sign-flip permutation, zeros kept | `signflip_pvalue`, 10,000 flips, `PERMUTATION_RNG_SEED = 61000` |
| 8 | T8 = divergence probability | per (seed, β), the share of the two redundancy levels whose selection **sets** differ; `mixedlm(y ~ C(beta) + Dc, groups=seed)`, one-sided Wald |
| 9 | exploratory only | `EXPLORATORY_ONLY`; test asserts none enter the family |
| 10 | SESOI | `SESOI_FRACTION_OF_OPTIMUM = 0.03` |
| 11 | cap | `MAX_CONFIRMATORY_SEEDS = 300` |
| 12 | freeze the rest | verified, §4 |

The 6.0 preflight runner is kept as a historical record and now refuses to run under 6.1
(its checks — T5, K1–K3, the 10-test family — no longer exist). Its only change is an
import redirected to `retired_tost` so that the refusal message is reached.

### Confirmatory family, exactly as coded

Unit: per (seed, β, budget), Δ_gap = score(token_aware) − score(greedy_objective), mean over
the two redundancy levels, **zeros retained**.

| id | budget | β levels | statistic | H₀ | H₁ |
|---|---|---|---|---|---|
| T1 | tight | all 6 | Friedman | 6 level distributions equal | some level differs |
| T2 | tight | +1.0 | sign-flip on seed means | mean Δ_gap = 0 | ≠ 0 |
| T3 | tight | +1.5 | sign-flip on seed means | mean Δ_gap = 0 | ≠ 0 |
| T4 | tight | +2.0 | sign-flip on seed means | mean Δ_gap = 0 | ≠ 0 |
| T6 | tight | all 6 | sign-flip on linear contrast scores (−5,−3,−1,1,3,5) | mean score = 0 | ≠ 0 |
| T7 | tight | all 6 | sign-flip on quadratic contrast scores (5,−1,−4,−4,−1,5) | mean score = 0 | ≠ 0 |
| T8 | tight | all 6 | slope of divergence share on centred D, mixed model | slope = 0 | slope > 0 |
| T9 | loose | all 6 | Friedman | equal | some level differs |
| T10 | medium | all 6 | sign-flip on quadratic contrast scores | mean score = 0 | ≠ 0 |

Holm over these nine. No pairwise β comparison exists anywhere in the family.

## 3. Items declared during implementation — for sign-off, not blocking

These are pre-data declarations that the signed-off text left at implementation level:

- **Permutation RNG seed = 61000.** The amendment required "a fixed RNG seed declared"; this
  is that declaration, made before any data.
- **T8 model family = linear mixed model** (a linear probability model), carried over
  literally from IMPL-3 with only the outcome replaced. A logistic mixed model is the
  obvious alternative. The text says "mixed model" without specifying.
- **G4′ is a stop-the-design invariant, not an exclusion rule** — the same semantics as
  every other locked invariant in this protocol.
- **Divergence = the index sets differ.** Selection order is ignored, as in the 6B.1
  analysis.

## 4. Audit results

| check | result |
|---|---|
| **Frozen factors vs the recorded 6.0 lock** | **PASS.** 39 constants identical. Changed only: `ELASTICITY_PROTOCOL_VERSION`, `CONFIRMATORY_FAMILY_SIZE`. Removed only the six TOST/planned-n constants, whose values are preserved exactly in `RETIRED_CONSTANTS`. Additions are amendment fields only. β levels, ρ_c, σ_r, σ_w, μ_r, μ_w, clips, N, d, K, γ, budgets, λ_obj, λ_mmr, CV guard, β tolerance, seeds, polynomial coefficients: unchanged. |
| **Code integrity** | **PASS.** `optimizer.py` facfb186ddcc · `benchmark.py` 5ce79b512ae3 · `embedder.py` c22a456f2f46 · controlled `generator.py` d7ccf7907794 · controlled `protocol.py` 566481725b48 · elasticity `generator.py` 351949f310ba — all equal to recorded values |
| **Step 4/5 artifacts** | **PASS, with a caveat.** 11 files, latest modification 2026-09-15 19:38 UTC (Step 5). `full_experiment.json`: 5400 rows, 1080/1080 ILP proven. **Caveat:** no byte-level hash baseline for `results/controlled/` was ever persisted — the 6.0 preflight compared hashes in memory and stored only the file count. Verification is therefore by modification time plus content QC. sha256 hashes are recorded **now** in the audit file as the baseline for every later step. |
| **Confirmatory seeds 0–59 untouched** | **PASS.** Every file under `results/elasticity/` was scanned: no compared-method outcome for any seed below 1000. Pilot rows contain seeds 1000–1004 only. The generator and saturation CSVs carry no outcome columns. |
| **Seeds 2000–2039 unused** | **PASS** |
| **Retired TOST artifact** | **PASS.** Byte-identical to the recorded sha256 (`f16f11017702…`). The analysis refuses the path, exposes no TOST functions, and no `TOST*` constant remains at protocol top level. Importing `analysis` does not load `retired_tost` (verified at runtime). The audit JSON's `modules_importing_retired_tost` field over-reports: it is a text search that also matches a docstring in `analysis.py` and an assertion string in the tests. The **only real import** is the historical 6.0 runner, which refuses to run. |
| **Tests** | **PASS.** `test_optimizer.py`, `test_controlled_generator.py`, `test_elasticity.py` — 0 failures (the elasticity suite is now 26 tests) |
| **Seed-count rule locked** | **FAIL → pilot gated off** |

`git` remains unavailable on this machine (Xcode licence not accepted), so no commit hash is
recorded.

## 5. Why the variance pilot was not run

The signed-off rule reads: *n = the maximum over the 9 confirmatory tests of the required n
at SESOI (3% of mean ILP optimum), power 0.80, α = .05/9, from
Var_seed = Var_inst·(1+ρ_red)/2; cap 300; confirmatory seeds 0…n−1.*

It cannot be evaluated as written:

| id | tests | undefined component |
|---|---|---|
| **U-SC1** | T1, T9 | No power formula and no SESOI exist for the Friedman omnibus test. |
| **U-SC2** | T8 | T8's outcome is a divergence **probability**. A SESOI of "3% of the ILP optimum" is on a different scale and defines no effect size for it. Any definition would also need extra pilot outputs — the variance of the divergence share and within-level sd(D) — that are **not** on the pilot's output whitelist. |
| **U-SC3** | T6, T7, T10 | The mapping from SESOI to a contrast pattern (a trend whose total change across the six levels equals δ) exists only in the 6B.1 scratch script, not in the signed-off text. |
| **U-SC4** | all | "0 … n−1" versus "extended only by that rule": whether n may fall **below** the preflight-validated 60 is not stated. |
| **U-SC5** | all | "3% of the mean ILP optimum" does not say the mean over what. Pooling over β and γ within a budget is the only reading that uses no per-β information, but it is a choice. |

Each item changes the n the pilot would report. Deciding any of them after seeing π, E[X₁²],
ρ_red, ρ_lvl and the optimum would let the decision depend on the data. The pilot is
therefore gated in code: `protocol.assert_seed_count_rule_locked()` raises while
`SEED_COUNT_RULE.status == "UNRESOLVED"`, and a test enforces it.

## 6. Decisions required to lock the rule (options, no recommendation made)

- **U-SC1 (Friedman):** (a) exclude T1/T9 from the maximum and state they are not powered by
  the rule; (b) power by simulation under a declared SESOI pattern (e.g. one level shifted by
  δ, or a linear trend with total change δ); (c) use the linear-contrast requirement as a
  documented proxy.
- **U-SC2 (T8):** (a) exclude T8 from the maximum and state it is not powered by the rule;
  (b) declare a probability-scale SESOI (e.g. an absolute change of X percentage points in
  divergence probability across a stated range of D) **and** extend the pilot whitelist to
  the variance of the divergence share plus within-level sd(D). The latter is a generator
  property, not an outcome.
- **U-SC3 (contrasts):** adopt the 6B.1 mapping explicitly (linear range 10, quadratic range
  9 over the orthogonal coefficients), or declare another.
- **U-SC4 (floor):** `n_final = max(60, n_required)`, or allow `n_final = n_required` below 60.
- **U-SC5 (optimum pooling):** pooled over β and γ within each budget, or another stated rule.

Also for sign-off: the §3 declarations.

## 7. Next step

1. Owner decides U-SC1 to U-SC5 and signs off the §3 declarations.
2. Record them in `SEED_COUNT_RULE` with `status = "LOCKED"`; extend the pilot output
   whitelist only if U-SC2(b) is chosen; bump the version to `6.2-seedrule-locked`.
3. Re-run `audit_amendment.py`. It must show `seed_count_rule_locked = true` with all other
   checks unchanged.
4. Only then implement and run the blinded variance pilot on seeds 2000–2039, with the
   whitelist enforced in code.
5. If n ≤ 300, report n and stop before the full experiment. If n > 300, stop: REDESIGN.
