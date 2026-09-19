# Step 6B — Elasticity experiment: implementation + preflight

## VERDICT: **FAIL — STOP.** Locked invariant G4 failed. The full experiment must not run.

37 of 38 locked invariants pass. One fails (G4), and three further problems that are
not locked-invariant failures threaten the design directly (§4). **No failure was
patched.** Each requires a new design decision.

Nothing in Step 4 or Step 5 was modified (Z1, Z2 pass). `experiments/controlled/*`
is reused read-only and hash-verified.

---

## 1. Files created

| path | role |
|---|---|
| `experiments/elasticity/__init__.py` | package |
| `experiments/elasticity/protocol.py` | frozen constants, **38 machine-checkable invariants**, **8 implementation decisions**, all written before any measurement |
| `experiments/elasticity/generator.py` | design A4 construction; reuses Step 4 RNG streams, embeddings, redundancy and budget rule read-only |
| `experiments/elasticity/analysis.py` | T1–T10 family; structure separated from inference; Holm; TOST margin rule; power; write-once margin guard |
| `experiments/elasticity/run_preflight.py` | preflight runner |
| `tests/test_elasticity.py` | 21 tests, all pass |
| `results/elasticity/preflight_report.json` | every invariant with pass/fail and detail |
| `results/elasticity/protocol_lock.json` | status `PREFLIGHT_FAIL_STOP` |
| `results/elasticity/preflight_generator_instances.csv`, `preflight_saturation.csv`, `pilot_rows.json` | raw diagnostics |
| `results/elasticity/tost_margin.json` | **written once; contingent — see §4.2** |

Runtime 0.6 min. `git` metadata is recorded as `unavailable`: `git` is currently blocked
on this machine by an unaccepted Xcode licence. Not faked.

## 2. Implementation decisions (resolved before measuring; all need sign-off)

| id | ambiguity in rev. 2 | decision |
|---|---|---|
| IMPL-1 | which latent is fixed | `η` (tokens) is the own latent: `w` bit-identical across all β and γ for a seed, so a CV rejection removes the whole seed and Friedman blocks stay complete |
| IMPL-2 | "unit = base instance" vs "pooled over redundancy; paired by seed" | value per (seed, β, budget) = mean over both redundancy levels; n = 60 seeds, matching the power analysis |
| IMPL-3 | no statistic named for T5–T8, T10 | contrasts: one-sample Wilcoxon on per-seed orthogonal-polynomial scores; T5: two one-sided Wilcoxon; T8: one-sided Wald from `mixedlm` |
| IMPL-4 | generator checks listed under a 5-seed pilot | run on confirmatory seeds 0–59, generator-only (3C.1 precedent). Pre-measurement arithmetic: SE over 5 seeds ≈ 0.25 > 0.15 tolerance |
| IMPL-5 | pilot seeds unspecified | 1000–1004, disjoint; confirmatory seeds outcome-blind during preflight |
| IMPL-6 | margin pooling / order of rounding | instance-level token-aware gaps at tight over all pilot instances; stability check before rounding |
| IMPL-7 | "the pilot's sd" undefined | sd of seed-level Δ_gap at tight, β = 2.0 (the T5 unit) |
| IMPL-8 | no tolerance beyond KS | added B3 (quantile tolerance 0.25σ) and D2 (usable-count difference ≤ 1.5) |

## 3. Invariant results

| id | result | key value |
|---|---|---|
| Z1 upstream hashes | PASS | all 4 match |
| Z2 Step 4/5 artifacts unchanged | PASS | 10 files byte-identical |
| A1 levels & ρ_c | PASS | ρ_c = β/2.5 exactly |
| **A2 β mean within 0.15** | PASS | worst \|dev\| 0.110 |
| **A3 β̂ > 1 at β = 2.0** | PASS | **60/60 (100%)** |
| B1 KS relevance | PASS | min p = 0.426 |
| B2 KS tokens | PASS | p = 1.000 (bit-identical, IMPL-1) |
| B3 quantile tolerance | PASS | log r max diff 0.103 (tol 0.313); log w 0.000 (tol 0.125) |
| C1 CV threshold | PASS | 0.30 |
| C2 rejection rate | PASS | **0 / 720** |
| C3 balanced | PASS | 0 per level |
| C4 no resampling | PASS | 60 accepted |
| D1 ≥ 4 usable docs | PASS | min 6 |
| D2 usable balanced | PASS | max level difference 0.38 |
| E1 D finite, in [0, 2] | PASS | range 0.107–0.569 |
| E2 within-level sd(D) > 0.02 | PASS | 0.065–0.080 |
| E3 D reproducible | PASS | 720/720 |
| F1 γ levels | PASS | 0.15 / 0.70 |
| F2 γ isolates similarity | PASS | 360/360 bit-identical r, w, β̂, D, topics |
| F3 γ raises similarity | PASS | 360/360; no overlap |
| G1 W_sat proven | PASS | 720/720 |
| G2 budget ordering | PASS | 720/720; max ρ deviation 0.0016 |
| G3 tight affords one doc | PASS | 0 failures |
| **G4 ≥ 2 docs at tight, both greedies** | **FAIL** | **12 / 120 violations** |
| G5 not all-identical | PASS | tight 0.33, medium 0.02, loose 0.20 |
| H1 ILP proven | PASS | 180/180 |
| H2 ILP runtime q95 < 5 s | PASS | q95 = 0.061 s |
| H3 no heuristic optimal | PASS | 0 / 720 |
| H4 objective accounting | PASS | 0 inconsistent; identity error 0.0 |
| I1 determinism | PASS | 3/3 probes identical, incl. ILP |
| J1–J5 T1–T10 structure | PASS | 10 tests; 1 or 6 levels; only the two greedies; redundancy pooled; exact rows |
| K1 margin write-once | PASS | written once |
| K2 m > 0 | PASS | m = 0.03 — **but see §4.2** |
| K3 power decision | PASS | `PENDING_USER_DECISION` |

### Realized β (confirmatory seeds 0–59, n = 60 per level)

| target | mean β̂ | sd | min | max | \|dev\| | share β̂ > 1 |
|---|---|---|---|---|---|---|
| −0.5 | −0.601 | 0.532 | −1.664 | +0.577 | 0.101 | 0.0% |
| 0.0 | −0.107 | 0.543 | −1.197 | +1.076 | 0.107 | 1.7% |
| +0.5 | +0.390 | 0.531 | −0.674 | +1.541 | 0.110 | 13.3% |
| +1.0 | +0.894 | 0.495 | −0.100 | +1.971 | 0.106 | 45.0% |
| +1.5 | +1.401 | 0.427 | +0.538 | +2.349 | 0.099 | 80.0% |
| **+2.0** | **+1.918** | 0.317 | **+1.276** | +2.637 | 0.082 | **100%** |

All six means sit ~0.1 below target, in the same direction. The levels share latents
per seed (IMPL-1), so one ensemble fluctuation moves all six together; the slope of
realized against target means is ≈ 1.01. Whether the offset is a small systematic bias
(from clipping/rounding) or a single correlated fluctuation was **not separately
verified**. A2 passes either way. Pilot seeds show smaller offsets (−0.53, −0.03, 0.47,
0.95, 1.44, 1.95), descriptive only.

**β > 1 is genuinely reached**, not merely targeted: every one of 60 instances at
β = 2.0 has β̂ > 1, minimum 1.276.

### Other diagnostics

- **Relevance marginal:** log r mean −2.99, sd 1.19–1.22 at every level. **Token marginal** identical at every level (IMPL-1).
- **Usable documents** (r > 0.05): mean 12.5, sd 2.5, min 6, max 20; below-threshold mean 12.5.
- **D:** by level 0.227 / 0.263 / 0.291 / 0.287 / 0.273 / 0.226 — peaks at β ≈ 0.5–1.0 as the design anticipated. 72% of values inside the expected [0.1, 0.3].
- **Redundancy:** low 0.033 vs moderate 0.122 mean similarity; paired difference +0.089, positive 360/360. Both levels are lower than in Step 4 because relevance is smaller.
- **W_sat:** mean 920, rising with β (707 → 1149). Unconstrained optimum holds 9.1 docs on average.
- **ILP:** saturation mean 0.035 s, budget solves q95 0.061 s.

## 4. Failures and design-level problems

### 4.1 G4 — locked invariant FAILED

**Criterion (design H9):** both greedy methods select ≥ 2 documents at the tight budget,
on every pilot instance.

**Observed:** 12 of 120 checks fail, all with **exactly 1 document selected** —
8 `greedy_objective`, 4 `greedy_token_aware`.

| β | −0.5 | 0 | +0.5 | +1.0 | +1.5 | +2.0 |
|---|---|---|---|---|---|---|
| violations (GO / TA) | 0/0 | 1/0 | 1/1 | 3/1 | 1/1 | 2/1 |

**Why this is not a heuristic malfunction.** The **ILP optimum itself** selects a single
document at tight in 7 of 60 pilot instances, and averages only **2.38 documents**
(distribution 1:7, 2:26, 3:24, 4:3). The tight budget (ρ = 0.25 of W_sat ≈ 230 tokens)
holds about two documents under this relevance marginal. At medium the ILP averages
5.1 documents; at loose, 8.1.

**Why the violating instances cannot simply be excluded.** Violations are concentrated
at **β ≥ 0.5** (11 of 12), which is exactly the regime the experiment exists to probe:
a value-greedy ranker buying one long, highly relevant document is the elasticity
mechanism itself. Excluding them would bias the β profile — seed selection correlated
with the manipulated factor.

**This also exposes tension in the guard itself.** G4 was written as a degeneracy check,
but a one-document optimum at β ≥ 1 may be a legitimate outcome rather than degeneracy.
Whether it is degeneracy or signal is a design question, not an implementation one.

### 4.2 The TOST margin rule is degenerate — `m = 0.03` was reached through float noise

**Not a locked-invariant failure** (K2 passes), but the rule did not work as designed.

- Token-aware greedy is **exactly optimal at tight in 47 of 60** pilot instances.
- Median gap = **0.0 exactly**. IQR = **2.78 × 10⁻¹⁷** — floating-point noise.
- The rule declares "unstable" when IQR > 2 × median. It evaluated `2.78e-17 > 0.0` →
  **True** → fallback `m = 0.03`.
- Without that noise, IQR would be 0, the check would read "stable", and the anchor would
  round to **m = 0.00**, failing K2.

So the value in `results/elasticity/tost_margin.json` is the literal output of the rule,
but it was decided by 3×10⁻¹⁷, not by any real instability. **The anchor "median gap of
the better method" is degenerate when that method is usually exactly optimal.** The file
is write-once and was not altered; it must not be relied on until the rule is redesigned.

### 4.3 T5 is infeasible at 60 seeds

sd of the T5 unit (seed-level Δ_gap, tight, β = 2.0) = **0.175** from 5 pilot seeds.
At m = 0.03 and n = 60, `m/SE = 1.33 < 1.645`, so **TOST power ≈ 0.00**.
**n ≈ 293** seeds would be needed for 0.80. Sensitivity: at half that sd, n ≈ 74; at
double, n ≈ 1171. The sd estimate rests on 5 seeds and is itself very uncertain.

Per the pre-registered rule this is **`PENDING_USER_DECISION`**: raise seeds to the
required n before the main run, or retain 60 and pre-declare T5 as underpowered and
inconclusive. Not decided silently.

### 4.4 Δ_gap is heavily zero-inflated at tight — a threat to T1–T8, not only T5

Instance-level Δ_gap is **exactly 0** in **40 of 60** pilot instances at tight
(28/60 medium, 35/60 loose); the two greedies select different sets in only 33% at
tight. Eight of the ten confirmatory tests use the tight budget. Wilcoxon with
`zero_method="wilcox"` discards zero differences and Friedman degrades with ties, so the
**effective sample size at tight is roughly a third of nominal**. The design's power
analysis (§G) assumed continuous, non-inflated differences and does not hold here.

## 5. Decisions required before any re-run

1. **G4.** Options: (a) reclassify G4 from a degeneracy guard to a reported descriptor,
   on the argument that one-document optima are legitimate at high β; (b) raise the tight
   ρ so the tight regime holds more documents — changes a locked factor and breaks
   comparability with Step 4; (c) raise N — changes a held-constant factor.
   **Excluding violating instances is not acceptable** (β-correlated).
2. **TOST margin rule.** The anchor is degenerate when the better method is usually
   exactly optimal. It needs a redefinition — e.g. a margin anchored to the objective's
   natural scale, or an explicit rule for a zero median — and a float tolerance in the
   stability check. The existing `tost_margin.json` must then be explicitly retired,
   not overwritten.
3. **T5 power:** raise seeds (~293 at the pilot sd) or pre-declare T5 inconclusive.
4. **Power for T1–T8 under zero-inflation:** the family's power needs to be re-estimated
   with the observed tie structure before a seed count is locked.
5. **Sign-off on IMPL-1 through IMPL-8.**

Items 1, 2 and 4 are interdependent: changing the tight ρ would also change the
zero-inflation and the margin anchor. They should be decided together.
