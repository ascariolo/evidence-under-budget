# Protocol 6.2 — **LOCKED** (`6.2-locked`)

**The seed-count rule is locked.** Every owner check passes on the final protocol bytes. The
variance pilot has **not** been run. Nothing was computed from seeds 0–59 or 2000–2039 in
this step.

| artifact | sha256 |
|---|---|
| `experiments/elasticity/protocol.py` (6.2-locked) | `0fe3412c890697bddcb8009620387fbd58d3d8c433e4fbd72e2dbd02fee6c22e` |
| `experiments/elasticity/seed_count.py` | `2bbc392f6b4ec4d6d1c0e9c37a3b3c4062e43ed9a9d6bab5af704b605a01e8cd` |
| lock-candidate audit `results/elasticity/amendment_6_2_lock_candidate_audit.json` | `c62b9ac5b50c1077…` (recorded inside the protocol as the lock basis) |
| lock audit | `results/elasticity/amendment_6_2_locked_audit.json` |

Sequence: `6.2-lock-candidate` → full audit (all checks true, lockable) → promote to
`6.2-locked` → full audit again on the final bytes (all checks true). Audit files are
version-named and none was overwritten.

## L.1 Resolutions in this revision

**U-SC2a-Q — percentile convention (locked, not computed).** For β level j, sort the 60
values `D_{s,j}`, s∈B, ascending as `x₍₀₎ ≤ … ≤ x₍₅₉₎` (0-based). Then `h = (N−1)·p` with
`N = 60`, `lo = ⌊h⌋`, `frac = h − lo`, and `P_p = x₍lo₎ + frac·(x₍lo+1₎ − x₍lo₎)`; if h is
an integer, `P_p = x₍h₎`.

- **P10:** h = 5.9 → `x₍5₎ + 0.9·(x₍6₎ − x₍5₎)`.
- **P90:** h = 53.1 → `x₍53₎ + 0.1·(x₍54₎ − x₍53₎)`.
- `ΔD = mean over the 6 β levels of [P90(D_j,B) − P10(D_j,B)]`.

h is computed as an exact rational, and **no library percentile function is used**
(verified by source inspection). The code stops unless there are exactly 60 finite D values
per level.

**U-SC-H — rule h, option (i), locked wording.** *"The pilot population is insufficient only
when a required statistic is undefined or fails an explicitly declared mathematical/data-
quality condition. Token-CV rejections are applied using the existing locked acceptance
policy; rejected seeds are excluded from P and reported. A rejection does not by itself stop
the pilot. The calculation stops only if, after applying the acceptance policy, any required
statistic cannot be computed or its declared mathematical/data-quality requirements are not
satisfied."*

**P = the accepted seeds among 2000–2039.** The declared conditions form a closed list,
each item deduced from the existing definitions:

- **H1:** every seed receives exactly one disposition.
- **H2:** every accepted seed has the complete grid (6 β × 2 γ × 3 budgets × both methods),
  as the definitions of ρ_red, ρ_lvl and S_s require.
- **H3:** |P| ≥ 2, required by ddof = 1 and by the between-seed Pearson correlation. **No
  other minimum count.**
- **H4:** rules a, b, c, e, f and the general invariant.

**`embedder.py` — restored.** Rewritten byte-exact from git object blob `f76e9826183a`
(commit `8fe4d34b2106`); sha256 is `c22a456f2f46` again. The changed hash was **not**
re-recorded. The event is documented as **RL-4**.

## L.2 Lock audit results (`amendment_6_2_locked_audit.json`)

| check | result |
|---|---|
| code-integrity hashes match (optimizer, benchmark, **embedder**, controlled generator/protocol, elasticity generator) | **PASS** |
| 1 U-SC0 = R2 · 2 frozen Friedman formula · 3 fixed generator-only ΔD · 4 B = seeds 0–59 | **PASS** |
| 5 edge rules explicit · 12 all formulas and edge rules frozen, no unresolved item | **PASS** |
| 10 percentile method explicit (text, no library default, exact on synthetic values) | **PASS** |
| 11 rule H operational (wording, H1–H4, implemented, P definition) | **PASS** |
| 6 pilot→n map deterministic · 7 no pilot output can alter definitions | **PASS** |
| 8 seeds 2000–2039 untouched (no file under `results/` references them) | **PASS** |
| 13 no ΔD computed · 14 no budget-level reference ILP on B · 15 no pilot variance/power/n via F (no numeric value for any such key anywhere under `results/`) | **PASS** |
| 9 no experimental comparison outcomes generated (only outcome file: 6B `pilot_rows.json`, seeds 1000–1004) | **PASS** |
| frozen factors, Step 4/5 artifacts, confirmatory seeds, retired TOST | **PASS** |
| tests: `test_optimizer`, `test_controlled_generator`, `test_elasticity`, `test_seed_count` | **0 failures** |

## L.3 Historical computations disclosed (not violations, recorded for completeness)

- **6B preflight:** the *unconstrained saturation* ILP (W_sat) on the 720 instances of seeds
  0–59 (`preflight_saturation.csv`), for outcome-blind budget construction. This is a
  different quantity from the budget-level optimum OPT_b, which **has not been computed**.
- **6B preflight:** per-instance generator diagnostics, including D, on seeds 0–59. The **ΔD
  statistic has not been computed** from them.
- **6B.1 repair analysis:** variance components and illustrative required-n on the **earlier
  pilot seeds 1000–1004** (`design_repair_analysis.json`). That is not population P, not the
  frozen function F, and predates R2; no value from it enters F.

## L.4 Reproducibility limitations

- **RL-1:** no git commit hash (Xcode licence).
- **RL-2:** no hash baseline for `results/controlled/` before 6.1.
- **RL-3:** the 6.1 audit record was overwritten on 2026-09-16 (documented).
- **RL-4:** accidental whitespace edit to `embedder.py`, detected and restored.

## L.5 Status of the pilot

With status `LOCKED`, `protocol.assert_seed_count_rule_locked()` now **passes**, so the
protocol **permits** the variance pilot. It has **not been run**, per the owner's instruction.
When authorised, the sequence is fixed:

1. Reference ILP on B → `OPT_b`, `δ_b`.
2. `ΔD`, `E_Q` on B (generator-only).
3. Pilot on P: token-CV dispositions, rule H, pilot quantities.
4. Substitute into the frozen `n_final`.
5. Report n; stop before the confirmatory run.

---

# Protocol amendment 6.2 — revision r2 (`6.2-proposed-r2`)

## Status: **NOT LOCKED.** Two definitions are still underspecified. One locked file changed on disk outside this work. Variance pilot **not run**; no experimental data generated.

Implements the owner's r2 decisions. The earlier `6.2-proposed` text is kept below,
marked superseded.

| | |
|---|---|
| `experiments/elasticity/protocol.py` sha256 | `c46c4d31e859173a…` |
| `experiments/elasticity/seed_count.py` | new: the frozen function, pure (no data access) |
| `tests/test_seed_count.py` | new: 20 tests, **synthetic inputs only** |
| audit | `results/elasticity/amendment_6_2_proposed_r2_audit.json` (version-named, refuses overwrite) |

---

## r2.1 U-SC0 — R2 (locked by owner)

**Before the pilot:** the complete deterministic function
`F : (π_b, M2_b, ρ_red,b, ρ_lvl,b, σ_S) ↦ n_final` is frozen, together with every **fixed**
quantity it uses: `δ_b`, `ΔD` and `E_Q`, all computed on B. Also frozen: the SESOI, the
tests, the effect-size mappings, power 0.80, α = .05/9 with Holm, every formula, the
minimum-60 rule, and all edge-case rules.

**After the pilot:** the observed pilot quantities are substituted into `F`. The pilot's
role is **parameter estimation only**.

In code: `seed_count.py` separates (1) the **fixed layer** (`fixed_*`, population B), (2)
the **pilot layer** (`pilot_quantities`, population P) and (3) the **map**
`n_final(fixed, pilot)`. `FixedQuantities` is immutable. AST checks verify that the fixed
layer references no pilot name and that the pilot layer references no SESOI, α, power,
contrast or ARE constant.

## r2.2 Populations and constants

- **P** = seeds 2000–2039 accepted under the locked token-CV policy.
- **B** = seeds 0–59, fixed and **independent of n**.
- k = 6; α = .05/9; power 0.80; nonzero means |x| > 1e-12; ρ tolerance 1e-10; n searched
  over 1…300; `n_final = max(60, maxₜ nₜ)`.

## r2.3 Fixed quantities (baseline population B)

- **U-SC5a, option (i):** `OPT_b = mean of the proven ILP optimum over s∈B × 6 β × 2 γ`
  (720 instances), using `W_max = build_budgets(ρ_b·W_sat)`, then `δ_b = 0.03·OPT_b`.
  Reference solver only. **This breaks the circularity: n does not determine the population
  that defines δ.** Running the reference ILP on B before n is locked is permitted.
  Heuristics, Δ_gap, and any treatment outcome on B are not.
- **U-SC2a, option (b):** `ΔD = mean over the 6 β levels j of (q90_j − q10_j)`, where the
  quantiles are taken of `D_{s,j}` over s∈B (one D per (s, β); D does not depend on γ or
  budget). This is a **generator-population-derived scale chosen before the variance
  pilot**; no alternative D contrast may be searched after the pilot. **Quantile method:
  unresolved (U-SC2a-Q).**
- `E_Q = mean over s∈B of Q_s`, with `Q_s = Σ_j Σ_γ Dc²_{s,j}` and Dc centred within β level
  over B.

## r2.4 Pilot quantities (population P)

- `π_b` = share of instances with |x| > 1e-12 at budget b.
- `M2_b` = mean of x² over the nonzero instances.
- `ρ_red,b` = Pearson over the pairs (x_{s,j,low,b}, x_{s,j,high,b}).
- `ρ_lvl,b` = mean of the 15 pairwise Pearson correlations between β columns of the
  seed-level redundancy-mean x.
- `σ_S` = sample SD (ddof = 1) of the T8 scores S_s at tight, with Dc and Y centred within
  β level over P.

## r2.5 The frozen map

- `σ²_b = (π_b·M2_b − δ_b²)(1 + ρ_red,b)/2`
- **T2–T4:** `n = max(1, ⌈(z₂·σ_tight/δ_tight)²⌉)`, with `z₂ = z_{1−α/2} + z_{0.80}`.
- **T6, T7 (tight), T10 (medium):** `L = δ_b·Σc²/(max c − min c)`;
  `sd = √(σ²_b·Σc²·(1 − ρ_lvl,b))`; `n = max(1, ⌈(z₂·sd/L)²⌉)`.
- **U-SC1, T1 (tight) and T9 (loose), Friedman, no hand-picked W:**
  - `f = (δ_b/σ_b)·√(1/(2k))` — Cohen's least-favourable pattern for range δ.
  - `ARE = 0.955·k/(k+1) = 0.8186`.
  - `λ(n) = ARE·n·k·f²/(1 − ρ_lvl,b)`.
  - `power(n) = P[χ′²(k−1, λ(n)) > χ²_{1−α; k−1}]`.
  - `n = min{n ∈ 1…300 : power(n) ≥ 0.80}`.
  - The implied `W_min = ARE·k·f²/((k−1)(1 − ρ_lvl,b))` is derived, never chosen. The 3%
    SESOI stays the underlying profile effect; the pilot supplies only σ and ρ_lvl to
    standardise it.
- **T8:** `b_min = 0.03/ΔD`. **0.03 is a probability-point difference across the ΔD
  contrast, not a slope of 0.03.** `n = max(1, ⌈(z₁·σ_S/(b_min·E_Q))²⌉)`, with
  `z₁ = z_{1−α} + z_{0.80}`. The estimand is unchanged: the within-β slope of divergence
  probability on D.
- `n_final = max(60, maxₜ nₜ)`. The pilot can raise n above 60 but can never lower it.

## r2.6 Edge-case rules (all raise `SeedCountStop`)

| rule | content |
|---|---|
| a | `π_b·M2_b − δ_b² ≤ 0` → STOP. The −δ² term is never dropped. |
| b | a required Pearson correlation is undefined (zero-variance vector / not finite) → STOP. No substitution. |
| c | require `−1/(k−1) ≤ ρ_lvl < 1` within tolerance 1e-10; materially outside → STOP; no clamping, no `n_t = 1`. Values within tolerance are used **unchanged**. A value that leaves `1 − ρ_lvl ≤ 0` is invalid → STOP. |
| d | n searched over 1…300; power < 0.80 at 300 (or a closed-form n > 300) → STOP, redesign |
| e | nonzero means \|x\| > 1e-12 |
| f | σ_S uses ddof = 1 |
| g | all centring is within β level; σ_S over P; E_Q over B |
| h | locked token-CV policy; rejected seeds are reported, never silently dropped; **"insufficient" is unresolved (U-SC-H)** |
| i | an unproven reference ILP in OPT_b → STOP; never substituted |
| general | *No edge-case rule may replace an undefined or mathematically invalid statistical quantity with an arbitrary numerical value merely to permit continuation. Degenerate or invalid inputs cause the seed-count calculation to stop.* Applied to σ²_b ≤ 0, σ_S ≤ 0, ΔD ≤ 0, E_Q ≤ 0, OPT_b ≤ 0, and any non-finite value. |

Rules a, b, c, d, i and the general invariant are each exercised by a synthetic-input test.

## r2.7 Remaining ambiguity — protocol NOT lockable

| id | issue | why it matters |
|---|---|---|
| **U-SC2a-Q** | ΔD needs the 10th and 90th percentiles of 60 values per β level, but the **quantile definition** is not specified (linear interpolation / Hyndman–Fan type 7, type 6, nearest-rank, lower, …) | different definitions give different ΔD, and therefore a different n for T8. Not chosen; `fixed_delta_D` raises `SeedCountNotLocked`. |
| **U-SC-H** | rule h's **"insufficient pilot population"** has no operational definition. (i) Insufficient only when a required statistic becomes undefined or invalid — already caught by rules a, b, f and the general invariant — so rejected seeds are reported and the calculation continues. (ii) Any token-CV rejection in P stops it. | P(≥1 rejection among 40 seeds) ≈ 10% at the measured 0.255% per-seed rate. Not chosen; `assert_pilot_sufficient` raises `SeedCountNotLocked`. |

## r2.8 Audit — owner items 1–9

| # | check | result |
|---|---|---|
| 1 | U-SC0 is R2 | **PASS** |
| 2 | U-SC1 is the frozen Friedman formula (protocol text and code agree) | **PASS** |
| 3 | U-SC2a has the fixed generator-only ΔD definition | **PASS** (definition); **FAIL** — quantile method unresolved |
| 4 | U-SC5a uses fixed seeds 0–59 | **PASS** |
| 5 | all edge-case rules explicit | **PASS** (rules a–i and the general invariant); **FAIL** — rule h not operational |
| 6 | pilot→n mapping deterministic once pilot quantities are supplied | **PASS** — `n_final(fixed, pilot)` is the only input path; repeated evaluation on synthetic inputs is identical; no randomness |
| 7 | no pilot output can alter the SESOI, contrast, test or formula | **PASS** — AST layer separation; the protocol formula text contains no pilot-seed reference |
| 8 | seeds 2000–2039 untouched | **PASS** — no file under `results/` references them |
| 9 | no experimental comparison outcomes generated | **PASS** — the only outcome-bearing file is `pilot_rows.json` (seeds 1000–1004, written 2026-09-16 09:52 UTC in 6B) |
| — | frozen factors, Step 4/5 artifacts, confirmatory seeds, retired TOST | PASS |
| — | tests | `test_optimizer`, `test_controlled_generator`, `test_elasticity`, `test_seed_count`: 0 failures |
| — | **code integrity** | **FAIL — `embedder.py`** (§r2.9) |

`protocol_lockable = false`.

## r2.9 Unexpected change to a locked file: `embedder.py`

The audit's hash lock fails for `embedder.py`: it is now `b528295b3a71`, against the
recorded `c22a456f2f46`, with modification time 2026-09-16 15:53 local.

- **No command in this work edited `embedder.py`.** The file was open in the IDE.
- The committed version was read from the `.git` object store (git itself is blocked,
  RL-1). The only difference is **one added line containing four spaces**, before
  `if n == 0:` in `build_corpus`.
- **The parse trees of the committed and working versions are identical**, so the change
  is behaviour-neutral.
- It was **not reverted**. The owner decides: restore the committed bytes (the hash lock
  then passes unchanged), or re-baseline the recorded hash with a documented reason.

## r2.10 Reproducibility limitations kept

- **RL-1:** no git commit hash (Xcode licence).
- **RL-2:** no hash baseline for `results/controlled/` before 6.1.
- **RL-3:** the 6.1 audit record was overwritten on 2026-09-16 (documented; the surviving
  record is in `docs/elasticity_amendment_6_1.md`).

## r2.11 Next step

Owner decides **U-SC2a-Q** (quantile definition), **U-SC-H** (operational meaning of
"insufficient") and the **`embedder.py`** disposition. Then re-audit. The rule may be
marked LOCKED only when the audit shows every owner check passing and no unresolved item.
Still no seeds 2000–2039, no reference ILP on B, and no ΔD computed until then.

---

# SUPERSEDED — `6.2-proposed` (r1) text below, retained for provenance

# Protocol amendment 6.2 (PROPOSED) — seed-count rule: resolution attempt

## Status: `6.2-proposed`. Rule **NOT locked**. Variance pilot **NOT run**. No experimental data generated.

The decisions were written into the protocol. Four statistical definitions still cannot be
written down without guessing, or without using pilot outputs, so they are recorded as
unresolved.

---

## 1. Exact changes

### Protocol (`experiments/elasticity/protocol.py`, sha256 `535e8313b068c451…`)

- **Version:** `6.2-proposed` (previous `6.1-amended`). No experimental factor changed: 39
  frozen constants are identical to the 6.0 lock, and only the version string and family
  size differ, as expected.
- **`DECLARATIONS_6_1`:**
  - RNG seed 61000: CONFIRMED.
  - G4′ failure stops the design: CONFIRMED.
  - Divergence means different document sets: CONFIRMED.
  - T8 linear-mixed-model carry-over: **NOT CONFIRMED**, re-specified below.
- **`N_MIN_CONFIRMATORY_SEEDS = 60`** (U-SC4): `n_final = max(60, max_t n_t)`, and STOP if
  it exceeds 300.
- **`T8_SPEC`** — frozen before any pilot:

| | |
|---|---|
| outcome | `Y ∈ {0,1}` per (seed, β, γ) at tight: 1 iff the selected document **sets** differ (instance level, both redundancy levels) |
| predictor | `Dc = D − mean_s D` within the β level |
| estimand | within-level slope of `P(Y=1)` on D, **in probability points per unit D** (identity link, level fixed effects) |
| estimator | `b̂ = Σ_s S_s / Σ Dc²`, with `S_s = Σ_{β,γ} Dc·(Y − Ȳ_β)` |
| test | one-sided seed-cluster **sign-flip test on per-seed scores `S_s`**, 10,000 flips, RNG seed 61000 |
| H₀ / H₁ | b = 0 / b > 0 |
| why identity link | The SESOI is declared on the absolute probability scale. A logistic model estimates a log-odds slope, and converting that to probability points depends on the fitted outcome probabilities, i.e. on the data. |
| why cluster sign-flip | No Gaussian random-effect assumption for a bounded binary outcome; seeds are the independent unit; consistent with the rest of the family |
| rejected | Gaussian mixed model on the seed-level share (6.1); logistic GLMM |

- **`SEED_COUNT_RULE`:** every per-test formula is written in full; none is evaluated.
  - **Unified SESOI:** `δ_b = 0.03·OPT_b` is the smallest absolute difference of interest
    between two expected values of Δ_gap at budget b — a level mean and zero (T2–T4), or the
    extreme values of a hypothesised profile (T1, T6, T7, T9, T10).
  - **T2–T4:** `n = ⌈(z·σ_seed,tight / δ_tight)²⌉`, with
    `σ²_seed = (π·E[X₁²] − δ²)(1+ρ_red)/2`. *Specified, conditional on U-SC0 and U-SC5a.*
  - **T6/T7/T10 (U-SC3):** pure pattern `m_k = δ_b·c_k / (max c − min c)`.
    - Contrast value: `L = δ_b·Σc²/(max c − min c)`, i.e. 7δ for linear and (84/9)δ for
      quadratic.
    - Score sd: `σ_seed·√(Σc²(1−ρ_lvl))`.
    - The mapping is written out and justified explicitly (for the quadratic, the range is
      the depth between the end levels and the middle levels), **not imported from the
      scratch script**. *Specified, conditional on U-SC0 and U-SC5a.*
  - **T1/T9 (U-SC1):** Cohen's (1988) least-favourable pattern for range δ, giving
    `f = (δ/σ_seed)·√(1/2k)`.
    - Friedman power via ARE (0.955·k/(k+1)): `λ_F = ARE·n·k·f²/(1−ρ_lvl)`, equivalently
      `W_min = ARE·k·f²/((k−1)(1−ρ_lvl))`.
    - n is the smallest value with noncentral χ² power ≥ 0.80.
    - Limitation: ties from zero-inflation make this n optimistic.
    - *UNRESOLVED — see U-SC0/U-SC1.*
  - **T8 (U-SC2):** `b_min = 0.03 / span_D`, and
    `n = ⌈(z₁·σ_S / (b_min·E[ΣDc²]))²⌉`. `E[ΣDc²]` is generator-only; `σ_S` comes from the
    pilot. *UNRESOLVED — span_D.*
  - **Pilot output whitelist:** `π_b`, `E[X₁²]_b`, `ρ_red,b`, `ρ_lvl,b` per budget, and `σ_S`
    at tight. **Not from the pilot:** `OPT_b` (confirmatory population, U-SC5) and
    `E[ΣDc²]` (generator-only).
- **`REPRODUCIBILITY_LIMITATIONS`:**
  - RL-1: no git commit hash, because of the Xcode licence.
  - RL-2: no pre-6.1 hash baseline for `results/controlled/`.
  - RL-3: the audit-overwrite incident (§4).

### Analysis (`experiments/elasticity/analysis.py`, sha256 `1f0bb2f5e68dadb7…`)

T8 is now `cluster_signflip_score_slope` on instance-level binary divergence, and
`t8_cluster_scores` implements `T8_SPEC`. `signflip_pvalue` gains `alternative="greater"`.
No mixed model remains in the analysis module.

### Audit (`experiments/elasticity/audit_amendment.py`)

- Classifies `retired_tost` references by **AST-parsed imports**, **textual mentions**, and
  a **runtime probe** that imports the active 6.x modules and checks `sys.modules`.
- The legacy 6.0 runner is reported separately and verified to refuse to run.
- Writes to a **version-named** file and **refuses to overwrite** an existing audit record.

### Tests

`tests/test_elasticity.py`: 29 tests, 0 failures. New tests cover T8's binary outcome and
set inequality, recovery of a known slope on the probability scale, the frozen T8 spec
(no mixed model), one-sided sign-flip, and the rule being specified but not locked.

---

## 2. Remaining unresolved items

### U-SC0 — the premise every other item depends on

**Claim:** no mapping from an absolute SESOI to a standardised effect size can be
independent of the noise scale.

**Proof:** every standardised minimum effect used for power — Kendall's W, Cohen's f, a
contrast d — is a function of `δ/σ` (and of `ρ_lvl` for repeated measures). `δ` is fixed a
priori (3% of the optimum), but `σ` exists only as a pilot output. Scaling the noise by any
factor `c` leaves `δ` unchanged and multiplies the standardised effect by `1/c`, so any
σ-free function of `δ` would have to give the same standardised effect for every noise
scale — which no function can. ∎

The owner must choose one reading:

- **R2** — the pilot supplies σ, ρ_red and ρ_lvl as *inputs* to formulas fixed now. This is
  the premise of the variance pilot, and no choice depends on pilot values.
- **R1** — declare standardised minimum effects directly as a-priori numbers (for example
  `W_min`, `f_min`, `d_min`). That abandons the 3% SESOI as the power basis, but n then
  becomes computable with no pilot.

### U-SC1 — Friedman

As signed, it asks for both "map the 3% SESOI to W" **and** "independent of pilot variance".
By U-SC0 those cannot both hold. Under R2 it resolves to the formula already written into
the protocol. Under R1 the owner must declare `W_min`. **I have not chosen a W value.**

### U-SC2a — T8 span of D

T8 tests a *slope*, so "0.03 probability points" defines an effect only together with a
declared span of D. Options:

- per one within-level SD of D, computed **generator-only** on the confirmatory population;
- per a fixed ΔD;
- across the within-level IQR of D.

### U-SC5a — circularity in the SESOI population

`δ_b` uses `OPT_b` over the locked population seeds `0…n_final−1`, but `n_final` depends on
`δ_b`. Options:

- **(i)** `S_SESOI = seeds 0–59`, the guaranteed minimum population under U-SC4, which does
  not depend on n. This requires solving budget-level ILPs on confirmatory seeds 0–59 before
  n is locked. That uses the reference solver only — no compared method, no Δ_gap — but it
  is still a pre-run computation on confirmatory instances, and it needs explicit
  permission.
- **(ii)** fixed-point iteration over n — may not converge; rejected.
- **(iii)** redefine `OPT_b` (for example, from the unconstrained saturation optimum
  already computed outcome-blind).

---

## 3. Is the seed-count rule computable without pilot outputs?

**No, on two separate counts:**

1. **Its definition is incomplete.** U-SC0, U-SC1, U-SC2a and U-SC5a are open, so the
   function mapping inputs to n is not yet fully defined.
2. **Under R2, it can never be computed without pilot outputs, by design.** The formulas
   require σ, ρ_red and ρ_lvl, which exist only after the pilot. The requirement that can
   hold is the weaker one: *the function is fixed before the pilot and no choice depends on
   pilot values.*

Only under **R1** — directly declared standardised minimum effects, plus a resolution of
U-SC2a and U-SC5a — would n be computable with no pilot at all. In that case the variance
pilot would be unnecessary.

The code gate remains closed: `assert_seed_count_rule_locked()` raises.

---

## 4. Audit results, including an incident

| check | result |
|---|---|
| frozen factors vs 6.0 lock | PASS — only the version and family size changed; 39 constants identical |
| code integrity | PASS — `optimizer.py`, `benchmark.py`, `embedder.py`, controlled generator and protocol, elasticity generator: all hashes match |
| Step 4/5 artifacts | PASS — 5,400 rows; 1,080/1,080 ILPs proven optimal |
| confirmatory seeds 0–59 | PASS — no outcome for any seed below 1000 |
| seeds 2000–2039 | unused |
| retired TOST artifact | PASS — byte-identical; the analysis refuses it |
| `retired_tost` static imports | **only** `run_preflight.py`, the legacy 6.0 executable, which refuses to run under the current protocol (verified) |
| `retired_tost` textual references only | `analysis.py` (docstring), `tests/test_elasticity.py` (assertion string) |
| active analysis runtime loads `retired_tost` | **False** |
| active analysis dependencies on `retired_tost` | **none** |
| tests | `test_optimizer`, `test_controlled_generator`, `test_elasticity`: 0 failures |
| seed-count rule locked | **no** — U-SC0, U-SC1, U-SC2a, U-SC5a |
| git | unavailable (RL-1) |

**Incident RL-3 — audit record overwritten.** A patch to the audit script failed on one
anchor and was never written. The audit command that followed therefore ran the **old**
script, which overwrote `results/elasticity/amendment_6_1_audit.json` at
2026-09-16 10:19:43 UTC with a run under `6.2-proposed`.

- **Lost:** the original 6.1 audit bytes.
- **Preserved:** the overwritten file, renamed to
  `amendment_6_1_audit.OVERWRITTEN_20260916T101943Z_old_script_run_under_6_2_proposed.json`.
  The 6.1 values survive as transcribed in `docs/elasticity_amendment_6_1.md` (protocol.py
  sha `10a638f3580bb6e5`, analysis.py sha `a05662c0eff73d0c`), whose pointer has been
  corrected.
- **Not affected:** no experimental data. The Step 4/5 data are unchanged; only an audit
  record was lost.
- **Prevention:** the audit script now writes version-named files and refuses to overwrite.

---

## 5. No experimental data generated

No generator instance, solver run or pilot was executed for any confirmatory, pilot or
variance-pilot seed. Seeds 2000–2039 were not used. No variance, power or n was calculated.
The only processes run were the unit tests (synthetic inputs), the audit (hashing, static
analysis, an import probe) and the legacy runner's refusal check, which exits before doing
any work. Every data file under `results/elasticity/` keeps its pre-step modification time.

---

## 6. Next step

The owner decides **U-SC0 (R1 or R2)**, then U-SC1 (automatic under R2; `W_min` under R1),
**U-SC2a (span of D)** and **U-SC5a (i / iii, and permission to solve reference ILPs on
seeds 0–59 if (i))**. After that: set `SEED_COUNT_RULE.status = "LOCKED"`, bump to
`6.2-locked`, and re-audit. Only then may the variance pilot run — or, under R1, n is
computed with no pilot.
