# Controlled Benchmark — Implementation and Validation (Step 3B)

**Scope.** Generator + validation suite + pilot only. The full 40-seed × 24-cell
experiment has **not** been run. No research conclusion appears in this document.

Specification: `docs/controlled_benchmark_design.md`.

---

## 1. Files created

| Path | Role |
|---|---|
| `experiments/controlled/__init__.py` | package marker |
| `experiments/controlled/protocol.py` | **frozen** design constants: factor levels, marginals, thresholds, deviations. Single source of truth imported by generator, tests and pilot so they cannot drift apart. |
| `experiments/controlled/generator.py` | instance synthesis, validation, saturation/budget construction |
| `experiments/controlled/run_pilot.py` | pilot runner (validation only) |
| `tests/test_controlled_generator.py` | 28-test validation suite |
| `results/controlled/pilot_report.json` / `.csv` | pilot diagnostics |
| `docs/controlled_benchmark_implementation.md` | this document |

**Files deliberately NOT modified:** `optimizer.py`, `benchmark.py`, `embedder.py`,
`tests/test_optimizer.py`, `README.md`, the v0 tag and artifacts,
`results/repaired_original/`. Verified by hash comparison against Step 2.

The controlled benchmark **consumes** `ContextKnapsack`, `SolveStatus` and
`SolveScope` from the repaired `optimizer.py`; no solver logic is duplicated.

---

## 2. Generator architecture

```
InstanceSpec(seed, rl_condition, redundancy_level, GeneratorConfig)
        |
        v
generate_instance()  -> ControlledInstance(relevance, tokens, similarity,
                                           embeddings, query_vector, topics,
                                           diagnostics)
        |                    validate_instance()  <- non-degeneracy guards
        v
compute_saturation() -> W_sat via proven-optimal unconstrained ILP
        |
        v  (condition D only) apply_adversarial(w_sat_pre) -> recompute W_sat
        v
build_budgets()      -> {tight, medium, loose} = round(rho * W_sat)
        |
        v
prepare_instance()   -> instance + saturation + budgets + metadata + diagnostics
```

An instance is fully specified by `(relevance, similarity, tokens)` — the only three
arrays any solver consumes — so **no text is generated and no encoder is involved**.
The controlled latent variables are the ground truth, per design section 5.1.

### Independent RNG streams

`np.random.SeedSequence(seed).spawn(3)` produces three independent generators:
`copula`, `topic`, `noise`. This is what makes the causal-isolation tests pass:
changing `γ` consumes only from the topic/noise streams, so it **cannot perturb the
relevance or token draws**. A single shared stream would have made every factor
interfere with every other one — the v0 failure mode in a new form.

---

## 3. Mathematical implementation

**Copula.** `ζ = z₁`, `η = ρ_c·z₁ + √(1−ρ_c²)·z₂` with `z ~ N(0, I)`. This is the
Cholesky factor of `[[1, ρ_c], [ρ_c, 1]]`; both marginals stay exactly standard normal
for every `ρ_c`, which is what preserves the relevance and token marginals across
conditions A–C.

**Relevance.** `r_i = r_min + (r_max − r_min) · Beta⁻¹(Φ(ζ_i); a, b)`,
`Beta(2, 3)` on `[0.05, 0.60]`.

**Tokens.** `w_i = clip(round(exp(μ_w + σ_w·η_i)), 8, 400)`, `μ_w = log 77`,
`σ_w = 0.50`. Integers, **exact by construction** — no tokenizer involved.

**Embeddings.** `q` is the first basis vector, so the orthogonal complement is simply
coordinates `1…d−1` (any other `q` is a rotation and changes nothing measurable).

```
e_i = r_i · q + √(1 − r_i²) · u_i ,    u_i ⊥ q ,  ‖u_i‖ = 1
```

giving `cos(q, e_i) = r_i` **exactly** — measured max absolute error `< 1e-10`.

**Redundancy.** Inside the subspace orthogonal to `q`:

```
u_i = normalize( √γ · t_{k(i)} + √(1−γ) · ε_i )
```

with `{t_k}` orthonormal (QR) and `ε_i` isotropic unit noise. Then

```
sim(e_i, e_j) = r_i r_j + √(1−r_i²)·√(1−r_j²)·cos(u_i, u_j)
```

with `cos(u_i, u_j) ≈ γ` within a topic and `≈ 0` across. `γ` therefore moves
inter-document similarity **without touching relevance**.

Topic membership is balanced and assigned from the topic stream **independently of
relevance**, so redundancy is not confounded with relevance.

**Similarity is a genuine Gram matrix** because it is `E Eᵀ` for explicit unit vectors —
symmetric, unit diagonal, PSD, entries in `[−1, 1]`, all asserted per instance. No
hand-written matrix is used anywhere; an arbitrary one could be unrealizable by any
encoder.

**Budgets.** `W_sat` = token mass of the unconstrained optimum, from a
**proven-optimal** ILP at `W_max = Σ w_i`. `W_max = round(ρ · W_sat)` for
`ρ ∈ {0.25, 0.60, 1.50}`. No absolute budget is hardcoded anywhere.

---

## 4. Condition encoding and metadata schema

`ControlledInstance.condition_metadata()` carries: `protocol_version`, `seed`,
`rl_condition` (`A`/`B`/`C`/`D`), `rl_condition_name`, `rho_c`, `target_beta`,
`redundancy_level`, `gamma`, `is_adversarial`, `n_docs`, `embed_dim`, `n_topics`,
`objective_lambda`, `mmr_lambda`, and the full `GeneratorConfig`.

`diagnostics` carries, per instance: `beta_target`, `beta_hat`, Pearson `r` + `p`,
Spearman `ρ` + `p`, relevance mean/std/min/max, token mean/std/min/max/CV/total,
similarity mean/std/min/max, `n_docs`, plus (condition D) `adversarial_indices`,
`adversarial_cap`, `adversarial_w_sat_pre`, `adversarial_tokens_after`.

`saturation` carries `w_sat`, `w_sat_proven`, `w_sat_status`, `w_sat_n_selected`,
`w_sat_score`, `corpus_token_mass`, `saturation_indices`, and for condition D also
`w_sat_pre_intervention`.

**The condition is declared, never inferred.** `target_beta` and `rho_c` come from the
spec; `beta_hat` and the realized correlations are recorded *alongside* and are never
used to define or relabel a condition. `test_condition_sign_is_not_inferred_from_realized_correlation`
pins this.

Arm E (realism anchor) is tagged `arm = "E_realism_anchor"` and is **never pooled**
with the synthetic factorial.

---

## 5. Validation tests — 28, all passing

| Group | Tests |
|---|---|
| **Determinism (item 9)** | identical arrays for the same seed+condition across all 8 cells; different seeds give genuinely different instances |
| **Embedding construction (item 4)** | `cos(q, e_i) = r_i` to `< 1e-10`; Gram symmetry / unit diagonal / PSD (`λ_min ≥ −1e-8`) / range; unit-norm embeddings |
| **Causal isolation — redundancy (item 11)** | changing `γ` leaves relevance, tokens, `β̂` and topics **bit-identical** while similarity moves; `γ` raises mean similarity by `> 0.05`; pooled relevance/token arrays identical across `γ` |
| **Causal isolation — relevance/length** | KS test: relevance and token marginals unchanged between A, B and C (`p > 0.01`); `β̂` realizes its target within tolerance; B and C have opposite-signed correlation, A near zero |
| **Causal isolation — budget & seed** | solving at three budgets leaves the documents untouched; changing seed preserves the declared condition |
| **Non-degeneracy (item 10)** | token CV ≥ 0.30 on every instance; **a near-constant length distribution is rejected with `DegenerateInstanceError`**; relevance std, similarity std; tokens are positive integers in range |
| **Condition D** | only the top-`k` relevant documents are inflated, others bit-identical; inflated documents stay individually affordable at the medium budget; A/B/C untouched |
| **Budget construction (item 7)** | `tight < medium < loose`; realized `ρ` within `0.02` of target; `W_sat` proven; loose does not bind, tight does |
| **ILP tractability (item 10)** | `PROVEN_OPTIMAL` in all 8 cells |
| **Divergence (item 14)** | the two greedies diverge in ≥ 3 conditions; **control test**: with constant `w` they are provably identical |
| **Metadata** | the condition round-trips — regenerating from recorded metadata reproduces the instance exactly |

Run: `python tests/test_controlled_generator.py` → **28 passed, 0 failures**.
`tests/test_optimizer.py` still passes 44/44 (unchanged).

---

## 6. Pilot protocol

4 seeds × 4 relevance-length conditions × 2 redundancy × 3 budgets = 96 synthetic
instance-budgets (32 instances), plus arm E at 3 seeds × 3 budgets on the **unchanged**
original text generator. 5 methods each. **Runtime 2.1 min**, inside the design's
~8-minute pilot budget.

`--seeds 4` rather than the design's 5 was chosen to stay inside the time budget with
margin; the pilot is validation, not inference, so the count carries no statistical
weight.

---

## 7. Pilot diagnostics

**Guards.** 32/32 synthetic instances valid, zero failures.

**ILP.** 105/105 budget solves and 35/35 saturation solves `proven_optimal`. No
incumbent was accepted anywhere.

**Elasticity** (the tolerance check is a 40-seed check; the 4-seed pilot values are
noisy by construction and are reported for completeness):

| Condition | target `β` | mean `β̂`, 40 seeds | \|diff\| | tol | pass |
|---|---|---|---|---|---|
| A | 0.00 | −0.034 | 0.034 | 0.10 | yes |
| B | +0.50 | +0.405 | 0.095 | 0.10 | yes (thin) |
| C | −0.50 | −0.439 | 0.061 | 0.10 | yes |

**Redundancy.**

| Level | `γ` | mean pairwise similarity |
|---|---|---|
| low | 0.15 | 0.093 |
| high | 0.70 | 0.178 |
| arm E (real text) | — | 0.330 |

**Token CV** (critical guard, threshold 0.30): synthetic min 0.315, mean 0.584;
0/160 instances below threshold across the 40 seeds the full run would use;
5th percentile 0.373. Arm E min 0.868.

**Budget separation.** `tight < medium < loose` on every instance; realized `ρ` within
0.02 of `{0.25, 0.60, 1.50}`.

**Divergence (item 14).** `greedy_objective` and `greedy_token_aware` selected
**different** documents in **95 / 96** synthetic instance-budgets, in **24 / 24** cells,
and **9 / 9** arm-E instance-budgets. The benchmark exposes the token-cost mechanism.

**Preliminary paired difference.** Recorded in the artifact per cell. It is **not
interpreted here and carries no statistical weight** — 4 seeds, no test, no correction.
Its only role is to show the mechanism can produce a non-zero difference, which it does.

---

## 8. Deviations from the design document

| ID | Issue | Status |
|---|---|---|
| **DEV-1** | Condition D's cap at `0.45·W_sat` is **circular** — `W_sat` is a function of the tokens, so the cap cannot be applied while the tokens are being assigned. | **Resolved as a two-pass procedure**: `W_sat` computed on pre-intervention tokens, cap taken against that, final `W_sat` recomputed after. Both recorded. Preserves the stated intent at the cost of one extra ILP solve for condition D only. **Needs sign-off.** |
| **DEV-2** | The design claims `γ = 0.15 → 0.2` and `γ = 0.70 → 0.45` mean similarity, "calibrated to the v0 corpus (0.405)". **Measured: 0.093 and 0.178.** At `K = 5`, cross-topic pairs contribute ≈ 0 and are 80% of all pairs, so mean similarity ceilings at `≈ r̄² + 0.93·γ/K ≈ 0.26` even at `γ = 1`. The v0 level is **unreachable**. | **NOT resolved unilaterally.** `γ` implemented exactly as specified — it still gives a genuine ≈2× contrast and is non-degenerate. Only the claim that "high" equals v0 redundancy is false. Changing `K`, `γ` or the construction would **redefine an experimental condition** and requires your decision. **OPEN.** |
| **DEV-3** | The design says `β̂` must be "within the tolerance specified by the design" but states no tolerance. | `BETA_TOLERANCE = 0.10` filled in, chosen before measuring, recorded in `protocol.py`. **Needs sign-off.** |

All three are machine-readable in `protocol.DESIGN_DEVIATIONS` and are copied into every
pilot artifact.

---

## 9. Known limitations

1. **DEV-2 is unresolved.** The synthetic "high redundancy" level is roughly half the
   redundancy of the real corpus (0.178 vs 0.330). Until decided, the redundancy factor
   should be read as "low vs moderate", not "low vs realistic".
2. **Condition B's elasticity margin is thin** — `|β̂ − target| = 0.095` against a
   tolerance of `0.10`. It passes, but a different seed range could fail it. Widening
   the tolerance after seeing this would be result-driven and has not been done.
3. **Token-CV headroom is thin.** Minimum observed 0.315 against a 0.30 threshold. No
   instance fails across the 40 seeds checked, but strict generation would abort the
   full run if one ever did. No resample policy is defined; adding one is a design
   decision.
4. **Synthetic relevance is strictly positive** (floor 0.05) while the real encoder can
   return negative cosine — arm E seed 2 has `r_min = −0.025`. The synthetic generator
   cannot reproduce that regime. Arm-E elasticity is therefore computed on the positive
   subset, with `beta_hat_n_used` and `beta_hat_n_dropped_nonpositive` recorded.
5. **Condition D is structural, not a fourth point on the copula scale**, and must not
   be pooled with A–C in any analysis that assumes shared marginals.
6. **The similarity floor** `sim ≥ r_i·r_j > 0` means no pair is ever orthogonal or
   negatively correlated. This matches the dense positive structure of the v0 corpus but
   is not universal across real retrieval sets.
7. **Arm E depends on the sentence-transformers model** and will silently differ if the
   model version changes; the backend name is recorded per instance.
8. **The pilot proves capability, not effect.** Nothing here says whether
   token-awareness helps, and no such statement should be extracted from
   `paired_diff_tokenaware_minus_objective`.

---

# Step 3C — Pre-flight validation across all 40 planned seeds

**Verdict: PASS on all 11 automated checks.** No stop condition triggered.
Runtime 9.5 min. Artifacts: `results/controlled/preflight_report.{json,csv}`.

**One conflict requires your decision before Step 4** — see "Unresolved concerns",
item U1. It is a conflict between two readings of the β criterion, not a generator
defect.

This section separates, deliberately: **protocol requirements** (what the locked spec
demands), **observed results** (what 40 seeds produced), **deviations**, and
**unresolved concerns**.

## 3C.0 Locked decisions

| Item | Decision |
|---|---|
| **DEV-2 (redundancy)** | **LOCKED.** `K = 5`, `γ ∈ {0.15, 0.70}` and the construction are **unchanged**. The factor is a **LOW vs MODERATE** redundancy contrast. The claim that `γ` was calibrated to the v0 corpus is **retracted** — struck in `docs/controlled_benchmark_design.md` §Factor 3 with the retraction recorded in place, and restated in `protocol.py`. No analysis may describe this factor as matching real-corpus redundancy. |
| **`BETA_TOLERANCE`** | **LOCKED at 0.10.** Not widened. Target β values unchanged. |
| **Seeds** | 0–39, contiguous. No seed resampled, excluded or reordered. |
| **`objective_lambda` / `mmr_lambda`** | 0.1 / 0.5, unchanged. |

## 3C.1 Protocol requirements vs observed results

### Generation
*Requirement:* every planned instance generates under the locked guards.
*Observed:* **320/320 generated**, zero rejections, zero validation failures.

### β realization (§3)
*Requirement (as implemented in `protocol.py`):* `|mean(β̂) − target| ≤ 0.10` per condition.

| cond | target | mean β̂ | sd | min | max | worst \|dev\| | \|mean−target\| | per-instance pass |
|---|---|---|---|---|---|---|---|---|
| A | +0.00 | −0.0338 | 0.1898 | −0.4038 | +0.3655 | 0.4038 | **0.0338** | 16/40 |
| B | +0.50 | +0.4051 | 0.1732 | −0.0669 | +0.7548 | 0.5669 | **0.0949** | 24/40 |
| C | −0.50 | −0.4388 | 0.1505 | −0.7210 | −0.1126 | 0.3874 | **0.0612** | 16/40 |
| D | n/a | (pre-intervention ≡ A) | | | | | | |

*Observed:* the **mean** criterion passes for A, B and C. The **per-instance** criterion
does not — see U1.

Condition D post-intervention β̂ = **+0.238** (sd 0.072). Pre-intervention it is
identical to A by construction, since D shares A's copula.

### Non-degeneracy (§4)

| metric | mean | sd | min | max | q05 | guard | below guard |
|---|---|---|---|---|---|---|---|
| token CV | 0.5114 | 0.0947 | **0.3149** | 0.8260 | 0.3727 | 0.30 | **0/320** |
| relevance sd | 0.1064 | 0.0144 | 0.0731 | 0.1452 | — | 0.05 | 0/320 |
| similarity sd | 0.1922 | 0.0623 | 0.1161 | 0.2662 | — | 0.01 | 0/320 |
| unique token lengths | 23.06 | 1.12 | **20** | 25 | — | >1 | 0/320 |
| unique relevance values | 25.00 | 0.00 | 25 | 25 | — | >1 | 0/320 |
| topics used | 5 | 0 | 5 | 5 | — | — | — |

No condition collapses toward constant length or constant relevance. Minimum token-CV
headroom is **+0.0149** — thin, and carried forward as concern U2.

### Causal isolation (§5)

*Requirement:* `γ` moves similarity and nothing else; `ρ_c` moves dependence only.

| intervention | quantity | observed |
|---|---|---|
| `γ` low→moderate | max abs change in **relevance** | **0.0** (exact) |
| | max abs change in **tokens** | **0.0** (exact) |
| | max abs change in **β̂** | **0.0** (exact) |
| | max abs change in **topic assignment** | **0.0** (exact) |
| | change in mean similarity | +0.0848 (sd 0.0044, min +0.0730, max +0.0943) |
| `ρ_c` A→B / A→C | relevance marginal, KS `p` | 1.000 / 1.000 |
| | token marginal, KS `p` | 0.723 / 0.969 |
| | topic structure identical across all 4 conditions | **True** |
| | D pre-intervention ≡ A | **True** |

The four exact zeros are the payoff from spawning independent `copula` / `topic` /
`noise` RNG streams: `γ` cannot consume from the streams that produce relevance or
tokens, so isolation is exact rather than approximate.

### Redundancy manipulation (§6) — LOW vs MODERATE

| level | `γ` | mean | sd | min | max |
|---|---|---|---|---|---|
| low | 0.15 | 0.0950 | 0.0160 | 0.0638 | 0.1257 |
| **moderate** (key `"high"`) | 0.70 | 0.1798 | 0.0148 | 0.1501 | 0.2098 |
| paired difference | | +0.0848 | 0.0044 | +0.0730 | +0.0943 |

Positive in **160/160** pairs. Cohen's `d = 5.50`, ratio **1.89×**, **ranges do not
overlap** (max low 0.1257 < min moderate 0.1501). A clean, non-degenerate separation.

**Arm E (real corpus) measures 0.3509.** The moderate level is roughly half that. This
is a LOW vs MODERATE contrast and must never be described as reaching real-corpus
redundancy.

### Budget construction (§7)

| quantity | mean | sd | min | max |
|---|---|---|---|---|
| `W_sat` | 1788.6 | 754.6 | 660 | 4528 |
| saturation fraction `W_sat / Σw` | 0.680 | 0.143 | 0.324 | 0.938 |

| level | target `ρ` | realized mean `ρ` | max abs deviation |
|---|---|---|---|
| tight | 0.25 | 0.24999 | 0.00059 |
| medium | 0.60 | 0.60000 | 0.00055 |
| loose | 1.50 | 1.50002 | 0.00074 |

`tight < medium < loose` in **320/320**. Saturation ILP proven optimal in **320/320**.
**0/320** instances where the tight budget cannot afford even the cheapest document, so
no budget level produces a degenerate empty-selection problem.

### Method divergence (§8) — generator validation, not a performance result

**958/960 (99.8%)** instance-budgets produced *different* selections between
`greedy_objective` and `greedy_token_aware`. **Zero cells** had no divergence.

| by budget | rate | by condition | rate |
|---|---|---|---|
| tight | 318/320 (99%) | A | 240/240 (100%) |
| medium | 320/320 (100%) | B | 240/240 (100%) |
| loose | 320/320 (100%) | C | 238/240 (99%) |
| | | D | 240/240 (100%) |

**Constant-length control: identical in every case**, as theory requires — with `w`
constant, `Δ_i/w_i ∝ Δ_i`. The control confirms the divergence above is caused by the
token-cost mechanism and not by anything else.

### ILP tractability (§9)

Representative seeds `[0, 13, 27]` × 8 cells × 3 budgets = **72 solves**:
**72/72 `proven_optimal`**, 72/72 feasible, 72/72 objective-consistent (the solver's own
objective agrees with the recomputed `Score(S)`).

| | mean | sd | min | max | q95 |
|---|---|---|---|---|---|
| representative solve (s) | 0.265 | 0.318 | 0.018 | 1.610 | 0.90 |
| saturation solve, all 320 (s) | 0.258 | 0.360 | 0.018 | **2.577** | 0.94 |

Model size is stable: 222–272 pair variables. No pathological instance appeared. These
are far below the original benchmark's worst case (7.3 s), so the full experiment is
comfortably tractable.

### Condition D audit (§10)

Procedure, as implemented:

1. Generate the instance with **unmodified** tokens.
2. Compute `w_sat_pre` from a proven-optimal unconstrained ILP **on that pre-intervention state**.
3. Inflate the top-`k = 4` relevant documents by `m = 6`, **capped at `0.45 · w_sat_pre`**.
4. Recompute the **final** `W_sat` on the modified instance (proven-optimal ILP).
5. Derive tight/medium/loose from the **final** `W_sat` only.

**Non-circularity:** the cap reads `w_sat_pre`, a function of the pre-intervention
tokens alone. The final `W_sat` is never fed back into the cap, so no fixed point is
required and no iteration can diverge. The dependence is a strict one-way chain
`tokens → w_sat_pre → cap → tokens' → W_sat → budgets`.

*Observed (80 instances):* `w_sat_pre` 1430.1 ± 336.0, final `W_sat` 2885.0 ± 545.6,
shift +1454.9 ± 270.6, cap 643.5 ± 151.2. Pre and final ILP **proven in 80/80**.
Adversarial documents individually affordable at the medium budget in **80/80**.

*Does D do what it is for?* The adversarial documents have mean density `r/w` = 0.00108
against 0.00351 for the rest — **3.25× less dense**, lower in **40/40** seeds, with mean
density rank 3.49 out of 25. Condition D genuinely creates the regime where the most
relevant documents are the worst value per token.

### Arm E (§11) — validated separately, never pooled

40 seeds on the **unchanged** original text generator.

| quantity | mean | sd | min | max |
|---|---|---|---|---|
| β̂ (positive subset) | +0.0759 | 0.0868 | −0.0736 | +0.3057 |
| Pearson `r`(relevance, tokens) | +0.0476 | 0.1374 | −0.3189 | +0.3550 |
| mean pairwise similarity | 0.3509 | 0.0374 | 0.2870 | 0.4304 |
| token CV | 1.0236 | 0.1120 | 0.8199 | 1.3469 |
| non-positive relevance values per seed | 0.20 | 0.41 | 0 | 1 |

**β̂ for arm E is computed on the positive-relevance subset only.** The real encoder can
return a negative cosine, for which the log-log slope is undefined; **8/40 seeds** have
at least one such document (at most 1 per seed). The dropped count is recorded per seed
as `beta_hat_n_used` / `n_relevance_nonpositive`. Handling is **deterministic** in
40/40, `W_sat` proven in 40/40, budgets ordered in 40/40.

Arm E is stored under `arm = "E_realism_anchor"` and is excluded from every synthetic
aggregate in this report.

### Determinism (§12)

Regenerating the same `(seed, condition, redundancy)` reproduces relevance, tokens,
similarity and topics **exactly** across all representative seeds × 8 cells. Different
seeds produce genuinely different instances in every condition.

## 3C.2 Deviations from the design document

| ID | Status after 3C |
|---|---|
| **DEV-1** — condition D cap circular in the design text | Resolved by the documented two-pass construction; audited above, 80/80 clean. **Implemented; sign-off requested.** |
| **DEV-2** — redundancy calibration claim false | **LOCKED.** Parameters unchanged; the claim is retracted and the factor is renamed in wording to LOW vs MODERATE. |
| **DEV-3** — β tolerance unspecified in the design | **LOCKED at 0.10.** |
| **DEV-4 (new)** — β criterion is ambiguous between per-condition mean and per-instance | See U1. **Unresolved.** |

## 3C.3 Unresolved concerns

**U1 — RESOLVED and LOCKED at step 3C.1.** *(was: the β criterion has two readings
and they disagree)*

The ambiguity was between the Step 3C instruction text, which asked to verify
`|β̂ − β_target| ≤ 0.10` "for each condition and seed", and `protocol.py`, which
implemented the criterion on the per-condition mean.

**Decision: β is a population-level parameter of the generator.** The locked criterion is

```
| mean(β̂ across seeds 0–39) − β_target | ≤ 0.10      per β condition
```

and the per-seed form is **explicitly not required**.

*Rationale.* `β̂` is estimated from only `N = 25` documents per realisation. The 40-seed
pre-flight measured per-instance `sd(β̂) = 0.15–0.19`, i.e. **1.5–1.9× the ±0.10
tolerance**, so a per-seed acceptance rule would partially select realisations according
to **estimator noise** rather than validate the underlying generator parameter. The
generator uses the exact target parameter — `ρ_c` is realised exactly by the copula —
and `β̂` is a finite-sample measurement of it.

*Consequences.* Per-seed `β̂` values remain recorded and reported for every instance,
because they quantify finite-sample estimator variability and act as a covariate in the
continuous companion model. **Reporting them is required; filtering on them is
forbidden.** The `n_pass` / `per_instance_pass_rate` fields already present in
`preflight_report.json` are descriptive variability statistics, **not** acceptance
gates, and must not be read as such.

*Status under the locked criterion:*

| cond | β_target | mean β̂ | \|mean − target\| | tolerance | verdict |
|---|---|---|---|---|---|
| A | +0.00 | −0.0338 | 0.0338 | 0.10 | **PASS** |
| B | +0.50 | +0.4051 | 0.0949 | 0.10 | **PASS** |
| C | −0.50 | −0.4388 | 0.0612 | 0.10 | **PASS** |

Standard error of the mean over 40 seeds is 0.024–0.030, so the aggregate criterion
carries 3–4× margin. **The Step 3C PASS verdict remains valid** — unchanged, because
`preflight_report.json` already evaluated and recorded this criterion
(`checks.beta.criterion`, `mean_within_tolerance`), so nothing was recomputed.

*What was NOT done:* no tolerance widened, no seed resampled or excluded, no β target
changed, no generator parameter changed, no `N` change, no optimizer/benchmark code
change. This was a protocol-definition change only. Recorded machine-readably as
`DEV-4` and `PROTOCOL_LOCK` in `experiments/controlled/protocol.py`.

**U2 — token-CV headroom is thin.** Minimum observed 0.3149 against the 0.30 guard,
headroom +0.0149; q05 = 0.3727. Nothing fails across the 40 planned seeds, so the full
run will not abort. But no resample policy exists, so a future seed range could halt
generation mid-run.

**U3 — redundancy does not reach real-corpus levels.** Moderate = 0.1798 vs arm E's
0.3509. Locked by decision; the interaction with redundancy is therefore estimated over
a narrower range than real retrieval exhibits, and conclusions about the redundancy
factor must be stated within that range.

**U4 — condition D roughly doubles `W_sat`** (1430 → 2885). This is expected — inflating
the cost of documents the optimum wants raises the optimum's token mass — and budgets
derive from the final `W_sat`, so tightness stays comparable. But D's *absolute* budgets
are about twice those of A–C, which is a further reason never to pool D with A–C.

**U5 — synthetic relevance is strictly positive** (floor 0.05) while arm E produces
non-positive cosines in 8/40 seeds. The synthetic generator cannot reproduce that regime.

## 3C.4 Protocol lock (step 3C.1)

| Item | State |
|---|---|
| Seeds | **0–39 fixed.** None resampled, excluded or reordered. |
| β criterion | `\|mean(β̂) − β_target\| ≤ 0.10` per condition — **locked**. Per-seed form explicitly not required. |
| β tolerance | 0.10 — **never widened**. |
| β targets | A `0.0`, B `+0.5`, C `−0.5`, D none — **unchanged**. |
| `N` | 25 — **unchanged**. |
| `γ` | `{low: 0.15, high: 0.70}`, `K = 5` — **unchanged**; LOW vs MODERATE contrast. |
| Budgets | `ρ ∈ {0.25, 0.60, 1.50}` of `W_sat` — **unchanged**. |
| `objective_lambda` / `mmr_lambda` | 0.1 / 0.5 — **unchanged**. |
| Generator code | **unchanged** at 3C.1. |
| Optimizer / benchmark code | **unchanged** since Step 1. |
| Step 3C verdict | **PASS, still valid** under the clarified criterion. |
| Open deviations | DEV-1 (two-pass `W_sat`) awaits sign-off; DEV-2, DEV-3, DEV-4 locked. |

The protocol is frozen. Any further change requires a new step and a new document.
