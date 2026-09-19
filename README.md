# Evidence Under Budget

**Does token-aware context selection recover better evidence? We pre-registered the question, ran it on HotpotQA, and the answer came back "no".**

This repository is a study, not a product pitch. It formulates LLM context selection as a
Quadratic Knapsack Problem, implements four heuristics plus an exact ILP, and then tests —
under a locked, pre-registered protocol — whether normalizing marginal gain by token cost
improves the evidence actually recovered. It does not, in the setting tested.

---

## Headline result

On 500 HotpotQA questions (distractor setting), comparing token-aware greedy (TA) against
plain objective greedy (GO) at three budgets, with supporting-fact coverage as the outcome:

| Budget $\rho$ | mean $\Delta_C$ | 95% CI | Holm-adjusted $p$ | ties |
|---|---|---|---|---|
| 0.10 | **−0.156** | [−0.182, −0.130] | 0.0003 | 264 / 500 |
| 0.25 | **−0.060** | [−0.080, −0.039] | 0.0003 | 352 / 500 |
| 0.50 | **−0.033** | [−0.048, −0.018] | 0.0003 | 414 / 500 |

Negative means token-aware selection recovers **fewer** annotated supporting facts than the
unnormalized baseline, at the same token budget. All three nulls are rejected. The effect is
concentrated in a minority of questions — most of the time the two methods tie.

At the two tight budgets TA simultaneously achieves the **higher** objective score. It
optimizes the stated objective better while recovering less annotated evidence — in this
setting the objective and the evidence are not aligned.

**No SESOI was specified**, so this repository makes no claim about whether these differences
are practically important. See [Scope](#what-this-does-and-does-not-support).

---

## Problem formulation

Let $D = \{d_1, \dots, d_N\}$ be the candidates returned for query $q$, with

- relevance $r_i = \cos(e_q, e_i) \in [-1, 1]$
- pairwise similarity $s_{ij} = \cos(e_i, e_j)$, clipped to $[0, 1]$ inside the solver
- token cost $w_i = |\mathrm{tokenize}(d_i)|$ under `cl100k_base`

Select $S \subseteq D$ maximizing

$$
\max_{S \subseteq D} \; \mathrm{Score}(S) = \sum_{i \in S} r_i \; - \; \lambda_{\mathrm{obj}} \sum_{i < j \,\in\, S} s_{ij}
\qquad \text{s.t.} \qquad \sum_{i \in S} w_i \le W_{\max}
$$

This is the Quadratic Knapsack Problem: a linear knapsack with a negative pairwise
interaction term, NP-hard. Setting $\lambda_{\mathrm{obj}} = 0$ recovers the linear knapsack;
equal $w_i$ with $\lambda_{\mathrm{obj}} > 0$ gives a summed-penalty relative of MMR.

The default is $\lambda_{\mathrm{obj}} = 0.1$. Relevance grows as $O(|S|)$ while the penalty
grows as $O(|S|^2)$, so a $\lambda$ borrowed from MMR — where the penalty is a **max**, not a
sum — is a scale error here rather than a preference: past a corpus-dependent value of
$\lambda$ every additional document is net-negative and the solvers stop early whatever the
budget allows. `benchmark.py --objective-lambda` exposes the sweep.

### Solvers

All heuristics share one **stopping policy**: candidate $i$ is eligible iff it is unselected,
$w_i \le$ remaining budget, and its marginal gain is positive,

$$
\Delta_i = r_i - \lambda_{\mathrm{obj}} \sum_{j \in S} s_{ij} > 0 .
$$

They differ only in how they rank the eligible candidates:

| Solver | Selection rule |
|---|---|
| `solve_top_k` | $\arg\max_i \; r_i$ |
| `solve_mmr` | $\arg\max_i \; \lambda_{\mathrm{mmr}} r_i - (1-\lambda_{\mathrm{mmr}}) \max_{j \in S} s_{ij}$ |
| `solve_greedy_objective` | $\arg\max_i \; \Delta_i$ |
| `solve_greedy` (token-aware) | $\arg\max_i \; \Delta_i / w_i$ |

`solve_greedy_objective` and `solve_greedy` differ **only** by the $1/w_i$ factor, which is
what makes their difference a clean measurement of token normalization.

### Exact solver

Binary $x_i$ selects document $i$; $y_{ij}$ linearizes the product $x_i x_j$:

$$
\max \sum_i r_i x_i - \lambda_{\mathrm{obj}} \sum_{i<j} s_{ij} y_{ij}
\quad \text{s.t.} \quad
\sum_i w_i x_i \le W_{\max}, \quad
y_{ij} \ge x_i + x_j - 1, \quad
x_i \in \{0,1\}, \; y_{ij} \ge 0
$$

Every $y_{ij}$ carries a non-positive objective coefficient, so the solver drives it to its
lower bound: $y$ can stay **continuous**, the usual $y_{ij} \le x_i$ constraints are
redundant, and pairs with $s_{ij} = 0$ are dropped from the model. The returned status
distinguishes a *proven* optimum from a time-limited incumbent — incumbents are returned
intact and never relabelled as optimal.

---

## The v7 experiment

The final phase is a pre-registered, locked protocol:
[`docs/final_validation_protocol.md`](docs/final_validation_protocol.md),
SHA-256 `ff807b7d…1d0daa5`, verified at runtime by the code before every run.

| Element | Choice |
|---|---|
| Data | HotpotQA dev, distractor setting, 500 questions, frozen permutation (seed 7000) |
| Candidate unit | sentence (all sentences of the 10 paragraphs) |
| Relevance | cosine against `all-MiniLM-L6-v2`, local, no paid API |
| Budgets | $W_{\max} = \lfloor \rho \cdot W_{\mathrm{pool}} \rfloor$, $\rho \in \{0.10, 0.25, 0.50\}$ |
| Outcome | supporting-fact coverage $C(S) = \|S \cap G(q)\| / \|G(q)\|$ |
| Estimand | $\Delta_C = C(\mathrm{TA}) - C(\mathrm{GO})$ |
| Confirmatory test | two-sided paired sign-flip permutation, 10,000 flips, Holm across exactly 3 budgets |
| SESOI | none specified, deliberately |

$W_{\mathrm{pool}}$ is the total token count of the candidate pool, computed before selection
and identical across all arms: $\rho$ measures relative budget tightness, not an absolute
token budget.

### Coverage by arm

| Arm | $\rho$ = 0.10 | 0.25 | 0.50 |
|---|---|---|---|
| top-k | 0.523 | 0.716 | **0.832** |
| greedy objective (GO) | **0.531** | **0.744** | 0.816 |
| greedy token-aware (TA) | 0.375 | 0.685 | 0.784 |
| MMR | 0.474 | 0.642 | 0.779 |
| token-aware + best-singleton | 0.375 | 0.685 | 0.784 |

### Objective score by arm

| Arm | $\rho$ = 0.10 | 0.25 | 0.50 |
|---|---|---|---|
| top-k | 2.081 | 3.154 | 3.471 |
| greedy objective (GO) | 2.103 | 3.294 | **3.610** |
| greedy token-aware (TA) | **2.408** | **3.370** | 3.545 |
| MMR | 1.885 | 3.000 | 3.446 |

TA leads on the objective at $\rho = 0.10$ and $0.25$ and trails GO at $\rho = 0.50$. Only
TA vs GO on coverage was tested; every other comparison in these two tables is descriptive.

### Secondary findings

- **The best-singleton guard never fired.** In all 1500 instances it returned exactly the
  token-aware selection. The pathological case from the budgeted-submodular literature (cheap
  high-density items crowding out one valuable expensive item) did not occur here.
- **The pre-hoc diagnostic failed.** $D = 1 - \tau_b(r,\, r/w)$ shows no association with
  $\Delta_C$: Spearman $+0.027$, $-0.069$, $-0.039$, all bootstrap CIs containing zero. It was
  pre-registered as exploratory; no threshold was searched for.
- **ILP**: 58 of 100 in-scope instances proved optimal within the 60 s cap (mean 12.0 s).
  The 42 time-limited incumbents are retained and labelled, and excluded from the optimality
  summary. Because the proven subset is selected by pool size, its numbers are not compared
  against the 500-question arms.

---

## What this does and does not support

**Supported.** Within this QKP formulation, on HotpotQA distractor, with sentence-level
candidates, MiniLM relevance, and $\rho \in \{0.10, 0.25, 0.50\}$: token normalization lowers
supporting-fact coverage relative to unnormalized greedy, most strongly at the tightest
budget, while raising the objective score at the two tight budgets.

**Not supported.** That token normalization is generally harmful — outside this objective,
corpus, granularity, relevance model, or budget range, nothing here applies. That any of
these differences matter practically: no SESOI exists, no external threshold for
supporting-fact coverage is established, and $p < 0.05$ does not answer that question. That
better packing produces better answers: no generative evaluator was used. Any causal claim
about corpus properties and $\Delta_C$. Any inferential claim involving top-k, MMR, the
guard, or the ILP, all of which are descriptive only.

---

## Quickstart

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Synthetic benchmark

```bash
python benchmark.py                  # default sweep, writes JSON + PNG
python benchmark.py --no-ilp --no-plot   # fast smoke run
```

| Flag | Default | Meaning |
|---|---|---|
| `--budgets` | `256 512 1024 2048` | $W_{\max}$ sweep |
| `--n-docs` | 25 | corpus size |
| `--objective-lambda` (alias `--lambda`) | 0.1 | redundancy weight $\lambda_{\mathrm{obj}}$ |
| `--mmr-lambda` | 0.5 | MMR trade-off weight, affects the MMR baseline only |
| `--repeats` | 3 | runs per budget; latency is the median |
| `--ilp-max-docs` | 0 | restrict the exact solver to the top-M by relevance (0 = no restriction, and a restricted result is a *pool* optimum, not a global one) |
| `--ilp-time-limit` | 300 | CBC time limit, seconds |
| `--no-gain-stop` | off | restores v0's fill-the-budget behaviour, for comparison only |
| `--seed` | 42 | corpus + run seed |

When the ILP cannot prove optimality the comparison column switches from `vs opt` to
`vs best`: a timed-out incumbent is never reported as ground truth.

### Library use

```python
from embedder import Embedder, build_corpus
from optimizer import ContextKnapsack

corpus = build_corpus(query, documents, Embedder())
knapsack = ContextKnapsack(corpus.relevance, corpus.similarity, corpus.tokens,
                           budget=2048, objective_lambda=0.1)

result = knapsack.solve_greedy()          # token-aware
baseline = knapsack.solve_greedy_objective()
exact = knapsack.solve_ilp(time_limit=60)

print(result.tokens_used, result.status, exact.is_proven_global_optimum)
```

### Reproducing v7

Requires `pyarrow` (parquet) in addition to `requirements.txt`. The HotpotQA parquet is
downloaded once and its SHA-256 recorded.

```bash
python -m experiments.final_validation.run_experiment --dry-run   # 20 questions, sanity only
python -m experiments.final_validation.run_experiment --full      # 500 questions, ~55 min
python -m experiments.final_validation.analyze \
    --raw results/final_validation/experiment_raw.json \
    --out results/final_validation/experiment_analysis.json
```

Both scripts verify the protocol hash at startup and refuse to run if the locked document
changed. Existing result artifacts are never overwritten — the runner aborts instead.

### Tests

```bash
pytest tests/                    # 166 tests
python tests/test_optimizer.py   # core suite standalone, no pytest needed
```

| Suite | Tests | Covers |
|---|---|---|
| `test_optimizer.py` | 44 | solver invariants, boundary conditions, budget and status guarantees |
| `test_controlled_generator.py` | 28 | controlled-benchmark instance generation |
| `test_elasticity.py` | 30 | elasticity protocol and analysis machinery |
| `test_seed_count.py` | 25 | frozen seed-count rule and its edge cases |
| `test_final_validation.py` | 39 | v7 protocol hash, coverage, pool identity, budgets, Holm family, ILP scope |

---

## Repository map

| Path | Role |
|---|---|
| `optimizer.py` | `ContextKnapsack`: top-k, MMR, greedy objective, token-aware greedy, exact ILP |
| `embedder.py` | encoding, token counting, cached $N \times N$ cosine matrix |
| `benchmark.py` | synthetic corpus generator, metrics, sweep driver, plots |
| `experiments/controlled/` | controlled synthetic benchmark (design → pilot → full run → audit) |
| `experiments/elasticity/` | mechanistic elasticity study, locked at v6.2 |
| `experiments/final_validation/` | **v7**: locked protocol, data loader, runner, analysis |
| `results/` | write-once artifacts, one directory per study |
| `docs/` | protocols, design memos, audits, post-mortems |
| `tests/` | 166 tests |

`embedder.py` ships a seeded hashing fallback used when sentence-transformers is unavailable.
It is semantically weaker and is **not** what the reported numbers use — the v7 runner
disables the fallback and aborts if the real model cannot load.

---

## How the study got here

| Stage | Outcome |
|---|---|
| **v0** (tag `v0-original-experiment`) | Original benchmark, preserved unmodified. Its headline claims did not survive audit. |
| **Repair** | ILP status handling, a real MMR baseline, a common stopping policy, corrected density and pooled-optimum terminology. v0's "MMR" was in fact greedy ascent on the objective; v0's top-k had no gain-based stop, which manufactured its collapse at large budgets. See `docs/experiment_repair.md`. |
| **Controlled benchmark** | 5400 rows over a factorial synthetic design, plus a post-run statistical audit. |
| **Elasticity (v6.2)** | Pre-registered mechanistic study. The seed-count rule hit a hard stop: the observed second moment was too small relative to the effect threshold, so no valid sample size existed. **Locked, stopped, and reported rather than rescued.** |
| **SESOI review** | The inherited 3% threshold had no scientific justification, and no external threshold exists in the objective's units. That closed the path to a confirmatory run on the synthetic estimand. |
| **v7** | Redesign around a downstream, externally meaningful outcome: supporting-fact coverage on real data. Locked, run, analyzed, audited. Result above. |

Artifacts are write-once and version-named; protocols are hashed and committed before the
run they govern. A stopped experiment is kept as a result, not deleted.

---

## Limitations

- **One dataset, one encoder, one granularity.** HotpotQA distractor, MiniLM, sentence-level
  candidates. No generalization claim beyond that.
- **Coverage is not answer quality.** No generative evaluator was used, by design (no paid
  APIs anywhere in this repository).
- **No SESOI.** Effect distributions are reported in full precisely because no practical
  threshold can be justified.
- **CBC incumbents are not bit-reproducible.** Across two runs of the same 20 instances, one
  time-limited ILP instance returned a different incumbent (same status, same coverage).
  Proven optima and all heuristic arms were identical.
- **The ILP proven subset is selected.** 58 of 100 instances closed within 60 s, and they
  tend to be the smaller pools.
- **Ties dominate.** Between 53% and 83% of questions yield $\Delta_C = 0$; mean and median
  tell different parts of the story and should be read together.
- **Git history is partial.** Most research phases were developed while commits were blocked
  in this environment; the locked v7 protocol and its lock record are committed, earlier
  phases are documented in `docs/` instead.

---

## License

No license file is present. Until one is added, no usage rights are granted — if you intend
this to be open source, add a `LICENSE` file.
