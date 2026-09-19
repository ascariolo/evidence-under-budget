r"""Comparative evaluation of context-packing strategies.

Runs Naive Top-K, true MMR, greedy ascent on the objective, the Token-Aware
Knapsack (greedy) and the exact ILP over a seeded synthetic corpus with
controlled redundancy.

Primary metrics (used for comparison and for any claim):

* **Score(S)** - the research objective itself.
* **Coverage** - fraction of distinct information clusters retained.
* **Redundancy** - ``sum_{i<j} sim(d_i, d_j)``, reported per-pair as well as
  totalled, because the total conflates redundancy with selection size.
* **Tokens used / documents selected**.

Diagnostic only (never used to rank methods):

* **Context Density Score** - deprecated, see docs/experiment_repair.md.
* **Token Savings (%)** - budget utilization, not quality.
* **Latency (ms)** - not comparable across solver classes; see ``--help``.

Usage::

    python benchmark.py                      # default sweep + plots
    python benchmark.py --no-ilp --no-plot   # fast smoke run
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from embedder import Embedder, EncodedCorpus, build_corpus
from optimizer import (ContextKnapsack, SelectionResult, SolveScope, SolveStatus,
                       DEFAULT_MMR_LAMBDA, DEFAULT_OBJECTIVE_LAMBDA)

RANDOM_SEED = 42

#: Bumped whenever the benchmark procedure changes in a way that makes older
#: result files non-comparable. v0 shipped no version field at all.
BENCHMARK_CONFIG_VERSION = "1.0-repaired"

#: Canonical method order for printing and plotting.
METHOD_ORDER = ("top_k", "mmr", "greedy_objective", "greedy_token_aware", "ilp")
METHOD_LABELS = {
    "top_k": "Naive Top-K",
    "mmr": "MMR (max-sim)",
    "greedy_objective": "Greedy on objective",
    "greedy_token_aware": "Token-Aware Knapsack",
    "ilp": "ILP (exact)",
}
METHOD_STYLES = {"top_k": "-", "mmr": "--", "greedy_objective": "-.",
                 "greedy_token_aware": "-", "ilp": ":"}
METHOD_COLORS = {
    "top_k": "#9aa0a6",
    "mmr": "#4c8dd9",
    "greedy_objective": "#8e6fbf",
    "greedy_token_aware": "#e0742a",
    "ilp": "#3f9a54",
}


# --------------------------------------------------------------------------- #
# Synthetic corpus generation
# --------------------------------------------------------------------------- #
# UNCHANGED FROM v0. The generator is deliberately frozen at this step: the
# repair is confined to solvers, metrics and reporting so that v0 and the
# repaired run remain comparable on identical data.
TOPIC_BANK: Dict[str, List[str]] = {
    "transformer attention": [
        "Self-attention lets every token attend to all other tokens in the sequence.",
        "Scaled dot-product attention divides the logits by the square root of the head dimension.",
        "Multi-head attention runs several attention operations in parallel subspaces.",
        "Attention weights are produced by a softmax over query-key inner products.",
    ],
    "vector retrieval": [
        "Dense retrieval encodes queries and passages into a shared embedding space.",
        "Approximate nearest neighbour indexes such as HNSW trade recall for latency.",
        "Cosine similarity ranks passages by the angle between their embeddings.",
        "Hybrid retrieval blends sparse BM25 scores with dense vector scores.",
    ],
    "context windows": [
        "Long context windows raise inference cost roughly quadratically with sequence length.",
        "Models degrade on facts placed in the middle of a long prompt.",
        "Context compression drops low-information spans before the prompt is sent.",
        "Token budgets force a trade-off between coverage and prompt latency.",
    ],
    "knapsack optimization": [
        "The knapsack problem maximizes value subject to a hard weight constraint.",
        "Greedy density heuristics sort items by value per unit of weight.",
        "The quadratic knapsack problem adds pairwise interaction terms to the objective.",
        "Integer linear programming yields exact solutions but scales exponentially.",
    ],
    "unrelated chatter": [
        "Sourdough starters need regular feeding at room temperature.",
        "The regional train timetable changes every December.",
        "Rain is expected across the northern valleys this weekend.",
        "Municipal recycling rules differ between neighbouring towns.",
    ],
}

PADDING = (
    "This passage restates the same background material at length, adding "
    "qualifications and transitional phrasing that carry little additional "
    "information for a reader who already understands the core claim. "
)


def generate_corpus(n_docs: int = 25, seed: int = RANDOM_SEED,
                    query_topics: Sequence[str] = ("transformer attention",
                                                   "vector retrieval",
                                                   "context windows"),
                    ) -> Tuple[str, List[str]]:
    r"""Build a seeded corpus with redundancy, length skew and distractors.

    Roughly half the documents paraphrase a small set of on-topic facts (so a
    relevance-only ranker will happily buy the same fact repeatedly), a third are
    padded to 5-10x their information content (token inefficiency) and the rest
    are off-topic distractors. Deterministic for a fixed ``seed``.

    UNCHANGED FROM v0 - see the note above ``TOPIC_BANK``.
    """
    rng = random.Random(seed)
    query = ("How does attention-based retrieval fit inside a limited LLM "
             "context window?")

    on_topic: List[str] = []
    for topic in query_topics:
        on_topic.extend(TOPIC_BANK[topic])
    off_topic = [s for t, sents in TOPIC_BANK.items()
                 if t not in query_topics for s in sents]

    documents: List[str] = []
    for i in range(n_docs):
        roll = rng.random()
        if roll < 0.45:                       # concise, on-topic, redundant
            base = rng.choice(on_topic)
            documents.append(base)
        elif roll < 0.75:                     # on-topic but token-inefficient
            base = rng.choice(on_topic)
            documents.append(base + " " + PADDING * rng.randint(2, 6))
        elif roll < 0.90:                     # multi-fact, genuinely dense
            picks = rng.sample(on_topic, k=min(3, len(on_topic)))
            documents.append(" ".join(picks))
        else:                                 # distractor
            documents.append(rng.choice(off_topic) + " " + PADDING * rng.randint(0, 2))
    rng.shuffle(documents)
    return query, documents


# --------------------------------------------------------------------------- #
# Reproducibility metadata
# --------------------------------------------------------------------------- #
UNAVAILABLE = "unavailable"


def _package_version(name: str) -> str:
    """Installed version of ``name``, or an explicit marker. Never invented."""
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return UNAVAILABLE


def _git_info() -> Dict[str, Any]:
    """Commit / tag / dirty state of the working tree, or explicit markers."""
    def run(args: Sequence[str]) -> str:
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=10,
                                 cwd=sys.path[0] or ".")
            return out.stdout.strip() if out.returncode == 0 else UNAVAILABLE
        except Exception:
            return UNAVAILABLE

    commit = run(["git", "rev-parse", "HEAD"])
    describe = run(["git", "describe", "--tags", "--always", "--dirty"])
    porcelain = run(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "describe": describe,
        "dirty": (UNAVAILABLE if porcelain == UNAVAILABLE
                  else bool(porcelain.strip())),
    }


@dataclass
class RunMetadata:
    """Everything needed to identify how a result file was produced.

    Any value that cannot be determined is recorded as the string
    ``"unavailable"`` rather than guessed or defaulted.
    """

    benchmark_config_version: str
    timestamp_utc: str
    seed: int
    objective_lambda: float
    mmr_lambda: float
    n_docs: int
    budgets: List[int]
    repeats: int
    ilp_repeats: int
    stopping_policy: str
    include_ilp: bool
    ilp_time_limit: Optional[float]
    ilp_max_docs: int
    ilp_solver: str
    embedding_backend: str
    embedding_model_requested: str
    embedding_fallback_used: bool
    tokenizer_encoding: str
    tokenizer_exact: bool
    query: str
    corpus_token_stats: Dict[str, float]
    git: Dict[str, Any]
    python_version: str
    platform: str
    packages: Dict[str, str]
    primary_metrics: List[str]
    diagnostic_metrics: List[str]
    notes: List[str] = field(default_factory=list)


def collect_metadata(embedder: Embedder, corpus: EncodedCorpus, *,
                     seed: int, objective_lambda: float, mmr_lambda: float,
                     n_docs: int, budgets: Sequence[int], repeats: int,
                     ilp_repeats: int, include_ilp: bool,
                     ilp_time_limit: Optional[float], ilp_max_docs: int,
                     stopping_policy: str, requested_model: str) -> RunMetadata:
    counter = embedder.token_counter
    tokens = corpus.tokens
    return RunMetadata(
        benchmark_config_version=BENCHMARK_CONFIG_VERSION,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        seed=seed,
        objective_lambda=objective_lambda,
        mmr_lambda=mmr_lambda,
        n_docs=n_docs,
        budgets=list(budgets),
        repeats=repeats,
        ilp_repeats=ilp_repeats,
        stopping_policy=stopping_policy,
        include_ilp=include_ilp,
        ilp_time_limit=ilp_time_limit,
        ilp_max_docs=ilp_max_docs,
        ilp_solver="PULP_CBC_CMD" if include_ilp else UNAVAILABLE,
        embedding_backend=corpus.backend_name,
        embedding_model_requested=requested_model,
        embedding_fallback_used=(corpus.backend_name != requested_model),
        tokenizer_encoding=getattr(counter, "encoding_name", UNAVAILABLE),
        tokenizer_exact=bool(getattr(counter, "exact", False)),
        query=corpus.query,
        corpus_token_stats={
            "min": int(tokens.min()) if len(tokens) else 0,
            "median": float(np.median(tokens)) if len(tokens) else 0.0,
            "max": int(tokens.max()) if len(tokens) else 0,
            "mean": float(tokens.mean()) if len(tokens) else 0.0,
            "total": int(tokens.sum()) if len(tokens) else 0,
        },
        git=_git_info(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        packages={name: _package_version(name) for name in
                  ("numpy", "pulp", "sentence-transformers", "tiktoken",
                   "matplotlib", "scipy", "torch")},
        primary_metrics=["score", "unique_coverage", "redundancy",
                         "redundancy_per_pair", "tokens_used", "n_selected"],
        diagnostic_metrics=["density_diagnostic", "token_savings_pct",
                            "latency_ms"],
        notes=[
            "latency_ms is NOT comparable across solver classes: heuristic "
            "timings are in-process NumPy loops, the ILP timing additionally "
            "covers model construction, LP file IO and CBC subprocess startup "
            "(see latency_breakdown per row).",
            "density_diagnostic is deprecated and must not be used to rank "
            "methods; it is maximized by near-empty selections.",
            "objective_lambda was selected on this evaluation corpus in v0 and "
            "is carried over unchanged; it is not a held-out value.",
            "The corpus generator is unchanged from v0.",
        ],
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass
class MethodMetrics:
    """Reported metrics for one method at one budget."""

    method: str
    budget: int
    # -- primary --------------------------------------------------------- #
    score: float
    unique_coverage: float
    redundancy: float
    redundancy_per_pair: Optional[float]
    relevance_sum: float
    tokens_used: int
    n_selected: int
    # -- provenance of the comparison ------------------------------------ #
    score_vs_reference: Optional[float]
    reference_kind: str
    solve_status: str
    solve_scope: str
    stop_reason: str
    # -- diagnostic (never used for ranking) ----------------------------- #
    density_diagnostic: float
    token_savings_pct: float
    latency_ms: float
    latency_runs: int
    latency_breakdown: Dict[str, Any]
    meta: Dict[str, Any]


def unique_coverage(result: SelectionResult, corpus: EncodedCorpus,
                    threshold: float = 0.85) -> float:
    r"""Fraction of distinct information clusters covered by the selection.

    Documents are greedily clustered at ``sim >= threshold``; the score is
    ``covered_clusters / total_clusters``. This is the "core information
    coverage" that Token Savings must not destroy.

    UNCHANGED FROM v0. Known limitations (clusters that no budget can reach are
    still counted in the denominator; off-topic distractor clusters count the
    same as on-topic ones) are recorded in docs/experiment_repair.md and are not
    addressed at this step.
    """
    n = len(corpus)
    if n == 0:
        return 0.0
    sim = corpus.similarity
    unassigned = np.ones(n, dtype=bool)
    order = np.argsort(-corpus.relevance, kind="stable")
    clusters: List[np.ndarray] = []
    for i in order:
        if not unassigned[i]:
            continue
        members = np.flatnonzero(unassigned & (sim[i] >= threshold))
        members = np.union1d(members, [i])
        unassigned[members] = False
        clusters.append(members)
    if not clusters:
        return 0.0
    selected = set(result.indices)
    covered = sum(1 for c in clusters if selected.intersection(c.tolist()))
    return covered / len(clusters)


def evaluate(result: SelectionResult, corpus: EncodedCorpus,
             reference_score: Optional[float] = None,
             reference_kind: str = "none",
             latency_runs: int = 1) -> MethodMetrics:
    """Convert a :class:`SelectionResult` into the reported metric row.

    ``reference_kind`` names what ``score_vs_reference`` is measured against:

    ``proven_global_optimum``
        an ILP run proved optimal over the full candidate set.
    ``proven_pool_optimum``
        an ILP run proved optimal over a *restricted* pool. This is a lower
        bound on the true optimum, so ratios against it can exceed 1.
    ``best_known``
        no proof available; the best objective any method reached.

    The ratio is suppressed when the reference is non-positive, because a ratio
    against a negative or near-zero reference is not an approximation quality.
    """
    ratio = None
    if reference_score is not None and reference_score > 1e-9:
        ratio = result.score / reference_score
    pairs = result.n_selected * (result.n_selected - 1) / 2.0
    return MethodMetrics(
        method=result.method,
        budget=result.budget,
        score=result.score,
        unique_coverage=unique_coverage(result, corpus),
        redundancy=result.redundancy,
        redundancy_per_pair=(result.redundancy / pairs) if pairs > 0 else None,
        relevance_sum=result.relevance_sum,
        tokens_used=result.tokens_used,
        n_selected=result.n_selected,
        score_vs_reference=ratio,
        reference_kind=reference_kind,
        solve_status=result.status.value,
        solve_scope=result.scope.value,
        stop_reason=str(result.meta.get("stop_reason", "n/a")),
        density_diagnostic=result.density_diagnostic,
        token_savings_pct=result.token_savings_pct,
        latency_ms=result.latency_ms,
        latency_runs=latency_runs,
        latency_breakdown={
            "model_build_ms": result.meta.get("model_build_ms"),
            "solver_ms": result.meta.get("solver_ms"),
            "includes_subprocess_spawn": result.method == "ilp",
        },
        meta={k: v for k, v in result.meta.items()
              if k not in ("model_build_ms", "solver_ms", "stop_reason")},
    )


# --------------------------------------------------------------------------- #
# Benchmark driver
# --------------------------------------------------------------------------- #
def solve_ilp_reference(corpus: EncodedCorpus, full: ContextKnapsack,
                        budget: int, objective_lambda: float, mmr_lambda: float,
                        max_docs: int, time_limit: float) -> SelectionResult:
    """Run the exact solver, optionally restricted to the top-``max_docs`` pool.

    The linearized QKP has ``O(N^2)`` pair variables, so CBC stops proving
    optimality well before a realistic candidate list is exhausted. Restricting
    it to the ``max_docs`` most relevant documents keeps the reference provably
    optimal **on a stated sub-problem**. Indices and the objective are mapped
    back to the full corpus, so every reported metric stays comparable across
    methods.

    The result carries ``scope = RESTRICTED_POOL`` in that case, which makes
    :attr:`SelectionResult.is_proven_global_optimum` false: a pool optimum is a
    lower bound on the global optimum, never a substitute for it. v0 copied the
    sub-problem's optimality flag through unqualified.
    """
    n = len(corpus)
    if max_docs <= 0 or n <= max_docs:
        return full.solve_ilp(time_limit=time_limit,
                              scope=SolveScope.FULL_CORPUS)

    pool = np.argsort(-corpus.relevance, kind="stable")[:max_docs]
    pool.sort()
    sub = ContextKnapsack(corpus.relevance[pool],
                          corpus.similarity[np.ix_(pool, pool)],
                          corpus.tokens[pool], budget=budget,
                          objective_lambda=objective_lambda,
                          mmr_lambda=mmr_lambda)
    sub_result = sub.solve_ilp(time_limit=time_limit,
                               scope=SolveScope.RESTRICTED_POOL)
    global_indices = [int(pool[i]) for i in sub_result.indices]
    score, relevance_sum, redundancy = full.score(global_indices)
    sub_result.indices = global_indices
    sub_result.score = score
    sub_result.relevance_sum = relevance_sum
    sub_result.redundancy = redundancy
    sub_result.meta["candidate_pool"] = int(max_docs)
    sub_result.meta["pool_of_n"] = int(n)
    return sub_result


def _merge_repeats(method: str, results: List[SelectionResult]) -> Tuple[SelectionResult, int]:
    """Collapse repeated runs into one canonical row plus a median latency.

    Every solver here is deterministic, so run 0 is *the* answer and the repeats
    exist only to measure latency. That is asserted rather than assumed: if the
    selections ever differ across repeats the run fails loudly instead of
    silently pairing one run's selection with another run's timing, which is
    what v0's ``best = results[0]`` did.
    """
    canonical = results[0]
    for other in results[1:]:
        if other.indices != canonical.indices:
            raise RuntimeError(
                f"non-deterministic solver '{method}': repeat selections differ "
                f"({canonical.indices} vs {other.indices}); latency medians "
                f"would not describe the reported selection")
    canonical.latency_ms = statistics.median(r.latency_ms for r in results)
    return canonical, len(results)


def run_benchmark(budgets: Sequence[int], n_docs: int = 25,
                  objective_lambda: float = DEFAULT_OBJECTIVE_LAMBDA,
                  mmr_lambda: float = DEFAULT_MMR_LAMBDA,
                  repeats: int = 3, ilp_repeats: int = 1,
                  include_ilp: bool = True, ilp_time_limit: float = 300.0,
                  ilp_max_docs: int = 0, seed: int = RANDOM_SEED,
                  stop_on_nonpositive_gain: bool = True,
                  ) -> Tuple[EncodedCorpus, Dict[int, Dict[str, MethodMetrics]], RunMetadata]:
    """Run every solver across ``budgets``; latency is the median of ``repeats``.

    The same latency procedure (median of N runs, N recorded per row) is applied
    to every method. ``ilp_repeats`` defaults to 1 only because the exact solver
    costs seconds per call; the count is recorded in the output rather than
    implied.
    """
    # All randomness in this pipeline is explicitly seeded at its source
    # (`random.Random(seed)` in generate_corpus, `default_rng` in the hashing
    # backend). This call seeds the legacy global NumPy RNG defensively so that
    # any future use of it is covered too; it is not load-bearing today.
    np.random.seed(seed)
    query, documents = generate_corpus(n_docs=n_docs, seed=seed)

    requested_model = "all-MiniLM-L6-v2"
    embedder = Embedder(model_name=requested_model)
    t0 = time.perf_counter()
    corpus = build_corpus(query, documents, embedder)
    encode_ms = (time.perf_counter() - t0) * 1000.0

    metadata = collect_metadata(
        embedder, corpus, seed=seed, objective_lambda=objective_lambda,
        mmr_lambda=mmr_lambda, n_docs=n_docs, budgets=budgets, repeats=repeats,
        ilp_repeats=ilp_repeats, include_ilp=include_ilp,
        ilp_time_limit=ilp_time_limit, ilp_max_docs=ilp_max_docs,
        stopping_policy=("positive_marginal_gain" if stop_on_nonpositive_gain
                         else "fill_budget_v0"),
        requested_model=requested_model)

    print(f"corpus: {len(corpus)} docs | backend: {corpus.backend_name} | "
          f"tokens: min={corpus.tokens.min()} median={int(np.median(corpus.tokens))} "
          f"max={corpus.tokens.max()} | encode {encode_ms:.0f} ms")
    print(f"query: {query!r}")
    print(f"objective_lambda={objective_lambda} mmr_lambda={mmr_lambda} "
          f"stopping={metadata.stopping_policy} seed={seed}")
    if metadata.embedding_fallback_used:
        print(f"WARNING: requested {requested_model!r} but ran on "
              f"{corpus.backend_name!r} - results are NOT comparable to v0")
    if include_ilp and 0 < ilp_max_docs < len(corpus):
        print(f"note: exact ILP runs on the top-{ilp_max_docs} documents by "
              f"relevance. Its optimum is a POOL optimum (a lower bound on the "
              f"global optimum over all {len(corpus)} documents), never a "
              f"proven global optimum.")
    print()

    table: Dict[int, Dict[str, MethodMetrics]] = {}
    for budget in budgets:
        knapsack = ContextKnapsack(corpus.relevance, corpus.similarity,
                                   corpus.tokens, budget=budget,
                                   objective_lambda=objective_lambda,
                                   mmr_lambda=mmr_lambda,
                                   stop_on_nonpositive_gain=stop_on_nonpositive_gain)
        runs: Dict[str, List[SelectionResult]] = {}
        for _ in range(max(1, repeats)):
            for method, result in knapsack.solve_all(include_ilp=False).items():
                runs.setdefault(method, []).append(result)
        if include_ilp:
            try:
                runs["ilp"] = [
                    solve_ilp_reference(corpus, knapsack, budget,
                                        objective_lambda, mmr_lambda,
                                        ilp_max_docs, ilp_time_limit)
                    for _ in range(max(1, ilp_repeats))]
            except RuntimeError as exc:
                print(f"  ILP skipped: {exc}")

        merged: Dict[str, Tuple[SelectionResult, int]] = {
            method: _merge_repeats(method, results)
            for method, results in runs.items()}

        ilp = merged.get("ilp", (None, 0))[0]
        if ilp is not None and ilp.is_proven_global_optimum:
            reference, reference_kind = ilp.score, "proven_global_optimum"
        elif ilp is not None and ilp.is_proven_pool_optimum:
            reference, reference_kind = ilp.score, "proven_pool_optimum"
        else:
            reference = max((r.score for r, _ in merged.values()), default=None)
            reference_kind = "best_known"
        table[budget] = {m: evaluate(r, corpus, reference, reference_kind, runs_n)
                         for m, (r, runs_n) in merged.items()}
        _print_budget_block(budget, table[budget])
    return corpus, table, metadata


_REFERENCE_HEADERS = {
    "proven_global_optimum": "vs opt",
    "proven_pool_optimum": "vs pool",
    "best_known": "vs best",
    "none": "vs ref",
}
_STATUS_SUFFIX = {
    SolveStatus.PROVEN_OPTIMAL.value: "",
    SolveStatus.FEASIBLE_NOT_PROVEN.value: " [incumbent]",
    SolveStatus.INFEASIBLE.value: " [infeasible]",
    SolveStatus.UNBOUNDED.value: " [unbounded]",
    SolveStatus.NOT_SOLVED.value: " [not solved]",
    SolveStatus.UNDEFINED.value: " [undefined]",
    SolveStatus.HEURISTIC.value: "",
}


def _print_budget_block(budget: int, rows: Dict[str, MethodMetrics]) -> None:
    kinds = {m.reference_kind for m in rows.values()}
    kind = next(iter(kinds)) if len(kinds) == 1 else "none"
    ref_label = _REFERENCE_HEADERS.get(kind, "vs ref")
    print(f"W_max = {budget} tokens")
    header = (f"  {'method':<24}{'score':>9}{'cover':>8}{'redund':>9}{'r/pair':>8}"
              f"{'tok':>7}{'docs':>6}{ref_label:>9}{'lat(ms)':>10}{'dens*':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for method in METHOD_ORDER:
        m = rows.get(method)
        if m is None:
            continue
        vs = f"{m.score_vs_reference:.3f}" if m.score_vs_reference is not None else "-"
        label = METHOD_LABELS[method] + _STATUS_SUFFIX.get(m.solve_status, "")
        if method == "ilp" and m.solve_scope == SolveScope.RESTRICTED_POOL.value:
            label += " [pool]"
        rpp = f"{m.redundancy_per_pair:.3f}" if m.redundancy_per_pair is not None else "-"
        print(f"  {label:<24}{m.score:>9.3f}{m.unique_coverage:>8.2f}"
              f"{m.redundancy:>9.3f}{rpp:>8}{m.tokens_used:>7}{m.n_selected:>6}"
              f"{vs:>9}{m.latency_ms:>10.2f}{m.density_diagnostic:>8.2f}")
    print("  * dens = deprecated diagnostic, NOT a ranking metric "
          "(maximized by near-empty selections).")
    print("    lat(ms) is not comparable across solver classes; see JSON "
          "latency_breakdown.")
    if kind == "proven_pool_optimum":
        print("    'vs pool' compares against a POOL optimum (a lower bound); "
              "ratios above 1.000 are possible and do not beat the true optimum.")
    elif kind == "best_known":
        print("    'vs best' compares against the best objective reached by any "
              "method; no optimality was proven at this budget.")
    print()


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_results(table: Dict[int, Dict[str, MethodMetrics]],
                 path: str = "benchmark_results.png") -> Optional[str]:
    """Four-panel comparison over the PRIMARY metrics only.

    Context Density was removed from the plot at the repair step: it is a
    deprecated diagnostic and plotting it invited reading it as a quality
    ranking. Latency was removed because it is not comparable across solver
    classes.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots")
        return None

    budgets = sorted(table)
    methods = [m for m in METHOD_ORDER if any(m in table[b] for b in budgets)]
    panels = [
        ("score", "Objective  $Score(S)$", False),
        ("unique_coverage", "Coverage (fraction of clusters)", False),
        ("redundancy_per_pair", "Redundancy per selected pair", False),
        ("tokens_used", "Tokens used", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, title, log_y) in zip(axes.ravel(), panels):
        for method in methods:
            xs = [b for b in budgets
                  if method in table[b] and getattr(table[b][method], metric) is not None]
            ys = [getattr(table[b][method], metric) for b in xs]
            ax.plot(xs, ys, marker="o", label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method], linestyle=METHOD_STYLES[method],
                    linewidth=1.8, markersize=4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Token budget $W_{max}$", fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)
        if log_y:
            ax.set_yscale("log")
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("Context-Knapsack solver comparison - primary metrics",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"plot written to {path}")
    return path


def dump_json(table: Dict[int, Dict[str, MethodMetrics]],
              metadata: RunMetadata, path: str) -> None:
    """Persist the raw metric rows **with** the metadata that identifies them.

    The v0 payload was metric rows only, so a result file could not be
    attributed to a seed, a lambda, a backend or a commit. The schema is now
    ``{"metadata": ..., "results": ...}``.
    """
    payload = {
        "metadata": asdict(metadata),
        "results": {str(b): {m: asdict(row) for m, row in rows.items()}
                    for b, rows in table.items()},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print(f"metrics written to {path}")


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[256, 512, 1024, 2048])
    parser.add_argument("--n-docs", type=int, default=25)
    parser.add_argument("--objective-lambda", dest="objective_lambda", type=float,
                        default=DEFAULT_OBJECTIVE_LAMBDA,
                        help="redundancy weight of the research objective")
    parser.add_argument("--lambda", dest="objective_lambda", type=float,
                        help="deprecated alias for --objective-lambda")
    parser.add_argument("--mmr-lambda", dest="mmr_lambda", type=float,
                        default=DEFAULT_MMR_LAMBDA,
                        help="MMR trade-off weight (baseline only; not the "
                             "objective's lambda)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="timing repeats for the heuristics")
    parser.add_argument("--ilp-repeats", type=int, default=1,
                        help="timing repeats for the exact solver")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--ilp-time-limit", type=float, default=300.0)
    parser.add_argument("--ilp-max-docs", type=int, default=0,
                        help="restrict the exact solver to the top-M documents "
                             "by relevance (0 = no restriction). The result is "
                             "then a POOL optimum, not a global optimum.")
    parser.add_argument("--no-gain-stop", action="store_true",
                        help="disable the common stopping policy and let every "
                             "heuristic fill the budget (v0 Top-K behaviour). "
                             "For ablation only.")
    parser.add_argument("--no-ilp", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-path", default="benchmark_results.png")
    parser.add_argument("--json-path", default="benchmark_results.json")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # stream progress when piped

    _, table, metadata = run_benchmark(
        budgets=args.budgets, n_docs=args.n_docs,
        objective_lambda=args.objective_lambda, mmr_lambda=args.mmr_lambda,
        repeats=args.repeats, ilp_repeats=args.ilp_repeats,
        include_ilp=not args.no_ilp, ilp_time_limit=args.ilp_time_limit,
        ilp_max_docs=args.ilp_max_docs, seed=args.seed,
        stop_on_nonpositive_gain=not args.no_gain_stop,
    )
    dump_json(table, metadata, args.json_path)
    if not args.no_plot:
        plot_results(table, args.plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
