r"""Comparative evaluation of context-packing strategies.

Runs Naive Top-K, classic MMR, Token-Aware Knapsack (greedy) and the exact ILP
ground truth over a seeded synthetic corpus with controlled redundancy, then
reports the three mandated metrics per method:

* **Latency (ms)** - wall-clock solve time per query (retrieval excluded).
* **Context Density Score** - net objective ``Score(S)`` per 1,000 tokens spent.
* **Token Savings (%)** - percentage of ``W_max`` left unspent.

Usage::

    python benchmark.py                      # default sweep + plots
    python benchmark.py --no-ilp --no-plot   # fast smoke run
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from embedder import Embedder, EncodedCorpus, build_corpus
from optimizer import ContextKnapsack, SelectionResult, DEFAULT_LAMBDA

RANDOM_SEED = 42
METHOD_LABELS = {
    "top_k": "Naive Top-K",
    "mmr": "MMR",
    "greedy_token_aware": "Token-Aware Knapsack",
    "ilp": "ILP (exact)",
}
# MMR frequently coincides with the ILP optimum on this corpus; dashing it keeps
# the overlapping curve visible instead of hidden under the exact solver's line.
METHOD_STYLES = {"top_k": "-", "mmr": "--", "greedy_token_aware": "-", "ilp": "-"}
METHOD_COLORS = {
    "top_k": "#9aa0a6",
    "mmr": "#4c8dd9",
    "greedy_token_aware": "#e0742a",
    "ilp": "#3f9a54",
}


# --------------------------------------------------------------------------- #
# Synthetic corpus generation
# --------------------------------------------------------------------------- #
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
# Metrics
# --------------------------------------------------------------------------- #
@dataclass
class MethodMetrics:
    """Reported metrics for one method at one budget."""

    method: str
    budget: int
    latency_ms: float
    density: float
    token_savings_pct: float
    score: float
    relevance_sum: float
    redundancy: float
    tokens_used: int
    n_selected: int
    unique_coverage: float
    score_vs_reference: Optional[float] = None
    reference_is_optimal: bool = False
    proven_optimal: bool = False


def unique_coverage(result: SelectionResult, corpus: EncodedCorpus,
                    threshold: float = 0.85) -> float:
    r"""Fraction of distinct information clusters covered by the selection.

    Documents are greedily clustered at ``sim >= threshold``; the score is
    ``covered_clusters / total_clusters``. This is the "core information
    coverage" that Token Savings must not destroy.
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
             reference_is_optimal: bool = False) -> MethodMetrics:
    """Convert a :class:`SelectionResult` into the reported metric row.

    ``reference_score`` is the ILP optimum when the solver proved optimality,
    otherwise the best objective any method reached (a lower bound on the true
    optimum). ``reference_is_optimal`` says which of the two it is, so an ILP
    run that hit its time limit is never reported as ground truth.
    """
    ratio = None
    if reference_score is not None and abs(reference_score) > 1e-9:
        ratio = result.score / reference_score
    return MethodMetrics(
        method=result.method,
        budget=result.budget,
        latency_ms=result.latency_ms,
        density=result.density,
        token_savings_pct=result.token_savings_pct,
        score=result.score,
        relevance_sum=result.relevance_sum,
        redundancy=result.redundancy,
        tokens_used=result.tokens_used,
        n_selected=result.n_selected,
        unique_coverage=unique_coverage(result, corpus),
        score_vs_reference=ratio,
        reference_is_optimal=reference_is_optimal,
        proven_optimal=result.optimal,
    )


# --------------------------------------------------------------------------- #
# Benchmark driver
# --------------------------------------------------------------------------- #
def solve_ilp_on_top_candidates(corpus: EncodedCorpus, full: ContextKnapsack,
                                budget: int, lambda_: float,
                                max_docs: int, time_limit: float) -> SelectionResult:
    """Run the exact solver, optionally restricted to the top-``max_docs`` pool.

    The linearized QKP has ``O(N^2)`` pair variables, so CBC stops proving
    optimality well before a realistic candidate list is exhausted. Restricting
    it to the ``max_docs`` most relevant documents keeps the ground truth
    *provably* optimal on a stated sub-problem instead of silently reporting a
    timed-out incumbent. Indices and the objective are mapped back to the full
    corpus, so every reported metric stays comparable across methods.
    """
    n = len(corpus)
    if max_docs <= 0 or n <= max_docs:
        return full.solve_ilp(time_limit=time_limit)

    pool = np.argsort(-corpus.relevance, kind="stable")[:max_docs]
    pool.sort()
    sub = ContextKnapsack(corpus.relevance[pool],
                          corpus.similarity[np.ix_(pool, pool)],
                          corpus.tokens[pool], budget=budget, lambda_=lambda_)
    sub_result = sub.solve_ilp(time_limit=time_limit)
    global_indices = [int(pool[i]) for i in sub_result.indices]
    score, relevance_sum, redundancy = full.score(global_indices)
    sub_result.indices = global_indices
    sub_result.score = score
    sub_result.relevance_sum = relevance_sum
    sub_result.redundancy = redundancy
    sub_result.meta["candidate_pool"] = int(max_docs)
    return sub_result


def run_benchmark(budgets: Sequence[int], n_docs: int = 25,
                  lambda_: float = DEFAULT_LAMBDA, repeats: int = 3,
                  include_ilp: bool = True, ilp_time_limit: float = 300.0,
                  ilp_max_docs: int = 0, seed: int = RANDOM_SEED,
                  ) -> Tuple[EncodedCorpus, Dict[int, Dict[str, MethodMetrics]]]:
    """Run every solver across ``budgets``; latency is the median of ``repeats``."""
    np.random.seed(seed)
    query, documents = generate_corpus(n_docs=n_docs, seed=seed)

    t0 = time.perf_counter()
    corpus = build_corpus(query, documents, Embedder())
    encode_ms = (time.perf_counter() - t0) * 1000.0

    print(f"corpus: {len(corpus)} docs | backend: {corpus.backend_name} | "
          f"tokens: min={corpus.tokens.min()} median={int(np.median(corpus.tokens))} "
          f"max={corpus.tokens.max()} | encode {encode_ms:.0f} ms")
    print(f"query: {query!r}")
    if include_ilp and 0 < ilp_max_docs < len(corpus):
        print(f"note: exact ILP runs on the top-{ilp_max_docs} documents by "
              f"relevance (marked '*' = optimal on that pool, not on all "
              f"{len(corpus)} documents)")
    print()

    table: Dict[int, Dict[str, MethodMetrics]] = {}
    for budget in budgets:
        knapsack = ContextKnapsack(corpus.relevance, corpus.similarity,
                                   corpus.tokens, budget=budget, lambda_=lambda_)
        runs: Dict[str, List[SelectionResult]] = {}
        for _ in range(max(1, repeats)):
            for method, result in knapsack.solve_all(include_ilp=False).items():
                runs.setdefault(method, []).append(result)
        if include_ilp:
            try:
                runs["ilp"] = [solve_ilp_on_top_candidates(
                    corpus, knapsack, budget, lambda_, ilp_max_docs, ilp_time_limit)]
            except RuntimeError as exc:
                print(f"  ILP skipped: {exc}")

        merged: Dict[str, SelectionResult] = {}
        for method, results in runs.items():
            best = results[0]
            best.latency_ms = statistics.median(r.latency_ms for r in results)
            merged[method] = best

        ilp = merged.get("ilp")
        if ilp is not None and ilp.optimal:
            reference, reference_is_optimal = ilp.score, True
        else:
            reference = max((r.score for r in merged.values()), default=None)
            reference_is_optimal = False
        table[budget] = {m: evaluate(r, corpus, reference, reference_is_optimal)
                         for m, r in merged.items()}
        _print_budget_block(budget, table[budget])
    return corpus, table


def _print_budget_block(budget: int, rows: Dict[str, MethodMetrics]) -> None:
    reference_is_optimal = any(m.reference_is_optimal for m in rows.values())
    ref_label = "vs opt" if reference_is_optimal else "vs best"
    print(f"W_max = {budget} tokens")
    header = (f"  {'method':<22}{'lat(ms)':>9}{'density':>10}{'save%':>8}"
              f"{'score':>9}{'redund':>9}{'cover':>8}{'docs':>6}{ref_label:>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for method in ("top_k", "mmr", "greedy_token_aware", "ilp"):
        m = rows.get(method)
        if m is None:
            continue
        vs = f"{m.score_vs_reference:.3f}" if m.score_vs_reference is not None else "-"
        label = METHOD_LABELS[method]
        if method == "ilp" and not m.proven_optimal:
            label += " [t/o]"
        elif method == "ilp" and m.reference_is_optimal:
            label += " *"
        print(f"  {label:<22}{m.latency_ms:>9.2f}{m.density:>10.3f}"
              f"{m.token_savings_pct:>8.1f}{m.score:>9.3f}{m.redundancy:>9.3f}"
              f"{m.unique_coverage:>8.2f}{m.n_selected:>6}{vs:>8}")
    if "ilp" in rows and not rows["ilp"].proven_optimal:
        print("  [t/o] ILP hit its time limit: its score is a feasible solution, "
              "not a proven optimum.")
    print()


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_results(table: Dict[int, Dict[str, MethodMetrics]],
                 path: str = "benchmark_results.png") -> Optional[str]:
    """Four-panel comparison: density, savings, redundancy, latency."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots")
        return None

    budgets = sorted(table)
    methods = [m for m in ("top_k", "mmr", "greedy_token_aware", "ilp")
               if any(m in table[b] for b in budgets)]
    panels = [
        ("density", "Context Density Score\n(net relevance per 1k tokens)", False),
        ("token_savings_pct", "Token Savings (%)", False),
        ("redundancy", "Redundancy  $\\sum_{i<j} sim(d_i,d_j)$", False),
        ("latency_ms", "Latency (ms, median)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (field, title, log_y) in zip(axes.ravel(), panels):
        for method in methods:
            xs = [b for b in budgets if method in table[b]]
            ys = [getattr(table[b][method], field) for b in xs]
            ax.plot(xs, ys, marker="o", label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method], linestyle=METHOD_STYLES[method],
                    linewidth=1.8, markersize=4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Token budget $W_{max}$", fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)
        if log_y:
            ax.set_yscale("log")
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("Agentic Context-Knapsack Optimizer - solver comparison",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"plot written to {path}")
    return path


def dump_json(table: Dict[int, Dict[str, MethodMetrics]], path: str) -> None:
    """Persist the raw metric rows for downstream analysis."""
    payload = {str(b): {m: asdict(row) for m, row in rows.items()}
               for b, rows in table.items()}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"metrics written to {path}")


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Defaults reproduce the table reported in README.md: N = 25 keeps the exact
    # ILP provably optimal within seconds, so the whole run finishes in ~1 min.
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[256, 512, 1024, 2048])
    parser.add_argument("--n-docs", type=int, default=25)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--ilp-time-limit", type=float, default=300.0)
    parser.add_argument("--ilp-max-docs", type=int, default=0,
                        help="restrict the exact solver to the top-M documents "
                             "by relevance (0 = no restriction)")
    parser.add_argument("--no-ilp", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-path", default="benchmark_results.png")
    parser.add_argument("--json-path", default="benchmark_results.json")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # stream progress when piped

    _, table = run_benchmark(
        budgets=args.budgets, n_docs=args.n_docs, lambda_=args.lambda_,
        repeats=args.repeats, include_ilp=not args.no_ilp,
        ilp_time_limit=args.ilp_time_limit, ilp_max_docs=args.ilp_max_docs,
        seed=args.seed,
    )
    dump_json(table, args.json_path)
    if not args.no_plot:
        plot_results(table, args.plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
