r"""Context-selection solvers under a hard token budget.

Problem
-------
Given ``N`` candidate documents with relevance ``r_i = rel(q, d_i)``, pairwise
similarity ``s_ij = sim(d_i, d_j)`` and token cost ``w_i``, select ``S`` to

.. math::

    \max_{S} \; Score(S) = \sum_{i \in S} rel(q, d_i)
                            - \lambda \sum_{i<j \in S} sim(d_i, d_j)
    \quad \text{s.t.} \quad \sum_{i \in S} w_i \le W_{max}

This is a Quadratic Knapsack Problem (NP-hard). Four solvers are provided:

``top_k``            naive relevance ranking, fills the budget (baseline).
``mmr``              classic Maximal Marginal Relevance, budget-truncated.
``greedy``           token-aware MMR: maximizes marginal gain **per token**.
``ilp``              exact ground truth via PuLP (linearized QKP).

All solvers consume the pre-computed matrices from
:func:`embedder.build_corpus`; none of them touch raw embeddings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Redundancy weight. The penalty is a *sum* over all selected pairs, so it grows
# as O(|S|^2) while relevance grows as O(|S|). Documents that survive retrieval are
# all on-topic, so their mean pairwise cosine is high (~0.25 on the benchmark
# corpus with all-MiniLM-L6-v2) while rel(q, d) peaks around 0.5: at |S| = 10 the
# penalty already outweighs total relevance for lambda >= 0.4, collapsing the
# selection to a handful of documents. 0.1 balances diversity and coverage.
DEFAULT_LAMBDA = 0.1


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class SelectionResult:
    """Outcome of one solver run.

    Attributes
    ----------
    method: solver name.
    indices: selected document indices, in selection order.
    tokens_used: total token cost of the selection.
    budget: the ``W_max`` the solver was run against.
    score: objective ``Score(S)``.
    relevance_sum: ``sum_i rel(q, d_i)`` over ``S`` (no penalty).
    redundancy: ``sum_{i<j} sim(d_i, d_j)`` over ``S``.
    latency_ms: wall-clock solve time in milliseconds.
    optimal: ``True`` only when an exact solver proved optimality.
    """

    method: str
    indices: List[int]
    tokens_used: int
    budget: int
    score: float
    relevance_sum: float
    redundancy: float
    latency_ms: float
    optimal: bool = False
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        return len(self.indices)

    @property
    def budget_utilization(self) -> float:
        """Fraction of ``W_max`` actually spent."""
        return self.tokens_used / self.budget if self.budget else 0.0

    @property
    def token_savings_pct(self) -> float:
        """Percentage of the budget left unspent."""
        return 100.0 * (1.0 - self.budget_utilization)

    @property
    def density(self) -> float:
        """Context Density Score: net objective per 1,000 tokens spent."""
        return 1000.0 * self.score / self.tokens_used if self.tokens_used else 0.0


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class ContextKnapsack:
    r"""Token-budgeted context packer.

    Parameters
    ----------
    relevance: ``(N,)`` ``rel(q, d_i)``.
    similarity: ``(N, N)`` symmetric ``sim(d_i, d_j)``; the diagonal is ignored.
    tokens: ``(N,)`` positive token cost per document.
    budget: ``W_max``, the hard token budget.
    lambda_: redundancy weight ``\lambda >= 0``. ``0`` reduces the objective to
        pure relevance; larger values buy diversity.

    Documents whose individual cost exceeds ``W_max`` are pruned once at
    construction (``infeasible_indices``) rather than re-checked in every loop.
    """

    def __init__(self, relevance: np.ndarray, similarity: np.ndarray,
                 tokens: np.ndarray, budget: int,
                 lambda_: float = DEFAULT_LAMBDA) -> None:
        self.relevance = np.asarray(relevance, dtype=np.float64).ravel()
        self.similarity = np.asarray(similarity, dtype=np.float64)
        self.tokens = np.asarray(tokens, dtype=np.int64).ravel()
        self.budget = int(budget)
        self.lambda_ = float(lambda_)

        n = self.relevance.shape[0]
        if self.tokens.shape[0] != n:
            raise ValueError(f"tokens has {self.tokens.shape[0]} entries, expected {n}")
        if n and self.similarity.shape != (n, n):
            raise ValueError(f"similarity must be ({n}, {n}), got {self.similarity.shape}")
        if self.lambda_ < 0:
            raise ValueError("lambda_ must be >= 0")
        if self.budget < 0:
            raise ValueError("budget must be >= 0")

        self.n = n
        if n:
            # Work on a defensive copy: zero diagonal, symmetrize, no negatives.
            self.similarity = np.array(self.similarity, dtype=np.float64, copy=True)
            self.similarity = 0.5 * (self.similarity + self.similarity.T)
            np.fill_diagonal(self.similarity, 0.0)
            np.clip(self.similarity, 0.0, None, out=self.similarity)

        self.feasible_mask = (self.tokens <= self.budget) & (self.tokens > 0)
        self.infeasible_indices = np.flatnonzero(~self.feasible_mask).tolist()

    # -- objective ---------------------------------------------------------- #
    def score(self, indices: Sequence[int]) -> Tuple[float, float, float]:
        """Return ``(score, relevance_sum, redundancy)`` for a selection."""
        idx = np.asarray(list(indices), dtype=np.int64)
        if idx.size == 0:
            return 0.0, 0.0, 0.0
        relevance_sum = float(self.relevance[idx].sum())
        # Upper triangle only: sum_{i<j} = (sum of submatrix) / 2 with zero diag.
        redundancy = float(self.similarity[np.ix_(idx, idx)].sum() / 2.0)
        return relevance_sum - self.lambda_ * redundancy, relevance_sum, redundancy

    def _result(self, method: str, indices: Sequence[int], latency_ms: float,
                optimal: bool = False, **meta: object) -> SelectionResult:
        indices = [int(i) for i in indices]
        score, relevance_sum, redundancy = self.score(indices)
        return SelectionResult(
            method=method,
            indices=indices,
            tokens_used=int(self.tokens[indices].sum()) if indices else 0,
            budget=self.budget,
            score=score,
            relevance_sum=relevance_sum,
            redundancy=redundancy,
            latency_ms=latency_ms,
            optimal=optimal,
            meta=dict(meta),
        )

    # -- baselines ---------------------------------------------------------- #
    def solve_top_k(self, k: Optional[int] = None) -> SelectionResult:
        """Naive retrieval: rank by ``rel(q, d_i)``, take while the budget holds.

        Redundancy is ignored entirely - this is the behaviour the framework is
        measured against. With ``k`` set, stops after ``k`` documents.
        """
        start = time.perf_counter()
        order = np.argsort(-self.relevance, kind="stable")
        selected: List[int] = []
        remaining = self.budget
        for i in order:
            if k is not None and len(selected) >= k:
                break
            cost = int(self.tokens[i])
            if 0 < cost <= remaining:
                selected.append(int(i))
                remaining -= cost
        latency = (time.perf_counter() - start) * 1000.0
        return self._result("top_k", selected, latency, k=k)

    def solve_mmr(self) -> SelectionResult:
        r"""Classic MMR: pick ``argmax`` marginal gain, ignoring token cost.

        .. math::

            i^* = \arg\max_{i \notin S} \; rel(q, d_i)
                  - \lambda \sum_{j \in S} sim(d_i, d_j)

        Selection stops when the budget can no longer fit any candidate or the
        best marginal gain is non-positive.
        """
        return self._greedy(token_aware=False, method="mmr")

    def solve_greedy(self) -> SelectionResult:
        r"""Token-aware MMR: maximize marginal gain **per token spent**.

        .. math::

            i^* = \arg\max_{i \notin S} \;
                  \frac{rel(q, d_i) - \lambda \sum_{j \in S} sim(d_i, d_j)}{w_i}

        This is the standard density-greedy heuristic for the (quadratic)
        knapsack: it prefers concise, non-redundant documents, so short
        high-value snippets are no longer crowded out by long ones.
        """
        return self._greedy(token_aware=True, method="greedy_token_aware")

    def _greedy(self, token_aware: bool, method: str) -> SelectionResult:
        """Shared greedy loop.

        Keeps a running ``penalty`` vector (``sum_{j in S} sim(i, j)`` for every
        candidate) updated with one vectorized add per iteration, so the whole
        loop costs ``O(N * |S|)`` without any per-vector Python arithmetic.
        """
        start = time.perf_counter()
        selected: List[int] = []
        if self.n == 0 or self.budget <= 0:
            return self._result(method, selected,
                                (time.perf_counter() - start) * 1000.0)

        available = self.feasible_mask.copy()
        penalty = np.zeros(self.n, dtype=np.float64)
        remaining = self.budget

        while True:
            fits = available & (self.tokens <= remaining)
            if not fits.any():
                break
            gain = self.relevance - self.lambda_ * penalty
            objective = gain / self.tokens if token_aware else gain
            objective = np.where(fits, objective, -np.inf)
            best = int(np.argmax(objective))
            if gain[best] <= 0.0:
                # No document can improve Score(S) any more; spending tokens on
                # it would strictly hurt the objective.
                break
            selected.append(best)
            remaining -= int(self.tokens[best])
            available[best] = False
            penalty += self.similarity[best]

        latency = (time.perf_counter() - start) * 1000.0
        return self._result(method, selected, latency)

    # -- exact -------------------------------------------------------------- #
    def solve_ilp(self, time_limit: Optional[float] = 60.0,
                  msg: bool = False) -> SelectionResult:
        r"""Exact ground truth via linearized QKP (PuLP / CBC).

        Binary ``x_i`` select documents; ``y_ij`` linearizes the product
        ``x_i x_j`` for ``i < j``:

        .. math::

            \max \sum_i r_i x_i - \lambda \sum_{i<j} s_{ij} y_{ij}
            \quad s.t. \quad \sum_i w_i x_i \le W_{max},
            \; y_{ij} \ge x_i + x_j - 1, \; y_{ij} \ge 0

        Because every ``y_ij`` carries a non-positive objective coefficient, the
        solver always pushes it to its lower bound, so ``y`` can stay continuous
        and the ``y <= x`` constraints are redundant. Pairs with ``s_ij == 0``
        are skipped entirely, which keeps the model sparse.

        Raises ``RuntimeError`` if PuLP is unavailable; ``optimal`` is ``False``
        on the result when the solver hit ``time_limit`` without proving
        optimality.
        """
        start = time.perf_counter()
        if self.n == 0 or self.budget <= 0:
            return self._result("ilp", [], (time.perf_counter() - start) * 1000.0,
                                optimal=True)
        try:
            import pulp
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise RuntimeError("PuLP is required for solve_ilp(); pip install pulp") from exc

        candidates = np.flatnonzero(self.feasible_mask).tolist()
        if not candidates:
            return self._result("ilp", [], (time.perf_counter() - start) * 1000.0,
                                optimal=True)

        problem = pulp.LpProblem("context_knapsack", pulp.LpMaximize)
        x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in candidates}

        objective = [self.relevance[i] * x[i] for i in candidates]
        n_pairs = 0
        if self.lambda_ > 0:
            for a, i in enumerate(candidates):
                for j in candidates[a + 1:]:
                    s = self.similarity[i, j]
                    if s <= 0.0:
                        continue
                    y = pulp.LpVariable(f"y_{i}_{j}", lowBound=0.0, upBound=1.0,
                                        cat="Continuous")
                    problem += y >= x[i] + x[j] - 1, f"link_{i}_{j}"
                    objective.append(-self.lambda_ * s * y)
                    n_pairs += 1

        problem += pulp.lpSum(objective)
        problem += (pulp.lpSum(int(self.tokens[i]) * x[i] for i in candidates)
                    <= self.budget), "token_budget"

        solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
        status = problem.solve(solver)
        selected = [i for i in candidates if x[i].value() is not None
                    and x[i].value() > 0.5]
        latency = (time.perf_counter() - start) * 1000.0
        return self._result(
            "ilp", selected, latency,
            optimal=(pulp.LpStatus[status] == "Optimal"),
            status=pulp.LpStatus[status], n_pairs=n_pairs,
            n_candidates=len(candidates),
        )

    # -- convenience -------------------------------------------------------- #
    def solve_all(self, include_ilp: bool = True,
                  ilp_time_limit: Optional[float] = 60.0) -> Dict[str, SelectionResult]:
        """Run every solver and return ``{method: SelectionResult}``."""
        results = {
            "top_k": self.solve_top_k(),
            "mmr": self.solve_mmr(),
            "greedy_token_aware": self.solve_greedy(),
        }
        if include_ilp:
            try:
                results["ilp"] = self.solve_ilp(time_limit=ilp_time_limit)
            except RuntimeError:
                pass
        return results


def from_corpus(corpus, budget: int, lambda_: float = DEFAULT_LAMBDA) -> ContextKnapsack:
    """Build a :class:`ContextKnapsack` from an :class:`embedder.EncodedCorpus`."""
    return ContextKnapsack(corpus.relevance, corpus.similarity, corpus.tokens,
                           budget=budget, lambda_=lambda_)
