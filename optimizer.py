r"""Context-selection solvers under a hard token budget.

Problem
-------
Given ``N`` candidate documents with relevance ``r_i = rel(q, d_i)``, pairwise
similarity ``s_ij = sim(d_i, d_j)`` and token cost ``w_i``, select ``S`` to

.. math::

    \max_{S} \; Score(S) = \sum_{i \in S} rel(q, d_i)
                            - \lambda_{obj} \sum_{i<j \in S} sim(d_i, d_j)
    \quad \text{s.t.} \quad \sum_{i \in S} w_i \le W_{max}

This is a Quadratic Knapsack Problem (NP-hard). The objective is unchanged from
the original (v0) study; only the solvers, their labelling and their stopping
semantics were repaired.

Solvers
-------
``top_k``              relevance ranking (baseline).
``mmr``                genuine Maximal Marginal Relevance (max-similarity).
``greedy_objective``   greedy ascent on ``Score(S)``; no token normalization.
``greedy_token_aware`` the proposed heuristic: marginal gain **per token**.
``ilp``                exact solver via PuLP (linearized QKP).

``greedy_objective`` is the method that v0 shipped under the name ``mmr``. It is
kept, under an accurate name, so the token-normalization effect can be isolated
from the choice of redundancy penalty.

Two distinct lambdas
--------------------
``objective_lambda`` weights the **summed** pairwise redundancy of the research
objective above. ``mmr_lambda`` is the convex weight of the **max**-similarity
MMR baseline. They parameterize different formulas and are not interchangeable;
see :meth:`ContextKnapsack.solve_mmr`.

All solvers consume the pre-computed matrices from
:func:`embedder.build_corpus`; none of them touch raw embeddings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Redundancy weight of the research objective. The penalty is a *sum* over all
# selected pairs, so it grows as O(|S|^2) while relevance grows as O(|S|).
# Documents that survive retrieval are all on-topic, so their mean pairwise
# cosine is high (~0.25 on the benchmark corpus with all-MiniLM-L6-v2) while
# rel(q, d) peaks around 0.5: at |S| = 10 the penalty already outweighs total
# relevance for lambda >= 0.4, collapsing the selection to a handful of
# documents. 0.1 balances diversity and coverage.
#
# PROVENANCE: this value was selected on the evaluation corpus itself, not on
# held-out data. It is carried over from v0 unchanged and deliberately NOT
# re-tuned here. See docs/experiment_repair.md, "Remaining limitations".
DEFAULT_OBJECTIVE_LAMBDA = 0.1

# Conventional MMR trade-off weight. NOT tuned, NOT fitted on this corpus, and
# NOT comparable to DEFAULT_OBJECTIVE_LAMBDA - it multiplies relevance in a
# convex combination against a max-similarity term, whereas objective_lambda
# multiplies a summed-similarity penalty. 0.5 is the neutral textbook default.
DEFAULT_MMR_LAMBDA = 0.5

#: Deprecated alias kept so existing callers keep working. Prefer
#: :data:`DEFAULT_OBJECTIVE_LAMBDA`, which says which lambda it is.
DEFAULT_LAMBDA = DEFAULT_OBJECTIVE_LAMBDA


# --------------------------------------------------------------------------- #
# Solver status
# --------------------------------------------------------------------------- #
class SolveStatus(str, Enum):
    """What is actually known about a returned selection.

    The distinction that matters scientifically is between
    :attr:`PROVEN_OPTIMAL` (the solver proved no better feasible selection
    exists) and :attr:`FEASIBLE_NOT_PROVEN` (the solver returned an incumbent,
    e.g. because it hit its time limit). v0 collapsed both into ``"Optimal"``
    because ``pulp.LpStatus`` reports *that a solution came back*, not that
    optimality was established.
    """

    PROVEN_OPTIMAL = "proven_optimal"
    FEASIBLE_NOT_PROVEN = "feasible_not_proven"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    NOT_SOLVED = "not_solved"
    UNDEFINED = "undefined"
    #: Heuristics carry no optimality guarantee at all.
    HEURISTIC = "heuristic_no_guarantee"


class SolveScope(str, Enum):
    """Which problem a status refers to.

    A selection proved optimal over a restricted candidate pool is *not* a
    global optimum: it is a lower bound on the true optimum over the full
    corpus. Heuristics run over the full corpus but prove nothing.
    """

    FULL_CORPUS = "full_corpus"
    RESTRICTED_POOL = "restricted_pool"


def classify_pulp_status(problem, pulp_module) -> SolveStatus:
    """Map a solved PuLP problem onto :class:`SolveStatus`.

    ``problem.status`` answers "did the solve terminate cleanly"; only
    ``problem.sol_status`` distinguishes a *proven* optimum
    (``LpSolutionOptimal``) from an *integer-feasible incumbent*
    (``LpSolutionIntegerFeasible``), which is what CBC returns when it stops on
    a time limit. v0 read only the former and therefore reported timed-out
    incumbents as proven optima.

    When ``sol_status`` is unavailable (older PuLP), the conservative branch is
    taken: a returned solution is reported as feasible-but-unproven rather than
    optimal.
    """
    status = getattr(problem, "status", None)
    sol_status = getattr(problem, "sol_status", None)

    if sol_status is None:
        if status == pulp_module.LpStatusOptimal:
            return SolveStatus.FEASIBLE_NOT_PROVEN
        if status == pulp_module.LpStatusInfeasible:
            return SolveStatus.INFEASIBLE
        if status == pulp_module.LpStatusUnbounded:
            return SolveStatus.UNBOUNDED
        if status == pulp_module.LpStatusNotSolved:
            return SolveStatus.NOT_SOLVED
        return SolveStatus.UNDEFINED

    if sol_status == pulp_module.LpSolutionOptimal:
        # Both must agree before anything is called proven.
        if status == pulp_module.LpStatusOptimal:
            return SolveStatus.PROVEN_OPTIMAL
        return SolveStatus.FEASIBLE_NOT_PROVEN
    if sol_status == pulp_module.LpSolutionIntegerFeasible:
        return SolveStatus.FEASIBLE_NOT_PROVEN
    if sol_status == pulp_module.LpSolutionInfeasible:
        return SolveStatus.INFEASIBLE
    if sol_status == pulp_module.LpSolutionUnbounded:
        return SolveStatus.UNBOUNDED
    if sol_status == pulp_module.LpSolutionNoSolutionFound:
        return SolveStatus.NOT_SOLVED
    return SolveStatus.UNDEFINED


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
    status: see :class:`SolveStatus`.
    scope: see :class:`SolveScope` - which problem ``status`` refers to.

    There is deliberately no writable ``optimal`` flag. Optimality is derived
    from ``status`` **and** ``scope`` together, so a pool-restricted or
    time-limited result cannot be relabelled as a proven global optimum by
    assignment, which is how v0 mislabelled both.
    """

    method: str
    indices: List[int]
    tokens_used: int
    budget: int
    score: float
    relevance_sum: float
    redundancy: float
    latency_ms: float
    status: SolveStatus = SolveStatus.HEURISTIC
    scope: SolveScope = SolveScope.FULL_CORPUS
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        return len(self.indices)

    @property
    def is_proven_global_optimum(self) -> bool:
        """True only for a proven optimum over the **full** candidate set."""
        return (self.status is SolveStatus.PROVEN_OPTIMAL
                and self.scope is SolveScope.FULL_CORPUS)

    @property
    def is_proven_pool_optimum(self) -> bool:
        """True for a proven optimum over a **restricted** pool.

        Such a value is a lower bound on the global optimum, never an upper one.
        """
        return (self.status is SolveStatus.PROVEN_OPTIMAL
                and self.scope is SolveScope.RESTRICTED_POOL)

    @property
    def optimal(self) -> bool:
        """Deprecated alias for :attr:`is_proven_global_optimum` (read-only)."""
        return self.is_proven_global_optimum

    @property
    def budget_utilization(self) -> float:
        """Fraction of ``W_max`` actually spent."""
        return self.tokens_used / self.budget if self.budget else 0.0

    @property
    def token_savings_pct(self) -> float:
        """Percentage of the budget left unspent.

        A budget-utilization statistic, not a quality measure: leaving tokens
        unspent is only good if coverage is retained, which this does not check.
        """
        return 100.0 * (1.0 - self.budget_utilization)

    @property
    def density_diagnostic(self) -> float:
        """DEPRECATED diagnostic: net objective per 1,000 tokens spent.

        Retained for continuity with v0 and for inspection only. **Not** a
        quality metric and never used to rank methods: the numerator carries an
        ``O(|S|^2)`` penalty while the denominator grows ``O(|S|)``, so the
        quantity is maximized by degenerate near-empty selections. See
        docs/experiment_repair.md section 7.
        """
        return 1000.0 * self.score / self.tokens_used if self.tokens_used else 0.0

    @property
    def density(self) -> float:
        """Deprecated alias for :attr:`density_diagnostic`."""
        return self.density_diagnostic


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
    objective_lambda: redundancy weight of the research objective,
        ``\lambda_{obj} >= 0``. ``0`` reduces the objective to pure relevance.
    mmr_lambda: trade-off weight of the MMR baseline only, in ``[0, 1]``. Has no
        effect on the objective or on any other solver.
    stop_on_nonpositive_gain: the common stopping policy (see
        :meth:`_select`). ``False`` restores v0's Top-K behaviour and exists
        only so the two policies can be compared.

    Documents whose individual cost exceeds ``W_max`` are pruned once at
    construction (``infeasible_indices``) rather than re-checked in every loop.
    """

    def __init__(self, relevance: np.ndarray, similarity: np.ndarray,
                 tokens: np.ndarray, budget: int,
                 objective_lambda: float = DEFAULT_OBJECTIVE_LAMBDA,
                 mmr_lambda: float = DEFAULT_MMR_LAMBDA,
                 stop_on_nonpositive_gain: bool = True,
                 lambda_: Optional[float] = None) -> None:
        if lambda_ is not None:  # deprecated keyword, kept for v0 callers
            objective_lambda = lambda_
        self.relevance = np.asarray(relevance, dtype=np.float64).ravel()
        self.similarity = np.asarray(similarity, dtype=np.float64)
        self.tokens = np.asarray(tokens, dtype=np.int64).ravel()
        self.budget = int(budget)
        self.objective_lambda = float(objective_lambda)
        self.mmr_lambda = float(mmr_lambda)
        self.stop_on_nonpositive_gain = bool(stop_on_nonpositive_gain)

        n = self.relevance.shape[0]
        if self.tokens.shape[0] != n:
            raise ValueError(f"tokens has {self.tokens.shape[0]} entries, expected {n}")
        if n and self.similarity.shape != (n, n):
            raise ValueError(f"similarity must be ({n}, {n}), got {self.similarity.shape}")
        if self.objective_lambda < 0:
            raise ValueError("objective_lambda must be >= 0")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")
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

    @property
    def lambda_(self) -> float:
        """Deprecated alias for :attr:`objective_lambda`."""
        return self.objective_lambda

    # -- objective ---------------------------------------------------------- #
    def score(self, indices: Sequence[int]) -> Tuple[float, float, float]:
        """Return ``(score, relevance_sum, redundancy)`` for a selection.

        Unchanged from v0: this is the research objective and is deliberately
        not part of the repair.
        """
        idx = np.asarray(list(indices), dtype=np.int64)
        if idx.size == 0:
            return 0.0, 0.0, 0.0
        relevance_sum = float(self.relevance[idx].sum())
        # Upper triangle only: sum_{i<j} = (sum of submatrix) / 2 with zero diag.
        redundancy = float(self.similarity[np.ix_(idx, idx)].sum() / 2.0)
        return relevance_sum - self.objective_lambda * redundancy, relevance_sum, redundancy

    def marginal_gain(self, candidate: int, selected: Sequence[int]) -> float:
        r"""Marginal contribution of ``candidate`` to ``Score(selected)``.

        .. math::

            \Delta_i = rel(q, d_i)
                       - \lambda_{obj} \sum_{j \in S} sim(d_i, d_j)

        This single definition drives the common stopping policy for **every**
        heuristic, independently of how each one ranks its candidates.
        """
        idx = np.asarray(list(selected), dtype=np.int64)
        penalty = float(self.similarity[candidate, idx].sum()) if idx.size else 0.0
        return float(self.relevance[candidate]) - self.objective_lambda * penalty

    def _result(self, method: str, indices: Sequence[int], latency_ms: float,
                status: SolveStatus = SolveStatus.HEURISTIC,
                scope: SolveScope = SolveScope.FULL_CORPUS,
                **meta: object) -> SelectionResult:
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
            status=status,
            scope=scope,
            meta=dict(meta),
        )

    # -- shared selection loop ---------------------------------------------- #
    def _select(self, priority_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                method: str, k: Optional[int] = None,
                **meta: object) -> SelectionResult:
        r"""Greedy selection under the **common stopping policy**.

        Every heuristic in this module runs through this loop and differs only
        in ``priority_fn``, which decides *which* eligible candidate is taken
        next. Eligibility, and therefore termination, is identical for all of
        them:

        A candidate ``i`` is eligible iff

        1. it has not been selected yet, and
        2. ``w_i <= remaining budget``, and
        3. ``\Delta_i = r_i - \lambda_{obj} \sum_{j \in S} s_ij > 0``
           (when ``stop_on_nonpositive_gain``, the default).

        Selection stops when no eligible candidate remains. Condition 3 is the
        repair: v0 applied it to the greedy solvers but not to Top-K, so Top-K
        alone kept buying documents that lowered ``Score(S)``, and the reported
        gap measured the stopping rule rather than the algorithm.

        Note that condition 3 is stated on the **objective's** marginal gain for
        every method, including MMR, whose own ranking score is unrelated to it.
        Ranking rule and termination rule are kept strictly separate.

        ``priority_fn(gain, max_sim_to_selected) -> (N,) priority vector``;
        higher is taken first.
        """
        start = time.perf_counter()
        selected: List[int] = []
        if self.n == 0 or self.budget <= 0:
            return self._result(method, selected,
                                (time.perf_counter() - start) * 1000.0,
                                stopping_policy=self._stopping_policy_name(),
                                **meta)

        available = self.feasible_mask.copy()
        penalty = np.zeros(self.n, dtype=np.float64)
        max_sim = np.zeros(self.n, dtype=np.float64)
        remaining = self.budget
        stop_reason = "no_candidate_fits_budget"

        while True:
            if k is not None and len(selected) >= k:
                stop_reason = "k_reached"
                break
            fits = available & (self.tokens <= remaining)
            if not fits.any():
                stop_reason = "no_candidate_fits_budget"
                break
            gain = self.relevance - self.objective_lambda * penalty
            if self.stop_on_nonpositive_gain:
                eligible = fits & (gain > 0.0)
                if not eligible.any():
                    stop_reason = "no_positive_marginal_gain"
                    break
            else:
                eligible = fits
            priority = np.where(eligible, priority_fn(gain, max_sim), -np.inf)
            best = int(np.argmax(priority))
            selected.append(best)
            remaining -= int(self.tokens[best])
            available[best] = False
            penalty += self.similarity[best]
            max_sim = np.maximum(max_sim, self.similarity[best])

        latency = (time.perf_counter() - start) * 1000.0
        return self._result(method, selected, latency,
                            stopping_policy=self._stopping_policy_name(),
                            stop_reason=stop_reason, **meta)

    def _stopping_policy_name(self) -> str:
        return ("positive_marginal_gain" if self.stop_on_nonpositive_gain
                else "fill_budget_v0")

    def _per_token(self, gain: np.ndarray) -> np.ndarray:
        """``gain / w`` computed only where ``w > 0`` (no divide-by-zero)."""
        out = np.full(self.n, -np.inf, dtype=np.float64)
        np.divide(gain, self.tokens, out=out, where=self.tokens > 0)
        return out

    # -- baselines ---------------------------------------------------------- #
    def solve_top_k(self, k: Optional[int] = None) -> SelectionResult:
        r"""Naive retrieval: rank by ``rel(q, d_i)``, take while the budget holds.

        .. math::

            i^* = \arg\max_{i \text{ eligible}} \; rel(q, d_i)

        Pure relevance ranking - no redundancy term, no token normalization;
        that is unchanged from v0 and is the point of the baseline. What changed
        is only that it now obeys the same stopping policy as every other
        heuristic (see :meth:`_select`). With ``k`` set, stops after ``k``
        documents.
        """
        return self._select(lambda gain, max_sim: self.relevance,
                            method="top_k", k=k, k_limit=k)

    def solve_mmr(self) -> SelectionResult:
        r"""Genuine Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

        .. math::

            i^* = \arg\max_{i \notin S} \;
                  \lambda_{mmr} \, rel(q, d_i)
                  - (1 - \lambda_{mmr}) \max_{j \in S} sim(d_i, d_j)

        The redundancy term is the **maximum** similarity to any already
        selected document, and ``\lambda_{mmr}`` is a convex weight.

        This differs from the research objective in both respects: the objective
        penalizes the **sum** over all selected pairs, weighted by
        ``\lambda_{obj}``, which is not a convex combination. v0 shipped
        summed-penalty greedy ascent under this name, which made the "MMR"
        baseline a greedy optimizer of the very objective being reported; the
        honest version of that solver is now
        :meth:`solve_greedy_objective`.

        For an empty ``S`` the max term is ``0``, so the first pick is the most
        relevant eligible document.
        """
        lam = self.mmr_lambda
        return self._select(
            lambda gain, max_sim: lam * self.relevance - (1.0 - lam) * max_sim,
            method="mmr", mmr_lambda=lam)

    def solve_greedy_objective(self) -> SelectionResult:
        r"""Greedy ascent on ``Score(S)`` itself; no token normalization.

        .. math::

            i^* = \arg\max_{i \notin S} \;
                  rel(q, d_i) - \lambda_{obj} \sum_{j \in S} sim(d_i, d_j)

        This is exactly the solver v0 labelled ``mmr``. It is the correct
        control for the proposed method: the two differ **only** by the
        ``1 / w_i`` factor, so the difference between them isolates token
        normalization.
        """
        return self._select(lambda gain, max_sim: gain,
                            method="greedy_objective")

    def solve_greedy(self) -> SelectionResult:
        r"""Token-aware greedy: maximize marginal gain **per token spent**.

        .. math::

            i^* = \arg\max_{i \notin S} \;
                  \frac{rel(q, d_i) - \lambda_{obj} \sum_{j \in S} sim(d_i, d_j)}{w_i}

        The standard density-greedy heuristic for the (quadratic) knapsack, and
        the method this study proposes. Unchanged from v0.
        """
        return self._select(lambda gain, max_sim: self._per_token(gain),
                            method="greedy_token_aware")

    # -- exact -------------------------------------------------------------- #
    def solve_ilp(self, time_limit: Optional[float] = 60.0,
                  msg: bool = False,
                  scope: SolveScope = SolveScope.FULL_CORPUS) -> SelectionResult:
        r"""Exact solver for the linearized QKP (PuLP / CBC).

        Binary ``x_i`` select documents; ``y_ij`` linearizes the product
        ``x_i x_j`` for ``i < j``:

        .. math::

            \max \sum_i r_i x_i - \lambda_{obj} \sum_{i<j} s_{ij} y_{ij}
            \quad s.t. \quad \sum_i w_i x_i \le W_{max},
            \; y_{ij} \ge x_i + x_j - 1, \; y_{ij} \ge 0

        Because every ``y_ij`` carries a non-positive objective coefficient, the
        solver always pushes it to its lower bound, so ``y`` can stay continuous
        and the ``y <= x`` constraints are redundant. Pairs with ``s_ij == 0``
        are skipped entirely, which keeps the model sparse. Note that the model
        therefore has ``N`` binaries and ``O(N^2)`` **continuous** variables.

        The returned ``status`` distinguishes a proven optimum from a
        time-limited incumbent (see :func:`classify_pulp_status`). Incumbents
        are returned intact - never discarded, never relabelled. ``scope``
        records whether the model covered the full corpus or a restricted pool.

        Raises ``RuntimeError`` if PuLP is unavailable.
        """
        start = time.perf_counter()
        if self.n == 0 or self.budget <= 0:
            # The empty selection is provably optimal when nothing can be bought.
            return self._result("ilp", [], (time.perf_counter() - start) * 1000.0,
                                status=SolveStatus.PROVEN_OPTIMAL, scope=scope,
                                reason="empty_or_zero_budget",
                                model_build_ms=0.0, solver_ms=0.0)
        try:
            import pulp
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise RuntimeError("PuLP is required for solve_ilp(); pip install pulp") from exc

        candidates = np.flatnonzero(self.feasible_mask).tolist()
        if not candidates:
            return self._result("ilp", [], (time.perf_counter() - start) * 1000.0,
                                status=SolveStatus.PROVEN_OPTIMAL, scope=scope,
                                reason="no_feasible_candidate",
                                model_build_ms=0.0, solver_ms=0.0)

        problem = pulp.LpProblem("context_knapsack", pulp.LpMaximize)
        x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in candidates}

        objective = [self.relevance[i] * x[i] for i in candidates]
        n_pairs = 0
        if self.objective_lambda > 0:
            for a, i in enumerate(candidates):
                for j in candidates[a + 1:]:
                    s = self.similarity[i, j]
                    if s <= 0.0:
                        continue
                    y = pulp.LpVariable(f"y_{i}_{j}", lowBound=0.0, upBound=1.0,
                                        cat="Continuous")
                    problem += y >= x[i] + x[j] - 1, f"link_{i}_{j}"
                    objective.append(-self.objective_lambda * s * y)
                    n_pairs += 1

        problem += pulp.lpSum(objective)
        problem += (pulp.lpSum(int(self.tokens[i]) * x[i] for i in candidates)
                    <= self.budget), "token_budget"

        build_ms = (time.perf_counter() - start) * 1000.0
        solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
        solve_start = time.perf_counter()
        problem.solve(solver)
        solver_ms = (time.perf_counter() - solve_start) * 1000.0

        status = classify_pulp_status(problem, pulp)
        if status is SolveStatus.INFEASIBLE:
            selected: List[int] = []
        else:
            # Keep the incumbent whatever its status: a feasible-but-unproven
            # solution is still a valid selection, it just is not a proof.
            selected = [i for i in candidates if x[i].value() is not None
                        and x[i].value() > 0.5]

        solver_objective = pulp.value(problem.objective)
        latency = (time.perf_counter() - start) * 1000.0
        result = self._result(
            "ilp", selected, latency, status=status, scope=scope,
            pulp_status=pulp.LpStatus[problem.status],
            pulp_sol_status=getattr(problem, "sol_status", None),
            solver_objective=(float(solver_objective)
                              if solver_objective is not None else None),
            n_pairs=n_pairs, n_candidates=len(candidates),
            time_limit_s=time_limit, model_build_ms=build_ms,
            solver_ms=solver_ms, solver_name="PULP_CBC_CMD",
        )
        # The solver's own objective value must agree with our recomputation of
        # Score(S) on the returned indices; a mismatch means the model and the
        # reported objective have drifted apart.
        if result.meta["solver_objective"] is not None and selected:
            result.meta["objective_consistent"] = bool(
                abs(result.meta["solver_objective"] - result.score) < 1e-6)
        return result

    # -- convenience -------------------------------------------------------- #
    def solve_all(self, include_ilp: bool = True,
                  ilp_time_limit: Optional[float] = 60.0) -> Dict[str, SelectionResult]:
        """Run every solver and return ``{method: SelectionResult}``."""
        results = {
            "top_k": self.solve_top_k(),
            "mmr": self.solve_mmr(),
            "greedy_objective": self.solve_greedy_objective(),
            "greedy_token_aware": self.solve_greedy(),
        }
        if include_ilp:
            try:
                results["ilp"] = self.solve_ilp(time_limit=ilp_time_limit)
            except RuntimeError:
                pass
        return results


def from_corpus(corpus, budget: int,
                objective_lambda: float = DEFAULT_OBJECTIVE_LAMBDA,
                mmr_lambda: float = DEFAULT_MMR_LAMBDA) -> ContextKnapsack:
    """Build a :class:`ContextKnapsack` from an :class:`embedder.EncodedCorpus`."""
    return ContextKnapsack(corpus.relevance, corpus.similarity, corpus.tokens,
                           budget=budget, objective_lambda=objective_lambda,
                           mmr_lambda=mmr_lambda)
