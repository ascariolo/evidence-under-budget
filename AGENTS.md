
# Agentic Context-Knapsack Optimizer

## Project Overview

An open-source Python framework that optimizes LLM context window packing for Agentic Search systems. Instead of naive Top-K document retrieval, this system treats context selection as a constrained optimization problem (Maximum Coverage / Quadratic Knapsack Problem). It maximizes semantic relevance and information diversity while strictly enforcing a token budget constraint ($W_{max}$).

## Core Scientific Problem

Standard RAG and Search pipelines suffer from three major inefficiencies:

1. **Redundancy:** Highly-ranked documents often duplicate information, wasting precious context space.
2. **Token Inefficiency:** Long documents with low information density crowd out concise, high-value snippets.
3. **Lost-in-the-Middle:** Bloated contexts increase LLM latency, cost, and hallucination rates.

This project implements a **Token-Aware Maximal Marginal Relevance (MMR)** greedy heuristic and compares it against an exact **Integer Linear Programming (ILP)** ground-truth solver and naive Top-K retrieval.

## Tech Stack & Dependencies

- **Language:** Python 3.10+
- **Embeddings & NLP:** `sentence-transformers` (`all-MiniLM-L6-v2`), `tiktoken` (`cl100k_base` tokenizer)
- **Math & Optimization:** `numpy`, `scipy`, `pulp` (ILP Solver)
- **Visualization:** `matplotlib`
- **Testing & Benchmarks:** `pytest` (optional), custom benchmark scripts

## Project Structure

- `embedder.py` — Embedding generation, cosine similarity matrix calculation, token counting.
- `optimizer.py` — Core optimization algorithms (`ContextKnapsack` class with Greedy Token-Aware MMR and ILP solvers).
- `benchmark.py` — Comparative evaluation engine and plot generator (Naive Top-K vs. MMR vs. Token-Aware Knapsack).
- `requirements.txt` — Python dependencies.
- `AGENTS.md` — Project context and development guidelines.

## Development & Execution Commands

- **Environment Setup:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```
- **Run Benchmark:**
  ```bash
  python benchmark.py
  ```

## Code Guidelines & Standards

### 1. Architectural & Research Rigor

- **Clean Modular Design:** Maintain strict separation of concerns between data processing (`embedder.py`), optimization solvers (`optimizer.py`), and evaluation/plotting (`benchmark.py`).
- **Mathematical Transparency:** Document formulas directly in function docstrings using standard notation (e.g., $Score(S) = \sum rel(q, d_i) - \lambda \sum sim(d_i, d_j)$).
- **Type Safety:** Use explicit Python type hints (`typing.List`, `typing.Tuple`, `numpy.ndarray`, etc.) for all function signatures.

### 2. Performance & Memory Optimization

- **Vectorized Matrix Operations:** Never loop over individual embedding vectors in Python. Use NumPy dot products and matrix operations for pairwise similarity computation.
- **CPU Compute Efficiency:** Design all algorithms to run quickly on standard CPU. Avoid GPU dependencies or heavy PyTorch overhead during runtime inference.
- **Sparse/Dense Matrix Handling:** Pre-compute and cache pairwise document similarity matrices $N \times N$ before running optimization iterations.

### 3. Reproducibility & Edge-Case Handling

- **Fixed Seeds:** Set deterministic random seeds (`numpy.random.seed(42)`) across all synthetic data generators and benchmark runs.
- **Boundary Conditions:** Gracefully handle edge cases such as:
  - Document lists where individual document token counts exceed $W_{max}$.
  - Empty or single-document inputs.
  - Documents with zero or negative similarity scores.

### 4. Zero Paid API Policy

- All embeddings, token counts, and optimization metrics must be computed deterministically using open local models (`sentence-transformers`, `tiktoken`, `PuLP`).
- Do not introduce dependencies on external paid API endpoints (e.g., OpenAI, Anthropic) in the core framework or benchmarking suite.

### 5. Benchmark Output Standards

- Measure and report three key metrics for all methods:
  - **Latency (ms):** Wall-clock execution time per query.
  - **Context Density Score:** Net relevance per 1,000 tokens spent.
  - **Token Savings (%):** Percentage of budget preserved while maintaining core information coverage.
