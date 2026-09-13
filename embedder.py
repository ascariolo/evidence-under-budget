"""Embedding generation, token counting and pairwise similarity.

This module is the only place that talks to embedding models and tokenizers.
Optimization logic lives in ``optimizer.py``; evaluation in ``benchmark.py``.

All backends are local and deterministic (Zero Paid API Policy):

* ``sentence-transformers`` (``all-MiniLM-L6-v2``) is the reference encoder.
* ``HashingBackend`` is a dependency-free fallback (random-projection bag of
  words) used when torch / sentence-transformers are unavailable. It is seeded,
  so results are reproducible, but semantically weaker.
* ``tiktoken`` (``cl100k_base``) counts tokens; a whitespace/punctuation
  heuristic is used when it is not installed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_ENCODING = "cl100k_base"
RANDOM_SEED = 42

_WORD_RE = re.compile(r"\w+|[^\w\s]")


# --------------------------------------------------------------------------- #
# Token counting
# --------------------------------------------------------------------------- #
class TokenCounter:
    """Counts tokens with ``tiktoken``'s ``cl100k_base`` encoding.

    Falls back to a heuristic (``~1.3`` tokens per word-like unit) when
    ``tiktoken`` is not installed, so the framework still runs offline.
    """

    def __init__(self, encoding_name: str = DEFAULT_ENCODING) -> None:
        self.encoding_name = encoding_name
        self._encoding = None
        self.exact = False
        try:  # pragma: no cover - depends on local install
            import tiktoken

            self._encoding = tiktoken.get_encoding(encoding_name)
            self.exact = True
        except Exception:
            self._encoding = None

    def count(self, text: str) -> int:
        """Return the token count of ``text`` (always ``>= 1`` for non-empty)."""
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return max(1, int(round(len(_WORD_RE.findall(text)) * 1.3)))

    def count_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Vectorized-friendly wrapper returning an ``int`` array of shape (N,)."""
        return np.asarray([self.count(t) for t in texts], dtype=np.int64)


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    """Convenience one-shot token count (creates a throwaway counter)."""
    return TokenCounter(encoding_name).count(text)


# --------------------------------------------------------------------------- #
# Embedding backends
# --------------------------------------------------------------------------- #
class HashingBackend:
    """Deterministic offline encoder: hashed bag-of-words + random projection.

    Each token is hashed into ``n_buckets`` and the resulting sparse count
    vector is projected into ``dim`` dimensions with a fixed Gaussian matrix
    (seed ``42``). Purely NumPy, no model download, no GPU.
    """

    name = "hashing-fallback"

    def __init__(self, dim: int = 384, n_buckets: int = 2 ** 14,
                 seed: int = RANDOM_SEED) -> None:
        self.dim = dim
        self.n_buckets = n_buckets
        rng = np.random.default_rng(seed)
        self._projection = rng.normal(
            scale=1.0 / np.sqrt(dim), size=(n_buckets, dim)
        ).astype(np.float32)

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.n_buckets

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        counts = np.zeros((len(texts), self.n_buckets), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _WORD_RE.findall(text.lower()):
                counts[i, self._bucket(token)] += 1.0
        # Sub-linear term weighting damps long-document magnitude bias.
        np.log1p(counts, out=counts)
        return counts @ self._projection


class SentenceTransformerBackend:
    """Wraps ``sentence-transformers`` (CPU inference, batched)."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME,
                 device: str = "cpu", batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer  # local import

        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )


class Embedder:
    """Encodes text into L2-normalized vectors and counts their tokens.

    Parameters
    ----------
    model_name:
        Sentence-transformers model id. Ignored when ``backend`` is given.
    backend:
        Pre-built backend instance (used by tests / offline runs).
    allow_fallback:
        When ``True`` (default) a failure to load sentence-transformers falls
        back to :class:`HashingBackend` instead of raising.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME,
                 backend: Optional[object] = None,
                 allow_fallback: bool = True,
                 encoding_name: str = DEFAULT_ENCODING) -> None:
        self.token_counter = TokenCounter(encoding_name)
        if backend is not None:
            self.backend = backend
        else:
            try:
                self.backend = SentenceTransformerBackend(model_name)
            except Exception:
                if not allow_fallback:
                    raise
                self.backend = HashingBackend()
        self.backend_name = getattr(self.backend, "name", type(self.backend).__name__)

    # -- encoding ----------------------------------------------------------- #
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized embeddings of shape ``(len(texts), dim)``."""
        if len(texts) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = np.asarray(self.backend.encode(list(texts)), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return l2_normalize(vectors)

    def count_tokens(self, texts: Sequence[str]) -> np.ndarray:
        """Token count per document, shape ``(N,)``, dtype ``int64``."""
        return self.token_counter.count_batch(texts)


# --------------------------------------------------------------------------- #
# Vectorized similarity helpers
# --------------------------------------------------------------------------- #
def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization; zero rows stay zero (no NaNs)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def cosine_similarity_matrix(a: np.ndarray, b: Optional[np.ndarray] = None,
                             assume_normalized: bool = False) -> np.ndarray:
    """Pairwise cosine similarity as a single dense matrix product.

    ``S = A_hat @ B_hat.T`` where ``X_hat`` is row-normalized. Never loops over
    individual vectors. Returns shape ``(n_a, n_b)`` (or ``(n_a, n_a)``).
    """
    a = np.asarray(a, dtype=np.float32)
    if a.size == 0:
        return np.zeros((a.shape[0], a.shape[0] if b is None else np.asarray(b).shape[0]),
                        dtype=np.float32)
    if not assume_normalized:
        a = l2_normalize(a)
    if b is None:
        return np.clip(a @ a.T, -1.0, 1.0)
    b = np.asarray(b, dtype=np.float32)
    if not assume_normalized:
        b = l2_normalize(b)
    return np.clip(a @ b.T, -1.0, 1.0)


# --------------------------------------------------------------------------- #
# Corpus container
# --------------------------------------------------------------------------- #
@dataclass
class EncodedCorpus:
    """Pre-computed artifacts shared by every solver in a single query run.

    Attributes
    ----------
    documents: raw texts, length ``N``.
    embeddings: ``(N, dim)`` L2-normalized document vectors.
    tokens: ``(N,)`` token cost of each document.
    relevance: ``(N,)`` cosine similarity ``rel(q, d_i)``.
    similarity: ``(N, N)`` cached pairwise ``sim(d_i, d_j)``, diagonal zeroed.
    """

    documents: List[str]
    embeddings: np.ndarray
    tokens: np.ndarray
    relevance: np.ndarray
    similarity: np.ndarray
    query: str = ""
    backend_name: str = ""

    def __len__(self) -> int:
        return len(self.documents)


def build_corpus(query: str, documents: Sequence[str],
                 embedder: Optional[Embedder] = None,
                 clip_negative_similarity: bool = True) -> EncodedCorpus:
    """Encode ``documents`` once and cache every matrix the solvers need.

    The ``N x N`` similarity matrix is computed up-front (single BLAS call) so
    the optimization loop never touches raw embeddings. Handles empty and
    single-document inputs. Negative cosine values are optionally clipped to
    ``0`` so the redundancy penalty can never *reward* a selection.
    """
    embedder = embedder or Embedder()
    documents = list(documents)
    n = len(documents)

    if n == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return EncodedCorpus([], np.zeros((0, 0), dtype=np.float32),
                             np.zeros((0,), dtype=np.int64), empty,
                             np.zeros((0, 0), dtype=np.float32), query,
                             embedder.backend_name)

    doc_vectors = embedder.encode(documents)
    query_vector = embedder.encode([query]) if query else np.zeros((1, doc_vectors.shape[1]),
                                                                   dtype=np.float32)
    relevance = cosine_similarity_matrix(query_vector, doc_vectors,
                                         assume_normalized=True)[0]
    similarity = cosine_similarity_matrix(doc_vectors, assume_normalized=True)
    np.fill_diagonal(similarity, 0.0)
    if clip_negative_similarity:
        np.clip(similarity, 0.0, 1.0, out=similarity)

    return EncodedCorpus(
        documents=documents,
        embeddings=doc_vectors,
        tokens=embedder.count_tokens(documents),
        relevance=relevance.astype(np.float32),
        similarity=similarity,
        query=query,
        backend_name=embedder.backend_name,
    )
