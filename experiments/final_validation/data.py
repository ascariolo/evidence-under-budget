"""HotpotQA loading for the LOCKED v7 protocol (sections 2 and 3).

Downloads the exact parquet named by the protocol once, records its SHA-256,
parses the distractor validation split into sentence-level candidate pools, and
applies *only* the two exclusions the protocol specifies. Excluded questions are
counted and never replaced by re-sampling.

Gold supporting facts are parsed here but are never handed to any solver: they
travel in a separate field of :class:`Question` and are read only when coverage
is scored, after selection.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import protocol as P

DATA_DIR = P.REPO_ROOT / "data" / "hotpotqa"
PARQUET_PATH = DATA_DIR / "validation-00000-of-00001.parquet"


@dataclass(frozen=True)
class Question:
    """One HotpotQA question with its sentence-level candidate pool.

    ``gold_indices`` are pool positions; they exist for scoring only. No field of
    this object is consulted by a solver except ``sentences`` (via embeddings)
    and ``tokens``.
    """

    qid: str
    question: str
    sentences: Tuple[str, ...]
    sentence_keys: Tuple[Tuple[str, int], ...]   # (paragraph title, sentence id)
    tokens: np.ndarray                            # (N,) int64, all > 0
    gold_indices: Tuple[int, ...]                 # positions in the pool
    n_gold_annotated: int                         # denominator of C(S)
    n_gold_unreachable: int                       # annotated but dropped (zero tokens)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.sentences)

    @property
    def w_pool(self) -> int:
        """W_pool: total token count of every candidate sentence (protocol 4).

        Computed from the candidate pool alone: independent of any arm, solver,
        gold label, or previously selected context.
        """
        return int(self.tokens.sum())


def download_once(url: str = P.DATASET_URL, dest: Path = PARQUET_PATH) -> Tuple[Path, str]:
    """Fetch the protocol's parquet if absent; return its path and SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=300) as response, open(tmp, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(dest)
    return dest, P.sha256_of(dest)


def _as_lists(value) -> List:
    """Parquet columns arrive as numpy arrays or lists depending on the reader."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def load_raw(path: Path = PARQUET_PATH) -> List[dict]:
    """Read the parquet into plain dicts, preserving row order."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


def build_questions(rows: Sequence[dict], token_counter) -> Tuple[List[Question], Dict[str, int]]:
    """Parse rows into pools, applying only the protocol's exclusions.

    Returns the kept questions in dataset order plus an exclusion ledger.
    """
    kept: List[Question] = []
    ledger = {
        "rows_total": len(rows),
        "excluded_missing_supporting_fact": 0,
        "excluded_empty_pool": 0,
        "sentences_dropped_zero_tokens": 0,
        "gold_unreachable_zero_tokens": 0,
        "questions_with_duplicate_titles": 0,
    }

    for row in rows:
        context = row["context"]
        titles = _as_lists(context["title"])
        paragraphs = [_as_lists(s) for s in _as_lists(context["sentences"])]

        # (title, sent_id) -> sentence text, first occurrence of a repeated title wins.
        catalogue: Dict[Tuple[str, int], str] = {}
        seen_titles = set()
        duplicate_title = False
        for title, sentences in zip(titles, paragraphs):
            if title in seen_titles:
                duplicate_title = True
            seen_titles.add(title)
            for sent_id, text in enumerate(sentences):
                catalogue.setdefault((title, sent_id), text)
        if duplicate_title:
            ledger["questions_with_duplicate_titles"] += 1

        facts = row["supporting_facts"]
        gold_keys = list(zip(_as_lists(facts["title"]), [int(i) for i in _as_lists(facts["sent_id"])]))

        # Exclusion 1 (protocol 2): annotation points at a sentence that is not there.
        if any(key not in catalogue for key in gold_keys) or not gold_keys:
            ledger["excluded_missing_supporting_fact"] += 1
            continue

        # Exclusion 2 (protocol 2): zero-token sentences leave the pool.
        keys = list(catalogue.keys())
        texts = [catalogue[k] for k in keys]
        counts = token_counter.count_batch(texts)
        keep = [i for i, c in enumerate(counts) if int(c) > 0]
        ledger["sentences_dropped_zero_tokens"] += len(keys) - len(keep)
        if not keep:
            ledger["excluded_empty_pool"] += 1
            continue

        pool_keys = [keys[i] for i in keep]
        pool_texts = [texts[i] for i in keep]
        pool_tokens = np.asarray([int(counts[i]) for i in keep], dtype=np.int64)
        position = {key: idx for idx, key in enumerate(pool_keys)}

        gold_unique = list(dict.fromkeys(gold_keys))          # annotation order, deduplicated
        gold_indices = tuple(position[k] for k in gold_unique if k in position)
        unreachable = len(gold_unique) - len(gold_indices)
        ledger["gold_unreachable_zero_tokens"] += unreachable

        kept.append(Question(
            qid=str(row["id"]),
            question=str(row["question"]),
            sentences=tuple(pool_texts),
            sentence_keys=tuple(pool_keys),
            tokens=pool_tokens,
            gold_indices=gold_indices,
            n_gold_annotated=len(gold_unique),
            n_gold_unreachable=unreachable,
            meta={"level": row.get("level"), "type": row.get("type")},
        ))

    ledger["questions_eligible"] = len(kept)
    return kept, ledger


def frozen_sample(questions: Sequence[Question], n: int = P.N_QUESTIONS,
                  seed: int = P.SAMPLE_SEED) -> List[Question]:
    """The protocol's sample: first ``n`` eligible questions of a frozen order.

    The permutation is drawn once from ``seed`` over the whole eligible set and
    never redrawn. Ineligible questions were removed *before* the draw, so no
    excluded question is ever replaced by re-sampling: the order is fixed and we
    simply stop after ``n``.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(questions))
    return [questions[int(i)] for i in order[:n]]


def load_sample(n: int = P.N_QUESTIONS) -> Tuple[List[Question], Dict[str, object]]:
    """Download (once), parse, exclude, and return the frozen sample plus a ledger."""
    from embedder import TokenCounter

    path, sha = download_once()
    rows = load_raw(path)
    counter = TokenCounter(P.TOKENIZER_ENCODING)
    questions, ledger = build_questions(rows, counter)
    sample = frozen_sample(questions, n=n)
    ledger = dict(ledger)
    ledger.update({
        "dataset_path": str(path),
        "dataset_sha256": sha,
        "dataset_url": P.DATASET_URL,
        "sample_seed": P.SAMPLE_SEED,
        "sample_size_requested": n,
        "sample_size_obtained": len(sample),
    })
    return sample, ledger


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    P.assert_locked()
    sample, ledger = load_sample()
    print(json.dumps(ledger, indent=2))
