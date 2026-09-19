"""Frozen constants of the LOCKED v7 final-validation protocol.

Single source of truth for every methodological decision of the final empirical
phase. The authority is ``docs/final_validation_protocol.md``; this module holds
a machine-readable copy of its constants plus a hash guard that refuses to run
if that document has changed.

Nothing here may be edited: the protocol is locked (see
``results/final_validation/protocol_lock.json``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DOC = REPO_ROOT / "docs" / "final_validation_protocol.md"
LOCK_RECORD = REPO_ROOT / "results" / "final_validation" / "protocol_lock.json"

#: SHA-256 of the locked protocol document, as recorded at lock time.
PROTOCOL_SHA256 = "ff807b7d389e2e0c8397acb4a2db90b3b790da155ec7784f37129dfc91d0daa5"
PROTOCOL_VERSION = "7.0-final-validation"
LOCK_COMMIT = "b392aabe60b31cd26703f19598ff3d11d9f75a7d"

# -- data ------------------------------------------------------------------- #
DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_SPLIT = "distractor/validation"
DATASET_URL = ("https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
               "distractor/validation-00000-of-00001.parquet")
N_QUESTIONS = 500
SAMPLE_SEED = 7000

# -- pool and objective ----------------------------------------------------- #
CANDIDATE_UNIT = "sentence"
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
TOKENIZER_ENCODING = "cl100k_base"
LAMBDA_OBJ = 0.1
LAMBDA_MMR = 0.5

# -- budgets ---------------------------------------------------------------- #
RHO_LEVELS: Tuple[float, ...] = (0.10, 0.25, 0.50)

# -- arms ------------------------------------------------------------------- #
ARM_TOP_K = "top_k"
ARM_GO = "greedy_objective"
ARM_TA = "greedy_token_aware"
ARM_MMR = "mmr"
ARM_TA_SINGLETON = "token_aware_best_singleton"
ARM_ILP = "ilp"
ARMS: Tuple[str, ...] = (ARM_TOP_K, ARM_GO, ARM_TA, ARM_MMR, ARM_TA_SINGLETON, ARM_ILP)

#: The only confirmatory comparison. Every other arm is descriptive.
CONFIRMATORY_COMPARISON = (ARM_TA, ARM_GO)
CONFIRMATORY_FAMILY_SIZE = 3

ILP_RHO = 0.25
ILP_N_QUESTIONS = 100
ILP_TIME_LIMIT_S = 60.0
ILP_THREADS = 1

# -- statistics ------------------------------------------------------------- #
BOOTSTRAP_SEED = 7001
BOOTSTRAP_RESAMPLES = 10000
PERMUTATION_SEED = 7002
PERMUTATION_FLIPS = 10000
ALPHA = 0.05
TEST_SIDEDNESS = "two-sided"
MULTIPLICITY = "holm"
SESOI = None  # deliberately unspecified; see protocol section 7

# -- dry run ---------------------------------------------------------------- #
DRY_RUN_N = 20


class ProtocolViolation(RuntimeError):
    """Raised when the locked protocol is missing, altered, or contradicted."""


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_hash() -> str:
    """SHA-256 of the protocol document as it exists on disk right now."""
    if not PROTOCOL_DOC.exists():
        raise ProtocolViolation(f"locked protocol document is missing: {PROTOCOL_DOC}")
    return sha256_of(PROTOCOL_DOC)


def assert_locked() -> str:
    """Refuse to proceed unless the locked protocol is byte-identical.

    Returns the verified SHA-256 so callers can record it in their artifacts.
    """
    observed = protocol_hash()
    if observed != PROTOCOL_SHA256:
        raise ProtocolViolation(
            "protocol document does not match the lock record.\n"
            f"  expected {PROTOCOL_SHA256}\n"
            f"  observed {observed}\n"
            "The protocol is LOCKED: restore the document, do not edit it."
        )
    if LOCK_RECORD.exists():
        record = json.loads(LOCK_RECORD.read_text())
        if record.get("status") != "LOCKED":
            raise ProtocolViolation(f"lock record status is {record.get('status')!r}, expected 'LOCKED'")
        if record.get("document_sha256") != PROTOCOL_SHA256:
            raise ProtocolViolation("lock record hash disagrees with this module")
    return observed


def frozen_constants() -> Dict[str, object]:
    """Constants to embed verbatim in every result artifact."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "lock_commit": LOCK_COMMIT,
        "dataset": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "n_questions": N_QUESTIONS,
        "sample_seed": SAMPLE_SEED,
        "rho_levels": list(RHO_LEVELS),
        "lambda_obj": LAMBDA_OBJ,
        "lambda_mmr": LAMBDA_MMR,
        "embedder": EMBEDDER_MODEL,
        "tokenizer": TOKENIZER_ENCODING,
        "candidate_unit": CANDIDATE_UNIT,
        "arms": list(ARMS),
        "confirmatory_comparison": list(CONFIRMATORY_COMPARISON),
        "confirmatory_family_size": CONFIRMATORY_FAMILY_SIZE,
        "ilp_scope": {"rho": ILP_RHO, "n_questions": ILP_N_QUESTIONS,
                      "time_limit_s": ILP_TIME_LIMIT_S, "threads": ILP_THREADS},
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "permutation_seed": PERMUTATION_SEED,
        "permutation_flips": PERMUTATION_FLIPS,
        "alpha": ALPHA,
        "sidedness": TEST_SIDEDNESS,
        "multiplicity": MULTIPLICITY,
        "sesoi": SESOI,
    }
