"""BOS-safe, document-separated token collection for pinned public corpora."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

DATASET = "Skylion007/openwebtext"
DATASET_CONFIG = "plain_text"
DATASET_REVISION = "79d93d786212f7344586290adb811d4ae6a1762c"
CTX = 128
PREPEND_BOS = True
CACHE = Path("data/cache")

SPLITS: dict[str, tuple[int, int]] = {
    "dev": (0, 20_000),
    "test": (20_000, 40_000),
    "train": (40_000, 2_000_000),
}


@dataclass(frozen=True)
class Corpus:
    """One pinned upstream text corpus and its document-index split ranges."""

    key: str
    repository: str
    config: str
    revision: str
    splits: dict[str, tuple[int, int]]

    def range(self, split: str) -> tuple[int, int]:
        if split not in self.splits:
            raise ValueError(
                f"unknown {self.key} split {split!r}; expected one of {list(self.splits)}"
            )
        return self.splits[split]


OPENWEBTEXT = Corpus(
    key="openwebtext",
    repository=DATASET,
    config=DATASET_CONFIG,
    revision=DATASET_REVISION,
    splits=SPLITS,
)

# FineWeb matches the reference GLP activation-data source. Document ranges are
# disjoint by construction: the frozen validation corpus never reaches document
# 100,000, and every training artifact is a prefix of the same training stream,
# so a 4M artifact is a strict subset of a 32M artifact collected from it.
FINEWEB = Corpus(
    key="fineweb",
    repository="HuggingFaceFW/fineweb",
    config="sample-10BT",
    revision="9bb295ddab0e05d785b879661af7260fed5140fc",
    splits={
        "val": (0, 40_000),
        "train": (100_000, 20_000_000),
    },
)

CORPORA: dict[str, Corpus] = {corpus.key: corpus for corpus in (OPENWEBTEXT, FINEWEB)}


def corpus_for(key: str) -> Corpus:
    if key not in CORPORA:
        raise ValueError(f"unknown corpus {key!r}; expected one of {list(CORPORA)}")
    return CORPORA[key]


def tokenizer_identity(tokenizer) -> str:  # noqa: ANN001
    identity = getattr(tokenizer, "name_or_path", None)
    if not identity:
        raise ValueError("tokenizer must expose a nonempty name_or_path for provenance")
    return str(identity)


def token_cache_path(  # noqa: ANN001
    split: str,
    n_seqs: int,
    ctx: int,
    tokenizer,
    cache_dir: Path | None = None,
    corpus: Corpus = OPENWEBTEXT,
) -> Path:
    directory = CACHE if cache_dir is None else cache_dir
    identity = {
        "dataset": corpus.repository,
        "dataset_config": corpus.config,
        "dataset_revision": corpus.revision,
        "tokenizer": tokenizer_identity(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "prepend_bos": PREPEND_BOS,
        "split": split,
        "n_seqs": n_seqs,
        "ctx": ctx,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return directory / f"{split}_{n_seqs}x{ctx}_{fingerprint}.npy"


def _validate_token_array(
    array: np.ndarray, expected_shape: tuple[int, int], bos: int
) -> None:
    if array.shape != expected_shape:
        raise ValueError(f"cached token shape {array.shape} != {expected_shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"cached token dtype must be integer, got {array.dtype}")
    bos_counts = np.count_nonzero(array == bos, axis=1)
    if PREPEND_BOS and (not np.all(array[:, 0] == bos) or not np.all(bos_counts == 1)):
        raise ValueError("each sequence must contain exactly one BOS at position zero")
    if not PREPEND_BOS and np.any(bos_counts):
        raise ValueError("BOS appeared although PREPEND_BOS is disabled")


def token_manifest_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".manifest.json")


def load_tokens(  # noqa: ANN001
    split: str,
    n_seqs: int,
    tokenizer,
    ctx: int = CTX,
    cache_dir: Path | None = None,
    corpus: Corpus = OPENWEBTEXT,
) -> torch.Tensor:
    """Return ``[n_seqs, ctx]`` tokens with one explicit BOS per document."""

    lower, upper = corpus.range(split)
    if n_seqs < 1 or ctx < 2:
        raise ValueError("n_seqs must be positive and ctx must be at least two")
    directory = CACHE if cache_dir is None else cache_dir
    path = token_cache_path(split, n_seqs, ctx, tokenizer, directory, corpus)
    if path.exists():
        cached = np.load(path)
        _validate_token_array(cached, (n_seqs, ctx), tokenizer.bos_token_id)
        return torch.from_numpy(cached).long()

    bos = tokenizer.bos_token_id
    needed_text_tokens = ctx - 1 if PREPEND_BOS else ctx
    rows: list[list[int]] = []
    documents_consumed = 0
    documents_skipped_short = 0
    raw_text_tokens = 0
    stream = load_dataset(
        corpus.repository,
        corpus.config,
        split="train",
        streaming=True,
        revision=corpus.revision,
    )
    for index, document in enumerate(stream):
        if index < lower:
            continue
        if index >= upper:
            break
        documents_consumed += 1
        token_ids = tokenizer.encode(document["text"], add_special_tokens=False)
        raw_text_tokens += len(token_ids)
        if len(token_ids) < needed_text_tokens:
            documents_skipped_short += 1
            continue
        row = ([bos] if PREPEND_BOS else []) + token_ids[:needed_text_tokens]
        if PREPEND_BOS and (row[0] != bos or row.count(bos) != 1):
            raise ValueError(
                "each sequence must contain exactly one BOS; tokenizer encoding must "
                "use add_special_tokens=False"
            )
        if not PREPEND_BOS and bos in row:
            raise ValueError("BOS appeared although PREPEND_BOS is disabled")
        rows.append(row)
        if len(rows) == n_seqs:
            break

    if len(rows) != n_seqs:
        raise RuntimeError(
            f"split {split!r} yielded {len(rows)}/{n_seqs} full documents in "
            f"range {lower}:{upper}"
        )
    array = np.asarray(rows, dtype=np.int32)
    _validate_token_array(array, (n_seqs, ctx), bos)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    token_manifest_path(path).write_text(
        json.dumps(
            {
                "corpus": corpus.key,
                "repository": corpus.repository,
                "repository_config": corpus.config,
                "repository_revision": corpus.revision,
                "split": split,
                "document_range": [lower, upper],
                "documents_consumed": documents_consumed,
                "documents_skipped_short": documents_skipped_short,
                "documents_kept": len(rows),
                "raw_text_tokens_seen": raw_text_tokens,
                "tokenizer": tokenizer_identity(tokenizer),
                "bos_token_id": int(bos),
                "prepend_bos": PREPEND_BOS,
                "ctx": ctx,
                "text_tokens_per_sequence": needed_text_tokens,
                "n_seqs": n_seqs,
            },
            indent=2,
        )
        + "\n"
    )
    return torch.from_numpy(array).long()
