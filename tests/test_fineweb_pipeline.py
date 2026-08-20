"""Tests for the FineWeb activation-data boundary used by the cheap-scaling experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from interp.activations import make_split
from interp.data import (
    CORPORA,
    FINEWEB,
    OPENWEBTEXT,
    corpus_for,
    load_tokens,
    token_cache_path,
    token_manifest_path,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class FakeTokenizer:
    bos_token_id = 99
    name_or_path = "fake-gpt2"

    def __init__(self, rows: dict[str, list[int]]) -> None:
        self.rows = rows

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return self.rows[text]


def _documents(count: int, length: int = 8) -> list[dict[str, str]]:
    return [{"text": f"doc{index}"} for index in range(count)]


def _fake_stream(monkeypatch: pytest.MonkeyPatch, documents: list[dict[str, str]]) -> list[dict]:
    import interp.data as data

    calls: list[dict] = []

    def fake_load_dataset(repository, config, **kwargs):  # noqa: ANN001
        calls.append({"repository": repository, "config": config, **kwargs})
        return documents

    monkeypatch.setattr(data, "load_dataset", fake_load_dataset)
    return calls


def test_fineweb_corpus_is_pinned_and_document_ranges_are_disjoint() -> None:
    assert FINEWEB.repository == "HuggingFaceFW/fineweb"
    assert FINEWEB.config == "sample-10BT"
    assert len(FINEWEB.revision) == 40 and int(FINEWEB.revision, 16) >= 0
    val_lower, val_upper = FINEWEB.range("val")
    train_lower, train_upper = FINEWEB.range("train")
    assert val_upper <= train_lower
    assert val_lower < val_upper and train_lower < train_upper
    assert set(CORPORA) == {"openwebtext", "fineweb"}
    assert corpus_for("fineweb") is FINEWEB
    with pytest.raises(ValueError, match="unknown corpus"):
        corpus_for("fineweb-edu")


def test_openwebtext_token_cache_identity_did_not_change_when_fineweb_was_added() -> None:
    tokenizer = FakeTokenizer({})
    root = Path("/tmp/does-not-need-to-exist")

    default_path = token_cache_path("train", 31_497, 128, tokenizer, root)
    explicit_path = token_cache_path("train", 31_497, 128, tokenizer, root, OPENWEBTEXT)
    fineweb_path = token_cache_path("train", 31_497, 128, tokenizer, root, FINEWEB)

    assert default_path == explicit_path
    assert fineweb_path != default_path


def test_fineweb_tokens_stream_the_pinned_revision_and_record_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents = [{"text": "long-a"}, {"text": "tiny"}, {"text": "long-b"}]
    tokenizer = FakeTokenizer(
        {"long-a": [1, 2, 3, 4], "tiny": [7], "long-b": [5, 6, 7, 8]}
    )
    calls = _fake_stream(monkeypatch, documents)
    corpus = FINEWEB.__class__(
        key="fineweb-test",
        repository=FINEWEB.repository,
        config=FINEWEB.config,
        revision=FINEWEB.revision,
        splits={"train": (0, 3)},
    )

    tokens = load_tokens("train", 2, tokenizer, ctx=4, cache_dir=tmp_path, corpus=corpus)

    assert tokens.tolist() == [[99, 1, 2, 3], [99, 5, 6, 7]]
    assert calls == [
        {
            "repository": FINEWEB.repository,
            "config": FINEWEB.config,
            "split": "train",
            "streaming": True,
            "revision": FINEWEB.revision,
        }
    ]
    cache_path = token_cache_path("train", 2, 4, tokenizer, tmp_path, corpus)
    manifest = json.loads(token_manifest_path(cache_path).read_text())
    assert manifest["repository_revision"] == FINEWEB.revision
    assert manifest["documents_consumed"] == 3
    assert manifest["documents_skipped_short"] == 1
    assert manifest["documents_kept"] == 2
    assert manifest["bos_token_id"] == 99
    assert manifest["prepend_bos"] is True
    assert manifest["text_tokens_per_sequence"] == 3


def test_fineweb_training_artifacts_are_nested_prefixes_of_one_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = FakeTokenizer({f"doc{index}": [index, index + 1, index + 2] for index in range(10)})
    _fake_stream(monkeypatch, _documents(10))
    corpus = FINEWEB.__class__(
        key="fineweb-test",
        repository=FINEWEB.repository,
        config=FINEWEB.config,
        revision=FINEWEB.revision,
        splits={"train": (0, 10)},
    )

    small = load_tokens("train", 3, tokenizer, ctx=4, cache_dir=tmp_path, corpus=corpus)
    large = load_tokens("train", 6, tokenizer, ctx=4, cache_dir=tmp_path, corpus=corpus)

    assert large[: len(small)].tolist() == small.tolist()
    assert len({tuple(row) for row in large.tolist()}) == 6


def test_fineweb_validation_documents_never_enter_the_training_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = FakeTokenizer({f"doc{index}": [index, index + 1, index + 2] for index in range(12)})
    _fake_stream(monkeypatch, _documents(12))
    corpus = FINEWEB.__class__(
        key="fineweb-test",
        repository=FINEWEB.repository,
        config=FINEWEB.config,
        revision=FINEWEB.revision,
        splits={"val": (0, 4), "train": (8, 12)},
    )

    validation = load_tokens("val", 3, tokenizer, ctx=4, cache_dir=tmp_path, corpus=corpus)
    training = load_tokens("train", 3, tokenizer, ctx=4, cache_dir=tmp_path, corpus=corpus)

    assert not {tuple(row) for row in validation.tolist()} & {
        tuple(row) for row in training.tolist()
    }


def test_fineweb_manifest_config_matches_the_pinned_corpus_and_its_own_arithmetic() -> None:
    manifest = yaml.safe_load((CONFIGS / "fineweb_activations_v1.yaml").read_text())
    corpus = manifest["corpus"]

    assert corpus["repository"] == FINEWEB.repository
    assert corpus["repository_config"] == FINEWEB.config
    assert corpus["repository_revision"] == FINEWEB.revision
    for split, values in manifest["splits"].items():
        assert list(FINEWEB.range(split)) == values["document_range"]

    total = 0
    for artifact in manifest["artifacts"]:
        per_seq = manifest["tokenization"]["activations_per_sequence"]
        assert artifact["n_seqs"] * per_seq == artifact["n_activations"]
        assert artifact["shape"] == [artifact["n_activations"], 768]
        assert artifact["bytes_float16"] == artifact["n_activations"] * 768 * 2
        split = make_split(
            artifact["n_activations"], per_seq, artifact["val_fraction"], artifact["split_seed"]
        )
        assert split.fingerprint() == artifact["split_fingerprint"]
        total += artifact["bytes_float16"]
    assert manifest["disk"]["total_bytes_float16"] == total
    assert manifest["storage"]["storage_dtype"] == "float16"
    assert set(manifest["protected_data"].values()) == {"forbidden"}


def test_frozen_validation_artifact_has_enough_internal_validation_sequences() -> None:
    manifest = yaml.safe_load((CONFIGS / "fineweb_activations_v1.yaml").read_text())
    validation = next(
        artifact for artifact in manifest["artifacts"] if artifact["split"] == "val"
    )
    split = make_split(
        validation["n_activations"], 127, validation["val_fraction"], validation["split_seed"]
    )
    sequences = np.unique(split.val // 127)

    assert len(sequences) == validation["internal_validation_sequences"]
    assert len(sequences) >= 256
