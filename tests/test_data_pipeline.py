"""Tests for the reusable BOS-safe activation-data boundary."""

from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from interp.activations import (
    ActivationDataset,
    artifact_hashes,
    load_activations,
    load_validation_report,
    make_split,
    split_stats,
    validate_activation_metadata,
)
from interp.data import DATASET_REVISION, OPENWEBTEXT, load_tokens, token_cache_path
from interp.model import (
    MODEL_NAME,
    MODEL_RESOLVED_NAME,
    MODEL_REVISION,
    STEERING_HOOK,
    load_model,
)
from interp.provenance import source_revision


def owt_with_splits(splits: dict[str, tuple[int, int]]):
    """OpenWebText with narrowed document ranges; cache identity is unchanged."""

    return replace(OPENWEBTEXT, splits=splits)


class FakeTokenizer:
    bos_token_id = 99
    name_or_path = "fake-gpt2"

    def __init__(self, rows: dict[str, list[int]]) -> None:
        self.rows = rows
        self.add_special_tokens: list[bool] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.add_special_tokens.append(add_special_tokens)
        return self.rows[text]


def test_canonical_model_and_hook_are_frozen() -> None:
    assert MODEL_NAME == "gpt2-small"
    assert MODEL_RESOLVED_NAME == "gpt2"
    assert MODEL_REVISION == "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def test_model_and_tokenizer_are_both_loaded_at_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import interp.model as model_module

    calls: dict[str, tuple] = {}

    class DummyTokenizer:
        pad_token = None
        eos_token = "<eos>"
        padding_side = "right"

    class DummyModel:
        tokenizer = DummyTokenizer()
        cfg = type("Cfg", (), {"n_layers": 12, "model_name": MODEL_RESOLVED_NAME})()
        hook_dict = {STEERING_HOOK: object()}

        def eval(self):
            calls["eval"] = ()

        def requires_grad_(self, value: bool):
            calls["requires_grad"] = (value,)

    def tokenizer_loader(name: str, *, revision: str):
        calls["tokenizer"] = (name, revision)
        return DummyTokenizer()

    def model_loader(name: str, **kwargs):
        calls["model"] = (name, kwargs)
        return DummyModel()

    monkeypatch.setattr(model_module.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(model_module.HookedTransformer, "from_pretrained", model_loader)

    load_model("cpu")

    assert calls["tokenizer"] == (MODEL_RESOLVED_NAME, MODEL_REVISION)
    assert calls["model"][1]["revision"] == MODEL_REVISION
    assert calls["model"][1]["tokenizer"].__class__ is DummyTokenizer
    assert STEERING_HOOK == "blocks.7.hook_resid_pre"


def test_worker_compatibility_dependencies_are_frozen() -> None:
    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]

    assert "torch==2.6.0" in dependencies
    assert "datasets==3.1.0" in dependencies


def test_token_loading_adds_exactly_one_bos_and_never_asks_tokenizer_for_specials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import interp.data as data

    documents = [{"text": "first"}, {"text": "short"}, {"text": "second"}]
    tokenizer = FakeTokenizer(
        {"first": [1, 2, 3, 4], "short": [7], "second": [5, 6, 7, 8]}
    )
    monkeypatch.setattr(data, "CACHE", tmp_path)
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: documents)

    got = load_tokens("train", 2, tokenizer, ctx=4, corpus=owt_with_splits({"train": (0, 3)}))

    assert got.tolist() == [[99, 1, 2, 3], [99, 5, 6, 7]]
    assert tokenizer.add_special_tokens == [False, False, False]


def test_token_loading_pins_dataset_revision_and_cache_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import interp.data as data

    calls: list[tuple[tuple, dict]] = []
    tokenizer = FakeTokenizer({"first": [1, 2, 3]})

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"text": "first"}]

    monkeypatch.setattr(data, "load_dataset", fake_load_dataset)
    load_tokens(
        "train", 1, tokenizer, ctx=4, cache_dir=tmp_path, corpus=owt_with_splits({"train": (0, 1)})
    )

    assert calls == [
        (
            (data.DATASET, data.DATASET_CONFIG),
            {"split": "train", "streaming": True, "revision": DATASET_REVISION},
        )
    ]
    assert token_cache_path("train", 1, 4, tokenizer, tmp_path).is_file()


def test_token_loading_fails_if_tokenizer_output_contains_a_second_bos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import interp.data as data

    tokenizer = FakeTokenizer({"bad": [99, 1, 2, 3]})
    monkeypatch.setattr(data, "CACHE", tmp_path)
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: [{"text": "bad"}])

    with pytest.raises(ValueError, match="exactly one BOS"):
        load_tokens("train", 1, tokenizer, ctx=4, corpus=owt_with_splits({"train": (0, 1)}))


def test_cached_tokens_are_rechecked_for_exactly_one_bos(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer({})
    np.save(
        token_cache_path("train", 1, 4, tokenizer, tmp_path),
        np.array([[99, 99, 1, 2]], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="exactly one BOS"):
        load_tokens("train", 1, tokenizer, ctx=4, cache_dir=tmp_path)


def test_canonical_split_is_document_disjoint_and_has_frozen_fingerprint() -> None:
    split = make_split(4_000_119, per_seq=127, val_fraction=0.05, seed=20_260_807)

    assert split.fingerprint() == "c34aec678a328131"
    assert len(split.train) == 3_800_094
    assert len(split.val) == 200_025
    assert not set((split.train // 127).tolist()) & set((split.val // 127).tolist())


def test_split_statistics_match_hand_calculated_population_values() -> None:
    array = np.array([[1.0, 4.0], [3.0, 8.0], [5.0, 12.0]], dtype=np.float16)
    ds = ActivationDataset(
        array=array,
        meta={"shape": [3, 2]},
        mean=np.zeros(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
    )

    mean, std = split_stats(ds, np.array([0, 2], dtype=np.int64), chunk=1)

    assert np.array_equal(mean, np.array([3.0, 8.0], dtype=np.float32))
    assert np.array_equal(std, np.array([2.0, 4.0], dtype=np.float32))


def write_activation_fixture(root: Path, *, status: str = "complete") -> None:
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    np.save(root / "tiny.npy", array)
    (root / "tiny.json").write_text(
        json.dumps(
            {
                "name": "tiny",
                "status": status,
                "shape": [2, 2],
                "storage_dtype": "float16",
                "bos_dropped": True,
                "steering_vectors_used": None,
            }
        )
    )
    np.savez(
        root / "tiny_stats.npz",
        mean=np.array([2.0, 3.0], dtype=np.float32),
        std=np.array([1.0, 1.0], dtype=np.float32),
    )


def test_activation_loader_checks_status_shape_dtype_and_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import interp.activations as activations

    write_activation_fixture(tmp_path)
    monkeypatch.setattr(activations, "ACT_DIR", tmp_path)

    ds = load_activations("tiny")

    assert ds.array.shape == (2, 2)
    assert ds.array.dtype == np.float16
    assert np.array_equal(ds.mean, np.array([2.0, 3.0], dtype=np.float32))
    assert np.array_equal(ds.std, np.array([1.0, 1.0], dtype=np.float32))


def test_exact_activation_metadata_rejects_wrong_split_or_identity() -> None:
    array = np.zeros((8, 4), dtype=np.float16)
    meta = {
        "name": "canonical",
        "status": "complete",
        "split": "train",
        "hook": STEERING_HOOK,
        "model": MODEL_NAME,
        "resolved_model_name": MODEL_RESOLVED_NAME,
        "model_revision": MODEL_REVISION,
        "compute_dtype": "float32",
        "storage_dtype": "float16",
        "shape": [8, 4],
        "ctx": 5,
        "bos_dropped": True,
        "n_seqs": 2,
        "steering_vectors_used": None,
        "dataset_revision": DATASET_REVISION,
        "dataset_repository": "Skylion007/openwebtext",
        "dataset_config": "plain_text",
        "tokenizer": "gpt2",
        "token_cache_sha256": "a" * 64,
    }
    dataset = ActivationDataset(array, meta, np.zeros(4), np.ones(4))
    expected = dict(
        expected_name="canonical",
        expected_split="train",
        expected_model=MODEL_NAME,
        expected_resolved_model_name=MODEL_RESOLVED_NAME,
        expected_model_revision=MODEL_REVISION,
        expected_hook=STEERING_HOOK,
        expected_ctx=5,
        expected_d_model=4,
        expected_dataset_repository="Skylion007/openwebtext",
        expected_dataset_config="plain_text",
        expected_dataset_revision=DATASET_REVISION,
        expected_tokenizer="gpt2",
    )

    validate_activation_metadata(dataset, **expected)
    for field, wrong in (
        ("name", "alias"),
        ("split", "dev"),
        ("model", "wrong-model"),
        ("ctx", 6),
        ("n_seqs", 3),
        ("dataset_revision", "moving-main"),
        ("tokenizer", "other-tokenizer"),
    ):
        corrupted = ActivationDataset(array, {**meta, field: wrong}, dataset.mean, dataset.std)
        with pytest.raises(ValueError, match=field):
            validate_activation_metadata(corrupted, **expected)


def test_interrupted_activation_artifact_is_not_a_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import interp.activations as activations

    write_activation_fixture(tmp_path, status="INTERRUPTED")
    monkeypatch.setattr(activations, "ACT_DIR", tmp_path)

    with pytest.raises(ValueError, match="INTERRUPTED"):
        load_activations("tiny")


def test_validation_report_hashes_are_checked_again_before_training(tmp_path: Path) -> None:
    write_activation_fixture(tmp_path)
    hashes = artifact_hashes("tiny", tmp_path)
    report = {
        "status": "VALID",
        "name": "tiny",
        "split_fingerprint": "split123",
        "sha256": hashes,
    }
    (tmp_path / "tiny_validation.json").write_text(json.dumps(report))

    got = load_validation_report(
        "tiny", tmp_path, expected_split_fingerprint="split123", verify_hashes=True
    )
    assert got == report

    with (tmp_path / "tiny.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_validation_report(
            "tiny", tmp_path, expected_split_fingerprint="split123", verify_hashes=True
        )


def test_snapshot_revision_covers_the_dependency_lock(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("VALUE = 1\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'tiny'\n")
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text("version = 1\n")

    before = source_revision(tmp_path)
    lock_path.write_text("version = 2\n")

    assert before.startswith("snapshot-sha256:")
    assert source_revision(tmp_path) != before


def test_random_access_advice_does_not_change_any_byte(tmp_path) -> None:
    """MADV_RANDOM is an I/O hint: identical bytes, identical batches, same order."""

    import numpy as np

    from interp.activations import _advise_random_access

    rng = np.random.default_rng(0)
    values = rng.normal(size=(2048, 16)).astype(np.float16)
    path = tmp_path / "array.npy"
    np.save(path, values)

    plain = np.load(path, mmap_mode="r")
    advised = np.load(path, mmap_mode="r")
    _advise_random_access(advised)

    index_rng = np.random.default_rng(7)
    for _ in range(5):
        indices = np.sort(index_rng.integers(0, values.shape[0], size=64))
        assert np.array_equal(np.array(plain[indices]), np.array(advised[indices]))
    assert np.array_equal(np.asarray(advised), values)
