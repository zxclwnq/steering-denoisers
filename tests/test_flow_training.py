"""Training-pipeline tests; all use synthetic activations and CPU."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from interp.activations import ActivationDataset, make_split, split_stats
from interp.data import DATASET, DATASET_CONFIG, DATASET_REVISION
from interp.flow_core import ActivationNormalizer, FlowMatcher, FlowModelConfig
from interp.train_flow import (
    FlowTrainingConfig,
    evaluate_flow,
    load_flow_checkpoint,
    load_training_config,
    save_flow_checkpoint,
    train_flow,
)

ROOT = Path(__file__).resolve().parents[1]


def synthetic_dataset(
    n_sequences: int = 100, per_seq: int = 4, d_model: int = 4
) -> ActivationDataset:
    rng = np.random.default_rng(42)
    array = rng.normal(size=(n_sequences * per_seq, d_model)).astype(np.float16)
    split = make_split(len(array), per_seq=per_seq, val_fraction=0.2, seed=7)
    hashes = {"array": "1" * 64, "metadata": "2" * 64, "statistics": "3" * 64}
    return ActivationDataset(
        array=array,
        meta={
            "name": "synthetic_train",
            "status": "complete",
            "split": "train",
            "hook": "blocks.7.hook_resid_pre",
            "model": "gpt2-small",
            "resolved_model_name": "gpt2",
            "model_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
            "compute_dtype": "float32",
            "storage_dtype": "float16",
            "shape": list(array.shape),
            "ctx": per_seq + 1,
            "bos_dropped": True,
            "n_seqs": n_sequences,
            "steering_vectors_used": None,
            "dataset_repository": DATASET,
            "dataset_config": DATASET_CONFIG,
            "dataset_revision": DATASET_REVISION,
            "tokenizer": "gpt2",
            "token_cache_sha256": "a" * 64,
            "full_validation_report": {
                "status": "VALID",
                "name": "synthetic_train",
                "split_fingerprint": split.fingerprint(),
                "sha256": hashes,
                "token_cache_sha256": "a" * 64,
            },
        },
        mean=array.astype(np.float32).mean(0),
        std=array.astype(np.float32).std(0),
    )


def tiny_training_config(dataset: ActivationDataset) -> FlowTrainingConfig:
    base = load_training_config(ROOT / "configs" / "flow_train_100k_v1.yaml")
    split = make_split(len(dataset), per_seq=4, val_fraction=0.2, seed=7)
    return replace(
        base,
        dataset_name="synthetic_train",
        per_seq=4,
        val_fraction=0.2,
        split_seed=7,
        split_fingerprint=split.fingerprint(),
        model=FlowModelConfig(4, 8, 2, 6, 5, 100.0),
        steps=3,
        batch_size=8,
        warmup_steps=1,
        eval_every=1,
        eval_batches=2,
        t_bins=(0.0, 0.5, 1.0),
        save_steps=(2,),
    )


def test_100k_config_is_a_new_dev_budget_variant_with_protected_data_forbidden() -> None:
    cfg = load_training_config(ROOT / "configs" / "flow_train_100k_v1.yaml")

    assert cfg.experiment_id == "clean_flow_100k_v1"
    assert cfg.experiment_class == "dev_method_development"
    assert cfg.steps == 100_000
    assert cfg.dataset_name == "resid7_train_4000k_v2"
    assert cfg.dataset_repository == DATASET
    assert cfg.dataset_revision == DATASET_REVISION
    assert cfg.per_seq == 127
    assert cfg.split_seed == 20_260_807
    assert cfg.split_fingerprint == "c34aec678a328131"
    assert cfg.training_seed == 0
    assert cfg.noise_seed == 20_260_812
    assert cfg.held_out == "forbidden"
    assert cfg.selection_metric == "val_flow_mse"
    assert cfg.model == FlowModelConfig(768, 1536, 3, 256, 768, 10_000.0)


def test_training_rejects_wrong_split_even_when_shape_and_fingerprint_match(
    tmp_path: Path,
) -> None:
    dataset = synthetic_dataset()
    dataset = replace(dataset, meta={**dataset.meta, "split": "dev"})

    with pytest.raises(ValueError, match="split"):
        train_flow(
            dataset,
            tiny_training_config(dataset),
            tmp_path / "wrong-split",
            device="cpu",
            progress=False,
        )


def test_checkpoint_round_trip_carries_model_normalizer_and_metadata(tmp_path: Path) -> None:
    torch.manual_seed(3)
    cfg = FlowModelConfig(4, 8, 1, 6, 5, 100.0)
    normalizer = ActivationNormalizer(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.ones(4))
    model = FlowMatcher(cfg, normalizer).eval()
    x = torch.randn(5, 4)
    t = torch.linspace(0.0, 1.0, 5)
    expected = model(x, t)
    path = tmp_path / "checkpoint.pt"

    save_flow_checkpoint(model, path, metadata={"step": 17, "dataset": "synthetic"})
    loaded, metadata, training_state = load_flow_checkpoint(path)

    assert torch.equal(loaded(x, t), expected)
    assert torch.equal(loaded.normalizer.mean, normalizer.mean)
    assert metadata == {"step": 17, "dataset": "synthetic"}
    assert training_state is None

    with pytest.raises(FileExistsError, match="overwrite"):
        save_flow_checkpoint(model, path, metadata={"step": 18})


def test_fixed_seed_evaluation_is_reproducible_and_reports_zero_control() -> None:
    dataset = synthetic_dataset(n_sequences=800)
    cfg = replace(
        tiny_training_config(dataset),
        batch_size=128,
        eval_batches=20,
    )
    split = make_split(len(dataset), cfg.per_seq, cfg.val_fraction, cfg.split_seed)
    normalizer = ActivationNormalizer(
        torch.from_numpy(dataset.mean), torch.from_numpy(dataset.std)
    )
    model = FlowMatcher(cfg.model, normalizer)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)

    first = evaluate_flow(model, dataset, split.val, cfg, torch.device("cpu"))
    second = evaluate_flow(model, dataset, split.val, cfg, torch.device("cpu"))

    assert first == second
    assert first["val_flow_mse"] == pytest.approx(first["zero_predictor_mse"])
    assert first["zero_predictor_mse"] == pytest.approx(2.0, abs=0.15)
    assert set(first["val_flow_mse_by_bin"]) == {"0.00-0.50", "0.50-1.00"}


def test_tiny_training_updates_parameters_and_writes_nonoverwriting_artifacts(
    tmp_path: Path,
) -> None:
    dataset = synthetic_dataset()
    cfg = tiny_training_config(dataset)
    run_dir = tmp_path / "run"

    metadata = train_flow(dataset, cfg, run_dir, device="cpu", progress=False)

    assert metadata["status"] == "complete"
    assert metadata["steps"] == 3
    assert metadata["split_fingerprint"] == cfg.split_fingerprint
    assert metadata["held_out_accessed"] is False
    assert metadata["environment"]["python"]
    assert metadata["environment"]["torch"] == torch.__version__
    assert metadata["used_config"]["steps"] == 3
    assert metadata["used_config"]["model"] == {
        "d_model": 4,
        "d_mlp": 8,
        "n_blocks": 2,
        "time_dim": 6,
        "time_hidden": 5,
        "max_period": 100.0,
        "activation_dim": None,
    }
    best_pointer = json.loads((run_dir / "best.json").read_text())
    assert (run_dir / best_pointer["checkpoint"]).is_file()
    assert best_pointer["checkpoint"].startswith("best_step_")
    assert (run_dir / "last.pt").is_file()
    assert (run_dir / "step_000002.pt").is_file()
    assert (run_dir / "meta.json").is_file()
    trained, last_meta, training_state = load_flow_checkpoint(run_dir / "last.pt")
    assert last_meta["step"] == 3
    assert training_state is not None
    assert training_state["optimizer"]
    assert training_state["scheduler"]
    split = make_split(len(dataset), cfg.per_seq, cfg.val_fraction, cfg.split_seed)
    mean, std = split_stats(dataset, split.train)
    torch.manual_seed(cfg.training_seed)
    initial = FlowMatcher(
        cfg.model,
        ActivationNormalizer(torch.from_numpy(mean), torch.from_numpy(std), cfg.norm_eps),
    )
    assert any(
        not torch.equal(initial_value, trained.state_dict()[name])
        for name, initial_value in initial.state_dict().items()
        if name not in {"normalizer.mean", "normalizer.std"}
    )
    assert len(metadata["history"]) == 3
    assert all(np.isfinite(row["train_loss"]) for row in metadata["history"])

    with pytest.raises(FileExistsError, match="already contains"):
        train_flow(dataset, cfg, run_dir, device="cpu", progress=False)


def test_resume_restores_exact_optimizer_scheduler_and_rng_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import interp.train_flow as training

    device = os.environ.get("INTERP_RESUME_TEST_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.fail("INTERP_RESUME_TEST_DEVICE=cuda but CUDA is unavailable")
    dataset = synthetic_dataset()
    cfg = replace(
        tiny_training_config(dataset),
        steps=4,
        eval_every=1,
        save_steps=(2,),
    )
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    expected_meta = train_flow(
        dataset, cfg, uninterrupted_dir, device=device, progress=False
    )

    original_save = training.save_flow_checkpoint

    def save_then_interrupt(model, path, *, metadata, training_state=None, **kwargs):  # noqa: ANN001
        original_save(
            model, path, metadata=metadata, training_state=training_state, **kwargs
        )
        if path.name == "step_000002.pt":
            raise KeyboardInterrupt("simulated interruption after immutable checkpoint")

    monkeypatch.setattr(training, "save_flow_checkpoint", save_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
        train_flow(dataset, cfg, resumed_dir, device=device, progress=False)
    monkeypatch.setattr(training, "save_flow_checkpoint", original_save)

    changed_report = {
        **dataset.meta["full_validation_report"],
        "sha256": {
            **dataset.meta["full_validation_report"]["sha256"],
            "array": "f" * 64,
        },
    }
    changed_dataset = replace(
        dataset,
        meta={**dataset.meta, "full_validation_report": changed_report},
    )
    with pytest.raises(ValueError, match="dataset_artifact_identity"):
        train_flow(
            changed_dataset,
            cfg,
            resumed_dir,
            device=device,
            progress=False,
            resume_checkpoint=resumed_dir / "step_000002.pt",
        )

    (resumed_dir / "status_RESUMED_from_000002.json").write_text("{}\n")

    resumed_meta = train_flow(
        dataset,
        cfg,
        resumed_dir,
        device=device,
        progress=False,
        resume_checkpoint=resumed_dir / "step_000002.pt",
    )
    expected_model, _, _ = load_flow_checkpoint(uninterrupted_dir / "last.pt")
    resumed_model, _, _ = load_flow_checkpoint(resumed_dir / "last.pt")

    assert resumed_meta["history"] == expected_meta["history"]
    assert resumed_meta["best_val_flow_mse"] == expected_meta["best_val_flow_mse"]
    assert all(
        torch.equal(expected_model.state_dict()[name], resumed_model.state_dict()[name])
        for name in expected_model.state_dict()
    )


def test_only_the_current_best_checkpoint_is_retained(tmp_path: Path) -> None:
    """keep: [best, last, configured_steps] must be real, not decorative."""

    dataset = synthetic_dataset()
    cfg = replace(tiny_training_config(dataset), steps=6, eval_every=1, save_steps=(2, 4))
    run_dir = tmp_path / "run"

    metadata = train_flow(dataset, cfg, run_dir, device="cpu", progress=False)

    best_files = sorted(path.name for path in run_dir.glob("best_step_*.pt"))
    step_files = sorted(path.name for path in run_dir.glob("step_*.pt"))
    pointer = json.loads((run_dir / "best.json").read_text())

    assert best_files == [metadata["best_checkpoint"]], best_files
    assert pointer["checkpoint"] == metadata["best_checkpoint"]
    assert (run_dir / metadata["best_checkpoint"]).is_file()
    assert step_files == ["step_000002.pt", "step_000004.pt"]
    assert (run_dir / "last.pt").is_file()


def test_retention_policy_must_be_declared_in_the_config(tmp_path: Path) -> None:
    import yaml

    raw = yaml.safe_load((ROOT / "configs" / "flow_train_100k_v1.yaml").read_text())
    raw["checkpoints"]["keep"] = ["best", "last", "configured_steps", "every_improvement"]
    path = tmp_path / "configs" / "drifted.yaml"
    path.parent.mkdir(parents=True)
    (tmp_path / "configs" / "flow_core_v1.yaml").write_text(
        (ROOT / "configs" / "flow_core_v1.yaml").read_text()
    )
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="retention policy"):
        load_training_config(path)
