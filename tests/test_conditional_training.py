"""End-to-end conditional training on synthetic activations: run, resume, provenance.

Everything here is CPU, tiny, and synthetic.  No real activation dataset, no GPU,
no protected direction is involved.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from interp.activations import ActivationDataset
from interp.conditional_flow import (
    MIN_TRAINING_RANK,
    ConditionalFlowMatcher,
    ConditionEncoderConfig,
    load_training_direction_pool,
    save_direction_pool,
)
from interp.flow_core import ActivationNormalizer, FlowMatcher, FlowModelConfig
from interp.train_flow import (
    LEGACY_NORMALIZER_EPS,
    load_flow_checkpoint,
    load_training_config,
    save_flow_checkpoint,
    train_flow,
)
from test_flow_training import synthetic_dataset, tiny_training_config

ROOT = Path(__file__).resolve().parents[1]
CONDITIONAL_CONFIG = ROOT / "configs" / "flow_train_conditional_60m_v1.yaml"


def _pool_manifest(path: Path, *, width: int = 4, rows: int = 6, seed: int = 5) -> Path:
    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn(rows, width, generator=generator)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    save_direction_pool(
        path,
        directions,
        ranks=tuple(range(MIN_TRAINING_RANK, MIN_TRAINING_RANK + rows)),
        source="synthetic_test_fixture",
        selection="blake2b_priority_rank",
        selection_seed=20260807,
    )
    return path


def _conditional_config(dataset: ActivationDataset, pool_path: Path, **overrides):
    """Tiny authorized conditional config over the synthetic dataset."""

    base = tiny_training_config(dataset)
    conditional = load_training_config(CONDITIONAL_CONFIG)
    return replace(
        base,
        experiment_id="conditional_synthetic_test",
        experiment_class="conditional_prior_method_development",
        raw={**base.raw, "status": "authorized"},
        conditioning=replace(
            conditional.conditioning,
            condition=ConditionEncoderConfig(cond_hidden=5),
            direction_pool=str(pool_path),
        ),
        steps=4,
        eval_every=1,
        save_steps=(2,),
        **overrides,
    )


def test_prepared_config_cannot_start_a_run(tmp_path: Path) -> None:
    """The shipped conditional config is not human-authorized and must not run."""

    dataset = synthetic_dataset()
    cfg = replace(
        _conditional_config(dataset, _pool_manifest(tmp_path / "pool.pt")),
        raw={"status": "prepared"},
    )

    with pytest.raises(PermissionError, match="human-authorized"):
        train_flow(dataset, cfg, tmp_path / "run", device="cpu", progress=False)
    assert not (tmp_path / "run").exists() or not any((tmp_path / "run").iterdir())


def test_shipped_conditional_config_is_authorized_and_conditional() -> None:
    """Authorized by explicit human instruction on 2026-08-15; the guard itself is
    still exercised by test_prepared_config_cannot_start_a_run."""

    cfg = load_training_config(CONDITIONAL_CONFIG)

    assert cfg.status == "authorized"
    assert cfg.conditioning is not None
    assert cfg.conditioning.coordinate == "linear_projection"
    assert cfg.conditioning.condition.cond_hidden == 256
    assert cfg.held_out == "forbidden"


def test_conditional_run_trains_and_records_pool_provenance(tmp_path: Path) -> None:
    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    cfg = _conditional_config(dataset, pool_path)
    run_dir = tmp_path / "run"

    metadata = train_flow(dataset, cfg, run_dir, device="cpu", progress=False)

    assert len(metadata["history"]) == 4
    assert all(row["val_flow_mse"] > 0 for row in metadata["history"])
    identity = metadata["direction_pool"]
    assert identity["split"] == "training_only"
    assert identity["excluded_splits"] == ["dev", "held_out"]
    assert identity["observed_min_rank"] >= MIN_TRAINING_RANK
    assert identity["digest"] == load_training_direction_pool(pool_path).provenance.digest
    assert metadata["held_out_accessed"] is False
    # The written run metadata must round-trip through JSON with the pool record.
    written = json.loads((run_dir / "meta.json").read_text())
    assert written["direction_pool"] == identity

    model, checkpoint_meta, _ = load_flow_checkpoint(run_dir / "last.pt")
    assert isinstance(model, ConditionalFlowMatcher)
    assert model.cond_cfg == cfg.conditioning.condition
    assert model.normalizer.eps == cfg.norm_eps
    assert checkpoint_meta["direction_pool"] == identity


def test_conditional_resume_reproduces_the_uninterrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import interp.train_flow as training

    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    cfg = _conditional_config(dataset, pool_path)
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    expected_meta = train_flow(dataset, cfg, uninterrupted, device="cpu", progress=False)

    original_save = training.save_flow_checkpoint

    def save_then_interrupt(model, path, *, metadata, training_state=None, **kwargs):  # noqa: ANN001
        original_save(model, path, metadata=metadata, training_state=training_state, **kwargs)
        if path.name == "step_000002.pt":
            raise KeyboardInterrupt("simulated interruption after immutable checkpoint")

    monkeypatch.setattr(training, "save_flow_checkpoint", save_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
        train_flow(dataset, cfg, resumed, device="cpu", progress=False)
    monkeypatch.setattr(training, "save_flow_checkpoint", original_save)

    resumed_meta = train_flow(
        dataset,
        cfg,
        resumed,
        device="cpu",
        progress=False,
        resume_checkpoint=resumed / "step_000002.pt",
    )

    expected_model, _, _ = load_flow_checkpoint(uninterrupted / "last.pt")
    resumed_model, _, _ = load_flow_checkpoint(resumed / "last.pt")
    assert resumed_meta["history"] == expected_meta["history"]
    assert resumed_meta["best_val_flow_mse"] == expected_meta["best_val_flow_mse"]
    assert all(
        torch.equal(expected_model.state_dict()[name], resumed_model.state_dict()[name])
        for name in expected_model.state_dict()
    )
    # The condition encoder actually moved during those steps.
    fresh = ConditionalFlowMatcher(
        cfg.model, cfg.conditioning.condition, expected_model.normalizer
    )
    assert not torch.equal(
        fresh.condition.direction.weight, expected_model.condition.direction.weight
    )


def test_resume_rejects_a_checkpoint_trained_on_a_different_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import interp.train_flow as training

    dataset = synthetic_dataset()
    cfg = _conditional_config(dataset, _pool_manifest(tmp_path / "pool.pt"))
    run_dir = tmp_path / "run"
    original_save = training.save_flow_checkpoint

    def save_then_interrupt(model, path, *, metadata, training_state=None, **kwargs):  # noqa: ANN001
        original_save(model, path, metadata=metadata, training_state=training_state, **kwargs)
        if path.name == "step_000002.pt":
            raise KeyboardInterrupt("simulated interruption")

    monkeypatch.setattr(training, "save_flow_checkpoint", save_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        train_flow(dataset, cfg, run_dir, device="cpu", progress=False)
    monkeypatch.setattr(training, "save_flow_checkpoint", original_save)

    # Regenerate the manifest at the same path: the config fingerprint is unchanged,
    # so only the recorded pool provenance can catch the swap.
    (tmp_path / "pool.pt").unlink()
    _pool_manifest(tmp_path / "pool.pt", seed=77, rows=7)
    with pytest.raises(ValueError, match="direction_pool"):
        train_flow(
            dataset,
            cfg,
            run_dir,
            device="cpu",
            progress=False,
            resume_checkpoint=run_dir / "step_000002.pt",
        )


def test_pool_width_must_match_the_activation_width(tmp_path: Path) -> None:
    dataset = synthetic_dataset()
    cfg = _conditional_config(dataset, _pool_manifest(tmp_path / "wide.pt", width=9))

    with pytest.raises(ValueError, match="direction pool width"):
        train_flow(dataset, cfg, tmp_path / "run", device="cpu", progress=False)


def test_unconditional_training_is_unchanged_by_the_conditional_path(
    tmp_path: Path,
) -> None:
    """The frozen unconditional trainer must produce the same run it always did."""

    dataset = synthetic_dataset()
    cfg = replace(tiny_training_config(dataset), steps=3, eval_every=1, save_steps=())
    run_dir = tmp_path / "run"

    metadata = train_flow(dataset, cfg, run_dir, device="cpu", progress=False)
    model, _, _ = load_flow_checkpoint(run_dir / "last.pt")

    assert cfg.conditioning is None
    assert "conditioning" not in metadata["used_config"]
    assert metadata["direction_pool"] is None
    assert isinstance(model, FlowMatcher) and not isinstance(model, ConditionalFlowMatcher)


def test_normalizer_eps_survives_the_checkpoint_round_trip(tmp_path: Path) -> None:
    normalizer = ActivationNormalizer(torch.zeros(4), torch.ones(4), eps=3e-3)
    model = FlowMatcher(FlowModelConfig(4, 8, 2, 6, 5, 100.0), normalizer)
    path = tmp_path / "checkpoint.pt"

    save_flow_checkpoint(model, path, metadata={})
    restored, _, _ = load_flow_checkpoint(path)

    assert restored.normalizer.eps == 3e-3
    h = torch.randn(3, 4)
    assert torch.allclose(restored.normalizer.normalize(h), normalizer.normalize(h))


def test_format_1_checkpoints_keep_their_implicit_eps(tmp_path: Path) -> None:
    """Historical checkpoints never stored eps; they must load with the value used."""

    legacy_model = FlowMatcher(
        FlowModelConfig(4, 8, 2, 6, 5, 100.0),
        ActivationNormalizer(torch.zeros(4), torch.ones(4), eps=LEGACY_NORMALIZER_EPS),
    )
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "format_version": 1,
            "model_config": {
                "d_model": 4,
                "d_mlp": 8,
                "n_blocks": 2,
                "time_dim": 6,
                "time_hidden": 5,
                "max_period": 100.0,
                "activation_dim": None,
            },
            "state_dict": legacy_model.state_dict(),
            "metadata": {"legacy": True},
            "training_state": None,
        },
        path,
    )

    restored, metadata, state = load_flow_checkpoint(path)

    assert restored.normalizer.eps == LEGACY_NORMALIZER_EPS
    assert metadata == {"legacy": True}
    assert state is None


def test_historical_config_fingerprints_are_unchanged() -> None:
    """Adding conditioning must not renumber the fingerprints of completed runs."""

    from interp.train_flow import _config_fingerprint, _used_config

    expected = {
        "flow_train_100k_v1":
            "89f589c4f5587782156e09041bb6caed68c4a321340fdd06abd3876e4934b4ab",
        "flow_train_scaling_wide60m_fw32m_v2":
            "79ecee9ff9a45062d99848f47dcb574d39f806925b26a5be391804f971ee8464",
        "flow_train_scaling_narrow16m_fw32m_v2":
            "c00f06a5e81c9bf619ebfcb8e1d1bcee0b7988fcf7164a92640504b2a0f5e6b3",
    }
    for name, fingerprint in expected.items():
        cfg = load_training_config(ROOT / "configs" / f"{name}.yaml")
        assert cfg.conditioning is None, name
        assert _config_fingerprint(_used_config(cfg)) == fingerprint, name
