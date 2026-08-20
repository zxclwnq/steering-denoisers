"""Tests for the frozen 2x2 capacity/data protocol and the wide 60M-class flow model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from interp.activations import file_sha256
from interp.flow_core import (
    ActivationNormalizer,
    FlowMatcher,
    FlowModelConfig,
    flow_parameter_count,
    load_flow_config,
    n_parameters,
)
from interp.prior_diagnostic import wide_glp_parameter_count
from interp.scaling import (
    APPROVED_SCALING_PROTOCOL_SHA256S,
    apply_selection_rule,
    load_scaling_protocol,
    validation_flow_loss,
)
from interp.train_flow import load_flow_checkpoint, load_training_config, save_flow_checkpoint

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
PROTOCOL = CONFIGS / "flow_scaling_2x2_v2.yaml"
PROTOCOL_V1 = CONFIGS / "flow_scaling_2x2_v1.yaml"
NARROW = CONFIGS / "flow_core_v1.yaml"
WIDE = CONFIGS / "flow_core_wide_60m_v1.yaml"


def _build(config_path: Path) -> tuple[FlowModelConfig, FlowMatcher]:
    cfg = load_flow_config(config_path)
    width = cfg.activation_width
    model = FlowMatcher(cfg, ActivationNormalizer(torch.zeros(width), torch.ones(width)))
    return cfg, model


def test_wide_model_has_the_exact_frozen_parameter_count() -> None:
    cfg, model = _build(WIDE)

    assert (cfg.activation_dim, cfg.d_model, cfg.d_mlp, cfg.n_blocks) == (768, 1536, 3072, 3)
    assert (cfg.time_dim, cfg.time_hidden) == (256, 768)
    assert n_parameters(model) == 60_407_808
    assert flow_parameter_count(cfg) == 60_407_808
    assert (
        wide_glp_parameter_count(
            activation_dim=768,
            d_model=1536,
            d_mlp=3072,
            n_blocks=3,
            time_dim=256,
            time_hidden=768,
        )
        == 60_407_808
    )


def test_narrow_model_is_unchanged_by_the_activation_dim_decoupling() -> None:
    cfg, model = _build(NARROW)

    assert cfg.activation_dim is None
    assert cfg.activation_width == 768
    assert n_parameters(model) == 16_147_200
    assert flow_parameter_count(cfg) == 16_147_200


def test_wide_model_maps_768_to_768_and_stays_tokenwise() -> None:
    _, model = _build(WIDE)
    x = torch.randn(6, 768)
    t = torch.rand(6, 1)

    out = model(x, t)
    permutation = torch.tensor([3, 1, 0, 5, 4, 2])
    permuted = model(x[permutation], t[permutation])

    assert out.shape == (6, 768)
    # Batched float32 matmul reorders reductions, so row-permutation invariance is
    # only exact to a few ULPs of the output scale (observed 1.3e-6 against a mean
    # magnitude near 0.45 on an eight-thread CPU). The tolerance is loosened to
    # 1e-5 and the contrast assertion below keeps the invariant meaningful: real
    # cross-token mixing moves other rows by orders of magnitude more than this.
    assert torch.allclose(out[permutation], permuted, atol=1e-5)

    disturbed = x.clone()
    disturbed[0] += 50.0
    moved = model(disturbed, t)
    assert torch.allclose(moved[1:], out[1:], atol=1e-5)
    assert float((moved[0] - out[0]).abs().max()) > 1e-2

    with pytest.raises(ValueError, match="activation_dim=768"):
        model(torch.randn(6, 1536), t)


def test_wide_model_trains_and_round_trips_through_a_checkpoint(tmp_path: Path) -> None:
    cfg = load_flow_config(WIDE)
    normalizer = ActivationNormalizer(torch.zeros(768), torch.ones(768))
    model = FlowMatcher(cfg, normalizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(8, 768)
    t = torch.rand(8, 1)
    target = torch.randn(8, 768)

    loss = (model(x, t) - target).square().mean()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer.step()

    path = tmp_path / "wide.pt"
    save_flow_checkpoint(model, path, metadata={"step": 1})
    reloaded, metadata, _ = load_flow_checkpoint(path)

    assert metadata["step"] == 1
    assert reloaded.cfg == cfg
    assert n_parameters(reloaded) == 60_407_808
    assert torch.allclose(reloaded(x, t), model(x, t), atol=1e-6)


def test_narrow_checkpoints_written_before_activation_dim_existed_still_load(
    tmp_path: Path,
) -> None:
    legacy = {
        "format_version": 1,
        "model_config": {
            "d_model": 8,
            "d_mlp": 16,
            "n_blocks": 1,
            "time_dim": 4,
            "time_hidden": 8,
            "max_period": 10000.0,
        },
        "metadata": {"step": 7},
        "training_state": None,
    }
    cfg = FlowModelConfig(**legacy["model_config"])
    model = FlowMatcher(cfg, ActivationNormalizer(torch.zeros(8), torch.ones(8)))
    legacy["state_dict"] = model.state_dict()
    path = tmp_path / "legacy.pt"
    torch.save(legacy, path)

    reloaded, metadata, _ = load_flow_checkpoint(path)

    assert metadata["step"] == 7
    assert reloaded.cfg.activation_dim is None
    assert reloaded.cfg.activation_width == 8


def test_declared_parameter_count_in_a_config_is_enforced(tmp_path: Path) -> None:
    raw = yaml.safe_load(WIDE.read_text())
    raw["architecture"]["n_params_expected"] = 60_407_809
    path = tmp_path / "wrong.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="parameters"):
        load_flow_config(path)


def test_scaling_protocol_is_frozen_and_self_consistent() -> None:
    protocol = load_scaling_protocol(PROTOCOL)

    assert protocol.experiment_id == "flow_scaling_2x2_v2"
    assert [arm.arm_id for arm in protocol.arms] == [
        "narrow16m_fw4m",
        "narrow16m_fw32m",
        "wide60m_fw4m",
        "wide60m_fw32m",
    ]
    assert protocol.optimizer_steps * protocol.batch_size == 256_000_000
    assert protocol.optimizer_steps == 250_000
    assert protocol.t_starts == (0.10, 0.25, 0.50)
    assert protocol.nfes == (1, 3, 5)
    assert (protocol.primary_t_start, protocol.primary_nfe) == (0.50, 1)
    assert protocol.validation_artifact == "resid7_fw_val_1024k_v1"
    assert max(arm.parameters for arm in protocol.arms) == 60_407_808


def test_scaling_protocol_rejects_an_edited_copy(tmp_path: Path) -> None:
    edited = tmp_path / "edited.yaml"
    edited.write_text(PROTOCOL.read_text() + "\n# drift\n")

    with pytest.raises(ValueError, match="SHA256"):
        load_scaling_protocol(edited)


def test_every_frozen_protocol_version_still_loads_at_its_pinned_digest() -> None:
    for path in (PROTOCOL_V1, PROTOCOL):
        protocol = load_scaling_protocol(path)
        assert (
            APPROVED_SCALING_PROTOCOL_SHA256S[protocol.experiment_id]
            == file_sha256(path)
        )
    v1 = load_scaling_protocol(PROTOCOL_V1)
    v2 = load_scaling_protocol(PROTOCOL)

    # v2 changes the training budget only; the design itself is untouched.
    assert v1.optimizer_steps == 100_000 and v2.optimizer_steps == 250_000
    assert [arm.arm_id for arm in v1.arms] == [arm.arm_id for arm in v2.arms]
    assert [arm.parameters for arm in v1.arms] == [arm.parameters for arm in v2.arms]
    assert [arm.dataset for arm in v1.arms] == [arm.dataset for arm in v2.arms]
    assert v1.validation_artifact == v2.validation_artifact
    assert (v1.t_starts, v1.nfes) == (v2.t_starts, v2.nfes)
    assert (v1.primary_t_start, v1.primary_nfe) == (v2.primary_t_start, v2.primary_nfe)
    assert v1.bootstrap_seed == v2.bootstrap_seed
    assert v1.tie_breakers == v2.tie_breakers


def test_every_arm_config_matches_the_protocol_and_shares_one_recipe() -> None:
    protocol = load_scaling_protocol(PROTOCOL)
    recipes = set()
    for arm in protocol.arms:
        config = load_training_config(REPO / arm.training_config)
        raw = config.raw

        assert config.experiment_class == "concept_independent_capacity_data_scaling"
        assert config.dataset_name == arm.dataset
        assert config.dataset_repository == "HuggingFaceFW/fineweb"
        assert config.dataset_config == "sample-10BT"
        assert config.steps == protocol.optimizer_steps
        assert config.batch_size == protocol.batch_size
        assert flow_parameter_count(config.model) == arm.parameters
        assert raw["compute_budget"]["total_activation_presentations"] == (
            protocol.total_activation_presentations
        )
        assert raw["compute_budget"]["unique_activation_tokens"] == arm.unique_activation_tokens
        assert raw["protected_data"]["held_out"] == "forbidden"
        recipes.add(
            (
                config.lr,
                config.weight_decay,
                config.warmup_steps,
                config.grad_clip,
                config.training_seed,
                config.noise_seed,
                config.eval_every,
                raw["training"]["schedule"],
                raw["training"]["dtype"],
            )
        )
    assert len(recipes) == 1, "the 2x2 must not vary the optimizer recipe between arms"


def test_arm_configs_only_vary_capacity_and_unique_data() -> None:
    protocol = load_scaling_protocol(PROTOCOL)
    widths = {
        arm.arm_id: load_training_config(REPO / arm.training_config).model
        for arm in protocol.arms
    }
    datasets = {arm.arm_id: arm.dataset for arm in protocol.arms}

    assert widths["narrow16m_fw4m"] == widths["narrow16m_fw32m"]
    assert widths["wide60m_fw4m"] == widths["wide60m_fw32m"]
    assert widths["narrow16m_fw4m"] != widths["wide60m_fw4m"]
    assert datasets["narrow16m_fw4m"] == datasets["wide60m_fw4m"]
    assert datasets["narrow16m_fw32m"] == datasets["wide60m_fw32m"]
    for cfg in widths.values():
        assert cfg.activation_width == 768
        assert cfg.n_blocks == 3


def _arm(arm_id: str, effects: np.ndarray, *, val: float, params: int, tokens: int) -> dict:
    return {
        "arm_id": arm_id,
        "primary_effects": effects.tolist(),
        "val_flow_mse": val,
        "parameters": params,
        "unique_activation_tokens": tokens,
    }


def test_selection_rule_resolves_a_clear_winner_without_steering_metrics() -> None:
    protocol = load_scaling_protocol(PROTOCOL)
    rng = np.random.default_rng(0)
    base = rng.normal(1.0, 0.05, size=protocol.n_sequences)
    better = base - 0.5

    result = apply_selection_rule(
        [
            _arm("narrow16m_fw4m", base, val=1.0, params=16_147_200, tokens=4_000_119),
            _arm("wide60m_fw32m", better, val=0.9, params=60_407_808, tokens=32_000_063),
        ],
        protocol,
    )

    assert result["selected_arm"] == "wide60m_fw32m"
    assert result["decided_by"] == "primary paired bootstrap"
    assert result["tied_with_leader"] == ["wide60m_fw32m"]
    assert result["steering_metrics_used"] is False


def test_unresolved_difference_falls_back_to_the_frozen_cheapness_tie_breakers() -> None:
    protocol = load_scaling_protocol(PROTOCOL)
    rng = np.random.default_rng(1)
    base = rng.normal(1.0, 0.05, size=protocol.n_sequences)
    # An independent arm whose per-sequence values differ by noise, not by a real
    # effect: the paired difference spreads across zero and must stay unresolved.
    barely_better = rng.normal(1.0, 0.05, size=protocol.n_sequences) - 1e-6

    result = apply_selection_rule(
        [
            _arm("narrow16m_fw4m", base, val=1.0, params=16_147_200, tokens=4_000_119),
            _arm("wide60m_fw32m", barely_better, val=1.0, params=60_407_808, tokens=32_000_063),
        ],
        protocol,
    )

    assert sorted(result["tied_with_leader"]) == ["narrow16m_fw4m", "wide60m_fw32m"]
    assert result["selected_arm"] == "narrow16m_fw4m"
    assert result["decided_by"] == "fewer parameters"


def test_selection_rule_rejects_a_wrong_sized_or_non_finite_arm() -> None:
    protocol = load_scaling_protocol(PROTOCOL)
    short = np.zeros(8)

    with pytest.raises(ValueError, match="per-sequence"):
        apply_selection_rule(
            [_arm("narrow16m_fw4m", short, val=1.0, params=16_147_200, tokens=4_000_119)],
            protocol,
        )
    with pytest.raises(ValueError, match="non-finite"):
        apply_selection_rule(
            [
                _arm(
                    "narrow16m_fw4m",
                    np.full(protocol.n_sequences, np.nan),
                    val=1.0,
                    params=16_147_200,
                    tokens=4_000_119,
                )
            ],
            protocol,
        )


def _zero_model(config_path: Path, width: int) -> FlowMatcher:
    _, model = _build(config_path)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    assert model.cfg.activation_width == width
    return model


def _tiny_validation_dataset(rows: int = 4096):
    from interp.activations import ActivationDataset

    rng = np.random.default_rng(3)
    array = rng.normal(0.0, 3.0, size=(rows, 768)).astype(np.float16)
    return ActivationDataset(
        array=array,
        meta={"shape": [rows, 768]},
        mean=np.zeros(768, dtype=np.float32),
        std=np.ones(768, dtype=np.float32),
    )


def test_validation_flow_loss_is_deterministic_and_noise_matched_across_capacities() -> None:
    protocol = load_scaling_protocol(PROTOCOL)
    dataset = _tiny_validation_dataset()
    indices = np.arange(len(dataset), dtype=np.int64)
    device = torch.device("cpu")
    narrow = _zero_model(NARROW, 768)
    wide = _zero_model(WIDE, 768)

    first = validation_flow_loss(narrow, dataset, indices, protocol, device)
    repeat = validation_flow_loss(narrow, dataset, indices, protocol, device)
    other = validation_flow_loss(wide, dataset, indices, protocol, device)

    assert first == repeat
    # A zero-weight velocity model predicts zero, so its flow MSE must equal the
    # zero-predictor control exactly, and both capacities must see identical
    # times and noise on identical rows.
    assert first["val_flow_mse"] == pytest.approx(first["zero_predictor_mse"], rel=1e-6)
    assert other["zero_predictor_mse"] == pytest.approx(first["zero_predictor_mse"], rel=1e-6)
    assert other["val_flow_mse_by_bin"] == pytest.approx(first["val_flow_mse_by_bin"], rel=1e-6)
    assert first["n_rows"] == 4096 and first["batches"] == 4
    assert first["seed"] == protocol.flow_loss_seed


def test_validation_flow_loss_refuses_a_partial_batch() -> None:
    protocol = load_scaling_protocol(PROTOCOL)
    dataset = _tiny_validation_dataset(rows=100)
    with pytest.raises(ValueError, match="at least one full batch"):
        validation_flow_loss(
            _zero_model(NARROW, 768),
            dataset,
            np.arange(100, dtype=np.int64),
            protocol,
            torch.device("cpu"),
        )
