"""Tangent objective inside the unified trainer: dispatch, provenance, resume gating.

Everything here is CPU, tiny, and synthetic. No real activation dataset, no GPU,
no protected direction is involved, and nothing is trained for real.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from interp.activations import ActivationDataset
from interp.conditional_flow import ConditionalFlowMatcher, ConditionEncoderConfig
from interp.tangent_flow import (
    ISOTROPIC_OBJECTIVE,
    TANGENT_OBJECTIVE,
    TangentFlowBatch,
    coordinate,
)
from interp.train_flow import (
    FlowObjectiveSpec,
    _config_fingerprint,
    _predict,
    _sample_batch,
    _used_config,
    load_flow_checkpoint,
    load_training_config,
    normalizer_identity,
    save_flow_checkpoint,
    train_flow,
)
from test_conditional_training import _conditional_config, _pool_manifest
from test_flow_training import synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]
TANGENT_CONFIG = ROOT / "configs" / "flow_train_tangent_narrow16m_fw32m_v1.yaml"
TANGENT_CORE = ROOT / "configs" / "flow_core_conditional_narrow16m_v1.yaml"


def _tangent_config(dataset: ActivationDataset, pool_path: Path, **overrides):
    return replace(
        _conditional_config(dataset, pool_path),
        experiment_id="tangent_synthetic_test",
        experiment_class="tangent_prior_method_development",
        flow_objective=FlowObjectiveSpec(
            type=TANGENT_OBJECTIVE, output_projection=True
        ),
        **overrides,
    )


# --------------------------------------------------------------------------
# prepared config (must not run)
# --------------------------------------------------------------------------


def test_tangent_config_is_authorized_and_scientifically_unchanged() -> None:
    """Authorized by explicit human instruction on 2026-08-16 for the T1 GPU run.

    Only the `status` field changed. `status` lives in `raw`, which `_used_config`
    drops, so the scientific fingerprint is unaffected -- asserted below against
    the value recorded before authorization.
    """

    cfg = load_training_config(TANGENT_CONFIG)

    assert cfg.status == "authorized"
    assert (
        _config_fingerprint(_used_config(cfg))
        == "e4af61135b0205cdcd6f196a61d5af464f0369b29e9bcaa471fd0764e7f85499"
    )
    assert cfg.objective_type == TANGENT_OBJECTIVE
    assert cfg.flow_objective.output_projection is True
    assert cfg.conditioning is not None
    assert cfg.held_out == "forbidden"
    assert cfg.experiment_class == "tangent_prior_method_development"
    # ~16M class, not the 60M one
    assert cfg.model.d_model == 768
    assert cfg.model.n_blocks == 3
    # the frozen recipe is untouched by authorization
    assert (cfg.steps, cfg.batch_size, cfg.lr) == (250000, 1024, 3.0e-4)
    assert (cfg.training_seed, cfg.noise_seed) == (0, 20260816)


def test_a_non_authorized_config_still_refuses_to_launch(tmp_path: Path) -> None:
    """The authorization guard itself, still proven on a prepared variant."""

    cfg = replace(load_training_config(TANGENT_CONFIG), raw={"status": "prepared"})
    assert cfg.status == "prepared"

    with pytest.raises(PermissionError, match="human-authorized"):
        train_flow(synthetic_dataset(), cfg, tmp_path / "run", device="cpu", progress=False)
    assert not (tmp_path / "run").exists() or not any((tmp_path / "run").iterdir())


def test_sixty_million_conditional_config_still_loads_unchanged() -> None:
    """The larger architecture must be instantiable later without code changes."""

    cfg = load_training_config(ROOT / "configs" / "flow_train_conditional_60m_v1.yaml")
    assert cfg.model.d_model == 1536
    assert cfg.objective_type == ISOTROPIC_OBJECTIVE


def test_tangent_objective_requires_conditioning(tmp_path: Path) -> None:
    import yaml

    raw = yaml.safe_load(TANGENT_CONFIG.read_text())
    raw.pop("conditioning")
    broken = tmp_path / "no_conditioning.yaml"
    broken.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="requires a conditioning section"):
        load_training_config(broken)


@pytest.mark.parametrize(
    ("field", "value"),
    [("held_out", "allowed"), ("dev", "allowed"), ("dev", None)],
)
def test_tangent_training_rejects_protected_data_access(
    tmp_path: Path, field: str, value: str | None
) -> None:
    """P13 regression: the tangent path must forbid BOTH dev and held-out."""

    import yaml

    raw = yaml.safe_load(TANGENT_CONFIG.read_text())
    assert raw["protected_data"] == {"held_out": "forbidden", "dev": "forbidden"}
    if value is None:
        raw["protected_data"].pop(field)
    else:
        raw["protected_data"][field] = value
    broken = tmp_path / "leaky.yaml"
    broken.write_text(yaml.safe_dump(raw))

    expected = "held-out access must be forbidden" if field == "held_out" else "dev = forbidden"
    with pytest.raises(ValueError, match=expected):
        load_training_config(broken)


def test_historical_isotropic_configs_keep_their_existing_contract() -> None:
    """Adding the dev requirement must not retroactively invalidate old configs."""

    import yaml

    for name in (
        "flow_train_scaling_narrow16m_fw32m_v2.yaml",
        "flow_train_conditional_60m_v1.yaml",
    ):
        path = ROOT / "configs" / name
        raw = yaml.safe_load(path.read_text())
        assert "flow_objective" not in raw
        assert "dev" not in raw["protected_data"]
        load_training_config(path)  # still loads


def test_unknown_objective_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="the non-default flow objectives"):
        FlowObjectiveSpec(type="something_else", output_projection=True)


# --------------------------------------------------------------------------
# fingerprint and RNG compatibility with the existing variants
# --------------------------------------------------------------------------


def test_isotropic_configs_keep_their_fingerprint_and_rng_consumption(
    tmp_path: Path,
) -> None:
    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    conditional = _conditional_config(dataset, pool_path)

    used = _used_config(conditional)
    assert "flow_objective" not in used
    assert conditional.objective_type == ISOTROPIC_OBJECTIVE

    # An isotropic batch must consume the generator exactly as before.
    model = ConditionalFlowMatcher(
        conditional.model,
        conditional.conditioning.condition,
        _normalizer(dataset, conditional),
    )
    from interp.conditional_flow import load_training_direction_pool

    pool = load_training_direction_pool(pool_path)
    h = torch.randn(6, conditional.model.activation_width)
    states = []
    for objective in (None, ISOTROPIC_OBJECTIVE):
        generator = torch.Generator().manual_seed(3)
        kwargs = {} if objective is None else {"objective": objective}
        _sample_batch(model, h, pool=pool, generator=generator, **kwargs)
        states.append(generator.get_state())
    assert torch.equal(states[0], states[1])


def test_tangent_config_fingerprint_differs_from_the_isotropic_one(
    tmp_path: Path,
) -> None:
    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    isotropic = _conditional_config(dataset, pool_path)
    tangent = _tangent_config(dataset, pool_path)
    assert _config_fingerprint(_used_config(isotropic)) != _config_fingerprint(
        _used_config(tangent)
    )


def _normalizer(dataset: ActivationDataset, cfg):  # noqa: ANN001, ANN202
    from interp.activations import make_split, split_stats
    from interp.flow_core import ActivationNormalizer

    split = make_split(len(dataset), cfg.per_seq, cfg.val_fraction, cfg.split_seed)
    mean, std = split_stats(dataset, split.train)
    return ActivationNormalizer(
        torch.from_numpy(mean), torch.from_numpy(std), cfg.norm_eps
    )


# --------------------------------------------------------------------------
# trainer dispatch
# --------------------------------------------------------------------------


def test_trainer_dispatch_produces_tangent_batches_and_tangent_predictions(
    tmp_path: Path,
) -> None:
    from interp.conditional_flow import load_training_direction_pool

    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    cfg = _tangent_config(dataset, pool_path)
    normalizer = _normalizer(dataset, cfg)
    model = ConditionalFlowMatcher(cfg.model, cfg.conditioning.condition, normalizer)
    pool = load_training_direction_pool(pool_path)

    h = torch.randn(8, cfg.model.activation_width)
    batch = _sample_batch(
        model,
        h,
        pool=pool,
        generator=torch.Generator().manual_seed(1),
        objective=cfg.objective_type,
    )
    assert isinstance(batch, TangentFlowBatch)
    assert coordinate(batch.target_velocity, batch.v_x).abs().max() < 1e-4

    projected = _predict(model, batch.x_t, batch.t, batch, output_projection=True)
    unprojected = _predict(model, batch.x_t, batch.t, batch, output_projection=False)
    assert coordinate(projected, batch.v_x).abs().max() < 1e-4
    assert coordinate(unprojected, batch.v_x).abs().max() > 1e-4


def test_tangent_run_records_objective_provenance_and_trains(tmp_path: Path) -> None:
    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    cfg = _tangent_config(dataset, pool_path)
    run_dir = tmp_path / "run"

    metadata = train_flow(dataset, cfg, run_dir, device="cpu", progress=False)

    identity = metadata["objective_identity"]
    assert identity["flow_objective"] == TANGENT_OBJECTIVE
    assert identity["condition_type"] == "direction_coordinate_film"
    assert identity["tangent_output_projection"] is True
    assert identity["normalizer"] == normalizer_identity(_normalizer(dataset, cfg))
    assert metadata["direction_pool"]["split"] == "training_only"
    assert metadata["held_out_accessed"] is False
    # the tangent-specific diagnostic is recorded for every evaluation
    assert all("val_raw_parallel_velocity_mean" in row for row in metadata["history"])

    model, checkpoint_meta, _ = load_flow_checkpoint(
        run_dir / "last.pt", expected_objective=TANGENT_OBJECTIVE
    )
    assert isinstance(model, ConditionalFlowMatcher)
    assert checkpoint_meta["objective_identity"] == identity


# --------------------------------------------------------------------------
# checkpoint round trip and incompatible loads
# --------------------------------------------------------------------------


def test_tangent_checkpoint_cannot_be_loaded_as_an_isotropic_one(tmp_path: Path) -> None:
    from interp.flow_core import ActivationNormalizer, FlowModelConfig

    cfg = FlowModelConfig(
        d_model=4, d_mlp=8, n_blocks=1, time_dim=4, time_hidden=4, max_period=10000.0
    )
    normalizer = ActivationNormalizer(torch.zeros(4), torch.ones(4), 1e-5)
    model = ConditionalFlowMatcher(cfg, ConditionEncoderConfig(cond_hidden=3), normalizer)

    path = tmp_path / "tangent.pt"
    save_flow_checkpoint(
        model, path, metadata={"step": 1}, flow_objective=TANGENT_OBJECTIVE
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["kind"] == "tangent_conditional_flow"
    assert payload["flow_objective"] == TANGENT_OBJECTIVE

    restored, _, _ = load_flow_checkpoint(path, expected_objective=TANGENT_OBJECTIVE)
    assert isinstance(restored, ConditionalFlowMatcher)
    for left, right in zip(
        model.state_dict().values(), restored.state_dict().values(), strict=True
    ):
        assert torch.equal(left, right)

    with pytest.raises(ValueError, match="trained on the .* objective"):
        load_flow_checkpoint(path, expected_objective=ISOTROPIC_OBJECTIVE)


def test_legacy_checkpoints_read_as_isotropic(tmp_path: Path) -> None:
    from interp.flow_core import ActivationNormalizer, FlowMatcher, FlowModelConfig

    cfg = FlowModelConfig(
        d_model=4, d_mlp=8, n_blocks=1, time_dim=4, time_hidden=4, max_period=10000.0
    )
    model = FlowMatcher(cfg, ActivationNormalizer(torch.zeros(4), torch.ones(4), 1e-5))
    path = tmp_path / "legacy.pt"
    save_flow_checkpoint(model, path, metadata={"step": 1})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["flow_objective"]
    path.unlink()
    torch.save(payload, path)

    load_flow_checkpoint(path, expected_objective=ISOTROPIC_OBJECTIVE)
    with pytest.raises(ValueError, match="trained on the 'isotropic' objective"):
        load_flow_checkpoint(path, expected_objective=TANGENT_OBJECTIVE)


def test_unconditional_model_cannot_carry_the_tangent_objective(tmp_path: Path) -> None:
    from interp.flow_core import ActivationNormalizer, FlowMatcher, FlowModelConfig

    cfg = FlowModelConfig(
        d_model=4, d_mlp=8, n_blocks=1, time_dim=4, time_hidden=4, max_period=10000.0
    )
    model = FlowMatcher(cfg, ActivationNormalizer(torch.zeros(4), torch.ones(4), 1e-5))
    with pytest.raises(ValueError, match="requires a conditional flow model"):
        save_flow_checkpoint(
            model, tmp_path / "x.pt", metadata={}, flow_objective=TANGENT_OBJECTIVE
        )


# --------------------------------------------------------------------------
# resume gating
# --------------------------------------------------------------------------


def test_resume_rejects_a_mismatched_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import interp.train_flow as training

    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    cfg = _tangent_config(dataset, pool_path)
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

    checkpoint = run_dir / "step_000002.pt"
    # Same run directory, same everything except the corruption geometry.
    isotropic = replace(
        _conditional_config(dataset, pool_path),
        experiment_id=cfg.experiment_id,
        experiment_class=cfg.experiment_class,
    )
    with pytest.raises(ValueError, match="trained on the .* objective"):
        train_flow(
            dataset,
            isotropic,
            run_dir,
            device="cpu",
            progress=False,
            resume_checkpoint=checkpoint,
        )


def test_resume_rejects_a_mismatched_output_projection_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import interp.train_flow as training

    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    cfg = _tangent_config(dataset, pool_path)
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

    unprojected = replace(
        cfg,
        flow_objective=FlowObjectiveSpec(type=TANGENT_OBJECTIVE, output_projection=False),
    )
    with pytest.raises(ValueError, match="config_fingerprint|objective identity"):
        train_flow(
            dataset,
            unprojected,
            run_dir,
            device="cpu",
            progress=False,
            resume_checkpoint=run_dir / "step_000002.pt",
        )


def test_tangent_resume_reproduces_the_uninterrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import interp.train_flow as training

    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    cfg = _tangent_config(dataset, pool_path)
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    expected = train_flow(dataset, cfg, uninterrupted, device="cpu", progress=False)

    original_save = training.save_flow_checkpoint

    def save_then_interrupt(model, path, *, metadata, training_state=None, **kwargs):  # noqa: ANN001
        original_save(model, path, metadata=metadata, training_state=training_state, **kwargs)
        if path.name == "step_000002.pt":
            raise KeyboardInterrupt("simulated interruption")

    monkeypatch.setattr(training, "save_flow_checkpoint", save_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        train_flow(dataset, cfg, resumed, device="cpu", progress=False)
    monkeypatch.setattr(training, "save_flow_checkpoint", original_save)

    actual = train_flow(
        dataset,
        cfg,
        resumed,
        device="cpu",
        progress=False,
        resume_checkpoint=resumed / "step_000002.pt",
    )
    assert actual["history"] == expected["history"]


# --------------------------------------------------------------------------
# post-stop experiment A: the variance-preserving objective in the same trainer
# --------------------------------------------------------------------------

VP_CONFIG = ROOT / "configs" / "flow_train_tangent_vp_narrow16m_fw32m_v1.yaml"


def _vp_config(dataset: ActivationDataset, pool_path: Path, **overrides):
    from interp.tangent_flow import VP_TANGENT_OBJECTIVE

    return replace(
        _conditional_config(dataset, pool_path),
        experiment_id="vp_tangent_synthetic_test",
        experiment_class="post_stop_method_development",
        flow_objective=FlowObjectiveSpec(
            type=VP_TANGENT_OBJECTIVE, output_projection=True
        ),
        **overrides,
    )


def test_vp_config_differs_from_the_frozen_one_only_in_the_path() -> None:
    """Experiment A changes one field. Everything else must be byte-identical.

    ``status`` is excluded: it is the human authorization gate and legitimately
    moves from ``prepared`` to ``authorized``. It lives in ``raw`` and is outside
    the config fingerprint, so flipping it does not change the scientific
    identity of the run.
    """

    import yaml

    frozen = yaml.safe_load(TANGENT_CONFIG.read_text())
    post_stop = yaml.safe_load(VP_CONFIG.read_text())
    assert post_stop["status"] in {"prepared", "authorized"}
    assert post_stop["flow_objective"]["type"] == "tangent_variance_preserving"
    assert post_stop["flow_objective"]["output_projection"] is True
    for section in ("data", "normalization", "training", "conditioning",
                    "checkpoints", "protected_data"):
        assert post_stop[section] == frozen[section], section
    # validation differs only in the prose note naming the objective
    assert {k: v for k, v in post_stop["validation"].items() if k != "in_run_note"} == {
        k: v for k, v in frozen["validation"].items() if k != "in_run_note"
    }
    assert post_stop["flow_core_config"] == frozen["flow_core_config"]
    # the paired-stream claim of protocol A.4 lives in the seed
    assert post_stop["training"]["noise_seed"] == frozen["training"]["noise_seed"]
    # and it is a new experiment, not an edit of the closed one
    assert post_stop["experiment_id"] != frozen["experiment_id"]
    assert post_stop["experiment_class"] == "post_stop_method_development"


def test_vp_config_loads_and_carries_the_variance_preserving_objective() -> None:
    from interp.tangent_flow import VP_TANGENT_OBJECTIVE

    cfg = load_training_config(VP_CONFIG)
    assert cfg.objective_type == VP_TANGENT_OBJECTIVE
    assert cfg.flow_objective.output_projection is True


def test_vp_config_fingerprint_differs_from_the_linear_tangent_one(tmp_path: Path) -> None:
    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    linear = _config_fingerprint(_used_config(_tangent_config(dataset, pool_path)))
    circle = _config_fingerprint(_used_config(_vp_config(dataset, pool_path)))
    assert linear != circle


def test_vp_trainer_dispatch_produces_variance_preserving_batches(tmp_path: Path) -> None:
    from interp.conditional_flow import load_training_direction_pool
    from interp.tangent_flow import VP_TANGENT_OBJECTIVE

    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    cfg = _vp_config(dataset, pool_path)
    normalizer = _normalizer(dataset, cfg)
    model = ConditionalFlowMatcher(cfg.model, cfg.conditioning.condition, normalizer)
    pool = load_training_direction_pool(pool_path)

    h = torch.randn(8, cfg.model.activation_width)
    batch = _sample_batch(
        model, h, pool=pool, generator=torch.Generator().manual_seed(1),
        objective=cfg.objective_type,
    )
    assert isinstance(batch, TangentFlowBatch)
    assert batch.objective == VP_TANGENT_OBJECTIVE
    # the constraint and the tangency of the target survive the trainer's plumbing
    assert (coordinate(batch.x_t, batch.v_x) - batch.c_x).abs().max() < 1e-4
    assert coordinate(batch.target_velocity, batch.v_x).abs().max() < 1e-4

    projected = _predict(model, batch.x_t, batch.t, batch, output_projection=True)
    assert coordinate(projected, batch.v_x).abs().max() < 1e-4


def test_vp_run_records_its_own_objective_and_cannot_be_read_as_the_linear_one(
    tmp_path: Path,
) -> None:
    from interp.tangent_flow import VP_TANGENT_OBJECTIVE

    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    cfg = _vp_config(dataset, pool_path)
    run_dir = tmp_path / "run"

    metadata = train_flow(dataset, cfg, run_dir, device="cpu", progress=False)
    assert metadata["objective_identity"]["flow_objective"] == VP_TANGENT_OBJECTIVE
    assert metadata["held_out_accessed"] is False

    model, checkpoint_meta, _ = load_flow_checkpoint(
        run_dir / "last.pt", expected_objective=VP_TANGENT_OBJECTIVE
    )
    assert isinstance(model, ConditionalFlowMatcher)
    assert checkpoint_meta["objective_identity"]["flow_objective"] == VP_TANGENT_OBJECTIVE
    with pytest.raises(ValueError, match="was trained on the"):
        load_flow_checkpoint(run_dir / "last.pt", expected_objective=TANGENT_OBJECTIVE)


def test_the_two_objectives_train_on_the_same_randomness(tmp_path: Path) -> None:
    """Protocol A.4's paired-stream claim, asserted through the trainer itself."""

    from interp.conditional_flow import load_training_direction_pool
    from interp.tangent_flow import VP_TANGENT_OBJECTIVE

    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    cfg = _tangent_config(dataset, pool_path)
    normalizer = _normalizer(dataset, cfg)
    model = ConditionalFlowMatcher(cfg.model, cfg.conditioning.condition, normalizer)
    pool = load_training_direction_pool(pool_path)
    h = torch.randn(16, cfg.model.activation_width)

    drawn = {}
    for objective in (TANGENT_OBJECTIVE, VP_TANGENT_OBJECTIVE):
        drawn[objective] = _sample_batch(
            model, h, pool=pool,
            generator=torch.Generator().manual_seed(20260816),
            objective=objective,
        )
    linear, circle = drawn[TANGENT_OBJECTIVE], drawn[VP_TANGENT_OBJECTIVE]
    assert torch.equal(linear.v_x, circle.v_x)
    assert torch.equal(linear.t, circle.t)
    assert torch.equal(linear.epsilon, circle.epsilon)
    assert not torch.allclose(linear.x_t, circle.x_t, atol=1e-2)
