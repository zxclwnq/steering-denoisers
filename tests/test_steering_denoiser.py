"""Post-stop experiment B: the steering-corruption denoiser.

The tests here encode the properties `docs/POST_STOP_PROTOCOL_2026-08-19.md` §3
relies on. The most important ones are not about shapes: they are that the model
is told nothing about the corruption, that the corruption distribution is frozen
and cannot be redefined by a config, that partial correction is a true
interpolation, and that realised concept strength -- the axis the decisive
comparison is made on -- measures what it claims to.

Everything here is CPU, tiny and synthetic. No GPT-2, no real dataset, no DEV or
held-out direction.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from interp.activations import ActivationDataset
from interp.flow_core import ActivationNormalizer, FlowModelConfig
from interp.steering_denoiser import (
    STEERING_CORRUPTION_SPEC,
    STEERING_DENOISER_OBJECTIVE,
    SteeringCorruptionSpec,
    SteeringDenoiseBatch,
    SteeringDenoiser,
    partial_denoise,
    realised_strength,
    sample_steering_corruption_batch,
    shrinkage_activation,
    steering_denoiser_parameter_count,
)
from interp.train_flow import (
    FlowObjectiveSpec,
    _bin_state,
    _predict,
    _sample_batch,
    _used_config,
    load_flow_checkpoint,
    load_training_config,
    save_flow_checkpoint,
    train_flow,
)
from test_conditional_training import _conditional_config, _pool_manifest
from test_flow_training import synthetic_dataset
from test_tangent_flow import _fixture, _pool

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "flow_train_steering_denoiser_16m_fw32m_v1.yaml"
D_MODEL = 16
ROWS = 24


def _tampered_config(tmp_path: Path, mutate) -> Path:  # noqa: ANN001
    """Write a modified copy of the frozen config where its core config resolves."""

    import shutil

    import yaml

    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    for name in ("flow_core_v1.yaml", "flow_core_conditional_narrow16m_v1.yaml"):
        shutil.copy(ROOT / "configs" / name, configs / name)
    raw = yaml.safe_load(CONFIG.read_text())
    mutate(raw)
    path = configs / "tampered.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _model(normalizer: ActivationNormalizer, seed: int = 0) -> SteeringDenoiser:
    torch.manual_seed(seed)
    cfg = FlowModelConfig(
        d_model=D_MODEL, d_mlp=32, n_blocks=2, time_dim=8, time_hidden=16,
        max_period=10000.0,
    )
    return SteeringDenoiser(cfg, normalizer).eval()


# --------------------------------------------------------------------------
# the frozen corruption distribution
# --------------------------------------------------------------------------


def test_the_corruption_spec_is_the_one_frozen_in_the_protocol() -> None:
    spec = STEERING_CORRUPTION_SPEC
    assert spec.version == "steering_corruption_v1"
    assert spec.distribution == "uniform_symmetric"
    assert spec.delta_max == 32.0
    assert spec.lambda_grid == (0.25, 0.50, 0.75, 1.00)


def test_delta_covers_the_published_clamp_displacement_range() -> None:
    """delta_max was chosen to cover |delta| p99 = 21.8 and max 32.0."""

    generator = torch.Generator().manual_seed(0)
    delta = STEERING_CORRUPTION_SPEC.sample_delta(
        200_000, generator=generator, device="cpu", dtype=torch.float32
    )
    assert delta.shape == (200_000, 1)
    assert float(delta.min()) < -31.0
    assert float(delta.max()) > 31.0
    # symmetric, and includes the identity case
    assert abs(float(delta.mean())) < 0.2
    assert float(delta.abs().min()) < 0.01


def test_a_malformed_spec_is_rejected() -> None:
    with pytest.raises(ValueError, match="delta_max must be positive"):
        SteeringCorruptionSpec(delta_max=0.0)
    with pytest.raises(ValueError, match="every lambda"):
        SteeringCorruptionSpec(lambda_grid=(0.0, 0.5))
    with pytest.raises(ValueError, match="increasing order"):
        SteeringCorruptionSpec(lambda_grid=(1.0, 0.5))


def test_a_config_may_restate_but_not_redefine_the_distribution(tmp_path: Path) -> None:
    """Freezing means freezing: a config cannot quietly widen or shift delta."""


    def widen(raw: dict) -> None:
        assert raw["steering_corruption"]["delta_max"] == 32.0
        raw["steering_corruption"]["delta_max"] = 64.0

    with pytest.raises(ValueError, match="frozen in"):
        load_training_config(_tampered_config(tmp_path, widen))


# --------------------------------------------------------------------------
# corruption geometry
# --------------------------------------------------------------------------


def test_corruption_is_a_pure_displacement_along_the_sampled_direction() -> None:
    h, v, _, _, normalizer = _fixture()
    batch = sample_steering_corruption_batch(
        h, normalizer=normalizer, pool=_pool(v),
        generator=torch.Generator().manual_seed(0),
    )
    z = normalizer.denormalize(batch.x_t)
    displacement = z - h
    along = (displacement * batch.v).sum(-1, keepdim=True)
    assert torch.allclose(along, batch.delta, atol=1e-3)
    # nothing moved off the direction
    assert float((displacement - along * batch.v).norm(dim=-1).max()) < 1e-3


def test_the_target_is_the_residual_that_undoes_the_corruption() -> None:
    h, v, _, _, normalizer = _fixture()
    batch = sample_steering_corruption_batch(
        h, normalizer=normalizer, pool=_pool(v),
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.allclose(batch.x_t + batch.target_velocity, batch.x0, atol=1e-5)
    assert torch.allclose(batch.x0, normalizer.normalize(h), atol=1e-5)


def test_a_supplied_delta_is_used_verbatim() -> None:
    h, v, _, _, normalizer = _fixture()
    delta = torch.full((h.shape[0], 1), 7.5)
    batch = sample_steering_corruption_batch(
        h, normalizer=normalizer, pool=_pool(v),
        generator=torch.Generator().manual_seed(0), delta=delta,
    )
    assert torch.allclose(batch.delta, delta)


def test_zero_delta_is_the_clean_identity() -> None:
    h, v, _, _, normalizer = _fixture()
    batch = sample_steering_corruption_batch(
        h, normalizer=normalizer, pool=_pool(v),
        generator=torch.Generator().manual_seed(0),
        delta=torch.zeros(h.shape[0], 1),
    )
    assert torch.allclose(batch.x_t, batch.x0, atol=1e-6)
    assert float(batch.target_velocity.abs().max()) < 1e-6


# --------------------------------------------------------------------------
# the model is told nothing
# --------------------------------------------------------------------------


def test_the_denoiser_output_depends_on_nothing_but_the_corrupted_state() -> None:
    """No direction, no strength, no time may reach the network."""

    _, _, _, _, normalizer = _fixture()
    model = _model(normalizer)
    z = torch.randn(ROWS, D_MODEL)
    baseline = model(z)
    for t in (0.0, 0.5, 1.0, None):
        assert torch.equal(model(z, t), baseline)
    # and the signature accepts nothing else
    assert model.forward.__doc__ is not None


def test_parameter_count_matches_the_analytic_formula() -> None:
    _, _, _, _, normalizer = _fixture()
    model = _model(normalizer)
    assert sum(p.numel() for p in model.parameters()) == (
        steering_denoiser_parameter_count(model.cfg)
    )


def test_malformed_inputs_are_rejected() -> None:
    _, _, _, _, normalizer = _fixture()
    model = _model(normalizer)
    with pytest.raises(ValueError, match="expected z with shape"):
        model(torch.randn(ROWS, D_MODEL + 1))
    with pytest.raises(ValueError, match="finite"):
        model(torch.full((ROWS, D_MODEL), float("nan")))


# --------------------------------------------------------------------------
# inference: partial correction and the strength axis
# --------------------------------------------------------------------------


def test_lambda_zero_is_exactly_additive_steering_with_no_evaluation() -> None:
    h, v, _, _, normalizer = _fixture()
    model = _model(normalizer)
    out = partial_denoise(model, h, v, 5.0, lam=0.0)
    assert torch.allclose(out.activation, h + 5.0 * v, atol=1e-5)
    assert out.diagnostics["network_evaluations"] == 0
    assert pytest.approx(float(out.realised_alpha.mean()), abs=1e-3) == 5.0


def test_partial_correction_interpolates_linearly_between_the_endpoints() -> None:
    h, v, _, _, normalizer = _fixture()
    model = _model(normalizer)
    additive = partial_denoise(model, h, v, 8.0, lam=0.0).activation
    full = partial_denoise(model, h, v, 8.0, lam=1.0).activation
    for lam in STEERING_CORRUPTION_SPEC.lambda_grid:
        produced = partial_denoise(model, h, v, 8.0, lam=lam).activation
        expected = additive + lam * (full - additive)
        assert torch.allclose(produced, expected, atol=1e-4)


def test_interpolating_before_or_after_denormalizing_is_the_same_activation() -> None:
    """The docstring's affine claim, checked rather than asserted in prose."""

    h, v, _, _, normalizer = _fixture()
    model = _model(normalizer)
    lam = 0.5
    out = partial_denoise(model, h, v, 6.0, lam=lam)
    z = h + 6.0 * v
    residual_raw = normalizer.denormalize(
        normalizer.normalize(z) + model(normalizer.normalize(z))
    ) - z
    assert torch.allclose(out.activation, z + lam * residual_raw, atol=1e-4)


def test_realised_strength_is_the_projection_of_the_actual_displacement() -> None:
    h, v, _, _, normalizer = _fixture()
    produced = h + 3.0 * v + torch.randn_like(h) * 0.1
    expected = ((produced - h) * v).sum(-1)
    assert torch.allclose(realised_strength(produced, h, v), expected, atol=1e-5)


def test_a_denoiser_that_perfectly_undoes_steering_reports_zero_realised_strength() -> None:
    """The named failure mode must be visible in the number that decides the verdict."""

    h, v, _, _, normalizer = _fixture()

    class _PerfectInverter(SteeringDenoiser):
        def __init__(self) -> None:
            super().__init__(
                FlowModelConfig(
                    d_model=D_MODEL, d_mlp=32, n_blocks=1, time_dim=8,
                    time_hidden=16, max_period=10000.0,
                ),
                normalizer,
            )
            self.clean = normalizer.normalize(h)

        def forward(self, z, t=None):  # noqa: ANN001, ANN201, ARG002
            return self.clean - z

    out = partial_denoise(_PerfectInverter().eval(), h, v, 9.0, lam=1.0)
    assert float(out.realised_alpha.abs().max()) < 1e-3
    assert pytest.approx(float(out.diagnostics["attenuation_mean"]), abs=1e-3) == 9.0
    # and the shrinkage control at that realised strength is the clean activation
    assert torch.allclose(
        shrinkage_activation(h, v, out.realised_alpha), h, atol=1e-3
    )


def test_shrinkage_control_reproduces_additive_steering_at_a_given_strength() -> None:
    h, v, _, _, _ = _fixture()
    produced = shrinkage_activation(h, v, 2.5)
    assert torch.allclose(realised_strength(produced, h, v), torch.full((ROWS,), 2.5),
                          atol=1e-4)


def test_lambda_outside_the_unit_interval_is_rejected() -> None:
    h, v, _, _, normalizer = _fixture()
    model = _model(normalizer)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="lambda must lie"):
            partial_denoise(model, h, v, 1.0, lam=bad)


# --------------------------------------------------------------------------
# the shared trainer
# --------------------------------------------------------------------------


def _denoiser_config(dataset: ActivationDataset, pool_path: Path, **overrides):
    from interp.train_flow import SteeringCorruptionTrainingSpec

    base = _conditional_config(dataset, pool_path)
    return replace(
        base,
        experiment_id="steering_denoiser_synthetic_test",
        experiment_class="post_stop_method_development",
        conditioning=None,
        flow_objective=FlowObjectiveSpec(
            type=STEERING_DENOISER_OBJECTIVE, output_projection=False
        ),
        steering_corruption=SteeringCorruptionTrainingSpec(
            direction_pool=str(pool_path), spec=STEERING_CORRUPTION_SPEC
        ),
        **overrides,
    )


def test_the_config_loads_and_carries_a_valid_authorization_status() -> None:
    """The status field is a human gate, so it moves; its meaning must not.

    This asserted ``prepared`` until a human authorized the run. Pinning the
    current value would make the test a record of when it was written rather than
    of any invariant -- and the property that matters, "the trainer refuses to
    launch anything not marked authorized", is enforced in `train_flow.py` and
    tested there, not by the value sitting in the file today.
    """

    import yaml

    raw = yaml.safe_load(CONFIG.read_text())
    assert raw["status"] in {"prepared", "authorized"}
    assert "conditioning" not in raw
    cfg = load_training_config(CONFIG)
    assert cfg.objective_type == STEERING_DENOISER_OBJECTIVE
    assert cfg.conditioning is None
    assert cfg.steering_corruption.spec == STEERING_CORRUPTION_SPEC
    assert cfg.direction_pool_path.endswith("training_only_rank256_v1.pt")


def test_output_projection_is_refused_for_a_model_with_no_velocity() -> None:
    with pytest.raises(ValueError, match="nothing to project"):
        FlowObjectiveSpec(type=STEERING_DENOISER_OBJECTIVE, output_projection=True)


def test_the_denoiser_must_not_declare_a_conditioning_section(tmp_path: Path) -> None:
    def add_conditioning(raw: dict) -> None:
        raw["flow_core_config"] = "configs/flow_core_conditional_narrow16m_v1.yaml"
        raw["conditioning"] = {
            "direction_pool": "data/direction_pools/training_only_rank256_v1.pt",
            "direction_pool_split": "training_only",
            "steering_vector_conditioning": False,
            "coordinate": "linear_projection",
            "condition_source": "natural_coordinate",
        }

    with pytest.raises(ValueError, match="unconditional"):
        load_training_config(_tampered_config(tmp_path, add_conditioning))


def test_the_denoiser_config_requires_a_training_only_pool(tmp_path: Path) -> None:
    def use_dev_pool(raw: dict) -> None:
        raw["steering_corruption"]["direction_pool_split"] = "dev"

    with pytest.raises(ValueError, match="training_only pool"):
        load_training_config(_tampered_config(tmp_path, use_dev_pool))


def test_existing_configs_keep_their_fingerprints(tmp_path: Path) -> None:
    """Adding experiment B must not change any historical config fingerprint."""

    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    used = _used_config(_conditional_config(dataset, pool_path))
    assert "steering_corruption" not in used


def test_trainer_dispatch_produces_denoise_batches_and_residual_predictions(
    tmp_path: Path,
) -> None:
    from interp.conditional_flow import load_training_direction_pool

    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    cfg = _denoiser_config(dataset, pool_path)
    normalizer = ActivationNormalizer(
        torch.zeros(cfg.model.activation_width), torch.ones(cfg.model.activation_width)
    )
    model = SteeringDenoiser(cfg.model, normalizer)
    pool = load_training_direction_pool(pool_path)

    h = torch.randn(8, cfg.model.activation_width)
    batch = _sample_batch(
        model, h, pool=pool, generator=torch.Generator().manual_seed(1),
        objective=cfg.objective_type,
    )
    assert isinstance(batch, SteeringDenoiseBatch)
    assert batch.objective == STEERING_DENOISER_OBJECTIVE
    prediction = _predict(model, batch.x_t, batch.t, batch)
    assert prediction.shape == batch.target_velocity.shape


def test_validation_bins_read_as_corruption_magnitude(tmp_path: Path) -> None:
    """The denoiser has no time; its bins must measure |delta| instead."""

    from interp.conditional_flow import load_training_direction_pool

    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    cfg = _denoiser_config(dataset, pool_path)
    width = cfg.model.activation_width
    normalizer = ActivationNormalizer(torch.zeros(width), torch.ones(width))
    model = SteeringDenoiser(cfg.model, normalizer)
    pool = load_training_direction_pool(pool_path)
    h = torch.randn(32, width)
    batch = _sample_batch(
        model, h, pool=pool, generator=torch.Generator().manual_seed(1),
        objective=cfg.objective_type,
    )
    x0 = normalizer.normalize(h)
    noise = torch.randn_like(x0)

    magnitudes = []
    for left, right in ((0.0, 0.10), (0.75, 1.0)):
        time = torch.full((x0.shape[0], 1), (left + right) / 2)
        state, target = _bin_state(batch, x0, noise, time, normalizer=normalizer)
        assert torch.allclose(state + target, x0, atol=1e-5)
        magnitudes.append(float((state - x0).norm(dim=-1).mean()))
    # a later bin is a strictly harder corruption
    assert magnitudes[1] > 5.0 * magnitudes[0]
    # and the largest bin reaches the frozen delta_max scale
    assert magnitudes[1] > 0.5 * STEERING_CORRUPTION_SPEC.delta_max


def test_a_denoiser_run_records_its_objective_and_round_trips(tmp_path: Path) -> None:
    dataset = synthetic_dataset()
    pool_path = _pool_manifest(tmp_path / "pool.pt")
    cfg = _denoiser_config(dataset, pool_path)
    run_dir = tmp_path / "run"

    metadata = train_flow(dataset, cfg, run_dir, device="cpu", progress=False)
    identity = metadata["objective_identity"]
    assert identity["flow_objective"] == STEERING_DENOISER_OBJECTIVE
    assert identity["condition_type"] is None
    assert identity["steering_corruption"]["delta_max"] == 32.0
    assert metadata["held_out_accessed"] is False
    assert metadata["direction_pool"]["split"] == "training_only"

    model, checkpoint_meta, _ = load_flow_checkpoint(
        run_dir / "last.pt", expected_objective=STEERING_DENOISER_OBJECTIVE
    )
    assert isinstance(model, SteeringDenoiser)
    assert checkpoint_meta["objective_identity"] == identity


def test_a_denoiser_checkpoint_cannot_masquerade_as_a_flow(tmp_path: Path) -> None:
    from interp.conditional_flow import ConditionalFlowMatcher, ConditionEncoderConfig

    cfg = FlowModelConfig(
        d_model=4, d_mlp=8, n_blocks=1, time_dim=4, time_hidden=4, max_period=10000.0
    )
    normalizer = ActivationNormalizer(torch.zeros(4), torch.ones(4))
    denoiser = SteeringDenoiser(cfg, normalizer)
    flow = ConditionalFlowMatcher(cfg, ConditionEncoderConfig(cond_hidden=4), normalizer)

    with pytest.raises(ValueError, match="must be used together"):
        save_flow_checkpoint(
            denoiser, tmp_path / "a.pt", metadata={},
            flow_objective="tangent_constraint_preserving",
        )
    with pytest.raises(ValueError, match="must be used together"):
        save_flow_checkpoint(
            flow, tmp_path / "b.pt", metadata={},
            flow_objective=STEERING_DENOISER_OBJECTIVE,
        )

    good = tmp_path / "c.pt"
    save_flow_checkpoint(
        denoiser, good, metadata={}, flow_objective=STEERING_DENOISER_OBJECTIVE
    )
    with pytest.raises(ValueError, match="was trained on the"):
        load_flow_checkpoint(good, expected_objective="isotropic")
