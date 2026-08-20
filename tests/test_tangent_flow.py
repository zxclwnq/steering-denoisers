"""T0 geometry invariants for the constraint-preserving tangent flow.

Every test here encodes a property the T1/T2 experiments depend on. The central
claims are that ``<x_t, v> = c`` for every ``t``, that the velocity target and
the used velocity are exactly tangent, and that the semantic coordinate survives
Euler integration without the numerical safeguard doing the work.
"""

from __future__ import annotations

import pytest
import torch

from interp.conditional_flow import (
    ConditionalFlowMatcher,
    ConditionEncoderConfig,
    DirectionPoolProvenance,
    TrainingDirectionPool,
    _digest_fields,
    _pool_digest,
    _unit_rows,
    clamp_seed,
    standardized_hyperplane,
)
from interp.constrained_flow import constrained_partial_noise
from interp.flow_core import ActivationNormalizer, FlowModelConfig
from interp.tangent_flow import (
    ISOTROPIC_OBJECTIVE,
    TANGENT_OBJECTIVE,
    clamp_then_tangent_flow,
    coordinate,
    raw_velocity_field,
    sample_tangent_flow_batch,
    tangent_flow_states,
    tangent_project,
    tangent_reverse_euler,
    unit_directions,
)

D_MODEL = 16
ROWS = 24
TIMES = (0.0, 0.01, 0.1, 0.25, 0.5, 0.9, 1.0)
TOL = 1e-4


def _fixture(seed: int = 20260816, *, real_scale: bool = True):  # noqa: ANN202
    generator = torch.Generator().manual_seed(seed)
    h = torch.randn(ROWS, D_MODEL, generator=generator) * 3.0
    v = torch.randn(ROWS, D_MODEL, generator=generator)
    v = v / v.norm(dim=-1, keepdim=True)
    noise = torch.randn(ROWS, D_MODEL, generator=generator)
    c_target = torch.randn(ROWS, 1, generator=generator) * 4.0
    if real_scale:
        mean = torch.randn(D_MODEL, generator=generator)
        std = torch.rand(D_MODEL, generator=generator) + 0.5
    else:
        mean, std = torch.zeros(D_MODEL), torch.ones(D_MODEL)
    return h, v, noise, c_target, ActivationNormalizer(mean, std, eps=1e-5)


def _pool(directions: torch.Tensor) -> TrainingDirectionPool:
    ranks = tuple(range(300, 300 + directions.shape[0]))
    fields = {
        "split": "training_only",
        "source": "synthetic_t0",
        "selection": "unit_test",
        "selection_seed": 0,
        "min_rank": 256,
        "excluded_splits": ["dev", "held_out"],
        "n_directions": len(ranks),
    }
    provenance = DirectionPoolProvenance(
        split="training_only",
        source="synthetic_t0",
        selection="unit_test",
        selection_seed=0,
        min_rank=256,
        excluded_splits=("dev", "held_out"),
        n_directions=len(ranks),
        digest=_pool_digest(directions, ranks, fields),
    )
    assert _digest_fields(provenance) == fields
    return TrainingDirectionPool(directions, ranks, provenance)


def _model(normalizer: ActivationNormalizer, seed: int = 0) -> ConditionalFlowMatcher:
    torch.manual_seed(seed)
    cfg = FlowModelConfig(
        d_model=D_MODEL, d_mlp=32, n_blocks=2, time_dim=8, time_hidden=16, max_period=10000.0
    )
    return ConditionalFlowMatcher(cfg, ConditionEncoderConfig(cond_hidden=8), normalizer).eval()


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


@pytest.mark.parametrize("t", TIMES)
def test_tangent_path_preserves_the_coordinate_at_every_time(t: float) -> None:
    h, v, noise, _, normalizer = _fixture()
    x0 = normalizer.normalize(h)
    v_x, c_x = standardized_hyperplane(normalizer, v, (h * v).sum(-1, keepdim=True))
    _, x_t, _ = tangent_flow_states(x0, v_x, c_x, noise, t)
    assert torch.allclose(coordinate(x_t, v_x), c_x, atol=TOL)


def test_tangent_noise_is_orthogonal_to_the_direction() -> None:
    h, v, noise, _, normalizer = _fixture()
    x0 = normalizer.normalize(h)
    v_x, c_x = standardized_hyperplane(normalizer, v, (h * v).sum(-1, keepdim=True))
    epsilon_perp, _, _ = tangent_flow_states(x0, v_x, c_x, noise, 0.5)
    assert coordinate(epsilon_perp, v_x).abs().max() < TOL
    # and it is a genuine projection, not a rescaling to zero
    assert epsilon_perp.norm(dim=-1).min() > 0.0


def test_velocity_target_is_tangent() -> None:
    h, v, noise, _, normalizer = _fixture()
    x0 = normalizer.normalize(h)
    v_x, c_x = standardized_hyperplane(normalizer, v, (h * v).sum(-1, keepdim=True))
    _, _, target = tangent_flow_states(x0, v_x, c_x, noise, 0.3)
    assert coordinate(target, v_x).abs().max() < TOL


def test_tangent_endpoints_are_the_clean_and_projected_noise_states() -> None:
    h, v, noise, _, normalizer = _fixture()
    x0 = normalizer.normalize(h)
    v_x, c_x = standardized_hyperplane(normalizer, v, (h * v).sum(-1, keepdim=True))
    eps_perp, at_zero, _ = tangent_flow_states(x0, v_x, c_x, noise, 0.0)
    _, at_one, _ = tangent_flow_states(x0, v_x, c_x, noise, 1.0)
    assert torch.allclose(at_zero, x0, atol=TOL)
    assert torch.allclose(at_one, eps_perp + c_x * v_x, atol=TOL)


@pytest.mark.parametrize("t", TIMES[1:])
def test_tangent_path_matches_the_independent_constrained_implementation(t: float) -> None:
    """Cross-check against ``constrained_flow``, derived and tested separately."""

    h, v, noise, _, normalizer = _fixture()
    x0 = normalizer.normalize(h)
    v_x, c_x = standardized_hyperplane(normalizer, v, (h * v).sum(-1, keepdim=True))
    _, x_t, _ = tangent_flow_states(x0, v_x, c_x, noise, t)
    assert torch.allclose(
        x_t, constrained_partial_noise(x0, noise, v_x, c_x, t_start=t), atol=TOL
    )


def test_tangent_projection_removes_exactly_the_parallel_part() -> None:
    _, v, noise, _, normalizer = _fixture()
    v_x, _ = standardized_hyperplane(normalizer, v, torch.zeros(ROWS, 1))
    projected = tangent_project(noise, v_x)
    assert coordinate(projected, v_x).abs().max() < TOL
    assert torch.allclose(noise - projected, coordinate(noise, v_x) * v_x, atol=TOL)


def test_sign_invariance_of_the_hyperplane_representation() -> None:
    """``(v, c)`` and ``(-v, -c)`` are the same hyperplane and must match exactly."""

    h, v, noise, c_target, normalizer = _fixture()
    x0 = normalizer.normalize(h)
    forward = standardized_hyperplane(normalizer, v, c_target)
    flipped = standardized_hyperplane(normalizer, -v, -c_target)
    assert torch.allclose(forward[0], flipped[0], atol=TOL)
    assert torch.allclose(forward[1], flipped[1], atol=TOL)
    a = tangent_flow_states(x0, *forward, noise, 0.4)
    b = tangent_flow_states(x0, *flipped, noise, 0.4)
    for left, right in zip(a, b, strict=True):
        assert torch.allclose(left, right, atol=TOL)


def test_raw_clamp_coordinate_equals_the_standardized_hyperplane() -> None:
    h, v, _, c_target, normalizer = _fixture()
    seed = clamp_seed(h, v, c_target)
    assert torch.allclose((seed * v).sum(-1, keepdim=True), c_target, atol=1e-3)
    v_x, c_x = standardized_hyperplane(normalizer, v, c_target)
    assert torch.allclose(coordinate(normalizer.normalize(seed), v_x), c_x, atol=TOL)


# --------------------------------------------------------------------------
# batch sampler
# --------------------------------------------------------------------------


def test_sampled_batch_satisfies_every_declared_invariant() -> None:
    h, v, _, _, normalizer = _fixture()
    pool = _pool(v)
    batch = sample_tangent_flow_batch(
        h,
        normalizer=normalizer,
        pool=pool,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.allclose(coordinate(batch.x0, batch.v_x), batch.c_x, atol=TOL)
    assert torch.allclose(coordinate(batch.x_t, batch.v_x), batch.c_x, atol=TOL)
    assert coordinate(batch.epsilon_perp, batch.v_x).abs().max() < TOL
    assert coordinate(batch.target_velocity, batch.v_x).abs().max() < TOL
    assert batch.t.shape == (ROWS, 1)
    assert bool(((batch.t >= 0.0) & (batch.t <= 1.0)).all())
    # different rows really do get different directions and coordinates
    assert len(torch.unique(batch.c_x)) > 1


def test_batch_semantics_are_per_row() -> None:
    """A per-row direction must give a per-row constraint, not a batch-wide one."""

    h, v, _, _, normalizer = _fixture()
    batch = sample_tangent_flow_batch(
        h, normalizer=normalizer, pool=_pool(v),
        generator=torch.Generator().manual_seed(11),
    )
    for row in range(ROWS):
        assert torch.allclose(
            (batch.x_t[row] * batch.v_x[row]).sum(), batch.c_x[row, 0], atol=TOL
        )


def test_sampler_refuses_a_device_mismatched_pool() -> None:
    h, v, _, _, normalizer = _fixture()
    pool = _pool(v).to(dtype=torch.float64)
    with pytest.raises(ValueError, match="direction pool is on"):
        sample_tangent_flow_batch(
            h, normalizer=normalizer, pool=pool,
            generator=torch.Generator().manual_seed(0),
        )


# --------------------------------------------------------------------------
# model output and integration
# --------------------------------------------------------------------------


def test_projected_model_velocity_is_tangent_while_the_raw_one_is_not() -> None:
    h, v, noise, _, normalizer = _fixture()
    model = _model(normalizer)
    x0 = normalizer.normalize(h)
    v_x, c_x = standardized_hyperplane(normalizer, v, (h * v).sum(-1, keepdim=True))
    _, x_t, _ = tangent_flow_states(x0, v_x, c_x, noise, 0.5)
    t = torch.full((ROWS, 1), 0.5)
    raw = raw_velocity_field(model, v_x, c_x)(x_t, t)
    projected = tangent_project(raw, v_x)
    assert coordinate(projected, v_x).abs().max() < TOL
    # the untrained network has no reason to be tangent on its own; if it were,
    # the projection test above would be vacuous
    assert coordinate(raw, v_x).abs().max() > TOL


@pytest.mark.parametrize("nfe", [1, 3, 5])
def test_coordinate_is_fixed_across_every_euler_step(nfe: int) -> None:
    h, v, noise, _, normalizer = _fixture()
    model = _model(normalizer)
    x0 = normalizer.normalize(h)
    v_x, c_x = standardized_hyperplane(normalizer, v, (h * v).sum(-1, keepdim=True))
    _, x_t, _ = tangent_flow_states(x0, v_x, c_x, noise, 0.5)
    _, stats = tangent_reverse_euler(
        raw_velocity_field(model, v_x, c_x),
        x_t, v_x, c_x, t_start=0.5, nfe=nfe, safeguard_projection=False,
    )
    # No safeguard at all: tangency alone holds the coordinate.
    assert stats["max_coordinate_drift"] < 1e-4
    assert stats["network_evaluations"] == nfe
    assert stats["projections"] == 0


def test_safeguard_only_corrects_float_drift_and_is_never_an_evaluation() -> None:
    h, v, noise, _, normalizer = _fixture()
    model = _model(normalizer)
    x0 = normalizer.normalize(h)
    v_x, c_x = standardized_hyperplane(normalizer, v, (h * v).sum(-1, keepdim=True))
    _, x_t, _ = tangent_flow_states(x0, v_x, c_x, noise, 0.5)
    guarded, stats = tangent_reverse_euler(
        raw_velocity_field(model, v_x, c_x),
        x_t, v_x, c_x, t_start=0.5, nfe=3, safeguard_projection=True,
    )
    bare, _ = tangent_reverse_euler(
        raw_velocity_field(model, v_x, c_x),
        x_t, v_x, c_x, t_start=0.5, nfe=3, safeguard_projection=False,
    )
    assert stats["network_evaluations"] == 3
    assert stats["projections"] == 3
    # The safeguard corrects float error only: it must not change the answer.
    assert stats["max_pre_projection_drift"] < 1e-4
    assert torch.allclose(guarded, bare, atol=1e-4)


# --------------------------------------------------------------------------
# clamp + tangent flow
# --------------------------------------------------------------------------


def test_t_start_zero_is_the_exact_hard_clamp_with_zero_evaluations() -> None:
    h, v, noise, c_target, normalizer = _fixture()
    model = _model(normalizer)
    out = clamp_then_tangent_flow(
        model, h, v, c_target, noise=noise, t_start=0.0, nfe=3
    )
    assert out.network_evaluations == 0
    assert out.projections == 0
    assert torch.equal(out.activation, out.seed)
    assert torch.allclose(out.activation, clamp_seed(h, v, c_target))
    assert torch.allclose(out.realised_coordinate, c_target.squeeze(-1), atol=1e-3)
    assert out.diagnostics["orthogonal_correction_norm_mean"] == 0.0


@pytest.mark.parametrize(("t_start", "nfe"), [(0.10, 1), (0.25, 3), (0.50, 1)])
def test_clamp_plus_tangent_flow_holds_the_requested_coordinate(
    t_start: float, nfe: int
) -> None:
    h, v, noise, c_target, normalizer = _fixture()
    model = _model(normalizer)
    out = clamp_then_tangent_flow(
        model, h, v, c_target, noise=noise, t_start=t_start, nfe=nfe
    )
    assert out.network_evaluations == nfe
    assert out.projections == nfe
    assert torch.allclose(out.realised_coordinate, c_target.squeeze(-1), atol=1e-2)
    assert float(out.diagnostics["coordinate_abs_error_max"]) < 1e-2
    # the flow must actually do something orthogonal, or the arm is vacuous
    assert float(out.diagnostics["orthogonal_correction_norm_mean"]) > 0.0
    # and the raw network velocity really does have a parallel part that was
    # discarded rather than applied.
    #
    # The bound is 1e-2, not > 0. An earlier version of this test asserted
    # > 0.0 and passed while the diagnostic was measuring an already-projected
    # vector: float residue of ~1e-8 is still "> 0". A floor six orders of
    # magnitude above that residue is what makes this test able to fail.
    assert float(out.diagnostics["raw_parallel_velocity_norm_mean"]) > 1e-2


class _KnownParallelVelocity(torch.nn.Module):
    """A model whose output has a deliberate, known component along ``v_x``.

    ``u_raw = base_perp + magnitude * v_x``, so the correct raw diagnostic is
    exactly ``magnitude`` and the correct used velocity is exactly ``base_perp``.
    """

    def __init__(self, normalizer: ActivationNormalizer, magnitude: float) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.magnitude = magnitude
        self.calls = 0

    def forward(  # noqa: D102
        self, x: torch.Tensor, t: torch.Tensor, v_x: torch.Tensor, c_x: torch.Tensor  # noqa: ARG002
    ) -> torch.Tensor:
        self.calls += 1
        base = torch.full_like(x, 0.05)
        return tangent_project(base, v_x) + self.magnitude * v_x


def test_raw_parallel_diagnostic_sees_the_unprojected_model_output() -> None:
    """P6 regression: the diagnostic must describe the model, not the projection."""

    h, v, noise, c_target, normalizer = _fixture()
    magnitude = 0.75
    model = _KnownParallelVelocity(normalizer, magnitude)

    out = clamp_then_tangent_flow(
        model, h, v, c_target, noise=noise, t_start=0.5, nfe=2
    )
    assert model.calls == 2
    # The raw diagnostic recovers the injected parallel component exactly...
    assert float(out.diagnostics["raw_parallel_velocity_norm_mean"]) == pytest.approx(
        magnitude, rel=1e-4
    )
    # ...and none of it reached the activation: the coordinate is untouched.
    assert float(out.diagnostics["max_pre_projection_drift"]) < 1e-4
    assert torch.allclose(out.realised_coordinate, c_target.squeeze(-1), atol=1e-2)


def test_used_velocity_is_tangent_even_when_the_model_is_maximally_parallel() -> None:
    h, v, noise, c_target, normalizer = _fixture()
    model = _KnownParallelVelocity(normalizer, magnitude=50.0)

    out = clamp_then_tangent_flow(
        model, h, v, c_target, noise=noise, t_start=0.5, nfe=3
    )
    # A velocity 50 units along v would wreck the coordinate if it were applied.
    assert float(out.diagnostics["raw_parallel_velocity_norm_mean"]) == pytest.approx(
        50.0, rel=1e-4
    )
    assert float(out.diagnostics["max_pre_projection_drift"]) < 1e-3
    assert float(out.diagnostics["coordinate_abs_error_max"]) < 1e-2


def test_tangent_inference_has_no_output_projection_switch() -> None:
    """P7 regression: a fake ablation switch is worse than none."""

    import inspect

    signature = inspect.signature(clamp_then_tangent_flow)
    assert "output_projection" not in signature.parameters
    assert "output_projection" not in inspect.signature(raw_velocity_field).parameters

    h, v, noise, c_target, normalizer = _fixture()
    out = clamp_then_tangent_flow(
        model=_model(normalizer), h=h, direction=v, c_target=c_target,
        noise=noise, t_start=0.5, nfe=1,
    )
    assert out.diagnostics["inference_output_projection"] == "always"


# --------------------------------------------------------------------------
# direction normalization (P14)
# --------------------------------------------------------------------------


def test_non_unit_directions_are_rejected_at_the_tangent_boundary() -> None:
    h, v, noise, c_target, normalizer = _fixture()
    model = _model(normalizer)
    # 1e-4 passes the shared 1e-3 guard but is 100x looser than the tangent bound.
    slightly_off = v * 1.0001
    assert _unit_rows(slightly_off) is not None  # the shared guard accepts it

    with pytest.raises(ValueError, match="unit directions to within"):
        unit_directions(slightly_off)
    with pytest.raises(ValueError, match="unit directions to within"):
        clamp_then_tangent_flow(
            model, h, slightly_off, c_target, noise=noise, t_start=0.5, nfe=1
        )


def test_a_loose_direction_would_have_missed_the_requested_coordinate() -> None:
    """Why the tolerance was tightened, stated as an executable fact."""

    h, v, _, _, _ = _fixture()
    loose = v * 1.0009  # inside the shared guard's 1e-3 limit
    assert _unit_rows(loose) is not None
    c_target = (h * v).sum(-1, keepdim=True) + 10.0
    landed = clamp_seed(h, loose, c_target)
    error = ((landed * loose).sum(-1, keepdim=True) - c_target).abs().max()
    # ~0.018: eighteen times the 1e-3 tolerance the T2 arm-matching gate allows,
    # from a direction the shared guard happily accepts.
    assert float(error) > 1e-2
    assert unit_directions(v) is not None


def test_clamp_plus_tangent_flow_is_sign_invariant() -> None:
    h, v, noise, c_target, normalizer = _fixture()
    model = _model(normalizer)
    forward = clamp_then_tangent_flow(
        model, h, v, c_target, noise=noise, t_start=0.25, nfe=3
    )
    flipped = clamp_then_tangent_flow(
        model, h, -v, -c_target, noise=noise, t_start=0.25, nfe=3
    )
    assert torch.allclose(forward.seed, flipped.seed, atol=1e-3)
    assert torch.allclose(forward.activation, flipped.activation, atol=1e-3)


def test_clamp_plus_tangent_flow_matches_the_clamp_along_the_direction_only() -> None:
    """The flow may move the orthogonal part; it may not move the coordinate."""

    h, v, noise, c_target, normalizer = _fixture()
    model = _model(normalizer)
    out = clamp_then_tangent_flow(
        model, h, v, c_target, noise=noise, t_start=0.5, nfe=3
    )
    delta = out.activation - out.seed
    parallel = (delta * v).sum(-1, keepdim=True)
    assert parallel.abs().max() < 1e-2
    assert (delta - parallel * v).norm(dim=-1).mean() > 0.0


def test_nonfinite_and_shape_inputs_are_rejected() -> None:
    h, v, noise, c_target, normalizer = _fixture()
    model = _model(normalizer)
    bad = h.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="h must be finite"):
        clamp_then_tangent_flow(model, bad, v, c_target, noise=noise, t_start=0.5, nfe=1)
    with pytest.raises(ValueError, match="noise must be"):
        clamp_then_tangent_flow(
            model, h, v, c_target, noise=noise[:2], t_start=0.5, nfe=1
        )


def test_objective_identifiers_are_distinct() -> None:
    assert ISOTROPIC_OBJECTIVE != TANGENT_OBJECTIVE
