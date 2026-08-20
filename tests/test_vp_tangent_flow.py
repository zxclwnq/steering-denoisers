"""Geometry invariants for the variance-preserving tangent path (post-stop A).

Every test encodes a property `docs/POST_STOP_PROTOCOL_2026-08-19.md` §2 relies
on. The central claims are that the quarter-circle path keeps ``<x_t, v> = c``
exactly, that its velocity target is the true time derivative and is tangent,
that it does **not** shrink the orthogonal scale the way the chord does, and
that the matched-severity map makes the two paths comparable at equal
noise-to-signal rather than equal ``t``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from interp.conditional_flow import ConditionalFlowMatcher, ConditionEncoderConfig
from interp.flow_core import ActivationNormalizer, FlowModelConfig
from interp.tangent_flow import (
    TANGENT_OBJECTIVE,
    VP_TANGENT_OBJECTIVE,
    clamp_then_tangent_flow,
    coordinate,
    matched_linear_time,
    matched_vp_time,
    sample_tangent_flow_batch,
    tangent_flow_states,
    tangent_path_states,
    vp_tangent_flow_states,
)
from test_tangent_flow import _fixture, _pool

TIMES = (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
TOL = 1e-4


def _hyperplane(h: torch.Tensor, v: torch.Tensor, normalizer: ActivationNormalizer):  # noqa: ANN202
    from interp.conditional_flow import standardized_hyperplane

    return standardized_hyperplane(normalizer, v, (h * v).sum(-1, keepdim=True))


# --------------------------------------------------------------------------
# the two invariants the constraint depends on
# --------------------------------------------------------------------------


@pytest.mark.parametrize("t", TIMES)
def test_vp_path_preserves_the_coordinate_at_every_time(t: float) -> None:
    h, v, noise, _, normalizer = _fixture()
    x0 = normalizer.normalize(h)
    v_x, c_x = _hyperplane(h, v, normalizer)
    _, x_t, _ = vp_tangent_flow_states(x0, v_x, c_x, noise, t)
    assert torch.allclose(coordinate(x_t, v_x), c_x, atol=TOL)


@pytest.mark.parametrize("t", TIMES)
def test_vp_velocity_target_is_tangent(t: float) -> None:
    h, v, noise, _, normalizer = _fixture()
    x0 = normalizer.normalize(h)
    v_x, c_x = _hyperplane(h, v, normalizer)
    _, _, target = vp_tangent_flow_states(x0, v_x, c_x, noise, t)
    assert coordinate(target, v_x).abs().max() < TOL


def test_vp_endpoints_are_the_clean_and_projected_noise_states() -> None:
    h, v, noise, _, normalizer = _fixture()
    x0 = normalizer.normalize(h)
    v_x, c_x = _hyperplane(h, v, normalizer)
    eps_perp, at_zero, _ = vp_tangent_flow_states(x0, v_x, c_x, noise, 0.0)
    _, at_one, _ = vp_tangent_flow_states(x0, v_x, c_x, noise, 1.0)
    assert torch.allclose(at_zero, x0, atol=TOL)
    assert torch.allclose(at_one, eps_perp + c_x * v_x, atol=TOL)


def test_the_two_paths_agree_only_at_the_shared_endpoints() -> None:
    """Same corruption endpoints, genuinely different interior. Not a rename."""

    h, v, noise, _, normalizer = _fixture()
    x0 = normalizer.normalize(h)
    v_x, c_x = _hyperplane(h, v, normalizer)
    for t in (0.0, 1.0):
        _, linear, _ = tangent_flow_states(x0, v_x, c_x, noise, t)
        _, circle, _ = vp_tangent_flow_states(x0, v_x, c_x, noise, t)
        assert torch.allclose(linear, circle, atol=TOL)
    for t in (0.25, 0.5, 0.75):
        _, linear, _ = tangent_flow_states(x0, v_x, c_x, noise, t)
        _, circle, _ = vp_tangent_flow_states(x0, v_x, c_x, noise, t)
        assert not torch.allclose(linear, circle, atol=1e-2)


# --------------------------------------------------------------------------
# independent re-derivation of the formulas
# --------------------------------------------------------------------------


@pytest.mark.parametrize("t", TIMES)
def test_vp_states_match_an_independent_numpy_implementation(t: float) -> None:
    """Re-derive with explicit projection matrices, using none of the module's helpers."""

    h, v, noise, _, normalizer = _fixture()
    x0 = normalizer.normalize(h)
    v_x, c_x = _hyperplane(h, v, normalizer)
    _, x_t, target = vp_tangent_flow_states(x0, v_x, c_x, noise, t)

    x0_n = x0.double().numpy()
    v_n = v_x.double().numpy()
    c_n = c_x.double().numpy()
    eps_n = noise.double().numpy()
    theta = (math.pi / 2.0) * t
    expected_x = np.empty_like(x0_n)
    expected_u = np.empty_like(x0_n)
    for row in range(x0_n.shape[0]):
        unit = v_n[row]
        projector = np.eye(unit.size) - np.outer(unit, unit)
        parallel = c_n[row, 0] * unit
        x_perp = projector @ x0_n[row]
        e_perp = projector @ eps_n[row]
        expected_x[row] = parallel + math.cos(theta) * x_perp + math.sin(theta) * e_perp
        expected_u[row] = (math.pi / 2.0) * (
            math.cos(theta) * e_perp - math.sin(theta) * x_perp
        )
    assert np.allclose(x_t.double().numpy(), expected_x, atol=1e-5)
    assert np.allclose(target.double().numpy(), expected_u, atol=1e-5)


@pytest.mark.parametrize("t", (0.1, 0.3, 0.5, 0.7, 0.9))
def test_vp_target_is_the_finite_difference_derivative_of_the_path(t: float) -> None:
    """``u*`` must be d(x_t)/dt, including the pi/2 factor, or Euler is wrong."""

    h, v, noise, _, normalizer = _fixture()
    x0 = normalizer.normalize(h).double()
    v_x, c_x = _hyperplane(h, v, normalizer)
    v_x, c_x, noise = v_x.double(), c_x.double(), noise.double()
    step = 1e-6
    _, ahead, _ = vp_tangent_flow_states(x0, v_x, c_x, noise, t + step)
    _, behind, _ = vp_tangent_flow_states(x0, v_x, c_x, noise, t - step)
    _, _, target = vp_tangent_flow_states(x0, v_x, c_x, noise, t)
    assert torch.allclose((ahead - behind) / (2 * step), target, atol=1e-4)


# --------------------------------------------------------------------------
# the property the experiment exists to test
# --------------------------------------------------------------------------


def test_orthogonal_scale_is_preserved_where_the_chord_halves_it() -> None:
    """The whole hypothesis: on standardized data the chord shrinks, the circle does not."""

    rows, width = 4096, 64
    generator = torch.Generator().manual_seed(20260819)
    x0 = torch.randn(rows, width, generator=generator)
    epsilon = torch.randn(rows, width, generator=generator)
    v_x = torch.zeros(rows, width)
    v_x[:, 0] = 1.0
    c_x = x0[:, :1].clone()

    for t in (0.25, 0.5, 0.75):
        _, chord, _ = tangent_flow_states(x0, v_x, c_x, epsilon, t)
        _, circle, _ = vp_tangent_flow_states(x0, v_x, c_x, epsilon, t)
        chord_var = float(chord[:, 1:].var())
        circle_var = float(circle[:, 1:].var())
        assert math.isclose(chord_var, (1 - t) ** 2 + t**2, rel_tol=0.05)
        assert math.isclose(circle_var, 1.0, rel_tol=0.05)
    # the shrinkage the experiment is aimed at is largest in the middle
    _, chord, _ = tangent_flow_states(x0, v_x, c_x, epsilon, 0.5)
    assert float(chord[:, 1:].var()) < 0.6


# --------------------------------------------------------------------------
# matched severity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("t_linear", (0.05, 0.10, 0.25, 0.50, 0.75, 0.95))
def test_matched_time_equates_the_noise_to_signal_ratio(t_linear: float) -> None:
    t_vp = matched_vp_time(t_linear)
    assert math.isclose(
        math.tan((math.pi / 2.0) * t_vp), t_linear / (1.0 - t_linear), rel_tol=1e-12
    )
    assert math.isclose(matched_linear_time(t_vp), t_linear, rel_tol=1e-12)


def test_matched_time_fixed_points_and_frozen_grid() -> None:
    for fixed in (0.0, 0.5, 1.0):
        assert math.isclose(matched_vp_time(fixed), fixed, abs_tol=1e-12)
    # the grid frozen in docs/POST_STOP_PROTOCOL_2026-08-19.md table A.3
    assert f"{matched_vp_time(0.10):.6f}" == "0.070447"
    assert f"{matched_vp_time(0.25):.6f}" == "0.204833"
    assert f"{matched_vp_time(0.75):.6f}" == "0.795167"


def test_matched_time_rejects_times_outside_the_unit_interval() -> None:
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            matched_vp_time(bad)
        with pytest.raises(ValueError):
            matched_linear_time(bad)


# --------------------------------------------------------------------------
# sampler, dispatch, and inference
# --------------------------------------------------------------------------


def test_the_two_samplers_consume_identical_randomness() -> None:
    """The paired-stream claim of protocol §A.4: same seed, same v, t and eps."""

    h, v, _, _, normalizer = _fixture()
    pool = _pool(v)
    batches = {}
    for objective in (TANGENT_OBJECTIVE, VP_TANGENT_OBJECTIVE):
        generator = torch.Generator().manual_seed(20260816)
        batches[objective] = sample_tangent_flow_batch(
            h,
            normalizer=normalizer,
            pool=pool,
            generator=generator,
            objective=objective,
        )
    linear, circle = batches[TANGENT_OBJECTIVE], batches[VP_TANGENT_OBJECTIVE]
    assert torch.equal(linear.v_x, circle.v_x)
    assert torch.equal(linear.c_x, circle.c_x)
    assert torch.equal(linear.t, circle.t)
    assert torch.equal(linear.epsilon, circle.epsilon)
    # only the path differs
    assert not torch.allclose(linear.x_t, circle.x_t, atol=1e-2)
    assert circle.objective == VP_TANGENT_OBJECTIVE
    assert linear.objective == TANGENT_OBJECTIVE


def test_sampled_vp_batch_satisfies_every_declared_invariant() -> None:
    h, v, _, _, normalizer = _fixture()
    generator = torch.Generator().manual_seed(0)
    batch = sample_tangent_flow_batch(
        h,
        normalizer=normalizer,
        pool=_pool(v),
        generator=generator,
        objective=VP_TANGENT_OBJECTIVE,
    )
    assert torch.allclose(coordinate(batch.x_t, batch.v_x), batch.c_x, atol=TOL)
    assert coordinate(batch.target_velocity, batch.v_x).abs().max() < TOL
    assert coordinate(batch.epsilon_perp, batch.v_x).abs().max() < TOL


def test_unknown_paths_are_rejected_by_dispatch_and_inference() -> None:
    h, v, noise, c_target, normalizer = _fixture()
    x0 = normalizer.normalize(h)
    v_x, c_x = _hyperplane(h, v, normalizer)
    with pytest.raises(ValueError, match="constraint-preserving"):
        tangent_path_states(x0, v_x, c_x, noise, 0.5, objective="isotropic")
    torch.manual_seed(0)
    cfg = FlowModelConfig(
        d_model=h.shape[1], d_mlp=32, n_blocks=2, time_dim=8, time_hidden=16,
        max_period=10000.0,
    )
    model = ConditionalFlowMatcher(cfg, ConditionEncoderConfig(cond_hidden=8), normalizer)
    with pytest.raises(ValueError, match="constraint-preserving"):
        clamp_then_tangent_flow(
            model.eval(), h, v, c_target, noise=noise, t_start=0.5, nfe=1,
            objective="isotropic",
        )


@pytest.mark.parametrize("nfe", (1, 3, 5))
def test_vp_inference_holds_the_requested_coordinate_across_every_step(nfe: int) -> None:
    h, v, noise, c_target, normalizer = _fixture()
    torch.manual_seed(0)
    cfg = FlowModelConfig(
        d_model=h.shape[1], d_mlp=32, n_blocks=2, time_dim=8, time_hidden=16,
        max_period=10000.0,
    )
    model = ConditionalFlowMatcher(cfg, ConditionEncoderConfig(cond_hidden=8), normalizer)
    out = clamp_then_tangent_flow(
        model.eval(),
        h,
        v,
        c_target,
        noise=noise,
        t_start=matched_vp_time(0.10),
        nfe=nfe,
        objective=VP_TANGENT_OBJECTIVE,
    )
    assert float((out.realised_coordinate - out.requested_coordinate).abs().max()) < 1e-3
    assert out.network_evaluations == nfe
    assert out.diagnostics["objective"] == VP_TANGENT_OBJECTIVE


def test_vp_t_start_zero_is_the_exact_hard_clamp() -> None:
    h, v, noise, c_target, normalizer = _fixture()
    torch.manual_seed(0)
    cfg = FlowModelConfig(
        d_model=h.shape[1], d_mlp=32, n_blocks=2, time_dim=8, time_hidden=16,
        max_period=10000.0,
    )
    model = ConditionalFlowMatcher(cfg, ConditionEncoderConfig(cond_hidden=8), normalizer)
    out = clamp_then_tangent_flow(
        model.eval(), h, v, c_target, noise=noise, t_start=0.0, nfe=1,
        objective=VP_TANGENT_OBJECTIVE,
    )
    assert out.network_evaluations == 0
    assert torch.allclose(out.activation, out.seed)
