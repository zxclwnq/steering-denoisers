"""Geometry invariants of the constrained conditional flow.

Every test here proves a property the evaluation depends on. The central ones
are that constrained forward noising leaves ``<x_t, v_x> = c_x`` exactly (which
ordinary SDEdit does not), and that projections never count as network
evaluations.
"""

from __future__ import annotations

import pytest
import torch

from interp.conditional_flow import clamp_seed, standardized_hyperplane
from interp.constrained_flow import (
    clamp_seed_flow,
    constrained_noise_endpoint,
    constrained_partial_noise,
    constrained_reverse_euler,
    correction_decomposition,
    hyperplane_project,
)
from interp.flow_core import ActivationNormalizer, linear_interpolate

D_MODEL = 16
ROWS = 12
TOL = 1e-4


class StubConditional:
    """Velocity with a deliberate component along v_x, so projection has work to do."""

    def __init__(self, mean=None, std=None, value: float = 0.3) -> None:  # noqa: ANN001
        self.normalizer = ActivationNormalizer(
            torch.zeros(D_MODEL) if mean is None else mean,
            torch.ones(D_MODEL) if std is None else std,
            eps=0.0,
        )
        self.value = value
        self.calls = 0

    def velocity_field(self, v_x: torch.Tensor, c_x: torch.Tensor):  # noqa: ANN201, ARG002
        def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
            self.calls += 1
            return torch.full_like(x, self.value)

        return velocity


def _fixture(seed: int = 20260815, *, real_scale: bool = False):  # noqa: ANN202
    generator = torch.Generator().manual_seed(seed)
    h = torch.randn(ROWS, D_MODEL, generator=generator) * 3.0
    v = torch.randn(ROWS, D_MODEL, generator=generator)
    v = v / v.norm(dim=-1, keepdim=True)
    noise = torch.randn(ROWS, D_MODEL, generator=generator)
    c_target = torch.randn(ROWS, 1, generator=generator) * 4.0
    if real_scale:
        mean = torch.randn(D_MODEL, generator=generator)
        std = torch.rand(D_MODEL, generator=generator) + 0.5
        return h, v, noise, c_target, mean, std
    return h, v, noise, c_target


# --- clamp ---------------------------------------------------------------


def test_clamp_puts_the_coordinate_exactly_on_target() -> None:
    h, v, _, c_target = _fixture()

    seeded = clamp_seed(h, v, c_target)

    realized = (seeded * v).sum(dim=-1, keepdim=True)
    assert torch.allclose(realized, c_target, atol=1e-5)


def test_clamp_only_arm_returns_the_clamp_with_zero_evaluations() -> None:
    h, v, noise, c_target = _fixture()
    model = StubConditional()

    out = clamp_seed_flow(
        model, h, v, c_target, noise=noise, t_start=0.5, nfe=3, arm="clamp_only"
    )

    assert torch.equal(out.activation, clamp_seed(h, v, c_target))
    assert out.network_evaluations == 0
    assert out.projections == 0
    assert model.calls == 0


# --- standardized equivalence -------------------------------------------


def test_raw_and_standardized_constraints_are_equivalent() -> None:
    """<h, v> = c_target must hold exactly when <x, v_x> = c_x under the normalizer."""

    h, v, _, c_target, mean, std = _fixture(real_scale=True)
    normalizer = ActivationNormalizer(mean, std, eps=1e-5)
    seeded = clamp_seed(h, v, c_target)

    v_x, c_x = standardized_hyperplane(normalizer, v, c_target)
    x = normalizer.normalize(seeded)

    assert torch.allclose((x * v_x).sum(dim=-1, keepdim=True), c_x, atol=1e-4)
    assert torch.allclose(v_x.norm(dim=-1), torch.ones(ROWS), atol=1e-5)


# --- constrained forward noising ----------------------------------------


def test_ordinary_sdedit_breaks_the_hyperplane() -> None:
    """The premise of the constrained arm: plain interpolation does not preserve c_x."""

    h, v, noise, c_target, mean, std = _fixture(real_scale=True)
    normalizer = ActivationNormalizer(mean, std, eps=1e-5)
    v_x, c_x = standardized_hyperplane(normalizer, v, c_target)
    x0 = normalizer.normalize(clamp_seed(h, v, c_target))

    x_t = linear_interpolate(x0, noise, 0.5)

    assert not torch.allclose((x_t * v_x).sum(dim=-1, keepdim=True), c_x, atol=1e-2)


@pytest.mark.parametrize("t_start", [0.0, 0.10, 0.25, 0.50, 0.90, 1.0])
def test_constrained_noising_preserves_the_hyperplane_at_every_time(t_start: float) -> None:
    h, v, noise, c_target, mean, std = _fixture(real_scale=True)
    normalizer = ActivationNormalizer(mean, std, eps=1e-5)
    v_x, c_x = standardized_hyperplane(normalizer, v, c_target)
    x0 = normalizer.normalize(clamp_seed(h, v, c_target))

    x_t = constrained_partial_noise(x0, noise, v_x, c_x, t_start=t_start)

    assert torch.allclose((x_t * v_x).sum(dim=-1, keepdim=True), c_x, atol=TOL)


def test_constrained_endpoint_is_the_projection_of_the_gaussian() -> None:
    """eps' = eps_perp + c_x v_x, i.e. eps projected onto the constraint."""

    _, _, noise, _, mean, std = _fixture(real_scale=True)
    generator = torch.Generator().manual_seed(3)
    v_x = torch.randn(ROWS, D_MODEL, generator=generator)
    v_x = v_x / v_x.norm(dim=-1, keepdim=True)
    c_x = torch.randn(ROWS, 1, generator=generator)

    endpoint = constrained_noise_endpoint(noise, v_x, c_x)

    parallel = (noise * v_x).sum(dim=-1, keepdim=True) * v_x
    expected = (noise - parallel) + c_x * v_x
    assert torch.allclose(endpoint, expected, atol=1e-5)
    assert torch.allclose((endpoint * v_x).sum(dim=-1, keepdim=True), c_x, atol=1e-5)


def test_constrained_noising_still_moves_the_orthogonal_subspace() -> None:
    """Preserving the coordinate must not accidentally freeze everything else."""

    h, v, noise, c_target, mean, std = _fixture(real_scale=True)
    normalizer = ActivationNormalizer(mean, std, eps=1e-5)
    v_x, c_x = standardized_hyperplane(normalizer, v, c_target)
    x0 = normalizer.normalize(clamp_seed(h, v, c_target))

    x_t = constrained_partial_noise(x0, noise, v_x, c_x, t_start=0.5)

    assert not torch.allclose(x_t, x0, atol=1e-3)


# --- projected integration ----------------------------------------------


def test_projected_euler_holds_the_constraint_after_every_step() -> None:
    h, v, noise, c_target, mean, std = _fixture(real_scale=True)
    normalizer = ActivationNormalizer(mean, std, eps=1e-5)
    v_x, c_x = standardized_hyperplane(normalizer, v, c_target)
    x0 = normalizer.normalize(clamp_seed(h, v, c_target))
    x_t = constrained_partial_noise(x0, noise, v_x, c_x, t_start=0.5)

    residuals = []

    def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
        residuals.append(float(((x * v_x).sum(dim=-1, keepdim=True) - c_x).abs().max()))
        return torch.full_like(x, 0.3)

    final, evaluations, projections = constrained_reverse_euler(
        velocity, x_t, v_x, c_x, t_start=0.5, nfe=5, project_each_step=True
    )

    assert evaluations == 5
    assert projections == 5
    assert max(residuals) < TOL
    assert float(((final * v_x).sum(dim=-1, keepdim=True) - c_x).abs().max()) < TOL


def test_unprojected_integration_leaves_the_hyperplane() -> None:
    """Without projection the learned velocity drifts off the constraint."""

    h, v, noise, c_target, mean, std = _fixture(real_scale=True)
    normalizer = ActivationNormalizer(mean, std, eps=1e-5)
    v_x, c_x = standardized_hyperplane(normalizer, v, c_target)
    x0 = normalizer.normalize(clamp_seed(h, v, c_target))
    x_t = constrained_partial_noise(x0, noise, v_x, c_x, t_start=0.5)

    def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
        return torch.full_like(x, 0.3)

    final, _, projections = constrained_reverse_euler(
        velocity, x_t, v_x, c_x, t_start=0.5, nfe=5, project_each_step=False
    )

    assert projections == 0
    assert float(((final * v_x).sum(dim=-1, keepdim=True) - c_x).abs().max()) > TOL


def test_hyperplane_projection_is_idempotent() -> None:
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(ROWS, D_MODEL, generator=generator)
    v_x = torch.randn(ROWS, D_MODEL, generator=generator)
    v_x = v_x / v_x.norm(dim=-1, keepdim=True)
    c_x = torch.randn(ROWS, 1, generator=generator)

    once = hyperplane_project(x, v_x, c_x)
    twice = hyperplane_project(once, v_x, c_x)

    assert torch.allclose(once, twice, atol=1e-6)


# --- NFE accounting ------------------------------------------------------


@pytest.mark.parametrize("nfe", [1, 3, 5])
@pytest.mark.parametrize("arm", ["sdedit", "tangent", "projected"])
def test_projection_never_counts_as_a_network_evaluation(arm: str, nfe: int) -> None:
    h, v, noise, c_target, mean, std = _fixture(real_scale=True)
    model = StubConditional(mean=mean, std=std)

    out = clamp_seed_flow(
        model, h, v, c_target, noise=noise, t_start=0.5, nfe=nfe, arm=arm
    )

    assert out.network_evaluations == nfe
    assert model.calls == nfe
    assert out.projections == (nfe if arm == "projected" else 0)


# --- sign canonicalization ----------------------------------------------


def test_negated_direction_and_target_give_the_same_trajectory() -> None:
    """(v, c) and (-v, -c) describe the same hyperplane, so results must match."""

    h, v, noise, c_target, mean, std = _fixture(real_scale=True)
    model = StubConditional(mean=mean, std=std)

    for arm in ("sdedit", "tangent", "projected"):
        positive = clamp_seed_flow(
            model, h, v, c_target, noise=noise, t_start=0.5, nfe=3, arm=arm
        )
        negated = clamp_seed_flow(
            model, h, -v, -c_target, noise=noise, t_start=0.5, nfe=3, arm=arm
        )
        assert torch.allclose(positive.activation, negated.activation, atol=1e-3), arm


# --- end-to-end constraint behaviour ------------------------------------


def test_projected_arm_preserves_the_raw_coordinate_end_to_end() -> None:
    h, v, noise, c_target, mean, std = _fixture(real_scale=True)
    model = StubConditional(mean=mean, std=std)

    out = clamp_seed_flow(
        model, h, v, c_target, noise=noise, t_start=0.5, nfe=3, arm="projected"
    )

    realized = (out.activation * v).sum(dim=-1, keepdim=True)
    assert torch.allclose(realized, c_target, atol=1e-3)


def test_correction_decomposition_is_orthogonal_for_the_projected_arm() -> None:
    h, v, noise, c_target, mean, std = _fixture(real_scale=True)
    model = StubConditional(mean=mean, std=std)

    out = clamp_seed_flow(
        model, h, v, c_target, noise=noise, t_start=0.5, nfe=3, arm="projected"
    )
    geometry = correction_decomposition(out.seed, out.activation, v)

    assert geometry["parallel_norm_mean"] < 1e-3
    assert geometry["orthogonal_norm_mean"] > 1e-3


def test_unknown_arm_is_rejected() -> None:
    h, v, noise, c_target = _fixture()

    with pytest.raises(ValueError, match="unknown arm"):
        clamp_seed_flow(
            StubConditional(), h, v, c_target, noise=noise, t_start=0.5, nfe=1, arm="additive"
        )
