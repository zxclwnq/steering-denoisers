"""Scientific invariants of the conditional Phase B intervention.

These encode the semantics stated in docs/PHASE_B_CONDITIONAL_PROTOCOL.md section 4,
in particular the two that differ from additive steering and are easy to get
silently wrong:

* ``t_start = 0`` is a null intervention at every alpha under ``seed_mode="clean"``,
  because the condition only enters through a model call;
* ``alpha = 0`` is NOT the identity, because the prior still noises and
  reintegrates, so the arm measures the prior's own reconstruction cost.
"""

from __future__ import annotations

import pytest
import torch

from interp.conditional_flow import (
    clamp_seed,
    conditional_clamp_steer,
    target_coordinate,
)
from interp.flow_core import ActivationNormalizer

D_MODEL = 8


class StubConditionalFlow:
    """Records the conditions it is asked for and returns a fixed velocity."""

    def __init__(self, value: float = 0.25) -> None:
        self.normalizer = ActivationNormalizer(
            torch.zeros(D_MODEL), torch.ones(D_MODEL), eps=0.0
        )
        self.value = value
        self.seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def velocity_field(self, v_x: torch.Tensor, c_x: torch.Tensor):  # noqa: ANN201
        self.seen.append((v_x.detach().clone(), c_x.detach().clone()))

        def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x, self.value)

        return velocity


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260815)
    h = torch.randn(5, D_MODEL, generator=generator)
    direction = torch.zeros(D_MODEL)
    direction[2] = 1.0
    noise = torch.randn(5, D_MODEL, generator=generator)
    return h, direction, noise


def test_clean_seed_never_forms_the_additive_steer() -> None:
    """seed_mode='clean' must feed clean h to the prior: the condition is the steer."""

    h, direction, noise = _fixture()

    out = conditional_clamp_steer(
        StubConditionalFlow(),
        h,
        direction,
        alpha=3.0,
        noise=noise,
        t_start=0.5,
        nfe=1,
        seed_mode="clean",
    )

    assert torch.equal(out.seed, h)
    additive = h + 3.0 * direction
    assert not torch.allclose(out.seed, additive)


def test_clamp_seed_reproduces_the_additive_steer_in_additive_mode() -> None:
    """The pre-existing seed_mode='clamp' path is exactly h + alpha*v."""

    h, direction, noise = _fixture()

    out = conditional_clamp_steer(
        StubConditionalFlow(),
        h,
        direction,
        alpha=3.0,
        noise=noise,
        t_start=0.5,
        nfe=1,
        seed_mode="clamp",
    )

    assert torch.allclose(out.seed, h + 3.0 * direction, atol=1e-6)


def test_requested_coordinate_is_natural_plus_alpha() -> None:
    h, direction, _ = _fixture()

    c_target = target_coordinate(h, direction, 2.5, mode="additive")

    expected = (h * direction).sum(dim=-1, keepdim=True) + 2.5
    assert torch.allclose(c_target, expected, atol=1e-6)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 5.0])
def test_t_start_zero_is_a_null_intervention_under_clean_seeding(alpha: float) -> None:
    """No model call means no steering channel, so h is returned untouched."""

    h, direction, noise = _fixture()
    model = StubConditionalFlow()

    out = conditional_clamp_steer(
        model,
        h,
        direction,
        alpha=alpha,
        noise=noise,
        t_start=0.0,
        nfe=1,
        seed_mode="clean",
    )

    assert torch.equal(out.activation, h)
    assert out.network_evaluations == 0
    assert model.seen == []


@pytest.mark.parametrize("alpha", [0.0, 1.0, 5.0])
def test_t_start_zero_under_clamp_seeding_keeps_the_additive_steer(alpha: float) -> None:
    """The two seed modes differ exactly here, which is why the protocol names it."""

    h, direction, noise = _fixture()

    out = conditional_clamp_steer(
        StubConditionalFlow(),
        h,
        direction,
        alpha=alpha,
        noise=noise,
        t_start=0.0,
        nfe=1,
        seed_mode="clamp",
    )

    assert torch.allclose(out.activation, h + alpha * direction, atol=1e-6)


def test_alpha_zero_is_not_the_identity_when_the_prior_runs() -> None:
    """The alpha=0 arm measures the prior's reconstruction cost, not a no-op."""

    h, direction, noise = _fixture()

    out = conditional_clamp_steer(
        StubConditionalFlow(),
        h,
        direction,
        alpha=0.0,
        noise=noise,
        t_start=0.5,
        nfe=1,
        seed_mode="clean",
    )

    assert not torch.allclose(out.activation, h, atol=1e-4)


def test_condition_carries_the_requested_coordinate_not_the_natural_one() -> None:
    h, direction, noise = _fixture()
    model = StubConditionalFlow()

    conditional_clamp_steer(
        model,
        h,
        direction,
        alpha=4.0,
        noise=noise,
        t_start=0.75,
        nfe=1,
        seed_mode="clean",
    )

    assert len(model.seen) == 1
    _, c_x = model.seen[0]
    # Unit std and zero mean make the standardized coordinate equal the raw one.
    expected = (h * direction).sum(dim=-1, keepdim=True) + 4.0
    assert torch.allclose(c_x, expected, atol=1e-5)


def test_identical_noise_and_grid_are_deterministic() -> None:
    h, direction, noise = _fixture()

    first = conditional_clamp_steer(
        StubConditionalFlow(), h, direction, alpha=2.0, noise=noise,
        t_start=0.75, nfe=3, seed_mode="clean",
    )
    second = conditional_clamp_steer(
        StubConditionalFlow(), h, direction, alpha=2.0, noise=noise,
        t_start=0.75, nfe=3, seed_mode="clean",
    )

    assert torch.equal(first.activation, second.activation)


def test_nfe_is_the_exact_network_evaluation_count() -> None:
    h, direction, noise = _fixture()

    for nfe in (1, 3, 5):
        out = conditional_clamp_steer(
            StubConditionalFlow(), h, direction, alpha=1.0, noise=noise,
            t_start=0.9, nfe=nfe, seed_mode="clean",
        )
        assert out.network_evaluations == nfe


def test_unknown_seed_mode_is_rejected() -> None:
    h, direction, noise = _fixture()

    with pytest.raises(ValueError, match="seed_mode"):
        conditional_clamp_steer(
            StubConditionalFlow(), h, direction, alpha=1.0, noise=noise,
            t_start=0.5, nfe=1, seed_mode="additive",
        )


def test_clean_seeding_leaves_clamp_seed_untouched_as_a_helper() -> None:
    """seed_mode must not change what clamp_seed itself means elsewhere."""

    h, direction, _ = _fixture()
    c_target = target_coordinate(h, direction, 1.5, mode="additive")

    seeded = clamp_seed(h, direction, c_target)

    realized = (seeded * direction).sum(dim=-1, keepdim=True)
    assert torch.allclose(realized, c_target, atol=1e-5)
