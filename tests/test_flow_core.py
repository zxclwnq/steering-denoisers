"""Independent mathematical tests for the clean rectified-flow core."""

from __future__ import annotations

import pytest
import torch

from interp.flow_core import (
    ActivationNormalizer,
    flow_matching_loss,
    linear_interpolate,
    sample_flow_batch,
    velocity_target,
)


def test_normalizer_round_trip_and_buffers_are_not_trainable() -> None:
    normalizer = ActivationNormalizer(
        mean=torch.tensor([1.0, -2.0, 0.5]),
        std=torch.tensor([2.0, 4.0, 0.25]),
        eps=1e-5,
    )
    h = torch.tensor([[3.0, 6.0, 1.0], [-1.0, -6.0, 0.0]])

    restored = normalizer.denormalize(normalizer.normalize(h))

    assert torch.allclose(restored, h, atol=1e-6)
    assert dict(normalizer.named_parameters()) == {}
    assert set(dict(normalizer.named_buffers())) == {"mean", "std"}


def test_normalizer_rejects_nonpositive_or_nonfinite_statistics() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        ActivationNormalizer(torch.zeros(2), torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        ActivationNormalizer(torch.tensor([0.0, float("nan")]), torch.ones(2))


def test_linear_interpolation_matches_hand_calculated_values() -> None:
    x0 = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    noise = torch.tensor([[5.0, -2.0], [3.0, 7.0]])
    t = torch.tensor([[0.25], [0.75]])

    got = linear_interpolate(x0, noise, t)

    want = torch.tensor([[2.0, 1.0], [2.0, 6.0]])
    assert torch.equal(got, want)


def test_interpolation_endpoints_and_velocity_are_exact() -> None:
    x0 = torch.tensor([[1.5, -2.0], [0.25, 4.0]])
    noise = torch.tensor([[-3.0, 8.0], [2.25, -1.0]])
    want_velocity = torch.tensor([[-4.5, 10.0], [2.0, -5.0]])

    assert torch.equal(linear_interpolate(x0, noise, 0.0), x0)
    assert torch.equal(linear_interpolate(x0, noise, 1.0), noise)
    assert torch.equal(velocity_target(x0, noise), want_velocity)


def test_velocity_target_is_the_path_derivative() -> None:
    x0 = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    noise = torch.tensor([[4.0, 6.0]], dtype=torch.float64)
    t = 0.37
    dt = 1e-6

    finite_difference = (
        linear_interpolate(x0, noise, t + dt) - linear_interpolate(x0, noise, t - dt)
    ) / (2.0 * dt)

    assert torch.allclose(
        finite_difference, torch.tensor([[3.0, 8.0]], dtype=torch.float64), atol=1e-9
    )


def test_sample_flow_batch_is_seeded_and_samples_time_per_activation() -> None:
    x0 = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10.0
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)

    a = sample_flow_batch(x0, generator=g1)
    b = sample_flow_batch(x0, generator=g2)

    assert torch.equal(a.x_t, b.x_t)
    assert torch.equal(a.t, b.t)
    assert torch.equal(a.target_velocity, b.target_velocity)
    assert a.t.shape == (8, 1)
    assert torch.unique(a.t).numel() > 1
    assert bool(((a.t >= 0.0) & (a.t <= 1.0)).all())
    reconstructed_noise = x0 + a.target_velocity
    assert torch.allclose(a.x_t, (1.0 - a.t) * x0 + a.t * reconstructed_noise)


def test_flow_matching_loss_is_mean_squared_velocity_error() -> None:
    prediction = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 1.0]])

    # Squared errors are [1, 0, 4, 4], so their mean is 9/4.
    assert flow_matching_loss(prediction, target).item() == pytest.approx(2.25)


@pytest.mark.parametrize("bad_t", [-0.01, 1.01, float("nan")])
def test_interpolation_rejects_invalid_flow_time(bad_t: float) -> None:
    x0 = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="flow time"):
        linear_interpolate(x0, torch.ones_like(x0), bad_t)
