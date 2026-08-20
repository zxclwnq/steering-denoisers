"""Analytic tests for clean reverse-Euler flow sampling."""

from __future__ import annotations

import pytest
import torch

from interp.flow_core import ActivationNormalizer, linear_interpolate
from interp.flow_sampling import (
    euler_time_grid,
    flow_correct,
    partial_noise,
    reverse_euler,
    sdedit_sample,
)


class RecordingVelocity:
    def __init__(self, velocity: torch.Tensor) -> None:
        self.velocity = velocity
        self.times: list[torch.Tensor] = []

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.times.append(t.detach().clone())
        return self.velocity.expand_as(x)


def test_euler_grid_is_the_hand_calculated_reverse_schedule() -> None:
    got = euler_time_grid(t_start=0.75, nfe=3, dtype=torch.float64)

    assert torch.equal(got, torch.tensor([0.75, 0.50, 0.25, 0.0], dtype=torch.float64))


def test_partial_noise_is_the_same_convex_path_used_for_training() -> None:
    x = torch.tensor([[2.0, -2.0], [4.0, 8.0]])
    noise = torch.tensor([[6.0, 2.0], [-4.0, 0.0]])

    got = partial_noise(x, noise, t_start=0.25)

    assert torch.equal(got, torch.tensor([[3.0, -1.0], [2.0, 6.0]]))


@pytest.mark.parametrize("nfe", [1, 3, 5])
def test_true_straight_path_velocity_recovers_x0_exactly(nfe: int) -> None:
    x0 = torch.tensor([[1.0, -2.0], [0.5, 4.0]], dtype=torch.float64)
    noise = torch.tensor([[5.0, 6.0], [-3.5, 0.0]], dtype=torch.float64)
    t_start = 0.6
    x_t = linear_interpolate(x0, noise, t_start)
    oracle = RecordingVelocity(noise - x0)

    got = reverse_euler(oracle, x_t, t_start=t_start, nfe=nfe)

    assert torch.allclose(got, x0, atol=1e-12)
    assert len(oracle.times) == nfe


def test_reverse_euler_calls_the_model_at_documented_times() -> None:
    x_t = torch.zeros(2, 3)
    oracle = RecordingVelocity(torch.ones(1, 3))

    reverse_euler(oracle, x_t, t_start=0.9, nfe=3)

    got = [float(t[0, 0]) for t in oracle.times]
    assert got == pytest.approx([0.9, 0.6, 0.3])


def test_reverse_sign_moves_toward_data_not_noise() -> None:
    x0 = torch.tensor([[0.0]])
    noise = torch.tensor([[10.0]])
    x_t = torch.tensor([[5.0]])  # t_start=0.5 on the exact path

    got = reverse_euler(RecordingVelocity(noise - x0), x_t, t_start=0.5, nfe=1)

    assert torch.equal(got, x0)


def test_zero_time_is_exact_identity_with_zero_model_evaluations() -> None:
    x = torch.tensor([[1.0, 2.0]])
    noise = torch.tensor([[9.0, -4.0]])
    oracle = RecordingVelocity(torch.ones_like(x))

    got = sdedit_sample(oracle, x, noise=noise, t_start=0.0, nfe=5)

    assert torch.equal(got, x)
    assert oracle.times == []


def test_flow_correct_round_trips_raw_activations_with_true_velocity() -> None:
    mean = torch.tensor([10.0, -4.0])
    std = torch.tensor([2.0, 0.5])
    normalizer = ActivationNormalizer(mean, std)
    h = torch.tensor([[12.0, -3.0], [8.0, -4.5]])
    x0 = normalizer.normalize(h)
    noise = torch.tensor([[2.0, -1.0], [-3.0, 4.0]])

    class ExactModel(RecordingVelocity):
        def __init__(self) -> None:
            super().__init__(noise - x0)
            self.normalizer = normalizer

    model = ExactModel()
    got = flow_correct(model, h, noise=noise, t_start=0.4, nfe=3)

    assert torch.allclose(got, h, atol=1e-6)
    assert len(model.times) == 3


@pytest.mark.parametrize(
    ("t_start", "nfe", "message"),
    [(-0.1, 1, "t_start"), (1.1, 1, "t_start"), (0.5, 0, "nfe")],
)
def test_sampler_rejects_invalid_schedule(t_start: float, nfe: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        euler_time_grid(t_start=t_start, nfe=nfe)


def test_sampler_rejects_nonfinite_velocity() -> None:
    class BadVelocity:
        def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x, float("nan"))

    with pytest.raises(ValueError, match="finite"):
        reverse_euler(BadVelocity(), torch.zeros(2, 3), t_start=0.5, nfe=1)
