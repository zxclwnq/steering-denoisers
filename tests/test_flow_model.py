"""Architecture tests derived from the frozen clean-flow specification."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from interp.flow_core import (
    ActivationNormalizer,
    FlowMatcher,
    FlowModelConfig,
    load_flow_config,
    n_parameters,
    sinusoidal_time_embedding,
    velocity_target,
)

ROOT = Path(__file__).resolve().parents[1]


def tiny_config() -> FlowModelConfig:
    return FlowModelConfig(
        d_model=4,
        d_mlp=8,
        n_blocks=2,
        time_dim=6,
        time_hidden=5,
        max_period=100.0,
    )


def tiny_model() -> FlowMatcher:
    torch.manual_seed(17)
    cfg = tiny_config()
    norm = ActivationNormalizer(torch.zeros(cfg.d_model), torch.ones(cfg.d_model))
    return FlowMatcher(cfg, norm)


def test_canonical_config_pins_the_written_architecture_and_objective() -> None:
    cfg = load_flow_config(ROOT / "configs" / "flow_core_v1.yaml")

    assert cfg == FlowModelConfig(
        d_model=768,
        d_mlp=1536,
        n_blocks=3,
        time_dim=256,
        time_hidden=768,
        max_period=10_000.0,
    )


def test_canonical_parameter_count_is_exactly_frozen() -> None:
    cfg = load_flow_config(ROOT / "configs" / "flow_core_v1.yaml")
    norm = ActivationNormalizer(torch.zeros(cfg.d_model), torch.ones(cfg.d_model))
    model = FlowMatcher(cfg, norm)

    assert n_parameters(model) == 16_147_200


def test_sinusoidal_embedding_has_known_zero_time_value() -> None:
    got = sinusoidal_time_embedding(torch.tensor([0.0]), dim=6, max_period=100.0)

    assert torch.equal(got, torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]))


def test_model_is_tokenwise_and_returns_one_velocity_per_activation() -> None:
    model = tiny_model()
    x_t = torch.randn(7, 4)
    t = torch.linspace(0.0, 1.0, 7)

    out = model(x_t, t)

    assert out.shape == x_t.shape
    assert torch.isfinite(out).all()


def test_scalar_and_expanded_time_inputs_are_equivalent() -> None:
    model = tiny_model().eval()
    x_t = torch.randn(5, 4)

    scalar = model(x_t, 0.25)
    expanded = model(x_t, torch.full((5, 1), 0.25))

    assert torch.equal(scalar, expanded)


def test_model_output_depends_on_time_for_fixed_activation() -> None:
    model = tiny_model().eval()
    x_t = torch.tensor([[0.5, -1.0, 2.0, 0.25]]).expand(3, -1)

    out = model(x_t, torch.tensor([0.0, 0.5, 1.0]))

    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(out[1], out[2])


def test_gradients_reach_every_trainable_parameter() -> None:
    model = tiny_model().train()
    x_t = torch.randn(12, 4)
    t = torch.linspace(0.05, 0.95, 12)

    model(x_t, t).square().mean().backward()

    missing = [name for name, p in model.named_parameters() if p.grad is None]
    nonfinite = [
        name
        for name, p in model.named_parameters()
        if p.grad is not None and not p.grad.isfinite().all()
    ]
    assert missing == []
    assert nonfinite == []


def test_tiny_model_overfits_one_fixed_flow_batch() -> None:
    torch.manual_seed(11)
    model = tiny_model().train()
    x0 = torch.randn(4, 4)
    noise = torch.randn(4, 4)
    t = torch.linspace(0.05, 0.95, 4)[:, None]
    x_t = (1.0 - t) * x0 + t * noise
    target = velocity_target(x0, noise)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    with torch.no_grad():
        initial = float((model(x_t, t) - target).square().mean())
    for _ in range(150):
        loss = (model(x_t, t) - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = float((model(x_t, t) - target).square().mean())

    assert final < initial * 0.05, (initial, final)


def test_model_signature_cannot_accept_steering_information() -> None:
    assert list(inspect.signature(FlowMatcher.forward).parameters) == ["self", "x_t", "t"]


@pytest.mark.parametrize(
    ("x_t", "t", "message"),
    [
        (torch.zeros(2, 3), 0.5, "activation_dim"),
        (torch.zeros(1, 2, 4), 0.5, r"\[batch, activation_dim\]"),
        (torch.zeros(2, 4), torch.zeros(3), "flow time"),
        (torch.zeros(2, 4), 1.1, "flow time"),
    ],
)
def test_model_fails_loudly_on_invalid_inputs(
    x_t: torch.Tensor, t: torch.Tensor | float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        tiny_model()(x_t, t)
