"""Mathematical core for linear-path flow matching in standardized activation space.

The convention is fixed throughout this module:

    t = 0  is the clean/data endpoint
    t = 1  is the Gaussian-noise endpoint
    x_t = (1 - t) x_0 + t epsilon
    dx_t / dt = epsilon - x_0

This module contains no GPT-2, SAE, steering-vector, dataset, or training-run logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch import nn


class ActivationNormalizer(nn.Module):
    """Fixed per-dimension train-split standardization statistics."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-5) -> None:
        super().__init__()
        if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
            raise ValueError("mean and std must be same-shaped one-dimensional tensors")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("normalization statistics must be finite")
        if not bool((std > 0).all()):
            raise ValueError("normalization std must be strictly positive")
        if not torch.isfinite(torch.tensor(eps)) or eps < 0:
            raise ValueError("normalization eps must be finite and nonnegative")
        self.eps = float(eps)
        self.register_buffer("mean", mean.detach().float().clone())
        self.register_buffer("std", std.detach().float().clone())

    def normalize(self, h: torch.Tensor) -> torch.Tensor:
        return (h - self.mean) / (self.std + self.eps)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean

    forward = normalize


@dataclass(frozen=True)
class FlowBatch:
    """One sampled flow-matching batch in standardized coordinates."""

    x_t: torch.Tensor
    t: torch.Tensor
    target_velocity: torch.Tensor


@dataclass(frozen=True)
class FlowModelConfig:
    """Explicit architecture parameters for the tokenwise velocity model."""

    d_model: int
    d_mlp: int
    n_blocks: int
    time_dim: int
    time_hidden: int
    max_period: float
    # Width of the modelled activation. ``None`` means "same as d_model", which is
    # the original narrow architecture and keeps existing checkpoints loadable and
    # config-equal after this field was introduced.
    activation_dim: int | None = None

    def __post_init__(self) -> None:
        for name in ("d_model", "d_mlp", "n_blocks", "time_dim", "time_hidden"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.activation_dim is not None and self.activation_dim <= 0:
            raise ValueError("activation_dim must be positive")
        if self.time_dim % 2:
            raise ValueError("time_dim must be even")
        if not math.isfinite(self.max_period) or self.max_period <= 0:
            raise ValueError("max_period must be finite and positive")

    @property
    def activation_width(self) -> int:
        """Input/output width of the velocity model in activation coordinates."""

        return self.d_model if self.activation_dim is None else self.activation_dim


def flow_parameter_count(cfg: FlowModelConfig) -> int:
    """Count ``FlowMatcher`` parameters analytically, without building the model."""

    width = cfg.activation_width
    input_projection = width * cfg.d_model + cfg.d_model
    time_embedding = cfg.time_dim * cfg.time_hidden + cfg.time_hidden
    time_embedding += cfg.time_hidden * cfg.d_model + cfg.d_model
    block = 2 * cfg.d_model
    block += 3 * (cfg.d_model * cfg.d_mlp + cfg.d_mlp)
    block += cfg.d_mlp * cfg.d_model + cfg.d_model
    output_norm = 2 * cfg.d_model
    output_projection = cfg.d_model * width + width
    return (
        input_projection
        + time_embedding
        + cfg.n_blocks * block
        + output_norm
        + output_projection
    )


def load_flow_config(path: Path) -> FlowModelConfig:
    """Load and validate the versioned clean-flow architecture specification."""

    raw = yaml.safe_load(path.read_text())
    if raw.get("version") != 1 or raw.get("status") != "clean_reimplementation":
        raise ValueError("expected clean flow-core config version 1")
    architecture = raw.get("architecture", {})
    time = raw.get("time_embedding", {})
    objective = raw.get("objective", {})
    expected_architecture = {
        "kind": "tokenwise_time_conditioned_swiglu",
        "norm": "layernorm",
        "time_conditioning": "multiplicative_gate",
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(f"architecture.{key} must be {expected!r}")
    expected_objective = {
        "kind": "flow_matching",
        "coordinates": "standardized_activation",
        "path": "linear_interpolation",
        "data_time": 0.0,
        "noise_time": 1.0,
        "target": "eps_minus_x0",
        "loss": "mean_squared_error",
    }
    for key, expected in expected_objective.items():
        if objective.get(key) != expected:
            raise ValueError(f"objective.{key} must be {expected!r}")
    if time.get("kind") != "continuous_sinusoidal":
        raise ValueError("time_embedding.kind must be 'continuous_sinusoidal'")
    if time.get("out_dim") != architecture.get("d_model"):
        raise ValueError("time_embedding.out_dim must equal architecture.d_model")
    activation_dim = architecture.get("activation_dim")
    cfg = FlowModelConfig(
        d_model=int(architecture["d_model"]),
        d_mlp=int(architecture["d_mlp"]),
        n_blocks=int(architecture["n_blocks"]),
        time_dim=int(time["dim"]),
        time_hidden=int(time["hidden"]),
        max_period=float(time["max_period"]),
        activation_dim=None if activation_dim is None else int(activation_dim),
    )
    expected_parameters = architecture.get("n_params_expected")
    if expected_parameters is not None and flow_parameter_count(cfg) != int(
        expected_parameters
    ):
        raise ValueError(
            f"architecture yields {flow_parameter_count(cfg)} parameters "
            f"!= declared {int(expected_parameters)}"
        )
    return cfg


def _validate_states(x0: torch.Tensor, noise: torch.Tensor) -> None:
    if x0.ndim != 2:
        raise ValueError(f"expected x0 with shape [batch, d_model], got {tuple(x0.shape)}")
    if x0.shape != noise.shape:
        raise ValueError(f"x0 shape {tuple(x0.shape)} != noise shape {tuple(noise.shape)}")
    if not x0.is_floating_point() or not noise.is_floating_point():
        raise ValueError("flow states must be floating-point tensors")
    if not torch.isfinite(x0).all() or not torch.isfinite(noise).all():
        raise ValueError("flow states must be finite")


def _time_for_state(t: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
    time = torch.as_tensor(t, device=x.device, dtype=x.dtype)
    if time.ndim == 0:
        pass
    elif time.ndim == 1 and time.shape[0] in (1, x.shape[0]):
        time = time[:, None]
    elif time.ndim == 2 and time.shape in ((1, 1), (x.shape[0], 1)):
        pass
    else:
        raise ValueError(
            "flow time must be scalar, [1], [batch], [1, 1], or [batch, 1]"
        )
    if not torch.isfinite(time).all() or not bool(((time >= 0.0) & (time <= 1.0)).all()):
        raise ValueError("flow time must be finite and inside [0, 1]")
    return time


def linear_interpolate(
    x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor | float
) -> torch.Tensor:
    """Return ``x_t = (1 - t) x_0 + t epsilon``."""

    _validate_states(x0, noise)
    time = _time_for_state(t, x0)
    return (1.0 - time) * x0 + time * noise


def velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Return the constant straight-path velocity ``epsilon - x_0``."""

    _validate_states(x0, noise)
    return noise - x0


def sample_flow_batch(
    x0: torch.Tensor,
    *,
    generator: torch.Generator,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> FlowBatch:
    """Sample per-activation flow times and Gaussian endpoints."""

    _validate_states(x0, torch.zeros_like(x0))
    if not (0.0 <= t_min < t_max <= 1.0):
        raise ValueError("expected 0 <= t_min < t_max <= 1")
    t = torch.rand(
        (x0.shape[0], 1),
        generator=generator,
        device=x0.device,
        dtype=x0.dtype,
    )
    t = t_min + (t_max - t_min) * t
    noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
    return FlowBatch(
        x_t=linear_interpolate(x0, noise, t),
        t=t,
        target_velocity=velocity_target(x0, noise),
    )


def flow_matching_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared velocity-prediction error over activations and dimensions."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}"
        )
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("velocity prediction and target must be finite")
    return (prediction - target).square().mean()


def sinusoidal_time_embedding(
    t: torch.Tensor, *, dim: int, max_period: float
) -> torch.Tensor:
    """Embed continuous flow time with geometrically spaced sine/cosine frequencies."""

    if dim <= 0 or dim % 2:
        raise ValueError("sinusoidal embedding dim must be positive and even")
    if not math.isfinite(max_period) or max_period <= 0:
        raise ValueError("max_period must be finite and positive")
    if not t.is_floating_point() or not torch.isfinite(t).all():
        raise ValueError("flow time must be a finite floating-point tensor")
    time = t.reshape(-1)
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / half
    )
    angles = time[:, None] * frequencies[None, :]
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)


class _TimeEmbedding(nn.Module):
    def __init__(self, cfg: FlowModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(cfg.time_dim, cfg.time_hidden)
        self.output = nn.Linear(cfg.time_hidden, cfg.d_model)
        self.activation = nn.SiLU()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        embedding = sinusoidal_time_embedding(
            t, dim=self.cfg.time_dim, max_period=self.cfg.max_period
        )
        return self.output(self.activation(self.input(embedding)))


class _FlowBlock(nn.Module):
    """Pre-norm residual SwiGLU block with multiplicative time gating."""

    def __init__(self, cfg: FlowModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.content = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.gate = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.time = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.down = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.activation = nn.SiLU()

    def forward(self, z: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(z)
        content = self.content(normalized)
        gate = self.gate(normalized) * self.time(time_embedding)
        return z + self.down(content * self.activation(gate))


class FlowMatcher(nn.Module):
    """Predict standardized-space velocity independently for each activation row."""

    def __init__(self, cfg: FlowModelConfig, normalizer: ActivationNormalizer) -> None:
        super().__init__()
        width = cfg.activation_width
        if normalizer.mean.shape != (width,):
            raise ValueError(
                f"normalizer width {normalizer.mean.shape[0]} != activation width {width}"
            )
        self.cfg = cfg
        self.normalizer = normalizer
        self.input = nn.Linear(width, cfg.d_model)
        self.time_embedding = _TimeEmbedding(cfg)
        self.blocks = nn.ModuleList(_FlowBlock(cfg) for _ in range(cfg.n_blocks))
        self.output_norm = nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model, width)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        if x_t.ndim != 2:
            raise ValueError(
                f"expected x_t with shape [batch, activation_dim], got {tuple(x_t.shape)}"
            )
        if x_t.shape[1] != self.cfg.activation_width:
            raise ValueError(
                f"expected activation_dim={self.cfg.activation_width}, got {x_t.shape[1]}"
            )
        if not x_t.is_floating_point() or not torch.isfinite(x_t).all():
            raise ValueError("x_t must be a finite floating-point tensor")
        time = _time_for_state(t, x_t).reshape(-1)
        if time.shape[0] == 1 and x_t.shape[0] != 1:
            time = time.expand(x_t.shape[0])
        embedded_time = self.time_embedding(time)
        z = self.input(x_t)
        for block in self.blocks:
            z = block(z, embedded_time)
        return self.output(self.output_norm(z))


def n_parameters(module: nn.Module) -> int:
    """Return the number of trainable and frozen parameter elements in a module."""

    return sum(parameter.numel() for parameter in module.parameters())
