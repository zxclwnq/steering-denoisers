"""Pure SDEdit-style partial noising and reverse explicit-Euler integration.

All states in ``partial_noise``, ``reverse_euler``, and ``sdedit_sample`` are in
standardized activation coordinates.  Noise is supplied by the caller so matched
experimental arms can reuse the identical Gaussian endpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch

from .flow_core import ActivationNormalizer, linear_interpolate

VelocityField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class NormalizedVelocityModel(Protocol):
    normalizer: ActivationNormalizer

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...


def euler_time_grid(
    t_start: float,
    nfe: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``nfe + 1`` uniform times from ``t_start`` down to zero."""

    if not isinstance(nfe, int) or isinstance(nfe, bool) or nfe < 1:
        raise ValueError(f"nfe must be a positive integer, got {nfe!r}")
    if not 0.0 <= t_start <= 1.0:
        raise ValueError(f"t_start must lie inside [0, 1], got {t_start}")
    if not dtype.is_floating_point:
        raise ValueError("Euler time-grid dtype must be floating point")
    return torch.linspace(t_start, 0.0, nfe + 1, device=device, dtype=dtype)


def partial_noise(x: torch.Tensor, noise: torch.Tensor, *, t_start: float) -> torch.Tensor:
    """Move a standardized activation toward an explicit Gaussian endpoint."""

    return linear_interpolate(x, noise, t_start)


@torch.no_grad()
def reverse_euler(
    velocity: VelocityField,
    x_t: torch.Tensor,
    *,
    t_start: float,
    nfe: int,
) -> torch.Tensor:
    """Integrate from ``t_start`` to data time zero with exactly ``nfe`` calls."""

    times = euler_time_grid(t_start, nfe, device=x_t.device, dtype=x_t.dtype)
    if x_t.ndim != 2 or not x_t.is_floating_point() or not torch.isfinite(x_t).all():
        raise ValueError("x_t must be a finite floating-point [batch, d_model] tensor")
    if t_start == 0.0:
        return x_t

    x = x_t
    for current, following in zip(times[:-1], times[1:], strict=True):
        t = current.expand(x.shape[0], 1)
        predicted = velocity(x, t)
        if predicted.shape != x.shape:
            raise ValueError(
                f"velocity shape {tuple(predicted.shape)} != state shape {tuple(x.shape)}"
            )
        if not predicted.is_floating_point() or not torch.isfinite(predicted).all():
            raise ValueError("predicted velocity must be finite and floating point")
        x = x + (following - current) * predicted
        if not torch.isfinite(x).all():
            raise ValueError("reverse Euler state became non-finite")
    return x


@torch.no_grad()
def sdedit_sample(
    velocity: VelocityField,
    x: torch.Tensor,
    *,
    noise: torch.Tensor,
    t_start: float,
    nfe: int,
) -> torch.Tensor:
    """Partially noise ``x`` and reverse-integrate it to the data endpoint."""

    if t_start == 0.0:
        # Validate the schedule even on the identity path, but perform no interpolation
        # or model evaluation. This makes zero-time semantics explicit and exact.
        euler_time_grid(t_start, nfe, device=x.device, dtype=x.dtype)
        if x.shape != noise.shape:
            raise ValueError(f"x shape {tuple(x.shape)} != noise shape {tuple(noise.shape)}")
        return x
    x_t = partial_noise(x, noise, t_start=t_start)
    return reverse_euler(velocity, x_t, t_start=t_start, nfe=nfe)


@torch.no_grad()
def flow_correct(
    model: NormalizedVelocityModel,
    h: torch.Tensor,
    *,
    noise: torch.Tensor,
    t_start: float,
    nfe: int,
) -> torch.Tensor:
    """Standardize raw activations, sample the flow, and denormalize the result."""

    if not hasattr(model, "normalizer"):
        raise ValueError("flow model must carry its train-split normalizer")
    if t_start == 0.0:
        euler_time_grid(t_start, nfe, device=h.device, dtype=h.dtype)
        if h.shape != noise.shape:
            raise ValueError(f"h shape {tuple(h.shape)} != noise shape {tuple(noise.shape)}")
        return h
    x = model.normalizer.normalize(h)
    sampled = sdedit_sample(model, x, noise=noise, t_start=t_start, nfe=nfe)
    return model.normalizer.denormalize(sampled)
