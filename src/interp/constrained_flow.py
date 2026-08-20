"""Constrained conditional flow: hold a semantic coordinate, naturalize the rest.

The question this module exists to answer is whether the learned prior can
improve the components of an activation *orthogonal* to a requested direction
while the coordinate along that direction is held exactly fixed.

## Why ordinary SDEdit cannot be reused unchanged

In standardized coordinates the constraint is ``<x, v_x> = c_x`` with
``||v_x|| = 1``. Ordinary forward noising gives

    x_t = (1 - t) x_0 + t eps
    <x_t, v_x> = (1 - t) c_x + t <eps, v_x>

which violates the constraint twice: the coordinate is shrunk by ``(1 - t)`` and
Gaussian mass is injected along ``v_x``.

## Constrained forward noising

Split ``x_0 = c_x v_x + x_0_perp`` and noise only the orthogonal part while
pinning the parallel part:

    x_t = c_x v_x + (1 - t)(x_0 - c_x v_x) + t eps_perp
        = (1 - t) x_0 + t (eps_perp + c_x v_x)

so ``<x_t, v_x> = (1 - t) c_x + t c_x = c_x`` for every ``t``.

The closed form is ordinary linear interpolation toward a modified endpoint

    eps' = eps_perp + c_x v_x = eps - (<eps, v_x> - c_x) v_x

which is exactly the affine projection of ``eps`` onto the constraint
hyperplane. Constrained noising is therefore ordinary SDEdit with the Gaussian
endpoint projected onto the hyperplane, and nothing else changes.

Reverse integration still leaves the hyperplane, because the learned velocity
has an unconstrained parallel component. The projected arm removes that by
re-projecting after every Euler step; projections are analytic and are never
counted as network evaluations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .conditional_flow import (
    ConditionalFlowMatcher,
    clamp_seed,
    standardized_hyperplane,
)
from .flow_core import linear_interpolate
from .flow_sampling import euler_time_grid

ARMS = ("clamp_only", "sdedit", "tangent", "projected")


def hyperplane_project(x: torch.Tensor, v_x: torch.Tensor, c_x: torch.Tensor) -> torch.Tensor:
    """Return ``x + (c_x - <x, v_x>) v_x``: the affine projection onto the constraint."""

    if x.ndim != 2 or v_x.shape != x.shape or c_x.shape != (x.shape[0], 1):
        raise ValueError("projection needs [rows, d] state, matching v_x, and [rows, 1] c_x")
    residual = c_x - (x * v_x).sum(dim=-1, keepdim=True)
    return x + residual * v_x


def constrained_noise_endpoint(
    noise: torch.Tensor, v_x: torch.Tensor, c_x: torch.Tensor
) -> torch.Tensor:
    """Project the Gaussian endpoint onto the constraint hyperplane.

    This is the only change constrained noising makes: ``eps -> eps_perp + c_x v_x``.
    """

    return hyperplane_project(noise, v_x, c_x)


def constrained_partial_noise(
    x0: torch.Tensor,
    noise: torch.Tensor,
    v_x: torch.Tensor,
    c_x: torch.Tensor,
    *,
    t_start: float,
) -> torch.Tensor:
    """Forward-noise while leaving ``<x_t, v_x> = c_x`` exactly invariant."""

    endpoint = constrained_noise_endpoint(noise, v_x, c_x)
    return linear_interpolate(x0, endpoint, t_start)


@dataclass(frozen=True)
class ConstrainedFlowOutput:
    activation: torch.Tensor
    seed: torch.Tensor
    network_evaluations: int
    projections: int


@torch.no_grad()
def constrained_reverse_euler(
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    v_x: torch.Tensor,
    c_x: torch.Tensor,
    *,
    t_start: float,
    nfe: int,
    project_each_step: bool,
) -> tuple[torch.Tensor, int, int]:
    """Reverse-integrate, optionally re-projecting onto the hyperplane each step."""

    times = euler_time_grid(t_start, nfe, device=x_t.device, dtype=x_t.dtype)
    if t_start == 0.0:
        return x_t, 0, 0
    x = x_t
    evaluations = 0
    projections = 0
    for current, following in zip(times[:-1], times[1:], strict=True):
        t = current.expand(x.shape[0], 1)
        predicted = velocity(x, t)
        evaluations += 1
        if predicted.shape != x.shape or not torch.isfinite(predicted).all():
            raise ValueError("constrained velocity must be finite and state-shaped")
        x = x + (following - current) * predicted
        if project_each_step:
            # Analytic constraint restoration: never a network evaluation.
            x = hyperplane_project(x, v_x, c_x)
            projections += 1
        if not torch.isfinite(x).all():
            raise ValueError("constrained reverse Euler state became non-finite")
    return x, evaluations, projections


@torch.no_grad()
def clamp_seed_flow(
    model: ConditionalFlowMatcher,
    h: torch.Tensor,
    direction: torch.Tensor,
    c_target: torch.Tensor,
    *,
    noise: torch.Tensor,
    t_start: float,
    nfe: int,
    arm: str,
) -> ConstrainedFlowOutput:
    """Hard-clamp the coordinate, then run one of the four prospectively defined arms.

    ``clamp_only``  the clamp itself, no flow, zero network evaluations;
    ``sdedit``      ordinary conditional SDEdit seeded from the clamp;
    ``tangent``     constrained forward noising, unconstrained reverse integration;
    ``projected``   constrained noising plus per-step hyperplane projection.
    """

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    if h.ndim != 2 or h.numel() == 0 or not h.is_floating_point():
        raise ValueError("h must be a nonempty floating-point [rows, d] tensor")
    if not torch.isfinite(h).all():
        raise ValueError("h must be finite")
    if noise.shape != h.shape or not torch.isfinite(noise).all():
        raise ValueError("noise must be a same-shaped finite tensor")

    seed = clamp_seed(h, direction, c_target)
    if arm == "clamp_only":
        return ConstrainedFlowOutput(
            activation=seed, seed=seed, network_evaluations=0, projections=0
        )

    v_x, c_x = standardized_hyperplane(model.normalizer, direction, c_target)
    if v_x.shape[0] == 1 and h.shape[0] != 1:
        v_x = v_x.expand(h.shape[0], v_x.shape[1])
    if c_x.shape[0] == 1 and h.shape[0] != 1:
        c_x = c_x.expand(h.shape[0], 1)
    v_x = v_x.to(device=h.device, dtype=h.dtype).contiguous()
    c_x = c_x.to(device=h.device, dtype=h.dtype).contiguous()

    x0 = model.normalizer.normalize(seed)
    if arm == "sdedit":
        x_t = linear_interpolate(x0, noise, t_start)
    else:
        x_t = constrained_partial_noise(x0, noise, v_x, c_x, t_start=t_start)

    sampled, evaluations, projections = constrained_reverse_euler(
        model.velocity_field(v_x, c_x),
        x_t,
        v_x,
        c_x,
        t_start=t_start,
        nfe=nfe,
        project_each_step=(arm == "projected"),
    )
    activation = model.normalizer.denormalize(sampled)
    if not torch.isfinite(activation).all():
        raise ValueError(f"{arm} produced a non-finite activation")
    return ConstrainedFlowOutput(
        activation=activation,
        seed=seed,
        network_evaluations=evaluations,
        projections=projections,
    )


@torch.no_grad()
def correction_decomposition(
    seed: torch.Tensor, out: torch.Tensor, direction: torch.Tensor
) -> dict[str, float]:
    """Split ``h_out - h_clamp`` into components parallel and orthogonal to ``v``.

    For the tangent and projected arms the parallel part is the quantity that
    should be near zero by construction, and the orthogonal part is the only
    channel through which the prior can improve the activation.
    """

    delta = out - seed
    parallel_signed = (delta * direction).sum(dim=-1, keepdim=True)
    orthogonal = delta - parallel_signed * direction
    return {
        "n_rows": int(delta.shape[0]),
        "correction_norm_mean": float(delta.norm(dim=-1).double().mean()),
        "parallel_norm_mean": float(parallel_signed.abs().double().mean()),
        "orthogonal_norm_mean": float(orthogonal.norm(dim=-1).double().mean()),
        "parallel_signed_mean": float(parallel_signed.double().mean()),
    }


def merge_decompositions(parts: list[dict[str, float]]) -> dict[str, float]:
    """Row-count-weighted mean across hook batches."""

    total = sum(int(part["n_rows"]) for part in parts)
    merged: dict[str, float] = {"n_rows": total}
    for key in parts[0]:
        if key == "n_rows":
            continue
        merged[key] = sum(
            float(part[key]) * int(part["n_rows"]) for part in parts
        ) / total
    return merged
