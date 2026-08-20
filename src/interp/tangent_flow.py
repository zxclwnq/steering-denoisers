"""Constraint-preserving tangent flow: corrupt and denoise inside ``<x, v> = c``.

The closed generic/conditional branch trained on isotropic corruption

    x_t = (1 - t) x_0 + t eps

and only at *inference* asked the model to correct an activation while holding a
semantic coordinate fixed.  Every ``x_t`` seen in training therefore left the
constraint hyperplane, while every ``x_t`` seen in the projected inference arm
lay exactly on it.  This module removes that train/test geometry mismatch by
making the tangent path the *training* distribution.

## The tangent path

For a clean standardized activation ``x_0``, a canonical standardized unit
direction ``v`` and its coordinate ``c = <x_0, v>``:

    x_par  = c v
    x_perp = x_0 - c v
    eps_perp = eps - <eps, v> v

    x_t = c v + (1 - t) x_perp + t eps_perp
        = (1 - t) x_0 + t (eps_perp + c v)

so ``<x_t, v> = c`` for *every* ``t``.  The velocity target is tangent by
construction:

    u* = eps_perp - x_perp,        <u*, v> = 0

## Enforcing tangency

The network predicts an unconstrained vector.  The velocity actually used for
the training loss, for Euler integration and for reconstruction is analytically
projected

    u_tangent = u - <u, v> v

so the constraint is an exact invariant rather than something the network is
hoped to learn.  The raw parallel component ``<u_raw, v>`` is measured *before*
projection and recorded as a diagnostic; it never moves the activation.

**Tangent inference always projects analytically.**  There is no inference-time
switch to disable it: the semantic-coordinate invariant is part of the method,
not an option.  The training-time ablation of
docs/PROPOSAL_CONSTRAINT_PRESERVING_TANGENT_FLOW.md §3.1 lives in the training
config (``flow_objective.output_projection``), where it genuinely changes the
loss the model is fitted against; that model is still integrated with tangent
velocities at inference.

This module contains no GPT-2, SAE, dataset, or evaluation-split logic, and it
never reads protected evaluation directions.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .conditional_flow import (
    ConditionalFlowMatcher,
    TrainingDirectionPool,
    _unit_rows,
    clamp_seed,
    standardized_hyperplane,
)
from .constrained_flow import hyperplane_project
from .flow_core import ActivationNormalizer, _time_for_state
from .flow_sampling import euler_time_grid

# Objective identifiers stored in checkpoints and training configs.  Every
# checkpoint written before this branch existed is implicitly ``ISOTROPIC``.
ISOTROPIC_OBJECTIVE = "isotropic"
TANGENT_OBJECTIVE = "tangent_constraint_preserving"
# Post-stop experiment A (docs/POST_STOP_PROTOCOL_2026-08-19.md §2): the same
# constraint-preserving geometry on a quarter-circle rather than a chord, so the
# orthogonal scale is preserved instead of shrinking to (1-t)^2 + t^2.
VP_TANGENT_OBJECTIVE = "tangent_variance_preserving"
# Objectives whose batches carry a (direction, coordinate) constraint and whose
# velocities are analytically projected.
TANGENT_OBJECTIVES = (TANGENT_OBJECTIVE, VP_TANGENT_OBJECTIVE)
FLOW_OBJECTIVES = (ISOTROPIC_OBJECTIVE, *TANGENT_OBJECTIVES)


# --------------------------------------------------------------------------
# tangent algebra
# --------------------------------------------------------------------------


# Every formula in this module assumes ||v|| = 1 exactly. The shared
# ``_unit_rows`` guard accepts 1e-3, which is far too loose here: a clamp of
# displacement d onto a direction of norm 1+delta lands at a coordinate error of
# roughly 2*d*delta, so delta = 1e-3 with d = 10 misses the requested coordinate
# by ~0.02 -- twenty times the 1e-3 tolerance the T2 arm-matching gate allows.
# The real training pool is unit to 2.7e-7, so this bound is comfortably met by
# the vectors actually used and does not require renormalizing them.
UNIT_NORM_TOLERANCE = 1e-6


def unit_directions(
    direction: torch.Tensor, *, tolerance: float = UNIT_NORM_TOLERANCE
) -> torch.Tensor:
    """Validate a ``[d]`` / ``[batch, d]`` direction tightly enough for the algebra.

    Delegates the shape and finiteness checks to the shared ``_unit_rows``, then
    applies the stricter norm bound the tangent formulas actually need. Vectors
    are validated, never rewritten: a pool direction is a canonicalized artifact
    and silently renormalizing it would break its digest correspondence.
    """

    rows = _unit_rows(direction)
    deviation = (rows.double().norm(dim=-1) - 1.0).abs()
    worst = float(deviation.max())
    if worst > tolerance:
        raise ValueError(
            f"tangent geometry needs unit directions to within {tolerance:g}; "
            f"worst row deviates by {worst:.3e}. Normalize upstream rather than "
            "relying on the looser shared guard."
        )
    return rows


def coordinate(x: torch.Tensor, v_x: torch.Tensor) -> torch.Tensor:
    """Return ``<x, v_x>`` as a ``[rows, 1]`` column."""

    if x.ndim != 2 or v_x.ndim != 2 or v_x.shape[-1] != x.shape[-1]:
        raise ValueError("coordinate needs [rows, d] state and matching [rows, d] direction")
    return (x * v_x).sum(dim=-1, keepdim=True)


def tangent_project(u: torch.Tensor, v_x: torch.Tensor) -> torch.Tensor:
    """Return ``u - <u, v_x> v_x``: the linear projection into ``v_x``-perp.

    This is ``hyperplane_project`` with a zero offset; it is spelled separately
    because a *velocity* has no affine part and must never acquire one.
    """

    return u - coordinate(u, v_x) * v_x


@dataclass(frozen=True)
class TangentFlowBatch:
    """One tangent-flow training batch, in standardized coordinates."""

    x0: torch.Tensor
    v_x: torch.Tensor
    c_x: torch.Tensor
    t: torch.Tensor
    epsilon: torch.Tensor
    epsilon_perp: torch.Tensor
    x_t: torch.Tensor
    target_velocity: torch.Tensor
    # Which constraint-preserving path produced ``x_t``. Carried on the batch so
    # the trainer's validation bins re-derive states on the same geometry the
    # batch was drawn from, rather than assuming the linear one.
    objective: str = TANGENT_OBJECTIVE


def _validate_tangent_states(
    x0: torch.Tensor,
    v_x: torch.Tensor,
    c_x: torch.Tensor,
    epsilon: torch.Tensor,
    t: torch.Tensor | float,
) -> torch.Tensor:
    """Shared shape/finiteness guard for every constraint-preserving path."""

    if x0.ndim != 2 or epsilon.shape != x0.shape:
        raise ValueError("x0 and epsilon must be same-shaped [rows, d] tensors")
    if v_x.shape[-1] != x0.shape[-1] or v_x.ndim != 2:
        raise ValueError("v_x must be [rows, d] or [1, d] matching the state width")
    if c_x.ndim != 2 or c_x.shape[-1] != 1:
        raise ValueError("c_x must be a [rows, 1] column")
    if not torch.isfinite(x0).all() or not torch.isfinite(epsilon).all():
        raise ValueError("tangent flow states must be finite")
    return _time_for_state(t, x0)


def tangent_flow_states(
    x0: torch.Tensor,
    v_x: torch.Tensor,
    c_x: torch.Tensor,
    epsilon: torch.Tensor,
    t: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(eps_perp, x_t, u_target)`` for the constraint-preserving path.

    Implements the boxed equations directly rather than routing through the
    isotropic sampler:

        eps_perp = eps - <eps, v> v
        x_t      = c v + (1 - t)(x0 - c v) + t eps_perp
        u_target = eps_perp - (x0 - c v)
    """

    time = _validate_tangent_states(x0, v_x, c_x, epsilon, t)

    parallel = c_x * v_x
    x0_perp = x0 - parallel
    epsilon_perp = tangent_project(epsilon, v_x)
    x_t = parallel + (1.0 - time) * x0_perp + time * epsilon_perp
    return epsilon_perp, x_t, epsilon_perp - x0_perp


_HALF_PI = math.pi / 2.0


def vp_tangent_flow_states(
    x0: torch.Tensor,
    v_x: torch.Tensor,
    c_x: torch.Tensor,
    epsilon: torch.Tensor,
    t: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(eps_perp, x_t, u_target)`` for the variance-preserving path.

    Same constraint, different interpolation. With ``theta = (pi/2) t`` the
    orthogonal part travels a quarter circle instead of a chord:

        eps_perp = eps - <eps, v> v
        x_t      = c v + cos(theta) (x0 - c v) + sin(theta) eps_perp
        u_target = (pi/2) ( -sin(theta) (x0 - c v) + cos(theta) eps_perp )

    ``<x_t, v> = c`` still holds for every ``t`` and ``<u_target, v> = 0`` still
    holds exactly, because both moving terms are v-orthogonal. The difference
    from :func:`tangent_flow_states` is scale: ``cos^2 + sin^2 = 1`` keeps the
    orthogonal variance constant, where the chord shrinks it to
    ``(1-t)^2 + t^2`` -- a factor of two at ``t = 0.5``.

    ``u_target`` is the genuine time derivative of ``x_t``, so the ``pi/2``
    factor is part of the target and not a normalization choice. It makes the
    velocity scale differ from the linear path's, which is why ``val_flow_mse``
    is not comparable across the two objectives.
    """

    time = _validate_tangent_states(x0, v_x, c_x, epsilon, t)
    theta = _HALF_PI * time
    parallel = c_x * v_x
    x0_perp = x0 - parallel
    epsilon_perp = tangent_project(epsilon, v_x)
    cos, sin = torch.cos(theta), torch.sin(theta)
    x_t = parallel + cos * x0_perp + sin * epsilon_perp
    target = _HALF_PI * (cos * epsilon_perp - sin * x0_perp)
    return epsilon_perp, x_t, target


_TANGENT_PATHS = {
    TANGENT_OBJECTIVE: tangent_flow_states,
    VP_TANGENT_OBJECTIVE: vp_tangent_flow_states,
}


def tangent_path_states(
    x0: torch.Tensor,
    v_x: torch.Tensor,
    c_x: torch.Tensor,
    epsilon: torch.Tensor,
    t: torch.Tensor | float,
    *,
    objective: str = TANGENT_OBJECTIVE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch to the corruption path named by ``objective``."""

    try:
        path = _TANGENT_PATHS[objective]
    except KeyError:
        raise ValueError(
            f"{objective!r} is not a constraint-preserving objective; "
            f"expected one of {TANGENT_OBJECTIVES}"
        ) from None
    return path(x0, v_x, c_x, epsilon, t)


def matched_vp_time(t_linear: float) -> float:
    """Map a linear-path time to the variance-preserving time of equal severity.

    Severity is the orthogonal noise-to-signal ratio: ``t/(1-t)`` on the chord,
    ``tan((pi/2) t)`` on the quarter circle. Equating them gives

        t_vp = (2/pi) arctan( t_lin / (1 - t_lin) ).

    Comparing the two paths at equal ``t`` would compare different amounts of
    corruption; every primary Experiment A claim is made at equal severity.
    ``t = 0``, ``t = 0.5`` and ``t = 1`` are fixed points of this map.
    """

    if not math.isfinite(t_linear) or not 0.0 <= t_linear <= 1.0:
        raise ValueError(f"linear path time must lie in [0, 1], got {t_linear!r}")
    if t_linear == 1.0:
        return 1.0
    return (2.0 / math.pi) * math.atan(t_linear / (1.0 - t_linear))


def matched_linear_time(t_vp: float) -> float:
    """Inverse of :func:`matched_vp_time`: ``t_lin = r / (1 + r)``, ``r = tan(theta)``."""

    if not math.isfinite(t_vp) or not 0.0 <= t_vp <= 1.0:
        raise ValueError(f"variance-preserving time must lie in [0, 1], got {t_vp!r}")
    if t_vp == 1.0:
        return 1.0
    ratio = math.tan(_HALF_PI * t_vp)
    return ratio / (1.0 + ratio)


def sample_tangent_flow_batch(
    h: torch.Tensor,
    *,
    normalizer: ActivationNormalizer,
    pool: TrainingDirectionPool,
    generator: torch.Generator,
    t_min: float = 0.0,
    t_max: float = 1.0,
    objective: str = TANGENT_OBJECTIVE,
) -> TangentFlowBatch:
    """Draw one tangent-corruption batch from raw activations.

    Directions come only from the training-only pool; the conditioned coordinate
    is each activation's own naturally occurring ``<h, v>``, mapped into
    standardized space by the existing exact hyperplane transform.
    """

    if h.ndim != 2 or not h.is_floating_point() or not torch.isfinite(h).all():
        raise ValueError("h must be a finite floating-point [rows, d] tensor")
    if not (0.0 <= t_min < t_max <= 1.0):
        raise ValueError("expected 0 <= t_min < t_max <= 1")
    if pool.directions.device != h.device or pool.directions.dtype != h.dtype:
        raise ValueError(
            f"direction pool is on {pool.directions.device}/{pool.directions.dtype}, "
            f"activations are on {h.device}/{h.dtype}; move the pool with .to()"
        )
    directions = unit_directions(pool.sample(h.shape[0], generator=generator))
    c_raw = (h * directions).sum(dim=-1, keepdim=True)
    v_x, c_x = standardized_hyperplane(normalizer, directions, c_raw)

    x0 = normalizer.normalize(h)
    t = torch.rand((h.shape[0], 1), generator=generator, device=h.device, dtype=h.dtype)
    t = t_min + (t_max - t_min) * t
    epsilon = torch.randn(h.shape, generator=generator, device=h.device, dtype=h.dtype)
    epsilon_perp, x_t, target = tangent_path_states(
        x0, v_x, c_x, epsilon, t, objective=objective
    )
    return TangentFlowBatch(
        x0=x0,
        v_x=v_x,
        c_x=c_x,
        t=t,
        epsilon=epsilon,
        epsilon_perp=epsilon_perp,
        x_t=x_t,
        target_velocity=target,
        objective=objective,
    )


def raw_velocity_field(
    model: ConditionalFlowMatcher, v_x: torch.Tensor, c_x: torch.Tensor
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Freeze the condition and expose the model's **unprojected** velocity.

    Projection deliberately happens in :func:`tangent_reverse_euler`, not here,
    so the parallel component can be measured on the genuine network output
    before it is removed. A field that projected on the way out would make the
    ``raw_parallel_velocity`` diagnostic measure its own projection and read as
    float noise regardless of what the model actually predicted.
    """

    def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return model(x, t, v_x, c_x)

    return velocity


# --------------------------------------------------------------------------
# inference: hard clamp, then tangent SDEdit
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TangentFlowOutput:
    """One clamp+tangent-flow result and the diagnostics that police it."""

    activation: torch.Tensor
    seed: torch.Tensor
    requested_coordinate: torch.Tensor
    realised_coordinate: torch.Tensor
    network_evaluations: int
    projections: int
    diagnostics: dict[str, float | int | bool]


@torch.no_grad()
def tangent_reverse_euler(
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    v_x: torch.Tensor,
    c_x: torch.Tensor,
    *,
    t_start: float,
    nfe: int,
    safeguard_projection: bool,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Integrate to ``t = 0`` with every step's velocity analytically tangent.

    ``velocity`` must return the model's **raw**, unprojected prediction. This
    function measures ``<u_raw, v>`` first and projects second, so the reported
    parallel component describes the network output rather than the projection's
    own rounding error. Projection is unconditional: tangent inference has no
    switch to disable it.

    ``safeguard_projection`` re-imposes the hyperplane after each step.  It is a
    numerical-stability guard only: the pre-projection drift is measured and
    reported so a reader can confirm it corrects float error and is not the
    mechanism producing semantic preservation.  Projections are analytic and are
    never counted as network evaluations.
    """

    times = euler_time_grid(t_start, nfe, device=x_t.device, dtype=x_t.dtype)
    if t_start == 0.0:
        return x_t, {
            "network_evaluations": 0,
            "projections": 0,
            "max_pre_projection_drift": 0.0,
            "max_coordinate_drift": 0.0,
            "raw_parallel_velocity_norm_mean": 0.0,
        }

    x = x_t
    evaluations = 0
    projections = 0
    pre_drift = 0.0
    drift = 0.0
    parallel_norms: list[float] = []
    for current, following in zip(times[:-1], times[1:], strict=True):
        t = current.expand(x.shape[0], 1)
        raw = velocity(x, t)
        evaluations += 1
        if raw.shape != x.shape or not torch.isfinite(raw).all():
            raise ValueError("tangent velocity must be finite and state-shaped")
        # Measure the parallel component of the RAW model output, then discard it.
        parallel_norms.append(float(coordinate(raw, v_x).abs().double().mean()))
        used = tangent_project(raw, v_x)
        x = x + (following - current) * used
        if not torch.isfinite(x).all():
            raise ValueError("tangent reverse Euler state became non-finite")
        step_drift = float((coordinate(x, v_x) - c_x).abs().double().max())
        pre_drift = max(pre_drift, step_drift)
        if safeguard_projection:
            x = hyperplane_project(x, v_x, c_x)
            projections += 1
            step_drift = float((coordinate(x, v_x) - c_x).abs().double().max())
        drift = max(drift, step_drift)
    return x, {
        "network_evaluations": evaluations,
        "projections": projections,
        "max_pre_projection_drift": pre_drift,
        "max_coordinate_drift": drift,
        "raw_parallel_velocity_norm_mean": (
            sum(parallel_norms) / len(parallel_norms) if parallel_norms else 0.0
        ),
    }


@torch.no_grad()
def clamp_then_tangent_flow(
    model: ConditionalFlowMatcher,
    h: torch.Tensor,
    direction: torch.Tensor,
    c_target: torch.Tensor,
    *,
    noise: torch.Tensor,
    t_start: float,
    nfe: int,
    safeguard_projection: bool = True,
    objective: str = TANGENT_OBJECTIVE,
) -> TangentFlowOutput:
    """Hard-clamp the semantic coordinate, then naturalize only the orthogonal part.

    ``h_clamp = h + (c_target - <h, v>) v`` fixes the coordinate in raw space.
    The constraint is carried into standardized space by the existing exact
    hyperplane transform, tangent-noised to ``t_start``, and reverse-integrated
    with analytically tangent velocities.  ``t_start = 0`` returns the clamp
    exactly, with zero network evaluations.

    There is deliberately no ``output_projection`` argument: tangent inference
    always projects. The training-time variant lives in the training config.
    """

    if h.ndim != 2 or h.numel() == 0 or not h.is_floating_point():
        raise ValueError("h must be a nonempty floating-point [rows, d] tensor")
    if not torch.isfinite(h).all():
        raise ValueError("h must be finite")
    if noise.shape != h.shape or not torch.isfinite(noise).all():
        raise ValueError("noise must be a same-shaped finite tensor")
    if objective not in TANGENT_OBJECTIVES:
        raise ValueError(f"{objective!r} is not a constraint-preserving objective")
    vector = unit_directions(direction).to(device=h.device, dtype=h.dtype)
    seed = clamp_seed(h, vector, c_target)
    requested = torch.as_tensor(c_target, device=h.device, dtype=h.dtype)
    requested = requested.reshape(-1, 1).expand(h.shape[0], 1)

    if t_start == 0.0:
        # Validate the schedule, evaluate nothing, and return the clamp exactly.
        euler_time_grid(t_start, nfe, device=h.device, dtype=h.dtype)
        activation = seed
        stats: dict[str, float | int] = {
            "network_evaluations": 0,
            "projections": 0,
            "max_pre_projection_drift": 0.0,
            "max_coordinate_drift": 0.0,
            "raw_parallel_velocity_norm_mean": 0.0,
        }
    else:
        v_x, c_x = standardized_hyperplane(model.normalizer, vector, requested)
        if v_x.shape[0] == 1 and h.shape[0] != 1:
            v_x = v_x.expand(h.shape[0], v_x.shape[1])
        if c_x.shape[0] == 1 and h.shape[0] != 1:
            c_x = c_x.expand(h.shape[0], 1)
        v_x = v_x.to(device=h.device, dtype=h.dtype).contiguous()
        c_x = c_x.to(device=h.device, dtype=h.dtype).contiguous()

        x_clamp = model.normalizer.normalize(seed)
        _, x_t, _ = tangent_path_states(
            x_clamp, v_x, c_x, noise, t_start, objective=objective
        )
        sampled, stats = tangent_reverse_euler(
            raw_velocity_field(model, v_x, c_x),
            x_t,
            v_x,
            c_x,
            t_start=t_start,
            nfe=nfe,
            safeguard_projection=safeguard_projection,
        )
        activation = model.normalizer.denormalize(sampled)
        if not torch.isfinite(activation).all():
            raise ValueError("tangent flow produced a non-finite activation")

    realised = (activation * vector).sum(dim=-1, keepdim=True)
    correction = activation - seed
    parallel = (correction * vector).sum(dim=-1, keepdim=True)
    orthogonal = correction - parallel * vector
    diagnostics: dict[str, float | int | bool] = {
        "n_rows": int(h.shape[0]),
        "t_start": float(t_start),
        "requested_nfe": int(nfe) if t_start != 0.0 else 0,
        "coordinate_abs_error_mean": float((realised - requested).abs().double().mean()),
        "coordinate_abs_error_max": float((realised - requested).abs().double().max()),
        "orthogonal_correction_norm_mean": float(orthogonal.norm(dim=-1).double().mean()),
        "parallel_correction_norm_mean": float(parallel.abs().double().mean()),
        "correction_norm_mean": float(correction.norm(dim=-1).double().mean()),
        "safeguard_projection": bool(safeguard_projection),
        "inference_output_projection": "always",
        "objective": objective,
        **stats,
    }
    return TangentFlowOutput(
        activation=activation,
        seed=seed,
        requested_coordinate=requested.squeeze(-1),
        realised_coordinate=realised.squeeze(-1),
        network_evaluations=int(stats["network_evaluations"]),
        projections=int(stats["projections"]),
        diagnostics=diagnostics,
    )
