"""Denoiser trained on steering-like corruption (post-stop experiment B).

Every prior in this programme was trained to invert *Gaussian* corruption and
then asked, at inference, to repair a *steering* edit. Cold Diffusion and SPAR
suggest training the restoration model on the degradation it will actually face:

    z = h + delta v,        D(z) -> h

with ``v`` drawn only from the training-only direction pool and ``delta`` from a
distribution frozen before any DEV result exists.

## The failure mode this module is built to expose

**The obvious solution is to undo the steering.** A model that learns
``D(h + delta v) = h`` perfectly has not repaired anything -- it has removed the
intervention. Comparing it to additive steering at equal *nominal* alpha would
score that removal as a quality win.

So every output carries its realised concept strength

    alpha_eff = <h' - h_clean, v>

and the decisive comparison in `docs/POST_STOP_PROTOCOL_2026-08-19.md` §B.5 is
made against additive and scalar-shrinkage controls at **equal alpha_eff**, not
at equal nominal alpha. Partial correction

    h'_lambda = z + lambda ( D(z) - z )

sweeps the strength axis over a frozen lambda grid so the matched comparison has
points to interpolate between; no single lambda is chosen after the fact.

## What this is not

It is not a flow. There is no time, no path, no integration, no NFE, and no
velocity. ``D`` sees only ``z``: it is not told ``v``, not told ``delta``, and
not told how corrupted its input is. It is a cheap residual MLP, deliberately
built from the same block as the frozen flow trunk so capacity is comparable.

This module contains no GPT-2, SAE, dataset, or evaluation-split logic, and it
never reads protected evaluation directions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .conditional_flow import TrainingDirectionPool, _unit_rows
from .flow_core import ActivationNormalizer, FlowModelConfig, _FlowBlock

STEERING_DENOISER_OBJECTIVE = "steering_corruption_denoising"


# --------------------------------------------------------------------------
# frozen corruption distribution
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SteeringCorruptionSpec:
    """The corruption distribution and inference grid, frozen before any result.

    ``delta_max`` was chosen from the already-published hard-clamp displacements
    of `results/tangent_t2_v1/raw_rows.npz` -- natural coordinate sd 7.52, clamp
    displacement sd 6.18, ``|delta|`` 99th percentile 21.8, observed maximum
    32.0 -- so uniform support on ``[-32, +32]`` covers the entire working range
    with room to spare, is symmetric, and includes ``delta = 0``. No other
    distribution is tried, and this field is not tuned after a result is seen.
    """

    version: str = "steering_corruption_v1"
    distribution: str = "uniform_symmetric"
    delta_max: float = 32.0
    # Frozen partial-correction grid. Reported in full; never narrowed post hoc.
    lambda_grid: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

    def __post_init__(self) -> None:
        if not self.delta_max > 0.0:
            raise ValueError("delta_max must be positive")
        if not self.lambda_grid or any(not 0.0 < lam <= 1.0 for lam in self.lambda_grid):
            raise ValueError("every lambda must lie in (0, 1]")
        if sorted(self.lambda_grid) != list(self.lambda_grid):
            raise ValueError("the lambda grid must be stated in increasing order")

    def sample_delta(
        self,
        rows: int,
        *,
        generator: torch.Generator,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Draw ``[rows, 1]`` steering strengths, uniform on ``[-delta_max, delta_max]``."""

        uniform = torch.rand((rows, 1), generator=generator, device=device, dtype=dtype)
        return (2.0 * uniform - 1.0) * self.delta_max


STEERING_CORRUPTION_SPEC = SteeringCorruptionSpec()


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


class SteeringDenoiser(nn.Module):
    """``D(z) = z + f(z)`` in standardized coordinates, with no conditioning.

    The trunk is the frozen flow block (`interp.flow_core._FlowBlock`) so the
    capacity is directly comparable to the priors this experiment is measured
    against. The block's multiplicative gate expects a conditioning vector; here
    it receives a single learned constant, which makes it an ordinary learned
    per-block scale rather than a channel through which corruption information
    could leak.

    ``forward`` returns the **residual** ``f(z)``, not ``D(z)``. Predicting the
    correction rather than the activation keeps the target small and centred,
    and makes partial correction ``z + lambda f(z)`` a one-line operation.
    """

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
        # One learned vector standing in for the flow's time embedding.
        self.constant = nn.Parameter(torch.zeros(cfg.d_model))
        self.blocks = nn.ModuleList(_FlowBlock(cfg) for _ in range(cfg.n_blocks))
        self.output_norm = nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model, width)

    def forward(self, z: torch.Tensor, t: torch.Tensor | float | None = None) -> torch.Tensor:
        """Predict the standardized-space residual ``x0 - z``.

        ``t`` exists only so this model is call-compatible with the shared
        trainer. It is accepted and ignored: a denoiser that could read the
        corruption level would not be solving the stated task.
        """

        del t
        if z.ndim != 2 or z.shape[1] != self.cfg.activation_width:
            raise ValueError(
                f"expected z with shape [batch, {self.cfg.activation_width}], "
                f"got {tuple(z.shape)}"
            )
        if not z.is_floating_point() or not torch.isfinite(z).all():
            raise ValueError("z must be a finite floating-point tensor")
        conditioning = self.constant.to(dtype=z.dtype)[None, :].expand(z.shape[0], -1)
        hidden = self.input(z)
        for block in self.blocks:
            hidden = block(hidden, conditioning)
        return self.output(self.output_norm(hidden))


def steering_denoiser_parameter_count(cfg: FlowModelConfig) -> int:
    """Count parameters analytically: the flow trunk minus its time embedding."""

    width = cfg.activation_width
    block = 2 * cfg.d_model
    block += 3 * (cfg.d_model * cfg.d_mlp + cfg.d_mlp)
    block += cfg.d_mlp * cfg.d_model + cfg.d_model
    return (
        width * cfg.d_model + cfg.d_model  # input projection
        + cfg.d_model  # the learned constant
        + cfg.n_blocks * block
        + 2 * cfg.d_model  # output norm
        + cfg.d_model * width + width  # output projection
    )


# --------------------------------------------------------------------------
# training batches
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SteeringDenoiseBatch:
    """One steering-corruption batch, in standardized coordinates.

    ``x_t`` and ``target_velocity`` are named for compatibility with the shared
    trainer's loss and checkpoint plumbing; here they mean the corrupted state
    and the residual to predict. Nothing on this batch is a flow quantity.
    """

    x0: torch.Tensor
    v: torch.Tensor
    delta: torch.Tensor
    x_t: torch.Tensor
    t: torch.Tensor
    target_velocity: torch.Tensor
    objective: str = STEERING_DENOISER_OBJECTIVE


def sample_steering_corruption_batch(
    h: torch.Tensor,
    *,
    normalizer: ActivationNormalizer,
    pool: TrainingDirectionPool,
    generator: torch.Generator,
    spec: SteeringCorruptionSpec = STEERING_CORRUPTION_SPEC,
    delta: torch.Tensor | None = None,
) -> SteeringDenoiseBatch:
    """Corrupt raw activations by steering, then standardize.

    The corruption is applied in **raw** space, ``z = h + delta v``, because that
    is where steering actually happens and where ``v`` is a unit vector. Only
    then is the pair standardized, so ``delta`` keeps the units the frozen T2
    displacements are quoted in.

    ``delta`` may be supplied to evaluate a specific strength; when omitted it is
    drawn from the frozen distribution.
    """

    if h.ndim != 2 or not h.is_floating_point() or not torch.isfinite(h).all():
        raise ValueError("h must be a finite floating-point [rows, d] tensor")
    if pool.directions.device != h.device or pool.directions.dtype != h.dtype:
        raise ValueError(
            f"direction pool is on {pool.directions.device}/{pool.directions.dtype}, "
            f"activations are on {h.device}/{h.dtype}; move the pool with .to()"
        )
    directions = _unit_rows(pool.sample(h.shape[0], generator=generator))
    if delta is None:
        delta = spec.sample_delta(
            h.shape[0], generator=generator, device=h.device, dtype=h.dtype
        )
    delta = delta.reshape(-1, 1).to(device=h.device, dtype=h.dtype)
    if delta.shape[0] != h.shape[0]:
        raise ValueError("delta must carry one strength per activation row")

    z = h + delta * directions
    x0 = normalizer.normalize(h)
    z_x = normalizer.normalize(z)
    return SteeringDenoiseBatch(
        x0=x0,
        v=directions,
        delta=delta,
        x_t=z_x,
        # A zero column, present only so the shared trainer can pass something.
        t=torch.zeros((h.shape[0], 1), device=h.device, dtype=h.dtype),
        target_velocity=x0 - z_x,
    )


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SteeringDenoiseOutput:
    """One partial-correction result and the controls that police it."""

    activation: torch.Tensor
    corrupted: torch.Tensor
    lam: float
    realised_alpha: torch.Tensor
    requested_alpha: torch.Tensor
    diagnostics: dict[str, float | int | str]


@torch.no_grad()
def partial_denoise(
    model: SteeringDenoiser,
    h_clean: torch.Tensor,
    direction: torch.Tensor,
    alpha: torch.Tensor | float,
    *,
    lam: float,
) -> SteeringDenoiseOutput:
    """Steer additively, then apply ``lambda`` of the denoiser's correction.

    ``lambda = 0`` is the additive baseline exactly (no network evaluation);
    ``lambda = 1`` is the full correction ``D(z)``.

    The interpolation is done in standardized space. Standardization is affine,
    so ``denormalize(z_x + lambda f) == z + lambda * denormalize_scale * f``:
    interpolating before or after denormalizing gives the same activation, and
    the choice cannot smuggle in a scale change.
    """

    if h_clean.ndim != 2 or not torch.isfinite(h_clean).all():
        raise ValueError("h_clean must be a finite [rows, d] tensor")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lambda must lie in [0, 1], got {lam!r}")
    vector = _unit_rows(direction).to(device=h_clean.device, dtype=h_clean.dtype)
    if vector.shape[0] == 1 and h_clean.shape[0] != 1:
        vector = vector.expand(h_clean.shape[0], vector.shape[1])
    requested = torch.as_tensor(alpha, device=h_clean.device, dtype=h_clean.dtype)
    requested = requested.reshape(-1, 1)
    if requested.shape[0] == 1 and h_clean.shape[0] != 1:
        requested = requested.expand(h_clean.shape[0], 1)

    corrupted = h_clean + requested * vector
    if lam == 0.0:
        activation = corrupted
        evaluations = 0
    else:
        z_x = model.normalizer.normalize(corrupted)
        residual = model(z_x)
        if residual.shape != z_x.shape or not torch.isfinite(residual).all():
            raise ValueError("the denoiser residual must be finite and state-shaped")
        activation = model.normalizer.denormalize(z_x + lam * residual)
        evaluations = 1
        if not torch.isfinite(activation).all():
            raise ValueError("partial denoising produced a non-finite activation")

    realised = realised_strength(activation, h_clean, vector)
    correction = activation - corrupted
    parallel = (correction * vector).sum(dim=-1, keepdim=True)
    orthogonal = correction - parallel * vector
    diagnostics: dict[str, float | int | str] = {
        "objective": STEERING_DENOISER_OBJECTIVE,
        "n_rows": int(h_clean.shape[0]),
        "lambda": float(lam),
        "network_evaluations": evaluations,
        "requested_alpha_mean": float(requested.double().mean()),
        "realised_alpha_mean": float(realised.double().mean()),
        # The attenuation channel, measured rather than assumed: how much of the
        # requested strength the correction gave back.
        "attenuation_mean": float((requested.squeeze(-1) - realised).double().mean()),
        "parallel_correction_norm_mean": float(parallel.abs().double().mean()),
        "orthogonal_correction_norm_mean": float(orthogonal.norm(dim=-1).double().mean()),
        "correction_norm_mean": float(correction.norm(dim=-1).double().mean()),
    }
    return SteeringDenoiseOutput(
        activation=activation,
        corrupted=corrupted,
        lam=float(lam),
        realised_alpha=realised,
        requested_alpha=requested.squeeze(-1),
        diagnostics=diagnostics,
    )


def realised_strength(
    produced: torch.Tensor, clean: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    """``alpha_eff = <h' - h_clean, v>``: the steering that actually survived.

    This is the x-axis of the matched-strength comparison. Reporting a quality
    win at a smaller ``alpha_eff`` than the baseline's is reporting attenuation.
    """

    if produced.shape != clean.shape:
        raise ValueError("produced and clean activations must have the same shape")
    vector = _unit_rows(direction).to(device=produced.device, dtype=produced.dtype)
    return ((produced - clean) * vector).sum(dim=-1)


def shrinkage_activation(
    h_clean: torch.Tensor, direction: torch.Tensor, alpha_eff: torch.Tensor | float
) -> torch.Tensor:
    """The scalar-shrinkage control: plain additive steering at ``alpha_eff``.

    This is the arm that must be beaten. If the denoiser's quality at a realised
    strength is no better than simply steering that much less, the method is an
    attenuator with extra steps.
    """

    vector = _unit_rows(direction).to(device=h_clean.device, dtype=h_clean.dtype)
    if vector.shape[0] == 1 and h_clean.shape[0] != 1:
        vector = vector.expand(h_clean.shape[0], vector.shape[1])
    strength = torch.as_tensor(alpha_eff, device=h_clean.device, dtype=h_clean.dtype)
    return h_clean + strength.reshape(-1, 1) * vector
