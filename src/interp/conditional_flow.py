"""Steering-aware conditional flow prior: condition on (direction, linear coordinate).

The unconditional prior in :mod:`interp.flow_core` models ``p(h)``.  This module
extends it to ``p(h | <h, v> = c)``: the same rectified-flow path, the same
velocity convention ``u = epsilon - x_0``, the same SwiGLU blocks and time
embedding, plus a condition embedding injected into every block.

Conditioning is on the *direction vector itself*, never on a feature ID, so an
unseen SAE direction can be requested zero-shot at inference time.

Raw-space constraint ``<h, v> = c_target`` is carried into standardized space
exactly.  With ``x = (h - mu) / scale`` where ``scale = std + eps`` is the
normalizer's effective divisor:

    q      = scale * v                      (elementwise)
    q_norm = ||q||
    v_x    = q / q_norm
    c_x    = (c_target - <mu, v>) / q_norm

because ``<x, q> = <h - mu, v>``.  The pair ``(v_x, c_x)`` is then sign
canonicalized so that ``(v, c)`` and ``(-v, -c)`` -- the same hyperplane -- give
one representation.

This module contains no GPT-2, SAE, dataset, or evaluation-split logic, and it
never reads protected evaluation directions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from .flow_core import (
    ActivationNormalizer,
    FlowMatcher,
    FlowModelConfig,
    _time_for_state,
    flow_parameter_count,
    load_flow_config,
    sample_flow_batch,
)
from .flow_sampling import euler_time_grid, sdedit_sample

# Rank floor for the training-only SAE direction pool
# (docs/STRUCTURED_CORRUPTION_PROPOSAL.md section 5).  Every accepted DEV or
# held-out concept direction sits below rank 64 in the same frozen ordering, so
# a pool bounded below by 256 is disjoint from both without ever enumerating an
# evaluation feature ID.
MIN_TRAINING_RANK = 256

TRAINING_ONLY_SPLIT = "training_only"

# Splits a training pool must explicitly declare it excludes.
EXCLUDED_EVALUATION_SPLITS = ("dev", "held_out")


# --------------------------------------------------------------------------
# hyperplane algebra
# --------------------------------------------------------------------------


def _unit_rows(direction: torch.Tensor) -> torch.Tensor:
    """Validate a ``[d]`` or ``[batch, d]`` unit direction and return it batched."""

    if direction.ndim == 1:
        direction = direction[None, :]
    if direction.ndim != 2 or direction.shape[1] == 0 or not direction.is_floating_point():
        raise ValueError("direction must be a floating-point [d] or [batch, d] tensor")
    if not torch.isfinite(direction).all():
        raise ValueError("direction must be finite")
    norms = direction.double().norm(dim=-1)
    if not bool(((norms - 1.0).abs() <= 1e-3).all()):
        raise ValueError(f"direction rows must be unit-normalized, got norms {norms.tolist()}")
    return direction


def _as_column(value: torch.Tensor | float, rows: int | None, name: str) -> torch.Tensor:
    """Coerce a scalar / ``[rows]`` / ``[rows, 1]`` quantity to a broadcastable column."""

    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.float()
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor[:, None]
    elif tensor.ndim != 2 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must be scalar, [rows], or [rows, 1]")
    if rows is not None and tensor.shape[0] not in (1, rows):
        raise ValueError(f"{name} has {tensor.shape[0]} rows, expected 1 or {rows}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor


def canonicalize_hyperplane(
    v_x: torch.Tensor, c_x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fix the sign so ``(v, c)`` and ``(-v, -c)`` map to one representation.

    The convention is: the largest-magnitude component of ``v_x`` is positive.
    Ties break on the lowest index, and an all-zero row is left untouched (it is
    rejected upstream by the ``q_norm`` guard).
    """

    pivot = v_x.abs().argmax(dim=-1, keepdim=True)
    sign = torch.sign(torch.gather(v_x, -1, pivot))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return v_x * sign, c_x * sign


def standardized_hyperplane(
    normalizer: ActivationNormalizer,
    direction: torch.Tensor,
    c_target: torch.Tensor | float,
    *,
    min_norm: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raw ``<h, v> = c_target`` to standardized ``<x, v_x> = c_x``.

    Returns canonicalized ``(v_x, c_x)`` with shapes ``[batch, d]`` and
    ``[batch, 1]``.
    """

    v = _unit_rows(direction)
    scale = (normalizer.std + normalizer.eps).to(device=v.device, dtype=v.dtype)
    mean = normalizer.mean.to(device=v.device, dtype=v.dtype)
    if scale.shape[-1] != v.shape[-1]:
        raise ValueError(
            f"normalizer width {scale.shape[-1]} != direction width {v.shape[-1]}"
        )
    c = _as_column(c_target, None, "c_target").to(device=v.device, dtype=v.dtype)
    rows = max(v.shape[0], c.shape[0])
    if v.shape[0] not in (1, rows) or c.shape[0] not in (1, rows):
        raise ValueError(
            f"direction rows {v.shape[0]} and c_target rows {c.shape[0]} do not broadcast"
        )
    if v.shape[0] == 1 and rows != 1:
        v = v.expand(rows, v.shape[1])

    q = (scale * v).double()
    q_norm = q.norm(dim=-1, keepdim=True)
    if not bool((q_norm > min_norm).all()):
        raise ValueError(
            f"degenerate standardized direction norm {q_norm.min().item()} <= {min_norm}"
        )
    v_x = (q / q_norm).to(v.dtype)
    offset = (v.double() * mean.double()).sum(dim=-1, keepdim=True)
    c_x = ((c.double() - offset) / q_norm).to(v.dtype)
    return canonicalize_hyperplane(v_x, c_x)


def target_coordinate(
    h: torch.Tensor, direction: torch.Tensor, alpha: torch.Tensor | float, *, mode: str
) -> torch.Tensor:
    """Return the requested raw coordinate ``c_target`` as ``[rows, 1]``.

    ``mode="additive"``  -> ``c_target = <h, v> + alpha``   (additive-equivalent)
    ``mode="absolute"``  -> ``c_target = alpha``            (hard clamp)
    """

    if h.ndim != 2:
        raise ValueError(f"expected h with shape [rows, d], got {tuple(h.shape)}")
    v = _unit_rows(direction).to(device=h.device, dtype=h.dtype)
    value = _as_column(alpha, h.shape[0], "alpha").to(device=h.device, dtype=h.dtype)
    if mode == "absolute":
        return value.expand(h.shape[0], 1).clone()
    if mode != "additive":
        raise ValueError(f"unknown coordinate mode {mode!r}, expected additive or absolute")
    return (h * v).sum(dim=-1, keepdim=True) + value


def clamp_seed(
    h: torch.Tensor, direction: torch.Tensor, c_target: torch.Tensor | float
) -> torch.Tensor:
    """Return ``h + (c_target - <h, v>) v``: the general seed with coordinate ``c_target``."""

    if h.ndim != 2 or not h.is_floating_point() or not torch.isfinite(h).all():
        raise ValueError("h must be a finite floating-point [rows, d] tensor")
    v = _unit_rows(direction).to(device=h.device, dtype=h.dtype)
    if v.shape[-1] != h.shape[-1]:
        raise ValueError(f"direction width {v.shape[-1]} != activation width {h.shape[-1]}")
    c = _as_column(c_target, h.shape[0], "c_target").to(device=h.device, dtype=h.dtype)
    return h + (c - (h * v).sum(dim=-1, keepdim=True)) * v


# --------------------------------------------------------------------------
# training-only direction pool
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectionPoolProvenance:
    """Where a training direction pool came from, and why it is training-only.

    Every field is required.  ``digest`` is over the direction and rank bytes plus
    the rest of this record, so a manifest cannot be edited -- or its directions
    swapped for evaluation ones -- without the load-time check failing.
    """

    split: str
    source: str
    selection: str
    selection_seed: int
    min_rank: int
    excluded_splits: tuple[str, ...]
    n_directions: int
    digest: str

    def record(self) -> dict[str, object]:
        """Serializable provenance for run metadata and checkpoint identity."""

        return {
            "split": self.split,
            "source": self.source,
            "selection": self.selection,
            "selection_seed": self.selection_seed,
            "min_rank": self.min_rank,
            "excluded_splits": list(self.excluded_splits),
            "n_directions": self.n_directions,
            "digest": self.digest,
        }


def _pool_digest(
    directions: torch.Tensor, ranks: tuple[int, ...], fields: dict[str, object]
) -> str:
    """Hash the directions, their ranks, and the provenance claims together."""

    payload = hashlib.sha256()
    payload.update(
        directions.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
    )
    payload.update(np.asarray(ranks, dtype=np.int64).tobytes())
    payload.update(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode())
    return payload.hexdigest()


class TrainingDirectionPool:
    """Directions the conditional prior may train on, gated on declared provenance.

    Construction fails unless the provenance record declares
    ``split == "training_only"``, excludes both evaluation splits, carries a
    selection procedure and seed, and puts every direction's rank at or above
    ``MIN_TRAINING_RANK``.  DEV and held-out directions occupy ranks far below that
    floor, so neither can enter the pool -- and no evaluation feature ID is ever
    read, named, or stored here.  The digest makes the claim tamper-evident.
    """

    def __init__(
        self,
        directions: torch.Tensor,
        ranks: tuple[int, ...] | list[int],
        provenance: DirectionPoolProvenance,
        *,
        floor: int = MIN_TRAINING_RANK,
    ) -> None:
        if not isinstance(provenance, DirectionPoolProvenance):
            raise TypeError("a direction pool requires a DirectionPoolProvenance record")
        if provenance.split != TRAINING_ONLY_SPLIT:
            raise ValueError(
                f"direction pool split must be {TRAINING_ONLY_SPLIT!r}, "
                f"got {provenance.split!r}"
            )
        missing = set(EXCLUDED_EVALUATION_SPLITS) - set(provenance.excluded_splits)
        if missing:
            raise ValueError(
                f"pool provenance must exclude every evaluation split; missing {sorted(missing)}"
            )
        if not provenance.source or not provenance.selection:
            raise ValueError("pool provenance needs a source and a selection procedure")
        if provenance.min_rank < floor:
            raise ValueError(
                f"pool provenance min_rank {provenance.min_rank} is below the "
                f"training-only floor {floor}"
            )
        vectors = _unit_rows(directions)
        ranks = tuple(int(rank) for rank in ranks)
        if len(ranks) != vectors.shape[0] or not ranks:
            raise ValueError("pool needs exactly one selection rank per direction")
        if len(set(ranks)) != len(ranks):
            raise ValueError("pool selection ranks must be unique")
        if min(ranks) < provenance.min_rank:
            raise ValueError(
                f"pool rank {min(ranks)} is below the declared floor {provenance.min_rank}; "
                "evaluation directions may not enter flow training"
            )
        if provenance.n_directions != len(ranks):
            raise ValueError(
                f"pool provenance declares {provenance.n_directions} directions, "
                f"found {len(ranks)}"
            )
        expected = _pool_digest(vectors, ranks, _digest_fields(provenance))
        if expected != provenance.digest:
            raise ValueError(
                "direction pool digest mismatch: manifest contents do not match "
                "their declared provenance"
            )
        self.directions = vectors
        self.ranks = ranks
        self.provenance = provenance

    def __len__(self) -> int:
        return int(self.directions.shape[0])

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> TrainingDirectionPool:
        """Return the same verified pool on another device/dtype."""

        moved = TrainingDirectionPool.__new__(TrainingDirectionPool)
        moved.directions = self.directions.to(device=device, dtype=dtype)
        moved.ranks = self.ranks
        moved.provenance = self.provenance
        return moved

    def sample(self, rows: int, *, generator: torch.Generator) -> torch.Tensor:
        """Draw ``rows`` directions with replacement, on the pool's own device."""

        if rows < 1:
            raise ValueError("must sample at least one direction")
        index = torch.randint(
            len(self), (rows,), generator=generator, device=self.directions.device
        )
        return self.directions[index]

    def identity(self) -> dict[str, object]:
        """Provenance record for run metadata and resume verification."""

        return {
            **self.provenance.record(),
            "observed_min_rank": min(self.ranks),
            "observed_max_rank": max(self.ranks),
        }


def _digest_fields(provenance: DirectionPoolProvenance) -> dict[str, object]:
    fields = provenance.record()
    fields.pop("digest")
    return fields


def save_direction_pool(
    path: Path,
    directions: torch.Tensor,
    ranks: tuple[int, ...] | list[int],
    *,
    source: str,
    selection: str,
    selection_seed: int,
    min_rank: int = MIN_TRAINING_RANK,
) -> DirectionPoolProvenance:
    """Write a training-only direction manifest with its provenance digest."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite a direction manifest: {path}")
    vectors = _unit_rows(directions)
    ranks = tuple(int(rank) for rank in ranks)
    fields: dict[str, object] = {
        "split": TRAINING_ONLY_SPLIT,
        "source": source,
        "selection": selection,
        "selection_seed": int(selection_seed),
        "min_rank": int(min_rank),
        "excluded_splits": list(EXCLUDED_EVALUATION_SPLITS),
        "n_directions": len(ranks),
    }
    provenance = DirectionPoolProvenance(
        split=TRAINING_ONLY_SPLIT,
        source=source,
        selection=selection,
        selection_seed=int(selection_seed),
        min_rank=int(min_rank),
        excluded_splits=EXCLUDED_EVALUATION_SPLITS,
        n_directions=len(ranks),
        digest=_pool_digest(vectors, ranks, fields),
    )
    # Validate before writing: a manifest on disk is always a loadable pool.
    TrainingDirectionPool(vectors, ranks, provenance)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "directions": vectors.detach().to(torch.float32).cpu(),
            "ranks": list(ranks),
            "provenance": provenance.record(),
        },
        path,
    )
    return provenance


def load_training_direction_pool(path: Path) -> TrainingDirectionPool:
    """Load a training-only direction manifest, refusing protected paths outright."""

    resolved = Path(path).resolve()
    if "protected" in resolved.parts:
        raise PermissionError(
            f"refusing to read a direction manifest under a protected path: {resolved}"
        )
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported direction-manifest format")
    record = dict(payload["provenance"])
    provenance = DirectionPoolProvenance(
        split=str(record["split"]),
        source=str(record["source"]),
        selection=str(record["selection"]),
        selection_seed=int(record["selection_seed"]),
        min_rank=int(record["min_rank"]),
        excluded_splits=tuple(str(item) for item in record["excluded_splits"]),
        n_directions=int(record["n_directions"]),
        digest=str(record["digest"]),
    )
    return TrainingDirectionPool(payload["directions"], payload["ranks"], provenance)


# --------------------------------------------------------------------------
# conditional architecture
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionEncoderConfig:
    """Parameters of the condition encoder added on top of the unconditional prior."""

    cond_hidden: int
    kind: str = "direction_coordinate_film"

    def __post_init__(self) -> None:
        if self.cond_hidden <= 0:
            raise ValueError("cond_hidden must be positive")
        if self.kind != "direction_coordinate_film":
            raise ValueError("the only implemented condition encoder is direction_coordinate_film")


def condition_parameter_count(cfg: FlowModelConfig, cond: ConditionEncoderConfig) -> int:
    """Count only the conditioning overhead, analytically."""

    hidden = cond.cond_hidden
    direction = cfg.activation_width * hidden + hidden
    film = 2 * (hidden + hidden)  # scale and shift, each Linear(1, hidden)
    output = hidden * cfg.d_model + cfg.d_model
    return direction + film + output


class _ConditionEmbedding(nn.Module):
    """Embed ``(v_x, c_x)`` with a FiLM-style interaction, not an additive scalar bias."""

    def __init__(self, cfg: FlowModelConfig, cond: ConditionEncoderConfig) -> None:
        super().__init__()
        hidden = cond.cond_hidden
        self.direction = nn.Linear(cfg.activation_width, hidden)
        self.scale = nn.Linear(1, hidden)
        self.shift = nn.Linear(1, hidden)
        self.output = nn.Linear(hidden, cfg.d_model)
        self.activation = nn.SiLU()

    def forward(self, v_x: torch.Tensor, c_x: torch.Tensor) -> torch.Tensor:
        direction = self.activation(self.direction(v_x))
        modulated = direction * self.scale(c_x) + self.shift(c_x)
        return self.output(self.activation(modulated))


class ConditionalFlowMatcher(FlowMatcher):
    """``f_theta(x_t, t, v_x, c_x)``: the unconditional prior plus hyperplane conditioning.

    The flow path, velocity target, block family, time embedding, standardization,
    and sampler semantics are inherited unchanged.  The condition embedding is
    added to the time embedding, so every block's multiplicative gate sees it.
    """

    def __init__(
        self,
        cfg: FlowModelConfig,
        cond: ConditionEncoderConfig,
        normalizer: ActivationNormalizer,
    ) -> None:
        super().__init__(cfg, normalizer)
        self.cond_cfg = cond
        self.condition = _ConditionEmbedding(cfg, cond)

    def forward(  # type: ignore[override]
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | float,
        v_x: torch.Tensor,
        c_x: torch.Tensor | float,
    ) -> torch.Tensor:
        width = self.cfg.activation_width
        if x_t.ndim != 2:
            raise ValueError(
                f"expected x_t with shape [batch, activation_dim], got {tuple(x_t.shape)}"
            )
        if x_t.shape[1] != width:
            raise ValueError(f"expected activation_dim={width}, got {x_t.shape[1]}")
        if not x_t.is_floating_point() or not torch.isfinite(x_t).all():
            raise ValueError("x_t must be a finite floating-point tensor")
        rows = x_t.shape[0]

        direction = v_x[None, :] if v_x.ndim == 1 else v_x
        if direction.ndim != 2 or direction.shape[1] != width:
            raise ValueError(f"v_x must be [d] or [batch, d] with d={width}")
        if not torch.isfinite(direction).all():
            raise ValueError("v_x must be finite")
        if direction.shape[0] == 1 and rows != 1:
            direction = direction.expand(rows, width)
        if direction.shape[0] != rows:
            raise ValueError(f"v_x has {direction.shape[0]} rows, expected 1 or {rows}")
        coordinate = _as_column(c_x, rows, "c_x").to(device=x_t.device, dtype=x_t.dtype)
        if coordinate.shape[0] == 1 and rows != 1:
            coordinate = coordinate.expand(rows, 1)

        time = _time_for_state(t, x_t).reshape(-1)
        if time.shape[0] == 1 and rows != 1:
            time = time.expand(rows)
        conditioning = self.time_embedding(time) + self.condition(
            direction.to(device=x_t.device, dtype=x_t.dtype), coordinate
        )
        z = self.input(x_t)
        for block in self.blocks:
            z = block(z, conditioning)
        return self.output(self.output_norm(z))

    def velocity_field(self, v_x: torch.Tensor, c_x: torch.Tensor):
        """Freeze the condition and expose the plain ``(x, t)`` field the sampler wants."""

        def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self(x, t, v_x, c_x)

        return velocity


def conditional_parameter_count(cfg: FlowModelConfig, cond: ConditionEncoderConfig) -> int:
    """Total conditional-model parameters, analytically."""

    return flow_parameter_count(cfg) + condition_parameter_count(cfg, cond)


@dataclass(frozen=True)
class ConditionalFlowConfig:
    """Architecture specification for the conditional variant."""

    model: FlowModelConfig
    condition: ConditionEncoderConfig
    coordinate: str
    feature_id_conditioning: bool


def load_conditional_flow_config(path: Path) -> ConditionalFlowConfig:
    """Load the conditional architecture spec, reusing the frozen base-flow validation."""

    model = load_flow_config(path)
    raw = yaml.safe_load(Path(path).read_text())
    conditioning = raw.get("conditioning")
    if not isinstance(conditioning, dict):
        raise ValueError("conditional config requires a 'conditioning' section")
    if conditioning.get("type") != "direction_coordinate":
        raise ValueError("conditioning.type must be 'direction_coordinate'")
    if conditioning.get("coordinate") != "linear_projection":
        raise ValueError("conditioning.coordinate must be 'linear_projection'")
    if conditioning.get("feature_id_conditioning") is not False:
        raise ValueError("feature-ID conditioning is not implemented and must be false")
    cond = ConditionEncoderConfig(
        cond_hidden=int(conditioning["cond_hidden"]),
        kind=str(conditioning.get("encoder", "direction_coordinate_film")),
    )
    expected = conditioning.get("n_params_expected")
    if expected is not None and condition_parameter_count(model, cond) != int(expected):
        raise ValueError(
            f"conditioning yields {condition_parameter_count(model, cond)} parameters "
            f"!= declared {int(expected)}"
        )
    return ConditionalFlowConfig(
        model=model,
        condition=cond,
        coordinate=str(conditioning["coordinate"]),
        feature_id_conditioning=False,
    )


# --------------------------------------------------------------------------
# training objective
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionalFlowBatch:
    """One conditional flow-matching batch; the flow part is the unconditional one."""

    x_t: torch.Tensor
    t: torch.Tensor
    target_velocity: torch.Tensor
    v_x: torch.Tensor
    c_x: torch.Tensor


def sample_conditional_flow_batch(
    h: torch.Tensor,
    *,
    normalizer: ActivationNormalizer,
    pool: TrainingDirectionPool,
    generator: torch.Generator,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> ConditionalFlowBatch:
    """Condition on each activation's own naturally occurring coordinate.

    The objective stays pure flow matching: no auxiliary coordinate loss, no
    steering corruption, no reconstruction term.  Directions come only from the
    training-only pool.
    """

    if h.ndim != 2 or not h.is_floating_point() or not torch.isfinite(h).all():
        raise ValueError("h must be a finite floating-point [rows, d] tensor")
    if pool.directions.device != h.device or pool.directions.dtype != h.dtype:
        raise ValueError(
            f"direction pool is on {pool.directions.device}/{pool.directions.dtype}, "
            f"activations are on {h.device}/{h.dtype}; move the pool with .to() so the "
            "sampling generator and the pool agree on device"
        )
    directions = pool.sample(h.shape[0], generator=generator)
    c_raw = (h * directions).sum(dim=-1, keepdim=True)
    v_x, c_x = standardized_hyperplane(normalizer, directions, c_raw)
    base = sample_flow_batch(
        normalizer.normalize(h), generator=generator, t_min=t_min, t_max=t_max
    )
    return ConditionalFlowBatch(
        x_t=base.x_t,
        t=base.t,
        target_velocity=base.target_velocity,
        v_x=v_x,
        c_x=c_x,
    )


def implied_x0(x_t: torch.Tensor, t: torch.Tensor | float, velocity: torch.Tensor) -> torch.Tensor:
    """Invert the linear path: ``x_0 = x_t - t * u`` for ``u = epsilon - x_0``."""

    time = _time_for_state(t, x_t)
    return x_t - time * velocity


# --------------------------------------------------------------------------
# conditional inference
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionalSteeringOutput:
    activation: torch.Tensor
    seed: torch.Tensor
    target_coordinate: torch.Tensor
    realized_coordinate: torch.Tensor
    coordinate_error: torch.Tensor
    network_evaluations: int
    final_projection: bool
    geometry: dict[str, float | int]


@torch.no_grad()
def conditional_clamp_steer(
    model: ConditionalFlowMatcher,
    h: torch.Tensor,
    direction: torch.Tensor,
    *,
    alpha: torch.Tensor | float,
    mode: str = "additive",
    noise: torch.Tensor,
    t_start: float,
    nfe: int,
    final_projection: bool = False,
    seed_mode: str = "clamp",
) -> ConditionalSteeringOutput:
    """Seed the requested coordinate, then regenerate a natural activation under it.

    ``final_projection`` re-imposes the coordinate by hard clamp after generation.
    It defaults to ``False`` on purpose: the scientific question is whether the
    conditional model respects the coordinate on its own.

    ``seed_mode`` selects what the prior is seeded with, which is a scientific
    choice rather than an implementation detail:

    ``"clamp"``  -> ``clamp_seed(h, v, c_target)``; in ``mode="additive"`` that is
    exactly ``h + alpha*v``, so the coordinate is already correct in the seed and
    the prior's job is to make it natural.

    ``"clean"``  -> ``h`` itself, so the requested coordinate reaches the model
    only through the condition. This is the arm defined by
    docs/PHASE_B_CONDITIONAL_PROTOCOL.md section 4, and it makes ``t_start = 0``
    a null intervention at every alpha.
    """

    if h.ndim != 2 or h.numel() == 0 or not h.is_floating_point():
        raise ValueError("h must be a nonempty floating-point [rows, d] tensor")
    if not torch.isfinite(h).all():
        raise ValueError("h must be finite")
    if noise.shape != h.shape or not noise.is_floating_point() or not torch.isfinite(noise).all():
        raise ValueError("noise must be a same-shaped finite floating-point tensor")
    if seed_mode not in ("clamp", "clean"):
        raise ValueError(f"unknown seed_mode {seed_mode!r}, expected clamp or clean")
    vector = _unit_rows(direction).to(device=h.device, dtype=h.dtype)
    c_target = target_coordinate(h, vector, alpha, mode=mode)
    seed = h if seed_mode == "clean" else clamp_seed(h, vector, c_target)

    if t_start == 0.0:
        # Validate the schedule, evaluate nothing, and return the seed exactly.
        euler_time_grid(t_start, nfe, device=h.device, dtype=h.dtype)
        activation = seed
        evaluations = 0
    else:
        v_x, c_x = standardized_hyperplane(model.normalizer, vector, c_target)
        sampled = sdedit_sample(
            model.velocity_field(v_x, c_x),
            model.normalizer.normalize(seed),
            noise=noise,
            t_start=t_start,
            nfe=nfe,
        )
        activation = model.normalizer.denormalize(sampled)
        evaluations = nfe
    if final_projection:
        activation = clamp_seed(activation, vector, c_target)
    if not torch.isfinite(activation).all():
        raise ValueError("conditional steering produced a non-finite activation")

    realized = (activation * vector).sum(dim=-1, keepdim=True)
    correction = activation - seed
    parallel = (correction * vector).sum(dim=-1, keepdim=True)
    orthogonal = correction - parallel * vector
    return ConditionalSteeringOutput(
        activation=activation,
        seed=seed,
        target_coordinate=c_target.squeeze(-1),
        realized_coordinate=realized.squeeze(-1),
        coordinate_error=(realized - c_target).squeeze(-1),
        network_evaluations=evaluations,
        final_projection=bool(final_projection),
        geometry={
            "n_rows": int(h.shape[0]),
            "coordinate_abs_error_mean": float((realized - c_target).abs().double().mean()),
            "seed_coordinate_mean": float((seed * vector).sum(-1).double().mean()),
            "correction_norm_mean": float(correction.norm(dim=-1).double().mean()),
            "parallel_correction_norm_mean": float(parallel.abs().double().mean()),
            "orthogonal_correction_norm_mean": float(orthogonal.norm(dim=-1).double().mean()),
            "clean_distance_mean": float((activation - h).norm(dim=-1).double().mean()),
        },
    )
