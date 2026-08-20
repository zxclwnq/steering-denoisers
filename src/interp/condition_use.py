"""Frozen concept-independent diagnostic: does the model actually use ``(v, c)``?

This is not a steering measurement and says nothing about steering quality. It asks one
question on real GPT-2 activations: when the flow model is handed the correct hyperplane
condition instead of a mismatched one, does its velocity prediction change in the way a
model that reads the condition would change?

Three matched arms share the same activations, the same flow times, the same Gaussian
endpoint, and the same model. Only the condition differs:

    correct            (v_i, c_i)      the row's own direction and coordinate
    shuffled_target    (v_i, c_j)      right direction, another row's coordinate
    shuffled_direction (v_j, c_j)      another row's condition entirely

A model that ignores the condition scores identically in all three arms.

The emphasis is on high ``t``, where ``x_t`` is mostly noise and the clean coordinate
cannot be recovered from the state alone. Everything is deterministic: rows, directions,
the permutation, and epsilon all come from frozen seeds, so the same diagnostic can be
re-run at any later checkpoint and compared directly.

No DEV or held-out direction is touched: directions come from the training-only pool.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .conditional_flow import (
    ConditionalFlowMatcher,
    TrainingDirectionPool,
    implied_x0,
    standardized_hyperplane,
)
from .flow_core import flow_matching_loss, linear_interpolate, velocity_target

ARMS = ("correct", "shuffled_target", "shuffled_direction")


@dataclass(frozen=True)
class ConditionUseSpec:
    """Frozen sampling plan. Changing any field makes a different diagnostic."""

    n_rows: int = 4096
    n_directions: int = 64
    row_seed: int = 20260815
    direction_seed: int = 20260816
    noise_seed: int = 20260817
    permutation_seed: int = 20260818
    t_values: tuple[float, ...] = (0.50, 0.75, 0.90, 1.00)
    version: str = "condition_use_v1"

    def __post_init__(self) -> None:
        if self.n_rows < 2 or self.n_directions < 2:
            raise ValueError("need at least two rows and two directions to shuffle")
        if not self.t_values or any(not 0.0 <= t <= 1.0 for t in self.t_values):
            raise ValueError("t values must lie inside [0, 1]")


# The frozen plan the real checkpoints are measured with.
FROZEN_SPEC = ConditionUseSpec()


def _select_rows(n_available: int, spec: ConditionUseSpec) -> np.ndarray:
    if n_available < spec.n_rows:
        raise ValueError(f"validation artifact has only {n_available} rows")
    rng = np.random.default_rng(spec.row_seed)
    return np.sort(rng.choice(n_available, size=spec.n_rows, replace=False))


def _derangement(n: int, seed: int) -> torch.Tensor:
    """A fixed permutation with no fixed point, so no row keeps its own condition."""

    generator = torch.Generator().manual_seed(seed)
    identity = torch.arange(n)
    for _ in range(100):
        order = torch.randperm(n, generator=generator)
        if not bool((order == identity).any()):
            return order
    # Deterministic fallback: a rotation has no fixed point for any n >= 2.
    return torch.roll(identity, 1)


@torch.no_grad()
def run_condition_use_diagnostic(
    model: ConditionalFlowMatcher,
    activations: np.ndarray,
    pool: TrainingDirectionPool,
    *,
    spec: ConditionUseSpec = FROZEN_SPEC,
    device: torch.device | str = "cpu",
) -> dict:
    """Return matched correct/shuffled results per flow time, plus overall summaries."""

    was_training = model.training
    model.eval()
    resolved = torch.device(device)
    rows = _select_rows(activations.shape[0], spec)
    h = torch.from_numpy(np.array(activations[rows], dtype=np.float32)).to(resolved)

    direction_rng = np.random.default_rng(spec.direction_seed)
    picked = np.sort(
        direction_rng.choice(len(pool), size=spec.n_directions, replace=False)
    )
    catalogue = pool.directions[picked].to(device=resolved, dtype=h.dtype)
    assignment = torch.from_numpy(
        direction_rng.integers(0, spec.n_directions, size=spec.n_rows)
    ).to(resolved)
    directions = catalogue[assignment]

    permutation = _derangement(spec.n_rows, spec.permutation_seed).to(resolved)
    coordinates = (h * directions).sum(dim=-1, keepdim=True)

    conditions = {
        "correct": (directions, coordinates),
        "shuffled_target": (directions, coordinates[permutation]),
        "shuffled_direction": (directions[permutation], coordinates[permutation]),
    }
    standardized = {
        arm: standardized_hyperplane(model.normalizer, vector, target)
        for arm, (vector, target) in conditions.items()
    }

    x0 = model.normalizer.normalize(h)
    noise_generator = torch.Generator(device="cpu").manual_seed(spec.noise_seed)
    epsilon = torch.randn(x0.shape, generator=noise_generator).to(resolved)
    target = velocity_target(x0, epsilon)

    by_t: list[dict] = []
    for t_value in spec.t_values:
        time = torch.full((x0.shape[0], 1), float(t_value), device=resolved)
        x_t = linear_interpolate(x0, epsilon, time)
        row: dict[str, object] = {"t": float(t_value)}
        predictions: dict[str, torch.Tensor] = {}
        for arm in ARMS:
            v_x, c_x = standardized[arm]
            prediction = model(x_t, time, v_x, c_x)
            predictions[arm] = prediction
            reconstructed = implied_x0(x_t, time, prediction)
            raw = model.normalizer.denormalize(reconstructed)
            realized = (raw * directions).sum(dim=-1, keepdim=True)
            row[f"flow_mse_{arm}"] = float(flow_matching_loss(prediction, target))
            # Coordinate error is always measured against the row's own true
            # coordinate, so a shuffled arm is expected to miss it.
            row[f"coordinate_abs_error_{arm}"] = float(
                (realized - coordinates).abs().double().mean()
            )
            row[f"requested_coordinate_abs_error_{arm}"] = float(
                (realized - conditions[arm][1]).abs().double().mean()
            )
        row["gap_shuffled_target"] = (
            row["flow_mse_shuffled_target"] - row["flow_mse_correct"]
        )
        row["gap_shuffled_direction"] = (
            row["flow_mse_shuffled_direction"] - row["flow_mse_correct"]
        )
        correct = predictions["correct"]
        scale = float(correct.norm(dim=-1).double().mean())
        row["prediction_norm_correct"] = scale
        for arm in ARMS[1:]:
            delta = float((predictions[arm] - correct).norm(dim=-1).double().mean())
            row[f"swap_sensitivity_{arm}"] = delta
            row[f"relative_swap_sensitivity_{arm}"] = delta / scale if scale else 0.0
        by_t.append(row)

    model.train(was_training)
    summary = {
        "spec": asdict(spec),
        "n_rows": int(spec.n_rows),
        "n_directions": int(spec.n_directions),
        "direction_pool": pool.identity(),
        "by_t": by_t,
        "mean_gap_shuffled_target": float(
            np.mean([row["gap_shuffled_target"] for row in by_t])
        ),
        "mean_gap_shuffled_direction": float(
            np.mean([row["gap_shuffled_direction"] for row in by_t])
        ),
        "max_gap_shuffled_target": float(
            np.max([row["gap_shuffled_target"] for row in by_t])
        ),
        "mean_relative_swap_sensitivity_target": float(
            np.mean([row["relative_swap_sensitivity_shuffled_target"] for row in by_t])
        ),
        "mean_relative_swap_sensitivity_direction": float(
            np.mean(
                [row["relative_swap_sensitivity_shuffled_direction"] for row in by_t]
            )
        ),
    }
    return summary


def condition_use_passes(
    summary: dict, *, min_gap: float = 0.0, min_relative_swap: float = 0.0
) -> bool:
    """Minimal mechanical PASS: the condition changes both the loss and the output.

    The thresholds are deliberately weak. This answers "is the condition being read at
    all", not "is it read well"; the numbers themselves are the evidence.
    """

    return (
        summary["mean_gap_shuffled_target"] > min_gap
        and summary["mean_gap_shuffled_direction"] > min_gap
        and summary["mean_relative_swap_sensitivity_target"] > min_relative_swap
        and summary["mean_relative_swap_sensitivity_direction"] > min_relative_swap
    )
