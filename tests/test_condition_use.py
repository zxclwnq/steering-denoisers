"""Tests for the frozen condition-use diagnostic.

The diagnostic's job is to separate a model that reads ``(v, c)`` from one that ignores
it, so the tests check exactly that: a condition-blind model must score a zero gap, and a
model that does use the condition must score a positive one.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from interp.condition_use import (
    FROZEN_SPEC,
    ConditionUseSpec,
    condition_use_passes,
    run_condition_use_diagnostic,
)
from interp.conditional_flow import (
    MIN_TRAINING_RANK,
    ConditionalFlowMatcher,
    ConditionEncoderConfig,
    load_training_direction_pool,
    save_direction_pool,
)
from interp.flow_core import ActivationNormalizer, FlowModelConfig

TINY = replace(FROZEN_SPEC, n_rows=64, n_directions=4, t_values=(0.5, 1.0))


def _pool(tmp_path, width: int = 6, rows: int = 8):
    generator = torch.Generator().manual_seed(4)
    directions = torch.randn(rows, width, generator=generator)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    path = tmp_path / "pool.pt"
    save_direction_pool(
        path,
        directions,
        ranks=tuple(range(MIN_TRAINING_RANK, MIN_TRAINING_RANK + rows)),
        source="synthetic_test_fixture",
        selection="blake2b_priority_rank",
        selection_seed=20260807,
    )
    return load_training_direction_pool(path)


def _model(width: int = 6, *, seed: int = 0) -> ConditionalFlowMatcher:
    torch.manual_seed(seed)
    return ConditionalFlowMatcher(
        FlowModelConfig(16, 32, 2, 8, 8, 100.0, activation_dim=width),
        ConditionEncoderConfig(cond_hidden=8),
        ActivationNormalizer(torch.zeros(width), torch.ones(width), eps=1e-5),
    )


def _activations(rows: int = 128, width: int = 6) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(rows, width)).astype(np.float32)


def test_diagnostic_is_deterministic(tmp_path) -> None:
    model, pool, activations = _model(), _pool(tmp_path), _activations()

    first = run_condition_use_diagnostic(model, activations, pool, spec=TINY)
    second = run_condition_use_diagnostic(model, activations, pool, spec=TINY)

    assert first["by_t"] == second["by_t"]
    assert first["direction_pool"]["digest"] == pool.provenance.digest


def test_condition_blind_model_shows_no_gap_and_fails(tmp_path) -> None:
    """A model whose condition encoder is zeroed must score an exactly zero gap."""

    model = _model()
    with torch.no_grad():
        for parameter in model.condition.parameters():
            parameter.zero_()

    summary = run_condition_use_diagnostic(model, _activations(), _pool(tmp_path), spec=TINY)

    for row in summary["by_t"]:
        assert row["gap_shuffled_target"] == pytest.approx(0.0, abs=1e-9)
        assert row["gap_shuffled_direction"] == pytest.approx(0.0, abs=1e-9)
        assert row["swap_sensitivity_shuffled_target"] == pytest.approx(0.0, abs=1e-9)
    assert condition_use_passes(summary) is False


def test_condition_using_model_shows_a_positive_gap(tmp_path) -> None:
    """Overfit the coordinate task, then the correct condition must win."""

    width, rows = 6, 256
    pool = _pool(tmp_path, width=width)
    model = _model(width)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(7)
    for _ in range(400):
        index = torch.randint(len(pool), (rows,), generator=generator)
        v = pool.directions[index]
        c = torch.empty(rows, 1).uniform_(-3.0, 3.0, generator=generator)
        z = torch.randn(rows, width, generator=generator)
        x0 = z + (c - (z * v).sum(dim=-1, keepdim=True)) * v
        epsilon = torch.randn(rows, width, generator=generator)
        time = torch.ones(rows, 1)
        loss = (model(epsilon, time, v, c) - (epsilon - x0)).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Activations whose coordinate along each direction is the only learnable signal.
    generator = torch.Generator().manual_seed(11)
    base = torch.randn(512, width, generator=generator)
    summary = run_condition_use_diagnostic(
        model, base.numpy(), pool, spec=replace(TINY, n_rows=256, n_directions=4)
    )

    assert summary["mean_gap_shuffled_target"] > 0.0
    assert summary["mean_gap_shuffled_direction"] > 0.0
    assert summary["mean_relative_swap_sensitivity_target"] > 0.0
    assert condition_use_passes(summary) is True
    high_t = [row for row in summary["by_t"] if row["t"] == 1.0][0]
    assert high_t["coordinate_abs_error_correct"] < high_t["coordinate_abs_error_shuffled_target"]


def test_spec_rejects_degenerate_plans() -> None:
    with pytest.raises(ValueError, match="at least two rows"):
        ConditionUseSpec(n_rows=1)
    with pytest.raises(ValueError, match="inside"):
        ConditionUseSpec(t_values=(1.5,))


def test_frozen_spec_is_the_documented_plan() -> None:
    assert FROZEN_SPEC.t_values == (0.50, 0.75, 0.90, 1.00)
    assert FROZEN_SPEC.n_rows == 4096
    assert FROZEN_SPEC.n_directions == 64
    assert FROZEN_SPEC.version == "condition_use_v1"
