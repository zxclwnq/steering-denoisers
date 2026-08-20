"""Plumbing of the prepared T1/T2 evaluator scripts, without GPT-2 or a checkpoint.

The substitution hooks are the part of these scripts that carries real risk:
row offsets across hook batches, the [sequence, position, d_model] reshape, and
the diagnostics that police the semantic coordinate. Those are driven directly
here so the future GPU run is not the first time they execute.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from interp.conditional_flow import (
    ConditionalFlowMatcher,
    ConditionEncoderConfig,
    clamp_seed,
)
from interp.flow_core import ActivationNormalizer, FlowModelConfig
from interp.tangent_eval import TANGENT_NATURALIZATION_SPEC
from interp.tangent_flow import TANGENT_OBJECTIVE, VP_TANGENT_OBJECTIVE

REPO = Path(__file__).resolve().parents[1]
D = 12
SEQUENCES = 6
POSITIONS = 5


def _module(name: str):  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flow() -> ConditionalFlowMatcher:
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(1)
    normalizer = ActivationNormalizer(
        torch.randn(D, generator=generator),
        torch.rand(D, generator=generator) + 0.5,
        1e-5,
    )
    cfg = FlowModelConfig(
        d_model=D, d_mlp=24, n_blocks=1, time_dim=4, time_hidden=8, max_period=10000.0
    )
    return ConditionalFlowMatcher(
        cfg, ConditionEncoderConfig(cond_hidden=4), normalizer
    ).eval()


def _inputs():  # noqa: ANN202
    generator = torch.Generator().manual_seed(20260816)
    activation = torch.randn(SEQUENCES, POSITIONS, D, generator=generator) * 2.0
    directions = torch.randn(SEQUENCES, D, generator=generator)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    noise = torch.randn(SEQUENCES, POSITIONS, D, generator=generator)
    return activation, directions, noise


@pytest.mark.parametrize("hook_batch", [SEQUENCES, 2, 1])
def test_t1_transform_survives_any_hook_batching(hook_batch: int) -> None:
    """Results must not depend on how the LM hook chunks the sequences."""

    t1 = _module("eval_tangent_reconstruction")
    activation, directions, noise = _inputs()
    flow = _flow()

    outputs = []
    for size in (SEQUENCES, hook_batch):
        transform = t1._TangentTransform(
            flow, directions, noise, t_start=0.5, nfe=3, reconstruct=True,
            objective=TANGENT_OBJECTIVE,
        )
        chunks = [
            transform(activation[start : start + size])
            for start in range(0, SEQUENCES, size)
        ]
        outputs.append((torch.cat(chunks), transform))

    whole, whole_transform = outputs[0]
    chunked, chunked_transform = outputs[1]
    assert torch.allclose(whole, chunked, atol=1e-5)
    assert whole.shape == activation.shape
    assert chunked_transform.evaluations == 3 * (SEQUENCES // hook_batch + (
        1 if SEQUENCES % hook_batch else 0
    ))
    assert whole_transform.evaluations == 3


def test_t1_corrupted_control_and_reconstruction_share_the_coordinate() -> None:
    t1 = _module("eval_tangent_reconstruction")
    activation, directions, noise = _inputs()
    flow = _flow()

    control = t1._TangentTransform(
        flow, directions, noise, t_start=0.5, nfe=1, reconstruct=False,
        objective=TANGENT_OBJECTIVE,
    )
    reconstructed = t1._TangentTransform(
        flow, directions, noise, t_start=0.5, nfe=1, reconstruct=True,
        objective=TANGENT_OBJECTIVE,
    )
    control(activation)
    reconstructed(activation)

    assert control.diagnostics()["network_evaluations"] == 0
    assert control.diagnostics()["nfe"] == 0
    assert reconstructed.diagnostics()["network_evaluations"] == 1

    from interp.tangent_eval import concatenate

    control_rows = concatenate(control.records)
    flow_rows = concatenate(reconstructed.records)
    # T1 conditions on the natural coordinate, so both arms sit on it exactly.
    assert control_rows["coordinate_abs_error"].max() < 1e-2
    assert flow_rows["coordinate_abs_error"].max() < 1e-2
    assert np.allclose(control_rows["c_target"], flow_rows["c_target"])
    # and the flow really moved the orthogonal part
    assert flow_rows["tangent_error"].mean() > 0.0


def test_t2_clamp_arm_is_the_exact_clamp_with_no_evaluations() -> None:
    t2 = _module("tangent_naturalization")
    activation, directions, noise = _inputs()
    flow = _flow()
    target = torch.linspace(-2.0, 2.0, SEQUENCES)

    transform = t2._ClampTangentTransform(
        flow, directions, target, noise, t_start=0.0, nfe=1,
        objective=TANGENT_OBJECTIVE,
    )
    produced = transform(activation)

    expected = clamp_seed(
        activation.reshape(-1, D),
        directions[:, None, :].expand(SEQUENCES, POSITIONS, D).reshape(-1, D),
        target[:, None].expand(SEQUENCES, POSITIONS).reshape(-1, 1),
    ).reshape_as(activation)
    assert torch.allclose(produced, expected)
    assert transform.diagnostics()["network_evaluations"] == 0
    assert transform.diagnostics()["nfe"] == 0


def test_t2_arms_agree_on_the_coordinate_and_the_gate_accepts_them() -> None:
    t2 = _module("tangent_naturalization")
    from interp.tangent_eval import assert_coordinate_match, concatenate

    activation, directions, noise = _inputs()
    flow = _flow()
    target = torch.linspace(-2.0, 2.0, SEQUENCES)

    clamp = t2._ClampTangentTransform(
        flow, directions, target, noise, t_start=0.0, nfe=1,
        objective=TANGENT_OBJECTIVE,
    )
    clamp(activation)
    clamp_rows = concatenate(clamp.records)

    for t_start in TANGENT_NATURALIZATION_SPEC.t_start:
        for nfe in TANGENT_NATURALIZATION_SPEC.nfe:
            arm = t2._ClampTangentTransform(
                flow, directions, target, noise, t_start=t_start, nfe=nfe,
                objective=TANGENT_OBJECTIVE,
            )
            arm(activation)
            rows = concatenate(arm.records)
            report = assert_coordinate_match(clamp_rows, rows, tolerance=1e-2)
            assert report["max_arm_coordinate_difference"] < 1e-2
            assert arm.diagnostics()["network_evaluations"] == nfe
            assert arm.diagnostics()["projections"] == nfe
            # the safeguard corrects float error only
            assert arm.diagnostics()["max_pre_projection_drift"] < 1e-3
            assert rows["orthogonal_correction_norm"].mean() > 0.0


def test_t2_transform_rejects_a_wrongly_shaped_activation() -> None:
    t2 = _module("tangent_naturalization")
    activation, directions, noise = _inputs()
    transform = t2._ClampTangentTransform(
        _flow(), directions, torch.zeros(SEQUENCES), noise, t_start=0.5, nfe=1,
        objective=TANGENT_OBJECTIVE,
    )
    with pytest.raises(ValueError, match=r"\[sequence, position, d_model\]"):
        transform(activation.reshape(-1, D))


@pytest.mark.parametrize(
    "script", ["eval_tangent_reconstruction", "tangent_naturalization"]
)
def test_prepared_scripts_expose_a_working_cli(script: str) -> None:
    result = subprocess.run(
        [sys.executable, f"scripts/{script}.py", "--help"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "PREPARED, NOT RUN" in result.stdout


# --------------------------------------------------------------------------
# post-stop experiment A: the evaluators must follow the checkpoint's own path
# --------------------------------------------------------------------------


def test_evaluation_plan_is_derived_from_the_checkpoint_not_an_operator_flag(
    tmp_path: Path,
) -> None:
    """A checkpoint can never be scored on the grid of the other corruption path."""

    from interp.tangent_eval import naturalization_spec_for, reconstruction_spec_for
    from interp.train_flow import checkpoint_objective, save_flow_checkpoint

    flow = _flow()
    for objective in (TANGENT_OBJECTIVE, VP_TANGENT_OBJECTIVE):
        path = tmp_path / f"{objective}.pt"
        save_flow_checkpoint(flow, path, metadata={}, flow_objective=objective)
        assert checkpoint_objective(path) == objective
        assert reconstruction_spec_for(objective).objective == objective
        assert naturalization_spec_for(objective).objective == objective

    # and the two plans are genuinely different grids with different cell keys
    linear = naturalization_spec_for(TANGENT_OBJECTIVE)
    circle = naturalization_spec_for(VP_TANGENT_OBJECTIVE)
    assert linear.primary_cell() != circle.primary_cell()
    assert linear.t_start != circle.t_start
    # the frozen historical labels are untouched
    assert linear.primary_cell() == "pooled_t0.10_nfe1_tangent_flow"


def test_isotropic_checkpoints_are_refused_by_both_constraint_evaluators(
    tmp_path: Path,
) -> None:
    from interp.tangent_eval import naturalization_spec_for, reconstruction_spec_for

    with pytest.raises(ValueError, match="no frozen T1 plan"):
        reconstruction_spec_for("isotropic")
    with pytest.raises(ValueError, match="no frozen T2 plan"):
        naturalization_spec_for("isotropic")


def test_vp_transforms_run_on_the_matched_grid_and_hold_the_coordinate() -> None:
    from interp.tangent_eval import (
        VP_TANGENT_NATURALIZATION_SPEC,
        assert_coordinate_match,
        concatenate,
    )

    t2 = _module("tangent_naturalization")
    activation, directions, noise = _inputs()
    flow = _flow()
    target = torch.linspace(-2.0, 2.0, SEQUENCES)

    clamp = t2._ClampTangentTransform(
        flow, directions, target, noise, t_start=0.0, nfe=1,
        objective=VP_TANGENT_OBJECTIVE,
    )
    clamp(activation)
    clamp_rows = concatenate(clamp.records)

    for t_start in VP_TANGENT_NATURALIZATION_SPEC.t_start:
        arm = t2._ClampTangentTransform(
            flow, directions, target, noise, t_start=t_start, nfe=1,
            objective=VP_TANGENT_OBJECTIVE,
        )
        arm(activation)
        rows = concatenate(arm.records)
        assert assert_coordinate_match(clamp_rows, rows, tolerance=1e-2)[
            "max_arm_coordinate_difference"
        ] < 1e-2
        assert arm.diagnostics()["objective"] == VP_TANGENT_OBJECTIVE
        assert rows["orthogonal_correction_norm"].mean() > 0.0


def test_the_two_paths_produce_different_states_through_the_evaluation_transform() -> None:
    """Guards against the VP arm silently degenerating into the linear one."""

    t2 = _module("tangent_naturalization")
    activation, directions, noise = _inputs()
    flow = _flow()
    target = torch.linspace(-2.0, 2.0, SEQUENCES)
    produced = {}
    for objective in (TANGENT_OBJECTIVE, VP_TANGENT_OBJECTIVE):
        transform = t2._ClampTangentTransform(
            flow, directions, target, noise, t_start=0.25, nfe=1, objective=objective
        )
        produced[objective] = transform(activation)
    assert not torch.allclose(
        produced[TANGENT_OBJECTIVE], produced[VP_TANGENT_OBJECTIVE], atol=1e-3
    )
