"""Matched-strength evaluation of experiment B: does the control really control?

The tests that matter here are the ones that would catch a *silently invalid*
comparison: a shrinkage arm that does not actually carry the denoiser's realised
strength, a row misalignment between the two passes, an alpha ladder that leaves
the natural support, or a verdict read from a lambda other than the frozen one.

CPU, tiny, synthetic. No GPT-2, no real dataset, no DEV or held-out direction.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from interp.flow_core import ActivationNormalizer, FlowModelConfig
from interp.natural_support import NATURAL_SUPPORT_SPEC, natural_coordinate_stats
from interp.steering_denoiser import SteeringDenoiser
from interp.steering_denoiser_eval import (
    STEERING_DENOISER_EVAL_SPEC,
    SteeringDenoiserEvalSpec,
    alpha_ladder,
    assert_strength_match,
    spec_payload,
    steering_denoiser_verdict,
    strength_summary,
)

REPO = Path(__file__).resolve().parents[1]
D = 12
SEQUENCES = 8
POSITIONS = 5
SPEC = STEERING_DENOISER_EVAL_SPEC


def _module(name: str):  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(seed: int = 0) -> SteeringDenoiser:
    torch.manual_seed(seed)
    cfg = FlowModelConfig(
        d_model=D, d_mlp=24, n_blocks=2, time_dim=8, time_hidden=16, max_period=10000.0
    )
    normalizer = ActivationNormalizer(torch.zeros(D), torch.ones(D) * 1.5, eps=1e-5)
    return SteeringDenoiser(cfg, normalizer).eval()


def _inputs():  # noqa: ANN202
    generator = torch.Generator().manual_seed(7)
    activation = torch.randn(SEQUENCES, POSITIONS, D, generator=generator) * 2.0
    directions = torch.randn(SEQUENCES, D, generator=generator)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    alpha = torch.linspace(-6.0, 6.0, SEQUENCES)
    return activation, directions, alpha


# --------------------------------------------------------------------------
# the frozen plan
# --------------------------------------------------------------------------


def test_the_spec_inherits_the_frozen_natural_support_plan() -> None:
    assert SPEC.plan is NATURAL_SUPPORT_SPEC
    assert SPEC.primary_lambda == 1.00
    assert SPEC.primary_lambda in SPEC.corruption.lambda_grid
    assert SPEC.primary_cell() == "pooled_lambda1.00_vs_matched_shrinkage"
    assert isinstance(spec_payload(SPEC), dict)


def test_a_primary_lambda_outside_the_frozen_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="primary lambda"):
        SteeringDenoiserEvalSpec(primary_lambda=0.6)


def test_alpha_ladder_stays_inside_each_direction_s_own_support() -> None:
    rng = np.random.default_rng(0)
    coordinates = rng.normal(0.0, 7.5, size=(6, 20_000))
    stats = natural_coordinate_stats(coordinates, NATURAL_SUPPORT_SPEC)
    ladder = alpha_ladder(stats, SPEC)

    assert set(ladder) == {"p50", "p75", "p90", "p95", "p99"}
    # the alpha = 0 control
    assert np.allclose(ladder["p50"], 0.0)
    # monotone rungs, all inside the observed range
    previous = ladder["p50"]
    for key in ("p75", "p90", "p95", "p99"):
        assert (ladder[key] > previous).all()
        previous = ladder[key]
    spread = np.array([entry["std"] for entry in stats])
    assert (ladder["p99"] < 4.0 * spread).all()


def test_alpha_ladder_rejects_statistics_without_the_needed_quantiles() -> None:
    with pytest.raises(ValueError, match="coordinate statistics lack"):
        alpha_ladder([{"p50": 0.0}], SPEC)
    with pytest.raises(ValueError, match="per-direction coordinate statistics"):
        alpha_ladder([], SPEC)


# --------------------------------------------------------------------------
# the control must really control
# --------------------------------------------------------------------------


def test_strength_match_accepts_equal_strengths_and_refuses_drift() -> None:
    left = {"realised_alpha": np.linspace(-3.0, 3.0, 40)}
    right = {"realised_alpha": left["realised_alpha"] + 1e-9}
    report = assert_strength_match(left, right)
    assert report["matched"] is True
    assert report["max_arm_strength_difference"] < 1e-8

    drifted = {"realised_alpha": left["realised_alpha"] + 0.05}
    with pytest.raises(ValueError, match="not a matched comparison"):
        assert_strength_match(left, drifted)


def test_matched_shrinkage_arm_reproduces_the_denoiser_strength_exactly() -> None:
    """The load-bearing claim of experiment B, checked end to end on the hook."""

    script = _module("eval_steering_denoiser")
    activation, directions, alpha = _inputs()
    model = _model()

    denoise = script._DenoiseTransform(model, directions, alpha, lam=1.0)
    denoise(activation)
    strengths = denoise.realised_alpha()

    shrink = script._MatchedShrinkageTransform(
        model, directions, alpha, target_strength=strengths, lam=1.0
    )
    produced = shrink(activation)

    report = assert_strength_match(
        script.concatenate(denoise.records), script.concatenate(shrink.records)
    )
    assert report["matched"] is True
    assert report["max_arm_strength_difference"] < 1e-3
    assert shrink.diagnostics()["network_evaluations"] == 0
    # and the control is pure additive steering: no orthogonal movement at all
    rows = script.concatenate(shrink.records)
    assert float(np.abs(rows["orthogonal_correction_norm"]).max()) < 1e-3
    assert produced.shape == activation.shape


@pytest.mark.parametrize("hook_batch", (1, 3, 8))
def test_the_two_passes_stay_aligned_under_any_hook_batching(hook_batch: int) -> None:
    """Row alignment is what makes the matched arm matched. Batching must not break it."""

    script = _module("eval_steering_denoiser")
    activation, directions, alpha = _inputs()
    model = _model()

    denoise = script._DenoiseTransform(model, directions, alpha, lam=0.75)
    for start in range(0, SEQUENCES, hook_batch):
        denoise(activation[start : start + hook_batch])
    strengths = denoise.realised_alpha()
    assert strengths.size == SEQUENCES * POSITIONS

    shrink = script._MatchedShrinkageTransform(
        model, directions, alpha, target_strength=strengths, lam=0.75
    )
    for start in range(0, SEQUENCES, hook_batch):
        shrink(activation[start : start + hook_batch])
    assert assert_strength_match(
        script.concatenate(denoise.records), script.concatenate(shrink.records)
    )["matched"]


def test_the_matched_arm_refuses_to_run_past_its_recorded_strengths() -> None:
    script = _module("eval_steering_denoiser")
    activation, directions, alpha = _inputs()
    model = _model()
    short = np.zeros(SEQUENCES * POSITIONS - 1)
    shrink = script._MatchedShrinkageTransform(
        model, directions, alpha, target_strength=short, lam=1.0
    )
    with pytest.raises(ValueError, match="same rows in the same order"):
        shrink(activation)


def test_lambda_zero_arm_is_additive_steering_with_no_evaluations() -> None:
    script = _module("eval_steering_denoiser")
    activation, directions, alpha = _inputs()
    model = _model()
    transform = script._DenoiseTransform(model, directions, alpha, lam=0.0)
    produced = transform(activation)

    expected = activation + alpha[:, None, None] * directions[:, None, :]
    assert torch.allclose(produced, expected, atol=1e-5)
    assert transform.diagnostics()["network_evaluations"] == 0
    assert transform.diagnostics()["arm"] == "additive"
    rows = script.concatenate(transform.records)
    assert np.allclose(rows["realised_alpha"], rows["requested_alpha"], atol=1e-3)


def test_the_denoiser_arm_actually_moves_the_activation() -> None:
    """Guards against an arm that silently degenerates into the baseline."""

    script = _module("eval_steering_denoiser")
    activation, directions, alpha = _inputs()
    model = _model()
    additive = script._DenoiseTransform(model, directions, alpha, lam=0.0)(activation)
    denoised = script._DenoiseTransform(model, directions, alpha, lam=1.0)(activation)
    assert not torch.allclose(additive, denoised, atol=1e-3)


# --------------------------------------------------------------------------
# the failure mode the experiment exists to detect
# --------------------------------------------------------------------------


def test_a_perfect_steering_inverter_is_scored_as_attenuation_not_repair() -> None:
    """A model that just undoes the steering must not look like a win."""

    script = _module("eval_steering_denoiser")
    activation, directions, alpha = _inputs()
    model = _model()
    clean_rows = activation.reshape(-1, D)

    class _Inverter(SteeringDenoiser):
        def forward(self, z, t=None):  # noqa: ANN001, ANN201, ARG002
            return self.normalizer.normalize(clean_rows) - z

    inverter = _Inverter(model.cfg, model.normalizer).eval()
    denoise = script._DenoiseTransform(inverter, directions, alpha, lam=1.0)
    denoise(activation)
    rows = script.concatenate(denoise.records)

    # It gave back all the concept ...
    assert float(np.abs(rows["realised_alpha"]).max()) < 1e-2
    summary = strength_summary(
        rows,
        np.zeros(SEQUENCES),
        np.zeros(SEQUENCES),
        np.zeros(SEQUENCES),
        np.arange(SEQUENCES) % 4,
        spec=SPEC,
    )
    assert abs(float(summary["retained_strength_fraction"])) < 1e-2
    # ... and its matched control is the clean activation, so the comparison is
    # exactly zero rather than a spurious quality win.
    shrink = script._MatchedShrinkageTransform(
        inverter, directions, alpha,
        target_strength=denoise.realised_alpha(), lam=1.0,
    )
    produced = shrink(activation)
    assert torch.allclose(produced, activation, atol=1e-2)


def test_realised_strength_is_reported_even_at_the_zero_alpha_rung() -> None:
    rows = {
        "requested_alpha": np.zeros(20),
        "realised_alpha": np.zeros(20),
        "orthogonal_correction_norm": np.zeros(20),
        "parallel_correction_norm": np.zeros(20),
    }
    summary = strength_summary(
        rows, np.zeros(20), np.zeros(20), np.zeros(20), np.arange(20) % 5, spec=SPEC
    )
    assert summary["retained_strength_fraction"] is None
    assert "retained_strength_note" in summary


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------


def _effects(mean: float, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        key: mean + rng.normal(0.0, 0.01, size=32)
        for key in ("p50", "p75", "p90", "p95", "p99")
    }


def test_a_clear_improvement_is_a_candidate_positive_and_nothing_stronger() -> None:
    verdict = steering_denoiser_verdict(_effects(-0.5), np.arange(32) % 8, spec=SPEC)
    assert verdict["verdict"] == "CANDIDATE_POSITIVE"
    assert "remaining controls" in verdict["interpretation"]
    assert verdict["primary_lambda"] == 1.00


def test_no_improvement_is_reported_as_attenuation() -> None:
    verdict = steering_denoiser_verdict(_effects(+0.5), np.arange(32) % 8, spec=SPEC)
    assert verdict["verdict"] == "NEGATIVE"
    assert "attenuation" in verdict["interpretation"]


def test_an_effect_indistinguishable_from_zero_is_not_a_positive() -> None:
    verdict = steering_denoiser_verdict(_effects(0.0), np.arange(32) % 8, spec=SPEC)
    assert verdict["verdict"] == "NEGATIVE"


def test_the_verdict_ships_per_direction_robustness() -> None:
    """A pooled mean can hide a result carried by a handful of directions.

    The protocol requires per-direction effects before a candidate positive may
    be believed, so the verdict must carry them rather than leave them to be
    recomputed by hand from the raw rows.
    """

    assignment = np.arange(32) % 8
    verdict = steering_denoiser_verdict(_effects(-0.5), assignment, spec=SPEC)
    pooled = verdict["pooled_effect"]
    assert set(pooled["per_direction_pooled_effect"]) == set(np.unique(assignment).tolist())
    assert pooled["fraction_directions_negative"] == 1.0
    assert pooled["lovo_max"] < 0.0


def test_a_result_carried_by_one_direction_is_visible_as_such() -> None:
    """One huge direction can drag the pooled mean negative on its own."""

    assignment = np.arange(32) % 8
    effects = _effects(+0.05)
    for key in effects:
        effects[key][assignment == 0] = -20.0
    pooled = steering_denoiser_verdict(effects, assignment, spec=SPEC)["pooled_effect"]
    assert pooled["pooled_mean"] < 0.0
    # ... yet only one direction of eight is actually negative, and dropping it
    # flips the pooled sign. Both facts are reported beside the mean.
    assert pooled["fraction_directions_negative"] == 0.125
    assert pooled["lovo_max"] > 0.0


def test_per_direction_effects_average_to_the_equal_weight_pooled_effect() -> None:
    """The breakdown must describe the same statistic the verdict decides on."""

    assignment = np.arange(32) % 8
    pooled = steering_denoiser_verdict(_effects(-0.3), assignment, spec=SPEC)["pooled_effect"]
    by_direction = pooled["per_direction_pooled_effect"]
    assert float(np.mean(list(by_direction.values()))) == pytest.approx(
        pooled["pooled_mean"], rel=1e-9, abs=1e-12
    )


def test_an_ineligible_run_issues_no_verdict_at_all() -> None:
    verdict = steering_denoiser_verdict(
        _effects(-0.5), np.arange(32) % 8, spec=SPEC,
        formal_eligible=False, ineligible_reason="debug-mode run",
    )
    assert verdict["verdict"] is None
    assert verdict["ineligible_reason"] == "debug-mode run"


def test_the_script_exposes_a_working_cli() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/eval_steering_denoiser.py", "--help"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "PREPARED, NOT RUN" in result.stdout
