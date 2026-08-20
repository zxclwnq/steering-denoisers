"""The D geometry must refuse to answer when it cannot answer honestly.

Three failure modes are worth a test here, and none of them is a type error:

* geometry running on a direction that failed, or never had, causal validation;
* a bootstrap for a single direction that silently reuses C's direction-clustered
  resampling, which would describe variability across directions we do not have;
* an empirical null so degenerate that "outside the random range" means nothing.

CPU, synthetic, no gemma, no GPU, no network.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "refusal_geometry", REPO / "scripts" / "refusal_geometry.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# the frozen D plan
# --------------------------------------------------------------------------


def test_the_d_spec_states_its_two_deviations_and_nothing_else() -> None:
    from interp.curvature import CURVATURE_SPEC

    d = _module().D_CURVATURE_SPEC
    assert d.min_bin_rows == 64 and CURVATURE_SPEC.min_bin_rows == 256
    assert d.n_random_directions == 128
    # everything that defines the statistic itself is untouched
    for field in (
        "cut_quantiles", "shuffle_seed", "split_half_seed", "split_half_repeats",
        "bootstrap_seed", "bootstrap_resamples", "confidence",
    ):
        assert getattr(d, field) == getattr(CURVATURE_SPEC, field), field


def test_random_axes_are_unit_vectors_and_deterministic() -> None:
    module = _module()
    axes = module.random_unit_directions(2048, module.D_CURVATURE_SPEC)
    assert axes.shape == (128, 2048)
    assert np.allclose(np.linalg.norm(axes, axis=1), 1.0)
    assert np.array_equal(
        axes, module.random_unit_directions(2048, module.D_CURVATURE_SPEC)
    )


# --------------------------------------------------------------------------
# the single-direction bootstrap
# --------------------------------------------------------------------------


def _linear_population(n: int = 3000, d: int = 12, seed: int = 0):
    """Activations whose conditional mean is exactly affine in the coordinate."""

    rng = np.random.default_rng(seed)
    direction = np.zeros(d)
    direction[0] = 1.0
    t = rng.normal(size=n)
    values = t[:, None] * direction + rng.normal(scale=0.1, size=(n, d))
    return values, direction


def test_the_bootstrap_resamples_prompts_not_directions() -> None:
    module = _module()
    values, direction = _linear_population()
    out = module.prompt_bootstrap(values, direction, None, module.D_CURVATURE_SPEC)
    assert out["resampling"] == "prompt_bootstrap_single_direction"
    assert out["n_resamples"] == module.D_CURVATURE_SPEC.bootstrap_resamples
    pooled = out["pooled_cos_secant_direction"]
    assert pooled["usable"] is True
    assert pooled["ci_lower"] < pooled["mean"] < pooled["ci_upper"]


def test_an_affine_population_shows_alignment_and_no_rotation() -> None:
    """A sanity anchor: the diagnostic must report the known answer here."""

    module = _module()
    values, direction = _linear_population()
    out = module.prompt_bootstrap(values, direction, None, module.D_CURVATURE_SPEC)
    assert out["pooled_cos_secant_direction"]["mean"] > 0.9
    assert out["pooled_cos_consecutive_secants"]["mean"] > 0.9


def test_every_rung_carries_its_own_interval() -> None:
    module = _module()
    values, direction = _linear_population()
    out = module.prompt_bootstrap(values, direction, None, module.D_CURVATURE_SPEC)
    rungs = out["cos_secant_direction_by_rung"]
    assert len(rungs) == module.D_CURVATURE_SPEC.n_secants
    for entry in rungs:
        if entry["usable"]:
            assert entry["ci_lower"] <= entry["mean"] <= entry["ci_upper"]


# --------------------------------------------------------------------------
# the empirical null
# --------------------------------------------------------------------------


def test_a_value_inside_the_random_spread_is_not_called_outside_it() -> None:
    module = _module()
    null = np.random.default_rng(0).normal(0.5, 0.05, 128)
    result = module.empirical_null(0.5, null, spec=module.D_CURVATURE_SPEC)
    assert result["usable"] is True
    assert result["outside_random_range"] is False
    assert result["above_random_interval"] is False
    assert 0.3 < result["fraction_of_random_axes_below"] < 0.7


def test_a_value_beyond_every_random_axis_is_reported_as_such() -> None:
    module = _module()
    null = np.random.default_rng(1).normal(0.3, 0.02, 128)
    result = module.empirical_null(0.95, null, spec=module.D_CURVATURE_SPEC)
    assert result["outside_random_range"] is True
    assert result["above_random_interval"] is True
    assert result["fraction_of_random_axes_below"] == 1.0


def test_a_null_too_small_to_mean_anything_is_marked_unusable() -> None:
    module = _module()
    result = module.empirical_null(0.5, np.array([0.4]), spec=module.D_CURVATURE_SPEC)
    assert result["usable"] is False


def test_a_nan_estimate_never_becomes_a_finding() -> None:
    module = _module()
    null = np.random.default_rng(2).normal(0.3, 0.02, 128)
    result = module.empirical_null(float("nan"), null, spec=module.D_CURVATURE_SPEC)
    assert result["usable"] is False


# --------------------------------------------------------------------------
# the causal gate blocks the geometry
# --------------------------------------------------------------------------


def _validation_dir(tmp_path: Path, verdict: str, sha: str = "a" * 64) -> Path:
    directory = tmp_path / "validation"
    directory.mkdir()
    (directory / "causal_validation.json").write_text(
        json.dumps({
            "verdict": {
                "verdict": verdict,
                "ablation_refusal_drop_pp": 80.0,
                "addition_refusal_rise_pp": 80.0,
            },
            "direction": {"sha256": sha},
        })
    )
    return directory


def test_geometry_refuses_to_run_after_a_failed_causal_validation(tmp_path: Path) -> None:
    validation = _validation_dir(tmp_path, "CAUSAL_CONTROL_FAIL")
    result = subprocess.run(
        [
            sys.executable, "scripts/refusal_geometry.py",
            "--validation", str(validation),
            "--direction", "data/refusal_direction/gemma-2b-it_direction.pt",
            "--metadata", "data/refusal_direction/gemma-2b-it_direction_metadata.json",
            "--splits", "data/refusal_direction/splits",
            "--out-dir", str(tmp_path / "out"),
        ],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "Experiment D stops at" in result.stderr
    # and it must not have downloaded or loaded a model first
    assert "CAUSAL_CONTROL_FAIL" in result.stderr


def test_geometry_refuses_a_direction_the_validation_did_not_validate(
    tmp_path: Path,
) -> None:
    """A pass receipt for a different vector must not license this one."""

    validation = _validation_dir(tmp_path, "CAUSAL_CONTROL_PASS", sha="b" * 64)
    result = subprocess.run(
        [
            sys.executable, "scripts/refusal_geometry.py",
            "--validation", str(validation),
            "--direction", "data/refusal_direction/gemma-2b-it_direction.pt",
            "--metadata", "data/refusal_direction/gemma-2b-it_direction_metadata.json",
            "--splits", "data/refusal_direction/splits",
            "--out-dir", str(tmp_path / "out"),
        ],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "not the same" in result.stderr


def test_the_script_exposes_a_working_cli() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/refusal_geometry.py", "--help"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "CAUSAL_CONTROL_PASS" in result.stdout


def test_the_validation_script_exposes_a_working_cli() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/refusal_causal_validation.py", "--help"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Both are required" in result.stdout


# --------------------------------------------------------------------------
# class balance inside the analysis
# --------------------------------------------------------------------------


def test_a_one_class_bin_leaves_the_balanced_analysis_short_of_secants() -> None:
    """The refusal coordinate separates the classes, so this is the expected shape.

    It must surface as "not enough usable secants" rather than as a confident
    number computed from bins that hold one class only.
    """

    module = _module()
    rng = np.random.default_rng(5)
    d = 8
    direction = np.zeros(d)
    direction[0] = 1.0
    # perfectly separated classes: harmless low on the coordinate, harmful high
    harmless = rng.normal(size=(2000, d)) + np.array([-6.0] + [0.0] * (d - 1))
    harmful = rng.normal(size=(2000, d)) + np.array([+6.0] + [0.0] * (d - 1))
    values = np.concatenate([harmless, harmful])
    classes = np.array([0] * 2000 + [1] * 2000)
    analysis = module.analyse(
        "separated", values, classes, direction, module.D_CURVATURE_SPEC
    )
    assert analysis["class_balanced"] is True
    assert analysis["usable_secants"] < module.D_CURVATURE_SPEC.n_secants


def test_a_mixed_population_can_still_be_balanced() -> None:
    module = _module()
    rng = np.random.default_rng(6)
    d = 8
    direction = np.zeros(d)
    direction[0] = 1.0
    values = rng.normal(size=(4000, d))
    classes = rng.integers(0, 2, size=4000)
    analysis = module.analyse(
        "mixed", values, classes, direction, module.D_CURVATURE_SPEC
    )
    assert analysis["usable_secants"] >= 2
    assert analysis["sufficient_for_a_claim"] is True
    assert analysis["random_axes"]["n_directions"] == 128


def test_an_analysis_reports_the_refusal_and_random_statistics_side_by_side() -> None:
    module = _module()
    values, direction = _linear_population(n=4000, d=8)
    analysis = module.analyse("plain", values, None, direction, module.D_CURVATURE_SPEC)
    for key in (
        "mean_cos_consecutive_secants",
        "mean_cos_secant_direction",
        "shortfall_below_ceiling",
    ):
        assert key in analysis["refusal_direction"]["statistics"]
        assert key in analysis["random_axes"]["statistics_mean"]
        assert key in analysis["refusal_vs_random"]
    # the planted direction is genuinely aligned, the random axes are not
    comparison = analysis["refusal_vs_random"]["mean_cos_secant_direction"]
    assert comparison["above_random_interval"] is True


def test_pytest_marker_free_module_imports_cleanly() -> None:
    assert _module().CLASS_HARMFUL == 1
    assert _module().CLASS_HARMLESS == 0
