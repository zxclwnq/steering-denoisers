"""The C controls must be able to return the answer that stops a claim.

The interesting failure mode is a control that says yes whatever the data say.
The local-tangent reading in particular is the one the curvature number tempts a
reader into, so `c1_alignment` is tested on synthetic per-direction records whose
answer is known by construction: alignment that peaks at the natural centre and
falls away must be accepted, and alignment that rises with concept strength --
which is what the real data do -- must be refused.

CPU, synthetic, no artifacts.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from interp.curvature import CURVATURE_SPEC

REPO = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "curvature_controls", REPO / "scripts" / "curvature_controls.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records(alignment_by_rung: list[float], *, n: int = 32, jitter: float = 0.01) -> list[dict]:
    rng = np.random.default_rng(0)
    return [
        {
            "cos_secant_direction": (
                np.array(alignment_by_rung) + rng.normal(0.0, jitter, size=5)
            ).tolist(),
            "cos_consecutive_secants": (0.70 + rng.normal(0.0, jitter, size=4)).tolist(),
            "shuffled_cos_consecutive_secants": (
                -0.45 + rng.normal(0.0, jitter, size=4)
            ).tolist(),
            "orthogonal_drift": (0.92 + rng.normal(0.0, jitter, size=5)).tolist(),
            "shuffled_orthogonal_drift": (0.999 + rng.normal(0.0, 1e-4, size=5)).tolist(),
            "split_half_pair_ceiling": (0.995 + rng.normal(0.0, 1e-3, size=4)).tolist(),
            "split_half_reliability": (0.995 + rng.normal(0.0, 1e-3, size=5)).tolist(),
        }
        for _ in range(n)
    ]


# --------------------------------------------------------------------------
# C1: the local-tangent reading must be earned
# --------------------------------------------------------------------------


def test_alignment_that_peaks_centrally_and_falls_away_supports_a_local_tangent() -> None:
    module = _module()
    records = _records([0.50, 0.70, 0.90, 0.70, 0.50])
    result = module.c1_alignment(records, CURVATURE_SPEC)
    assert result["local_tangent_reading_supported"] is True
    assert result["upper_tail_minus_central"]["ci_upper"] < 0.0
    assert result["lower_tail_minus_central"]["ci_upper"] < 0.0
    assert "good local tangent" in result["interpretation"]


def test_alignment_that_rises_with_concept_strength_refuses_the_local_tangent() -> None:
    """The shape the real data have: v tracks the motion *better* further out."""

    module = _module()
    records = _records([0.29, 0.35, 0.36, 0.37, 0.50])
    result = module.c1_alignment(records, CURVATURE_SPEC)
    assert result["local_tangent_reading_supported"] is False
    assert result["upper_tail_minus_central"]["mean"] > 0.0
    assert result["upper_tail_minus_central"]["ci_lower"] > 0.0
    assert "do not support calling v a local tangent" in result["interpretation"]


def test_flat_alignment_is_not_a_local_tangent_either() -> None:
    module = _module()
    result = module.c1_alignment(_records([0.40] * 5), CURVATURE_SPEC)
    assert result["local_tangent_reading_supported"] is False


def test_every_rung_is_reported_with_its_bin_pair_and_an_interval() -> None:
    module = _module()
    by_rung = module.c1_alignment(_records([0.29, 0.35, 0.36, 0.37, 0.50]), CURVATURE_SPEC)[
        "by_rung"
    ]
    assert len(by_rung) == CURVATURE_SPEC.n_secants
    assert by_rung[2]["bins"] == "bin2(p25..p50)->bin3"
    assert by_rung[4]["bins"] == "bin4(p75..p90)->bin5"
    for entry in by_rung:
        assert entry["ci_lower"] < entry["mean"] < entry["ci_upper"]
        assert entry["cluster"] == "direction"


# --------------------------------------------------------------------------
# C2 / C3
# --------------------------------------------------------------------------


def test_an_unpaired_difference_recovers_a_known_gap() -> None:
    module = _module()
    left = np.full(32, 0.30) + np.random.default_rng(1).normal(0, 0.01, 32)
    right = np.full(32, 0.20) + np.random.default_rng(2).normal(0, 0.01, 32)
    diff = module._independent_difference(left, right, spec=CURVATURE_SPEC)
    assert diff["mean"] == pytest.approx(0.10, abs=0.01)
    assert diff["excludes_zero"] is True
    assert diff["resampling"] == "independent_direction_clusters_both_sides"


def test_two_identical_populations_do_not_exclude_zero() -> None:
    module = _module()
    values = np.random.default_rng(3).normal(0.3, 0.01, 32)
    diff = module._independent_difference(values, values.copy(), spec=CURVATURE_SPEC)
    assert diff["excludes_zero"] is False


def test_the_random_control_is_reported_absent_rather_than_faked() -> None:
    module = _module()
    result = module.c2_random_control(_records([0.4] * 5), None, CURVATURE_SPEC)
    assert result["available"] is False
    assert "re-run the diagnostic" in result["note"]


def test_curvature_matching_an_arbitrary_axis_is_not_called_concept_specific() -> None:
    module = _module()
    records = _records([0.4] * 5)
    result = module.c2_random_control(records, _records([0.4] * 5), CURVATURE_SPEC)
    assert result["curvature_exceeds_an_arbitrary_axis"] is False
    assert "not distinguishable" in result["interpretation"]


def test_shuffling_must_visibly_destroy_the_structure() -> None:
    module = _module()
    result = module.c3_shuffle_control(_records([0.4] * 5), CURVATURE_SPEC)
    assert result["structure_is_destroyed_by_shuffling"] is True
    assert result["real_minus_shuffled"]["ci_lower"] > 0.0


def test_a_pipeline_that_curves_without_real_labels_is_called_an_artefact() -> None:
    module = _module()
    records = _records([0.4] * 5)
    for record in records:  # shuffled looks exactly like the real thing
        record["shuffled_cos_consecutive_secants"] = record["cos_consecutive_secants"]
    result = module.c3_shuffle_control(records, CURVATURE_SPEC)
    assert result["structure_is_destroyed_by_shuffling"] is False
    assert "artefact" in result["interpretation"]


# --------------------------------------------------------------------------
# C4 and the script
# --------------------------------------------------------------------------


def test_reliability_reports_unanimity_and_leave_one_direction_out() -> None:
    module = _module()
    result = module.c4_reliability(_records([0.4] * 5), CURVATURE_SPEC)
    assert result["sign_is_unanimous"] is True
    assert result["n_directions_positive"] == 32
    lovo = result["leave_one_direction_out"]
    assert lovo["lovo_min"] <= result["shortfall_interval"]["mean"] <= lovo["lovo_max"]


def test_a_shortfall_carried_by_one_direction_is_not_robust() -> None:
    module = _module()
    records = _records([0.4] * 5)
    for index, record in enumerate(records):
        # every direction sits exactly on its ceiling except one
        record["cos_consecutive_secants"] = (
            [0.0] * 4 if index == 0 else list(record["split_half_pair_ceiling"])
        )
    result = module.c4_reliability(records, CURVATURE_SPEC)
    assert result["sign_is_unanimous"] is False
    assert result["robust_to_dropping_any_direction"] is False


# --------------------------------------------------------------------------
# C5: the exploratory join
# --------------------------------------------------------------------------


def _curvature_source(indices: list[int], curvatures: list[float]) -> dict:
    return {
        "direction_pool_indices": indices,
        "per_direction": [
            {
                "cos_consecutive_secants": [1.0 - k] * 4,
                "split_half_pair_ceiling": [1.0] * 4,
            }
            for k in curvatures
        ],
    }


def _outcome(indices: list[int], effects: list[float], path: Path) -> Path:
    path.write_text(
        json.dumps({
            "experiment": "steering_denoiser_matched_strength_v1",
            "direction_pool_indices": indices,
            "verdict": {
                "pooled_effect": {
                    "per_direction_pooled_effect": {
                        str(position): value for position, value in enumerate(effects)
                    },
                    "weighting": "equal_quantile_weight",
                }
            },
        })
    )
    return path


def test_c5_needs_the_same_directions_on_both_sides(tmp_path: Path) -> None:
    """The frozen C draw is disjoint from the steering draw: that must be refused."""

    module = _module()
    source = _curvature_source([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    outcome = _outcome([90, 91, 92, 93], [0.1, 0.2, 0.3, 0.4], tmp_path / "b.json")
    result = module.c5_curvature_vs_outcome(source, outcome, CURVATURE_SPEC)
    assert result["available"] is False
    assert result["n_shared_directions"] == 0
    assert "different directions" in result["reason"]


def test_c5_joins_on_pool_index_not_on_position(tmp_path: Path) -> None:
    """The two artifacts order their directions independently.

    Joining by position would silently pair the wrong direction's curvature with
    the wrong direction's outcome and still produce a plausible correlation.
    """

    module = _module()
    n = 24
    indices = list(range(100, 100 + n))
    curvatures = [0.01 * i for i in range(n)]
    # the outcome lists the same directions in reversed order
    effects = [0.01 * i for i in range(n)][::-1]
    source = _curvature_source(indices, curvatures)
    outcome = _outcome(indices[::-1], effects, tmp_path / "b.json")
    result = module.c5_curvature_vs_outcome(source, outcome, CURVATURE_SPEC)
    assert result["available"] is True
    # reversed indices with reversed effects means each direction keeps its own
    # value, so the relationship is perfectly positive -- a positional join would
    # have produced -1.0 instead.
    assert result["correlations"]["kappa_v"]["spearman_rho"] == pytest.approx(1.0)


def test_c5_reports_a_null_as_a_null(tmp_path: Path) -> None:
    module = _module()
    n = 32
    indices = list(range(n))
    rng = np.random.default_rng(11)
    source = _curvature_source(indices, rng.normal(0.3, 0.05, n).tolist())
    outcome = _outcome(indices, rng.normal(0.0, 0.01, n).tolist(), tmp_path / "b.json")
    result = module.c5_curvature_vs_outcome(source, outcome, CURVATURE_SPEC)
    assert result["available"] is True
    assert result["relationship_detected"] is False
    assert "does not predict" in result["interpretation"]
    assert result["class"] == "exploratory"


def test_the_script_exposes_a_working_cli_and_reads_no_activations() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/curvature_controls.py", "--help"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Trains nothing" in result.stdout


def test_the_analysis_refuses_to_overwrite_an_existing_result(tmp_path: Path) -> None:
    (tmp_path / "curvature_controls.json").write_text(json.dumps({"already": "here"}))
    result = subprocess.run(
        [
            sys.executable, "scripts/curvature_controls.py",
            "--result", "results/curvature_c_v1", "--out-dir", str(tmp_path),
        ],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "refusing to overwrite a scientific result" in result.stderr
