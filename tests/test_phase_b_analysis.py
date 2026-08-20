"""Scientific invariants of the frozen Phase-B analysis layer and the wide rerun release."""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from interp.phase_b_analysis import (
    ANALYSIS_VERSION,
    ROW_METRICS,
    equal_alpha_paired,
    load_release_rows,
    matched_projection,
    paired_release_difference,
    primary_rows,
    resample_matrix,
    vector_bootstrap,
)
from interp.phase_b_evaluator import (
    APPROVED_EVALUATOR_CONFIG_SHA256S,
    load_evaluator_config,
)

ROOT = Path(__file__).resolve().parents[1]
NARROW_CONFIG = ROOT / "configs" / "flow_phase_b_evaluator_v1.yaml"
WIDE_CONFIG = ROOT / "configs" / "flow_phase_b_evaluator_wide60m_v1.yaml"
VECTORS = ("a", "b", "c", "d")
THRESHOLD = 0.02


def _row(
    *,
    vector: str = "a",
    alpha: float = 0.0,
    prompt_id: int = 0,
    seed: int = 0,
    method: str = "flow",
    arm_id: str = "flow_t050_nfe1",
    projection: float | None = None,
    is_stress: bool = False,
    **metrics: float,
) -> dict:
    values = {name: 0.0 for name in ROW_METRICS}
    values.update(metrics)
    return {
        "schema_version": "clean_phase_b_row_v1",
        "release_id": "release",
        "metric_versions": {},
        "split": "dev",
        "method": method,
        "arm_id": arm_id,
        "vector": vector,
        "alpha": alpha,
        "alpha_hex": float(alpha).hex(),
        "alpha_hat": alpha,
        "is_stress": is_stress,
        "prompt_id": prompt_id,
        "generation_seed": seed,
        "metrics": values,
        "geometry": None
        if projection is None
        else {
            "realized_projection_mean": projection,
            "retained_fraction_mean": None,
            "correction_norm_mean": 0.0,
            "parallel_correction_norm_mean": 0.0,
            "orthogonal_correction_norm_mean": 0.0,
            "correction_cosine_mean": 0.0,
        },
    }


def _grid(projection_of, *, nll_of, method: str, rep_of=None) -> tuple[dict, ...]:
    """One row per (vector, alpha) with a single prompt/seed family."""

    rows = []
    for vector in VECTORS:
        for alpha in (0.0, 1.0, 2.0, 3.0):
            rows.append(
                _row(
                    vector=vector,
                    alpha=alpha,
                    method=method,
                    arm_id="flow_t050_nfe1" if method == "flow" else method,
                    projection=projection_of(vector, alpha) if method == "flow" else None,
                    nll=nll_of(vector, alpha),
                    repetition_rate=0.0 if rep_of is None else rep_of(vector, alpha),
                )
            )
    return tuple(rows)


def _matched(flow, baseline, *, scale=1.0, threshold=THRESHOLD):
    matrix = resample_matrix(VECTORS, seed=1, n_resamples=64)
    return matched_projection(
        flow,
        baseline,
        VECTORS,
        matrix,
        coordinate_scale=scale,
        repetition_threshold=threshold,
        confidence=0.95,
        metrics=("nll",),
    )


@pytest.fixture(autouse=True)
def _small_primary_grid(monkeypatch):
    monkeypatch.setattr("interp.phase_b_analysis.PRIMARY_ROWS", len(VECTORS) * 4)


def test_matched_projection_interpolates_linearly_inside_the_bracket():
    flow = _grid(lambda v, a: 1.5 if a == 0.0 else 0.0, nll_of=lambda v, a: 10.0, method="flow")
    baseline = _grid(None, nll_of=lambda v, a: a, method="additive")
    summary, records = _matched(flow, baseline)
    supported = [item for item in records if item["status"] == "supported"]
    interpolated = {
        item["metrics"]["nll"]["matched"] for item in supported if item["alpha_hat"] == 0.0
    }
    # target 1.5 sits halfway between coordinates 1.0 and 2.0, whose NLLs are 1.0 and 2.0.
    assert interpolated == {1.5}
    assert {item["weight"] for item in supported if item["alpha_hat"] == 0.0} == {0.5}
    assert summary["extrapolation"] == "forbidden"


def test_matched_projection_refuses_to_extrapolate_beyond_the_observed_grid():
    flow = _grid(lambda v, a: 99.0, nll_of=lambda v, a: 1.0, method="flow")
    baseline = _grid(None, nll_of=lambda v, a: a, method="additive")
    with pytest.raises(ValueError, match="no supported comparison points"):
        _matched(flow, baseline)


def test_matched_projection_uses_the_shrinkage_coordinate_not_nominal_alpha():
    flow = _grid(lambda v, a: 1.6, nll_of=lambda v, a: 0.0, method="flow")
    baseline = _grid(None, nll_of=lambda v, a: a, method="shrinkage_k080")
    summary, records = _matched(flow, baseline, scale=0.8)
    supported = [item for item in records if item["status"] == "supported"]
    # Coordinates are 0, .8, 1.6, 2.4, so the target lands exactly on the alpha=2.0
    # node. Under nominal alpha it would instead have bracketed alphas 1 and 2.
    assert {item["bracket_upper_alpha_hat"] for item in supported} == {2.0}
    assert {item["weight"] for item in supported} == {1.0}
    assert {item["metrics"]["nll"]["matched"] for item in supported} == {2.0}
    assert summary["counts"]["supported"] == len(VECTORS) * 4


def test_degenerate_flow_row_and_degenerate_bracket_both_leave_the_frontier():
    flow = _grid(
        lambda v, a: 1.5 if a == 0.0 else 0.5,
        nll_of=lambda v, a: 0.0,
        method="flow",
        rep_of=lambda v, a: 1.0 if (v == "a" and a == 0.0) else 0.0,
    )
    baseline = _grid(
        None,
        nll_of=lambda v, a: a,
        method="additive",
        rep_of=lambda v, a: 1.0 if (v == "b" and a == 2.0) else 0.0,
    )
    summary, records = _matched(flow, baseline)
    statuses = {(item["vector"], item["alpha_hat"]): item["status"] for item in records}
    assert statuses[("a", 0.0)] == "degenerate_flow"
    assert statuses[("b", 0.0)] == "degenerate_bracket"
    assert statuses[("c", 0.0)] == "supported"
    assert summary["counts"] == {"supported": 14, "unsupported": 0, "degenerate": 2}


def test_a_direction_with_no_supported_points_fails_loudly():
    flow = _grid(lambda v, a: 1.5, nll_of=lambda v, a: 0.0, method="flow")
    baseline = _grid(
        None,
        nll_of=lambda v, a: a,
        method="additive",
        rep_of=lambda v, a: 1.0 if (v == "b" and a == 2.0) else 0.0,
    )
    with pytest.raises(ValueError, match=r"no supported comparison points .*'b'"):
        _matched(flow, baseline)


def test_stress_alphas_never_enter_the_primary_grid():
    rows = (
        *_grid(lambda v, a: 0.0, nll_of=lambda v, a: 0.0, method="flow"),
        _row(vector="a", alpha=9.0, is_stress=True, projection=0.0),
    )
    assert len(primary_rows(rows)) == len(VECTORS) * 4


def test_vector_bootstrap_reports_signs_and_leave_one_vector_out():
    matrix = resample_matrix(VECTORS, seed=7, n_resamples=128)
    result = vector_bootstrap(
        {"a": 1.0, "b": 1.0, "c": 1.0, "d": -3.0}, VECTORS, matrix, confidence=0.95
    )
    assert result["vector_signs"] == {"positive": 3, "negative": 1}
    assert result["mean"] == pytest.approx(0.0)
    assert result["leave_one_vector_out"]["d"] == pytest.approx(1.0)
    assert result["leave_one_vector_out"]["a"] == pytest.approx(-1.0 / 3.0)


def test_resample_matrix_is_deterministic_and_shared():
    first = resample_matrix(VECTORS, seed=11, n_resamples=32)
    second = resample_matrix(VECTORS, seed=11, n_resamples=32)
    assert np.array_equal(first, second)
    assert first.shape == (32, len(VECTORS))


def test_vector_bootstrap_rejects_a_partial_vector_set():
    matrix = resample_matrix(VECTORS, seed=3, n_resamples=8)
    with pytest.raises(ValueError, match="exactly the frozen vector set"):
        vector_bootstrap({"a": 1.0, "b": 2.0}, VECTORS, matrix, confidence=0.95)


def test_equal_alpha_pairing_requires_identical_continuation_cells():
    flow = _grid(lambda v, a: 0.0, nll_of=lambda v, a: 1.0, method="flow")
    baseline = tuple(
        row
        for row in _grid(None, nll_of=lambda v, a: 0.0, method="additive")
        if not (row["vector"] == "a" and row["alpha"] == 0.0)
    )
    matrix = resample_matrix(VECTORS, seed=5, n_resamples=8)
    with pytest.raises(ValueError):
        equal_alpha_paired(flow, baseline, VECTORS, matrix, confidence=0.95, metrics=("nll",))


def test_release_difference_is_wide_minus_narrow_and_needs_a_resolved_ci():
    matrix = resample_matrix(VECTORS, seed=13, n_resamples=2000)
    narrow = vector_bootstrap(dict.fromkeys(VECTORS, 1.0), VECTORS, matrix, confidence=0.95)
    wide = vector_bootstrap(dict.fromkeys(VECTORS, 0.5), VECTORS, matrix, confidence=0.95)
    result = paired_release_difference(wide, narrow, VECTORS, matrix, confidence=0.95)
    assert result["mean"] == pytest.approx(-0.5)
    assert result["narrow_mean"] == pytest.approx(1.0)
    assert result["wide_mean"] == pytest.approx(0.5)
    assert result["improved"] is True

    noisy = vector_bootstrap(
        {"a": -1.0, "b": 1.0, "c": -1.0, "d": 1.0}, VECTORS, matrix, confidence=0.95
    )
    unresolved = paired_release_difference(noisy, narrow, VECTORS, matrix, confidence=0.95)
    assert unresolved["improved"] is False


def test_release_rows_reject_a_foreign_release_identity(tmp_path):
    path = tmp_path / "arm.jsonl"
    path.write_text(json.dumps(_row(projection=0.0)) + "\n")
    (tmp_path / "arm.meta.json").write_text(
        json.dumps({"status": "complete", "release_id": "other", "row_count": 2880})
    )
    with pytest.raises(ValueError, match="not the frozen release identity"):
        load_release_rows(
            path,
            release_id="other",
            schema_version="clean_phase_b_row_v1",
            metric_versions={},
            expected_method="flow",
            expected_arm="flow_t050_nfe1",
        )


def test_wide_release_keeps_the_narrow_dev_design_byte_identical():
    narrow_text = NARROW_CONFIG.read_text()
    wide_text = WIDE_CONFIG.read_text()
    marker = "\nbaselines:\n"
    narrow_tail = narrow_text[narrow_text.index(marker) :].splitlines()
    wide_tail = wide_text[wide_text.index(marker) :].splitlines()
    changed = [
        line
        for line in difflib.unified_diff(narrow_tail, wide_tail, n=0, lineterm="")
        if line[:1] in {"+", "-"} and not line.startswith(("+++", "---"))
    ]
    # Below `baselines:` only the output root moves and the release_requires list
    # names the scaling provenance instead of the Phase-A report. Every vector,
    # prompt, alpha, arm, control feature, metric version, and statistical rule is
    # byte identical, which is what makes narrow and wide an apples-to-apples pair.
    assert changed == [
        "-server_output_root: /workspace/results/clean_flow_phase_b_dev_v1",
        "+server_output_root: /workspace/results/clean_flow_phase_b_dev_wide60m_v1",
        "-  - phase_a_report_sha256",
        "+  - scaling_protocol_sha256",
        "+  - scaling_selection_report_sha256",
        "+  - scaling_arm_report_sha256",
    ], changed


def test_wide_release_shares_the_frozen_dev_protocol_and_grid():
    narrow = load_evaluator_config(NARROW_CONFIG)
    wide = load_evaluator_config(WIDE_CONFIG)
    assert wide.experiment_id != narrow.experiment_id
    assert wide.arms == narrow.arms
    assert wide.phase_b is not None and wide.phase_b.vectors == narrow.phase_b.vectors
    assert wide.phase_b.alpha_hat == narrow.phase_b.alpha_hat
    assert wide.phase_b.prompts == narrow.phase_b.prompts
    assert wide.phase_b.generation_seeds == narrow.phase_b.generation_seeds
    assert wide.parent_protocol_sha256 == narrow.parent_protocol_sha256
    assert wide.bootstrap_seed == narrow.bootstrap_seed
    assert wide.repetition_threshold == narrow.repetition_threshold
    assert wide.control_features == narrow.control_features
    assert wide.baselines == narrow.baselines
    assert wide.server_output_root != narrow.server_output_root


def test_wide_prior_is_the_concept_independent_selection_at_the_frozen_width():
    wide = load_evaluator_config(WIDE_CONFIG)
    prior = wide.flow_prior
    assert prior is not None
    assert prior.provenance_kind == "concept_independent_scaling_selection"
    assert prior.selected_arm == "wide60m_fw32m"
    assert prior.activation_width == wide.phase_b.d_model == 768
    assert prior.parameters == 60_407_808
    assert wide.flow_checkpoint_sha256 != narrow_checkpoint_sha256()
    report = ROOT / "results" / "flow_scaling_2x2_v2" / "reports" / "selection.json"
    if not report.is_file():
        pytest.skip("2x2 selection report is not present in this checkout")
    selection = json.loads(report.read_text())
    assert selection["selection"]["selected_arm"] == prior.selected_arm
    assert selection["selection"]["phase_b_used"] is False
    assert selection["selection"]["steering_metrics_used"] is False


def narrow_checkpoint_sha256() -> str:
    return load_evaluator_config(NARROW_CONFIG).flow_checkpoint_sha256


def test_a_substitute_prior_may_not_reuse_the_narrow_checkpoint_identity(tmp_path):
    raw = yaml.safe_load(WIDE_CONFIG.read_text())
    raw["flow_checkpoint"]["sha256"] = narrow_checkpoint_sha256()
    path = tmp_path / "spoofed.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    APPROVED_EVALUATOR_CONFIG_SHA256S["clean_flow_phase_b_dev_wide60m_v1"] = __import__(
        "interp.activations", fromlist=["file_sha256"]
    ).file_sha256(path)
    try:
        with pytest.raises(ValueError, match="must not reuse the narrow Phase-A checkpoint"):
            load_evaluator_config(path)
    finally:
        APPROVED_EVALUATOR_CONFIG_SHA256S["clean_flow_phase_b_dev_wide60m_v1"] = (
            "83bcc7344cfdded76a24965f6f7d0ae329ab2b58b32a90fd1e250ca910520dbd"
        )


def test_unregistered_evaluator_release_is_refused(tmp_path):
    raw = yaml.safe_load(WIDE_CONFIG.read_text())
    raw["experiment_id"] = "clean_flow_phase_b_dev_wide60m_v2"
    path = tmp_path / "unregistered.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ValueError, match="unknown evaluator release"):
        load_evaluator_config(path)


CAPACITY_CONFIG = ROOT / "configs" / "flow_phase_b_evaluator_narrow16m_fw32m_v1.yaml"


def test_capacity_control_release_moves_only_the_prior():
    """The capacity arm must differ from the wide release in capacity and nothing else."""

    wide = load_evaluator_config(WIDE_CONFIG)
    control = load_evaluator_config(CAPACITY_CONFIG)
    assert control.experiment_id != wide.experiment_id
    assert control.server_output_root != wide.server_output_root
    assert control.flow_checkpoint_sha256 != wide.flow_checkpoint_sha256
    # everything that defines the measurement is shared
    assert control.arms == wide.arms
    assert control.baselines == wide.baselines
    assert control.control_features == wide.control_features
    assert control.metric_versions == wide.metric_versions
    assert control.parent_protocol_sha256 == wide.parent_protocol_sha256
    assert control.bootstrap_seed == wide.bootstrap_seed
    assert control.bootstrap_resamples == wide.bootstrap_resamples
    assert control.repetition_threshold == wide.repetition_threshold
    assert control.phase_b.vectors == wide.phase_b.vectors
    assert control.phase_b.prompts == wide.phase_b.prompts
    assert control.phase_b.alpha_hat == wide.phase_b.alpha_hat
    assert control.phase_b.generation_seeds == wide.phase_b.generation_seeds
    assert control.phase_b.off_distribution_norm == wide.phase_b.off_distribution_norm

    narrow_prior, wide_prior = control.flow_prior, wide.flow_prior
    assert narrow_prior is not None and wide_prior is not None
    # identical training data, statistics, and budget; only architecture differs
    assert narrow_prior.training_dataset == wide_prior.training_dataset
    assert narrow_prior.training_statistics_sha256 == wide_prior.training_statistics_sha256
    assert narrow_prior.selection_protocol_sha256 == wide_prior.selection_protocol_sha256
    assert narrow_prior.selection_report_sha256 == wide_prior.selection_report_sha256
    assert narrow_prior.history_entries == wide_prior.history_entries
    assert narrow_prior.architecture_config != wide_prior.architecture_config
    assert narrow_prior.parameters == 16_147_200
    assert wide_prior.parameters == 60_407_808
    assert narrow_prior.activation_width == wide_prior.activation_width == 768
    assert narrow_prior.provenance_kind == "concept_independent_scaling_capacity_control"
    # the loader refuses any prior whose config does not assert this, so reaching
    # here already proves steering evidence did not choose the checkpoint
    assert yaml.safe_load(CAPACITY_CONFIG.read_text())["flow_prior"][
        "selected_by_steering_evidence"
    ] is False


def test_capacity_control_role_refuses_the_selected_arm(tmp_path):
    """A capacity control must be a sibling arm, never the arm the rule selected."""

    reports = ROOT / "results" / "flow_scaling_2x2_v2" / "reports"
    if not (reports / "selection.json").is_file():
        pytest.skip("2x2 selection report is not present in this checkout")
    from interp.scaling import validate_selected_flow_prior

    kwargs = dict(
        expected_arm="wide60m_fw32m",
        training_experiment_id="flow_scaling_wide60m_fw32m_v2",
        checkpoint_path=Path("/workspace/checkpoints/x/best_step_249500.pt"),
        run_meta_path=Path("/workspace/checkpoints/x/meta.json"),
        best_pointer_path=Path("/workspace/checkpoints/x/best.json"),
    )
    with pytest.raises(ValueError, match="cannot also be its own capacity control"):
        validate_selected_flow_prior(
            ROOT / "configs" / "flow_scaling_2x2_v2.yaml",
            reports / "selection.json",
            reports / "wide60m_fw32m.json",
            role="capacity_control",
            **kwargs,
        )


def test_selected_role_still_refuses_a_non_selected_arm(tmp_path):
    """Widening the role must not have loosened the original selected-arm guard."""

    reports = ROOT / "results" / "flow_scaling_2x2_v2" / "reports"
    if not (reports / "selection.json").is_file():
        pytest.skip("2x2 selection report is not present in this checkout")
    from interp.scaling import validate_selected_flow_prior

    with pytest.raises(ValueError, match="frozen rule selected"):
        validate_selected_flow_prior(
            ROOT / "configs" / "flow_scaling_2x2_v2.yaml",
            reports / "selection.json",
            reports / "narrow16m_fw32m.json",
            expected_arm="narrow16m_fw32m",
            training_experiment_id="flow_scaling_narrow16m_fw32m_v2",
            checkpoint_path=Path("/workspace/checkpoints/x/best_step_249500.pt"),
            run_meta_path=Path("/workspace/checkpoints/x/meta.json"),
            best_pointer_path=Path("/workspace/checkpoints/x/best.json"),
            role="selected",
        )


def test_unknown_flow_prior_role_is_refused():
    from interp.scaling import validate_selected_flow_prior

    reports = ROOT / "results" / "flow_scaling_2x2_v2" / "reports"
    if not (reports / "selection.json").is_file():
        pytest.skip("2x2 selection report is not present in this checkout")
    with pytest.raises(ValueError, match="unsupported flow-prior role"):
        validate_selected_flow_prior(
            ROOT / "configs" / "flow_scaling_2x2_v2.yaml",
            reports / "selection.json",
            reports / "narrow16m_fw32m.json",
            expected_arm="narrow16m_fw32m",
            training_experiment_id="flow_scaling_narrow16m_fw32m_v2",
            checkpoint_path=Path("/workspace/checkpoints/x/best_step_249500.pt"),
            run_meta_path=Path("/workspace/checkpoints/x/meta.json"),
            best_pointer_path=Path("/workspace/checkpoints/x/best.json"),
            role="whatever_is_convenient",
        )


def _stub_analysis(release_id, checkpoint, comparison, prior=True):
    return {
        "status": "complete",
        "analysis_version": ANALYSIS_VERSION,
        "release_id": release_id,
        "flow_checkpoint_sha256": checkpoint,
        "epsilon_cell_digest": "e" * 64,
        "statistics": {"bootstrap_seed": 1, "bootstrap_resamples": 4, "bootstrap_confidence": 0.95},
        "vectors": ["a", "b"],
        "protected_data": {"held_out_accessed": False},
        "flow_prior": {"comparison_release_id": comparison} if prior else None,
        "arms": {},
    }


def _pairing_error(narrow, wide, tmp_path):
    """Run only the pairing guards of the compare subcommand."""

    import subprocess
    import sys

    np_, wp = tmp_path / "n.json", tmp_path / "w.json"
    np_.write_text(json.dumps(narrow))
    wp.write_text(json.dumps(wide))
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_flow_phase_b.py"),
            "compare",
            "--narrow",
            str(np_),
            "--wide",
            str(wp),
            "--output",
            str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return proc.stderr


def test_sibling_releases_of_one_comparison_may_be_paired(tmp_path):
    """A capacity control and the selected arm are siblings, not a direct pair."""

    common = "clean_flow_phase_b_dev_v1_original"
    narrow = _stub_analysis("release_narrow16m", "a" * 64, common)
    wide = _stub_analysis("release_wide60m", "b" * 64, common)
    err = _pairing_error(narrow, wide, tmp_path)
    assert "neither a declared comparison pair nor siblings" not in err


def test_unrelated_releases_are_still_refused(tmp_path):
    """Widening the pairing rule must not allow two arbitrary releases."""

    narrow = _stub_analysis("release_narrow16m", "a" * 64, "some_other_release")
    wide = _stub_analysis("release_wide60m", "b" * 64, "a_different_release")
    err = _pairing_error(narrow, wide, tmp_path)
    assert "neither a declared comparison pair nor siblings" in err


def test_pairing_still_requires_matched_epsilon(tmp_path):
    common = "clean_flow_phase_b_dev_v1_original"
    narrow = _stub_analysis("release_narrow16m", "a" * 64, common)
    wide = _stub_analysis("release_wide60m", "b" * 64, common)
    wide["epsilon_cell_digest"] = "f" * 64
    err = _pairing_error(narrow, wide, tmp_path)
    assert "did not draw the same epsilon" in err
