"""T1/T2 evaluation harness: matched controls, validity gates, frozen verdicts.

No GPT-2, no activation artifact, no checkpoint. These test the analysis logic
that decides what a future GPU run is allowed to claim.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from interp.conditional_flow import (
    ConditionalFlowMatcher,
    ConditionEncoderConfig,
    clamp_seed,
)
from interp.flow_core import ActivationNormalizer, FlowModelConfig
from interp.tangent_eval import (
    T1_MIN_RECOVERED_FRACTION,
    TANGENT_NATURALIZATION_SPEC,
    TANGENT_RECONSTRUCTION_SPEC,
    assert_coordinate_match,
    clustered_bootstrap_mean_ci,
    concatenate,
    natural_coordinate,
    naturalization_summary,
    reconstruction_summary,
    t1_verdict,
    t2_cell_report,
    t2_experiment_verdict,
    tangent_corrupted_activation,
    tangent_geometry,
)
from interp.tangent_flow import clamp_then_tangent_flow

D = 12
ROWS = 20


def _fixture(seed: int = 20260816):  # noqa: ANN202
    generator = torch.Generator().manual_seed(seed)
    h = torch.randn(ROWS, D, generator=generator) * 3.0
    v = torch.randn(ROWS, D, generator=generator)
    v = v / v.norm(dim=-1, keepdim=True)
    noise = torch.randn(ROWS, D, generator=generator)
    c_target = torch.randn(ROWS, 1, generator=generator) * 2.0
    mean = torch.randn(D, generator=generator)
    std = torch.rand(D, generator=generator) + 0.5
    return h, v, noise, c_target, ActivationNormalizer(mean, std, 1e-5)


def _null_model(normalizer: ActivationNormalizer) -> ConditionalFlowMatcher:
    """A conditional flow whose predicted velocity is identically zero."""

    torch.manual_seed(0)
    cfg = FlowModelConfig(
        d_model=D, d_mlp=16, n_blocks=1, time_dim=4, time_hidden=8, max_period=10000.0
    )
    model = ConditionalFlowMatcher(cfg, ConditionEncoderConfig(cond_hidden=4), normalizer)
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.zero_()
    return model.eval()


# --------------------------------------------------------------------------
# matched control
# --------------------------------------------------------------------------


@pytest.mark.parametrize("t_start", [0.10, 0.25, 0.50, 0.75])
def test_corrupted_control_is_exactly_the_flow_starting_state(t_start: float) -> None:
    """A zero-velocity flow must return its own seed: the control is the same path."""

    h, v, noise, c_target, normalizer = _fixture()
    model = _null_model(normalizer)
    control = tangent_corrupted_activation(
        model, h, v, c_target, noise=noise, t_start=t_start
    )
    out = clamp_then_tangent_flow(
        model, h, v, c_target, noise=noise, t_start=t_start, nfe=3
    )
    assert torch.allclose(control, out.activation, atol=1e-4)


def test_corrupted_control_at_zero_time_is_the_clamp() -> None:
    h, v, noise, c_target, normalizer = _fixture()
    model = _null_model(normalizer)
    control = tangent_corrupted_activation(
        model, h, v, c_target, noise=noise, t_start=0.0
    )
    assert torch.allclose(control, clamp_seed(h, v, c_target))


def test_corrupted_control_preserves_the_coordinate() -> None:
    h, v, noise, c_target, normalizer = _fixture()
    model = _null_model(normalizer)
    control = tangent_corrupted_activation(
        model, h, v, c_target, noise=noise, t_start=0.5
    )
    assert torch.allclose(
        (control * v).sum(-1, keepdim=True), c_target, atol=1e-2
    )


def test_natural_coordinate_makes_the_clamp_a_no_op() -> None:
    """T1 conditions on the activation's own coordinate, so nothing is clamped away."""

    h, v, _, _, _ = _fixture()
    c = natural_coordinate(h, v)
    assert torch.allclose(clamp_seed(h, v, c), h, atol=1e-4)


# --------------------------------------------------------------------------
# per-row geometry
# --------------------------------------------------------------------------


def test_tangent_geometry_separates_parallel_from_orthogonal_error() -> None:
    h, v, _, _, _ = _fixture()
    c = natural_coordinate(h, v)
    # a purely parallel perturbation
    parallel_only = h + 0.7 * v
    rows = tangent_geometry(h, parallel_only, v, c)
    assert np.allclose(rows["tangent_error"], 0.0, atol=1e-5)
    assert np.allclose(rows["parallel_error"], 0.7, atol=1e-4)
    assert np.allclose(rows["coordinate_abs_error"], 0.7, atol=1e-4)

    # a purely orthogonal perturbation
    delta = torch.randn(ROWS, D, generator=torch.Generator().manual_seed(3))
    delta = delta - (delta * v).sum(-1, keepdim=True) * v
    rows = tangent_geometry(h, h + delta, v, c)
    assert np.allclose(rows["parallel_error"], 0.0, atol=1e-5)
    assert np.allclose(rows["coordinate_abs_error"], 0.0, atol=1e-4)
    assert (rows["tangent_error"] > 0.0).all()


def test_concatenate_joins_hook_batches_in_order() -> None:
    parts = [
        {"a": np.array([1.0, 2.0]), "b": np.array([3.0])},
        {"a": np.array([4.0]), "b": np.array([5.0, 6.0])},
    ]
    joined = concatenate(parts)
    assert joined["a"].tolist() == [1.0, 2.0, 4.0]
    assert joined["b"].tolist() == [3.0, 5.0, 6.0]
    with pytest.raises(ValueError, match="nothing to concatenate"):
        concatenate([])


# --------------------------------------------------------------------------
# T1 summary and gate
# --------------------------------------------------------------------------


def _t1_rows(n: int = 64) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "relative_l2_to_clean": rng.random(n),
        "cosine_to_clean": rng.random(n),
        "tangent_error": rng.random(n),
        "parallel_error": np.zeros(n),
        "coordinate_abs_error": np.full(n, 1e-6),
        "c_target": rng.random(n),
        "c_realised": rng.random(n),
        "c0_clean": rng.random(n),
    }


def _clusters(n: int = 64, n_directions: int = 16) -> np.ndarray:
    """Direction label per validation sequence, as the frozen plan assigns them."""

    return np.tile(np.arange(n_directions), n // n_directions)


def _arms(
    reconstructed: np.ndarray,
    clean: np.ndarray,
    corrupted: np.ndarray,
    *,
    spec=TANGENT_RECONSTRUCTION_SPEC,  # noqa: ANN001
) -> dict[str, dict]:
    """An arms dict keyed exactly as the T1 evaluator writes it."""

    clusters = _clusters(len(clean))
    return {
        spec.primary_cell(): reconstruction_summary(
            _t1_rows(len(clean)), reconstructed, clean, corrupted, clusters
        ),
        spec.corruption_cell(): reconstruction_summary(
            _t1_rows(len(clean)), corrupted, clean, corrupted, clusters
        ),
    }


def test_reconstruction_summary_reports_recovered_damage() -> None:
    clean = np.full(32, 3.0)
    corrupted = np.full(32, 4.0)
    reconstructed = np.full(32, 3.2)
    summary = reconstruction_summary(_t1_rows(32), reconstructed, clean, corrupted, _clusters(32))
    assert summary["corruption_delta_lm"] == pytest.approx(1.0)
    assert summary["delta_lm_vs_clean"] == pytest.approx(0.2)
    assert summary["recovered_damage"] == pytest.approx(0.8)
    assert summary["recovered_fraction"] == pytest.approx(0.8)


def test_recovered_fraction_is_omitted_when_corruption_did_no_damage() -> None:
    clean = np.full(32, 3.0)
    summary = reconstruction_summary(_t1_rows(32), clean, clean, clean, _clusters(32))
    assert summary["recovered_fraction"] is None
    assert "meaningless" in summary["recovered_fraction_note"]


def test_t1_threshold_is_frozen() -> None:
    """Human-approved 2026-08-16, before any real T1 result.

    A pragmatic preregistered convention, not a derived constant. Changing it
    after a T1 result is observed makes the gate post-hoc, so this test exists
    to make that edit loud rather than silent.
    """

    assert T1_MIN_RECOVERED_FRACTION == 0.25
    verdict = t1_verdict(
        _arms(np.full(64, 3.2), np.full(64, 3.0), np.full(64, 4.0)),
    )
    assert verdict["min_recovered_fraction"] == 0.25
    assert "frozen 2026-08-16" in verdict["threshold_status"]
    assert "not a theoretically derived constant" in verdict["threshold_status"]


def test_t1_gate_passes_only_on_a_material_and_significant_recovery() -> None:
    clean = np.full(64, 3.0)
    corrupted = np.full(64, 4.0) + np.random.default_rng(1).normal(0, 0.01, 64)

    good = t1_verdict(_arms(np.full(64, 3.2), clean, corrupted))
    weak = t1_verdict(_arms(np.full(64, 3.95), clean, corrupted))

    assert good["verdict"] == "PASS"
    assert weak["verdict"] == "FAIL"
    assert "do not proceed to T2" in weak["consequence"]


def test_t1_gate_fails_when_the_corruption_did_nothing() -> None:
    clean = np.full(64, 3.0)
    assert t1_verdict(_arms(clean, clean, clean))["verdict"] == "FAIL"


def test_t1_primary_cell_is_frozen_and_order_independent() -> None:
    """P1 regression: reordering the diagnostic grid must not move the decision."""

    from dataclasses import replace

    spec = TANGENT_RECONSTRUCTION_SPEC
    assert (spec.primary_t_start, spec.primary_nfe) == (0.50, 1)
    assert spec.primary_cell() == "t0.50_nfe1_tangent"

    reordered = replace(
        spec, t_start=tuple(reversed(spec.t_start)), nfe=tuple(reversed(spec.nfe))
    )
    assert reordered.t_start[0] != spec.t_start[0]
    assert reordered.nfe[0] != spec.nfe[0]
    # ...and yet the gate reads the same cell.
    assert reordered.primary_cell() == spec.primary_cell()
    assert reordered.corruption_cell() == spec.corruption_cell()

    clean, corrupted = np.full(64, 3.0), np.full(64, 4.0)
    arms = _arms(np.full(64, 3.2), clean, corrupted)
    assert (
        t1_verdict(arms, spec=reordered)["verdict"]
        == t1_verdict(arms, spec=spec)["verdict"]
    )
    assert t1_verdict(arms, spec=reordered)["primary_cell"] == spec.primary_cell()


def test_a_primary_cell_outside_the_grid_is_rejected() -> None:
    from dataclasses import replace

    with pytest.raises(ValueError, match="primary t_start must be evaluated"):
        replace(TANGENT_RECONSTRUCTION_SPEC, primary_t_start=0.99)
    with pytest.raises(ValueError, match="primary NFE must be evaluated"):
        replace(TANGENT_RECONSTRUCTION_SPEC, primary_nfe=7)


def test_a_missing_primary_cell_fails_loudly() -> None:
    with pytest.raises(KeyError, match="primary cell"):
        t1_verdict({"some_other_cell": {}})


def test_an_ineligible_run_gets_no_formal_verdict() -> None:
    """P2 regression: a non-selected checkpoint cannot receive a T1 verdict."""

    clean, corrupted = np.full(64, 3.0), np.full(64, 4.0)
    arms = _arms(np.full(64, 3.2), clean, corrupted)

    formal = t1_verdict(arms)
    assert formal["verdict"] == "PASS"

    diagnostic = t1_verdict(
        arms, formal_eligible=False, ineligible_reason="checkpoint was not selected"
    )
    assert diagnostic["verdict"] == "DIAGNOSTIC_ONLY"
    assert diagnostic["would_have_been"] == "PASS"
    assert diagnostic["ineligible_reason"] == "checkpoint was not selected"
    assert diagnostic["formal_verdict_eligible"] is False
    assert "cannot authorize T2" in diagnostic["consequence"]


# --------------------------------------------------------------------------
# T2 summary, coordinate gate, verdict
# --------------------------------------------------------------------------


def _t2_rows(n: int, c_realised: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2)
    return {
        "relative_l2_to_clean": rng.random(n),
        "cosine_to_clean": rng.random(n),
        "tangent_error": rng.random(n),
        "parallel_error": np.zeros(n),
        "coordinate_abs_error": np.full(n, 1e-6),
        "c_target": c_realised,
        "c_realised": c_realised,
        "c0_clean": rng.random(n),
        "orthogonal_correction_norm": rng.random(n),
    }


def test_coordinate_gate_rejects_an_attenuated_flow_arm() -> None:
    """The exact failure mode the closed branch produced must be refused outright."""

    target = np.full(50, 4.0)
    clamp = _t2_rows(50, target)
    honest = _t2_rows(50, target.copy())
    attenuated = _t2_rows(50, target * 0.9)

    report = assert_coordinate_match(clamp, honest)
    assert report["max_arm_coordinate_difference"] == 0.0

    with pytest.raises(ValueError, match="fixed semantic coordinate"):
        assert_coordinate_match(clamp, attenuated)


def _pooled(paired_offset: float, n: int = 16, *, per_direction=None) -> dict:  # noqa: ANN001
    """A pooled primary-cell entry shaped exactly as the T2 evaluator builds it."""

    assignment = np.repeat(np.arange(4), n // 4)
    clamp_nll = np.full(n, 3.0)
    flow_nll = clamp_nll + paired_offset
    if per_direction is not None:
        for index, offset in enumerate(per_direction):
            flow_nll[assignment == index] = clamp_nll[assignment == index] + offset
    rows = _t2_rows(n, np.full(n, 2.0))
    entry = {
        **naturalization_summary(
            rows, flow_nll, clamp_nll, assignment,
            bootstrap_seed=1, bootstrap_resamples=400, confidence=0.95,
        ),
        "coordinate_match": {"max_arm_coordinate_difference": 0.0},
    }
    return entry


def test_naturalization_summary_primary_statistic_and_direction_signs() -> None:
    summary = _pooled(-0.05)
    assert summary["primary_paired_delta_nll_vs_clamp"]["mean"] == pytest.approx(-0.05)
    assert summary["fraction_directions_negative"] == 1.0
    assert summary["lovo_paired_delta_nll_max"] < 0.0
    assert t2_experiment_verdict(summary)["verdict"] == "PASS"


def test_t2_uses_the_clustered_bootstrap_for_its_formal_gate() -> None:
    """P5 regression: the canonical interval must be the direction-clustered one."""

    summary = _pooled(-0.05)
    canonical = summary["primary_paired_delta_nll_vs_clamp"]
    assert canonical["unit"] == "direction_cluster_then_sequence"
    assert canonical["n_units"] == 4          # directions, not sequences
    assert canonical["n_observations"] == 16  # sequences
    # The sequence-level interval is kept but explicitly marked non-canonical.
    legacy = summary["paired_delta_nll_vs_clamp_sequence_level"]
    assert legacy["unit"] == "validation_sequence"
    assert "not\n" not in legacy["note"]
    assert "non-canonical" in legacy["note"]
    assert t2_experiment_verdict(summary)["bootstrap_unit"] == canonical["unit"]


def test_t2_experiment_verdict_fails_on_a_heterogeneous_or_positive_effect() -> None:
    assert t2_experiment_verdict(_pooled(+0.05))["verdict"] == "FAIL"
    assert "stops entirely" in t2_experiment_verdict(_pooled(+0.05))["consequence"]

    # three directions improve, one gets much worse: not homogeneous
    mixed = _pooled(-0.05, per_direction=[-0.05, -0.05, -0.05, +0.5])
    assert mixed["fraction_directions_negative"] == 0.75
    assert t2_experiment_verdict(mixed)["verdict"] == "FAIL"


def test_t2_experiment_rule_is_one_frozen_cell_not_any_cell_passes() -> None:
    """P9 regression: there is exactly one experiment-level decision."""

    spec = TANGENT_NATURALIZATION_SPEC
    assert (spec.primary_t_start, spec.primary_nfe) == (0.10, 1)
    assert spec.primary_pools_quantiles is True
    assert spec.primary_cell() == "pooled_t0.10_nfe1_tangent_flow"

    verdict = t2_experiment_verdict(_pooled(-0.05))
    assert verdict["scope"] == "experiment_level"
    assert "NOT any-cell-passes" in verdict["rule"]
    assert verdict["pools_target_quantiles"] is True

    # A per-cell report is explicitly not a verdict and carries no PASS key.
    cell = t2_cell_report(_pooled(-0.05))
    assert cell["scope"] == "single_cell_diagnostic"
    assert cell["not_the_experiment_verdict"] is True
    assert "verdict" not in cell


def test_t2_verdict_fails_if_the_arms_drifted_apart_in_coordinate() -> None:
    drifted = _pooled(-0.05)
    drifted["coordinate_match"] = {"max_arm_coordinate_difference": 0.5}
    verdict = t2_experiment_verdict(drifted)
    assert verdict["coordinate_held_fixed"] is False
    assert verdict["verdict"] == "FAIL"


def test_t2_ineligible_run_gets_no_formal_verdict() -> None:
    verdict = t2_experiment_verdict(
        _pooled(-0.05), formal_eligible=False, ineligible_reason="debug-mode run"
    )
    assert verdict["verdict"] == "DIAGNOSTIC_ONLY"
    assert verdict["would_have_been"] == "PASS"


# --------------------------------------------------------------------------
# clustered bootstrap mechanics (P5)
# --------------------------------------------------------------------------


def test_clustered_bootstrap_is_deterministic() -> None:
    effects = np.random.default_rng(0).normal(size=64)
    clusters = _clusters(64, 8)
    first = clustered_bootstrap_mean_ci(
        effects, clusters, seed=7, n_resamples=300, confidence=0.95
    )
    second = clustered_bootstrap_mean_ci(
        effects, clusters, seed=7, n_resamples=300, confidence=0.95
    )
    assert first == second
    different = clustered_bootstrap_mean_ci(
        effects, clusters, seed=8, n_resamples=300, confidence=0.95
    )
    assert different["ci_lower"] != first["ci_lower"]
    assert different["mean"] == first["mean"]  # the point estimate is not resampled


def test_clustered_bootstrap_groups_by_direction_not_by_row() -> None:
    """The cluster structure must actually be used, not merely recorded."""

    # Effects are constant within a direction and differ sharply across them, so
    # a direction-clustered interval must be much wider than a row-level one.
    clusters = _clusters(64, 8)
    effects = np.array([float(c) for c in clusters])
    clustered = clustered_bootstrap_mean_ci(
        effects, clusters, seed=3, n_resamples=800, confidence=0.95
    )
    assert clustered["n_units"] == 8
    assert clustered["n_observations"] == 64
    width = clustered["ci_upper"] - clustered["ci_lower"]
    assert width > 1.0, clustered

    # With every row its own cluster the interval collapses toward the row-level one.
    singleton = clustered_bootstrap_mean_ci(
        effects, np.arange(64), seed=3, n_resamples=800, confidence=0.95
    )
    assert singleton["n_units"] == 64
    assert (singleton["ci_upper"] - singleton["ci_lower"]) < width


def test_clustered_bootstrap_validates_its_inputs() -> None:
    effects = np.zeros(8)
    with pytest.raises(ValueError, match="one cluster label"):
        clustered_bootstrap_mean_ci(
            effects, np.zeros(4), seed=0, n_resamples=10, confidence=0.95
        )
    with pytest.raises(ValueError, match="nonempty"):
        clustered_bootstrap_mean_ci(
            np.array([]), np.array([]), seed=0, n_resamples=10, confidence=0.95
        )
    with pytest.raises(ValueError, match="must be finite"):
        clustered_bootstrap_mean_ci(
            np.array([np.nan]), np.array([0]), seed=0, n_resamples=10, confidence=0.95
        )


def test_t2_inference_grid_stays_cheap() -> None:
    """The protocol fixes a cheap primary grid; widening it is a protocol change."""

    assert TANGENT_NATURALIZATION_SPEC.t_start == (0.10, 0.25, 0.50)
    assert TANGENT_NATURALIZATION_SPEC.nfe == (1, 3)


# --------------------------------------------------------------------------
# formal T2 aggregation: equal quantile weight, quantile rows kept together
# --------------------------------------------------------------------------


def _quantile_effects(n_seq: int = 64, n_dir: int = 32, seed: int = 0):  # noqa: ANN202
    rng = np.random.default_rng(seed)
    assignment = np.repeat(np.arange(n_dir), n_seq // n_dir)
    effects = {q: rng.normal(-0.02, 0.05, n_seq)
               for q in ("q50", "q75", "q90", "q95", "q99")}
    return effects, assignment


def test_pooled_effect_gives_each_quantile_equal_weight() -> None:
    """P5/§5 regression: one quantile must not dominate via row counts."""

    from interp.tangent_eval import equal_weight_pooled_effect

    assignment = np.repeat(np.arange(4), 4)
    effects = {
        "q50": np.full(16, -1.0),
        "q75": np.full(16, -1.0),
        "q90": np.full(16, -1.0),
        "q95": np.full(16, -1.0),
        "q99": np.full(16, +3.0),
    }
    point = equal_weight_pooled_effect(effects, assignment)
    # equal quantile weight: (-1 -1 -1 -1 +3)/5 = -0.2
    assert point["pooled_mean"] == pytest.approx(-0.2)
    assert point["per_quantile_mean"]["q99"] == pytest.approx(3.0)
    assert point["n_quantiles"] == 5
    assert point["weighting"] == "equal_quantile_weight"


def test_pooled_effect_direction_signs_and_lovo_use_the_pooled_statistic() -> None:
    from interp.tangent_eval import equal_weight_pooled_effect

    assignment = np.repeat(np.arange(4), 4)
    effects = {q: np.full(16, -0.05) for q in ("q50", "q75", "q90", "q95", "q99")}
    # one direction is much worse
    for q in effects:
        effects[q] = effects[q].copy()
        effects[q][assignment == 3] = +0.60
    point = equal_weight_pooled_effect(effects, assignment)
    assert point["fraction_directions_negative"] == 0.75
    assert point["lovo_max"] > 0.0  # dropping a good direction leaves the bad one
    assert point["per_direction_pooled_effect"][3] == pytest.approx(0.60)


def test_hierarchical_bootstrap_keeps_quantile_rows_together() -> None:
    """§6 regression: a drawn sequence must bring all five quantiles with it.

    Constructed so row-independent resampling and grouped resampling cannot
    agree: within every sequence the five quantiles sum to zero, so any draw
    that keeps them together yields exactly zero, while independent row draws
    would scatter around zero.
    """

    from interp.tangent_eval import hierarchical_quantile_bootstrap

    n_seq, n_dir = 64, 32
    assignment = np.repeat(np.arange(n_dir), n_seq // n_dir)
    base = np.random.default_rng(5).normal(size=n_seq)
    effects = {
        "q50": base, "q75": base, "q90": base, "q95": base,
        "q99": -4.0 * base,  # the five sum to zero for every sequence
    }
    out = hierarchical_quantile_bootstrap(
        effects, assignment, seed=1, n_resamples=300, confidence=0.95
    )
    assert out["mean"] == pytest.approx(0.0, abs=1e-12)
    assert out["ci_lower"] == pytest.approx(0.0, abs=1e-12)
    assert out["ci_upper"] == pytest.approx(0.0, abs=1e-12)
    assert out["quantile_rows_kept_together"] is True
    assert out["unit"] == "direction_cluster_then_sequence"
    assert out["n_units"] == n_dir and out["n_sequences"] == n_seq
    assert out["n_observations"] == n_seq * 5


def test_hierarchical_bootstrap_is_deterministic_and_seed_sensitive() -> None:
    from interp.tangent_eval import hierarchical_quantile_bootstrap

    effects, assignment = _quantile_effects()
    a = hierarchical_quantile_bootstrap(effects, assignment, seed=7, n_resamples=200)
    b = hierarchical_quantile_bootstrap(effects, assignment, seed=7, n_resamples=200)
    c = hierarchical_quantile_bootstrap(effects, assignment, seed=8, n_resamples=200)
    assert a == b
    assert a["mean"] == c["mean"]           # point estimate is not resampled
    assert a["ci_lower"] != c["ci_lower"]   # interval is


def test_direction_clustering_widens_the_interval_versus_ignoring_it() -> None:
    """The cluster structure must actually bind, not merely be recorded."""

    from interp.tangent_eval import hierarchical_quantile_bootstrap

    n_seq, n_dir = 64, 32
    assignment = np.repeat(np.arange(n_dir), n_seq // n_dir)
    # effect depends only on direction: clustering is the whole story
    per_dir = np.random.default_rng(11).normal(size=n_dir)
    values = per_dir[assignment]
    effects = {q: values for q in ("q50", "q75", "q90", "q95", "q99")}

    clustered = hierarchical_quantile_bootstrap(
        effects, assignment, seed=3, n_resamples=600
    )
    singleton = hierarchical_quantile_bootstrap(
        effects, np.arange(n_seq), seed=3, n_resamples=600
    )
    assert clustered["n_units"] == n_dir
    assert singleton["n_units"] == n_seq
    assert (clustered["ci_upper"] - clustered["ci_lower"]) > (
        singleton["ci_upper"] - singleton["ci_lower"]
    )


def test_t2_pooled_cell_feeds_the_frozen_experiment_verdict() -> None:
    from interp.tangent_eval import t2_pooled_cell

    effects, assignment = _quantile_effects()
    cell = t2_pooled_cell(
        effects, assignment,
        {"coordinate_abs_error_max": 1e-6,
         "coordinate_match": {"max_arm_coordinate_difference": 0.0}},
        seed=20260906, n_resamples=400,
    )
    assert cell["scope"] == "pooled_equal_quantile_weight"
    assert set(cell["quantiles_used"]) == {"q50", "q75", "q90", "q95", "q99"}
    assert cell["n_paired_rows"] == 64 * 5
    verdict = t2_experiment_verdict(cell)
    assert verdict["scope"] == "experiment_level"
    assert verdict["bootstrap_unit"] == "direction_cluster_then_sequence"
    assert verdict["verdict"] in {"PASS", "FAIL"}


def test_pooled_effect_rejects_ragged_or_empty_input() -> None:
    from interp.tangent_eval import equal_weight_pooled_effect

    with pytest.raises(ValueError, match="no target quantiles"):
        equal_weight_pooled_effect({}, np.arange(4))
    with pytest.raises(ValueError, match="one effect per sequence"):
        equal_weight_pooled_effect(
            {"q50": np.zeros(4), "q75": np.zeros(3)}, np.arange(4)
        )
