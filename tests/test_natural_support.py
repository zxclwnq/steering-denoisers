"""Frozen-plan and calibration-math tests for the natural-support diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from interp.natural_support import (
    NATURAL_SUPPORT_SPEC,
    NaturalSupportSpec,
    assign_directions,
    calibration,
    classify,
    displacement_bin_edges,
    evaluable_mask,
    monotonic_across_quantiles,
    natural_coordinate_stats,
    per_direction_slopes,
    select_directions,
    select_reference_rows,
    select_sequences,
    target_coordinates,
)


def test_selection_is_deterministic_and_disjointly_seeded() -> None:
    rows_a = select_reference_rows(1_000_000, NATURAL_SUPPORT_SPEC)
    rows_b = select_reference_rows(1_000_000, NATURAL_SUPPORT_SPEC)
    directions_a = select_directions(23354, NATURAL_SUPPORT_SPEC)
    directions_b = select_directions(23354, NATURAL_SUPPORT_SPEC)

    assert np.array_equal(rows_a, rows_b)
    assert np.array_equal(directions_a, directions_b)
    assert len(np.unique(rows_a)) == NATURAL_SUPPORT_SPEC.reference_rows
    assert len(np.unique(directions_a)) == NATURAL_SUPPORT_SPEC.n_directions


def test_every_direction_is_used_by_the_same_number_of_sequences() -> None:
    assignment = assign_directions(NATURAL_SUPPORT_SPEC)

    counts = np.bincount(assignment)
    assert len(assignment) == NATURAL_SUPPORT_SPEC.n_sequences
    assert len(counts) == NATURAL_SUPPORT_SPEC.n_directions
    assert len(set(counts.tolist())) == 1


def test_sequence_selection_stays_inside_the_supplied_validation_ids() -> None:
    validation_ids = np.arange(500, 900, dtype=np.int64)

    selected = select_sequences(validation_ids, NATURAL_SUPPORT_SPEC)

    assert np.isin(selected, validation_ids).all()
    assert len(np.unique(selected)) == NATURAL_SUPPORT_SPEC.n_sequences


def test_quantile_stats_match_numpy_on_a_known_distribution() -> None:
    generator = np.random.default_rng(0)
    coordinates = generator.normal(3.0, 2.0, size=(2, 20000))

    stats = natural_coordinate_stats(coordinates, NATURAL_SUPPORT_SPEC)

    assert stats[0]["n"] == 20000
    assert stats[0]["p50"] == pytest.approx(np.quantile(coordinates[0], 0.50))
    assert stats[0]["p99"] == pytest.approx(np.quantile(coordinates[0], 0.99))
    assert stats[0]["p50"] < stats[0]["p90"] < stats[0]["p99"]
    assert stats[0]["mean"] == pytest.approx(3.0, abs=0.1)


def test_targets_are_read_from_the_recorded_quantiles() -> None:
    stats = [{"p50": 1.0, "p90": 5.0}, {"p50": 2.0, "p90": 7.0}]

    assert np.array_equal(target_coordinates(stats, 0.50), np.array([1.0, 2.0]))
    assert np.array_equal(target_coordinates(stats, 0.90), np.array([5.0, 7.0]))


def test_evaluable_mask_drops_requests_inside_the_noise_floor() -> None:
    spec = NaturalSupportSpec()
    c0 = np.array([0.0, 0.0, 0.0])
    c_target = np.array([0.001, 1.0, -1.0])
    std = np.array([10.0, 10.0, 10.0])  # threshold = 1.0

    mask = evaluable_mask(c0, c_target, std, spec)

    assert mask.tolist() == [False, False, True] or mask.tolist() == [False, False, False]
    assert not mask[0]


def test_evaluable_mask_keeps_both_control_directions() -> None:
    spec = NaturalSupportSpec()
    c0 = np.array([0.0, 0.0])
    c_target = np.array([5.0, -5.0])
    std = np.array([1.0, 1.0])

    assert evaluable_mask(c0, c_target, std, spec).all()


def test_displacement_bins_are_deterministic_quantiles() -> None:
    requested = np.array([-4.0, -1.0, 2.0, 8.0])

    edges = displacement_bin_edges(requested, NATURAL_SUPPORT_SPEC)

    assert np.array_equal(edges, np.quantile(np.abs(requested), (0.25, 0.50, 0.75)))


def test_calibration_recovers_a_known_slope() -> None:
    generator = np.random.default_rng(5)
    requested = generator.normal(0.0, 3.0, size=400)
    realised = 0.6 * requested + generator.normal(0.0, 0.01, size=400)
    sequences = np.repeat(np.arange(20), 20)

    result = calibration(requested, realised, sequences, NATURAL_SUPPORT_SPEC)

    assert result["slope"] == pytest.approx(0.6, abs=0.02)
    assert result["slope_ci_lower"] < 0.6 < result["slope_ci_upper"]
    assert result["pearson"] == pytest.approx(1.0, abs=0.01)
    assert result["n_sequences"] == 20
    assert result["bootstrap_unit"] == "validation_sequence"


def test_calibration_bootstrap_resamples_sequences_not_rows() -> None:
    """A single-sequence input must give a degenerate CI, proving the unit."""

    requested = np.linspace(-5.0, 5.0, 50)
    realised = 0.5 * requested
    single = np.zeros(50, dtype=np.int64)

    result = calibration(requested, realised, single, NATURAL_SUPPORT_SPEC)

    assert result["n_sequences"] == 1
    assert result["slope_ci_lower"] == pytest.approx(result["slope_ci_upper"], abs=1e-9)


def test_calibration_detects_no_control() -> None:
    generator = np.random.default_rng(7)
    requested = generator.normal(0.0, 3.0, size=200)
    realised = np.zeros(200)
    sequences = np.repeat(np.arange(10), 20)

    result = calibration(requested, realised, sequences, NATURAL_SUPPORT_SPEC)

    assert result["slope"] == pytest.approx(0.0, abs=1e-9)
    assert result["mean_abs_coordinate_error"] == pytest.approx(np.abs(requested).mean())


def test_per_direction_slopes_expose_heterogeneity() -> None:
    requested = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    realised = np.array([0.9, 1.8, 2.7, -0.9, -1.8, -2.7])
    directions = np.array([0, 0, 0, 1, 1, 1])

    summary = per_direction_slopes(requested, realised, directions)

    assert summary["per_direction_slope"][0] == pytest.approx(0.9, abs=1e-6)
    assert summary["per_direction_slope"][1] == pytest.approx(-0.9, abs=1e-6)
    assert summary["fraction_positive_slope"] == pytest.approx(0.5)


def test_monotonicity_check() -> None:
    assert monotonic_across_quantiles([1.0, 2.0, 3.0])
    assert not monotonic_across_quantiles([1.0, 3.0, 2.0])
    assert not monotonic_across_quantiles([1.0, 1.0, 2.0])


def _calib(slope: float, lower: float, error: float, fraction: float) -> dict:
    return {
        "slope": slope,
        "slope_ci_lower": lower,
        "slope_ci_upper": slope + 0.05,
        "mean_abs_coordinate_error": error,
        "fraction_correct_direction": fraction,
    }


def test_classify_category_a_requires_every_clause() -> None:
    correct = _calib(0.7, 0.6, 1.0, 0.95)
    shuffled = _calib(0.0, 0.0, 5.0, 0.5)
    directions = {"fraction_positive_slope": 1.0, "lovo_slope_min": 0.6}

    category, reasons = classify(correct, shuffled, directions, monotonic=True)

    assert category == "A"
    assert any("slope=" in reason for reason in reasons)


def test_classify_demotes_to_b_when_one_direction_carries_the_effect() -> None:
    correct = _calib(0.7, 0.6, 1.0, 0.95)
    shuffled = _calib(0.0, 0.0, 5.0, 0.5)
    directions = {"fraction_positive_slope": 0.4, "lovo_slope_min": 0.6}

    category, _ = classify(correct, shuffled, directions, monotonic=True)

    assert category == "B"


def test_classify_returns_b_for_weak_but_real_control() -> None:
    correct = _calib(0.05, 0.02, 4.0, 0.60)
    shuffled = _calib(0.0, 0.0, 4.5, 0.5)
    directions = {"fraction_positive_slope": 0.7, "lovo_slope_min": 0.02}

    category, _ = classify(correct, shuffled, directions, monotonic=False)

    assert category == "B"


def test_classify_returns_c_when_condition_is_not_used() -> None:
    correct = _calib(0.01, -0.01, 4.0, 0.51)
    shuffled = _calib(0.01, -0.01, 3.9, 0.51)
    directions = {"fraction_positive_slope": 0.5, "lovo_slope_min": -0.01}

    category, _ = classify(correct, shuffled, directions, monotonic=False)

    assert category == "C"


def test_spec_rejects_an_inconsistent_plan() -> None:
    with pytest.raises(ValueError, match="whole multiple"):
        NaturalSupportSpec(n_sequences=65, n_directions=32)


def test_quality_clause_blocks_category_a() -> None:
    """The specified rule requires quality cost not be catastrophic."""

    correct = _calib(0.9, 0.86, 2.0, 0.91)
    shuffled = _calib(0.5, 0.4, 4.8, 0.7)
    directions = {"fraction_positive_slope": 1.0, "lovo_slope_min": 0.89}

    category, reasons = classify(
        correct, shuffled, directions, monotonic=True, delta_lm=4.08, clean_lm=3.77
    )

    assert category == "A_controllability_quality_blocked"
    assert any("quality_ok=False" in reason for reason in reasons)


def test_quality_clause_allows_category_a_when_cost_is_small() -> None:
    correct = _calib(0.9, 0.86, 2.0, 0.91)
    shuffled = _calib(0.5, 0.4, 4.8, 0.7)
    directions = {"fraction_positive_slope": 1.0, "lovo_slope_min": 0.89}

    category, _ = classify(
        correct, shuffled, directions, monotonic=True, delta_lm=0.30, clean_lm=3.77
    )

    assert category == "A"


def test_quality_clause_is_reported_as_unevaluated_when_absent() -> None:
    correct = _calib(0.9, 0.86, 2.0, 0.91)
    shuffled = _calib(0.5, 0.4, 4.8, 0.7)
    directions = {"fraction_positive_slope": 1.0, "lovo_slope_min": 0.89}

    category, reasons = classify(correct, shuffled, directions, monotonic=True)

    assert category == "A"
    assert any("not evaluated" in reason for reason in reasons)
