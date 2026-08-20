"""The covariance controls must be able to take the curvature result away.

The dangerous failure here is the mirror of C's own: a control that confirms
whatever it is given. So the central tests build populations whose answer is
known by construction --- an anisotropic Gaussian, whose conditional mean is
*exactly linear* however curved the pipeline makes it look, and a genuinely
bending population --- and require the controls to tell them apart.

CPU, synthetic, no artifacts, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from interp.covariance_controls import (
    COVARIANCE_CONTROL_SPEC,
    conditional_fit_comparison,
    covariance_coordinates,
    covariance_linear_direction,
    covariance_verdict,
    gaussian_surrogate,
    match_directions,
    residual_conditional_means,
    split_sequences,
)
from interp.curvature import CURVATURE_SPEC, CurvatureSpec, bin_indices

SPEC = CurvatureSpec(min_bin_rows=64, bootstrap_resamples=200)


def _anisotropic(d: int = 24, seed: int = 0) -> np.ndarray:
    """A covariance with a wide spectrum, like a residual stream."""

    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(d, d)))[0]
    eigenvalues = np.geomspace(30.0, 0.05, d)
    return basis @ np.diag(eigenvalues) @ basis.T


def _gaussian_population(n: int = 20000, d: int = 24, seed: int = 1):
    covariance = _anisotropic(d, seed)
    rng = np.random.default_rng(seed + 100)
    values = rng.multivariate_normal(np.zeros(d), covariance, size=n)
    return values, covariance


# --------------------------------------------------------------------------
# C6.1 the covariance-predicted direction
# --------------------------------------------------------------------------


def test_the_conditional_slope_is_sigma_v_over_projected_variance() -> None:
    """The identity the whole control rests on, checked against a direct estimate."""

    values, covariance = _gaussian_population(n=200_000, d=8, seed=3)
    v = np.zeros(8)
    v[0] = 1.0
    predicted = covariance_linear_direction(covariance, v)

    # empirical conditional slope: regress h on c
    c = values @ v
    design = np.stack([np.ones_like(c), c], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    empirical = coefficients[1]
    cosine = float(
        empirical @ predicted["slope"]
        / (np.linalg.norm(empirical) * np.linalg.norm(predicted["slope"]))
    )
    assert cosine == pytest.approx(1.0, abs=1e-3)


def test_an_anisotropic_covariance_makes_sigma_v_point_away_from_v() -> None:
    """Which is exactly why cos(d_k, v) cannot be read on its own."""

    covariance = _anisotropic(24, seed=5)
    rng = np.random.default_rng(7)
    v = rng.normal(size=24)
    result = covariance_linear_direction(covariance, v)
    assert 0.0 < result["cos_v_sigma_v"] < 0.95
    assert result["projected_variance"] > 0.0


def test_an_isotropic_covariance_leaves_sigma_v_parallel_to_v() -> None:
    """The degenerate case: no anisotropy, no confound."""

    v = np.array([0.6, 0.8, 0.0])
    result = covariance_linear_direction(3.0 * np.eye(3), v)
    assert result["cos_v_sigma_v"] == pytest.approx(1.0)
    assert np.allclose(result["unit"], v / np.linalg.norm(v))


def test_a_degenerate_direction_is_refused() -> None:
    with pytest.raises(ValueError, match="projected variance"):
        covariance_linear_direction(np.zeros((4, 4)), np.array([1.0, 0, 0, 0]))


# --------------------------------------------------------------------------
# C6.2 held-out linear vs quadratic
# --------------------------------------------------------------------------


def test_the_split_is_by_sequence_never_by_row() -> None:
    sequence = np.repeat(np.arange(200), 10)
    train, test = split_sequences(sequence, COVARIANCE_CONTROL_SPEC.split_seed)
    assert train.sum() + test.sum() == sequence.size
    assert not set(sequence[train]).intersection(set(sequence[test]))
    # deterministic
    again, _ = split_sequences(sequence, COVARIANCE_CONTROL_SPEC.split_seed)
    assert np.array_equal(train, again)


def test_a_gaussian_population_gives_the_quadratic_no_held_out_advantage() -> None:
    """The key negative control: exactly linear conditional mean, however
    anisotropic the covariance is."""

    values, _ = _gaussian_population(n=40_000, d=16, seed=11)
    sequence = np.repeat(np.arange(values.shape[0] // 20), 20)
    train, test = split_sequences(sequence, COVARIANCE_CONTROL_SPEC.split_seed)
    v = np.zeros(16)
    v[3] = 1.0
    out = conditional_fit_comparison(values, v, train, test)
    # the quadratic term can only fit noise here
    assert abs(out["relative_improvement"]) < 1e-3


def test_a_genuinely_curved_population_is_detected() -> None:
    """A conditional mean with a real quadratic term must be found."""

    rng = np.random.default_rng(13)
    n, d = 40_000, 16
    c = rng.normal(0.0, 3.0, size=n)
    w = rng.normal(size=d)
    u = rng.normal(size=d)
    values = c[:, None] * w + (c**2)[:, None] * u + rng.normal(size=(n, d))
    v = w / np.linalg.norm(w)
    sequence = np.repeat(np.arange(n // 20), 20)
    train, test = split_sequences(sequence, COVARIANCE_CONTROL_SPEC.split_seed)
    out = conditional_fit_comparison(values, v, train, test)
    assert out["delta_mse_linear_minus_quadratic"] > 0.0
    assert out["relative_improvement"] > 0.1


def test_the_comparison_never_fits_and_scores_on_the_same_rows() -> None:
    values, _ = _gaussian_population(n=8000, d=8, seed=17)
    sequence = np.repeat(np.arange(400), 20)
    train, test = split_sequences(sequence, COVARIANCE_CONTROL_SPEC.split_seed)
    v = np.zeros(8)
    v[0] = 1.0
    out = conditional_fit_comparison(values, v, train, test)
    assert out["n_train_rows"] + out["n_test_rows"] == values.shape[0]
    assert out["per_row_delta"].size == out["n_test_rows"]


def test_residual_conditional_means_vanish_for_a_gaussian_population() -> None:
    values, _ = _gaussian_population(n=60_000, d=16, seed=19)
    v = np.zeros(16)
    v[2] = 1.0
    sequence = np.repeat(np.arange(values.shape[0] // 20), 20)
    train, test = split_sequences(sequence, COVARIANCE_CONTROL_SPEC.split_seed)
    assignment = bin_indices(values @ v, SPEC)
    out = residual_conditional_means(
        values, v, train, test, assignment, SPEC.n_bins, SPEC.min_bin_rows
    )
    ratios = [r for r in out["residual_to_secant_ratio"] if r is not None]
    assert ratios, "no usable bins"
    assert max(ratios) < 0.10


# --------------------------------------------------------------------------
# C6.3 the surrogate
# --------------------------------------------------------------------------


def test_the_surrogate_reproduces_the_covariance_and_is_not_isotropic() -> None:
    covariance = _anisotropic(24, seed=23)
    synthetic, receipt = gaussian_surrogate(
        np.zeros(24), covariance, 200_000, COVARIANCE_CONTROL_SPEC.surrogate_seed
    )
    assert synthetic.shape == (200_000, 24)
    assert receipt["relative_covariance_error"] < 0.05
    assert receipt["isotropic"] is False
    achieved = np.cov(synthetic, rowvar=False)
    spectrum = np.linalg.eigvalsh(achieved)
    assert spectrum.max() / spectrum.min() > 100.0


def test_the_surrogate_is_deterministic_under_its_frozen_seed() -> None:
    covariance = _anisotropic(8, seed=29)
    a, _ = gaussian_surrogate(np.zeros(8), covariance, 500, 4242)
    b, _ = gaussian_surrogate(np.zeros(8), covariance, 500, 4242)
    assert np.array_equal(a, b)


def test_a_singular_covariance_is_handled_rather_than_crashing() -> None:
    """An empirical covariance can be rank-deficient; Cholesky would fail here."""

    basis = np.linalg.qr(np.random.default_rng(31).normal(size=(6, 6)))[0]
    covariance = basis @ np.diag([5.0, 3.0, 1.0, 0.0, 0.0, 0.0]) @ basis.T
    synthetic, receipt = gaussian_surrogate(np.zeros(6), covariance, 5000, 7)
    assert np.isfinite(synthetic).all()
    assert receipt["smallest_eigenvalue"] < 1e-8


# --------------------------------------------------------------------------
# C6.4 matching
# --------------------------------------------------------------------------


def _pool(d: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, d))
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def test_matching_improves_balance_on_both_variables() -> None:
    covariance = _anisotropic(24, seed=37)
    candidates = _pool(24, 4000, 41)
    # concept directions deliberately drawn to have unusual projected variance
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    concept = eigenvectors[:, -6:].T  # the highest-variance directions
    matched = match_directions(covariance, concept, candidates)
    for key in ("projected_variance", "cos_v_sigma_v"):
        balance = matched["balance"][key]
        assert balance["mean_abs_difference_after"] < balance["mean_abs_difference_before"]
    assert matched["directions"].shape == concept.shape


def test_matching_is_without_replacement() -> None:
    covariance = _anisotropic(16, seed=43)
    candidates = _pool(16, 2000, 47)
    concept = _pool(16, 12, 53)
    matched = match_directions(covariance, concept, candidates)
    assert len(set(matched["indices"].tolist())) == 12


def test_matched_directions_really_carry_similar_projected_variance() -> None:
    covariance = _anisotropic(24, seed=59)
    candidates = _pool(24, 8000, 61)
    concept = _pool(24, 10, 67)
    matched = match_directions(covariance, concept, candidates)
    concept_stats = covariance_coordinates(covariance, concept)
    matched_stats = covariance_coordinates(covariance, matched["directions"])
    ratio = matched_stats["projected_variance"] / concept_stats["projected_variance"]
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0)


def test_the_matching_rule_is_recorded_with_the_result() -> None:
    covariance = _anisotropic(8, seed=71)
    matched = match_directions(covariance, _pool(8, 4, 73), _pool(8, 500, 79))
    assert "without replacement" in matched["rule"]
    assert matched["n_candidates"] == 500


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------


def _interval(lower: float, mean: float = 1.0) -> dict:
    return {"usable": True, "mean": mean, "ci_lower": lower, "ci_upper": lower + 1.0}


def test_every_control_agreeing_gives_the_strong_outcome() -> None:
    verdict = covariance_verdict(
        _interval(0.05), _interval(0.03), _interval(0.01), residual_structure_present=True
    )
    assert verdict["outcome"] == "CURVATURE_BEYOND_COVARIANCE"


def test_losing_the_concept_specific_excess_downgrades_the_claim() -> None:
    verdict = covariance_verdict(
        _interval(0.05), _interval(-0.02), _interval(0.01), residual_structure_present=True
    )
    assert verdict["outcome"] == "CURVATURE_NOT_CONCEPT_SPECIFIC_AFTER_MATCHING"
    assert "covariance anisotropy" in verdict["why"]


def test_a_linear_conditional_mean_gives_the_weak_outcome() -> None:
    """The outcome that takes the original finding away must be reachable."""

    verdict = covariance_verdict(
        _interval(-0.01), _interval(-0.02), _interval(-0.001),
        residual_structure_present=False,
    )
    assert verdict["outcome"] == "CURVATURE_EXPLAINED_BY_COVARIANCE"
    assert "does not survive" in verdict["why"]


def test_the_strong_outcome_needs_the_residual_structure_too() -> None:
    verdict = covariance_verdict(
        _interval(0.05), _interval(0.03), _interval(0.01), residual_structure_present=False
    )
    assert verdict["outcome"] != "CURVATURE_BEYOND_COVARIANCE"


def test_every_verdict_carries_the_prohibited_claims_and_the_cos_note() -> None:
    verdict = covariance_verdict(
        _interval(0.05), _interval(0.03), _interval(0.01), residual_structure_present=True
    )
    assert "SAE features are correlational" in verdict["prohibited_claims"]
    assert "confounded by activation covariance" in verdict["cos_dk_v_note"]


def test_the_frozen_spec_matches_the_protocol() -> None:
    assert COVARIANCE_CONTROL_SPEC.polynomial_degree == 2
    assert COVARIANCE_CONTROL_SPEC.n_candidates == 20000
    assert COVARIANCE_CONTROL_SPEC.split_seed == 20260920
    assert COVARIANCE_CONTROL_SPEC.candidate_seed == 20260921
    assert COVARIANCE_CONTROL_SPEC.surrogate_seed == 20260922
    assert CURVATURE_SPEC.min_bin_rows == 256  # C's own floor is untouched
