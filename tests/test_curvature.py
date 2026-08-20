"""Post-stop experiment C: the curvature diagnostic must not manufacture curvature.

The dangerous failure mode here is a false positive. Finite-sample noise in each
bin mean always pushes ``cos(d_k, d_k+1)`` below 1, so a diagnostic without a
calibrated ceiling would report curvature for perfectly linear data and hand the
programme a tempting post-hoc explanation for its negative results.

The central tests therefore run the whole pipeline on synthetic data with a
*known* answer: an exactly affine conditional mean must come back
NO_STRONG_CURVATURE, and a trajectory that genuinely rotates must come back
CURVATURE_DETECTED.

CPU, tiny, synthetic. No GPT-2, no real dataset, no DEV or held-out direction.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from interp.curvature import (
    CURVATURE_SPEC,
    CurvatureSpec,
    bin_indices,
    bin_means,
    bootstrap_shortfall,
    curvature_verdict,
    direction_curvature,
    pooled_curvature,
    secant_geometry,
    spec_payload,
)

REPO = Path(__file__).resolve().parents[1]
WIDTH = 48
ROWS = 20_000
SEQUENCE = np.repeat(np.arange(ROWS // 100), 100)
SPEC = CurvatureSpec(min_bin_rows=64, bootstrap_resamples=400)


def _trajectory(kappa: float, seed: int, *, width: int = WIDTH, rows: int = ROWS):  # noqa: ANN202
    """``h = c (cos(kc) w + sin(kc) a) + noise``: affine at k = 0, bending otherwise."""

    rng = np.random.default_rng(seed)
    v = rng.normal(size=width)
    v /= np.linalg.norm(v)
    w = rng.normal(size=width)
    w /= np.linalg.norm(w)
    w = w + v
    w /= np.linalg.norm(w)
    a = rng.normal(size=width)
    a -= (a @ w) * w
    a /= np.linalg.norm(a)
    c = rng.normal(0.0, 3.0, size=rows)
    angle = kappa * c
    curve = np.cos(angle)[:, None] * w[None, :] + np.sin(angle)[:, None] * a[None, :]
    return c[:, None] * curve + rng.normal(0.0, 1.0, size=(rows, width)), v


def _records(kappa: float, n_directions: int = 12) -> list[dict[str, object]]:
    records = []
    for index in range(n_directions):
        activations, v = _trajectory(kappa, 100 + index)
        records.append(
            direction_curvature(activations, v, SEQUENCE, spec=SPEC, seed_offset=index)
        )
    return records


# --------------------------------------------------------------------------
# binning and secant algebra
# --------------------------------------------------------------------------


def test_the_frozen_plan_makes_six_bins_and_five_secants() -> None:
    assert CURVATURE_SPEC.cut_quantiles == (0.10, 0.25, 0.50, 0.75, 0.90)
    assert CURVATURE_SPEC.n_bins == 6
    assert CURVATURE_SPEC.n_secants == 5
    assert isinstance(spec_payload(CURVATURE_SPEC), dict)


def test_a_malformed_plan_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        CurvatureSpec(cut_quantiles=(0.0, 0.5))
    with pytest.raises(ValueError, match="must increase"):
        CurvatureSpec(cut_quantiles=(0.5, 0.25))
    with pytest.raises(ValueError, match="at least two coordinate cuts"):
        CurvatureSpec(cut_quantiles=(0.5,))


def test_bins_partition_the_rows_at_the_declared_quantiles() -> None:
    coordinate = np.linspace(-10.0, 10.0, 10_000)
    assignment = bin_indices(coordinate, CURVATURE_SPEC)
    assert assignment.min() == 0
    assert assignment.max() == CURVATURE_SPEC.n_bins - 1
    shares = np.bincount(assignment, minlength=CURVATURE_SPEC.n_bins) / assignment.size
    # p10/p25/p50/p75/p90 cuts -> shares 0.10, 0.15, 0.25, 0.25, 0.15, 0.10
    assert np.allclose(shares, [0.10, 0.15, 0.25, 0.25, 0.15, 0.10], atol=0.01)
    # bins are ordered by coordinate
    for k in range(CURVATURE_SPEC.n_bins - 1):
        assert coordinate[assignment == k].max() <= coordinate[assignment == k + 1].min()


def test_an_undersampled_bin_yields_nan_rather_than_a_noisy_mean() -> None:
    spec = CurvatureSpec(min_bin_rows=1000)
    values = np.random.default_rng(0).normal(size=(500, 4))
    assignment = bin_indices(values[:, 0], spec)
    mu, counts = bin_means(values, assignment, spec)
    assert counts.sum() == 500
    assert np.isnan(mu).all()


def test_secant_geometry_matches_an_independent_computation() -> None:
    rng = np.random.default_rng(3)
    mu = rng.normal(size=(6, WIDTH))
    v = rng.normal(size=WIDTH)
    v /= np.linalg.norm(v)
    geometry = secant_geometry(mu, v)

    for k in range(5):
        d = mu[k + 1] - mu[k]
        parallel = float(d @ v)
        orthogonal = d - parallel * v
        assert geometry["cos_secant_direction"][k] == pytest.approx(
            parallel / np.linalg.norm(d)
        )
        assert geometry["orthogonal_drift"][k] == pytest.approx(
            np.linalg.norm(orthogonal) / np.linalg.norm(d)
        )
    for k in range(4):
        left, right = mu[k + 1] - mu[k], mu[k + 2] - mu[k + 1]
        assert geometry["cos_consecutive_secants"][k] == pytest.approx(
            left @ right / (np.linalg.norm(left) * np.linalg.norm(right))
        )


def test_a_perfectly_affine_conditional_mean_has_parallel_secants() -> None:
    """The definition the whole diagnostic rests on, with no noise at all."""

    rng = np.random.default_rng(4)
    v = rng.normal(size=WIDTH)
    v /= np.linalg.norm(v)
    w = rng.normal(size=WIDTH)
    c = np.linspace(-3.0, 3.0, 6)
    mu = c[:, None] * w[None, :] + rng.normal(size=WIDTH)[None, :]
    geometry = secant_geometry(mu, v)
    assert np.allclose(geometry["cos_consecutive_secants"], 1.0, atol=1e-10)


# --------------------------------------------------------------------------
# the calibrations, and the answers they enable
# --------------------------------------------------------------------------


def test_linear_data_is_not_reported_as_curved() -> None:
    """The false positive this diagnostic exists to avoid."""

    records = _records(0.0)
    pooled = pooled_curvature(records, SPEC)
    interval = bootstrap_shortfall(records, SPEC)
    verdict = curvature_verdict(pooled, interval=interval)
    assert verdict["verdict"] == "NO_STRONG_CURVATURE"
    assert verdict["mean_shortfall_below_ceiling"] < 0.02
    assert "must NOT be used to explain" in verdict["interpretation"]


def test_genuinely_rotating_data_is_detected() -> None:
    records = _records(0.12)
    pooled = pooled_curvature(records, SPEC)
    interval = bootstrap_shortfall(records, SPEC)
    verdict = curvature_verdict(pooled, interval=interval)
    assert verdict["verdict"] == "CURVATURE_DETECTED"
    assert interval["ci_lower"] > verdict["margin"]
    assert interval["fraction_directions_positive"] > 0.8


def test_the_noise_floor_is_what_separates_the_two_answers() -> None:
    """Without the ceiling, linear data would look curved at this sample size."""

    pooled = pooled_curvature(_records(0.0), SPEC)
    observed = np.asarray(pooled["cos_consecutive_secants_by_rung"], dtype=np.float64)
    ceiling = np.asarray(pooled["split_half_pair_ceiling_by_pair"], dtype=np.float64)
    # the raw angle is visibly below 1 even though the data are exactly linear
    assert observed.mean() < 0.999
    # and the ceiling accounts for essentially all of that gap
    assert abs(float(np.mean(ceiling - observed))) < 0.02


def test_the_shuffle_control_destroys_the_structure() -> None:
    record = _records(0.0, n_directions=1)[0]
    observed = np.asarray(record["cos_consecutive_secants"], dtype=np.float64)
    shuffled = np.asarray(record["shuffled_cos_consecutive_secants"], dtype=np.float64)
    assert np.nanmean(observed) > 0.9
    assert np.nanmean(shuffled) < 0.5


def test_a_verdict_without_any_usable_rung_is_withheld() -> None:
    pooled = {
        "cos_consecutive_secants_by_rung": [float("nan")] * 4,
        "split_half_pair_ceiling_by_pair": [float("nan")] * 4,
    }
    verdict = curvature_verdict(pooled)
    assert verdict["verdict"] is None
    assert "no rung" in verdict["reason"]


def test_the_bootstrap_clusters_on_directions_not_rows() -> None:
    records = _records(0.12)
    interval = bootstrap_shortfall(records, SPEC)
    assert interval["cluster"] == "direction"
    assert interval["n_directions"] == len(records)
    assert interval["ci_lower"] < interval["mean"] < interval["ci_upper"]
    with pytest.raises(ValueError, match="at least two directions"):
        bootstrap_shortfall(records[:1], SPEC)


def test_a_point_estimate_alone_cannot_declare_curvature() -> None:
    """A wide interval must veto a borderline point estimate."""

    pooled = {
        "cos_consecutive_secants_by_rung": [0.90, 0.90, 0.90, 0.90],
        "split_half_pair_ceiling_by_pair": [0.99, 0.99, 0.99, 0.99],
    }
    wide = {"ci_lower": -0.30, "ci_upper": 0.40, "mean": 0.09}
    assert curvature_verdict(pooled, interval=wide)["verdict"] == "NO_STRONG_CURVATURE"
    tight = {"ci_lower": 0.07, "ci_upper": 0.11, "mean": 0.09}
    assert curvature_verdict(pooled, interval=tight)["verdict"] == "CURVATURE_DETECTED"


# --------------------------------------------------------------------------
# the script
# --------------------------------------------------------------------------


def test_the_script_exposes_a_working_cli_and_needs_no_language_model() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/curvature_diagnostic.py", "--help"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "trains nothing" in result.stdout

    spec = importlib.util.spec_from_file_location(
        "curvature_diagnostic", REPO / "scripts" / "curvature_diagnostic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # random control axes are unit vectors and are not pool directions
    axes = module._random_directions(WIDTH)
    assert axes.shape == (CURVATURE_SPEC.n_random_directions, WIDTH)
    assert np.allclose(np.linalg.norm(axes, axis=1), 1.0)


def test_class_balancing_changes_nothing_when_no_classes_are_supplied() -> None:
    """Experiment D needs class-balanced bin means; C must be untouched by that.

    The frozen C path is `classes=None`, and it has to stay bit-identical, or
    every published C number silently moves.
    """

    from interp.curvature import bin_means, direction_curvature

    rng = np.random.default_rng(3)
    values = rng.normal(size=(4000, 6))
    assignment = rng.integers(0, CURVATURE_SPEC.n_bins, size=4000)
    mu_a, counts_a = bin_means(values, assignment, CURVATURE_SPEC)
    mu_b, counts_b = bin_means(values, assignment, CURVATURE_SPEC, classes=None)
    assert np.array_equal(np.nan_to_num(mu_a), np.nan_to_num(mu_b))
    assert np.array_equal(counts_a, counts_b)

    sequence = np.arange(4000)
    direction = rng.normal(size=6)
    record_a = direction_curvature(values, direction, sequence, spec=CURVATURE_SPEC)
    record_b = direction_curvature(
        values, direction, sequence, spec=CURVATURE_SPEC, classes=None
    )
    assert set(record_a) == set(record_b)
    for key in record_a:
        assert np.allclose(
            np.nan_to_num(np.asarray(record_a[key], dtype=np.float64)),
            np.nan_to_num(np.asarray(record_b[key], dtype=np.float64)),
        ), key


def test_class_balanced_bin_means_ignore_class_proportion() -> None:
    """The whole point: a bin's mean must not move when its mixture moves.

    The refusal coordinate separates harmful from harmless by construction, so an
    unbalanced bin mean would track the class mixture rather than any geometry.
    """

    from interp.curvature import bin_means

    spec = CURVATURE_SPEC
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    n = spec.min_bin_rows

    def build(n_a: int, n_b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.concatenate([np.tile(a, (n_a, 1)), np.tile(b, (n_b, 1))])
        classes = np.array([0] * n_a + [1] * n_b)
        return values, np.zeros(n_a + n_b, dtype=np.int64), classes

    # bin 0 is 90% class A in one case and 10% class A in the other
    values_1, assignment_1, classes_1 = build(9 * n, n)
    values_2, assignment_2, classes_2 = build(n, 9 * n)
    mu_1, _ = bin_means(values_1, assignment_1, spec, classes=classes_1)
    mu_2, _ = bin_means(values_2, assignment_2, spec, classes=classes_2)
    assert np.allclose(mu_1[0], mu_2[0])
    assert np.allclose(mu_1[0], 0.5 * (a + b))
    # the unbalanced mean would have moved a long way
    plain_1, _ = bin_means(values_1, assignment_1, spec)
    plain_2, _ = bin_means(values_2, assignment_2, spec)
    assert not np.allclose(plain_1[0], plain_2[0])


def test_a_bin_missing_a_class_is_unusable_rather_than_half_balanced() -> None:
    """A one-class bin cannot be class-balanced, so it must not be reported.

    Silently falling back to the plain mean there would reintroduce exactly the
    composition effect the balancing exists to remove.
    """

    from interp.curvature import bin_means

    spec = CURVATURE_SPEC
    n = spec.min_bin_rows
    # both classes exist in the data, but bin 0 holds only class 0
    values = np.concatenate([np.tile(np.array([1.0, 0.0]), (n, 1)), np.zeros((n, 2))])
    assignment = np.array([0] * n + [1] * n, dtype=np.int64)
    classes = np.array([0] * n + [1] * n, dtype=np.int64)
    mu, counts = bin_means(values, assignment, spec, classes=classes)
    assert np.isnan(mu[0]).all()
    assert np.isnan(mu[1]).all()
    assert counts[0] == n


def test_class_balancing_needs_min_rows_from_every_class(monkeypatch) -> None:
    from interp.curvature import bin_means

    spec = CURVATURE_SPEC
    n = spec.min_bin_rows
    values = np.concatenate([np.zeros((n, 2)), np.ones((3, 2))])
    assignment = np.zeros(n + 3, dtype=np.int64)
    classes = np.array([0] * n + [1] * 3)
    mu, _ = bin_means(values, assignment, spec, classes=classes)
    assert np.isnan(mu[0]).all()


def test_the_direction_bootstrap_clusters_and_keeps_pairs_together() -> None:
    """Controls resample the same unit the frozen shortfall interval does.

    A statistic and its contrast must be resampled jointly, or the contrast gets
    an interval built from directions that were never compared with each other.
    """

    from interp.curvature import bootstrap_direction_mean

    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    result = bootstrap_direction_mean(values, spec=CURVATURE_SPEC)
    assert result["mean"] == pytest.approx(4.5)
    assert result["cluster"] == "direction"
    assert result["n_directions"] == 8
    assert result["ci_lower"] < 4.5 < result["ci_upper"]
    # deterministic under the frozen seed
    assert result == bootstrap_direction_mean(values, spec=CURVATURE_SPEC)


def test_the_direction_bootstrap_ignores_directions_with_no_estimate() -> None:
    from interp.curvature import bootstrap_direction_mean

    values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    result = bootstrap_direction_mean(values, spec=CURVATURE_SPEC)
    assert result["mean"] == pytest.approx(3.0)
    assert result["n_directions"] == 3


def test_the_direction_bootstrap_refuses_a_single_direction() -> None:
    from interp.curvature import bootstrap_direction_mean

    with pytest.raises(ValueError, match="at least two directions"):
        bootstrap_direction_mean(np.array([1.0, np.nan]), spec=CURVATURE_SPEC)


def test_the_direction_bootstrap_reproduces_the_frozen_shortfall_interval() -> None:
    """The helper must be the same resampling the frozen interval already used.

    If it is not, every control interval in the report would be built on a
    different rule from the primary one it is placed beside.
    """

    from interp.curvature import (
        _direction_shortfall,
        bootstrap_direction_mean,
        bootstrap_shortfall,
    )

    rng = np.random.default_rng(4)
    records = [
        {
            "cos_consecutive_secants": rng.normal(0.7, 0.05, size=4).tolist(),
            "split_half_pair_ceiling": rng.normal(0.99, 0.002, size=4).tolist(),
        }
        for _ in range(32)
    ]
    frozen = bootstrap_shortfall(records, CURVATURE_SPEC)
    helper = bootstrap_direction_mean(
        np.array([_direction_shortfall(r) for r in records]), spec=CURVATURE_SPEC
    )
    for field in ("mean", "ci_lower", "ci_upper", "n_directions"):
        assert helper[field] == pytest.approx(frozen[field])


def _script_module():
    spec = importlib.util.spec_from_file_location(
        "curvature_diagnostic", REPO / "scripts" / "curvature_diagnostic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direction_selection_is_the_frozen_seeded_draw_by_default() -> None:
    module = _script_module()
    picked, provenance = module._select_directions(4096, None)
    assert picked.size == CURVATURE_SPEC.n_directions
    assert (np.diff(picked) > 0).all()
    assert provenance["mode"] == "frozen_seeded_draw"
    assert provenance["preregistered"] is True
    # deterministic
    assert np.array_equal(picked, module._select_directions(4096, None)[0])


def test_supplied_directions_are_marked_exploratory_not_preregistered(
    tmp_path: Path,
) -> None:
    """Reusing another experiment's directions is an operator choice.

    It is the only way to ask whether curvature relates to that experiment's
    per-direction outcomes, but it is a selection the frozen plan did not make,
    so the payload must say so rather than look like the preregistered draw.
    """

    source = tmp_path / "other_result.json"
    source.write_text(json.dumps({"direction_pool_indices": [7, 3, 11, 3]}))
    picked, provenance = module_picked = _script_module()._select_directions(
        4096, source
    )
    assert np.array_equal(picked, np.array([3, 7, 11]))
    assert provenance["mode"] == "supplied_indices"
    assert provenance["preregistered"] is False
    assert provenance["source"].endswith("other_result.json")
    assert "exploratory" in provenance["note"]
    del module_picked


def test_supplied_directions_must_actually_exist_in_the_pool(tmp_path: Path) -> None:
    source = tmp_path / "other_result.json"
    source.write_text(json.dumps({"direction_pool_indices": [7, 999_999]}))
    with pytest.raises(ValueError, match="outside the pool"):
        _script_module()._select_directions(4096, source)


def test_a_source_without_direction_indices_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "other_result.json"
    source.write_text(json.dumps({"something_else": 1}))
    with pytest.raises(ValueError, match="direction_pool_indices"):
        _script_module()._select_directions(4096, source)


def test_row_selection_is_deterministic_and_inside_the_artifact() -> None:
    spec = importlib.util.spec_from_file_location(
        "curvature_diagnostic", REPO / "scripts" / "curvature_diagnostic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    available = 1_024_128
    rows = module._select_rows(available)
    assert rows.size == CURVATURE_SPEC.n_rows
    assert rows.min() >= 0 and rows.max() < available
    assert np.unique(rows).size == rows.size
    assert (np.diff(rows) > 0).all()
    # deterministic
    assert np.array_equal(rows, module._select_rows(available))
    with pytest.raises(ValueError, match="fewer than the frozen"):
        module._select_rows(10)
