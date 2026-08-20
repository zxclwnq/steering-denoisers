"""C6: how much of the Experiment C curvature survives covariance control?

Experiment D established that ``cos(d_k, v)`` is dominated by how much variance a
direction carries, which makes an ordinary random unit axis a poor reference. The
same confound reaches into C, and there is an exact reason why. For a Gaussian,

    E[h | v'h = c] = mu + (Sigma v / v'Sigma v) (c - v'mu)

so the linear direction of the conditional trajectory is ``Sigma v``, not ``v``.
A residual stream is strongly anisotropic, so a conditional mean that moves
"away from v" is the *expected* behaviour of a purely linear, purely Gaussian
population --- not evidence of anything bending.

This module supplies the controls that separate the two: the covariance-predicted
linear direction, a held-out linear-versus-quadratic conditional fit, a
covariance-matched Gaussian surrogate, and covariance-matched random directions.
The statistics themselves stay `interp.curvature`; nothing here invents a new
measure, and nothing here rewrites a frozen C artifact.

Frozen in `docs/EXPERIMENT_C6_PROTOCOL.md` before any C6 number existed.
`preregistered: false` --- this is a post-hoc control of an existing result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CovarianceControlSpec:
    """Frozen C6 plan. Seeds and the matching rule are fixed before any result."""

    version: str = "curvature_covariance_controls_c6_v1"
    split_seed: int = 20260920
    candidate_seed: int = 20260921
    surrogate_seed: int = 20260922
    n_candidates: int = 20000
    n_principal_components: int = 8
    # Degree 2 only: the question is whether the conditional mean is nonlinear at
    # all, not how nonlinear. A flexible model would answer a different question.
    polynomial_degree: int = 2
    # Below this the residual conditional means are too small for an angle to
    # mean anything, and a cosine between two near-zero vectors is noise.
    min_residual_norm_ratio: float = 0.05

    def payload(self) -> dict:
        return asdict(self)


COVARIANCE_CONTROL_SPEC = CovarianceControlSpec()


# --------------------------------------------------------------------------
# C6.1 the covariance-predicted linear direction
# --------------------------------------------------------------------------


def covariance_linear_direction(covariance: np.ndarray, v: np.ndarray) -> dict:
    """``Sigma v``: the direction a purely linear Gaussian population would move in.

    ``b_v`` is the conditional-mean slope per unit of the coordinate; ``t_v`` is
    its unit vector, which is what an angle can be taken against.
    """

    v = np.asarray(v, dtype=np.float64).reshape(-1)
    v = v / np.linalg.norm(v)
    sigma_v = np.asarray(covariance, dtype=np.float64) @ v
    projected = float(v @ sigma_v)
    if not np.isfinite(projected) or projected <= 0.0:
        raise ValueError("projected variance must be positive and finite")
    norm = float(np.linalg.norm(sigma_v))
    return {
        "projected_variance": projected,
        "slope": sigma_v / projected,
        "unit": sigma_v / norm,
        "sigma_v_norm": norm,
        # How far the covariance-predicted direction sits from v itself. This is
        # the quantity that makes cos(d_k, v) uninterpretable on its own.
        "cos_v_sigma_v": projected / norm,
    }


# --------------------------------------------------------------------------
# C6.2 held-out linear vs quadratic conditional model
# --------------------------------------------------------------------------


def split_sequences(sequence: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Halve the data by SEQUENCE, never by row.

    Rows from one document are not independent, so a row-level split would leak
    the fitted trajectory into its own evaluation.
    """

    sequences = np.asarray(sequence, dtype=np.int64)
    unique = np.unique(sequences)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    left = set(shuffled[: len(shuffled) // 2].tolist())
    mask = np.fromiter(
        (sid in left for sid in sequences), dtype=bool, count=sequences.size
    )
    return mask, ~mask


def _design(coordinate: np.ndarray, degree: int, centre: float, scale: float) -> np.ndarray:
    z = (np.asarray(coordinate, dtype=np.float64) - centre) / scale
    return np.stack([z**power for power in range(degree + 1)], axis=1)


def conditional_fit_comparison(
    activations: np.ndarray,
    v: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    spec: CovarianceControlSpec = COVARIANCE_CONTROL_SPEC,
) -> dict:
    """Does a quadratic in ``c`` predict held-out activations better than a line?

    This is the direct test of "is ``E[h|c]`` nonlinear", and unlike an angle it
    cannot be confounded by covariance anisotropy: both models see exactly the
    same coordinate and the same output space, and both are scored on rows
    neither of them was fitted on.
    """

    values = np.asarray(activations, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    v = v / np.linalg.norm(v)
    coordinate = values @ v
    centre = float(coordinate[train].mean())
    scale = float(coordinate[train].std())
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("the coordinate has no spread on the training half")

    out: dict[str, object] = {
        "coordinate_centre": centre,
        "coordinate_scale": scale,
        "n_train_rows": int(train.sum()),
        "n_test_rows": int(test.sum()),
    }
    errors: dict[int, np.ndarray] = {}
    for degree in (1, spec.polynomial_degree):
        design_train = _design(coordinate[train], degree, centre, scale)
        coefficients, *_ = np.linalg.lstsq(design_train, values[train], rcond=None)
        residual = values[test] - _design(coordinate[test], degree, centre, scale) @ coefficients
        errors[degree] = np.square(residual).mean(axis=1)
        out[f"mse_degree{degree}"] = float(errors[degree].mean())
    out["delta_mse_linear_minus_quadratic"] = float(
        errors[1].mean() - errors[spec.polynomial_degree].mean()
    )
    out["relative_improvement"] = float(
        (errors[1].mean() - errors[spec.polynomial_degree].mean()) / errors[1].mean()
    )
    out["per_row_delta"] = errors[1] - errors[spec.polynomial_degree]
    return out


def _ratio(residual_norm: np.ndarray, secant_norm: np.ndarray) -> list[float | None]:
    """Residual size relative to the local secant it would have to explain.

    A residual mean is only interesting next to how far the conditional mean
    moves between bins; in absolute nats-free units it says nothing.
    """

    out: list[float | None] = []
    for index, residual in enumerate(residual_norm):
        reference = secant_norm[min(index, secant_norm.size - 1)]
        if not (np.isfinite(residual) and np.isfinite(reference)) or reference == 0.0:
            out.append(None)
        else:
            out.append(float(residual / reference))
    return out


def residual_conditional_means(
    activations: np.ndarray,
    v: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    assignment: np.ndarray,
    n_bins: int,
    min_bin_rows: int,
) -> dict:
    """Conditional means of what the held-out linear model failed to predict.

    If the conditional geometry is entirely linear, these are zero up to noise
    and carry no systematic profile.
    """

    values = np.asarray(activations, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    v = v / np.linalg.norm(v)
    coordinate = values @ v
    centre = float(coordinate[train].mean())
    scale = float(coordinate[train].std())
    design_train = _design(coordinate[train], 1, centre, scale)
    coefficients, *_ = np.linalg.lstsq(design_train, values[train], rcond=None)
    residual = values[test] - _design(coordinate[test], 1, centre, scale) @ coefficients

    bins = np.asarray(assignment, dtype=np.int64)[test]
    rho = np.full((n_bins, values.shape[1]), np.nan, dtype=np.float64)
    raw = np.full((n_bins, values.shape[1]), np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for k in range(n_bins):
        rows = bins == k
        counts[k] = int(rows.sum())
        if counts[k] >= min_bin_rows:
            rho[k] = residual[rows].mean(axis=0)
            raw[k] = values[test][rows].mean(axis=0)
    with np.errstate(invalid="ignore"):
        residual_norm = np.linalg.norm(rho, axis=1)
        secant_norm = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    return {
        "bin_counts": counts.tolist(),
        "residual_mean_norm_by_bin": [
            None if not np.isfinite(x) else float(x) for x in residual_norm
        ],
        "raw_secant_norm_by_rung": [
            None if not np.isfinite(x) else float(x) for x in secant_norm
        ],
        "residual_to_secant_ratio": _ratio(residual_norm, secant_norm),
        "residual_means": rho,
    }


# --------------------------------------------------------------------------
# C6.3 covariance-matched Gaussian surrogate
# --------------------------------------------------------------------------


def gaussian_surrogate(
    mean: np.ndarray, covariance: np.ndarray, n_rows: int, seed: int
) -> tuple[np.ndarray, dict]:
    """Draw ``N(mu_hat, Sigma_hat)``: same second-order structure, nothing higher.

    Its population conditional mean is linear by construction, so whatever
    curvature the pipeline reports on it is produced by finite-sample estimation
    and the pipeline itself, not by the data bending.

    Sampling goes through the eigendecomposition rather than a Cholesky factor
    because an empirical covariance can be singular, and the reproduction
    accuracy of the spectrum is reported rather than assumed.
    """

    covariance = np.asarray(covariance, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    clipped = np.clip(eigenvalues, 0.0, None)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=(n_rows, covariance.shape[0]))
    synthetic = (noise * np.sqrt(clipped)) @ eigenvectors.T + np.asarray(mean, dtype=np.float64)
    achieved = np.cov(synthetic, rowvar=False)
    denominator = float(np.linalg.norm(covariance))
    return synthetic, {
        "method": "eigendecomposition_of_the_empirical_covariance",
        "n_rows": int(n_rows),
        "n_negative_eigenvalues_clipped": int((eigenvalues < 0.0).sum()),
        "smallest_eigenvalue": float(eigenvalues.min()),
        "largest_eigenvalue": float(eigenvalues.max()),
        "relative_covariance_error": float(
            np.linalg.norm(achieved - covariance) / denominator
        )
        if denominator
        else None,
        "isotropic": False,
    }


# --------------------------------------------------------------------------
# C6.4 covariance-matched random directions
# --------------------------------------------------------------------------


def covariance_coordinates(covariance: np.ndarray, directions: np.ndarray) -> dict:
    """The two matching variables: projected variance, and place in the spectrum."""

    directions = np.asarray(directions, dtype=np.float64)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    sigma_d = directions @ np.asarray(covariance, dtype=np.float64)
    projected = np.einsum("ij,ij->i", directions, sigma_d)
    norms = np.linalg.norm(sigma_d, axis=1)
    return {
        "projected_variance": projected,
        "cos_v_sigma_v": projected / norms,
    }


def match_directions(
    covariance: np.ndarray,
    concept: np.ndarray,
    candidates: np.ndarray,
) -> dict:
    """Nearest candidate per concept direction in standardized (log s^2, a) space.

    Matching is without replacement, so the null has as many distinct directions
    as the concept set and cannot be one lucky candidate reused.

    ``log s^2`` because projected variance spans orders of magnitude; ``a``
    because two directions can share a variance and still sit very differently
    relative to the covariance spectrum.
    """

    concept_stats = covariance_coordinates(covariance, concept)
    candidate_stats = covariance_coordinates(covariance, candidates)
    pool = np.stack(
        [np.log(candidate_stats["projected_variance"]), candidate_stats["cos_v_sigma_v"]],
        axis=1,
    )
    centre = pool.mean(axis=0)
    scale = pool.std(axis=0)
    scale[scale == 0.0] = 1.0
    pool_z = (pool - centre) / scale
    target = np.stack(
        [np.log(concept_stats["projected_variance"]), concept_stats["cos_v_sigma_v"]],
        axis=1,
    )
    target_z = (target - centre) / scale

    picked: list[int] = []
    taken = np.zeros(pool_z.shape[0], dtype=bool)
    for row in target_z:
        distance = np.linalg.norm(pool_z - row, axis=1)
        distance[taken] = np.inf
        index = int(np.argmin(distance))
        taken[index] = True
        picked.append(index)
    picked_array = np.asarray(picked, dtype=np.int64)

    matched = {
        key: value[picked_array] for key, value in candidate_stats.items()
    }
    unmatched_reference = {
        key: value[: len(picked)] for key, value in candidate_stats.items()
    }
    return {
        "indices": picked_array,
        "directions": np.asarray(candidates, dtype=np.float64)[picked_array],
        "rule": (
            "nearest neighbour without replacement in standardized "
            "(log projected variance, cos(v, Sigma v)) space"
        ),
        "n_candidates": int(candidates.shape[0]),
        "balance": _balance(concept_stats, matched, unmatched_reference),
    }


def _balance(concept: dict, matched: dict, unmatched: dict) -> dict:
    """Did the matching actually work? Reported before and after, per variable."""

    out = {}
    for key in ("projected_variance", "cos_v_sigma_v"):
        target = concept[key]
        transform = np.log if key == "projected_variance" else (lambda x: x)
        after = np.abs(transform(target) - transform(matched[key]))
        before = np.abs(transform(target) - transform(unmatched[key]))
        out[key] = {
            "concept_mean": float(np.mean(target)),
            "matched_mean": float(np.mean(matched[key])),
            "unmatched_mean": float(np.mean(unmatched[key])),
            "mean_abs_difference_after": float(np.mean(after)),
            "max_abs_difference_after": float(np.max(after)),
            "mean_abs_difference_before": float(np.mean(before)),
            "scale": "log" if key == "projected_variance" else "linear",
        }
    return out


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------


def covariance_verdict(
    real_minus_gaussian: dict,
    real_minus_matched_random: dict,
    quadratic_improvement: dict,
    residual_structure_present: bool,
) -> dict:
    """One of the three preregistered readings, by rule.

    Order follows the protocol: the strongest claim requires every control to
    agree, the middle one keeps the nonlinearity but drops the concept-specific
    part, and the weakest is issued when the controls remove the effect.
    """

    def positive(block: dict) -> bool:
        return bool(block.get("usable", True)) and float(block.get("ci_lower", 0.0)) > 0.0

    beyond_gaussian = positive(real_minus_gaussian)
    beyond_matched = positive(real_minus_matched_random)
    nonlinear = positive(quadratic_improvement)

    if beyond_gaussian and beyond_matched and nonlinear and residual_structure_present:
        outcome = "CURVATURE_BEYOND_COVARIANCE"
        why = (
            "curvature exceeds both a covariance-matched Gaussian and "
            "covariance-matched random directions, a held-out quadratic beats the "
            "linear conditional model, and the residual conditional means keep "
            "systematic structure. The conditional mean varies nonlinearly with "
            "the coordinate beyond finite-sample noise and second-order geometry"
        )
    elif nonlinear:
        outcome = "CURVATURE_NOT_CONCEPT_SPECIFIC_AFTER_MATCHING"
        why = (
            "the conditional mean is still nonlinear on held-out rows, but the "
            "excess curvature of concept directions over ordinary random axes is "
            "substantially explained by covariance anisotropy: against a "
            "covariance-matched null the concept directions do not stand out"
        )
    else:
        outcome = "CURVATURE_EXPLAINED_BY_COVARIANCE"
        why = (
            "under the linear and covariance controls the effect does not "
            "survive: the original curvature statistic is largely explained by "
            "anisotropic covariance geometry and finite-sample conditioning"
        )
    return {
        "gate": "C6",
        "outcome": outcome,
        "why": why,
        "curvature_exceeds_gaussian_surrogate": beyond_gaussian,
        "curvature_exceeds_covariance_matched_random": beyond_matched,
        "held_out_quadratic_beats_linear": nonlinear,
        "residual_conditional_structure_present": bool(residual_structure_present),
        "prohibited_claims": [
            "curvature causes steering failure",
            "v is or is not causal",
            "v is or is not a valid intervention direction",
            "SAE features are correlational",
            "tangent alignment predicts steerability",
        ],
        "cos_dk_v_note": (
            "cos(d_k, v) is confounded by activation covariance and is reported "
            "as a descriptive statistic only"
        ),
    }
