"""Is the concept trajectory linear? (post-stop experiment C)

The whole programme assumes concept strength is ``<h, v>`` for one fixed ``v``.
If natural activations instead lie along a *curved* trajectory, ``v`` is only a
local tangent: strong additive steering travels in a straight line away from the
natural states, and "orthogonal correction relative to a fixed ``v``" is not the
same thing as "correction toward the natural manifold".

This module measures that, and nothing else. It trains nothing and proposes no
intervention.

## The measurement

For one direction ``v`` and natural activations ``h``:

    c      = <h, v>
    bins   = the coordinate quantiles p10/p25/p50/p75/p90 (5 cuts -> 6 bins)
    mu_k   = E[h | c in bin k]
    d_k    = mu_{k+1} - mu_k                      (5 secants)
    d_par  = <d_k, v> v ,  d_perp = d_k - d_par

    r_k          = ||d_perp|| / ||d_k||
    cos(d_k, v)
    cos(d_k, d_{k+1})                             (4 consecutive pairs)

A conditional mean that is *affine* in the coordinate, ``E[h | c] = mu_0 + c w``,
gives ``d_k`` parallel to a single fixed ``w`` at every rung, so
``cos(d_k, d_{k+1}) = 1`` exactly. **Systematic rotation of ``d_k`` with
increasing concept strength is the curvature signal.** ``r_k`` and ``cos(d_k, v)``
describe how far the local trajectory tilts away from ``v`` itself, which is a
different (and weaker) statement: a large ``r_k`` alone only says other
coordinates co-vary with ``c``, not that the trajectory bends.

## Why the noise floor is not optional

Each ``mu_k`` is a finite-sample mean, so each ``d_k`` carries independent
estimation noise, and independent noise *always* pushes ``cos(d_k, d_{k+1})``
below 1. Reading a value below 1 as curvature without calibrating that bias is
the easiest way to manufacture a false positive here. Two calibrations are
therefore computed alongside every estimate:

* **shuffle control** -- coordinate labels permuted across rows, destroying any
  real relationship. Gives the null.
* **split-half reliability** -- ``d_k`` estimated independently on two disjoint
  halves of the sequences, then correlated with itself. Gives the *ceiling*:
  ``cos(d_k, d_{k+1})`` cannot exceed roughly this even under perfect linearity.

Curvature is claimed only when the consecutive-secant angle sits clearly below
the split-half ceiling. Random unit directions and a sequence-level bootstrap are
computed too.

Concept-independent by construction: frozen validation activations, training-only
pool directions, no DEV or held-out data, no LLM judge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

CUT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass(frozen=True)
class CurvatureSpec:
    """Frozen sampling and analysis plan for the curvature diagnostic."""

    version: str = "concept_curvature_v1"
    # Coordinate quantile cuts. Five cuts make six bins and five secants.
    cut_quantiles: tuple[float, ...] = CUT_QUANTILES
    # Directions and rows. The direction count matches natural_support_v1 so the
    # diagnostic speaks about the same population of directions the steering
    # experiments used; the seed differs because this is a different experiment.
    n_directions: int = 32
    direction_seed: int = 20260913
    n_rows: int = 262144
    row_seed: int = 20260914
    # Controls.
    n_random_directions: int = 32
    random_direction_seed: int = 20260915
    shuffle_seed: int = 20260916
    split_half_seed: int = 20260917
    split_half_repeats: int = 8
    bootstrap_seed: int = 20260918
    bootstrap_resamples: int = 1000
    confidence: float = 0.95
    # A bin must hold at least this many rows for its mean to be reported.
    min_bin_rows: int = 256

    def __post_init__(self) -> None:
        if len(self.cut_quantiles) < 2:
            raise ValueError("need at least two coordinate cuts")
        if not all(0.0 < q < 1.0 for q in self.cut_quantiles):
            raise ValueError("cut quantiles must lie strictly inside (0, 1)")
        if list(self.cut_quantiles) != sorted(self.cut_quantiles):
            raise ValueError("cut quantiles must increase")
        if self.n_directions < 2 or self.n_rows < 2:
            raise ValueError("need at least two directions and two rows")
        if self.min_bin_rows < 2:
            raise ValueError("a bin mean needs at least two rows")

    @property
    def n_bins(self) -> int:
        return len(self.cut_quantiles) + 1

    @property
    def n_secants(self) -> int:
        return self.n_bins - 1


CURVATURE_SPEC = CurvatureSpec()


def spec_payload(spec: CurvatureSpec = CURVATURE_SPEC) -> dict:
    return asdict(spec)


# --------------------------------------------------------------------------
# binning and secants
# --------------------------------------------------------------------------


def bin_indices(coordinate: np.ndarray, spec: CurvatureSpec = CURVATURE_SPEC) -> np.ndarray:
    """Assign each row to a coordinate-quantile bin, ``0 .. n_bins - 1``."""

    values = np.asarray(coordinate, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("coordinate must be a nonempty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("coordinates must be finite")
    edges = np.quantile(values, spec.cut_quantiles)
    return np.searchsorted(edges, values, side="right").astype(np.int64)


def bin_means(
    activations: np.ndarray,
    assignment: np.ndarray,
    spec: CurvatureSpec = CURVATURE_SPEC,
    classes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mu, counts)`` with ``mu[k]`` the mean activation of bin ``k``.

    Bins below ``min_bin_rows`` get a NaN mean rather than a noisy one, so an
    undersampled tail cannot quietly contribute a spurious rotation.

    ``classes`` is for Experiment D, where the coordinate is a direction built to
    separate two prompt classes. Sorting a mixed pool by such a coordinate largely
    sorts by class, so an ordinary bin mean would track the class mixture rather
    than any geometry. When it is supplied, a bin's mean is the *unweighted
    average of its per-class means*, which is invariant to the proportions; a bin
    that does not hold ``min_bin_rows`` of every class cannot be balanced at all
    and is returned as NaN rather than quietly falling back to the plain mean.

    ``classes=None`` is the frozen Experiment C path and is bit-identical to the
    version that had no such parameter.
    """

    values = np.asarray(activations, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("activations must be [rows, d]")
    groups = np.asarray(assignment, dtype=np.int64)
    if groups.shape != (values.shape[0],):
        raise ValueError("assignment must give one bin per activation row")
    labels = None
    if classes is not None:
        labels = np.asarray(classes, dtype=np.int64)
        if labels.shape != (values.shape[0],):
            raise ValueError("classes must give one label per activation row")
        present = np.unique(labels)
    mu = np.full((spec.n_bins, values.shape[1]), np.nan, dtype=np.float64)
    counts = np.zeros(spec.n_bins, dtype=np.int64)
    for k in range(spec.n_bins):
        rows = groups == k
        counts[k] = int(rows.sum())
        if counts[k] < spec.min_bin_rows:
            continue
        if labels is None:
            mu[k] = values[rows].mean(axis=0)
            continue
        per_class = []
        for label in present:
            member = rows & (labels == label)
            if int(member.sum()) < spec.min_bin_rows:
                per_class = []
                break
            per_class.append(values[member].mean(axis=0))
        if per_class:
            mu[k] = np.mean(per_class, axis=0)
    return mu, counts


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return vector / norm


def secant_geometry(
    mu: np.ndarray, direction: np.ndarray
) -> dict[str, np.ndarray]:
    """Orthogonal drift and local angles of the secants ``d_k = mu_{k+1} - mu_k``.

    Bins whose mean is NaN propagate NaN into the secants that touch them, and
    every summary here is NaN-aware, so an undersampled bin removes its own rungs
    instead of contaminating the rest.
    """

    means = np.asarray(mu, dtype=np.float64)
    v = _unit(np.asarray(direction, dtype=np.float64).reshape(-1))
    if means.ndim != 2 or means.shape[1] != v.size:
        raise ValueError("bin means and direction must share a width")
    secants = means[1:] - means[:-1]
    parallel = secants @ v
    orthogonal = secants - parallel[:, None] * v[None, :]
    norms = np.linalg.norm(secants, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        drift = np.linalg.norm(orthogonal, axis=1) / norms
        cos_v = parallel / norms
    consecutive = np.full(max(secants.shape[0] - 1, 0), np.nan, dtype=np.float64)
    for k in range(secants.shape[0] - 1):
        left, right = secants[k], secants[k + 1]
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        if np.isfinite(denominator) and denominator > 0.0:
            consecutive[k] = float(left @ right / denominator)
    return {
        "secant": secants,
        "orthogonal_drift": drift,
        "cos_secant_direction": cos_v,
        "cos_consecutive_secants": consecutive,
        "secant_norm": norms,
        "parallel_component": parallel,
    }


# --------------------------------------------------------------------------
# one direction, with its controls
# --------------------------------------------------------------------------


def direction_curvature(
    activations: np.ndarray,
    direction: np.ndarray,
    sequence: np.ndarray,
    *,
    spec: CurvatureSpec = CURVATURE_SPEC,
    seed_offset: int = 0,
    classes: np.ndarray | None = None,
) -> dict[str, object]:
    """Full curvature record for one direction: estimate, null, and ceiling.

    ``sequence`` gives each row's originating sequence. It is the resampling and
    splitting unit, because activations from one document are not independent.
    """

    values = np.asarray(activations, dtype=np.float64)
    v = _unit(np.asarray(direction, dtype=np.float64).reshape(-1))
    sequences = np.asarray(sequence, dtype=np.int64)
    if sequences.shape != (values.shape[0],):
        raise ValueError("sequence must give one identifier per activation row")

    coordinate = values @ v
    assignment = bin_indices(coordinate, spec)
    mu, counts = bin_means(values, assignment, spec, classes=classes)
    observed = secant_geometry(mu, v)

    # Null: the same rows, coordinate labels permuted, so any structure is noise.
    shuffle_rng = np.random.default_rng(spec.shuffle_seed + seed_offset)
    shuffled = secant_geometry(
        bin_means(values, shuffle_rng.permutation(assignment), spec, classes=classes)[0], v
    )

    # Ceiling: the same estimate on two disjoint halves of the SEQUENCES, averaged
    # over several independent splits. One split is itself a noisy estimate of the
    # noise floor, and a noisy ceiling makes the curvature comparison noisy in
    # exactly the direction that would fake a result.
    unique = np.unique(sequences)
    split_rng = np.random.default_rng(spec.split_half_seed + seed_offset)
    per_split = np.full((spec.split_half_repeats, spec.n_secants), np.nan, dtype=np.float64)
    for repeat in range(spec.split_half_repeats):
        left_ids = set(
            split_rng.choice(
                unique, size=max(len(unique) // 2, 1), replace=False
            ).tolist()
        )
        left_rows = np.fromiter(
            (sid in left_ids for sid in sequences), dtype=bool, count=sequences.size
        )
        halves = []
        for rows in (left_rows, ~left_rows):
            if rows.sum() < spec.min_bin_rows * spec.n_bins:
                halves.append(None)
                continue
            half_assignment = bin_indices(coordinate[rows], spec)
            half_classes = None if classes is None else np.asarray(classes)[rows]
            halves.append(
                secant_geometry(
                    bin_means(values[rows], half_assignment, spec, classes=half_classes)[0],
                    v,
                )
            )
        if any(half is None for half in halves):
            continue
        for k in range(spec.n_secants):
            left, right = halves[0]["secant"][k], halves[1]["secant"][k]
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            if np.isfinite(denominator) and denominator > 0.0:
                per_split[repeat, k] = float(left @ right / denominator)
    with np.errstate(invalid="ignore"):
        half_reliability = np.nanmean(per_split, axis=0)
    # Spearman-Brown: a split-half correlation describes estimates built from HALF
    # the sequences, which are noisier than the full-data secants the observed
    # angle is computed from. Without this step the ceiling is systematically too
    # low and every direction looks mildly curved.
    with np.errstate(invalid="ignore"):
        reliability = np.where(
            half_reliability > 0.0,
            2.0 * half_reliability / (1.0 + half_reliability),
            half_reliability,
        )

    # cos(d_k, d_{k+1}) is attenuated by the independent noise in BOTH secants,
    # so its ceiling is the geometric mean of the two reliabilities -- the
    # standard correction-for-attenuation form. Comparing the pair angle against
    # a single secant's reliability would compare different quantities.
    pair_ceiling = np.full(max(spec.n_secants - 1, 0), np.nan, dtype=np.float64)
    for k in range(pair_ceiling.size):
        left, right = reliability[k], reliability[k + 1]
        if np.isfinite(left) and np.isfinite(right) and left > 0.0 and right > 0.0:
            pair_ceiling[k] = float(np.sqrt(left * right))

    return {
        "bin_counts": counts.tolist(),
        "bin_coordinate_means": [
            float(coordinate[assignment == k].mean()) if counts[k] else float("nan")
            for k in range(spec.n_bins)
        ],
        "orthogonal_drift": observed["orthogonal_drift"].tolist(),
        "cos_secant_direction": observed["cos_secant_direction"].tolist(),
        "cos_consecutive_secants": observed["cos_consecutive_secants"].tolist(),
        "secant_norm": observed["secant_norm"].tolist(),
        "shuffled_orthogonal_drift": shuffled["orthogonal_drift"].tolist(),
        "shuffled_cos_consecutive_secants": (
            shuffled["cos_consecutive_secants"].tolist()
        ),
        "split_half_reliability_raw": half_reliability.tolist(),
        "split_half_reliability": reliability.tolist(),
        "split_half_pair_ceiling": pair_ceiling.tolist(),
        "n_rows": int(values.shape[0]),
        "n_sequences": int(len(unique)),
    }


# --------------------------------------------------------------------------
# pooling across directions, and the reporting rule
# --------------------------------------------------------------------------


def _nanmean(values: list[list[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.nanmean(array, axis=0)


def pooled_curvature(
    records: list[dict[str, object]], spec: CurvatureSpec = CURVATURE_SPEC
) -> dict[str, object]:
    """Average each rung over directions, keeping the controls beside the estimate."""

    if not records:
        raise ValueError("need at least one direction record")
    pooled = {
        "n_directions": len(records),
        "orthogonal_drift_by_rung": _nanmean(
            [r["orthogonal_drift"] for r in records]
        ).tolist(),
        "cos_secant_direction_by_rung": _nanmean(
            [r["cos_secant_direction"] for r in records]
        ).tolist(),
        "cos_consecutive_secants_by_rung": _nanmean(
            [r["cos_consecutive_secants"] for r in records]
        ).tolist(),
        "shuffled_orthogonal_drift_by_rung": _nanmean(
            [r["shuffled_orthogonal_drift"] for r in records]
        ).tolist(),
        "shuffled_cos_consecutive_secants_by_rung": _nanmean(
            [r["shuffled_cos_consecutive_secants"] for r in records]
        ).tolist(),
        "split_half_reliability_by_rung": _nanmean(
            [r["split_half_reliability"] for r in records]
        ).tolist(),
        "split_half_pair_ceiling_by_pair": _nanmean(
            [r["split_half_pair_ceiling"] for r in records]
        ).tolist(),
        "n_bins": spec.n_bins,
        "n_secants": spec.n_secants,
    }
    return pooled


def _direction_shortfall(record: dict[str, object]) -> float:
    """One direction's mean gap between its ceiling and its observed angle."""

    observed = np.asarray(record["cos_consecutive_secants"], dtype=np.float64)
    ceiling = np.asarray(record["split_half_pair_ceiling"], dtype=np.float64)
    usable = np.isfinite(observed) & np.isfinite(ceiling)
    if not usable.any():
        return float("nan")
    return float(np.mean(ceiling[usable] - observed[usable]))


def bootstrap_direction_mean(
    per_direction: np.ndarray, *, spec: CurvatureSpec = CURVATURE_SPEC
) -> dict[str, float | int | str]:
    """Direction-clustered interval on any per-direction quantity.

    Same cluster, seed, resample count and confidence as
    :func:`bootstrap_shortfall`, so a control interval placed beside the primary
    one was built by the same rule. Pass a per-direction *difference* to get a
    paired contrast: the directions are then resampled jointly by construction.
    """

    usable = np.asarray(per_direction, dtype=np.float64)
    usable = usable[np.isfinite(usable)]
    if usable.size < 2:
        raise ValueError("need at least two directions with a usable estimate")
    rng = np.random.default_rng(spec.bootstrap_seed)
    draws = usable[rng.integers(0, usable.size, size=(spec.bootstrap_resamples, usable.size))]
    means = draws.mean(axis=1)
    tail = (1.0 - spec.confidence) / 2.0
    return {
        "mean": float(usable.mean()),
        "ci_lower": float(np.quantile(means, tail)),
        "ci_upper": float(np.quantile(means, 1.0 - tail)),
        "confidence": float(spec.confidence),
        "n_directions": int(usable.size),
        "n_resamples": int(spec.bootstrap_resamples),
        "fraction_directions_positive": float((usable > 0.0).mean()),
        "cluster": "direction",
    }


def bootstrap_shortfall(
    records: list[dict[str, object]], spec: CurvatureSpec = CURVATURE_SPEC
) -> dict[str, float | int]:
    """Resample DIRECTIONS to put an interval on the pooled curvature shortfall.

    Directions are the cluster: rows within a direction share its geometry, so
    resampling rows would understate the uncertainty that matters. A curvature
    claim resting on a handful of unusual directions fails here.
    """

    per_direction = np.array([_direction_shortfall(r) for r in records], dtype=np.float64)
    usable = per_direction[np.isfinite(per_direction)]
    if usable.size < 2:
        raise ValueError("need at least two directions with a usable shortfall")
    rng = np.random.default_rng(spec.bootstrap_seed)
    draws = usable[rng.integers(0, usable.size, size=(spec.bootstrap_resamples, usable.size))]
    means = draws.mean(axis=1)
    tail = (1.0 - spec.confidence) / 2.0
    return {
        "mean": float(usable.mean()),
        "ci_lower": float(np.quantile(means, tail)),
        "ci_upper": float(np.quantile(means, 1.0 - tail)),
        "confidence": float(spec.confidence),
        "n_directions": int(usable.size),
        "n_resamples": int(spec.bootstrap_resamples),
        "fraction_directions_positive": float((usable > 0.0).mean()),
        "cluster": "direction",
    }


def curvature_verdict(
    pooled: dict[str, object],
    random_pooled: dict[str, object] | None = None,
    *,
    interval: dict[str, float | int] | None = None,
    margin: float = 0.02,
) -> dict[str, object]:
    """The frozen reporting rule of `docs/POST_STOP_PROTOCOL_2026-08-19.md` §C.4.

    Curvature is reported only when the consecutive-secant angle falls clearly
    below what the split-half reliability says the estimate could have achieved.
    ``margin`` is how far below "clearly" means. It was fixed at 0.02 by
    calibration on SYNTHETIC data, before any real activation was touched: a
    generator with an exactly linear conditional mean produces a shortfall of
    ~0.004 at this sample size, while a trajectory rotating by ~15 degrees across
    the observed coordinate range produces ~0.025. It is not tuned afterwards.

    When ``interval`` (from :func:`bootstrap_shortfall`) is supplied, curvature
    additionally requires its lower bound to clear the margin, so the verdict does
    not rest on a point estimate.

    This function issues a description, never a licence to explain away a
    negative result: §C.4 forbids using nonlinear geometry as an explanation when
    no curvature is found.
    """

    observed = np.asarray(pooled["cos_consecutive_secants_by_rung"], dtype=np.float64)
    ceiling = np.asarray(pooled["split_half_pair_ceiling_by_pair"], dtype=np.float64)
    usable = np.isfinite(observed) & np.isfinite(ceiling)
    if not usable.any():
        return {
            "gate": "C",
            "verdict": None,
            "reason": "no rung had both an estimate and a reliability ceiling",
        }
    shortfall = ceiling[usable] - observed[usable]
    curved = bool(np.mean(shortfall) > margin)
    if interval is not None:
        curved = curved and float(interval["ci_lower"]) > margin
    verdict: dict[str, object] = {
        "gate": "C",
        "margin": float(margin),
        "mean_cos_consecutive_secants": float(np.mean(observed[usable])),
        "mean_split_half_pair_ceiling": float(np.mean(ceiling[usable])),
        "mean_shortfall_below_ceiling": float(np.mean(shortfall)),
        "rungs_used": int(usable.sum()),
        "shortfall_interval": interval,
        "verdict": "CURVATURE_DETECTED" if curved else "NO_STRONG_CURVATURE",
        "interpretation": (
            "the local direction of the natural concept trajectory rotates with "
            "concept strength by more than estimation noise explains; a fixed v is "
            "a local tangent, not a global axis"
            if curved
            else "no strong evidence that the concept trajectory bends relative to "
            "the chosen direction within the natural range; nonlinear geometry "
            "must NOT be used to explain the programme's negative results"
        ),
    }
    if random_pooled is not None:
        verdict["random_direction_reference"] = {
            "mean_cos_consecutive_secants": float(
                np.nanmean(
                    np.asarray(
                        random_pooled["cos_consecutive_secants_by_rung"], dtype=np.float64
                    )
                )
            ),
            "mean_orthogonal_drift": float(
                np.nanmean(
                    np.asarray(
                        random_pooled["orthogonal_drift_by_rung"], dtype=np.float64
                    )
                )
            ),
            "note": (
                "random unit directions are not concept directions; this is a "
                "reference for what the statistics look like on an arbitrary axis, "
                "not a null hypothesis test"
            ),
        }
    return verdict
