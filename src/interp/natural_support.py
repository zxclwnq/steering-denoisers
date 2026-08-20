"""Natural-support controllability diagnostic for the conditional flow prior.

Scientific question: can the conditional model control ``<h, v>`` when the
requested coordinate stays inside the natural coordinate distribution the model
was trained on?

The C1 pre-check (`results/conditional_c1_precheck_v1/`) requested coordinates
far outside that support and found saturation. This diagnostic removes the
extrapolation confound by drawing every target from the direction's own natural
quantiles, so a failure here cannot be explained by out-of-distribution targets.

Concept-independent by construction: frozen validation activations, training-only
pool directions, no DEV or held-out data, no LLM judge.

Everything selectable is frozen in ``NATURAL_SUPPORT_SPEC`` before any
controllability number is computed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

RECORDED_QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
TARGET_QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.99)
ARMS = ("correct", "self", "shuffled_target", "unconditional")


@dataclass(frozen=True)
class NaturalSupportSpec:
    """Frozen sampling and analysis plan. Changing a field makes a new diagnostic."""

    version: str = "natural_support_v1"
    reference_rows: int = 262144
    reference_row_seed: int = 20260901
    n_directions: int = 32
    direction_seed: int = 20260902
    n_sequences: int = 64
    sequence_seed: int = 20260903
    noise_seed: int = 20260904
    permutation_seed: int = 20260905
    bootstrap_seed: int = 20260906
    bootstrap_resamples: int = 2000
    confidence: float = 0.95
    target_quantiles: tuple[float, ...] = TARGET_QUANTILES
    recorded_quantiles: tuple[float, ...] = RECORDED_QUANTILES
    t_start: tuple[float, ...] = (0.50, 0.75, 0.90)
    nfe_primary: int = 1
    nfe_optional: int = 3
    # A requested target counts as "meaningfully different" from c0 only when it
    # exceeds this fraction of the direction's own natural coordinate spread.
    min_displacement_frac: float = 0.10
    # Displacement bins are quartiles of |requested_delta| over evaluated rows.
    displacement_bin_quantiles: tuple[float, ...] = (0.25, 0.50, 0.75)

    def __post_init__(self) -> None:
        if self.n_directions < 2 or self.n_sequences < 2:
            raise ValueError("need at least two directions and two sequences")
        if self.n_sequences % self.n_directions:
            raise ValueError("n_sequences must be a whole multiple of n_directions")
        if not all(0.0 < q < 1.0 for q in self.target_quantiles):
            raise ValueError("target quantiles must lie strictly inside (0, 1)")
        if self.min_displacement_frac <= 0.0:
            raise ValueError("min_displacement_frac must be positive")


NATURAL_SUPPORT_SPEC = NaturalSupportSpec()


def select_reference_rows(n_available: int, spec: NaturalSupportSpec) -> np.ndarray:
    """Rows used only to estimate natural coordinate quantiles."""

    if n_available < spec.reference_rows:
        raise ValueError(f"validation artifact has only {n_available} rows")
    rng = np.random.default_rng(spec.reference_row_seed)
    return np.sort(rng.choice(n_available, size=spec.reference_rows, replace=False))


def select_directions(n_available: int, spec: NaturalSupportSpec) -> np.ndarray:
    if n_available < spec.n_directions:
        raise ValueError("direction pool is smaller than the frozen direction count")
    rng = np.random.default_rng(spec.direction_seed)
    return np.sort(rng.choice(n_available, size=spec.n_directions, replace=False))


def select_sequences(validation_sequence_ids: np.ndarray, spec: NaturalSupportSpec) -> np.ndarray:
    """Frozen evaluation sequences, drawn only from the internal validation split."""

    ids = np.asarray(validation_sequence_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) < spec.n_sequences:
        raise ValueError("too few internal-validation sequences for the frozen plan")
    rng = np.random.default_rng(spec.sequence_seed)
    return np.sort(rng.choice(ids, size=spec.n_sequences, replace=False))


def assign_directions(spec: NaturalSupportSpec) -> np.ndarray:
    """One direction index per evaluation sequence, round-robin and deterministic."""

    repeats = spec.n_sequences // spec.n_directions
    return np.tile(np.arange(spec.n_directions, dtype=np.int64), repeats)


def natural_coordinate_stats(
    coordinates: np.ndarray, spec: NaturalSupportSpec
) -> list[dict[str, float | int]]:
    """Per-direction natural coordinate distribution over the reference rows."""

    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("coordinates must be [direction, reference_row]")
    if not np.isfinite(values).all():
        raise ValueError("natural coordinates must be finite")
    stats: list[dict[str, float | int]] = []
    for row in values:
        entry: dict[str, float | int] = {
            "n": int(row.size),
            "mean": float(row.mean()),
            "std": float(row.std(ddof=0)),
            "min": float(row.min()),
            "max": float(row.max()),
        }
        for q, value in zip(
            spec.recorded_quantiles, np.quantile(row, spec.recorded_quantiles), strict=True
        ):
            entry[f"p{int(round(q * 100)):02d}"] = float(value)
        stats.append(entry)
    return stats


def target_coordinates(
    stats: list[dict[str, float | int]], quantile: float
) -> np.ndarray:
    """The requested absolute coordinate for each direction at one target quantile."""

    key = f"p{int(round(quantile * 100)):02d}"
    return np.array([float(entry[key]) for entry in stats], dtype=np.float64)


def evaluable_mask(
    c0: np.ndarray,
    c_target: np.ndarray,
    direction_std: np.ndarray,
    spec: NaturalSupportSpec,
) -> np.ndarray:
    """Keep only rows whose request is meaningfully different from the natural value."""

    requested = np.asarray(c_target, dtype=np.float64) - np.asarray(c0, dtype=np.float64)
    threshold = spec.min_displacement_frac * np.asarray(direction_std, dtype=np.float64)
    return np.abs(requested) > threshold


def displacement_bin_edges(requested: np.ndarray, spec: NaturalSupportSpec) -> np.ndarray:
    """Deterministic quantile bin edges over |requested_delta|."""

    magnitude = np.abs(np.asarray(requested, dtype=np.float64))
    if magnitude.size == 0:
        raise ValueError("cannot bin an empty displacement vector")
    return np.quantile(magnitude, spec.displacement_bin_quantiles)


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.stack([x, np.ones_like(x)], axis=1)
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(solution[0]), float(solution[1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(values: np.ndarray) -> np.ndarray:
        order = values.argsort()
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        return ranks

    return _pearson(rank(x), rank(y))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(np.sqrt((x_centered**2).sum() * (y_centered**2).sum()))
    if denominator == 0.0:
        return float("nan")
    return float((x_centered * y_centered).sum() / denominator)


def calibration(
    requested: np.ndarray,
    realised: np.ndarray,
    sequence_index: np.ndarray,
    spec: NaturalSupportSpec,
) -> dict[str, float | int]:
    """Fit realised_delta ~ slope * requested_delta + intercept.

    The bootstrap resamples *sequences*, never rows: positions inside one
    sequence are not independent observations.
    """

    x = np.asarray(requested, dtype=np.float64)
    y = np.asarray(realised, dtype=np.float64)
    groups = np.asarray(sequence_index, dtype=np.int64)
    if x.shape != y.shape or x.shape != groups.shape or x.size == 0:
        raise ValueError("calibration inputs must be same-shaped and nonempty")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("calibration inputs must be finite")

    slope, intercept = _ols(x, y)
    unique = np.unique(groups)
    by_group = {int(g): np.flatnonzero(groups == g) for g in unique}
    generator = np.random.default_rng(spec.bootstrap_seed)
    slopes = np.empty(spec.bootstrap_resamples, dtype=np.float64)
    for draw in range(spec.bootstrap_resamples):
        picked = generator.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([by_group[int(unique[index])] for index in picked])
        slopes[draw] = _ols(x[rows], y[rows])[0]
    tail = (1.0 - spec.confidence) / 2.0
    lower, upper = np.quantile(slopes, [tail, 1.0 - tail])
    return {
        "slope": slope,
        "slope_ci_lower": float(lower),
        "slope_ci_upper": float(upper),
        "intercept": intercept,
        "pearson": _pearson(x, y),
        "spearman": _spearman(x, y),
        "mean_abs_coordinate_error": float(np.abs(y - x).mean()),
        "median_abs_coordinate_error": float(np.median(np.abs(y - x))),
        "fraction_correct_direction": float((np.sign(y) == np.sign(x)).mean()),
        "n_rows": int(x.size),
        "n_sequences": int(len(unique)),
        "bootstrap_resamples": int(spec.bootstrap_resamples),
        "bootstrap_unit": "validation_sequence",
    }


def per_direction_slopes(
    requested: np.ndarray, realised: np.ndarray, direction_index: np.ndarray
) -> dict[str, object]:
    """Slope per direction plus the leave-one-direction-out range of the pooled slope."""

    x = np.asarray(requested, dtype=np.float64)
    y = np.asarray(realised, dtype=np.float64)
    directions = np.asarray(direction_index, dtype=np.int64)
    unique = np.unique(directions)
    slopes: dict[int, float] = {}
    for value in unique:
        rows = directions == value
        if rows.sum() >= 2:
            slopes[int(value)] = _ols(x[rows], y[rows])[0]
    lovo = []
    for value in unique:
        rows = directions != value
        if rows.sum() >= 2:
            lovo.append(_ols(x[rows], y[rows])[0])
    values = np.array(list(slopes.values()), dtype=np.float64)
    return {
        "per_direction_slope": slopes,
        "n_directions": int(len(slopes)),
        "slope_min": float(values.min()) if values.size else float("nan"),
        "slope_max": float(values.max()) if values.size else float("nan"),
        "slope_median": float(np.median(values)) if values.size else float("nan"),
        "fraction_positive_slope": float((values > 0).mean()) if values.size else float("nan"),
        "lovo_slope_min": float(np.min(lovo)) if lovo else float("nan"),
        "lovo_slope_max": float(np.max(lovo)) if lovo else float("nan"),
    }


def monotonic_across_quantiles(mean_realised_by_quantile: list[float]) -> bool:
    values = np.asarray(mean_realised_by_quantile, dtype=np.float64)
    return bool(np.all(np.diff(values) > 0))


def classify(
    correct: dict[str, float | int],
    shuffled: dict[str, float | int],
    direction_summary: dict[str, object],
    *,
    monotonic: bool,
    delta_lm: float | None = None,
    clean_lm: float | None = None,
    catastrophic_delta_lm_frac: float = 0.25,
) -> tuple[str, list[str]]:
    """Assign category A / B / C from the frozen decision rule.

    A requires every clause; B is "condition is read but control is weak"; C is
    "no useful conditional control".

    The quality clause ("quality cost is not catastrophic inside natural
    support") is part of the specified rule. ``delta_lm`` is the total LM cost of
    the operating point and ``clean_lm`` the unmodified baseline; a cost above
    ``catastrophic_delta_lm_frac`` of the clean loss blocks category A. When
    ``delta_lm`` is not supplied the clause cannot be evaluated and the returned
    category is reported as controllability-only.
    """

    slope = float(correct["slope"])
    slope_lower = float(correct["slope_ci_lower"])
    error_gap = float(shuffled["mean_abs_coordinate_error"]) - float(
        correct["mean_abs_coordinate_error"]
    )
    fraction = float(correct["fraction_correct_direction"])
    positive = float(direction_summary["fraction_positive_slope"])
    lovo_min = float(direction_summary["lovo_slope_min"])

    reasons = [
        f"slope={slope:.4f} (CI {slope_lower:.4f}..{float(correct['slope_ci_upper']):.4f})",
        f"shuffled-minus-correct coordinate error gap={error_gap:+.4f}",
        f"fraction moving in requested direction={fraction:.3f}",
        f"directions with positive slope={positive:.3f}",
        f"LOVO slope min={lovo_min:.4f}",
        f"monotonic across target quantiles={monotonic}",
    ]

    quality_ok: bool | None = None
    if delta_lm is not None and clean_lm is not None:
        quality_ok = float(delta_lm) <= catastrophic_delta_lm_frac * float(clean_lm)
        reasons.append(
            f"delta_lm={float(delta_lm):+.3f} vs clean={float(clean_lm):.3f} "
            f"(budget {catastrophic_delta_lm_frac:.0%}); quality_ok={quality_ok}"
        )
    else:
        reasons.append("delta_lm not supplied; quality clause not evaluated")

    condition_is_read = error_gap > 0.0
    if not condition_is_read and slope < 0.05:
        return "C", reasons
    controllable = (
        monotonic
        and slope_lower > 0.10
        and condition_is_read
        and fraction > 0.80
        and positive > 0.80
        and lovo_min > 0.05
    )
    if controllable and quality_ok is False:
        # Every controllability clause holds but the operating point is ruinous.
        # The taxonomy has no slot for this, so name it rather than force-fit.
        return "A_controllability_quality_blocked", reasons
    if controllable:
        return "A", reasons
    return "B", reasons


def spec_payload(spec: NaturalSupportSpec = NATURAL_SUPPORT_SPEC) -> dict:
    return asdict(spec)
