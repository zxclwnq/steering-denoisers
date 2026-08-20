"""Frozen vector-aware analysis of a completed clean Phase-B DEV release.

The generation stage writes immutable raw rows; nothing there is analysed. This
module is the versioned analysis layer: it recomputes every headline quantity
from the raw rows only, using the statistical and matching contract frozen in
the evaluator config.

Scientific rules encoded here rather than left to the caller:

* the statistical unit is the steering direction, never the generation row;
* realised-projection matching interpolates only between bracketing alphas of the
  same ``(vector, prompt_id, generation_seed)`` family, with no extrapolation and
  no clipping;
* a comparison point is discarded when the flow row or either bracketing baseline
  row fails the frozen repetition gate;
* stress alphas never enter the primary grid;
* one resample matrix is shared by every comparison in one analysis.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ANALYSIS_VERSION = "clean_phase_b_analysis_v1"
PRIMARY_ROWS = 2_400
TOTAL_ROWS = 2_880
ROW_METRICS = (
    "nll",
    "lexicon_score",
    "sae_act_target",
    "sae_act_control_mean",
    "sae_act_control_max",
    "repetition_rate",
    "dist_1",
    "dist_2",
    "dist_3",
)


@dataclass(frozen=True)
class CellKey:
    vector: str
    alpha_hex: str
    prompt_id: int
    generation_seed: int


def _cell(row: dict) -> CellKey:
    return CellKey(
        vector=row["vector"],
        alpha_hex=row["alpha_hex"],
        prompt_id=row["prompt_id"],
        generation_seed=row["generation_seed"],
    )


def load_release_rows(
    path: Path,
    *,
    release_id: str,
    schema_version: str,
    metric_versions: dict[str, str],
    expected_method: str,
    expected_arm: str,
) -> tuple[dict, ...]:
    """Read one immutable artifact and refuse anything but its exact frozen shape."""

    meta_path = path.with_suffix(".meta.json")
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"incomplete Phase-B artifact: {path}")
    meta = json.loads(meta_path.read_text())
    if meta.get("status") != "complete" or meta.get("release_id") != release_id:
        raise ValueError(f"{path} is not a completed artifact of release {release_id}")
    if int(meta.get("row_count", -1)) != TOTAL_ROWS:
        raise ValueError(f"{path} receipt does not record {TOTAL_ROWS} rows")

    rows: list[dict] = []
    seen: set[CellKey] = set()
    with path.open() as handle:
        for number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if (
                row.get("schema_version") != schema_version
                or row.get("release_id") != release_id
                or row.get("metric_versions") != metric_versions
                or row.get("split") != "dev"
                or row.get("method") != expected_method
                or row.get("arm_id") != expected_arm
            ):
                raise ValueError(f"{path} row {number} is not the frozen release identity")
            key = _cell(row)
            if key in seen:
                raise ValueError(f"{path} row {number} duplicates a continuation cell")
            seen.add(key)
            values = row["metrics"]
            if not all(math.isfinite(float(values[name])) for name in ROW_METRICS):
                raise ValueError(f"{path} row {number} carries a non-finite metric")
            rows.append(row)
    if len(rows) != TOTAL_ROWS:
        raise ValueError(f"{path} holds {len(rows)} rows, not {TOTAL_ROWS}")
    return tuple(rows)


def primary_rows(rows: tuple[dict, ...]) -> tuple[dict, ...]:
    """Drop the stress alphas; the primary grid is the only frontier-defining set."""

    kept = tuple(row for row in rows if row["is_stress"] is False)
    if len(kept) != PRIMARY_ROWS:
        raise ValueError(f"primary grid holds {len(kept)} rows, not {PRIMARY_ROWS}")
    return kept


def resample_matrix(vectors: tuple[str, ...], *, seed: int, n_resamples: int) -> np.ndarray:
    """One shared bootstrap index matrix over steering directions."""

    if len(vectors) != len(set(vectors)) or not vectors:
        raise ValueError("bootstrap vectors must be a nonempty unique sequence")
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(vectors), size=(n_resamples, len(vectors)))


def vector_bootstrap(
    per_vector: dict[str, float],
    vectors: tuple[str, ...],
    matrix: np.ndarray,
    *,
    confidence: float,
) -> dict:
    """Bootstrap the mean of the per-direction means with the shared resample matrix."""

    if set(per_vector) != set(vectors):
        raise ValueError("bootstrap input must cover exactly the frozen vector set")
    values = np.array([per_vector[name] for name in vectors], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("bootstrap input must be finite")
    draws = values[matrix].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    positive = int((values > 0).sum())
    return {
        "mean": float(values.mean()),
        "ci_lower": float(np.quantile(draws, tail)),
        "ci_upper": float(np.quantile(draws, 1.0 - tail)),
        "confidence": confidence,
        "n_vectors": int(len(values)),
        "n_resamples": int(matrix.shape[0]),
        "unit": "vector_mean",
        "vector_means": {name: float(value) for name, value in zip(vectors, values, strict=True)},
        "vector_signs": {"positive": positive, "negative": int(len(values) - positive)},
        "leave_one_vector_out": {
            name: float(np.delete(values, index).mean()) for index, name in enumerate(vectors)
        },
    }


def _vector_means(pairs: list[tuple[str, float]], vectors: tuple[str, ...]) -> dict[str, float]:
    sums: dict[str, float] = {name: 0.0 for name in vectors}
    counts: dict[str, int] = {name: 0 for name in vectors}
    for name, value in pairs:
        sums[name] += value
        counts[name] += 1
    missing = [name for name in vectors if counts[name] == 0]
    if missing:
        raise ValueError(f"no supported comparison points for directions {missing}")
    return {name: sums[name] / counts[name] for name in vectors}


def descriptive(rows: tuple[dict, ...]) -> dict:
    """Unweighted primary-grid means, plus the geometry means the protocol requires."""

    grid = primary_rows(rows)
    out = {
        name: float(np.mean([float(row["metrics"][name]) for row in grid])) for name in ROW_METRICS
    }
    out["n_rows"] = len(grid)
    out["degenerate_rows"] = int(
        sum(float(row["metrics"]["repetition_rate"]) > 0.0 for row in grid)
    )
    if grid[0]["geometry"] is not None:
        out["realized_projection_mean"] = float(
            np.mean([float(row["geometry"]["realized_projection_mean"]) for row in grid])
        )
        for name in (
            "correction_norm_mean",
            "parallel_correction_norm_mean",
            "orthogonal_correction_norm_mean",
            "correction_cosine_mean",
        ):
            values = [
                float(row["geometry"][name]) for row in grid if row["geometry"][name] is not None
            ]
            out[name] = float(np.mean(values))
        retained = [
            float(row["geometry"]["retained_fraction_mean"])
            for row in grid
            if row["geometry"]["retained_fraction_mean"] is not None
        ]
        out["retained_fraction_mean"] = float(np.mean(retained))
        out["retained_fraction_rows"] = len(retained)
        out["alpha_zero_delta_projection"] = float(
            np.mean(
                [
                    float(row["geometry"]["realized_projection_mean"])
                    for row in grid
                    if row["alpha"] == 0.0
                ]
            )
        )
    return out


def equal_alpha_paired(
    flow: tuple[dict, ...],
    baseline: tuple[dict, ...],
    vectors: tuple[str, ...],
    matrix: np.ndarray,
    *,
    confidence: float,
    metrics: tuple[str, ...] = ROW_METRICS,
) -> dict:
    """Paired flow-minus-baseline effects at equal nominal alpha.

    This is the weak comparison: it holds nominal alpha fixed, not realised
    steering strength, so an improvement here is also what pure attenuation
    produces.
    """

    by_cell = {_cell(row): row for row in primary_rows(baseline)}
    grid = primary_rows(flow)
    if {_cell(row) for row in grid} != set(by_cell):
        raise ValueError("paired comparison requires identical continuation cells")
    out = {}
    for name in metrics:
        pairs = [
            (
                row["vector"],
                float(row["metrics"][name]) - float(by_cell[_cell(row)]["metrics"][name]),
            )
            for row in grid
        ]
        out[name] = vector_bootstrap(
            _vector_means(pairs, vectors), vectors, matrix, confidence=confidence
        )
    out["n_pairs"] = len(grid)
    return out


def _baseline_curves(
    baseline: tuple[dict, ...], *, coordinate_scale: float
) -> dict[tuple[str, int, int], list[dict]]:
    curves: dict[tuple[str, int, int], list[dict]] = {}
    for row in primary_rows(baseline):
        key = (row["vector"], row["prompt_id"], row["generation_seed"])
        curves.setdefault(key, []).append(
            {"coordinate": float(row["alpha"]) * coordinate_scale, "row": row}
        )
    for key, points in curves.items():
        points.sort(key=lambda item: item["coordinate"])
        coordinates = [item["coordinate"] for item in points]
        if len(set(coordinates)) != len(coordinates):
            raise ValueError(f"baseline curve {key} has duplicate coordinates")
    return curves


def matched_projection(
    flow: tuple[dict, ...],
    baseline: tuple[dict, ...],
    vectors: tuple[str, ...],
    matrix: np.ndarray,
    *,
    coordinate_scale: float,
    repetition_threshold: float,
    confidence: float,
    metrics: tuple[str, ...] = ROW_METRICS,
) -> tuple[dict, list[dict]]:
    """Compare flow against the baseline curve at equal realised steering strength.

    The baseline coordinate is its exact realised displacement along the unit
    direction: ``alpha`` for additive steering and ``kappa * alpha`` for scalar
    shrinkage. Interpolation is linear and strictly inside the observed bracket.
    """

    curves = _baseline_curves(baseline, coordinate_scale=coordinate_scale)
    supported: dict[str, list[tuple[str, float]]] = {name: [] for name in metrics}
    records: list[dict] = []
    counts = {"supported": 0, "unsupported": 0, "degenerate": 0}
    weights: list[float] = []

    for row in primary_rows(flow):
        key = (row["vector"], row["prompt_id"], row["generation_seed"])
        target = float(row["geometry"]["realized_projection_mean"])
        points = curves[key]
        record = {
            "arm_id": row["arm_id"],
            "vector": row["vector"],
            "prompt_id": row["prompt_id"],
            "generation_seed": row["generation_seed"],
            "alpha_hat": row["alpha_hat"],
            "target_projection": target,
        }
        if float(row["metrics"]["repetition_rate"]) > repetition_threshold:
            counts["degenerate"] += 1
            records.append({**record, "status": "degenerate_flow"})
            continue
        lower = None
        for left, right in zip(points, points[1:], strict=False):
            if left["coordinate"] <= target <= right["coordinate"]:
                lower, upper = left, right
                break
        if lower is None:
            counts["unsupported"] += 1
            records.append({**record, "status": "outside_bracket"})
            continue
        if any(
            float(item["row"]["metrics"]["repetition_rate"]) > repetition_threshold
            for item in (lower, upper)
        ):
            counts["degenerate"] += 1
            records.append({**record, "status": "degenerate_bracket"})
            continue
        span = upper["coordinate"] - lower["coordinate"]
        weight = 0.0 if span == 0.0 else (target - lower["coordinate"]) / span
        counts["supported"] += 1
        weights.append(weight)
        differences = {}
        for name in metrics:
            interpolated = (1.0 - weight) * float(lower["row"]["metrics"][name]) + weight * float(
                upper["row"]["metrics"][name]
            )
            difference = float(row["metrics"][name]) - interpolated
            supported[name].append((row["vector"], difference))
            differences[name] = {
                "flow": float(row["metrics"][name]),
                "matched": interpolated,
                "difference": difference,
            }
        records.append(
            {
                **record,
                "status": "supported",
                "bracket_lower_alpha_hat": lower["row"]["alpha_hat"],
                "bracket_upper_alpha_hat": upper["row"]["alpha_hat"],
                "bracket_lower_coordinate": lower["coordinate"],
                "bracket_upper_coordinate": upper["coordinate"],
                "weight": weight,
                "metrics": differences,
            }
        )

    summary = {
        name: vector_bootstrap(
            _vector_means(supported[name], vectors), vectors, matrix, confidence=confidence
        )
        for name in metrics
    }
    summary["counts"] = counts
    summary["coordinate_scale"] = coordinate_scale
    summary["mean_bracket_weight"] = float(np.mean(weights)) if weights else None
    summary["extrapolation"] = "forbidden"
    summary["clipping"] = "forbidden"
    summary["repetition_threshold"] = repetition_threshold
    # Applied symmetrically: dropping degenerate flow rows flatters flow, dropping
    # degenerate bracket rows penalises it, and the frozen contract excludes both
    # from the valid frontier. Any variant that filters only one side of the
    # comparison biases the frontier and must not be used.
    summary["degeneracy_rule"] = (
        "flow row and both bracketing baseline rows must satisfy "
        "repetition_rate <= repetition_threshold"
    )
    return summary, records


def nfe_effects(
    arms: dict[str, tuple[dict, ...]],
    vectors: tuple[str, ...],
    matrix: np.ndarray,
    *,
    confidence: float,
    metrics: tuple[str, ...] = ("nll", "sae_act_target", "lexicon_score", "repetition_rate"),
) -> dict:
    """Paired higher-NFE minus NFE-1 effects at each fixed ``t_start``."""

    out = {}
    for tag in ("t010", "t025", "t050"):
        reference = arms[f"flow_{tag}_nfe1"]
        for nfe in (3, 5):
            out[f"{tag}_nfe{nfe}_minus_nfe1"] = equal_alpha_paired(
                arms[f"flow_{tag}_nfe{nfe}"],
                reference,
                vectors,
                matrix,
                confidence=confidence,
                metrics=metrics,
            )
    return out


def paired_release_difference(
    wide: dict,
    narrow: dict,
    vectors: tuple[str, ...],
    matrix: np.ndarray,
    *,
    confidence: float,
) -> dict:
    """Bootstrap ``wide - narrow`` over directions from two per-vector mean maps.

    Both inputs are ``vector_bootstrap`` outputs computed on the same eight
    frozen directions, so the difference is paired at the scientific unit even
    though the two releases generated different continuations.
    """

    difference = {
        name: float(wide["vector_means"][name]) - float(narrow["vector_means"][name])
        for name in vectors
    }
    result = vector_bootstrap(difference, vectors, matrix, confidence=confidence)
    result["narrow_mean"] = float(narrow["mean"])
    result["wide_mean"] = float(wide["mean"])
    result["improved"] = bool(result["ci_upper"] < 0.0)
    result["worsened"] = bool(result["ci_lower"] > 0.0)
    return result
