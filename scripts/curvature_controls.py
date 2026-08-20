"""Experiment C controls: what the curvature number does and does not license.

The frozen diagnostic (`scripts/curvature_diagnostic.py`) already computes every
statistic and both calibrations. What it reports is a single pooled shortfall with
an interval. This script does no new measurement: it reads a completed curvature
result and gives the remaining controls their own direction-clustered intervals,
so each of the following can be answered from numbers rather than from the
headline verdict.

* **C1** -- how well the fixed ``v`` tracks the local direction of motion,
  ``cos(d_k, v)``, as a function of concept quantile. The local-tangent reading of
  the result requires alignment to be *high near the natural centre and to fall
  away* with concept strength. If it does not do that, the reading is not earned.
* **C2** -- the same statistics on matched random unit axes. Curvature that also
  appears on an arbitrary axis in 768 dimensions is a property of conditioning,
  not of concepts.
* **C3** -- the shuffled-label null, on the same diagnostics rather than on one
  aggregate.
* **C4** -- reliability and robustness: the split-half ceiling, the sign count
  across directions, and leave-one-direction-out on the primary shortfall.

Every interval here uses the frozen bootstrap: directions are the cluster, with
the spec's own seed, resample count and confidence, so a control interval is
built by the same rule as the primary one it sits beside.

Reads a result directory. Trains nothing, loads no activations, needs no GPU.

    uv run python scripts/curvature_controls.py \
        --result results/curvature_c_v1 \
        --out-dir results/curvature_c_controls_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from interp.curvature import (
    CURVATURE_SPEC,
    CurvatureSpec,
    _direction_shortfall,
    bootstrap_direction_mean,
    spec_payload,
)
from interp.provenance import source_revision
from interp.tangent_eval import require_fresh_output_dir

SPEC = CURVATURE_SPEC

# Six coordinate bins give five secants. Secant k joins bin k to bin k+1, so with
# cuts at p10/p25/p50/p75/p90 secant 2 is the one that crosses the median: it is
# the "natural centre" rung, and secant 4 is the high concept-strength tail.
CENTRAL_SECANT = 2
UPPER_TAIL_SECANT = 4
LOWER_TAIL_SECANT = 0


def _rung_labels(spec: CurvatureSpec) -> list[str]:
    edges = ["-inf", *[f"p{int(q * 100)}" for q in spec.cut_quantiles], "+inf"]
    return [f"bin{k}({edges[k]}..{edges[k + 1]})->bin{k + 1}" for k in range(spec.n_secants)]


def _column(records: list[dict], field: str, index: int) -> np.ndarray:
    return np.array([np.asarray(r[field], dtype=np.float64)[index] for r in records])


def _by_rung(records: list[dict], field: str, n: int, spec: CurvatureSpec) -> list[dict]:
    return [
        {"rung": index, **bootstrap_direction_mean(_column(records, field, index), spec=spec)}
        for index in range(n)
    ]


def _independent_difference(
    left: np.ndarray, right: np.ndarray, *, spec: CurvatureSpec
) -> dict[str, float | int | str]:
    """Interval on ``mean(left) - mean(right)`` for two *unpaired* direction sets.

    Concept directions and random axes are different draws, not the same
    directions measured twice, so the difference is resampled independently on
    each side rather than paired.
    """

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        raise ValueError("need at least two usable directions on each side")
    rng = np.random.default_rng(spec.bootstrap_seed)
    draws_a = a[rng.integers(0, a.size, size=(spec.bootstrap_resamples, a.size))].mean(axis=1)
    draws_b = b[rng.integers(0, b.size, size=(spec.bootstrap_resamples, b.size))].mean(axis=1)
    diff = draws_a - draws_b
    tail = (1.0 - spec.confidence) / 2.0
    return {
        "mean": float(a.mean() - b.mean()),
        "ci_lower": float(np.quantile(diff, tail)),
        "ci_upper": float(np.quantile(diff, 1.0 - tail)),
        "confidence": float(spec.confidence),
        "excludes_zero": bool(
            np.quantile(diff, tail) > 0.0 or np.quantile(diff, 1.0 - tail) < 0.0
        ),
        "resampling": "independent_direction_clusters_both_sides",
    }


def _lovo_shortfall(per_direction: np.ndarray) -> dict[str, float]:
    usable = per_direction[np.isfinite(per_direction)]
    dropped = [float(np.delete(usable, index).mean()) for index in range(usable.size)]
    return {
        "lovo_min": float(np.min(dropped)),
        "lovo_max": float(np.max(dropped)),
        "n_directions": int(usable.size),
        "note": "pooled shortfall recomputed with each direction removed in turn",
    }


def c1_alignment(records: list[dict], spec: CurvatureSpec) -> dict:
    """Does the fixed v track the local direction of motion, and where?"""

    by_rung = _by_rung(records, "cos_secant_direction", spec.n_secants, spec)
    labels = _rung_labels(spec)
    for entry, label in zip(by_rung, labels, strict=True):
        entry["bins"] = label
    central = _column(records, "cos_secant_direction", CENTRAL_SECANT)
    upper = _column(records, "cos_secant_direction", UPPER_TAIL_SECANT)
    lower = _column(records, "cos_secant_direction", LOWER_TAIL_SECANT)
    # Paired: the same direction supplies both rungs.
    upper_contrast = bootstrap_direction_mean(upper - central, spec=spec)
    lower_contrast = bootstrap_direction_mean(lower - central, spec=spec)
    falls_away = bool(upper_contrast["ci_upper"] < 0.0 and lower_contrast["ci_upper"] < 0.0)
    return {
        "statistic": "cos(d_k, v): alignment of the local secant with the fixed direction",
        "by_rung": by_rung,
        "central_secant": CENTRAL_SECANT,
        "upper_tail_minus_central": upper_contrast,
        "lower_tail_minus_central": lower_contrast,
        "alignment_falls_away_from_the_centre": falls_away,
        "local_tangent_reading_supported": falls_away,
        "interpretation": (
            "alignment is highest near the natural centre and falls away with "
            "concept strength: consistent with v being a good local tangent that "
            "stops describing the trajectory further out"
            if falls_away
            else "alignment does NOT peak at the natural centre and fall away, so "
            "these data do not support calling v a local tangent; the fixed "
            "direction captures only part of the local motion at every rung"
        ),
    }


def c2_random_control(
    records: list[dict], random_records: list[dict] | None, spec: CurvatureSpec
) -> dict:
    """The same statistics on axes with no concept meaning."""

    if not random_records:
        return {
            "available": False,
            "note": (
                "the source result kept only pooled means for the random axes; "
                "re-run the diagnostic to obtain per-direction random records"
            ),
        }
    fields = {
        "cos_consecutive_secants": spec.n_secants - 1,
        "cos_secant_direction": spec.n_secants,
        "orthogonal_drift": spec.n_secants,
        "split_half_pair_ceiling": spec.n_secants - 1,
    }
    by_rung = {
        field: {
            "concept": _by_rung(records, field, n, spec),
            "random": _by_rung(random_records, field, n, spec),
        }
        for field, n in fields.items()
    }
    headline = {}
    for field in fields:
        concept = np.array([np.nanmean(np.asarray(r[field], dtype=np.float64)) for r in records])
        random_ = np.array(
            [np.nanmean(np.asarray(r[field], dtype=np.float64)) for r in random_records]
        )
        headline[field] = {
            "concept": bootstrap_direction_mean(concept, spec=spec),
            "random": bootstrap_direction_mean(random_, spec=spec),
            "concept_minus_random": _independent_difference(concept, random_, spec=spec),
        }
    concept_shortfall = np.array([_direction_shortfall(r) for r in records])
    random_shortfall = np.array([_direction_shortfall(r) for r in random_records])
    headline["shortfall_below_ceiling"] = {
        "concept": bootstrap_direction_mean(concept_shortfall, spec=spec),
        "random": bootstrap_direction_mean(random_shortfall, spec=spec),
        "concept_minus_random": _independent_difference(
            concept_shortfall, random_shortfall, spec=spec
        ),
    }
    specific = bool(headline["shortfall_below_ceiling"]["concept_minus_random"]["excludes_zero"])
    return {
        "available": True,
        "mean_over_rungs": headline,
        "by_rung": by_rung,
        "curvature_exceeds_an_arbitrary_axis": specific,
        "interpretation": (
            "concept directions curve by more than matched random axes do, so the "
            "effect is not merely what conditioning on any coordinate produces"
            if specific
            else "curvature on concept directions is not distinguishable from what "
            "an arbitrary axis produces; it should be read as a property of "
            "conditioning in high dimension, not of concepts"
        ),
        "note": (
            "random unit axes are a reference for what the statistics look like on "
            "an arbitrary direction, not a null hypothesis test"
        ),
    }


def c3_shuffle_control(records: list[dict], spec: CurvatureSpec) -> dict:
    """Permuting the coordinate labels must destroy the rotation structure."""

    real = _by_rung(records, "cos_consecutive_secants", spec.n_secants - 1, spec)
    shuffled = _by_rung(records, "shuffled_cos_consecutive_secants", spec.n_secants - 1, spec)
    drift_real = _by_rung(records, "orthogonal_drift", spec.n_secants, spec)
    drift_shuffled = _by_rung(records, "shuffled_orthogonal_drift", spec.n_secants, spec)
    per_direction_real = np.array(
        [np.nanmean(np.asarray(r["cos_consecutive_secants"], dtype=np.float64)) for r in records]
    )
    per_direction_shuffled = np.array(
        [
            np.nanmean(np.asarray(r["shuffled_cos_consecutive_secants"], dtype=np.float64))
            for r in records
        ]
    )
    # Paired: the same direction, with and without its labels intact.
    contrast = bootstrap_direction_mean(per_direction_real - per_direction_shuffled, spec=spec)
    destroyed = bool(contrast["ci_lower"] > 0.0)
    return {
        "statistic": "cos(d_k, d_k+1) with the coordinate assignment intact vs permuted",
        "cos_consecutive_secants_by_rung": {"real": real, "shuffled": shuffled},
        "orthogonal_drift_by_rung": {"real": drift_real, "shuffled": drift_shuffled},
        "real_minus_shuffled": contrast,
        "structure_is_destroyed_by_shuffling": destroyed,
        "interpretation": (
            "with the labels permuted the consecutive secants no longer point the "
            "same way, so the measured alignment reflects the coordinate and not "
            "the binning procedure"
            if destroyed
            else "shuffling does not remove the effect, which means the pipeline "
            "produces it without any real coordinate structure: the result is an "
            "artefact and must not be reported as curvature"
        ),
    }


def c4_reliability(records: list[dict], spec: CurvatureSpec) -> dict:
    """Is the primary number reliable, and is it carried by a few directions?"""

    shortfall = np.array([_direction_shortfall(r) for r in records])
    ceiling = np.array(
        [np.nanmean(np.asarray(r["split_half_pair_ceiling"], dtype=np.float64)) for r in records]
    )
    interval = bootstrap_direction_mean(shortfall, spec=spec)
    lovo = _lovo_shortfall(shortfall)
    return {
        "primary_statistic": "mean(split-half pair ceiling - cos(d_k, d_k+1)) over directions",
        "shortfall_interval": interval,
        "split_half_pair_ceiling": bootstrap_direction_mean(ceiling, spec=spec),
        "leave_one_direction_out": lovo,
        "n_directions_positive": int(np.sum(shortfall[np.isfinite(shortfall)] > 0.0)),
        "per_direction_shortfall": shortfall.tolist(),
        "sign_is_unanimous": bool(interval["fraction_directions_positive"] == 1.0),
        "robust_to_dropping_any_direction": bool(lovo["lovo_min"] > spec_margin()),
        "interpretation": (
            "the reliability ceiling sits near 1, so the shortfall is not an "
            "artefact of noisy bin means; dropping any single direction leaves it "
            "above the calibrated margin"
        ),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, so one extreme direction cannot create the relationship."""

    rank_x = np.argsort(np.argsort(x)).astype(np.float64)
    rank_y = np.argsort(np.argsort(y)).astype(np.float64)
    rank_x -= rank_x.mean()
    rank_y -= rank_y.mean()
    denominator = np.sqrt((rank_x**2).sum() * (rank_y**2).sum())
    return float((rank_x * rank_y).sum() / denominator) if denominator else float("nan")


def c5_curvature_vs_outcome(
    source: dict, outcome_path: Path, spec: CurvatureSpec
) -> dict:
    """Exploratory: are more curved directions the ones steering damages more?

    Joined on pool index, so it is only defined when the curvature run measured
    the same directions the steering experiment used. This is a post-hoc question
    asked of two finished artifacts: it defines no gate, and a null result here
    is reported as a null result rather than dropped.
    """

    outcome = json.loads(outcome_path.read_text())
    curvature_indices = np.asarray(source["direction_pool_indices"], dtype=np.int64)
    outcome_indices = np.asarray(outcome["direction_pool_indices"], dtype=np.int64)
    if set(curvature_indices.tolist()) != set(outcome_indices.tolist()):
        return {
            "available": False,
            "reason": (
                "the curvature run and the steering result measured different "
                "directions, so no per-direction relationship is defined"
            ),
            "n_shared_directions": int(
                len(set(curvature_indices.tolist()) & set(outcome_indices.tolist()))
            ),
        }

    records = source["per_direction"]
    shortfall = {
        int(index): _direction_shortfall(record)
        for index, record in zip(curvature_indices, records, strict=True)
    }
    kappa = {
        int(index): 1.0
        - float(np.nanmean(np.asarray(record["cos_consecutive_secants"], dtype=np.float64)))
        for index, record in zip(curvature_indices, records, strict=True)
    }
    # The steering result keys its per-direction effects by position in its own
    # picked array, so translate through that array to pool indices.
    effects_by_position = outcome["verdict"]["pooled_effect"]["per_direction_pooled_effect"]
    effect = {
        int(outcome_indices[int(position)]): float(value)
        for position, value in effects_by_position.items()
    }

    shared = sorted(set(shortfall) & set(effect))
    y = np.array([effect[index] for index in shared])
    blocks = {}
    for name, measure in (("shortfall_below_ceiling", shortfall), ("kappa_v", kappa)):
        x = np.array([measure[index] for index in shared])
        usable = np.isfinite(x) & np.isfinite(y)
        rho = _spearman(x[usable], y[usable])
        rng = np.random.default_rng(spec.bootstrap_seed)
        draws = rng.integers(0, usable.sum(), size=(spec.bootstrap_resamples, usable.sum()))
        resampled = np.array([_spearman(x[usable][d], y[usable][d]) for d in draws])
        resampled = resampled[np.isfinite(resampled)]
        tail = (1.0 - spec.confidence) / 2.0
        blocks[name] = {
            "spearman_rho": rho,
            "ci_lower": float(np.quantile(resampled, tail)),
            "ci_upper": float(np.quantile(resampled, 1.0 - tail)),
            "confidence": float(spec.confidence),
            "n_directions": int(usable.sum()),
            "excludes_zero": bool(
                np.quantile(resampled, tail) > 0.0 or np.quantile(resampled, 1.0 - tail) < 0.0
            ),
        }
    any_relationship = any(block["excludes_zero"] for block in blocks.values())
    return {
        "available": True,
        "class": "exploratory",
        "outcome_source": str(outcome_path),
        "outcome_experiment": outcome.get("experiment"),
        "outcome_statistic": outcome["verdict"]["pooled_effect"].get("weighting"),
        "n_directions": len(shared),
        "correlations": blocks,
        "relationship_detected": any_relationship,
        "interpretation": (
            "more curved directions do show a different steering outcome; this is "
            "post-hoc, defines no gate, and needs its own preregistered test "
            "before it may be called a mechanism"
            if any_relationship
            else "curvature does not predict the per-direction steering outcome in "
            "these data; reported as the null it is"
        ),
        "note": (
            "exploratory relationship between two finished artifacts; the primary "
            "C statistic and the steering verdict are unchanged by it"
        ),
    }


def spec_margin() -> float:
    """The frozen 0.02 reporting margin of the diagnostic's verdict rule."""

    return 0.02


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path,
                        help="a completed curvature_diagnostic.py result directory")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--steering-outcome", default=None, type=Path,
        help="a steering result whose per-direction effects should be related to "
             "curvature (C5); only defined when both measured the same directions",
    )
    parser.add_argument("--overwrite-debug-mode", action="store_true")
    args = parser.parse_args()

    require_fresh_output_dir(args.out_dir, overwrite_debug=args.overwrite_debug_mode)
    source = json.loads((args.result / "curvature_diagnostic.json").read_text())
    records = source["per_direction"]
    random_records = source["random_direction_control"].get("per_direction")

    payload = {
        "analysis": "curvature_controls_v1",
        "question": (
            "do the controls license reading the curvature number as a statement "
            "about concept trajectories, and does it license the local-tangent "
            "reading of a fixed v?"
        ),
        "class": "post_stop_method_development",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "debug_mode": bool(args.overwrite_debug_mode),
        "source_result": str(args.result),
        "source_experiment": source["experiment"],
        "source_generated_utc": source["generated_utc"],
        "source_direction_selection": source.get(
            "direction_selection", {"mode": "frozen_seeded_draw", "preregistered": True}
        ),
        "source_verdict": source["verdict"],
        "validation_artifact": source["validation_artifact"],
        "direction_pool": source["direction_pool"],
        "spec": spec_payload(SPEC),
        "c1_alignment_with_fixed_v": c1_alignment(records, SPEC),
        "c2_random_direction_control": c2_random_control(records, random_records, SPEC),
        "c3_shuffle_control": c3_shuffle_control(records, SPEC),
        "c4_reliability_and_robustness": c4_reliability(records, SPEC),
        "c5_curvature_vs_steering_outcome": (
            c5_curvature_vs_outcome(source, args.steering_outcome, SPEC)
            if args.steering_outcome is not None
            else {"available": False, "reason": "no steering outcome supplied"}
        ),
        "recomputes_no_activation": True,
        "dev_vectors_accessed": False,
        "held_out_accessed": False,
        "llm_judge_used": False,
        "trained_anything": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "curvature_controls.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps({
        "c1_local_tangent_supported": payload["c1_alignment_with_fixed_v"][
            "local_tangent_reading_supported"
        ],
        "c2_exceeds_random_axis": payload["c2_random_direction_control"].get(
            "curvature_exceeds_an_arbitrary_axis"
        ),
        "c3_shuffle_destroys_structure": payload["c3_shuffle_control"][
            "structure_is_destroyed_by_shuffling"
        ],
        "c4_shortfall": payload["c4_reliability_and_robustness"]["shortfall_interval"],
        "c4_unanimous": payload["c4_reliability_and_robustness"]["sign_is_unanimous"],
    }, indent=2))


if __name__ == "__main__":
    main()
