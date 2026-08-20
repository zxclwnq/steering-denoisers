"""Exploratory: does the C6 nonlinearity predict per-direction failure in B?

The measure was fixed before looking at anything: ``x_v = MSE_linear -
MSE_quadratic`` on the held-out half, the most directly controlled nonlinearity
statistic C6 produces. The outcome was fixed too: B's per-direction
matched-strength ``delta NLL``, never the nominal-alpha gain. Projected variance
enters as a control because C6 showed the geometry statistics track it.

The join was **not** possible from the originally frozen artifacts: Experiment B
inherits the `natural_support_v1` direction draw and C6 inherits the Experiment C
draw, and the two sets share no direction. Rather than substitute one of the
statistics that *is* available for B's directions --- the curvature shortfall,
the tail-versus-centre contrast, either cosine --- which would be choosing a
measure after learning the chosen one was unavailable, the same C6.2 measure was
recomputed for B's draw under explicit authorization
(`scripts/c6_nonlinearity_on_steering_directions.py`). Everything about that
computation is inherited unchanged from the C6 protocol except the direction set,
which is itself frozen by `natural_support_v1`.

Both sides are joined on pool index, never on position.

    uv run python scripts/c6_b_direction_join.py \
        --b results/steering_denoiser_b_v1 \
        --c6 results/c6_nonlinearity_steering_directions_v1 \
        --out-dir results/c6_b_direction_join_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from interp.curvature import CURVATURE_SPEC
from interp.provenance import source_revision
from interp.tangent_eval import require_fresh_output_dir

# Fixed before the join was inspected.
PRIMARY_MEASURE = "mse_linear_minus_mse_quadratic_held_out"
PRIMARY_OUTCOME = "matched_strength_paired_delta_nll"
BOOTSTRAP_SEED = 20260923


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rank_x = np.argsort(np.argsort(x)).astype(np.float64)
    rank_y = np.argsort(np.argsort(y)).astype(np.float64)
    rank_x -= rank_x.mean()
    rank_y -= rank_y.mean()
    denominator = np.sqrt((rank_x**2).sum() * (rank_y**2).sum())
    return float((rank_x * rank_y).sum() / denominator) if denominator else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--c6", required=True, type=Path,
                        help="a directory holding the C6.2 nonlinearity measure "
                             "for the same directions Experiment B evaluated")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--overwrite-debug-mode", action="store_true")
    args = parser.parse_args()

    require_fresh_output_dir(args.out_dir, overwrite_debug=args.overwrite_debug_mode)
    b = json.loads((args.b / "steering_denoiser.json").read_text())
    # Either the original C6 run or the same measure recomputed for B's draw.
    c6_path = args.c6 / "nonlinearity.json"
    if not c6_path.is_file():
        c6_path = args.c6 / "covariance_controls.json"
    c6 = json.loads(c6_path.read_text())
    fit = c6.get("c6_2_held_out_conditional_fit", c6)
    variances = c6.get("c6_1_covariance_predicted_direction", {}).get(
        "projected_variance", {}
    ).get("concept", c6.get("projected_variance"))

    b_ids = [int(i) for i in b["direction_pool_indices"]]
    c6_ids = [int(i) for i in c6["direction_pool_indices"]]
    shared = sorted(set(b_ids) & set(c6_ids))

    # The join sanity checks, run before any statistic.
    checks = {
        "b_n_directions": len(b_ids),
        "c6_n_directions": len(c6_ids),
        "b_ids_unique": len(set(b_ids)) == len(b_ids),
        "c6_ids_unique": len(set(c6_ids)) == len(c6_ids),
        "n_shared_directions": len(shared),
        "b_direction_seed_source": "natural_support_v1 plan (inherited by Experiment B)",
        "c6_direction_selection": c6.get("direction_selection", {
            "source": "Experiment C frozen draw",
            "direction_seed": CURVATURE_SPEC.direction_seed,
        }),
        "b_outcome_available": "per_direction_pooled_effect"
        in b["verdict"]["pooled_effect"],
        "c6_measure_available": "delta_mse_linear_minus_quadratic" in fit,
        "c6_measure_source": str(c6_path),
        "held_out_accessed": False,
    }

    payload = {
        "experiment": "c6_b_direction_join_v1",
        "preregistered": False,
        "exploratory_post_hoc": True,
        "question": (
            "does the nonlinearity of the conditional geometry beyond covariance "
            "predict how badly steering repair fails, per direction?"
        ),
        "primary_measure": PRIMARY_MEASURE,
        "primary_outcome": PRIMARY_OUTCOME,
        "variance_control": "log(v' Sigma v)",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "b_source": str(args.b),
        "c6_source": str(args.c6),
        "b_direction_pool_indices": b_ids,
        "c6_direction_pool_indices": c6_ids,
        "shared_direction_pool_indices": shared,
        "join_checks": checks,
        "modified_frozen_artifacts": False,
        "dev_vectors_accessed": False,
        "held_out_accessed": False,
        "llm_judge_used": False,
        "trained_anything": False,
    }

    if len(shared) < 3:
        payload["verdict"] = {
            "outcome": "JOIN_NOT_POSSIBLE_FROM_FROZEN_ARTIFACTS",
            "why": (
                "Experiment B and Experiment C6 measured disjoint direction sets. "
                "B inherits the natural_support_v1 draw; C6 inherits the "
                f"Experiment C draw (seed {CURVATURE_SPEC.direction_seed}). They "
                f"share {len(shared)} directions, so the preregistered measure "
                "does not exist for any direction B evaluated."
            ),
            "why_no_substitute": (
                "the curvature shortfall, the tail-versus-centre contrast and both "
                "cosine statistics ARE available for B's directions, via "
                "results/curvature_c_steering_directions_v1. They were excluded in "
                "advance and are not substituted here: swapping the measure after "
                "seeing that the chosen one is unavailable is the exact failure the "
                "pre-specification exists to prevent."
            ),
            "what_would_be_required": (
                "recomputing the held-out linear-versus-quadratic conditional fit "
                "for B's 32 directions against the 262144 x 768 activation matrix "
                "-- a new job over activations, cancelled by the analysis's own rule"
            ),
            "statistics_computed": False,
        }
    else:
        b_effect = b["verdict"]["pooled_effect"]["per_direction_pooled_effect"]
        b_order = {index: position for position, index in enumerate(b_ids)}
        c6_order = {index: position for position, index in enumerate(c6_ids)}
        # Joined on pool index on both sides, never on position: the two runs
        # order their directions independently, and a positional join would pair
        # the wrong direction's geometry with the wrong direction's outcome.
        outcome = np.array(
            [b_effect[str(b_order[i])] for i in shared], dtype=np.float64
        )
        measure = np.array(
            [fit["delta_mse_linear_minus_quadratic"][c6_order[i]] for i in shared],
            dtype=np.float64,
        )
        variance = np.array(
            [variances[c6_order[i]] for i in shared], dtype=np.float64
        )
        if not np.isfinite(measure).all() or not np.isfinite(outcome).all():
            raise ValueError("the join produced non-finite values")
        if (variance <= 0.0).any():
            raise ValueError("log-variance requires a positive projected variance")
        design = np.stack([np.ones_like(measure), measure, np.log(variance)], axis=1)
        beta, *_ = np.linalg.lstsq(design, outcome, rcond=None)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws_rho, draws_beta = [], []
        for _ in range(2000):
            index = rng.integers(0, outcome.size, size=outcome.size)
            draws_rho.append(_spearman(measure[index], outcome[index]))
            fit, *_ = np.linalg.lstsq(design[index], outcome[index], rcond=None)
            draws_beta.append(fit[1])
        # Partial association: both sides residualized on log-variance.
        control = np.stack([np.ones_like(variance), np.log(variance)], axis=1)
        measure_residual = measure - control @ np.linalg.lstsq(
            control, measure, rcond=None
        )[0]
        outcome_residual = outcome - control @ np.linalg.lstsq(
            control, outcome, rcond=None
        )[0]
        payload["direction_rows"] = [
            {
                "direction_pool_index": int(i),
                "delta_mse_linear_minus_quadratic": float(measure[k]),
                "matched_delta_nll": float(outcome[k]),
                "projected_variance": float(variance[k]),
            }
            for k, i in enumerate(shared)
        ]
        payload["statistics"] = {
            "spearman_rho": _spearman(measure, outcome),
            "partial_spearman_rho_after_log_variance": _spearman(
                measure_residual, outcome_residual
            ),
            "beta_0": float(beta[0]),
            "beta_2_log_variance": float(beta[2]),
            "beta_1": float(beta[1]),
            "beta_1_ci": [
                float(np.quantile(draws_beta, 0.025)),
                float(np.quantile(draws_beta, 0.975)),
            ],
            "rho_ci": [
                float(np.quantile(draws_rho, 0.025)),
                float(np.quantile(draws_rho, 0.975)),
            ],
            "n_directions": int(outcome.size),
        }
        lower, upper = payload["statistics"]["beta_1_ci"]
        payload["verdict"] = {
            "outcome": "ASSOCIATION_DETECTED"
            if lower > 0.0
            else "OPPOSITE_ASSOCIATION"
            if upper < 0.0
            else "NO_ASSOCIATION",
            "statistics_computed": True,
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "direction_join.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["verdict"], indent=2))


if __name__ == "__main__":
    main()
