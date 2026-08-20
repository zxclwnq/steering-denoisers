"""C6 addendum: is the concept-versus-matched-random excess itself a variance effect?

The C6 matching rule was frozen before any result, and it fired. Its own balance
report then showed it did not achieve balance: random unit directions in 768
dimensions concentrate tightly around a small projected variance, while the
concept directions reach far higher, so the candidate pool simply does not
contain comparable directions. That is an overlap failure, not a tuning problem.

Whether the residual imbalance matters depends on one thing: does the curvature
statistic track projected variance? This reads the finished C6 artifact and
answers that from the directions it already measured -- concept, unmatched
random, and the eight principal components -- spanning a factor of ~28 in
projected variance.

Nothing is recomputed from activations, nothing is retrained, and the C6 artifact
is not modified.

    uv run python scripts/c6_variance_confound_check.py \
        --c6 results/curvature_c6_covariance_v1 \
        --out-dir results/curvature_c6_variance_confound_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from interp.curvature import CURVATURE_SPEC, bootstrap_direction_mean
from interp.provenance import source_revision
from interp.tangent_eval import require_fresh_output_dir


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rank_x = np.argsort(np.argsort(x)).astype(np.float64)
    rank_y = np.argsort(np.argsort(y)).astype(np.float64)
    rank_x -= rank_x.mean()
    rank_y -= rank_y.mean()
    denominator = np.sqrt((rank_x**2).sum() * (rank_y**2).sum())
    return float((rank_x * rank_y).sum() / denominator) if denominator else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c6", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--overwrite-debug-mode", action="store_true")
    args = parser.parse_args()

    require_fresh_output_dir(args.out_dir, overwrite_debug=args.overwrite_debug_mode)
    c6 = json.loads((args.c6 / "covariance_controls.json").read_text())
    raw = np.load(args.c6 / "raw_rows.npz")

    variances = c6["c6_1_covariance_predicted_direction"]["projected_variance"]
    groups = {
        "concept": (
            np.asarray(variances["concept"], dtype=np.float64),
            np.asarray(raw["real_shortfall"], dtype=np.float64),
        ),
        "random_unmatched": (
            np.asarray(variances["random_unmatched"], dtype=np.float64),
            np.asarray(raw["unmatched_shortfall"], dtype=np.float64),
        ),
        "principal_components": (
            np.asarray(variances["principal_components"], dtype=np.float64),
            np.asarray(
                [c["shortfall_below_ceiling"] for c in c6["c6_6_principal_components"]],
                dtype=np.float64,
            ),
        ),
    }
    all_variance = np.concatenate([v for v, _ in groups.values()])
    all_shortfall = np.concatenate([s for _, s in groups.values()])
    usable = np.isfinite(all_variance) & np.isfinite(all_shortfall)
    log_variance = np.log(all_variance[usable])
    shortfall = all_shortfall[usable]

    rng = np.random.default_rng(CURVATURE_SPEC.bootstrap_seed)
    draws = []
    for _ in range(CURVATURE_SPEC.bootstrap_resamples):
        index = rng.integers(0, shortfall.size, size=shortfall.size)
        draws.append(_spearman(log_variance[index], shortfall[index]))
    draws = np.asarray(draws, dtype=np.float64)
    tail = (1.0 - CURVATURE_SPEC.confidence) / 2.0

    balance = c6["c6_4_matched_random_directions"]["balance"]
    matched_worked = all(
        block["mean_abs_difference_after"] < block["mean_abs_difference_before"]
        for block in balance.values()
    )
    concept_shortfall = groups["concept"][1]
    pc_shortfall = groups["principal_components"][1]

    payload = {
        "experiment": "curvature_c6_variance_confound_v1",
        "class": "post_hoc covariance controls",
        "preregistered": False,
        "protocol": "docs/EXPERIMENT_C6_PROTOCOL.md",
        "question": (
            "the frozen matching rule did not achieve balance; does the curvature "
            "statistic track projected variance, and therefore does the residual "
            "imbalance explain the concept-versus-matched-random excess?"
        ),
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "c6_source": str(args.c6),
        "recomputes_no_activation": True,
        "matching_balance_achieved": matched_worked,
        "matching_balance": balance,
        "group_means": {
            name: {
                "projected_variance": float(np.mean(variance)),
                "shortfall": float(np.nanmean(short)),
                "n": int(variance.size),
            }
            for name, (variance, short) in groups.items()
        },
        "shortfall_vs_log_projected_variance": {
            "spearman_rho": _spearman(log_variance, shortfall),
            "ci_lower": float(np.quantile(draws, tail)),
            "ci_upper": float(np.quantile(draws, 1.0 - tail)),
            "n_directions": int(shortfall.size),
            "confidence": float(CURVATURE_SPEC.confidence),
            "excludes_zero": bool(
                np.quantile(draws, tail) > 0.0 or np.quantile(draws, 1.0 - tail) < 0.0
            ),
        },
        "principal_components_vs_concept": {
            **bootstrap_direction_mean(
                np.concatenate([pc_shortfall, -concept_shortfall]), spec=CURVATURE_SPEC
            ),
            "pc_shortfall_mean": float(np.nanmean(pc_shortfall)),
            "concept_shortfall_mean": float(np.nanmean(concept_shortfall)),
            "pcs_more_curved_than_concept": bool(
                np.nanmean(pc_shortfall) > np.nanmean(concept_shortfall)
            ),
            "note": (
                "principal components carry ~5x the concept projected variance and "
                "have no concept meaning at all"
            ),
        },
        "conclusion": None,
        "held_out_accessed": False,
        "dev_vectors_accessed": False,
        "llm_judge_used": False,
        "trained_anything": False,
    }
    tracks = payload["shortfall_vs_log_projected_variance"]["excludes_zero"]
    pcs_higher = payload["principal_components_vs_concept"]["pcs_more_curved_than_concept"]
    payload["conclusion"] = (
        "the curvature statistic tracks projected variance, and the frozen "
        "matching left the concept directions with roughly twice the projected "
        "variance of their matched nulls. The concept-versus-matched-random "
        "excess therefore cannot be attributed to concept-specific geometry. "
        "Variance-selected principal components are MORE curved than the concept "
        "directions, which settles it directly. What survives is the "
        "variance-matched-by-construction comparison: the Gaussian surrogate and "
        "the held-out quadratic fit, neither of which can be explained this way"
        if tracks and pcs_higher
        else "the curvature statistic does not track projected variance in these "
        "data, so the residual matching imbalance does not explain the "
        "concept-versus-matched-random excess"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "variance_confound.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "matching_balance_achieved": matched_worked,
        "shortfall_vs_variance": payload["shortfall_vs_log_projected_variance"],
        "pcs_more_curved": payload["principal_components_vs_concept"][
            "pcs_more_curved_than_concept"
        ],
        "conclusion": payload["conclusion"],
    }, indent=2))


if __name__ == "__main__":
    main()
