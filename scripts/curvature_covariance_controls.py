"""C6: run the covariance controls on the Experiment C activations.

No training, no language model, no held-out data, and no frozen C artifact is
rewritten. This reads the same validation activations, the same 32 directions and
the same binning C used, and asks how much of the curvature survives once the
anisotropy of the residual stream is controlled for.

Six controls, all frozen in `docs/EXPERIMENT_C6_PROTOCOL.md` before any number:

* C6.1 the covariance-predicted linear direction `Sigma v`;
* C6.2 held-out linear versus quadratic conditional model, plus residual
  conditional means (the main test);
* C6.3 a covariance-matched Gaussian surrogate, whose conditional mean is linear
  by construction;
* C6.4 covariance-matched random directions, replacing the old unmatched null;
* C6.5 the rising tail-versus-centre profile against that matched null;
* C6.6 PC1..PC8, to show how the pipeline behaves on variance-selected axes.

    uv run python scripts/curvature_covariance_controls.py \
        --activation-dir /workspace/data/fineweb_activations \
        --token-cache-dir /workspace/data/fineweb_token_cache \
        --name resid7_fw_val_1024k_v1 \
        --pool data/direction_pools/training_only_rank256_v1.pt \
        --out-dir /workspace/results/curvature_c6_covariance_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from interp.conditional_flow import load_training_direction_pool
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
from interp.curvature import (
    CURVATURE_SPEC,
    _direction_shortfall,
    bin_indices,
    bin_means,
    bootstrap_direction_mean,
    direction_curvature,
    secant_geometry,
    spec_payload,
)
from interp.provenance import source_revision
from interp.tangent_eval import load_validated_evaluation_bundle, require_fresh_output_dir

SPEC = CURVATURE_SPEC
C6 = COVARIANCE_CONTROL_SPEC
PER_SEQ = 127
VAL_FRACTION = 0.05
SPLIT_SEED = 20260807
D_MODEL = 768
CENTRAL_SECANT = 2
UPPER_TAIL_SECANT = 4


def _select_rows(n_available: int) -> np.ndarray:
    rng = np.random.default_rng(SPEC.row_seed)
    return np.sort(rng.choice(n_available, size=SPEC.n_rows, replace=False))


def _paired_difference(left: np.ndarray, right: np.ndarray) -> dict:
    """Direction-clustered interval on a paired per-direction difference."""

    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    usable = difference[np.isfinite(difference)]
    if usable.size < 2:
        return {"usable": False, "n_directions": int(usable.size)}
    interval = bootstrap_direction_mean(usable, spec=SPEC)
    return {"usable": True, **interval}


def _pooled(records: list[dict], field: str) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.array(
            [np.nanmean(np.asarray(r[field], dtype=np.float64)) for r in records]
        )


def _profile_contrast(records: list[dict]) -> np.ndarray:
    """Tail minus centre of the cos(d_k, v) profile, per direction."""

    out = []
    for record in records:
        profile = np.asarray(record["cos_secant_direction"], dtype=np.float64)
        out.append(profile[UPPER_TAIL_SECANT] - profile[CENTRAL_SECANT])
    return np.array(out, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-dir", required=True, type=Path)
    parser.add_argument("--token-cache-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--hook", default="blocks.7.hook_resid_pre")
    parser.add_argument("--overwrite-debug-mode", action="store_true")
    args = parser.parse_args()

    require_fresh_output_dir(args.out_dir, overwrite_debug=args.overwrite_debug_mode)

    bundle = load_validated_evaluation_bundle(
        args.name, args.activation_dir, args.token_cache_dir,
        hook=args.hook, per_seq=PER_SEQ, val_fraction=VAL_FRACTION,
        split_seed=SPLIT_SEED, d_model=D_MODEL,
    )
    rows = _select_rows(bundle.activations.shape[0])
    activations = np.array(bundle.activations[rows], dtype=np.float64)
    sequence = rows // PER_SEQ
    print(f"rows {activations.shape}, sequences {np.unique(sequence).size}", flush=True)

    pool = load_training_direction_pool(args.pool).to(dtype=torch.float32)
    picked = np.sort(
        np.random.default_rng(SPEC.direction_seed).choice(
            len(pool), size=SPEC.n_directions, replace=False
        )
    )
    directions = pool.directions[picked].double().cpu().numpy()
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    mean = activations.mean(axis=0)
    centred = activations - mean
    covariance = (centred.T @ centred) / (activations.shape[0] - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    print("covariance estimated", flush=True)

    # ---- C6.1 the covariance-predicted linear direction -------------------
    linear_directions = [covariance_linear_direction(covariance, v) for v in directions]
    real_records = [
        direction_curvature(activations, v, sequence, spec=SPEC, seed_offset=i)
        for i, v in enumerate(directions)
    ]
    cos_with_sigma_v = []
    for v, linear in zip(directions, linear_directions, strict=True):
        # Same bins as C -- binning uses the coordinate v'h, which Sigma v does
        # not change; only the direction the secants are measured against does.
        assignment = bin_indices(activations @ v, SPEC)
        mu, _ = bin_means(activations, assignment, SPEC)
        cos_with_sigma_v.append(
            np.asarray(
                secant_geometry(mu, linear["unit"])["cos_secant_direction"], dtype=np.float64
            )
        )
    cos_with_sigma_v = np.stack(cos_with_sigma_v)
    print("C6.1 done", flush=True)

    # ---- C6.2 held-out linear vs quadratic --------------------------------
    train, test = split_sequences(sequence, C6.split_seed)
    fits, residuals = [], []
    for v in directions:
        fit = conditional_fit_comparison(activations, v, train, test, spec=C6)
        assignment = bin_indices(activations @ v, SPEC)
        residual = residual_conditional_means(
            activations, v, train, test, assignment, SPEC.n_bins, SPEC.min_bin_rows
        )
        fits.append(fit)
        residuals.append(residual)
    delta_mse = np.array([f["delta_mse_linear_minus_quadratic"] for f in fits])
    relative = np.array([f["relative_improvement"] for f in fits])
    ratios = [
        [x for x in r["residual_to_secant_ratio"] if x is not None] for r in residuals
    ]
    mean_ratio = np.array([np.mean(x) if x else np.nan for x in ratios])
    print("C6.2 done", flush=True)

    # ---- C6.3 covariance-matched Gaussian surrogate -----------------------
    synthetic, surrogate_receipt = gaussian_surrogate(
        mean, covariance, activations.shape[0], C6.surrogate_seed
    )
    surrogate_records = [
        direction_curvature(synthetic, v, sequence, spec=SPEC, seed_offset=i)
        for i, v in enumerate(directions)
    ]
    print("C6.3 done", flush=True)

    # ---- C6.4 covariance-matched random directions ------------------------
    candidate_rng = np.random.default_rng(C6.candidate_seed)
    candidates = candidate_rng.normal(size=(C6.n_candidates, D_MODEL))
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
    matched = match_directions(covariance, directions, candidates)
    matched_records = [
        direction_curvature(activations, v, sequence, spec=SPEC, seed_offset=500 + i)
        for i, v in enumerate(matched["directions"])
    ]
    # The historical unmatched control, kept for continuity only.
    unmatched = candidates[: SPEC.n_directions]
    unmatched_records = [
        direction_curvature(activations, v, sequence, spec=SPEC, seed_offset=900 + i)
        for i, v in enumerate(unmatched)
    ]
    print("C6.4 done", flush=True)

    # ---- C6.6 PCA sanity control ------------------------------------------
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    pcs = vt[: C6.n_principal_components]
    pc_records = [
        direction_curvature(activations, v, sequence, spec=SPEC, seed_offset=700 + i)
        for i, v in enumerate(pcs)
    ]
    pc_stats = covariance_coordinates(covariance, pcs)
    print("C6.6 done", flush=True)

    # ---- comparisons -------------------------------------------------------
    real_shortfall = np.array([_direction_shortfall(r) for r in real_records])
    surrogate_shortfall = np.array([_direction_shortfall(r) for r in surrogate_records])
    matched_shortfall = np.array([_direction_shortfall(r) for r in matched_records])
    unmatched_shortfall = np.array([_direction_shortfall(r) for r in unmatched_records])

    real_cos = _pooled(real_records, "cos_consecutive_secants")
    surrogate_cos = _pooled(surrogate_records, "cos_consecutive_secants")
    matched_cos = _pooled(matched_records, "cos_consecutive_secants")

    real_minus_gaussian = _paired_difference(real_shortfall, surrogate_shortfall)
    real_minus_matched = _paired_difference(real_shortfall, matched_shortfall)
    quadratic = bootstrap_direction_mean(delta_mse, spec=SPEC)
    quadratic["usable"] = True
    profile_difference = _paired_difference(
        _profile_contrast(real_records), _profile_contrast(matched_records)
    )
    residual_present = bool(np.nanmean(mean_ratio) > C6.min_residual_norm_ratio)

    verdict = covariance_verdict(
        real_minus_gaussian, real_minus_matched, quadratic, residual_present
    )

    concept_stats = covariance_coordinates(covariance, directions)
    random_stats = covariance_coordinates(covariance, unmatched)
    payload = {
        "experiment": C6.version,
        "class": "post_hoc covariance controls",
        "preregistered": False,
        "protocol": "docs/EXPERIMENT_C6_PROTOCOL.md",
        "question": (
            "how much of the Experiment C curvature survives a control for "
            "second-order covariance geometry and finite-sample conditioning?"
        ),
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "debug_mode": bool(args.overwrite_debug_mode),
        "validation_artifact": bundle.identity,
        "direction_pool": pool.identity(),
        "direction_pool_indices": picked.tolist(),
        "curvature_spec": spec_payload(SPEC),
        "c6_spec": C6.payload(),
        "n_rows": int(activations.shape[0]),
        "n_sequences": int(np.unique(sequence).size),
        "covariance": {
            "estimator": "empirical, full 768x768, float64",
            "total_variance": float(np.trace(covariance)),
            "top_eigenvalues": eigenvalues[:8].tolist(),
            "condition_number": float(eigenvalues[0] / max(eigenvalues[-1], 1e-30)),
        },
        "c6_1_covariance_predicted_direction": {
            "projected_variance": {
                "concept": concept_stats["projected_variance"].tolist(),
                "random_unmatched": random_stats["projected_variance"].tolist(),
                "principal_components": pc_stats["projected_variance"].tolist(),
            },
            "cos_v_sigma_v": {
                "concept": concept_stats["cos_v_sigma_v"].tolist(),
                "random_unmatched": random_stats["cos_v_sigma_v"].tolist(),
                "principal_components": pc_stats["cos_v_sigma_v"].tolist(),
            },
            "cos_dk_sigma_v_by_rung": np.nanmean(cos_with_sigma_v, axis=0).tolist(),
            "cos_dk_sigma_v_pooled": float(np.nanmean(cos_with_sigma_v)),
            "cos_dk_v_pooled": float(np.nanmean(_pooled(real_records, "cos_secant_direction"))),
            "note": (
                "descriptive only: a high cos(d_k, Sigma v) is not evidence of "
                "causalness or of intervention quality"
            ),
        },
        "c6_2_held_out_conditional_fit": {
            "split": "by sequence, seed 20260920",
            "n_train_rows": int(train.sum()),
            "n_test_rows": int(test.sum()),
            "mse_linear": [f["mse_degree1"] for f in fits],
            "mse_quadratic": [f["mse_degree2"] for f in fits],
            "delta_mse_linear_minus_quadratic": delta_mse.tolist(),
            "relative_improvement": relative.tolist(),
            "delta_mse_interval": quadratic,
            "relative_improvement_interval": bootstrap_direction_mean(relative, spec=SPEC),
            "residual_to_secant_ratio_mean": mean_ratio.tolist(),
            "residual_structure_present": residual_present,
            "residual_threshold": C6.min_residual_norm_ratio,
        },
        "c6_3_gaussian_surrogate": {
            "receipt": surrogate_receipt,
            "real_shortfall": real_shortfall.tolist(),
            "surrogate_shortfall": surrogate_shortfall.tolist(),
            "real_shortfall_mean": float(np.nanmean(real_shortfall)),
            "surrogate_shortfall_mean": float(np.nanmean(surrogate_shortfall)),
            "real_cos_consecutive_secants_mean": float(np.nanmean(real_cos)),
            "surrogate_cos_consecutive_secants_mean": float(np.nanmean(surrogate_cos)),
            "real_minus_surrogate_shortfall": real_minus_gaussian,
            "caveat": (
                "surrogate rows are independent while real rows within a sequence "
                "are correlated, so the surrogate has more effective independence "
                "at equal row count; the shortfall below the split-half ceiling is "
                "primary precisely because the ceiling absorbs the noise level"
            ),
        },
        "c6_4_matched_random_directions": {
            "rule": matched["rule"],
            "n_candidates": matched["n_candidates"],
            "balance": matched["balance"],
            "concept_shortfall_mean": float(np.nanmean(real_shortfall)),
            "matched_shortfall_mean": float(np.nanmean(matched_shortfall)),
            "unmatched_shortfall_mean": float(np.nanmean(unmatched_shortfall)),
            "concept_cos_consecutive_secants_mean": float(np.nanmean(real_cos)),
            "matched_cos_consecutive_secants_mean": float(np.nanmean(matched_cos)),
            "concept_minus_matched_shortfall": real_minus_matched,
            "concept_minus_unmatched_shortfall": _paired_difference(
                real_shortfall, unmatched_shortfall
            ),
        },
        "c6_5_profile_contrast": {
            "definition": "cos(d_k, v) at secant 4 minus secant 2, unchanged from C",
            "concept_mean": float(np.nanmean(_profile_contrast(real_records))),
            "matched_null_mean": float(np.nanmean(_profile_contrast(matched_records))),
            "difference": profile_difference,
        },
        "c6_6_principal_components": [
            {
                "component": index,
                "projected_variance": float(pc_stats["projected_variance"][index]),
                "cos_v_sigma_v": float(pc_stats["cos_v_sigma_v"][index]),
                "shortfall_below_ceiling": _direction_shortfall(record),
                "mean_cos_consecutive_secants": float(
                    np.nanmean(np.asarray(record["cos_consecutive_secants"], dtype=np.float64))
                ),
                "mean_cos_secant_direction": float(
                    np.nanmean(np.asarray(record["cos_secant_direction"], dtype=np.float64))
                ),
            }
            for index, record in enumerate(pc_records)
        ],
        "verdict": verdict,
        "rewrote_frozen_c_artifacts": False,
        "dev_vectors_accessed": False,
        "held_out_accessed": False,
        "llm_judge_used": False,
        "trained_anything": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "covariance_controls.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(
        args.out_dir / "raw_rows.npz",
        real_shortfall=real_shortfall,
        surrogate_shortfall=surrogate_shortfall,
        matched_shortfall=matched_shortfall,
        unmatched_shortfall=unmatched_shortfall,
        delta_mse=delta_mse,
        relative_improvement=relative,
        cos_dk_sigma_v=cos_with_sigma_v,
        concept_profile_contrast=_profile_contrast(real_records),
        matched_profile_contrast=_profile_contrast(matched_records),
        direction_pool_indices=picked,
        matched_indices=matched["indices"],
    )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
