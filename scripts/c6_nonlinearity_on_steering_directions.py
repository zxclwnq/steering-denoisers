"""The C6.2 nonlinearity measure, computed for the directions Experiment B used.

C6 measured its nonlinearity on the Experiment C direction draw, which shares no
direction with the steering experiments. To ask whether that nonlinearity relates
to per-direction steering-repair failure, the measure has to exist for B's
directions, and it does not. This computes it, and nothing else.

Everything about the measurement is inherited unchanged from
`docs/EXPERIMENT_C6_PROTOCOL.md`: the same activation rows and row seed, the same
sequence-level train/test split and split seed, the same standardization, the
same degree-2 ceiling. **Only the direction set differs**, and it is not chosen
here either --- it is `natural_support_v1`'s frozen draw, the one Experiment B
inherits, so no direction is selected by anybody for this analysis.

This does not modify the C6 artifact and does not re-run any model. It is
`exploratory_post_hoc: true`.

    uv run python scripts/c6_nonlinearity_on_steering_directions.py \
        --activation-dir /workspace/data/fineweb_activations \
        --token-cache-dir /workspace/data/fineweb_token_cache \
        --name resid7_fw_val_1024k_v1 \
        --pool data/direction_pools/training_only_rank256_v1.pt \
        --out-dir /workspace/results/c6_nonlinearity_steering_directions_v1
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
    split_sequences,
)
from interp.curvature import CURVATURE_SPEC, spec_payload
from interp.natural_support import NATURAL_SUPPORT_SPEC, select_directions
from interp.provenance import source_revision
from interp.tangent_eval import load_validated_evaluation_bundle, require_fresh_output_dir

SPEC = CURVATURE_SPEC
C6 = COVARIANCE_CONTROL_SPEC
PER_SEQ = 127
VAL_FRACTION = 0.05
SPLIT_SEED = 20260807
D_MODEL = 768


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
    rng = np.random.default_rng(SPEC.row_seed)
    rows = np.sort(rng.choice(bundle.activations.shape[0], size=SPEC.n_rows, replace=False))
    activations = np.array(bundle.activations[rows], dtype=np.float64)
    sequence = rows // PER_SEQ
    print(f"rows {activations.shape}, sequences {np.unique(sequence).size}", flush=True)

    pool = load_training_direction_pool(args.pool).to(dtype=torch.float32)
    picked = select_directions(len(pool), NATURAL_SUPPORT_SPEC)
    directions = pool.directions[picked].double().cpu().numpy()
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    centred = activations - activations.mean(axis=0)
    covariance = (centred.T @ centred) / (activations.shape[0] - 1)
    stats = covariance_coordinates(covariance, directions)
    print("covariance estimated", flush=True)

    train, test = split_sequences(sequence, C6.split_seed)
    fits = [conditional_fit_comparison(activations, v, train, test, spec=C6)
            for v in directions]
    print("fits done", flush=True)

    payload = {
        "experiment": "c6_nonlinearity_steering_directions_v1",
        "class": "post_hoc covariance controls",
        "preregistered": False,
        "exploratory_post_hoc": True,
        "protocol": "docs/EXPERIMENT_C6_PROTOCOL.md (section 3, measure unchanged)",
        "question": (
            "the C6.2 held-out nonlinearity measure, for the directions Experiment "
            "B evaluated"
        ),
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "debug_mode": bool(args.overwrite_debug_mode),
        "validation_artifact": bundle.identity,
        "direction_pool": pool.identity(),
        "direction_pool_indices": picked.tolist(),
        "direction_selection": {
            "source": "natural_support_v1 frozen draw, inherited by Experiment B",
            "direction_seed": NATURAL_SUPPORT_SPEC.direction_seed,
            "chosen_for_this_analysis": False,
        },
        "curvature_spec": spec_payload(SPEC),
        "c6_spec": C6.payload(),
        "n_rows": int(activations.shape[0]),
        "n_train_rows": int(train.sum()),
        "n_test_rows": int(test.sum()),
        "mse_linear": [f["mse_degree1"] for f in fits],
        "mse_quadratic": [f["mse_degree2"] for f in fits],
        "delta_mse_linear_minus_quadratic": [
            f["delta_mse_linear_minus_quadratic"] for f in fits
        ],
        "relative_improvement": [f["relative_improvement"] for f in fits],
        "projected_variance": stats["projected_variance"].tolist(),
        "cos_v_sigma_v": stats["cos_v_sigma_v"].tolist(),
        "modified_frozen_artifacts": False,
        "dev_vectors_accessed": False,
        "held_out_accessed": False,
        "llm_judge_used": False,
        "trained_anything": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "nonlinearity.json").write_text(json.dumps(payload, indent=2) + "\n")
    delta = np.array(payload["delta_mse_linear_minus_quadratic"])
    print(json.dumps({
        "n_directions": len(picked),
        "delta_mse_mean": float(delta.mean()),
        "delta_mse_positive": int((delta > 0).sum()),
        "projected_variance_mean": float(np.mean(payload["projected_variance"])),
    }, indent=2))


if __name__ == "__main__":
    main()
