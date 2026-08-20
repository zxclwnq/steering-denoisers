"""Experiment C: does the natural concept trajectory bend away from a fixed v?

For each training-only direction, bin frozen validation activations by their own
coordinate ``<h, v>``, take the secants between consecutive bin means, and ask
whether those secants keep pointing the same way as concept strength increases.
A conditional mean that is affine in the coordinate gives secants that never
rotate; systematic rotation is curvature.

Every estimate is reported beside two calibrations that make it interpretable: a
shuffled-label null and a split-half reliability ceiling. Random unit directions
are computed as a reference, and the pooled shortfall carries a
direction-clustered bootstrap interval.

This script trains nothing, evaluates no language model, and proposes no
intervention. It needs only the validation activation artifact and the
training-only direction pool.

Concept-independent throughout: no DEV vectors, no held-out data, no LLM judge.

    uv run python scripts/curvature_diagnostic.py \
        --activation-dir /workspace/data/fineweb_activations \
        --token-cache-dir /workspace/data/fineweb_token_cache \
        --name resid7_fw_val_1024k_v1 \
        --pool data/direction_pools/training_only_rank256_v1.pt \
        --out-dir /workspace/results/curvature_c_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from interp.conditional_flow import load_training_direction_pool
from interp.curvature import (
    CURVATURE_SPEC,
    bootstrap_shortfall,
    curvature_verdict,
    direction_curvature,
    pooled_curvature,
    spec_payload,
)
from interp.provenance import source_revision
from interp.tangent_eval import (
    load_validated_evaluation_bundle,
    require_fresh_output_dir,
)

SPEC = CURVATURE_SPEC
PER_SEQ = 127
VAL_FRACTION = 0.05
SPLIT_SEED = 20260807
D_MODEL = 768


def _select_rows(n_available: int) -> np.ndarray:
    """Frozen evaluation rows, drawn from the whole validation artifact.

    The population is the entire `resid7_fw_val_1024k_v1` artifact, not its
    internal 5% document split. That artifact is already concept-independent
    held-back data -- corpus documents 0..40000, disjoint from the training
    documents 100000+ -- and it is the same population
    `interp.natural_support.select_reference_rows` draws from when it estimates
    natural coordinate quantiles. Using the internal 5% split instead would leave
    only ~51k rows, too few for six coordinate bins across 32 directions.
    """

    if n_available < SPEC.n_rows:
        raise ValueError(
            f"validation artifact has {n_available} rows, fewer than the frozen "
            f"{SPEC.n_rows}"
        )
    rng = np.random.default_rng(SPEC.row_seed)
    return np.sort(rng.choice(n_available, size=SPEC.n_rows, replace=False))


def _select_directions(
    n_pool: int, indices_from: Path | None
) -> tuple[np.ndarray, dict[str, object]]:
    """Which pool directions to measure, and an honest record of who chose them.

    The frozen plan draws its own directions from ``SPEC.direction_seed``. Passing
    ``--direction-indices-from`` instead reuses the direction set of another
    result, which is the only way to ask whether curvature relates to that
    experiment's per-direction outcomes -- and is an operator choice the frozen
    plan never made. The returned provenance says which of the two happened, so a
    reader can never mistake a reused selection for the preregistered draw.
    """

    if indices_from is None:
        rng = np.random.default_rng(SPEC.direction_seed)
        picked = np.sort(rng.choice(n_pool, size=SPEC.n_directions, replace=False))
        return picked, {
            "mode": "frozen_seeded_draw",
            "preregistered": True,
            "direction_seed": SPEC.direction_seed,
        }

    payload = json.loads(Path(indices_from).read_text())
    if "direction_pool_indices" not in payload:
        raise ValueError(
            f"{indices_from} carries no direction_pool_indices to reuse"
        )
    picked = np.unique(np.asarray(payload["direction_pool_indices"], dtype=np.int64))
    if picked.size == 0 or picked.min() < 0 or picked.max() >= n_pool:
        raise ValueError(f"{indices_from} names directions outside the pool")
    return picked, {
        "mode": "supplied_indices",
        "preregistered": False,
        "source": str(indices_from),
        "source_experiment": payload.get("experiment"),
        "note": (
            "exploratory: these directions were chosen to match another "
            "experiment's set, not drawn by the frozen plan; the primary C "
            "result is the frozen_seeded_draw run and this does not replace it"
        ),
    }


def _random_directions(width: int) -> np.ndarray:
    """Control axes: unit vectors with no concept meaning at all."""

    rng = np.random.default_rng(SPEC.random_direction_seed)
    raw = rng.normal(size=(SPEC.n_random_directions, width))
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-dir", required=True, type=Path)
    parser.add_argument("--token-cache-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--overwrite-debug-mode", action="store_true",
                        help="permit writing into a non-empty result directory; "
                             "marks the run as a discardable debug run")
    parser.add_argument("--hook", default="blocks.7.hook_resid_pre")
    parser.add_argument(
        "--direction-indices-from", default=None, type=Path,
        help="reuse the direction_pool_indices of another result instead of the "
             "frozen seeded draw; marks the run exploratory and non-preregistered",
    )
    args = parser.parse_args()

    require_fresh_output_dir(args.out_dir, overwrite_debug=args.overwrite_debug_mode)

    bundle = load_validated_evaluation_bundle(
        args.name, args.activation_dir, args.token_cache_dir,
        hook=args.hook, per_seq=PER_SEQ, val_fraction=VAL_FRACTION,
        split_seed=SPLIT_SEED, d_model=D_MODEL,
    )
    rows = _select_rows(bundle.activations.shape[0])
    activations = np.array(bundle.activations[rows], dtype=np.float64)
    # Sequence identity is the resampling and splitting unit: activations from one
    # document are not independent observations.
    sequence = rows // PER_SEQ

    pool = load_training_direction_pool(args.pool).to(dtype=torch.float32)
    picked, direction_selection = _select_directions(
        len(pool), args.direction_indices_from
    )
    directions = pool.directions[picked].double().cpu().numpy()

    records = [
        direction_curvature(activations, directions[index], sequence,
                            spec=SPEC, seed_offset=index)
        for index in range(directions.shape[0])
    ]
    pooled = pooled_curvature(records, SPEC)
    interval = bootstrap_shortfall(records, SPEC)

    random_records = [
        direction_curvature(activations, axis, sequence, spec=SPEC,
                            seed_offset=1000 + index)
        for index, axis in enumerate(_random_directions(activations.shape[1]))
    ]
    random_pooled = pooled_curvature(random_records, SPEC)
    random_interval = bootstrap_shortfall(random_records, SPEC)

    verdict = curvature_verdict(pooled, random_pooled, interval=interval)

    payload = {
        "experiment": SPEC.version,
        "question": (
            "within the natural coordinate range, does the local direction of the "
            "concept trajectory rotate away from the fixed SAE direction?"
        ),
        "primary_statistic": (
            "mean(split-half pair ceiling - cos(d_k, d_k+1)), pooled over "
            "directions; positive beyond the margin = curvature"
        ),
        "spec": spec_payload(SPEC),
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "debug_mode": bool(args.overwrite_debug_mode),
        "validation_artifact": bundle.identity,
        "direction_pool": pool.identity(),
        "direction_pool_indices": picked.tolist(),
        "direction_selection": direction_selection,
        "n_rows": int(activations.shape[0]),
        "n_sequences": int(np.unique(sequence).size),
        "pooled": pooled,
        "shortfall_interval": interval,
        "random_direction_control": {
            "pooled": random_pooled,
            "shortfall_interval": random_interval,
            # Kept per direction, exactly as for the real axes, so the reference
            # can be given the same intervals rather than only a point estimate.
            "per_direction": random_records,
            "note": (
                "random unit axes are not concept directions; this shows what the "
                "statistics look like on an arbitrary axis and is a reference, not "
                "a null hypothesis test"
            ),
        },
        "per_direction": records,
        "verdict": verdict,
        "dev_vectors_accessed": False,
        "held_out_accessed": False,
        "llm_judge_used": False,
        "trained_anything": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "curvature_diagnostic.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    np.savez_compressed(
        args.out_dir / "raw_rows.npz",
        orthogonal_drift=np.array([r["orthogonal_drift"] for r in records]),
        cos_secant_direction=np.array([r["cos_secant_direction"] for r in records]),
        cos_consecutive_secants=np.array(
            [r["cos_consecutive_secants"] for r in records]
        ),
        split_half_pair_ceiling=np.array(
            [r["split_half_pair_ceiling"] for r in records]
        ),
        shuffled_cos_consecutive_secants=np.array(
            [r["shuffled_cos_consecutive_secants"] for r in records]
        ),
        bin_coordinate_means=np.array([r["bin_coordinate_means"] for r in records]),
        direction_pool_indices=picked,
        **{
            f"random_{field}": np.array([r[field] for r in random_records])
            for field in (
                "orthogonal_drift",
                "cos_secant_direction",
                "cos_consecutive_secants",
                "split_half_pair_ceiling",
                "shuffled_cos_consecutive_secants",
                "bin_coordinate_means",
            )
        },
    )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
