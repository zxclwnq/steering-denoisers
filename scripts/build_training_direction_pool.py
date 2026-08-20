"""Build the training-only SAE direction pool for the conditional flow prior.

The pool is defined by the frozen selection rule, not by any semantic judgement:

  1. candidate universe = SAE features that have a gpt-4o-mini Neuronpedia description
     and whose firing rate on the corrected `dev` corpus split lies in [1e-4, 2e-2];
  2. order = the frozen BLAKE2b priority ordering, seed 20260807, the same ordering the
     concept directions were drawn from (vendored in data/frozen_selection/);
  3. training-only pool = every candidate at rank >= 256.

Ranks below the floor are sliced off and never inspected, printed, or written. Every
accepted concept direction sits below rank 64 in this ordering, so the pool is disjoint
from both evaluation splits without any protected manifest being opened. This script
reads nothing under configs/protected/ and nothing that enumerates an evaluation split.

Run:

    uv run python scripts/build_training_direction_pool.py \
        --out data/direction_pools/training_only_rank256_v1.pt
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from interp.conditional_flow import (
    MIN_TRAINING_RANK,
    load_training_direction_pool,
    save_direction_pool,
)
from interp.phase_b import FrozenSAESpec, load_frozen_sae

FROZEN_SELECTION = Path("data/frozen_selection")
SAE_SNAPSHOT = Path(
    "/home/oleg/.cache/huggingface/hub/models--jbloom--GPT2-Small-SAEs-Reformatted"
    "/snapshots/57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9/blocks.7.hook_resid_pre"
)

# The frozen candidate rule (configs/steering_vectors_v2.yaml provenance header).
LABELLER = "gpt-4o-mini"
FIRING_RATE_RANGE = (1e-4, 2e-2)
SELECTION_SEED = 20260807
EXPECTED_UNIVERSE = 23610

# Identity of the frozen SAE, already pinned in configs/flow_phase_b_dev_v1.yaml.
SAE_SPEC = FrozenSAESpec(
    release="gpt2-small-res-jb",
    repo_id="jbloom/GPT2-Small-SAEs-Reformatted",
    revision="57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9",
    neuronpedia_id="gpt2-small/7-res-jb",
    model_name="gpt2-small",
    hook="blocks.7.hook_resid_pre",
    d_in=768,
    d_sae=24576,
    config_filename="cfg.json",
    weights_filename="sae_weights.safetensors",
    config_sha256="93d39f5eefeb5c254bf45c871fddc9527619d3626eeb0bd015e5f7330945f88e",
    weights_sha256="47bfb75008fdd7ebf068044c0c3a212606aaa3f5dc05f1d1a7cffe502002c0b6",
)

# The frozen direction convention, recorded alongside the SAE identity.
SAE_DIRECTION_CONVENTION = "unit_normalized_W_dec_row"


def _load_stable_order(selection_py: Path):
    """Load the vendored ordering rule without colliding with this repo's package."""

    spec = importlib.util.spec_from_file_location("frozen_selection_rule", selection_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SELECTION_SEED != SELECTION_SEED:
        raise ValueError(
            f"vendored selection seed {module.SELECTION_SEED} != frozen {SELECTION_SEED}"
        )
    return module.stable_order


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_universe(explanations: Path, stats_path: Path) -> list[int]:
    """The frozen candidate pool: described features alive in the firing-rate band."""

    described: set[int] = set()
    for line in explanations.open():
        row = json.loads(line)
        if row["explanationModelName"] == LABELLER:
            described.add(int(row["index"]))
    stats = np.load(stats_path, allow_pickle=True)
    rate = stats["firing_rate"]
    low, high = FIRING_RATE_RANGE
    return [feature for feature in sorted(described) if low <= rate[feature] <= high]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frozen-selection", type=Path, default=FROZEN_SELECTION)
    parser.add_argument("--sae-dir", type=Path, default=SAE_SNAPSHOT)
    parser.add_argument("--min-rank", type=int, default=MIN_TRAINING_RANK)
    args = parser.parse_args()

    provenance_record = json.loads((args.frozen_selection / "PROVENANCE.json").read_text())
    if provenance_record.get("protected_inputs_used") is not False:
        raise ValueError("vendored selection inputs are not marked protection-free")

    stable_order = _load_stable_order(args.frozen_selection / "selection.py")
    universe = candidate_universe(
        args.frozen_selection / "explanations_gpt2-small_7-res-jb.jsonl",
        args.frozen_selection / "feature_stats_7-res-jb_dev.npz",
    )
    if len(universe) != EXPECTED_UNIVERSE:
        raise ValueError(
            f"candidate universe is {len(universe)}, expected the frozen "
            f"{EXPECTED_UNIVERSE}; the vendored inputs are not the frozen ones"
        )
    ordered = stable_order(universe, SELECTION_SEED)
    # The only contact with the ordering: everything above the floor, nothing below it.
    training_features = ordered[args.min_rank :]
    training_ranks = tuple(range(args.min_rank, len(ordered)))

    sae = load_frozen_sae(SAE_SPEC, args.sae_dir, device="cpu")
    directions = torch.nn.functional.normalize(
        sae.w_dec[training_features].detach().float(), dim=-1
    )

    inputs = provenance_record["inputs"]
    provenance = save_direction_pool(
        args.out,
        directions,
        training_ranks,
        source=(
            f"sae:{SAE_SPEC.release}/{SAE_SPEC.hook}@{SAE_SPEC.revision}"
            f" weights_sha256:{SAE_SPEC.weights_sha256}"
            f" direction:{SAE_DIRECTION_CONVENTION}"
            f" universe:{len(universe)}/{SAE_SPEC.d_sae}"
            f" labeller:{LABELLER}"
            f" firing_rate:[{FIRING_RATE_RANGE[0]:g},{FIRING_RATE_RANGE[1]:g}]"
            f" explanations_sha256:{inputs['explanations']['sha256']}"
            f" feature_stats_sha256:{inputs['feature_stats']['sha256']}"
            f" selection_rule_sha256:{inputs['selection_rule']['sha256']}"
        ),
        selection="blake2b_priority_rank/interp.selection.stable_order@v2",
        selection_seed=SELECTION_SEED,
        min_rank=args.min_rank,
    )

    reloaded = load_training_direction_pool(args.out)
    if reloaded.provenance != provenance:
        raise ValueError("reloaded pool provenance differs from the saved record")
    norms = reloaded.directions.norm(dim=-1)
    summary = {
        **reloaded.identity(),
        "candidate_universe": len(universe),
        "discarded_below_floor": args.min_rank,
        "direction_norm_min": float(norms.min()),
        "direction_norm_max": float(norms.max()),
        "manifest_sha256": _file_sha256(args.out),
        "manifest_path": str(args.out),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
