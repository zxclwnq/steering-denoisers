"""Concept-independent C1 pre-check: is a requested coordinate actually realised?

PHASE_B_CONDITIONAL_PROTOCOL.md section 13 makes C1 the gate: if the conditional
prior does not realise the requested coordinate, C2-C4 are uninterpretable and
the DEV sweep should not run. This script tests C1 on frozen validation
activations and training-only pool directions, so it touches no DEV vector and
no held-out data and needs no protocol freeze.

    uv run python scripts/conditional_c1_precheck.py \
        --checkpoint /workspace/checkpoints/<run>/best_step_249500.pt \
        --activation-dir /workspace/data/fineweb_activations \
        --name resid7_fw_val_1024k_v1 \
        --pool data/direction_pools/training_only_rank256_v1.pt \
        --out /workspace/results/c1_precheck.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from interp.activations import activation_paths
from interp.conditional_flow import conditional_clamp_steer, load_training_direction_pool
from interp.train_flow import load_flow_checkpoint

# Frozen sampling plan, deliberately mirroring condition_use_v1 so the two
# diagnostics describe the same rows and directions.
N_ROWS = 4096
N_DIRECTIONS = 64
ROW_SEED = 20260815
DIRECTION_SEED = 20260816
NOISE_SEED = 20260817
ACTIVATION_NORM_MEAN = 88.76
ALPHA_HAT = (0.1, 0.3, 0.5, 1.0)
T_START = (0.50, 0.75, 0.90)
NFE = (1, 3)
SEED_MODES = ("clean", "clamp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--activation-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, meta, _ = load_flow_checkpoint(args.checkpoint, device=device)
    model.eval()
    if not hasattr(model, "velocity_field"):
        raise ValueError("checkpoint is not a conditional flow model")

    array_path, meta_path, _ = activation_paths(args.name, args.activation_dir)
    metadata = json.loads(meta_path.read_text())
    if metadata.get("split") != "val" or not metadata.get("bos_dropped"):
        raise ValueError("the pre-check runs on the BOS-dropped validation artifact")
    activations = np.load(array_path, mmap_mode="r")
    rng = np.random.default_rng(ROW_SEED)
    rows = np.sort(rng.choice(activations.shape[0], size=N_ROWS, replace=False))
    h = torch.from_numpy(np.array(activations[rows], dtype=np.float32)).to(device)

    pool = load_training_direction_pool(args.pool).to(device=device, dtype=torch.float32)
    direction_rng = np.random.default_rng(DIRECTION_SEED)
    picked = np.sort(direction_rng.choice(len(pool), size=N_DIRECTIONS, replace=False))
    catalogue = pool.directions[picked].to(device=device, dtype=h.dtype)
    assignment = torch.from_numpy(
        direction_rng.integers(0, N_DIRECTIONS, size=N_ROWS)
    ).to(device)
    directions = catalogue[assignment]

    generator = torch.Generator(device="cpu").manual_seed(NOISE_SEED)
    noise = torch.randn(h.shape, generator=generator).to(device)

    digest = hashlib.sha256()
    digest.update(args.checkpoint.read_bytes())

    results = []
    for seed_mode in SEED_MODES:
        for alpha_hat in ALPHA_HAT:
            alpha = alpha_hat * ACTIVATION_NORM_MEAN
            for t_start in T_START:
                for nfe in NFE:
                    out = conditional_clamp_steer(
                        model,
                        h,
                        directions,
                        alpha=alpha,
                        mode="additive",
                        noise=noise,
                        t_start=t_start,
                        nfe=nfe,
                        seed_mode=seed_mode,
                    )
                    displacement = ((out.activation - h) * directions).sum(dim=-1)
                    results.append(
                        {
                            "seed_mode": seed_mode,
                            "alpha_hat": alpha_hat,
                            "alpha": alpha,
                            "t_start": t_start,
                            "nfe": nfe,
                            "r_retain_mean": float(
                                (displacement / alpha).double().mean()
                            ),
                            "r_retain_std": float((displacement / alpha).double().std()),
                            "coordinate_abs_error_mean": float(
                                out.coordinate_error.abs().double().mean()
                            ),
                            "clean_distance_mean": float(
                                (out.activation - h).norm(dim=-1).double().mean()
                            ),
                        }
                    )

    payload = {
        "diagnostic": "conditional_c1_precheck_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest.hexdigest(),
        "checkpoint_step": meta.get("step"),
        "experiment_id": meta.get("experiment_id"),
        "validation_activations": args.name,
        "direction_pool": pool.identity(),
        "n_rows": N_ROWS,
        "n_directions": N_DIRECTIONS,
        "alpha_convention": f"alpha = alpha_hat * {ACTIVATION_NORM_MEAN}",
        "held_out_accessed": False,
        "dev_vectors_accessed": False,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    for row in results:
        print(
            f"{row['seed_mode']:5s} a_hat={row['alpha_hat']:<4} t={row['t_start']:.2f} "
            f"nfe={row['nfe']} r_retain={row['r_retain_mean']:+.4f} "
            f"coord_err={row['coordinate_abs_error_mean']:8.3f}"
        )


if __name__ == "__main__":
    main()
