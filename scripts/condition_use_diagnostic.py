"""Run the frozen condition-use diagnostic on one conditional-flow checkpoint.

Concept-independent: frozen validation activations, training-only pool directions, no
DEV or held-out data of any kind. Writes one JSON result per checkpoint.

    uv run python scripts/condition_use_diagnostic.py \
        --checkpoint /workspace/checkpoints/<run>/step_010000.pt \
        --activation-dir /workspace/data/fineweb_activations \
        --name resid7_fw_val_1024k_v1 \
        --pool data/direction_pools/training_only_rank256_v1.pt \
        --out /workspace/results/condition_use/step_010000.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from interp.activations import activation_paths
from interp.condition_use import (
    FROZEN_SPEC,
    condition_use_passes,
    run_condition_use_diagnostic,
)
from interp.conditional_flow import ConditionalFlowMatcher, load_training_direction_pool
from interp.train_flow import load_flow_checkpoint


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--activation-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, checkpoint_meta, _ = load_flow_checkpoint(args.checkpoint, device)
    if not isinstance(model, ConditionalFlowMatcher):
        raise ValueError("condition-use diagnostic requires a conditional checkpoint")
    pool = load_training_direction_pool(args.pool).to(device=device, dtype=torch.float32)

    array_path, meta_path, _ = activation_paths(args.name, args.activation_dir)
    metadata = json.loads(meta_path.read_text())
    if metadata.get("split") != "val" or not metadata.get("bos_dropped"):
        raise ValueError("the diagnostic runs on the BOS-dropped validation artifact")
    activations = np.load(array_path, mmap_mode="r")

    summary = run_condition_use_diagnostic(
        model, activations, pool, spec=FROZEN_SPEC, device=device
    )
    summary["checkpoint"] = str(args.checkpoint)
    summary["checkpoint_sha256"] = _file_sha256(args.checkpoint)
    summary["checkpoint_step"] = checkpoint_meta.get("step")
    summary["experiment_id"] = checkpoint_meta.get("experiment_id")
    summary["config_fingerprint"] = checkpoint_meta.get("config_fingerprint")
    summary["val_flow_mse_at_checkpoint"] = checkpoint_meta.get("val_flow_mse")
    summary["validation_activations"] = {
        "name": args.name,
        "shape": metadata["shape"],
        "token_cache_sha256": metadata.get("token_cache_sha256"),
    }
    summary["mechanical_pass"] = condition_use_passes(summary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_t"}, indent=2))
    for row in summary["by_t"]:
        print(
            f"t={row['t']:.2f} "
            f"mse correct={row['flow_mse_correct']:.5f} "
            f"shuf_c={row['flow_mse_shuffled_target']:.5f} "
            f"shuf_v={row['flow_mse_shuffled_direction']:.5f} "
            f"gap_c={row['gap_shuffled_target']:+.5f} "
            f"gap_v={row['gap_shuffled_direction']:+.5f} "
            f"coord_err={row['coordinate_abs_error_correct']:.3f} "
            f"swap_rel={row['relative_swap_sensitivity_shuffled_target']:.4f}"
        )


if __name__ == "__main__":
    main()
