"""Constrained naturalization diagnostic: hard clamp vs constrained conditional flow.

Question: with the semantic coordinate held exactly, can the learned prior improve
the orthogonal components enough to beat an ordinary hard clamp?

Concept-independent. Frozen validation activations, training-only pool directions,
targets reused from the frozen natural_support_v1 quantile plan. No DEV vectors,
no held-out data, no LLM judge. Trains nothing.

    uv run python scripts/constrained_naturalization.py \
        --checkpoint /workspace/checkpoints/<run>/best_step_249500.pt \
        --activation-dir /workspace/data/fineweb_activations \
        --token-cache-dir /workspace/data/fineweb_token_cache \
        --name resid7_fw_val_1024k_v1 \
        --pool data/direction_pools/training_only_rank256_v1.pt \
        --out-dir /workspace/results/constrained_naturalization_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from interp.activations import activation_paths, file_sha256, make_split
from interp.conditional_flow import load_training_direction_pool
from interp.constrained_flow import (
    clamp_seed_flow,
    correction_decomposition,
    merge_decompositions,
)
from interp.functional import sequence_delta_lm, sequence_lm_losses
from interp.model import load_model, resolve_device
from interp.natural_support import (
    NATURAL_SUPPORT_SPEC,
    assign_directions,
    natural_coordinate_stats,
    select_directions,
    select_reference_rows,
    select_sequences,
    spec_payload,
    target_coordinates,
)
from interp.phase_a import paired_bootstrap_mean_ci
from interp.provenance import source_revision
from interp.train_flow import load_flow_checkpoint

SPEC = NATURAL_SUPPORT_SPEC
PER_SEQ = 127
VAL_FRACTION = 0.05
SPLIT_SEED = 20260807
# Frozen cheap grid: the point is whether condition-consistent seeding permits
# useful correction at LOW flow time, so t=0.75/0.90 are deliberately excluded.
T_START = (0.10, 0.25, 0.50)
NFE = (1, 3)
FLOW_ARMS = ("sdedit", "tangent", "projected")


class _ConstrainedTransform:
    """Substitution hook applying one constrained arm, recording geometry."""

    def __init__(
        self,
        flow,  # noqa: ANN001
        directions: torch.Tensor,
        c_target: torch.Tensor,
        noise: torch.Tensor,
        *,
        t_start: float,
        nfe: int,
        arm: str,
    ) -> None:
        self.flow = flow
        self.directions = directions
        self.c_target = c_target
        self.noise = noise
        self.t_start = t_start
        self.nfe = nfe
        self.arm = arm
        self.offset = 0
        self.records: list[dict[str, np.ndarray]] = []
        self.geometry_parts: list[dict[str, float]] = []
        self.evaluations = 0
        self.projections = 0

    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.ndim != 3:
            raise ValueError("transform expects [sequence, position, d_model]")
        batch, positions, width = activation.shape
        rows = slice(self.offset, self.offset + batch)
        v = self.directions[rows].to(device=activation.device, dtype=activation.dtype)
        noise = self.noise[rows].to(device=activation.device, dtype=activation.dtype)
        target = self.c_target[rows].to(device=activation.device, dtype=activation.dtype)

        flat = activation.reshape(-1, width)
        flat_noise = noise.reshape(-1, width)
        flat_v = v[:, None, :].expand(batch, positions, width).reshape(-1, width)
        flat_c = target[:, None].expand(batch, positions).reshape(-1, 1)

        out = clamp_seed_flow(
            self.flow, flat, flat_v, flat_c,
            noise=flat_noise, t_start=self.t_start, nfe=self.nfe, arm=self.arm,
        )
        transformed = out.activation
        realized = (transformed * flat_v).sum(dim=-1, keepdim=True)
        geometry = correction_decomposition(out.seed, transformed, flat_v)
        clean_norm = flat.norm(dim=-1)
        self.records.append(
            {
                "coordinate_abs_error": (realized - flat_c)
                .abs().squeeze(-1).double().cpu().numpy(),
                "c_target": flat_c.squeeze(-1).double().cpu().numpy(),
                "c_realised": realized.squeeze(-1).double().cpu().numpy(),
                "c0_clean": (flat * flat_v).sum(dim=-1).double().cpu().numpy(),
                "relative_l2_to_clean": (
                    (transformed - flat).norm(dim=-1) / clean_norm
                ).double().cpu().numpy(),
                "cosine_to_clean": torch.nn.functional.cosine_similarity(
                    transformed, flat, dim=-1
                ).double().cpu().numpy(),
                "sequence": np.repeat(
                    np.arange(self.offset, self.offset + batch, dtype=np.int64), positions
                ),
            }
        )
        self.geometry_parts.append(geometry)
        self.evaluations += out.network_evaluations
        self.projections += out.projections
        self.offset += batch
        return transformed.reshape_as(activation)

    def collected(self) -> dict[str, np.ndarray]:
        return {
            key: np.concatenate([r[key] for r in self.records]) for key in self.records[0]
        }


def _run_arm(
    flow, language_model, tokens, directions, c_target, noise, clean_losses,  # noqa: ANN001
    *, t_start: float, nfe: int, arm: str, hook: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, _ConstrainedTransform]:
    transform = _ConstrainedTransform(
        flow, directions, c_target, noise, t_start=t_start, nfe=nfe, arm=arm
    )
    losses = sequence_delta_lm(
        language_model, tokens, transform, hook=hook, skip_bos=True, clean=clean_losses
    )
    return (
        transform.collected(),
        losses["transformed"].numpy(),
        losses["delta"].numpy(),
        transform,
    )


def _summarize(
    rows: dict[str, np.ndarray],
    nll: np.ndarray,
    delta_lm: np.ndarray,
    transform: _ConstrainedTransform,
    clamp_nll: np.ndarray,
    assignment: np.ndarray,
) -> dict:
    paired = nll - clamp_nll
    bootstrap = paired_bootstrap_mean_ci(
        paired, seed=SPEC.bootstrap_seed, n_resamples=SPEC.bootstrap_resamples,
        confidence=SPEC.confidence,
    )
    by_direction: dict[int, float] = {}
    for index in np.unique(assignment):
        by_direction[int(index)] = float(paired[assignment == index].mean())
    values = np.array(list(by_direction.values()))
    lovo = [
        float(paired[assignment != index].mean()) for index in np.unique(assignment)
    ]
    return {
        "arm": transform.arm,
        "t_start": transform.t_start,
        "nfe": transform.nfe,
        "network_evaluations": transform.evaluations,
        "projections": transform.projections,
        "mean_nll": float(nll.mean()),
        "mean_delta_lm_vs_clean": float(delta_lm.mean()),
        "paired_delta_nll_vs_clamp": bootstrap,
        "coordinate_abs_error_mean": float(rows["coordinate_abs_error"].mean()),
        "coordinate_abs_error_median": float(np.median(rows["coordinate_abs_error"])),
        "relative_l2_to_clean_mean": float(rows["relative_l2_to_clean"].mean()),
        "cosine_to_clean_mean": float(rows["cosine_to_clean"].mean()),
        "correction": merge_decompositions(transform.geometry_parts),
        "per_direction_paired_delta_nll": by_direction,
        "fraction_directions_negative": float((values < 0).mean()),
        "lovo_paired_delta_nll_min": float(np.min(lovo)),
        "lovo_paired_delta_nll_max": float(np.max(lovo)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--activation-dir", required=True, type=Path)
    parser.add_argument("--token-cache-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--hook", default="blocks.7.hook_resid_pre")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    flow, meta, _ = load_flow_checkpoint(args.checkpoint, device=device)
    flow.eval()
    if not hasattr(flow, "velocity_field"):
        raise ValueError("checkpoint is not a conditional flow model")

    array_path, meta_path, _ = activation_paths(args.name, args.activation_dir)
    metadata = json.loads(meta_path.read_text())
    if metadata.get("split") != "val" or not metadata.get("bos_dropped"):
        raise ValueError("diagnostic requires the BOS-dropped validation artifact")
    activations = np.load(array_path, mmap_mode="r")

    pool = load_training_direction_pool(args.pool).to(device=device, dtype=torch.float32)
    picked = select_directions(len(pool), SPEC)
    directions = pool.directions[picked].to(device=device, dtype=torch.float32)

    # Identical frozen plan as natural_support_v1 -> identical targets.
    reference_rows = select_reference_rows(activations.shape[0], SPEC)
    reference = torch.from_numpy(
        np.array(activations[reference_rows], dtype=np.float32)
    ).to(device)
    natural = (reference @ directions.T).T.double().cpu().numpy()
    del reference
    stats = natural_coordinate_stats(natural, SPEC)

    split = make_split(activations.shape[0], PER_SEQ, VAL_FRACTION, SPLIT_SEED)
    sequence_ids = select_sequences(np.unique(split.val // PER_SEQ), SPEC)
    assignment = assign_directions(SPEC)

    language_model = load_model(str(device))
    cache_path = args.token_cache_dir / metadata["token_cache_file"]
    tokens = torch.from_numpy(np.load(cache_path))[torch.from_numpy(sequence_ids)].long()

    sequence_directions = directions[torch.from_numpy(assignment).to(device)]
    generator = torch.Generator(device="cpu").manual_seed(SPEC.noise_seed)
    noise = torch.randn(
        SPEC.n_sequences, metadata["ctx"] - 1, directions.shape[1], generator=generator
    )
    clean_losses = sequence_lm_losses(
        language_model, tokens, hook=args.hook, skip_bos=True
    )

    arms: dict[str, dict] = {}
    raw: dict[str, np.ndarray] = {}
    for quantile in SPEC.target_quantiles:
        label = f"q{int(round(quantile * 100)):02d}"
        per_direction_target = target_coordinates(stats, quantile)
        target = torch.from_numpy(
            per_direction_target[assignment].astype(np.float32)
        ).to(device)

        clamp_rows, clamp_nll, clamp_delta, clamp_transform = _run_arm(
            flow, language_model, tokens, sequence_directions, target, noise,
            clean_losses, t_start=0.0, nfe=0, arm="clamp_only", hook=args.hook,
        )
        arms[f"{label}_clamp_only"] = _summarize(
            clamp_rows, clamp_nll, clamp_delta, clamp_transform, clamp_nll, assignment
        )
        for field, values in clamp_rows.items():
            raw[f"{label}_clamp_only_{field}"] = values

        for t_start in T_START:
            for nfe in NFE:
                for arm in FLOW_ARMS:
                    rows, nll, delta, transform = _run_arm(
                        flow, language_model, tokens, sequence_directions, target,
                        noise, clean_losses, t_start=t_start, nfe=nfe, arm=arm,
                        hook=args.hook,
                    )
                    tag = f"{label}_t{t_start:.2f}_nfe{nfe}_{arm}"
                    entry = _summarize(
                        rows, nll, delta, transform, clamp_nll, assignment
                    )
                    entry["target_quantile"] = quantile
                    requested = rows["c_target"] - rows["c0_clean"]
                    up = requested > 0
                    for name, mask in (("up", up), ("down", ~up)):
                        if mask.sum() > 0:
                            entry[f"{name}_coordinate_abs_error_mean"] = float(
                                rows["coordinate_abs_error"][mask].mean()
                            )
                    arms[tag] = entry
                    for field, values in rows.items():
                        raw[f"{tag}_{field}"] = values
        print(f"finished target {label}")

    payload = {
        "diagnostic": "constrained_naturalization_v1",
        "question": (
            "can the learned prior improve orthogonal components at a fixed "
            "semantic coordinate, beating a hard clamp?"
        ),
        "spec": spec_payload(SPEC),
        "grid": {"t_start": list(T_START), "nfe": list(NFE), "arms": list(FLOW_ARMS)},
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": meta.get("step"),
        "validation_artifact": args.name,
        "validation_artifact_sha256": file_sha256(array_path),
        "clean_lm_loss": float(clean_losses.mean()),
        "direction_pool": pool.identity(),
        "direction_pool_indices": picked.tolist(),
        "sequence_ids": sequence_ids.tolist(),
        "direction_assignment": assignment.tolist(),
        "natural_coordinate_stats": stats,
        "arms": arms,
        "sae_metrics": "not run: pool artifact carries no SAE feature ids",
        "lexicon_metrics": "not run: lexicons exist only for DEV vectors",
        "degeneration_metrics": "not run: diagnostic substitutes activations, does not generate",
        "dev_vectors_accessed": False,
        "held_out_accessed": False,
        "llm_judge_used": False,
        "trained_anything": False,
    }
    report = args.out_dir / "constrained_naturalization.json"
    report.write_text(json.dumps(payload, indent=2))
    np.savez_compressed(args.out_dir / "raw_rows.npz", **raw)
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
