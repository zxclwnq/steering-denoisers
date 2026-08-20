"""Natural-support controllability diagnostic on one conditional-flow checkpoint.

Answers: can the conditional prior control ``<h, v>`` when the requested target
stays inside the natural coordinate distribution it was trained on?

Concept-independent. Frozen validation activations, training-only pool
directions, no DEV vectors, no held-out data, no LLM judge. Trains nothing.

    uv run python scripts/natural_support_controllability.py \
        --checkpoint /workspace/checkpoints/<run>/best_step_249500.pt \
        --activation-dir /workspace/data/fineweb_activations \
        --token-cache-dir /workspace/data/fineweb_token_cache \
        --name resid7_fw_val_1024k_v1 \
        --pool data/direction_pools/training_only_rank256_v1.pt \
        --out-dir /workspace/results/natural_support_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from interp.activations import activation_paths, file_sha256, make_split
from interp.conditional_flow import (
    conditional_clamp_steer,
    load_training_direction_pool,
)
from interp.flow_sampling import flow_correct
from interp.functional import sequence_delta_lm, sequence_lm_losses
from interp.model import load_model, resolve_device
from interp.natural_support import (
    NATURAL_SUPPORT_SPEC,
    assign_directions,
    calibration,
    classify,
    displacement_bin_edges,
    evaluable_mask,
    monotonic_across_quantiles,
    natural_coordinate_stats,
    per_direction_slopes,
    select_directions,
    select_reference_rows,
    select_sequences,
    spec_payload,
    target_coordinates,
)
from interp.provenance import source_revision
from interp.train_flow import load_flow_checkpoint

SPEC = NATURAL_SUPPORT_SPEC
PER_SEQ = 127
VAL_FRACTION = 0.05
SPLIT_SEED = 20260807


class _CoordinateTransform:
    """Substitution hook applying one conditional arm and recording coordinates."""

    def __init__(
        self,
        flow,  # noqa: ANN001
        directions: torch.Tensor,
        c_target: torch.Tensor | None,
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
        self.evaluations = 0

    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.ndim != 3:
            raise ValueError("transform expects [sequence, position, d_model]")
        batch, positions, width = activation.shape
        stop = self.offset + batch
        rows = slice(self.offset, stop)
        v = self.directions[rows].to(device=activation.device, dtype=activation.dtype)
        noise = self.noise[rows].to(device=activation.device, dtype=activation.dtype)
        flat = activation.reshape(-1, width)
        flat_noise = noise.reshape(-1, width)
        flat_v = v[:, None, :].expand(batch, positions, width).reshape(-1, width)
        c0 = (flat * flat_v).sum(dim=-1)

        if self.arm == "unconditional":
            transformed = flow_correct(
                self.flow, flat, noise=flat_noise, t_start=self.t_start, nfe=self.nfe
            )
            requested = c0
        else:
            if self.arm == "self":
                requested = c0
            else:
                target = self.c_target[rows].to(
                    device=activation.device, dtype=activation.dtype
                )
                requested = target[:, None].expand(batch, positions).reshape(-1)
            out = conditional_clamp_steer(
                self.flow,
                flat,
                flat_v,
                alpha=requested[:, None],
                mode="absolute",
                noise=flat_noise,
                t_start=self.t_start,
                nfe=self.nfe,
                seed_mode="clean",
            )
            transformed = out.activation
        if not torch.isfinite(transformed).all():
            raise ValueError(f"{self.arm} transform produced a non-finite activation")

        realised = (transformed * flat_v).sum(dim=-1)
        clean_norm = flat.norm(dim=-1)
        relative = (transformed - flat).norm(dim=-1) / clean_norm
        cosine = torch.nn.functional.cosine_similarity(transformed, flat, dim=-1)
        self.records.append(
            {
                "c0": c0.double().cpu().numpy(),
                "c_target": requested.double().cpu().numpy(),
                "c_realised": realised.double().cpu().numpy(),
                "relative_l2": relative.double().cpu().numpy(),
                "cosine": cosine.double().cpu().numpy(),
                "sequence": np.repeat(
                    np.arange(self.offset, stop, dtype=np.int64), positions
                ),
            }
        )
        self.evaluations += 0 if self.t_start == 0.0 else self.nfe
        self.offset = stop
        return transformed.reshape_as(activation)

    def collected(self) -> dict[str, np.ndarray]:
        return {
            key: np.concatenate([record[key] for record in self.records])
            for key in self.records[0]
        }


def _arm_rows(
    flow,  # noqa: ANN001
    language_model,  # noqa: ANN001
    tokens: torch.Tensor,
    directions: torch.Tensor,
    c_target: torch.Tensor | None,
    noise: torch.Tensor,
    clean_losses: torch.Tensor,
    *,
    t_start: float,
    nfe: int,
    arm: str,
    hook: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    transform = _CoordinateTransform(
        flow, directions, c_target, noise, t_start=t_start, nfe=nfe, arm=arm
    )
    losses = sequence_delta_lm(
        language_model, tokens, transform, hook=hook, skip_bos=True, clean=clean_losses
    )
    return transform.collected(), losses["delta"].numpy(), transform.evaluations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--activation-dir", required=True, type=Path)
    parser.add_argument("--token-cache-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--unconditional-checkpoint", type=Path)
    parser.add_argument("--hook", default="blocks.7.hook_resid_pre")
    parser.add_argument("--device", default=None)
    parser.add_argument("--with-nfe3", action="store_true")
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
    if getattr(pool, "split", "training_only") != "training_only":
        raise ValueError("direction pool must be the training-only split")
    picked = select_directions(len(pool), SPEC)
    directions = pool.directions[picked].to(device=device, dtype=torch.float32)

    # --- 1. natural coordinate reference distributions -------------------
    reference_rows = select_reference_rows(activations.shape[0], SPEC)
    reference = torch.from_numpy(
        np.array(activations[reference_rows], dtype=np.float32)
    ).to(device)
    natural = (reference @ directions.T).T.double().cpu().numpy()
    del reference
    stats = natural_coordinate_stats(natural, SPEC)
    direction_std = np.array([entry["std"] for entry in stats], dtype=np.float64)

    # --- evaluation sequences --------------------------------------------
    split = make_split(activations.shape[0], PER_SEQ, VAL_FRACTION, SPLIT_SEED)
    validation_ids = np.unique(split.val // PER_SEQ)
    sequence_ids = select_sequences(validation_ids, SPEC)
    assignment = assign_directions(SPEC)

    language_model = load_model(str(device))
    cache_path = args.token_cache_dir / metadata["token_cache_file"]
    tokens = torch.from_numpy(np.load(cache_path))[torch.from_numpy(sequence_ids)].long()
    if tokens.shape != (SPEC.n_sequences, metadata["ctx"]):
        raise ValueError(f"selected token matrix has shape {tuple(tokens.shape)}")

    sequence_directions = directions[torch.from_numpy(assignment).to(device)]
    generator = torch.Generator(device="cpu").manual_seed(SPEC.noise_seed)
    noise = torch.randn(
        SPEC.n_sequences, metadata["ctx"] - 1, directions.shape[1], generator=generator
    )
    clean_losses = sequence_lm_losses(
        language_model, tokens, hook=args.hook, skip_bos=True
    )

    unconditional = None
    if args.unconditional_checkpoint is not None:
        unconditional, _, _ = load_flow_checkpoint(
            args.unconditional_checkpoint, device=device
        )
        unconditional.eval()

    nfe_values = [SPEC.nfe_primary] + ([SPEC.nfe_optional] if args.with_nfe3 else [])
    permutation_rng = np.random.default_rng(SPEC.permutation_seed)
    raw: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}

    for nfe in nfe_values:
        for t_start in SPEC.t_start:
            key_prefix = f"t{t_start:.2f}_nfe{nfe}"

            # target-independent arms, computed once per (t_start, nfe)
            for arm, model in (("self", flow), ("unconditional", unconditional)):
                if model is None:
                    continue
                rows, delta_lm, evaluations = _arm_rows(
                    model, language_model, tokens, sequence_directions, None, noise,
                    clean_losses, t_start=t_start, nfe=nfe, arm=arm, hook=args.hook,
                )
                summary[f"{key_prefix}_{arm}"] = {
                    "arm": arm,
                    "t_start": t_start,
                    "nfe": nfe,
                    "network_evaluations": evaluations,
                    "mean_delta_lm": float(delta_lm.mean()),
                    "mean_relative_l2": float(rows["relative_l2"].mean()),
                    "mean_cosine": float(rows["cosine"].mean()),
                    "mean_abs_coordinate_shift": float(
                        np.abs(rows["c_realised"] - rows["c0"]).mean()
                    ),
                }
                for field, values in rows.items():
                    raw[f"{key_prefix}_{arm}_{field}"] = values

            for quantile in SPEC.target_quantiles:
                per_direction_target = target_coordinates(stats, quantile)
                target = torch.from_numpy(
                    per_direction_target[assignment].astype(np.float32)
                ).to(device)
                shuffled_order = permutation_rng.permutation(SPEC.n_sequences)
                shuffled = target[torch.from_numpy(shuffled_order).to(device)]

                for arm, arm_target in (
                    ("correct", target),
                    ("shuffled_target", shuffled),
                ):
                    rows, delta_lm, evaluations = _arm_rows(
                        flow, language_model, tokens, sequence_directions, arm_target,
                        noise, clean_losses, t_start=t_start, nfe=nfe, arm=arm,
                        hook=args.hook,
                    )
                    tag = f"{key_prefix}_q{int(round(quantile * 100)):02d}_{arm}"
                    direction_index = assignment[rows["sequence"]]
                    keep = evaluable_mask(
                        rows["c0"], rows["c_target"], direction_std[direction_index], SPEC
                    )
                    requested = (rows["c_target"] - rows["c0"])[keep]
                    realised = (rows["c_realised"] - rows["c0"])[keep]
                    entry: dict[str, object] = {
                        "arm": arm,
                        "t_start": t_start,
                        "nfe": nfe,
                        "target_quantile": quantile,
                        "network_evaluations": evaluations,
                        "n_rows_total": int(len(keep)),
                        "n_rows_evaluated": int(keep.sum()),
                        "mean_delta_lm": float(delta_lm.mean()),
                        "mean_relative_l2": float(rows["relative_l2"].mean()),
                        "mean_cosine": float(rows["cosine"].mean()),
                        "mean_requested_coordinate": float(rows["c_target"][keep].mean()),
                        "mean_realised_coordinate": float(rows["c_realised"][keep].mean()),
                        "mean_requested_displacement": float(requested.mean()),
                        "mean_realised_displacement": float(realised.mean()),
                        "control_fraction": float(realised.sum() / requested.sum()),
                    }
                    if keep.sum() >= 2:
                        entry["calibration"] = calibration(
                            requested, realised, rows["sequence"][keep], SPEC
                        )
                        entry["directions"] = per_direction_slopes(
                            requested, realised, direction_index[keep]
                        )
                        up = requested > 0
                        for label, mask in (("up", up), ("down", ~up)):
                            if mask.sum() >= 2:
                                entry[f"{label}_control"] = {
                                    "n_rows": int(mask.sum()),
                                    "mean_requested_displacement": float(
                                        requested[mask].mean()
                                    ),
                                    "mean_realised_displacement": float(
                                        realised[mask].mean()
                                    ),
                                    "control_fraction": float(
                                        realised[mask].sum() / requested[mask].sum()
                                    ),
                                    "fraction_correct_direction": float(
                                        (np.sign(realised[mask]) == np.sign(requested[mask])).mean()
                                    ),
                                }
                        edges = displacement_bin_edges(requested, SPEC)
                        bins = np.digitize(np.abs(requested), edges)
                        entry["displacement_bins"] = {
                            "edges": edges.tolist(),
                            "control_fraction": [
                                float(realised[bins == b].sum() / requested[bins == b].sum())
                                if (bins == b).sum() > 0
                                else None
                                for b in range(len(edges) + 1)
                            ],
                        }
                    summary[tag] = entry
                    for field, values in rows.items():
                        raw[f"{tag}_{field}"] = values

    # --- classification, primary grid only (nfe=1) ------------------------
    decision: dict[str, object] = {}
    for t_start in SPEC.t_start:
        prefix = f"t{t_start:.2f}_nfe{SPEC.nfe_primary}"
        correct_entries = [
            summary[f"{prefix}_q{int(round(q * 100)):02d}_correct"]
            for q in SPEC.target_quantiles
        ]
        shuffled_entries = [
            summary[f"{prefix}_q{int(round(q * 100)):02d}_shuffled_target"]
            for q in SPEC.target_quantiles
        ]
        usable = [e for e in correct_entries if "calibration" in e]
        if not usable:
            continue
        pooled_correct = max(usable, key=lambda e: e["n_rows_evaluated"])
        pooled_shuffled = max(
            [e for e in shuffled_entries if "calibration" in e],
            key=lambda e: e["n_rows_evaluated"],
        )
        monotonic = monotonic_across_quantiles(
            [float(e["mean_realised_coordinate"]) for e in correct_entries]
        )
        category, reasons = classify(
            pooled_correct["calibration"],
            pooled_shuffled["calibration"],
            pooled_correct["directions"],
            monotonic=monotonic,
        )
        decision[prefix] = {
            "category": category,
            "reasons": reasons,
            "monotonic_across_target_quantiles": monotonic,
            "realised_coordinate_by_target_quantile": {
                f"p{int(round(q * 100)):02d}": float(e["mean_realised_coordinate"])
                for q, e in zip(SPEC.target_quantiles, correct_entries, strict=True)
            },
            "requested_coordinate_by_target_quantile": {
                f"p{int(round(q * 100)):02d}": float(e["mean_requested_coordinate"])
                for q, e in zip(SPEC.target_quantiles, correct_entries, strict=True)
            },
        }

    payload = {
        "diagnostic": SPEC.version,
        "spec": spec_payload(SPEC),
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": meta.get("step"),
        "experiment_id": meta.get("experiment_id"),
        "unconditional_checkpoint": (
            None if args.unconditional_checkpoint is None
            else str(args.unconditional_checkpoint)
        ),
        "unconditional_checkpoint_sha256": (
            None if args.unconditional_checkpoint is None
            else file_sha256(args.unconditional_checkpoint)
        ),
        "validation_artifact": args.name,
        "validation_artifact_sha256": file_sha256(array_path),
        "token_cache_file": metadata["token_cache_file"],
        "direction_pool": pool.identity(),
        "direction_pool_indices": picked.tolist(),
        "sequence_ids": sequence_ids.tolist(),
        "direction_assignment": assignment.tolist(),
        "natural_coordinate_stats": stats,
        "arms": summary,
        "decision": decision,
        "dev_vectors_accessed": False,
        "held_out_accessed": False,
        "llm_judge_used": False,
        "trained_anything": False,
    }
    report = args.out_dir / "natural_support_controllability.json"
    report.write_text(json.dumps(payload, indent=2))
    np.savez_compressed(args.out_dir / "raw_rows.npz", **raw)
    print(f"wrote {report}")
    for key, value in decision.items():
        print(f"{key}: category {value['category']}")
        for reason in value["reasons"]:
            print(f"    {reason}")


if __name__ == "__main__":
    main()
