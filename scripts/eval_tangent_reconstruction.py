"""T1: does the tangent-trained flow solve its own matched reconstruction task?

Tangent-corrupt a clean validation activation at its own natural coordinate,
reconstruct it with the tangent-trained flow, and compare against the corrupted
control and the clean identity. Optionally run the closed branch's isotropic
conditional prior through the identical inference path as a reference.

Concept-independent throughout: frozen validation activations, training-only
pool directions, no DEV vectors, no held-out data, no LLM judge, no training.

PREPARED, NOT RUN. Requires a trained tangent checkpoint, which requires the
human to authorize configs/flow_train_tangent_narrow16m_fw32m_v1.yaml first.

    uv run python scripts/eval_tangent_reconstruction.py \
        --checkpoint /workspace/checkpoints/<run>/best_step_XXXXXX.pt \
        --activation-dir /workspace/data/fineweb_activations \
        --token-cache-dir /workspace/data/fineweb_token_cache \
        --name resid7_fw_val_1024k_v1 \
        --pool data/direction_pools/training_only_rank256_v1.pt \
        --out-dir /workspace/results/tangent_reconstruction_t1_v1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from interp.activations import file_sha256
from interp.conditional_flow import ConditionalFlowMatcher, load_training_direction_pool
from interp.flow_core import flow_matching_loss, n_parameters
from interp.functional import sequence_delta_lm, sequence_lm_losses
from interp.model import load_model, resolve_device
from interp.natural_support import assign_directions, select_directions, select_sequences
from interp.provenance import source_revision
from interp.tangent_eval import (
    STOP_RULE,
    TangentReconstructionSpec,
    concatenate,
    load_validated_evaluation_bundle,
    natural_coordinate,
    reconstruction_spec_for,
    reconstruction_summary,
    require_fresh_output_dir,
    spec_payload,
    t1_verdict,
    tangent_corrupted_activation,
    tangent_geometry,
    unselected_checkpoint_receipt,
    verify_direction_pool,
    verify_selected_checkpoint,
    write_t1_receipt,
)
from interp.tangent_flow import (
    ISOTROPIC_OBJECTIVE,
    TANGENT_OBJECTIVES,
    clamp_then_tangent_flow,
    sample_tangent_flow_batch,
    tangent_project,
)
from interp.train_flow import checkpoint_objective, load_flow_checkpoint

PER_SEQ = 127
VAL_FRACTION = 0.05
SPLIT_SEED = 20260807


class _TangentTransform:
    """Substitution hook: tangent-corrupt at the natural coordinate, then reconstruct.

    ``reconstruct=False`` gives the corrupted control, which is by construction
    the exact state the reconstruction arm integrates from.
    """

    def __init__(
        self,
        flow: ConditionalFlowMatcher,
        directions: torch.Tensor,
        noise: torch.Tensor,
        *,
        t_start: float,
        nfe: int,
        reconstruct: bool,
        objective: str,
    ) -> None:
        self.flow = flow
        self.directions = directions
        self.noise = noise
        self.t_start = t_start
        self.nfe = nfe
        self.reconstruct = reconstruct
        self.objective = objective
        self.offset = 0
        self.records: list[dict[str, np.ndarray]] = []
        self.evaluations = 0
        self.projections = 0
        self.max_coordinate_drift = 0.0
        self.max_pre_projection_drift = 0.0
        self.raw_parallel_velocity: list[float] = []

    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.ndim != 3:
            raise ValueError("transform expects [sequence, position, d_model]")
        batch, positions, width = activation.shape
        rows = slice(self.offset, self.offset + batch)
        v = self.directions[rows].to(device=activation.device, dtype=activation.dtype)
        noise = self.noise[rows].to(device=activation.device, dtype=activation.dtype)

        clean = activation.reshape(-1, width)
        flat_noise = noise.reshape(-1, width)
        flat_v = v[:, None, :].expand(batch, positions, width).reshape(-1, width)
        # T1 conditions on each activation's own coordinate: the clamp is a no-op
        # and the only intervention is the tangent corruption.
        c = natural_coordinate(clean, flat_v)

        if self.reconstruct:
            out = clamp_then_tangent_flow(
                self.flow, clean, flat_v, c,
                noise=flat_noise, t_start=self.t_start, nfe=self.nfe,
                objective=self.objective,
            )
            produced = out.activation
            self.evaluations += out.network_evaluations
            self.projections += out.projections
            self.max_coordinate_drift = max(
                self.max_coordinate_drift, float(out.diagnostics["max_coordinate_drift"])
            )
            self.max_pre_projection_drift = max(
                self.max_pre_projection_drift,
                float(out.diagnostics["max_pre_projection_drift"]),
            )
            self.raw_parallel_velocity.append(
                float(out.diagnostics["raw_parallel_velocity_norm_mean"])
            )
        else:
            produced = tangent_corrupted_activation(
                self.flow, clean, flat_v, c, noise=flat_noise, t_start=self.t_start,
                objective=self.objective,
            )

        self.records.append(tangent_geometry(clean, produced, flat_v, c))
        self.offset += batch
        return produced.reshape_as(activation)

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "t_start": self.t_start,
            "objective": self.objective,
            "nfe": self.nfe if self.reconstruct else 0,
            "network_evaluations": self.evaluations,
            "projections": self.projections,
            "max_coordinate_drift": self.max_coordinate_drift,
            "max_pre_projection_drift": self.max_pre_projection_drift,
            "raw_parallel_velocity_norm_mean": (
                float(np.mean(self.raw_parallel_velocity))
                if self.raw_parallel_velocity
                else 0.0
            ),
        }


def _run(flow, language_model, tokens, directions, noise, clean, *, hook, **kwargs):  # noqa: ANN001, ANN201
    transform = _TangentTransform(flow, directions, noise, **kwargs)
    losses = sequence_delta_lm(
        language_model, tokens, transform, hook=hook, skip_bos=True, clean=clean
    )
    return concatenate(transform.records), losses["transformed"].numpy(), transform


@torch.no_grad()
def _validation_tangent_mse(flow, activations, pool, device, spec) -> dict[str, float]:  # noqa: ANN001
    """Recompute the checkpoint's own objective on frozen validation rows."""

    rng = np.random.default_rng(spec.mse_row_seed)
    rows = np.sort(
        rng.choice(activations.shape[0], size=min(spec.mse_rows, activations.shape[0]),
                   replace=False)
    )
    generator = torch.Generator(device=device).manual_seed(spec.mse_seed)
    picks = np.random.default_rng(spec.mse_seed).integers(
        0, len(rows), size=(spec.mse_batches, spec.mse_batch_size)
    )
    losses: list[float] = []
    controls: list[float] = []
    parallels: list[float] = []
    for index in range(spec.mse_batches):
        h = torch.from_numpy(
            np.array(activations[rows[picks[index]]], dtype=np.float32)
        ).to(device)
        batch = sample_tangent_flow_batch(
            h, normalizer=flow.normalizer, pool=pool, generator=generator,
            objective=spec.objective,
        )
        raw = flow(batch.x_t, batch.t, batch.v_x, batch.c_x)
        prediction = tangent_project(raw, batch.v_x)
        losses.append(float(flow_matching_loss(prediction, batch.target_velocity)))
        controls.append(float(batch.target_velocity.square().mean()))
        parallels.append(float((raw * batch.v_x).sum(-1).abs().mean()))
    return {
        "objective": spec.objective,
        "val_tangent_flow_mse": float(np.mean(losses)),
        "zero_predictor_mse": float(np.mean(controls)),
        "raw_parallel_velocity_mean": float(np.mean(parallels)),
        "n_batches": spec.mse_batches,
        "batch_size": spec.mse_batch_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--isotropic-checkpoint",
        type=Path,
        default=None,
        help="optional closed-branch conditional prior, run through the identical "
        "tangent inference path as a reference arm",
    )
    parser.add_argument("--activation-dir", required=True, type=Path)
    parser.add_argument("--token-cache-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="training run directory holding best.json and meta.json; "
                             "required for a formal T1 verdict")
    parser.add_argument("--allow-unselected-checkpoint", action="store_true",
                        help="evaluate a checkpoint that is not the run's "
                             "concept-independent selection; the result is marked "
                             "diagnostic-only and cannot carry a formal T1 verdict")
    parser.add_argument("--overwrite-debug-mode", action="store_true",
                        help="permit writing into a non-empty result directory; "
                             "marks the run as a discardable debug run")
    parser.add_argument("--hook", default="blocks.7.hook_resid_pre")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    require_fresh_output_dir(args.out_dir, overwrite_debug=args.overwrite_debug_mode)

    # The evaluation plan is derived from the checkpoint's own recorded corruption
    # path, never from an operator flag, so a model can never be scored on a grid
    # it was not trained for.
    objective = checkpoint_objective(args.checkpoint)
    if objective not in TANGENT_OBJECTIVES:
        raise ValueError(
            f"T1 evaluates constraint-preserving checkpoints; {args.checkpoint} "
            f"records objective {objective!r}"
        )
    spec: TangentReconstructionSpec = reconstruction_spec_for(objective)
    flow, meta, _ = load_flow_checkpoint(
        args.checkpoint, device=device, expected_objective=objective
    )
    if not isinstance(flow, ConditionalFlowMatcher):
        raise ValueError("the tangent objective requires a conditional flow checkpoint")
    flow.eval()

    # P2: the checkpoint must BE the run's concept-independent selection, not
    # merely a path someone typed.
    if args.allow_unselected_checkpoint:
        selection = unselected_checkpoint_receipt(
            args.checkpoint, meta, "operator passed --allow-unselected-checkpoint"
        )
        formal_eligible = False
        ineligible_reason = "checkpoint was not verified as the run's selection"
    elif args.run_dir is None:
        raise ValueError(
            "--run-dir is required for a formal T1 verdict so the checkpoint can be "
            "verified against the run's best.json/meta.json; pass "
            "--allow-unselected-checkpoint for a diagnostic-only run"
        )
    else:
        selection = verify_selected_checkpoint(
            args.checkpoint, meta, run_dir=args.run_dir,
            expected_objective=objective,
        )
        formal_eligible = True
        ineligible_reason = None

    # P4: one fully validated bundle, not two loosely related files.
    bundle = load_validated_evaluation_bundle(
        args.name, args.activation_dir, args.token_cache_dir,
        hook=args.hook, per_seq=PER_SEQ, val_fraction=VAL_FRACTION,
        split_seed=SPLIT_SEED, d_model=flow.cfg.activation_width,
    )
    activations = bundle.activations
    metadata = bundle.meta

    pool = load_training_direction_pool(args.pool).to(device=device, dtype=torch.float32)
    # P3: the evaluation pool must be the pool the checkpoint trained on.
    pool_receipt = verify_direction_pool(meta, pool)
    picked = select_directions(len(pool), spec)
    directions = pool.directions[picked].to(device=device, dtype=torch.float32)

    sequence_ids = select_sequences(np.unique(bundle.split.val // PER_SEQ), spec)
    assignment = assign_directions(spec)
    sequence_directions = directions[torch.from_numpy(assignment).to(device)]

    language_model = load_model(str(device))
    tokens = bundle.tokens[torch.from_numpy(sequence_ids)].long()
    generator = torch.Generator(device="cpu").manual_seed(spec.noise_seed)
    noise = torch.randn(
        spec.n_sequences, metadata["ctx"] - 1, directions.shape[1], generator=generator
    )
    clean_losses = sequence_lm_losses(language_model, tokens, hook=args.hook, skip_bos=True)
    clean = clean_losses.numpy()

    reference = {"name": "identity", "mean_nll": float(clean.mean()), "delta_lm": 0.0}
    priors: dict[str, ConditionalFlowMatcher] = {spec.arm_label: flow}
    isotropic_reference: dict | None = None
    if args.isotropic_checkpoint is not None:
        isotropic, isotropic_meta, _ = load_flow_checkpoint(
            args.isotropic_checkpoint, device=device,
            expected_objective=ISOTROPIC_OBJECTIVE,
        )
        if not isinstance(isotropic, ConditionalFlowMatcher):
            raise ValueError("the reference prior must be a conditional flow checkpoint")
        if isotropic.cfg.activation_width != flow.cfg.activation_width:
            raise ValueError(
                "reference prior activation width differs; it cannot even be run on "
                "this artifact"
            )
        if not torch.allclose(
            isotropic.normalizer.mean.cpu(), flow.normalizer.mean.cpu()
        ) or not torch.allclose(isotropic.normalizer.std.cpu(), flow.normalizer.std.cpu()):
            raise ValueError(
                "reference prior was standardized differently; running it on this "
                "artifact would not be interpretable at all"
            )
        isotropic.eval()
        # P11: matching activation width does NOT make this an objective-only
        # control. Capacity, training budget, data and noise stream all differ,
        # so it is labelled for what it is and is excluded from the formal gate.
        matched = (
            isotropic.cfg == flow.cfg
            and (isotropic_meta.get("direction_pool") or {}).get("digest")
            == pool_receipt.get("digest")
            and isotropic_meta.get("config_fingerprint") is not None
        )
        isotropic_reference = {
            "label": "diagnostic_unmatched_reference",
            "checkpoint": str(args.isotropic_checkpoint),
            "checkpoint_sha256": file_sha256(args.isotropic_checkpoint),
            "model_config": asdict(isotropic.cfg),
            "n_parameters": n_parameters(isotropic),
            "tangent_model_n_parameters": n_parameters(flow),
            "architecture_identical_to_tangent_model": isotropic.cfg == flow.cfg,
            "direction_pool": isotropic_meta.get("direction_pool"),
            "config_fingerprint": isotropic_meta.get("config_fingerprint"),
            "objective_identity": isotropic_meta.get("objective_identity"),
            "all_matching_criteria_satisfied": bool(matched),
            "excluded_from_formal_gate": True,
            "interpretation_limit": (
                "this checkpoint differs from the tangent model in capacity, "
                "training budget, training data and noise stream. It shows how a "
                "historical prior behaves on the tangent task; it CANNOT isolate "
                "the effect of the objective. A causal objective-only comparison "
                "needs a separately trained matched 16M isotropic control, which "
                "is not authorized."
            ),
        }
        priors["diagnostic_unmatched_reference"] = isotropic

    arms: dict[str, dict] = {}
    raw: dict[str, np.ndarray] = {}
    for t_start in spec.t_start:
        corrupt_rows, corrupt_nll, corrupt_transform = _run(
            flow, language_model, tokens, sequence_directions, noise, clean_losses,
            hook=args.hook, t_start=t_start, nfe=1, reconstruct=False,
            objective=objective,
        )
        label = f"t{t_start:.2f}_corrupted"
        arms[label] = {
            **reconstruction_summary(
                corrupt_rows, corrupt_nll, clean, corrupt_nll, assignment
            ),
            **corrupt_transform.diagnostics(),
            "arm": "corrupted_control",
        }
        for field, values in corrupt_rows.items():
            raw[f"{label}_{field}"] = values

        for prior_name, prior in priors.items():
            for nfe in spec.nfe:
                rows, nll, transform = _run(
                    prior, language_model, tokens, sequence_directions, noise,
                    clean_losses, hook=args.hook, t_start=t_start, nfe=nfe,
                    reconstruct=True, objective=objective,
                )
                tag = f"t{t_start:.2f}_nfe{nfe}_{prior_name}"
                arms[tag] = {
                    **reconstruction_summary(rows, nll, clean, corrupt_nll, assignment),
                    **transform.diagnostics(),
                    "arm": prior_name,
                }
                for field, values in rows.items():
                    raw[f"{tag}_{field}"] = values
        print(f"finished t_start {t_start:.2f}")

    # P1: the gate reads spec.primary_cell(), never a tuple position.
    verdict = t1_verdict(
        arms,
        spec=spec,
        formal_eligible=formal_eligible and not args.overwrite_debug_mode,
        ineligible_reason=(
            "debug-mode run" if args.overwrite_debug_mode else ineligible_reason
        ),
    )

    payload = {
        "experiment": spec.version,
        "question": (
            "does the tangent-trained flow solve its own matched tangent "
            "reconstruction task on frozen validation activations?"
        ),
        "stop_rule": STOP_RULE,
        "spec": spec_payload(spec),
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": meta.get("step"),
        "checkpoint_objective_identity": meta.get("objective_identity"),
        "checkpoint_selection": selection,
        "formal_verdict_eligible": bool(verdict["formal_verdict_eligible"]),
        "debug_mode": bool(args.overwrite_debug_mode),
        "isotropic_reference": isotropic_reference,
        "validation_artifact": bundle.identity,
        "clean_reference": reference,
        "validation_tangent_mse": _validation_tangent_mse(
            flow, activations, pool, device, spec
        ),
        "direction_pool": pool_receipt,
        "direction_pool_indices": picked.tolist(),
        "sequence_ids": sequence_ids.tolist(),
        "direction_assignment": assignment.tolist(),
        "arms": arms,
        "t1_gate": verdict,
        "dev_vectors_accessed": False,
        "held_out_accessed": False,
        "llm_judge_used": False,
        "trained_anything": False,
    }
    report = args.out_dir / "tangent_reconstruction.json"
    report.write_text(json.dumps(payload, indent=2))
    np.savez_compressed(args.out_dir / "raw_rows.npz", **raw)
    receipt = write_t1_receipt(args.out_dir / "t1_receipt.json", payload)
    print(f"wrote {report}")
    print(f"T1 gate ({verdict['primary_cell']}): {verdict['verdict']}")
    print(f"  {verdict['consequence']}")
    if receipt["verdict"] != "PASS" or not receipt["formal_verdict_eligible"]:
        print("  T2 is NOT authorized by this receipt.")


if __name__ == "__main__":
    main()
