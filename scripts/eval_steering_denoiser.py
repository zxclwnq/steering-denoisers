"""Experiment B: is the steering-trained denoiser repairing, or just undoing?

Steer additively along a training-only direction, apply lambda of the denoiser's
correction, and compare language-model quality against scalar shrinkage evaluated
at the **same realised concept strength** -- never at the same nominal alpha.

    additive:   h + alpha v
    denoiser:   h'_lambda = z + lambda ( D(z) - z ),   z = h + alpha v
    shrinkage:  h + alpha_eff v,  alpha_eff = <h'_lambda - h, v>

If the denoiser's apparent quality advantage disappears once shrinkage is given
the same alpha_eff, the method removes steering rather than repairing it. That is
the outcome `docs/POST_STOP_PROTOCOL_2026-08-19.md` §B.6 names in advance as the
expected failure mode.

Concept-independent throughout: frozen validation activations, training-only pool
directions, no DEV vectors, no held-out data, no LLM judge, no training.

PREPARED, NOT RUN. Requires a trained denoiser checkpoint, which requires the
human to authorize configs/flow_train_steering_denoiser_16m_fw32m_v1.yaml first.

    uv run python scripts/eval_steering_denoiser.py \
        --checkpoint /workspace/checkpoints/<run>/best_step_XXXXXX.pt \
        --activation-dir /workspace/data/fineweb_activations \
        --token-cache-dir /workspace/data/fineweb_token_cache \
        --name resid7_fw_val_1024k_v1 \
        --pool data/direction_pools/training_only_rank256_v1.pt \
        --out-dir /workspace/results/steering_denoiser_b_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from interp.activations import file_sha256
from interp.conditional_flow import load_training_direction_pool
from interp.functional import sequence_delta_lm, sequence_lm_losses
from interp.model import load_model, resolve_device
from interp.natural_support import (
    NATURAL_SUPPORT_SPEC,
    assign_directions,
    natural_coordinate_stats,
    select_directions,
    select_reference_rows,
    select_sequences,
)
from interp.provenance import source_revision
from interp.steering_denoiser import (
    STEERING_DENOISER_OBJECTIVE,
    SteeringDenoiser,
    partial_denoise,
    realised_strength,
    shrinkage_activation,
)
from interp.steering_denoiser_eval import (
    STEERING_DENOISER_EVAL_SPEC,
    STOP_RULE_B,
    alpha_ladder,
    assert_strength_match,
    spec_payload,
    steering_denoiser_verdict,
    strength_summary,
)
from interp.tangent_eval import (
    concatenate,
    load_validated_evaluation_bundle,
    require_fresh_output_dir,
    verify_direction_pool,
    verify_selected_checkpoint,
)
from interp.train_flow import checkpoint_objective, load_flow_checkpoint

SPEC = STEERING_DENOISER_EVAL_SPEC
PLAN = NATURAL_SUPPORT_SPEC
PER_SEQ = 127
VAL_FRACTION = 0.05
SPLIT_SEED = 20260807


def _records(
    clean: torch.Tensor,
    produced: torch.Tensor,
    direction: torch.Tensor,
    requested: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Per-row geometry and the two strengths the whole experiment turns on."""

    correction = produced - (clean + requested * direction)
    parallel = (correction * direction).sum(dim=-1, keepdim=True)
    orthogonal = correction - parallel * direction
    difference = produced - clean
    return {
        "requested_alpha": requested.squeeze(-1).double().cpu().numpy(),
        "realised_alpha": realised_strength(produced, clean, direction)
        .double()
        .cpu()
        .numpy(),
        "parallel_correction_norm": parallel.abs().squeeze(-1).double().cpu().numpy(),
        "orthogonal_correction_norm": orthogonal.norm(dim=-1).double().cpu().numpy(),
        "relative_l2_to_clean": (
            difference.norm(dim=-1) / clean.norm(dim=-1).clamp_min(1e-12)
        )
        .double()
        .cpu()
        .numpy(),
        "cosine_to_clean": torch.cosine_similarity(produced, clean, dim=-1)
        .double()
        .cpu()
        .numpy(),
    }


class _DenoiseTransform:
    """Substitution hook: additive steering, then ``lambda`` of the correction.

    ``lam = 0`` is the additive baseline exactly, with zero network evaluations.
    The realised strength of every row is retained so the matched shrinkage arm
    can be constructed from it.
    """

    def __init__(
        self,
        model: SteeringDenoiser,
        directions: torch.Tensor,
        alpha: torch.Tensor,
        *,
        lam: float,
    ) -> None:
        self.model = model
        self.directions = directions
        self.alpha = alpha
        self.lam = lam
        self.offset = 0
        self.records: list[dict[str, np.ndarray]] = []
        self.realised: list[np.ndarray] = []
        self.evaluations = 0

    def _slice(self, activation: torch.Tensor):  # noqa: ANN202
        if activation.ndim != 3:
            raise ValueError("transform expects [sequence, position, d_model]")
        batch, positions, width = activation.shape
        rows = slice(self.offset, self.offset + batch)
        v = self.directions[rows].to(device=activation.device, dtype=activation.dtype)
        a = self.alpha[rows].to(device=activation.device, dtype=activation.dtype)
        clean = activation.reshape(-1, width)
        flat_v = v[:, None, :].expand(batch, positions, width).reshape(-1, width)
        flat_a = a[:, None].expand(batch, positions).reshape(-1, 1)
        return batch, clean, flat_v, flat_a

    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        batch, clean, flat_v, flat_a = self._slice(activation)
        out = partial_denoise(self.model, clean, flat_v, flat_a, lam=self.lam)
        self.evaluations += int(out.diagnostics["network_evaluations"])
        self.records.append(_records(clean, out.activation, flat_v, flat_a))
        self.realised.append(out.realised_alpha.double().cpu().numpy())
        self.offset += batch
        return out.activation.reshape_as(activation)

    def realised_alpha(self) -> np.ndarray:
        return np.concatenate(self.realised)

    def diagnostics(self) -> dict[str, float | int | str]:
        return {
            "arm": "denoise" if self.lam > 0.0 else "additive",
            "lambda": float(self.lam),
            "network_evaluations": self.evaluations,
        }


class _MatchedShrinkageTransform(_DenoiseTransform):
    """Additive steering at a supplied per-row strength: the matched control.

    The strengths come from a completed denoiser pass over the identical
    sequences in the identical order, so row ``i`` here carries exactly the
    concept strength row ``i`` of that pass ended up with. Nothing is denoised;
    the network is never evaluated.
    """

    def __init__(
        self,
        model: SteeringDenoiser,
        directions: torch.Tensor,
        alpha: torch.Tensor,
        *,
        target_strength: np.ndarray,
        lam: float,
    ) -> None:
        super().__init__(model, directions, alpha, lam=0.0)
        self.target_strength = np.asarray(target_strength, dtype=np.float64)
        self.matched_for_lambda = float(lam)
        self.row_offset = 0

    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        batch, clean, flat_v, flat_a = self._slice(activation)
        span = clean.shape[0]
        if self.row_offset + span > self.target_strength.size:
            raise ValueError(
                "matched shrinkage ran past the recorded strengths; the two passes "
                "must cover the same rows in the same order"
            )
        strength = torch.from_numpy(
            self.target_strength[self.row_offset : self.row_offset + span]
        ).to(device=clean.device, dtype=clean.dtype)
        produced = shrinkage_activation(clean, flat_v, strength)
        # Records keep the DENOISER's requested alpha so the two arms' rows line
        # up column for column; the realised strength is the matched one.
        self.records.append(_records(clean, produced, flat_v, flat_a))
        self.realised.append(
            realised_strength(produced, clean, flat_v).double().cpu().numpy()
        )
        self.row_offset += span
        self.offset += batch
        return produced.reshape_as(activation)

    def diagnostics(self) -> dict[str, float | int | str]:
        return {
            "arm": "matched_shrinkage",
            "matched_for_lambda": self.matched_for_lambda,
            "network_evaluations": 0,
        }


def _run(transform, language_model, tokens, clean, *, hook):  # noqa: ANN001, ANN201
    losses = sequence_delta_lm(
        language_model, tokens, transform, hook=hook, skip_bos=True, clean=clean
    )
    return concatenate(transform.records), losses["transformed"].numpy(), transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--activation-dir", required=True, type=Path)
    parser.add_argument("--token-cache-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="training run directory holding best.json and meta.json; "
                             "required for a formal verdict")
    parser.add_argument("--allow-unselected-checkpoint", action="store_true",
                        help="evaluate a checkpoint that is not the run's "
                             "concept-independent selection; diagnostic only")
    parser.add_argument("--overwrite-debug-mode", action="store_true",
                        help="permit writing into a non-empty result directory; "
                             "marks the run as a discardable debug run")
    parser.add_argument("--hook", default="blocks.7.hook_resid_pre")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    require_fresh_output_dir(args.out_dir, overwrite_debug=args.overwrite_debug_mode)

    objective = checkpoint_objective(args.checkpoint)
    if objective != STEERING_DENOISER_OBJECTIVE:
        raise ValueError(
            f"experiment B evaluates steering denoisers; {args.checkpoint} records "
            f"objective {objective!r}"
        )
    model, meta, _ = load_flow_checkpoint(
        args.checkpoint, device=device, expected_objective=objective
    )
    if not isinstance(model, SteeringDenoiser):
        raise ValueError("the steering-denoising objective requires a SteeringDenoiser")
    model.eval()

    if args.allow_unselected_checkpoint:
        selection = {
            "verified": False,
            "reason": "operator passed --allow-unselected-checkpoint",
        }
        formal_eligible = False
        ineligible_reason = "checkpoint was not verified as the run's selection"
    elif args.run_dir is None:
        raise ValueError(
            "--run-dir is required for a formal verdict so the checkpoint can be "
            "verified against the run's best.json/meta.json; pass "
            "--allow-unselected-checkpoint for a diagnostic-only run"
        )
    else:
        selection = verify_selected_checkpoint(
            args.checkpoint, meta, run_dir=args.run_dir, expected_objective=objective
        )
        formal_eligible = True
        ineligible_reason = None

    bundle = load_validated_evaluation_bundle(
        args.name, args.activation_dir, args.token_cache_dir,
        hook=args.hook, per_seq=PER_SEQ, val_fraction=VAL_FRACTION,
        split_seed=SPLIT_SEED, d_model=model.cfg.activation_width,
    )
    activations = bundle.activations

    pool = load_training_direction_pool(args.pool).to(device=device, dtype=torch.float32)
    pool_receipt = verify_direction_pool(meta, pool)
    picked = select_directions(len(pool), PLAN)
    directions = pool.directions[picked].to(device=device, dtype=torch.float32)

    # The alpha ladder is read off each direction's own natural support, using the
    # frozen natural_support_v1 reference rows -- the same construction the closed
    # branch used for its clamp targets.
    reference_rows = select_reference_rows(activations.shape[0], PLAN)
    reference = torch.from_numpy(
        np.array(activations[reference_rows], dtype=np.float32)
    ).to(device)
    natural = (reference @ directions.T).T.double().cpu().numpy()
    del reference
    stats = natural_coordinate_stats(natural, PLAN)
    ladder = alpha_ladder(stats, SPEC)

    sequence_ids = select_sequences(np.unique(bundle.split.val // PER_SEQ), PLAN)
    assignment = assign_directions(PLAN)
    sequence_directions = directions[torch.from_numpy(assignment).to(device)]

    language_model = load_model(str(device))
    tokens = bundle.tokens[torch.from_numpy(sequence_ids)].long()
    clean_losses = sequence_lm_losses(language_model, tokens, hook=args.hook, skip_bos=True)
    clean = clean_losses.numpy()

    arms: dict[str, dict] = {}
    raw: dict[str, np.ndarray] = {}
    primary_effects: dict[str, np.ndarray] = {}
    primary_validity: dict[str, dict] = {}

    for key, per_direction in ladder.items():
        alpha = torch.from_numpy(per_direction[assignment]).to(
            device=device, dtype=torch.float32
        )

        # Context arm: plain additive steering at the nominal alpha. Reported so a
        # reader can see the nominal-alpha comparison, never used as the decision.
        additive_rows, additive_nll, additive_transform = _run(
            _DenoiseTransform(model, sequence_directions, alpha, lam=0.0),
            language_model, tokens, clean_losses, hook=args.hook,
        )
        arms[f"{key}_additive"] = {
            "target_quantile": key,
            **strength_summary(
                additive_rows, additive_nll, additive_nll, clean, assignment, spec=SPEC
            ),
            **additive_transform.diagnostics(),
        }
        for field, values in additive_rows.items():
            raw[f"{key}_additive_{field}"] = values

        for lam in SPEC.corruption.lambda_grid:
            denoise_rows, denoise_nll, denoise_transform = _run(
                _DenoiseTransform(model, sequence_directions, alpha, lam=lam),
                language_model, tokens, clean_losses, hook=args.hook,
            )
            # THE control: shrinkage carrying exactly this arm's realised strength.
            shrink_rows, shrink_nll, shrink_transform = _run(
                _MatchedShrinkageTransform(
                    model, sequence_directions, alpha,
                    target_strength=denoise_transform.realised_alpha(), lam=lam,
                ),
                language_model, tokens, clean_losses, hook=args.hook,
            )
            match = assert_strength_match(
                denoise_rows, shrink_rows, tolerance=SPEC.strength_match_tolerance
            )

            denoise_tag = f"{key}_{SPEC.arm_label(lam)}"
            shrink_tag = f"{key}_{SPEC.shrinkage_label(lam)}"
            arms[denoise_tag] = {
                "target_quantile": key,
                # Paired against the matched control, not against nominal additive.
                **strength_summary(
                    denoise_rows, denoise_nll, shrink_nll, clean, assignment, spec=SPEC
                ),
                **denoise_transform.diagnostics(),
                "strength_match": match,
                "delta_nll_vs_nominal_additive": float(
                    denoise_nll.mean() - additive_nll.mean()
                ),
            }
            arms[shrink_tag] = {
                "target_quantile": key,
                **strength_summary(
                    shrink_rows, shrink_nll, additive_nll, clean, assignment, spec=SPEC
                ),
                **shrink_transform.diagnostics(),
            }
            for field, values in denoise_rows.items():
                raw[f"{denoise_tag}_{field}"] = values
            for field, values in shrink_rows.items():
                raw[f"{shrink_tag}_{field}"] = values

            if lam == SPEC.primary_lambda:
                # One paired effect per validation SEQUENCE, aligned across
                # quantiles so the bootstrap keeps a sequence's rungs together.
                primary_effects[key] = denoise_nll - shrink_nll
                primary_validity[key] = match
        print(f"finished rung {key}")

    if not primary_effects:
        raise RuntimeError("the frozen primary lambda produced no data")
    verdict = steering_denoiser_verdict(
        primary_effects,
        assignment,
        spec=SPEC,
        formal_eligible=formal_eligible and not args.overwrite_debug_mode,
        ineligible_reason=(
            "debug-mode run" if args.overwrite_debug_mode else ineligible_reason
        ),
    )
    verdict["per_quantile_strength_match"] = primary_validity

    payload = {
        "experiment": SPEC.version,
        "question": (
            "matched on realised concept strength, does a denoiser trained on "
            "steering-like corruption beat simply steering that much less?"
        ),
        "primary_statistic": (
            "NLL_denoiser(lambda=1) - NLL_shrinkage_at_same_alpha_eff "
            "(negative = genuine repair)"
        ),
        "stop_rule": STOP_RULE_B,
        "spec": spec_payload(SPEC),
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": meta.get("step"),
        "checkpoint_objective_identity": meta.get("objective_identity"),
        "checkpoint_selection": selection,
        "formal_verdict_eligible": bool(verdict["formal_verdict_eligible"]),
        "debug_mode": bool(args.overwrite_debug_mode),
        "validation_artifact": bundle.identity,
        "clean_lm_loss": float(clean.mean()),
        "direction_pool": pool_receipt,
        "direction_pool_indices": picked.tolist(),
        "sequence_ids": sequence_ids.tolist(),
        "direction_assignment": assignment.tolist(),
        "alpha_ladder": {key: values.tolist() for key, values in ladder.items()},
        "natural_coordinate_stats": stats,
        "arms": arms,
        "verdict": verdict,
        "dev_vectors_accessed": False,
        "held_out_accessed": False,
        "llm_judge_used": False,
        "trained_anything": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "steering_denoiser.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    np.savez_compressed(args.out_dir / "raw_rows.npz", **raw)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
