"""T2: hard clamp vs hard clamp + tangent-trained flow, at a fixed semantic coordinate.

Primary statistic:

    delta_NLL = NLL_tangent_flow - NLL_clamp        (negative = useful naturalization)

Both arms hard-clamp the same activation to the same natural-support target
coordinate. The tangent flow may then move only the orthogonal degrees of
freedom; the coordinate is enforced analytically and checked row by row, so the
flow arm cannot win by attenuating the coordinate.

Reuses the frozen natural_support_v1 plan verbatim: same directions, same
sequences, same target quantiles, same seeds, so the hard-clamp baseline is
directly comparable to results/constrained_naturalization_v1/.

Concept-independent: frozen validation activations, training-only pool
directions, no DEV vectors, no held-out data, no LLM judge, no training.

PREPARED, NOT RUN. Requires a trained tangent checkpoint that has passed T1.

    uv run python scripts/tangent_naturalization.py \
        --checkpoint /workspace/checkpoints/<run>/best_step_XXXXXX.pt \
        --activation-dir /workspace/data/fineweb_activations \
        --token-cache-dir /workspace/data/fineweb_token_cache \
        --name resid7_fw_val_1024k_v1 \
        --pool data/direction_pools/training_only_rank256_v1.pt \
        --out-dir /workspace/results/tangent_naturalization_t2_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from interp.activations import file_sha256
from interp.conditional_flow import ConditionalFlowMatcher, load_training_direction_pool
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
from interp.provenance import source_revision
from interp.tangent_eval import (
    STOP_RULE,
    assert_coordinate_match,
    concatenate,
    load_validated_evaluation_bundle,
    naturalization_spec_for,
    naturalization_summary,
    reconstruction_spec_for,
    require_fresh_output_dir,
    t2_cell_report,
    t2_experiment_verdict,
    t2_pooled_cell,
    tangent_geometry,
    verify_direction_pool,
    verify_t1_pass_receipt,
)
from interp.tangent_flow import TANGENT_OBJECTIVES, clamp_then_tangent_flow
from interp.train_flow import checkpoint_objective, load_flow_checkpoint

SPEC = NATURAL_SUPPORT_SPEC
PER_SEQ = 127
VAL_FRACTION = 0.05
SPLIT_SEED = 20260807


class _ClampTangentTransform:
    """Substitution hook: hard clamp, then optionally the tangent-trained flow.

    ``t_start = 0`` is arm A (clamp only, zero network evaluations); any other
    ``t_start`` is arm B. Both arms take the identical clamp, so the seed and the
    semantic coordinate are shared by construction rather than by coincidence.
    """

    def __init__(
        self,
        flow: ConditionalFlowMatcher,
        directions: torch.Tensor,
        c_target: torch.Tensor,
        noise: torch.Tensor,
        *,
        t_start: float,
        nfe: int,
        objective: str,
    ) -> None:
        self.flow = flow
        self.directions = directions
        self.c_target = c_target
        self.noise = noise
        self.t_start = t_start
        self.nfe = nfe
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
        target = self.c_target[rows].to(device=activation.device, dtype=activation.dtype)

        clean = activation.reshape(-1, width)
        flat_noise = noise.reshape(-1, width)
        flat_v = v[:, None, :].expand(batch, positions, width).reshape(-1, width)
        flat_c = target[:, None].expand(batch, positions).reshape(-1, 1)

        out = clamp_then_tangent_flow(
            self.flow, clean, flat_v, flat_c,
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

        record = tangent_geometry(clean, produced, flat_v, flat_c)
        # The correction is measured against the clamp, which is what the flow
        # arm is compared to; against clean it would mix in the clamp itself.
        correction = produced - out.seed
        parallel = (correction * flat_v).sum(dim=-1, keepdim=True)
        orthogonal = correction - parallel * flat_v
        record["orthogonal_correction_norm"] = orthogonal.norm(dim=-1).double().cpu().numpy()
        record["parallel_correction_norm"] = parallel.abs().squeeze(-1).double().cpu().numpy()
        record["sequence"] = np.repeat(
            np.arange(self.offset, self.offset + batch, dtype=np.int64), positions
        )
        self.records.append(record)
        self.offset += batch
        return produced.reshape_as(activation)

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "t_start": self.t_start,
            "objective": self.objective,
            "nfe": self.nfe if self.t_start != 0.0 else 0,
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


def _run(flow, language_model, tokens, directions, target, noise, clean, *, hook, **kwargs):  # noqa: ANN001, ANN201
    transform = _ClampTangentTransform(flow, directions, target, noise, **kwargs)
    losses = sequence_delta_lm(
        language_model, tokens, transform, hook=hook, skip_bos=True, clean=clean
    )
    return (
        concatenate(transform.records),
        losses["transformed"].numpy(),
        losses["delta"].numpy(),
        transform,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--activation-dir", required=True, type=Path)
    parser.add_argument("--token-cache-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--t1-receipt", required=True, type=Path,
                        help="t1_receipt.json from a formal T1 PASS on this exact "
                             "checkpoint and pool; T2 refuses to run without it")
    parser.add_argument("--overwrite-debug-mode", action="store_true",
                        help="permit writing into a non-empty result directory; "
                             "marks the run as a discardable debug run")
    parser.add_argument("--hook", default="blocks.7.hook_resid_pre")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    require_fresh_output_dir(args.out_dir, overwrite_debug=args.overwrite_debug_mode)

    # As in T1: the frozen inference grid comes from the checkpoint's own
    # recorded corruption path, so a model cannot be scored on a grid it was not
    # trained for. For the variance-preserving path this grid is the
    # MATCHED-SEVERITY image of the linear one.
    objective = checkpoint_objective(args.checkpoint)
    if objective not in TANGENT_OBJECTIVES:
        raise ValueError(
            f"T2 evaluates constraint-preserving checkpoints; {args.checkpoint} "
            f"records objective {objective!r}"
        )
    t2_spec = naturalization_spec_for(objective)
    flow, meta, _ = load_flow_checkpoint(
        args.checkpoint, device=device, expected_objective=objective
    )
    if not isinstance(flow, ConditionalFlowMatcher):
        raise ValueError("the tangent objective requires a conditional flow checkpoint")
    flow.eval()

    pool = load_training_direction_pool(args.pool).to(device=device, dtype=torch.float32)
    pool_receipt = verify_direction_pool(meta, pool)
    # P8: T2 is only meaningful after a formal T1 PASS on the same model, on the
    # same corruption path. The T1 spec comes from the same checkpoint-recorded
    # objective as the T2 grid, so the arms can never authorize each other.
    t1_receipt = verify_t1_pass_receipt(
        args.t1_receipt,
        checkpoint_sha256=file_sha256(args.checkpoint),
        pool_identity=pool_receipt,
        objective_identity=meta.get("objective_identity"),
        spec=reconstruction_spec_for(objective),
    )

    bundle = load_validated_evaluation_bundle(
        args.name, args.activation_dir, args.token_cache_dir,
        hook=args.hook, per_seq=PER_SEQ, val_fraction=VAL_FRACTION,
        split_seed=SPLIT_SEED, d_model=flow.cfg.activation_width,
    )
    activations = bundle.activations
    metadata = bundle.meta

    picked = select_directions(len(pool), SPEC)
    directions = pool.directions[picked].to(device=device, dtype=torch.float32)

    # Identical frozen plan as natural_support_v1 -> identical targets, so the
    # clamp baseline is comparable to results/constrained_naturalization_v1/.
    reference_rows = select_reference_rows(activations.shape[0], SPEC)
    reference = torch.from_numpy(
        np.array(activations[reference_rows], dtype=np.float32)
    ).to(device)
    natural = (reference @ directions.T).T.double().cpu().numpy()
    del reference
    stats = natural_coordinate_stats(natural, SPEC)

    sequence_ids = select_sequences(np.unique(bundle.split.val // PER_SEQ), SPEC)
    assignment = assign_directions(SPEC)
    sequence_directions = directions[torch.from_numpy(assignment).to(device)]

    language_model = load_model(str(device))
    tokens = bundle.tokens[torch.from_numpy(sequence_ids)].long()
    generator = torch.Generator(device="cpu").manual_seed(SPEC.noise_seed)
    noise = torch.randn(
        SPEC.n_sequences, metadata["ctx"] - 1, directions.shape[1], generator=generator
    )
    clean_losses = sequence_lm_losses(language_model, tokens, hook=args.hook, skip_bos=True)

    arms: dict[str, dict] = {}
    raw: dict[str, np.ndarray] = {}
    cell_reports: dict[str, dict] = {}
    # Accumulators for the single experiment-level statistic: the primary
    # operating point, pooled over the target quantiles.
    primary_effects: dict[str, np.ndarray] = {}
    primary_rows: list[dict[str, np.ndarray]] = []
    primary_clamp_rows: list[dict[str, np.ndarray]] = []
    primary_diagnostics: list[dict] = []
    primary_validity: dict[str, dict] = {}
    for quantile in SPEC.target_quantiles:
        label = f"q{int(round(quantile * 100)):02d}"
        target = torch.from_numpy(
            target_coordinates(stats, quantile)[assignment].astype(np.float32)
        ).to(device)

        # Arm A: hard clamp only, NFE 0.
        clamp_rows, clamp_nll, clamp_delta, clamp_transform = _run(
            flow, language_model, tokens, sequence_directions, target, noise,
            # nfe is validated but never spent: t_start = 0 is the exact clamp.
            clean_losses, hook=args.hook, t_start=0.0, nfe=1, objective=objective,
        )
        arms[f"{label}_clamp_only"] = {
            "arm": "clamp_only",
            "target_quantile": quantile,
            "mean_delta_lm_vs_clean": float(clamp_delta.mean()),
            **naturalization_summary(
                clamp_rows, clamp_nll, clamp_nll, assignment,
                bootstrap_seed=SPEC.bootstrap_seed,
                bootstrap_resamples=SPEC.bootstrap_resamples,
                confidence=SPEC.confidence,
            ),
            **clamp_transform.diagnostics(),
        }
        for field, values in clamp_rows.items():
            raw[f"{label}_clamp_only_{field}"] = values

        # Arm B: the same clamp, then the tangent-trained flow.
        for t_start in t2_spec.t_start:
            for nfe in t2_spec.nfe:
                rows, nll, delta, transform = _run(
                    flow, language_model, tokens, sequence_directions, target, noise,
                    clean_losses, hook=args.hook, t_start=t_start, nfe=nfe,
                    objective=objective,
                )
                tag = f"{label}_t{t_start:.2f}_nfe{nfe}_{t2_spec.arm_label}"
                entry = {
                    "arm": "clamp_plus_tangent_flow",
                    "target_quantile": quantile,
                    "mean_delta_lm_vs_clean": float(delta.mean()),
                    **naturalization_summary(
                        rows, nll, clamp_nll, assignment,
                        bootstrap_seed=SPEC.bootstrap_seed,
                        bootstrap_resamples=SPEC.bootstrap_resamples,
                        confidence=SPEC.confidence,
                    ),
                    **transform.diagnostics(),
                    # Refuses to report a cell whose arms drifted apart in
                    # coordinate; an attenuated "win" is not a result.
                    "coordinate_match": assert_coordinate_match(clamp_rows, rows),
                }
                requested = rows["c_target"] - rows["c0_clean"]
                up = requested > 0
                for name, mask in (("up", up), ("down", ~up)):
                    if mask.sum() > 0:
                        entry[f"{name}_coordinate_abs_error_mean"] = float(
                            rows["coordinate_abs_error"][mask].mean()
                        )
                        entry[f"{name}_orthogonal_correction_norm_mean"] = float(
                            rows["orthogonal_correction_norm"][mask].mean()
                        )
                        entry[f"{name}_n_rows"] = int(mask.sum())
                arms[tag] = entry
                cell_reports[tag] = t2_cell_report(entry)
                if (
                    t_start == t2_spec.primary_t_start
                    and nfe == t2_spec.primary_nfe
                ):
                    # One paired effect per validation SEQUENCE, aligned across
                    # quantiles so the bootstrap can keep a sequence's five
                    # quantile observations together.
                    primary_effects[label] = nll - clamp_nll
                    primary_rows.append(rows)
                    primary_clamp_rows.append(clamp_rows)
                    primary_diagnostics.append(transform.diagnostics())
                    primary_validity[label] = entry["coordinate_match"]
                for field, values in rows.items():
                    raw[f"{tag}_{field}"] = values
        print(f"finished target {label}")

    # ------------------------------------------------------------------
    # THE experiment-level statistic: one frozen cell, pooled over quantiles.
    # ------------------------------------------------------------------
    if not primary_effects:
        raise RuntimeError("the frozen primary cell produced no data")
    pooled_rows = concatenate(primary_rows)
    pooled_clamp_rows = concatenate(primary_clamp_rows)
    geometry = {
        "target_quantiles": list(SPEC.target_quantiles),
        "t_start": t2_spec.primary_t_start,
        "nfe": t2_spec.primary_nfe,
        "coordinate_abs_error_mean": float(pooled_rows["coordinate_abs_error"].mean()),
        "coordinate_abs_error_max": float(pooled_rows["coordinate_abs_error"].max()),
        "orthogonal_correction_norm_mean": float(
            pooled_rows["orthogonal_correction_norm"].mean()
        ),
        "parallel_correction_norm_mean": float(
            pooled_rows["parallel_correction_norm"].mean()
        ),
        "relative_l2_to_clean_mean": float(pooled_rows["relative_l2_to_clean"].mean()),
        "cosine_to_clean_mean": float(pooled_rows["cosine_to_clean"].mean()),
        # Every quantile cell already passed assert_coordinate_match individually;
        # this is the pooled restatement of that invariant.
        "coordinate_match": assert_coordinate_match(pooled_clamp_rows, pooled_rows),
        "per_quantile_coordinate_match": primary_validity,
        "network_evaluations": sum(
            int(d["network_evaluations"]) for d in primary_diagnostics
        ),
        "projections": sum(int(d["projections"]) for d in primary_diagnostics),
        "max_coordinate_drift": max(
            float(d["max_coordinate_drift"]) for d in primary_diagnostics
        ),
        "max_pre_projection_drift": max(
            float(d["max_pre_projection_drift"]) for d in primary_diagnostics
        ),
        "raw_parallel_velocity_norm_mean": float(
            np.mean([d["raw_parallel_velocity_norm_mean"] for d in primary_diagnostics])
        ),
    }
    # Formal statistic: equal quantile weight; bootstrap keeps each drawn
    # sequence's five quantile observations together (frozen 2026-08-16).
    pooled = t2_pooled_cell(
        primary_effects, assignment, geometry,
        seed=SPEC.bootstrap_seed,
        n_resamples=SPEC.bootstrap_resamples,
        confidence=SPEC.confidence,
    )
    arms[t2_spec.primary_cell()] = pooled
    verdict = t2_experiment_verdict(
        pooled,
        spec=t2_spec,
        formal_eligible=not args.overwrite_debug_mode,
        ineligible_reason="debug-mode run" if args.overwrite_debug_mode else None,
    )

    payload = {
        "experiment": "tangent_naturalization_t2_v1",
        "question": (
            "at a semantic coordinate held exactly fixed, does the tangent-trained "
            "flow lower LM NLL relative to a hard clamp alone?"
        ),
        "primary_statistic": "NLL_tangent_flow - NLL_clamp (negative = useful)",
        "stop_rule": STOP_RULE,
        "spec": spec_payload(SPEC),
        "grid": {"t_start": list(t2_spec.t_start), "nfe": list(t2_spec.nfe)},
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": meta.get("step"),
        "checkpoint_objective_identity": meta.get("objective_identity"),
        "t1_receipt": t1_receipt,
        "debug_mode": bool(args.overwrite_debug_mode),
        "validation_artifact": bundle.identity,
        "clean_lm_loss": float(clean_losses.mean()),
        "direction_pool": pool_receipt,
        "direction_pool_indices": picked.tolist(),
        "sequence_ids": sequence_ids.tolist(),
        "direction_assignment": assignment.tolist(),
        "natural_coordinate_stats": stats,
        "arms": arms,
        "t2_experiment_verdict": verdict,
        "t2_cell_diagnostics": cell_reports,
        "decision_rule": (
            "ONE frozen operating point (t_start "
            f"{t2_spec.primary_t_start:.2f}, NFE {t2_spec.primary_nfe}) pooled "
            "across the five natural-support target quantiles. Per-cell entries "
            "in t2_cell_diagnostics are diagnostics: a favourable cell there is "
            "NOT a T2 pass, because 30 cells guarantee some favourable ones."
        ),
        "comparable_baseline": "results/constrained_naturalization_v1/ (same frozen plan)",
        "sae_metrics": "not run: pool artifact carries no SAE feature ids",
        "lexicon_metrics": "not run: lexicons exist only for DEV vectors",
        "degeneration_metrics": "not run: diagnostic substitutes activations, does not generate",
        "dev_vectors_accessed": False,
        "held_out_accessed": False,
        "llm_judge_used": False,
        "trained_anything": False,
    }
    report = args.out_dir / "tangent_naturalization.json"
    report.write_text(json.dumps(payload, indent=2))
    np.savez_compressed(args.out_dir / "raw_rows.npz", **raw)
    print(f"wrote {report}")
    print(f"T2 experiment verdict ({verdict['primary_cell']}): {verdict['verdict']}")
    print(f"  paired delta NLL = {verdict['paired_delta_nll_mean']:+.5f} nats "
          f"CI {verdict['paired_delta_nll_ci'][0]:+.5f}..{verdict['paired_delta_nll_ci'][1]:+.5f}")
    favourable = sum(
        1 for report in cell_reports.values() if report["cell_favourable"]
    )
    print(f"  ({favourable}/{len(cell_reports)} diagnostic cells favourable; "
          "this is NOT the decision)")


if __name__ == "__main__":
    main()
