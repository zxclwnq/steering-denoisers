"""Evaluate `hard clamp + tangent flow` on the frozen DEV steering protocol.

Why this exists
---------------
The T1/T2 evaluations substitute activations and score NLL; they never generate
text, so the tangent method had no Dist-1/2/3 or repetition numbers, and no
point on the DEV concept/quality Pareto. This script closes that gap by running
the branch's final method through the *same* frozen DEV protocol as additive,
the assignment's Gaussian denoiser, shrinkage and the isotropic flow arms:
identical prompts, alpha grid, seeds, decoding settings and metrics.

The intervention
----------------
At each generated position, the activation is hard-clamped to the
additive-equivalent coordinate and then naturalized by the tangent flow, with
the coordinate held exactly:

    c_target = <h, v> + alpha        (so the clamp alone is exactly h + alpha*v)
    h_out    = clamp_then_tangent_flow(h_clamp, v, c_target)

Note the clamp arm on this grid *is* additive by construction, which is why only
the flow arm is generated here.

Status
------
POST-HOC. This cell is not part of any frozen protocol: the DEV protocol was
frozen before the tangent branch existed. It is reported as a descriptive arm
alongside the frozen ones and carries no preregistered gate. It cannot change
the frozen T2 result.

    uv run python scripts/eval_tangent_on_dev.py \
        --config configs/flow_phase_b_dev_v1.yaml \
        --checkpoint <tangent checkpoint> \
        --t-start 0.10 --nfe 1 \
        --out-dir results/tangent_dev_posthoc_v1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from interp.activations import file_sha256
from interp.conditional_flow import ConditionalFlowMatcher
from interp.evaluation_metrics import score_continuations, text_metrics
from interp.flow_steering import FlowGenerationSession, FlowNoiseCell, FlowSteeringOutput
from interp.model import load_model, resolve_device
from interp.phase_b import load_frozen_sae, load_phase_b_config
from interp.provenance import source_revision
from interp.tangent_flow import TANGENT_OBJECTIVE, clamp_then_tangent_flow
from interp.train_flow import load_flow_checkpoint


class TangentGenerationSession(FlowGenerationSession):
    """The frozen generation session with the tangent transform substituted in.

    Only ``_steer`` changes. Matched per-cell noise, position bookkeeping,
    receipts, geometry aggregation and the off-distribution guard are inherited
    unchanged from the reviewed base class, so this arm is generated under
    exactly the same mechanics as the frozen ones.
    """

    @torch.no_grad()
    def _steer(self, valid_h: torch.Tensor, valid_noise: torch.Tensor) -> FlowSteeringOutput:
        vector = self.direction.to(device=valid_h.device, dtype=valid_h.dtype)
        rows = vector[None, :].expand(valid_h.shape[0], vector.shape[0])
        # Additive-equivalent target: clamping to it reproduces h + alpha*v
        # exactly, so the flow is the only thing that differs from additive.
        c_target = (valid_h * rows).sum(dim=-1, keepdim=True) + self.alpha
        additive = valid_h + self.alpha * vector
        out = clamp_then_tangent_flow(
            self.model,
            valid_h,
            rows,
            c_target,
            noise=valid_noise,
            t_start=self.t_start,
            nfe=self.nfe,
        )
        # Same off-distribution rule as the frozen arms: if the additive state is
        # already far outside the activation shell, fall back to it untouched.
        guarded = additive.norm(dim=-1) > self.off_distribution_norm
        activation = torch.where(guarded.unsqueeze(-1), additive, out.activation)
        if not torch.isfinite(activation).all():
            raise ValueError("tangent-steered activation became non-finite")
        return FlowSteeringOutput(
            activation=activation,
            guarded=guarded,
            network_evaluations=out.network_evaluations,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sae-dir", required=True, type=Path,
                        help="directory holding the SHA-pinned frozen SAE files")
    parser.add_argument("--t-start", type=float, default=0.10)
    parser.add_argument("--nfe", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    final = args.out_dir / "tangent_dev.jsonl"
    if final.exists():
        raise FileExistsError(f"{final} already exists; results are immutable")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    cfg = load_phase_b_config(args.config)
    model = load_model(str(device))
    sae = load_frozen_sae(cfg.sae, args.sae_dir, device=str(device))
    flow, meta, _ = load_flow_checkpoint(
        args.checkpoint, device=device, expected_objective=TANGENT_OBJECTIVE
    )
    if not isinstance(flow, ConditionalFlowMatcher):
        raise ValueError("the tangent objective requires a conditional flow checkpoint")
    flow.eval()

    control_features = {
        v.name: [o.feature for o in cfg.vectors if o.name != v.name][:4]
        for v in cfg.vectors
    }
    grid = tuple(cfg.alpha_hat) + tuple(cfg.alpha_hat_stress)
    shards = args.out_dir / "shards"
    shards.mkdir(exist_ok=True)
    rows: list[dict] = []
    for vector in cfg.vectors:
        shard = shards / f"{vector.name}.jsonl"
        if shard.exists():          # resume: this vector already completed
            with shard.open() as handle:
                done = [json.loads(line) for line in handle]
            rows.extend(done)
            print(f"skip complete vector {vector.name} ({len(done)} rows)")
            continue
        vector_rows: list[dict] = []
        direction = sae.decoder_directions([vector.feature])[0].to(device)
        for alpha_hat in grid:
            alpha = float(alpha_hat) * cfg.activation_norm_mean
            for seed in cfg.generation_seeds:
                prompt_ids = tuple(range(len(cfg.prompts)))
                prompts = [cfg.prompts[i] for i in prompt_ids]
                tokens = model.to_tokens(prompts)
                cells = tuple(
                    FlowNoiseCell(vector.name, alpha, pid, seed) for pid in prompt_ids
                )
                session = TangentGenerationSession(
                    flow, direction, alpha=alpha, t_start=args.t_start, nfe=args.nfe,
                    cells=cells, prompt_width=tokens.shape[1],
                    max_new_tokens=cfg.max_new_tokens,
                    off_distribution_norm=cfg.off_distribution_norm,
                    noise_namespace=cfg.noise_namespace,
                )
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)

                def apply(activation: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001, ARG001, B023
                    return session.apply(activation)  # noqa: B023

                with model.hooks(fwd_hooks=[(cfg.hook, apply)]):
                    generated = model.generate(
                        tokens, max_new_tokens=cfg.max_new_tokens, stop_at_eos=False,
                        do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
                        top_k=cfg.top_k if cfg.top_k > 0 else None,
                        freq_penalty=cfg.freq_penalty, use_past_kv_cache=True,
                        return_type="tokens", verbose=False,
                    )
                continuations = model.tokenizer.batch_decode(
                    generated[:, tokens.shape[1]:], skip_special_tokens=True
                )
                # Same scoring path as the frozen arms: clean GPT-2 conditional
                # NLL plus target/control SAE activation on the continuation.
                feature_ids = [vector.feature, *control_features[vector.name]]
                scored = score_continuations(
                    model, sae, cfg.hook, prompts, continuations, feature_ids,
                    batch_size=16,
                )
                for pid, continuation, row in zip(
                    prompt_ids, continuations, scored, strict=True
                ):
                    tm = text_metrics(continuation, vector.lexicon)
                    controls = row.sae_feature_means[1:]
                    vector_rows.append({
                        "method": "clamp_plus_tangent",
                        "arm_id": f"tangent_t{args.t_start:.2f}_nfe{args.nfe}",
                        "vector": vector.name, "feature": vector.feature,
                        "alpha": alpha, "alpha_hat": float(alpha_hat),
                        "is_stress": float(alpha_hat) in tuple(cfg.alpha_hat_stress),
                        "prompt_id": pid, "generation_seed": seed,
                        "continuation": continuation,
                        "metrics": {
                            **asdict(tm),
                            "nll": row.nll,
                            "sae_act_target": row.sae_feature_means[0],
                            "sae_act_control_mean": sum(controls) / len(controls),
                            "sae_act_control_max": max(controls),
                        },
                    })
        with shard.open("w") as handle:
            for row in vector_rows:
                handle.write(json.dumps(row) + "\n")
        rows.extend(vector_rows)
        print(f"finished vector {vector.name}: +{len(vector_rows)} rows "
              f"({len(rows)} total)", flush=True)

    out = args.out_dir / "tangent_dev.jsonl"
    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (args.out_dir / "meta.json").write_text(json.dumps({
        "status": "post_hoc_descriptive",
        "not_a_frozen_protocol_arm": True,
        "cannot_change_frozen_t2_result": True,
        "method": "clamp_plus_tangent (additive-equivalent coordinate)",
        "t_start": args.t_start, "nfe": args.nfe,
        "row_count": len(rows),
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_objective": meta.get("objective_identity"),
        "config": str(args.config),
        "decoding": {"max_new_tokens": cfg.max_new_tokens, "temperature": cfg.temperature,
                     "top_p": cfg.top_p, "top_k": cfg.top_k},
        "held_out_accessed": False, "llm_judge_used": False, "trained_anything": False,
    }, indent=2) + "\n")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
