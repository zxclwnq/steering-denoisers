"""Experiment D, phase 1: prove the refusal direction is causal, here, first.

A positive control has to earn the name. Before any geometry is computed, this
script reproduces both halves of the published behavioural effect on an
independent test split:

* ablating the direction at every layer must stop the model refusing harmful
  instructions;
* adding it at layer 10 must make the model refuse harmless ones.

Both are required. Removing refusal alone would not show mediation -- a large
enough perturbation degrades any model -- and inducing refusal alone would not
show the direction is what the model uses. If either half fails, the verdict is
CAUSAL_CONTROL_FAIL and `scripts/refusal_geometry.py` refuses to run.

Everything selectable is frozen in `docs/EXPERIMENT_D_PROTOCOL.md` and
`interp.refusal_control.REFUSAL_CONTROL_SPEC`, both fixed before any D number
existed. The direction is the published artifact, verified by content hash.

Greedy decoding, so the completions are deterministic given the model.

    uv run python scripts/refusal_causal_validation.py \
        --direction data/refusal_direction/gemma-2b-it_direction.pt \
        --metadata data/refusal_direction/gemma-2b-it_direction_metadata.json \
        --splits data/refusal_direction/splits \
        --out-dir results/refusal_causal_validation_d_v1
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from interp.provenance import source_revision
from interp.refusal_control import (
    REFUSAL_CONTROL_SPEC,
    ablation_intervention,
    addition_intervention,
    causal_validation_verdict,
    is_refusal,
    load_refusal_direction,
    refusal_rate,
    refusal_score,
    spec_payload,
    tokenize_instructions,
)
from interp.tangent_eval import require_fresh_output_dir

SPEC = REFUSAL_CONTROL_SPEC


def load_split(splits_dir: Path, name: str) -> list[dict]:
    return json.loads((Path(splits_dir) / f"{name}.json").read_text())


def sample_prompts(dataset: list[dict], n: int, seed: int) -> list[dict]:
    """Frozen subsample. `random.sample` with a fixed seed, as upstream does."""

    rng = random.Random(seed)
    return rng.sample(dataset, min(n, len(dataset)))


@torch.no_grad()
def mean_refusal_score(model, tokenizer, instructions: list[str], intervention) -> float:
    """Logit-level refusal readout, averaged over prompts."""

    scores = []
    for start in range(0, len(instructions), SPEC.scoring_batch_size):
        batch = instructions[start : start + SPEC.scoring_batch_size]
        tokens = tokenize_instructions(tokenizer, batch)
        with intervention():
            logits = model(
                input_ids=tokens.input_ids.to(model.device),
                attention_mask=tokens.attention_mask.to(model.device),
            ).logits
        scores.append(refusal_score(logits).cpu())
    return float(torch.cat(scores).mean())


@torch.no_grad()
def generate(model, tokenizer, instructions: list[str], intervention) -> list[str]:
    """Greedy completions, upstream's generation config."""

    config = GenerationConfig(max_new_tokens=SPEC.max_new_tokens, do_sample=False)
    config.pad_token_id = tokenizer.pad_token_id
    completions: list[str] = []
    for start in range(0, len(instructions), SPEC.generation_batch_size):
        batch = instructions[start : start + SPEC.generation_batch_size]
        tokens = tokenize_instructions(tokenizer, batch)
        with intervention():
            produced = model.generate(
                input_ids=tokens.input_ids.to(model.device),
                attention_mask=tokens.attention_mask.to(model.device),
                generation_config=config,
            )
        produced = produced[:, tokens.input_ids.shape[-1] :]
        completions.extend(
            tokenizer.decode(row, skip_special_tokens=True).strip() for row in produced
        )
        print(f"  generated {len(completions)}/{len(instructions)}", flush=True)
    return completions


def null_intervention():
    import contextlib

    return contextlib.nullcontext()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-path", default=SPEC.model_path)
    parser.add_argument("--overwrite-debug-mode", action="store_true")
    args = parser.parse_args()

    require_fresh_output_dir(args.out_dir, overwrite_debug=args.overwrite_debug_mode)
    direction = load_refusal_direction(args.direction, args.metadata, spec=SPEC)
    print(f"direction verified: {direction.receipt}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    model.requires_grad_(False)

    unit = direction.unit.to(model.device)
    raw = direction.raw.to(model.device)

    harmful = sample_prompts(
        load_split(args.splits, "harmful_test"), SPEC.n_eval_prompts, SPEC.eval_sample_seed
    )
    harmless = sample_prompts(
        load_split(args.splits, "harmless_test"), SPEC.n_eval_prompts, SPEC.eval_sample_seed
    )
    harmful_instructions = [row["instruction"] for row in harmful]
    harmless_instructions = [row["instruction"] for row in harmless]

    def ablation():
        return ablation_intervention(model, unit)

    def addition():
        return addition_intervention(
            model, raw, direction.layer, SPEC.addition_coefficient
        )

    print("scoring harmful baseline...", flush=True)
    harmful_baseline_score = mean_refusal_score(
        model, tokenizer, harmful_instructions, null_intervention
    )
    print("scoring harmful ablated...", flush=True)
    harmful_ablated_score = mean_refusal_score(
        model, tokenizer, harmful_instructions, ablation
    )
    print("scoring harmless baseline...", flush=True)
    harmless_baseline_score = mean_refusal_score(
        model, tokenizer, harmless_instructions, null_intervention
    )
    print("scoring harmless with addition...", flush=True)
    harmless_addition_score = mean_refusal_score(
        model, tokenizer, harmless_instructions, addition
    )

    print("generating harmful baseline...", flush=True)
    harmful_baseline = generate(model, tokenizer, harmful_instructions, null_intervention)
    print("generating harmful ablated...", flush=True)
    harmful_ablated = generate(model, tokenizer, harmful_instructions, ablation)
    print("generating harmless baseline...", flush=True)
    harmless_baseline = generate(model, tokenizer, harmless_instructions, null_intervention)
    print("generating harmless with addition...", flush=True)
    harmless_addition = generate(model, tokenizer, harmless_instructions, addition)

    verdict = causal_validation_verdict(
        harmful_baseline_rate=refusal_rate(harmful_baseline),
        harmful_ablated_rate=refusal_rate(harmful_ablated),
        harmless_baseline_rate=refusal_rate(harmless_baseline),
        harmless_addition_rate=refusal_rate(harmless_addition),
        harmful_baseline_score=harmful_baseline_score,
        harmful_ablated_score=harmful_ablated_score,
        harmless_baseline_score=harmless_baseline_score,
        harmless_addition_score=harmless_addition_score,
        spec=SPEC,
    )

    arms = {
        "harmful_baseline": (harmful_instructions, harmful_baseline),
        "harmful_ablated": (harmful_instructions, harmful_ablated),
        "harmless_baseline": (harmless_instructions, harmless_baseline),
        "harmless_addition": (harmless_instructions, harmless_addition),
    }
    payload = {
        "experiment": SPEC.version,
        "phase": "causal_validation",
        "class": "post_stop_method_development",
        "question": (
            "does the published refusal direction reproduce its behavioural effect "
            "on this machine, in both directions, before any geometry is computed?"
        ),
        "protocol": "docs/EXPERIMENT_D_PROTOCOL.md",
        "spec": spec_payload(SPEC),
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "debug_mode": bool(args.overwrite_debug_mode),
        "model_path": args.model_path,
        "torch_dtype": "bfloat16",
        "device": str(model.device),
        "direction": direction.receipt,
        "n_harmful": len(harmful_instructions),
        "n_harmless": len(harmless_instructions),
        "eval_split": "test (disjoint from the train/val splits the direction used)",
        "verdict": verdict,
        "completions": {
            name: [
                {
                    "prompt": prompt,
                    "response": response,
                    "is_refusal": bool(is_refusal(response)),
                }
                for prompt, response in zip(prompts, responses, strict=True)
            ]
            for name, (prompts, responses) in arms.items()
        },
        "geometry_metrics_used_for_selection": False,
        "dev_vectors_accessed": False,
        "held_out_accessed": False,
        "llm_judge_used": False,
        "trained_anything": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "causal_validation.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
