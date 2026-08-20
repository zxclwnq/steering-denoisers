"""Run one frozen DEV-only operational smoke for the Phase B steering hook."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import math
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from interp.activations import file_sha256
from interp.flow_steering import FlowGenerationSession, FlowNoiseCell
from interp.model import (
    MODEL_NAME,
    MODEL_RESOLVED_NAME,
    MODEL_REVISION,
    load_model,
    resolve_device,
)
from interp.phase_a import (
    lifecycle_record,
    load_phase_a_config,
    validate_checkpoint_payload,
    validate_frozen_checkpoint,
    write_failure_receipt,
    write_immutable_json,
)
from interp.phase_b import (
    load_frozen_sae,
    load_phase_b_config,
    operational_smoke_claims,
    prepare_smoke_input,
    validate_phase_a_evidence,
)
from interp.provenance import source_revision
from interp.train_flow import load_flow_checkpoint


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _finite_nested(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite_nested(item) for item in value.values())
    if isinstance(value, float):
        return math.isfinite(value)
    return True


@torch.no_grad()
def _smoke_once(model, flow, direction: torch.Tensor, cfg) -> dict:  # noqa: ANN001
    smoke = cfg.smoke
    prompt = cfg.prompts[smoke.prompt_id]
    tokens, prepend_bos = prepare_smoke_input(model, prompt)
    alpha = cfg.activation_norm_mean * smoke.alpha_hat
    cell = FlowNoiseCell(
        vector=smoke.vector,
        alpha=alpha,
        prompt_id=smoke.prompt_id,
        generation_seed=smoke.generation_seed,
    )
    session = FlowGenerationSession(
        flow,
        direction,
        alpha=alpha,
        t_start=smoke.t_start,
        nfe=smoke.nfe,
        cells=(cell,),
        prompt_width=tokens.shape[1],
        max_new_tokens=smoke.max_new_tokens,
        off_distribution_norm=cfg.off_distribution_norm,
        noise_namespace=cfg.noise_namespace,
    )

    torch.manual_seed(smoke.generation_seed)
    if tokens.device.type == "cuda":
        torch.cuda.manual_seed_all(smoke.generation_seed)

    def apply(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001, ARG001
        return session.apply(act)

    with model.hooks(fwd_hooks=[(cfg.hook, apply)]):
        generated = model.generate(
            tokens,
            max_new_tokens=smoke.max_new_tokens,
            stop_at_eos=False,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k if cfg.top_k > 0 else None,
            freq_penalty=cfg.freq_penalty,
            use_past_kv_cache=True,
            prepend_bos=prepend_bos,
            return_type="tokens",
            verbose=False,
            output_logits=True,
            return_dict_in_generate=True,
        )
    sequences = generated.sequences
    logits = torch.stack(generated.logits, dim=1)
    if sequences.shape != (1, tokens.shape[1] + smoke.max_new_tokens):
        raise ValueError("smoke generation returned an unexpected token shape")
    if logits.shape[:2] != (1, smoke.max_new_tokens) or not torch.isfinite(logits).all():
        raise ValueError("smoke generation returned non-finite or malformed logits")
    receipt = session.receipt()
    expected_calls = smoke.max_new_tokens
    if receipt["hook_calls"] != expected_calls:
        raise ValueError("smoke hook-call count differs from TransformerLens cached generation")
    if receipt["flow_network_evaluations"] != expected_calls * smoke.nfe:
        raise ValueError("smoke flow-network evaluation count differs from the frozen NFE")
    if not _finite_nested(receipt):
        raise ValueError("smoke steering geometry is non-finite")
    return {
        "prompt_token_count": int(tokens.shape[1]),
        "generated_token_count": smoke.max_new_tokens,
        "generated_tokens_sha256": _tensor_sha256(sequences[:, tokens.shape[1] :]),
        "generation_logits_sha256": _tensor_sha256(logits),
        "receipt": receipt,
    }


def _run(args: argparse.Namespace) -> dict:
    cfg = load_phase_b_config(args.config)
    phase_a = load_phase_a_config(args.phase_a_config)
    phase_a_evidence = validate_phase_a_evidence(
        cfg, args.phase_a_config, args.phase_a_report
    )
    selection = validate_frozen_checkpoint(phase_a.checkpoint_path, phase_a)
    if selection["checkpoint_sha256"] != cfg.checkpoint_sha256:
        raise ValueError("revalidated checkpoint differs from the Phase B freeze")

    device = resolve_device(args.device)
    flow, checkpoint_metadata, _ = load_flow_checkpoint(phase_a.checkpoint_path, device)
    checkpoint_payload = validate_checkpoint_payload(checkpoint_metadata, phase_a)
    if flow.cfg.d_model != cfg.d_model:
        raise ValueError("flow width differs from the frozen Phase B hook width")

    sae = load_frozen_sae(cfg.sae, args.sae_dir, device="cpu")
    vector = next((item for item in cfg.vectors if item.name == cfg.smoke.vector), None)
    if vector is None or vector.split != "dev":
        raise ValueError("predetermined smoke vector is absent from the DEV manifest")
    direction = sae.decoder_directions([vector.feature])[0].to(device)
    direction_norm = float(direction.double().norm())
    if not math.isclose(direction_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("smoke SAE direction is not unit normalized")

    language_model = load_model(str(device))
    if (
        cfg.model_name != MODEL_NAME
        or cfg.resolved_model_name != MODEL_RESOLVED_NAME
        or cfg.model_revision != MODEL_REVISION
        or language_model.cfg.d_model != cfg.d_model
        or cfg.hook not in language_model.hook_dict
    ):
        raise ValueError("loaded GPT-2 does not match the frozen Phase B model interface")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    first = _smoke_once(language_model, flow, direction, cfg)
    second = _smoke_once(language_model, flow, direction, cfg)
    if first != second:
        raise ValueError("repeated Phase B smoke was not deterministic")

    return {
        "status": "complete",
        "research_status": "NOT_EVALUATED",
        "experiment_id": cfg.experiment_id,
        "experiment_class": cfg.experiment_class,
        "source_revision": source_revision(),
        "config": {
            "path": str(args.config),
            "sha256": file_sha256(args.config),
            "frozen": cfg.raw,
        },
        "phase_a_evidence": phase_a_evidence,
        "checkpoint_selection": selection,
        "checkpoint_payload": checkpoint_payload,
        "model": {
            "name": MODEL_NAME,
            "resolved_name": MODEL_RESOLVED_NAME,
            "revision": MODEL_REVISION,
            "hook": cfg.hook,
        },
        "sae": {
            "release": cfg.sae.release,
            "repo_id": cfg.sae.repo_id,
            "revision": cfg.sae.revision,
            "hook": cfg.sae.hook,
            "config_path": str(args.sae_dir / cfg.sae.config_filename),
            "config_sha256": cfg.sae.config_sha256,
            "weights_path": str(args.sae_dir / cfg.sae.weights_filename),
            "weights_sha256": cfg.sae.weights_sha256,
            "feature": vector.feature,
            "direction_sha256": _tensor_sha256(direction.float()),
            "direction_norm": direction_norm,
        },
        "protected_data": {
            "dev_directions_loaded": [vector.name],
            "final_evaluation_directions_loaded": False,
            "held_out_accessed": False,
        },
        "smoke": {
            **operational_smoke_claims(),
            "vector": vector.name,
            "prompt_id": cfg.smoke.prompt_id,
            "generation_seed": cfg.smoke.generation_seed,
            "alpha": cfg.activation_norm_mean * cfg.smoke.alpha_hat,
            "alpha_hat": cfg.smoke.alpha_hat,
            "t_start": cfg.smoke.t_start,
            "nfe": cfg.smoke.nfe,
            "repeat_count": 2,
            "deterministic_repeat": True,
            **first,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformer_lens": importlib.metadata.version("transformer-lens"),
            "transformers": importlib.metadata.version("transformers"),
            "safetensors": importlib.metadata.version("safetensors"),
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase-a-config", required=True, type=Path)
    parser.add_argument("--phase-a-report", required=True, type=Path)
    parser.add_argument("--sae-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    command = [sys.executable, *sys.argv]
    started_utc = datetime.now(UTC).isoformat()

    try:
        report = _run(args)
        report.update(
            lifecycle_record(
                status="complete",
                command=command,
                started_utc=started_utc,
                finished_utc=datetime.now(UTC).isoformat(),
            )
        )
        write_immutable_json(args.output, report)
    except KeyboardInterrupt as error:
        write_failure_receipt(
            args.output,
            error,
            status="INTERRUPTED",
            command=command,
            started_utc=started_utc,
        )
        raise
    except Exception as error:
        write_failure_receipt(
            args.output,
            error,
            status="INVALID",
            command=command,
            started_utc=started_utc,
        )
        raise

    print(f"Phase B operational smoke complete: {args.output}")


if __name__ == "__main__":
    main()
