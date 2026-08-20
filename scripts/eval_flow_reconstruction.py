"""Run the frozen validation-only Phase A functional reconstruction gate."""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from interp.activations import (
    file_sha256,
    load_activations,
    load_validation_report,
    validate_activation_metadata,
)
from interp.data import load_tokens, token_cache_path
from interp.model import (
    MODEL_NAME,
    MODEL_RESOLVED_NAME,
    MODEL_REVISION,
    load_model,
    resolve_device,
)
from interp.phase_a import (
    evaluate_phase_a,
    lifecycle_record,
    load_phase_a_config,
    select_validation_sequence_ids,
    validate_checkpoint_payload,
    validate_frozen_checkpoint,
    write_failure_receipt,
    write_immutable_json,
)
from interp.provenance import source_revision
from interp.train_flow import load_flow_checkpoint


def _run(args: argparse.Namespace) -> dict:
    cfg = load_phase_a_config(args.config)
    selection_receipt = validate_frozen_checkpoint(cfg.checkpoint_path, cfg)

    validation = load_validation_report(
        cfg.dataset_name,
        args.activation_dir,
        expected_split_fingerprint=cfg.split_fingerprint,
        verify_hashes=True,
    )
    if validation.get("sha256") != cfg.artifact_sha256:
        raise ValueError("validated activation artifact SHA set differs from Phase A freeze")
    if validation.get("token_cache_sha256") != cfg.token_cache_sha256:
        raise ValueError("validated token-cache SHA differs from Phase A freeze")
    dataset = load_activations(cfg.dataset_name, args.activation_dir)
    dataset = replace(dataset, meta={**dataset.meta, "full_validation_report": validation})
    validate_activation_metadata(
        dataset,
        expected_name=cfg.dataset_name,
        expected_split=cfg.activation_split,
        expected_model=cfg.model_name,
        expected_resolved_model_name=cfg.resolved_model_name,
        expected_model_revision=cfg.model_revision,
        expected_hook=cfg.hook,
        expected_ctx=cfg.ctx,
        expected_d_model=768,
        expected_dataset_repository=cfg.dataset_repository,
        expected_dataset_config=cfg.dataset_config,
        expected_dataset_revision=cfg.dataset_revision,
        expected_tokenizer=cfg.tokenizer,
    )

    device = resolve_device(args.device)
    flow, checkpoint_metadata, _ = load_flow_checkpoint(cfg.checkpoint_path, device)
    checkpoint_payload_receipt = validate_checkpoint_payload(checkpoint_metadata, cfg)
    if flow.cfg.d_model != dataset.d_model:
        raise ValueError(
            f"flow width {flow.cfg.d_model} != validated activation width {dataset.d_model}"
        )
    if (
        not torch.isfinite(flow.normalizer.mean).all()
        or not torch.isfinite(flow.normalizer.std).all()
        or not bool((flow.normalizer.std > 0).all())
    ):
        raise ValueError("checkpoint normalizer must be finite with strictly positive std")

    language_model = load_model(str(device))
    if cfg.model_name != MODEL_NAME or cfg.resolved_model_name != MODEL_RESOLVED_NAME:
        raise ValueError("canonical model name differs from the Phase A freeze")
    if cfg.model_revision != MODEL_REVISION:
        raise ValueError("canonical model revision differs from the Phase A freeze")
    cache_path = token_cache_path(
        cfg.activation_split,
        int(dataset.meta["n_seqs"]),
        cfg.ctx,
        language_model.tokenizer,
        args.token_cache_dir,
    )
    tokens = load_tokens(
        cfg.activation_split,
        int(dataset.meta["n_seqs"]),
        language_model.tokenizer,
        ctx=cfg.ctx,
        cache_dir=args.token_cache_dir,
    )
    observed_token_sha = file_sha256(cache_path)
    if observed_token_sha != cfg.token_cache_sha256:
        raise ValueError(
            f"token-cache SHA {observed_token_sha} != frozen {cfg.token_cache_sha256}"
        )
    sequence_ids = select_validation_sequence_ids(len(dataset), cfg)
    selected_tokens = tokens[torch.from_numpy(sequence_ids)]
    if selected_tokens.shape != (cfg.n_sequences, cfg.ctx):
        raise ValueError("selected Phase A token matrix has the wrong shape")

    evaluation = evaluate_phase_a(
        flow, language_model, selected_tokens, sequence_ids, cfg
    )
    now = datetime.now(UTC).isoformat()
    return {
        "status": "complete",
        "research_status": evaluation["research_status"],
        "generated_utc": now,
        "experiment_id": cfg.experiment_id,
        "experiment_class": cfg.experiment_class,
        "source_revision": source_revision(),
        "config": {
            "path": str(args.config),
            "sha256": file_sha256(args.config),
            "frozen": cfg.raw,
        },
        "checkpoint_selection": selection_receipt,
        "checkpoint_payload": checkpoint_payload_receipt,
        "activation_artifact": {
            "root": str(args.activation_dir),
            "validation_report": validation,
        },
        "tokens": {
            "cache_path": str(cache_path),
            "sha256": observed_token_sha,
            "corpus_split": cfg.activation_split,
            "internal_split": cfg.internal_split,
        },
        "model": {
            "name": MODEL_NAME,
            "resolved_name": MODEL_RESOLVED_NAME,
            "revision": MODEL_REVISION,
            "hook": cfg.hook,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "protected_data": {
            "steering_vectors_loaded": False,
            "dev_directions_loaded": False,
            "held_out_accessed": False,
        },
        "evaluation": evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--activation-dir", required=True, type=Path)
    parser.add_argument("--token-cache-dir", required=True, type=Path)
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

    gate = report["evaluation"]["gate"]
    print(
        f"Phase A {gate['status']}: mean(d)={gate['primary_effect']['mean']:.6f}, "
        f"95% CI=[{gate['primary_effect']['ci_lower']:.6f}, "
        f"{gate['primary_effect']['ci_upper']:.6f}]"
    )
    print(f"wrote immutable report: {args.output}")


if __name__ == "__main__":
    main()
