"""Run the frozen concept-independent flow-prior checkpoint and NFE diagnostic."""

from __future__ import annotations

import argparse
import json
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
from interp.prior_diagnostic import (
    assert_exact_file_sha256,
    load_prior_diagnostic_config,
    nfe_marginals,
    summarize_phase_a,
    wide_glp_parameter_count,
)
from interp.provenance import source_revision
from interp.train_flow import load_flow_checkpoint


def _load_validation_inputs(args: argparse.Namespace, phase_cfg):  # noqa: ANN001
    validation = load_validation_report(
        phase_cfg.dataset_name,
        args.activation_dir,
        expected_split_fingerprint=phase_cfg.split_fingerprint,
        verify_hashes=True,
    )
    if validation.get("sha256") != phase_cfg.artifact_sha256:
        raise ValueError("validated activation artifact SHA set differs from frozen Phase A")
    if validation.get("token_cache_sha256") != phase_cfg.token_cache_sha256:
        raise ValueError("validated token-cache SHA differs from frozen Phase A")
    dataset = load_activations(phase_cfg.dataset_name, args.activation_dir)
    dataset = replace(dataset, meta={**dataset.meta, "full_validation_report": validation})
    validate_activation_metadata(
        dataset,
        expected_name=phase_cfg.dataset_name,
        expected_split=phase_cfg.activation_split,
        expected_model=phase_cfg.model_name,
        expected_resolved_model_name=phase_cfg.resolved_model_name,
        expected_model_revision=phase_cfg.model_revision,
        expected_hook=phase_cfg.hook,
        expected_ctx=phase_cfg.ctx,
        expected_d_model=768,
        expected_dataset_repository=phase_cfg.dataset_repository,
        expected_dataset_config=phase_cfg.dataset_config,
        expected_dataset_revision=phase_cfg.dataset_revision,
        expected_tokenizer=phase_cfg.tokenizer,
    )
    return dataset, validation


def _load_tokens(args: argparse.Namespace, phase_cfg, dataset, language_model):  # noqa: ANN001
    cache_path = token_cache_path(
        phase_cfg.activation_split,
        int(dataset.meta["n_seqs"]),
        phase_cfg.ctx,
        language_model.tokenizer,
        args.token_cache_dir,
    )
    tokens = load_tokens(
        phase_cfg.activation_split,
        int(dataset.meta["n_seqs"]),
        language_model.tokenizer,
        ctx=phase_cfg.ctx,
        cache_dir=args.token_cache_dir,
    )
    assert_exact_file_sha256(cache_path, phase_cfg.token_cache_sha256)
    sequence_ids = select_validation_sequence_ids(len(dataset), phase_cfg)
    selected = tokens[torch.from_numpy(sequence_ids)]
    if selected.shape != (phase_cfg.n_sequences, phase_cfg.ctx):
        raise ValueError("selected validation token matrix has the wrong shape")
    return selected, sequence_ids, cache_path


def _evaluate_checkpoint(spec, phase_cfg, language_model, tokens, sequence_ids, nfes):  # noqa: ANN001
    assert_exact_file_sha256(spec.path, spec.sha256)
    flow, metadata, _ = load_flow_checkpoint(spec.path, language_model.cfg.device)
    cell_cfg = replace(
        phase_cfg,
        checkpoint_path=spec.path,
        checkpoint_sha256=spec.sha256,
        checkpoint_step=spec.step,
        nfes=tuple(nfes),
    )
    payload = validate_checkpoint_payload(metadata, cell_cfg)
    if flow.cfg.d_model != 768:
        raise ValueError("diagnostic checkpoint must use the frozen 768-wide activation model")
    if (
        not torch.isfinite(flow.normalizer.mean).all()
        or not torch.isfinite(flow.normalizer.std).all()
        or not bool((flow.normalizer.std > 0).all())
    ):
        raise ValueError("checkpoint normalizer must be finite and strictly positive")
    evaluation = evaluate_phase_a(flow, language_model, tokens, sequence_ids, cell_cfg)
    return payload, evaluation


def _run(args: argparse.Namespace) -> dict:
    cfg = load_prior_diagnostic_config(args.config)
    project_root = args.config.resolve().parents[1]
    parent_path = project_root / cfg.parent_phase_a_path
    assert_exact_file_sha256(parent_path, cfg.parent_phase_a_sha256)
    phase_cfg = load_phase_a_config(parent_path)

    assert_exact_file_sha256(cfg.run_meta_path, cfg.run_meta_sha256)
    assert_exact_file_sha256(cfg.best_pointer_path, cfg.best_pointer_sha256)
    run_meta = json.loads(cfg.run_meta_path.read_text())
    history = run_meta.get("history", [])
    if run_meta.get("status") != "complete" or len(history) != cfg.expected_history_entries:
        raise ValueError("training run metadata is not the frozen complete 200-entry trajectory")
    if run_meta.get("experiment_id") != phase_cfg.training_experiment_id:
        raise ValueError("training experiment identity differs from frozen Phase A")
    history_by_step = {int(item["step"]): item for item in history}
    if any(spec.step not in history_by_step for spec in cfg.checkpoints):
        raise ValueError("checkpoint grid is missing from the training validation history")
    selection_receipt = validate_frozen_checkpoint(phase_cfg.checkpoint_path, phase_cfg)

    dataset, validation = _load_validation_inputs(args, phase_cfg)
    device = resolve_device(args.device)
    language_model = load_model(str(device))
    if (
        phase_cfg.model_name != MODEL_NAME
        or phase_cfg.resolved_model_name != MODEL_RESOLVED_NAME
        or phase_cfg.model_revision != MODEL_REVISION
    ):
        raise ValueError("canonical GPT-2 identity differs from frozen Phase A")
    selected_tokens, sequence_ids, cache_path = _load_tokens(
        args, phase_cfg, dataset, language_model
    )

    checkpoint_results = {}
    noise_hashes = set()
    for spec in cfg.checkpoints:
        payload, evaluation = _evaluate_checkpoint(
            spec,
            phase_cfg,
            language_model,
            selected_tokens,
            sequence_ids,
            cfg.checkpoint_nfes,
        )
        noise_hashes.add(evaluation["noise"]["sha256"])
        checkpoint_results[str(spec.step)] = {
            "checkpoint": {
                "path": str(spec.path),
                "step": spec.step,
                "sha256": spec.sha256,
                "payload": payload,
            },
            "validation_record": history_by_step[spec.step],
            "summary": summarize_phase_a(evaluation),
            "evaluation": evaluation,
        }
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected_spec = next(spec for spec in cfg.checkpoints if spec.step == cfg.nfe_sweep_step)
    nfe_payload, nfe_evaluation = _evaluate_checkpoint(
        selected_spec,
        phase_cfg,
        language_model,
        selected_tokens,
        sequence_ids,
        cfg.nfe_sweep,
    )
    noise_hashes.add(nfe_evaluation["noise"]["sha256"])
    if len(noise_hashes) != 1:
        raise RuntimeError("matched Phase-A epsilon changed across checkpoint or NFE evaluations")

    capacity = cfg.wide_capacity_design
    wide_parameters = wide_glp_parameter_count(
        activation_dim=int(capacity["activation_dim"]),
        d_model=int(capacity["d_model"]),
        d_mlp=int(capacity["d_mlp"]),
        n_blocks=int(capacity["n_blocks"]),
        time_dim=int(capacity["time_dim"]),
        time_hidden=int(capacity["time_hidden"]),
    )
    return {
        "status": "complete",
        "experiment_id": cfg.experiment_id,
        "experiment_class": "concept_independent_dev_diagnostic",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "config": {
            "path": str(args.config),
            "sha256": file_sha256(args.config),
            "frozen": cfg.raw,
        },
        "parent_phase_a": {
            "path": str(parent_path),
            "sha256": cfg.parent_phase_a_sha256,
            "checkpoint_selection": selection_receipt,
        },
        "training_run": {
            "meta_path": str(cfg.run_meta_path),
            "meta_sha256": cfg.run_meta_sha256,
            "best_pointer_path": str(cfg.best_pointer_path),
            "best_pointer_sha256": cfg.best_pointer_sha256,
            "history": history,
        },
        "activation_artifact": {"root": str(args.activation_dir), "validation": validation},
        "tokens": {
            "cache_path": str(cache_path),
            "sha256": phase_cfg.token_cache_sha256,
            "validation_sequence_ids": [int(value) for value in sequence_ids],
        },
        "model": {
            "name": MODEL_NAME,
            "resolved_name": MODEL_RESOLVED_NAME,
            "revision": MODEL_REVISION,
            "hook": phase_cfg.hook,
        },
        "checkpoint_sweep": checkpoint_results,
        "nfe_sweep": {
            "checkpoint": {
                "path": str(selected_spec.path),
                "step": selected_spec.step,
                "sha256": selected_spec.sha256,
                "payload": nfe_payload,
            },
            "summary": summarize_phase_a(nfe_evaluation),
            "marginals": nfe_marginals(nfe_evaluation),
            "evaluation": nfe_evaluation,
        },
        "matched_noise": {
            "sha256": next(iter(noise_hashes)),
            "same_across_checkpoints_t_start_and_nfe": True,
        },
        "wide_capacity_design": {**capacity, "n_parameters": wide_parameters},
        "protected_data": {
            "phase_b_rerun": False,
            "steering_vectors_loaded": False,
            "dev_directions_loaded": False,
            "held_out_accessed": False,
            "llm_judges_loaded": False,
            "training_run_launched": False,
            "activation_collection_launched": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
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
    print(f"wrote immutable diagnostic: {args.output}")


if __name__ == "__main__":
    main()
