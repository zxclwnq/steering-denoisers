"""Evaluate one trained 2x2 scaling arm concept-independently on the frozen FineWeb validation set.

No steering vector, DEV direction, held-out artifact, Phase-B row, or LLM judge is
loaded. The measurement is the frozen Phase-A grid plus a shared validation
flow-loss diagnostic, with epsilon matched across arms.
"""

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
    make_split,
    validate_activation_metadata,
)
from interp.data import corpus_for, load_tokens, token_cache_path
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
    select_validation_sequence_ids,
    validate_frozen_checkpoint,
    write_failure_receipt,
    write_immutable_json,
)
from interp.provenance import source_revision
from interp.scaling import arm_phase_a_config, load_scaling_protocol, validation_flow_loss
from interp.train_flow import load_flow_checkpoint


def _resolve_run(run_dir: Path) -> tuple[Path, Path, Path, dict, int, int]:
    run_meta_path = run_dir / "meta.json"
    best_pointer_path = run_dir / "best.json"
    run_meta = json.loads(run_meta_path.read_text())
    best = json.loads(best_pointer_path.read_text())
    if run_meta.get("status") != "complete":
        raise ValueError(f"training run {run_dir} is not complete")
    checkpoint_path = run_dir / str(best["checkpoint"])
    history = run_meta["history"]
    step = int(str(checkpoint_path.stem).removeprefix("best_step_"))
    return checkpoint_path, run_meta_path, best_pointer_path, run_meta, step, len(history)


def _run(args: argparse.Namespace) -> dict:
    protocol = load_scaling_protocol(args.config)
    arm = protocol.arm(args.arm)
    corpus = corpus_for("fineweb")

    validation = load_validation_report(
        protocol.validation_artifact,
        args.activation_dir,
        expected_split_fingerprint=protocol.split_fingerprint,
        verify_hashes=True,
    )
    dataset = load_activations(protocol.validation_artifact, args.activation_dir)
    dataset = replace(dataset, meta={**dataset.meta, "full_validation_report": validation})
    validate_activation_metadata(
        dataset,
        expected_name=protocol.validation_artifact,
        expected_split=protocol.validation_split,
        expected_model=MODEL_NAME,
        expected_resolved_model_name=MODEL_RESOLVED_NAME,
        expected_model_revision=MODEL_REVISION,
        expected_hook=protocol.hook,
        expected_ctx=protocol.ctx,
        expected_d_model=768,
        expected_dataset_repository=corpus.repository,
        expected_dataset_config=corpus.config,
        expected_dataset_revision=corpus.revision,
        expected_tokenizer="gpt2",
    )

    (
        checkpoint_path,
        run_meta_path,
        best_pointer_path,
        run_meta,
        step,
        history_entries,
    ) = _resolve_run(args.run_dir)
    if run_meta.get("experiment_id") != args.training_experiment_id:
        raise ValueError("run metadata experiment id differs from the requested arm run")
    if run_meta.get("held_out_accessed") is not False:
        raise ValueError("run metadata does not prove held-out remained untouched")
    if run_meta.get("dataset") != arm.dataset:
        raise ValueError(
            f"run trained on {run_meta.get('dataset')!r}, arm expects {arm.dataset!r}"
        )
    if int(run_meta.get("steps", 0)) != protocol.optimizer_steps:
        raise ValueError("run optimizer steps differ from the compute-matched protocol")
    if int(run_meta.get("batch_size", 0)) != protocol.batch_size:
        raise ValueError("run batch size differs from the compute-matched protocol")

    device = resolve_device(args.device)
    language_model = load_model(str(device))
    flow, checkpoint_metadata, _ = load_flow_checkpoint(checkpoint_path, device)
    if flow.cfg.activation_width != dataset.d_model:
        raise ValueError(
            f"flow activation width {flow.cfg.activation_width} != validation width "
            f"{dataset.d_model}"
        )
    parameters = sum(parameter.numel() for parameter in flow.parameters())
    if parameters != arm.parameters:
        raise ValueError(f"checkpoint has {parameters} parameters != frozen {arm.parameters}")

    cache_path = token_cache_path(
        protocol.validation_split,
        int(dataset.meta["n_seqs"]),
        protocol.ctx,
        language_model.tokenizer,
        args.token_cache_dir,
        corpus,
    )
    tokens = load_tokens(
        protocol.validation_split,
        int(dataset.meta["n_seqs"]),
        language_model.tokenizer,
        ctx=protocol.ctx,
        cache_dir=args.token_cache_dir,
        corpus=corpus,
    )
    token_cache_sha256 = file_sha256(cache_path)
    if token_cache_sha256 != dataset.meta["token_cache_sha256"]:
        raise ValueError("validation token cache differs from the collected artifact")

    cfg = arm_phase_a_config(
        protocol,
        arm,
        checkpoint_path=checkpoint_path,
        run_meta_path=run_meta_path,
        best_pointer_path=best_pointer_path,
        checkpoint_sha256=file_sha256(checkpoint_path),
        run_meta_sha256=file_sha256(run_meta_path),
        best_pointer_sha256=file_sha256(best_pointer_path),
        checkpoint_step=step,
        history_entries=history_entries,
        training_experiment_id=args.training_experiment_id,
        validation_artifact_sha256=validation["sha256"],
        validation_token_cache_sha256=token_cache_sha256,
        dataset_repository=corpus.repository,
        dataset_config=corpus.config,
        dataset_revision=corpus.revision,
    )
    selection_receipt = validate_frozen_checkpoint(checkpoint_path, cfg)
    if int(checkpoint_metadata.get("step", -1)) != step:
        raise ValueError("checkpoint payload step differs from the best-pointer step")
    training_identity = {
        "trained_on": checkpoint_metadata.get("dataset"),
        "training_split_fingerprint": checkpoint_metadata.get("split_fingerprint"),
        "training_dataset_artifact_identity": checkpoint_metadata.get(
            "dataset_artifact_identity"
        ),
        "config_fingerprint": checkpoint_metadata.get("config_fingerprint"),
        "source_revision": checkpoint_metadata.get("source_revision"),
    }
    if training_identity["trained_on"] != arm.dataset:
        raise ValueError("checkpoint was not trained on the arm's activation artifact")

    sequence_ids = select_validation_sequence_ids(len(dataset), cfg)
    selected_tokens = tokens[torch.from_numpy(sequence_ids)]
    evaluation = evaluate_phase_a(flow, language_model, selected_tokens, sequence_ids, cfg)
    split = make_split(len(dataset), cfg.per_seq, cfg.val_fraction, cfg.split_seed)
    flow_loss = validation_flow_loss(flow, dataset, split.val, protocol, device)

    primary_key = f"t{protocol.primary_t_start:.2f}_nfe{protocol.primary_nfe}"
    return {
        "status": "complete",
        "experiment_id": protocol.experiment_id,
        "arm": arm.arm_id,
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "protocol": {"path": str(args.config), "sha256": file_sha256(args.config)},
        "arm_spec": {
            "training_config": str(arm.training_config),
            "flow_core_config": str(arm.flow_core_config),
            "parameters": parameters,
            "dataset": arm.dataset,
            "unique_activation_tokens": arm.unique_activation_tokens,
            "total_activation_presentations": protocol.total_activation_presentations,
            "optimizer_steps": protocol.optimizer_steps,
        },
        "checkpoint_selection": selection_receipt,
        "training_identity": training_identity,
        "validation_artifact": {
            "root": str(args.activation_dir),
            "validation_report": validation,
            "token_cache_path": str(cache_path),
            "token_cache_sha256": token_cache_sha256,
        },
        "model": {
            "name": MODEL_NAME,
            "resolved_name": MODEL_RESOLVED_NAME,
            "revision": MODEL_REVISION,
            "hook": protocol.hook,
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
            "phase_b_loaded": False,
            "llm_judge_used": False,
        },
        "validation_flow_loss": flow_loss,
        "phase_a": evaluation,
        "selection_inputs": {
            "primary_cell": primary_key,
            "primary_effects": evaluation["cells"][primary_key]["delta_lm_per_sequence"],
            "mean_reconstructed_delta_lm": evaluation["cells"][primary_key]["mean_delta_lm"],
            "val_flow_mse": flow_loss["val_flow_mse"],
            "parameters": parameters,
            "unique_activation_tokens": arm.unique_activation_tokens,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--training-experiment-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
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
            args.output, error, status="INTERRUPTED", command=command, started_utc=started_utc
        )
        raise
    except Exception as error:
        write_failure_receipt(
            args.output, error, status="INVALID", command=command, started_utc=started_utc
        )
        raise

    selection = report["selection_inputs"]
    print(
        f"arm {report['arm']}: mean reconstructed delta LM at {selection['primary_cell']} = "
        f"{selection['mean_reconstructed_delta_lm']:.6f}, "
        f"val_flow_mse = {selection['val_flow_mse']:.6f}"
    )
    print(f"wrote immutable report: {args.output}")


if __name__ == "__main__":
    main()
