"""Tiny FineWeb -> activations -> training -> checkpoint -> Phase-A smoke for the 2x2 setup.

Operational only. No steering, no DEV direction, no held-out artifact, no Phase B,
no LLM judge. Nothing produced here may be used as scientific evidence; it exists
to prove the pipeline and the wide model are executable before an expensive run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from interp.activations import (
    ActivationDataset,
    file_sha256,
    load_activations,
    load_validation_report,
    make_split,
    split_stats,
)
from interp.flow_core import (
    ActivationNormalizer,
    FlowMatcher,
    flow_matching_loss,
    load_flow_config,
    n_parameters,
    sample_flow_batch,
)
from interp.model import load_model, resolve_device
from interp.phase_a import (
    PhaseAConfig,
    evaluate_phase_a,
    lifecycle_record,
    select_validation_sequence_ids,
    write_failure_receipt,
    write_immutable_json,
)
from interp.provenance import source_revision
from interp.train_flow import load_flow_checkpoint, load_training_config, train_flow

PER_SEQ = 127
CTX = 128
REPO = Path(__file__).resolve().parents[1]


def _run_script(script: str, arguments: list[str]) -> None:
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *arguments],
        check=True,
        cwd=REPO,
    )


def _collect(name: str, n_seqs: int, out_dir: Path, cache_dir: Path, device: str) -> None:
    _run_script(
        "collect_activations.py",
        [
            "--corpus", "fineweb",
            "--split", "train",
            "--n-tokens", str(n_seqs * PER_SEQ),
            "--name", name,
            "--output-dir", str(out_dir),
            "--token-cache-dir", str(cache_dir),
            "--device", device,
        ],
    )


def _training_config(work: Path, flow_core: Path, dataset: str, n_tokens: int, steps: int) -> Path:
    """Write a throwaway training config beside a copy of the frozen architecture config."""

    configs = work / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    shutil.copy(flow_core, configs / flow_core.name)
    fingerprint = make_split(n_tokens, PER_SEQ, 0.05, 20_260_807).fingerprint()
    config = {
        "version": 1,
        "experiment_id": "flow_scaling_smoke_v1",
        "experiment_class": "concept_independent_capacity_data_scaling",
        "status": "authorized",
        "flow_core_config": f"configs/{flow_core.name}",
        "data": {
            "dataset": dataset,
            "split": "train",
            "repository": "HuggingFaceFW/fineweb",
            "repository_config": "sample-10BT",
            "repository_revision": "9bb295ddab0e05d785b879661af7260fed5140fc",
            "tokenizer": "gpt2",
            "per_seq": PER_SEQ,
            "val_fraction": 0.05,
            "split_seed": 20_260_807,
            "split_fingerprint": fingerprint,
            "bos_dropped": True,
            "steering_vectors_used": False,
        },
        "normalization": {
            "statistics_from": "train_split_only",
            "eps": 1.0e-5,
            "accumulation_dtype": "float64",
        },
        "training": {
            "seed": 0,
            "noise_seed": 20_260_812,
            "steps": steps,
            "batch_size": 1024,
            "optimizer": "adamw",
            "lr": 3.0e-4,
            "weight_decay": 0.01,
            "schedule": "cosine",
            "warmup_steps": 10,
            "grad_clip": 1.0,
            "dtype": "float32",
            "eval_every": max(steps // 4, 1),
        },
        "validation": {
            "seed": 0,
            "batches": 2,
            "t_bins": [0.0, 0.10, 0.25, 0.50, 0.75, 1.0],
            "selection_metric": "val_flow_mse",
            "selection_mode": "min",
        },
        "checkpoints": {
            "save_steps": [steps],
            "keep": ["best", "last", "configured_steps"],
            "include_optimizer_state": True,
        },
        "protected_data": {"held_out": "forbidden"},
    }
    path = configs / "flow_train_scaling_smoke_v1.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _wide_model_checks(flow_core: Path, device: torch.device) -> dict:
    """Exact parameter count, finite forward/backward, one update, and a tiny overfit."""

    cfg = load_flow_config(flow_core)
    width = cfg.activation_width
    torch.manual_seed(0)
    normalizer = ActivationNormalizer(torch.zeros(width), torch.ones(width))
    model = FlowMatcher(cfg, normalizer).to(device).float().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=0.01)
    generator = torch.Generator(device=device).manual_seed(0)
    fixed = torch.randn(1024, width, generator=generator, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    losses: list[float] = []
    started = time.perf_counter()
    for _ in range(100):
        batch = sample_flow_batch(fixed, generator=generator)
        loss = flow_matching_loss(model(batch.x_t, batch.t), batch.target_velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise FloatingPointError("wide-model smoke produced a non-finite loss or gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if losses[-1] >= losses[0]:
        raise ValueError(f"wide model did not reduce loss: {losses[0]} -> {losses[-1]}")
    return {
        "config": str(flow_core),
        "activation_dim": width,
        "d_model": cfg.d_model,
        "d_mlp": cfg.d_mlp,
        "n_blocks": cfg.n_blocks,
        "parameters": n_parameters(model),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "steps": len(losses),
        "steps_per_second": len(losses) / elapsed,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "batch_size": 1024,
    }


def _phase_a_config(dataset_name: str, fingerprint: str, n_sequences: int) -> PhaseAConfig:
    zero = "0" * 64
    return PhaseAConfig(
        experiment_id="flow_scaling_smoke_v1",
        experiment_class="operational_smoke",
        checkpoint_path=Path("smoke"),
        run_meta_path=Path("smoke"),
        best_pointer_path=Path("smoke"),
        checkpoint_sha256=zero,
        run_meta_sha256=zero,
        best_pointer_sha256=zero,
        checkpoint_step=0,
        training_experiment_id="flow_scaling_smoke_v1",
        expected_history_entries=0,
        selection_metric="val_flow_mse",
        selection_mode="min",
        dataset_name=dataset_name,
        activation_split="train",
        internal_split="validation_only",
        dataset_repository="HuggingFaceFW/fineweb",
        dataset_config="sample-10BT",
        dataset_revision="9bb295ddab0e05d785b879661af7260fed5140fc",
        tokenizer="gpt2",
        model_name="gpt2-small",
        resolved_model_name="gpt2",
        model_revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
        hook="blocks.7.hook_resid_pre",
        ctx=CTX,
        per_seq=PER_SEQ,
        val_fraction=0.05,
        split_seed=20_260_807,
        split_fingerprint=fingerprint,
        token_cache_sha256=zero,
        artifact_sha256={"array": zero, "metadata": zero, "statistics": zero},
        n_sequences=n_sequences,
        sequence_selection="first_sorted_internal_validation_sequences",
        lm_batch_size=4,
        noise_seed=0,
        bootstrap_seed=20_260_813,
        bootstrap_resamples=200,
        bootstrap_confidence=0.95,
        t_starts=(0.10, 0.25, 0.50),
        nfes=(1,),
        skip_bos=True,
        identity_nfe=1,
        primary_t_start=0.50,
        primary_nfe=1,
        steering_vectors="forbidden",
        dev_directions="forbidden",
        held_out="forbidden",
        raw={"smoke": True},
    )


def _run(args: argparse.Namespace) -> dict:
    device = resolve_device(args.device)
    work: Path = args.work_dir
    activations = work / "activations"
    replay = work / "activations_replay"
    cache = work / "cache"
    name = f"smoke_fw_{args.n_seqs}seq_v1"
    n_tokens = args.n_seqs * PER_SEQ

    _collect(name, args.n_seqs, activations, cache, str(device))
    _run_script(
        "validate_activations.py",
        [
            "--corpus", "fineweb",
            "--name", name,
            "--activation-dir", str(activations),
            "--token-cache-dir", str(cache),
            "--expected-split", "train",
            "--device", str(device),
        ],
    )
    _collect(name, args.n_seqs, replay, cache, str(device))
    first_hash = file_sha256(activations / f"{name}.npy")
    replay_hash = file_sha256(replay / f"{name}.npy")
    if first_hash != replay_hash:
        raise ValueError("FineWeb activation collection is not deterministic")

    report = json.loads((activations / f"{name}_validation.json").read_text())
    dataset = load_activations(name, activations)
    if dataset.array.dtype != np.float16:
        raise ValueError("activation storage must be float16")
    meta = dataset.meta
    if meta["padding_activations_discarded"] != 0 or not meta["bos_dropped"]:
        raise ValueError("collection padding/BOS receipts are wrong")
    if meta["bos_activations_discarded"] != args.n_seqs or meta["valid_activations"] != n_tokens:
        raise ValueError("collection token accounting is wrong")
    split = make_split(len(dataset), PER_SEQ, 0.05, 20_260_807)
    mean, std = split_stats(dataset, split.train)
    if not np.isfinite(mean).all() or not (std > 0).all():
        raise ValueError("float64 train-split statistics are invalid")

    wide = _wide_model_checks(args.flow_core_config, device)

    validation_report = load_validation_report(
        name, activations, expected_split_fingerprint=split.fingerprint(), verify_hashes=True
    )
    training_config_path = _training_config(
        work, args.flow_core_config, name, len(dataset), args.steps
    )
    training_config = load_training_config(training_config_path)
    run_dir = work / "run"
    started = time.perf_counter()
    training = train_flow(
        ActivationDataset(
            array=dataset.array,
            meta={**dataset.meta, "full_validation_report": validation_report},
            mean=dataset.mean,
            std=dataset.std,
        ),
        training_config,
        run_dir,
        device=device,
        progress=False,
    )
    training_seconds = time.perf_counter() - started
    checkpoint_path = run_dir / str(training["best_checkpoint"])
    flow, checkpoint_meta, checkpoint_state = load_flow_checkpoint(checkpoint_path, device)
    if checkpoint_state is None:
        raise ValueError("smoke checkpoint carries no resume state")
    if n_parameters(flow) != wide["parameters"]:
        raise ValueError("reloaded checkpoint parameter count changed")

    language_model = load_model(str(device))
    phase_cfg = _phase_a_config(name, split.fingerprint(), args.phase_a_sequences)
    sequence_ids = select_validation_sequence_ids(len(dataset), phase_cfg)
    tokens_path = cache / meta["token_cache_file"]
    tokens = torch.from_numpy(np.load(tokens_path)).long()
    phase_a = evaluate_phase_a(
        flow, language_model, tokens[torch.from_numpy(sequence_ids)], sequence_ids, phase_cfg
    )
    if phase_a["identity"]["mean_delta_lm"] != 0.0:
        raise ValueError("t_start=0 identity is not exact in the smoke path")
    if phase_a["identity"]["flow_evaluations"] != 0:
        raise ValueError("t_start=0 identity called the flow network")

    return {
        "status": "complete",
        "kind": "operational_smoke",
        "research_status": "NOT_EVALUATED",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "fineweb_collection": {
            "artifact": name,
            "n_seqs": args.n_seqs,
            "n_activations": len(dataset),
            "array_sha256": first_hash,
            "deterministic_replay_sha256": replay_hash,
            "validation_report": report,
            "metadata": meta,
        },
        "wide_model": wide,
        "training": {
            "config": str(training_config_path),
            "steps": training_config.steps,
            "seconds": training_seconds,
            "steps_per_second": training_config.steps / training_seconds,
            "best_checkpoint": training["best_checkpoint"],
            "best_val_flow_mse": training["best_val_flow_mse"],
            "history": training["history"],
            "checkpoint_metadata": checkpoint_meta,
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
        },
        "phase_a": {
            "n_sequences": args.phase_a_sequences,
            "mean_clean_lm_loss": phase_a["mean_clean_lm_loss"],
            "identity": phase_a["identity"],
            "corruptions": {
                key: value["mean_delta_lm"] for key, value in phase_a["corruptions"].items()
            },
            "cells": {key: value["mean_delta_lm"] for key, value in phase_a["cells"].items()},
            "noise_sha256": phase_a["noise"]["sha256"],
        },
        "protected_data": {
            "steering_vectors_loaded": False,
            "dev_directions_loaded": False,
            "held_out_accessed": False,
            "phase_b_loaded": False,
            "llm_judge_used": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-seqs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--phase-a-sequences", type=int, default=8)
    parser.add_argument(
        "--flow-core-config", type=Path, default=REPO / "configs" / "flow_core_wide_60m_v1.yaml"
    )
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

    print(json.dumps({key: report[key] for key in ("status", "wide_model", "phase_a")}, indent=2))
    print(f"wrote immutable smoke receipt: {args.output}")


if __name__ == "__main__":
    main()
