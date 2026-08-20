#!/usr/bin/env python3
"""Freeze, smoke, and run the exact clean Phase B DEV evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import torch
from huggingface_hub import snapshot_download

from interp.activations import file_sha256
from interp.evaluation_metrics import score_continuations, text_metrics
from interp.flow_steering import FlowGenerationSession, FlowNoiseCell
from interp.model import MODEL_RESOLVED_NAME, MODEL_REVISION, load_model, resolve_device
from interp.phase_a import load_phase_a_config, validate_frozen_checkpoint
from interp.phase_b import load_frozen_sae, validate_phase_a_evidence
from interp.phase_b_evaluator import (
    BaselineContinuation,
    EvaluatorConfig,
    FlowArm,
    load_evaluator_config,
    prepare_generation_batch,
    validate_and_import_baseline,
    validate_cross_baseline_coverage,
)
from interp.provenance import source_revision
from interp.scaling import validate_selected_flow_prior
from interp.train_flow import load_flow_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "flow_phase_b_evaluator_v1.yaml"
DEFAULT_PHASE_A = ROOT / "configs" / "flow_phase_a_100k_v1.yaml"
DEFAULT_PHASE_A_REPORT = (
    ROOT
    / "results"
    / "remote_clean_flow_phase_a_100k_v1_d0a304d2"
    / "clean_flow_phase_a_100k_v1_d0a304d2.json"
)
DEFAULT_SAE = Path("/workspace/data/sae/gpt2-small-res-jb_57d08a4f/blocks.7.hook_resid_pre")
DEFAULT_BASELINES = Path("/workspace/data/phase_b_baselines")
DEFAULT_SCALING_REPORTS = Path("/workspace/results/flow_scaling_2x2_v2")


def _json_bytes(value: object) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"{text}\n".encode()


def _write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _tree_artifacts(snapshot: Path) -> dict[str, dict[str, str]]:
    model_names = {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "pytorch_model.bin",
    }
    tokenizer_names = {
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }

    def hashes(names: set[str]) -> dict[str, str]:
        return {
            path.relative_to(snapshot).as_posix(): file_sha256(path)
            for path in sorted(snapshot.rglob("*"))
            if path.is_file() and path.name in names
        }

    model, tokenizer = hashes(model_names), hashes(tokenizer_names)
    if "config.json" not in model or not ({"model.safetensors", "pytorch_model.bin"} & set(model)):
        raise ValueError("resolved GPT-2 snapshot lacks exact model artifacts")
    if not {"merges.txt", "vocab.json"}.issubset(tokenizer):
        raise ValueError("resolved GPT-2 snapshot lacks exact tokenizer artifacts")
    return {"model": model, "tokenizer": tokenizer}


def _artifact_digest(mapping: dict[str, str]) -> str:
    return hashlib.sha256(_json_bytes(mapping)).hexdigest()


def _baseline_inputs(
    cfg: EvaluatorConfig, directory: Path
) -> dict[str, tuple[BaselineContinuation, ...]]:
    imported = {
        spec.method: validate_and_import_baseline(
            spec,
            directory / spec.raw_filename,
            directory / spec.meta_filename,
            cfg,
        )
        for spec in cfg.baselines
    }
    validate_cross_baseline_coverage(imported)
    return imported


def _flow_prior_provenance(args: argparse.Namespace, cfg: EvaluatorConfig) -> tuple[dict, dict]:
    """Return (evidence, checkpoint_selection) for the prior this release evaluates."""

    prior = cfg.flow_prior
    if prior is None:
        phase_a = load_phase_a_config(args.phase_a_config)
        evidence = validate_phase_a_evidence(cfg.phase_b, args.phase_a_config, args.phase_a_report)
        selection = validate_frozen_checkpoint(phase_a.checkpoint_path, phase_a)
        if phase_a.checkpoint_path != cfg.flow_checkpoint_path:
            raise ValueError("Phase A checkpoint path differs from the evaluator manifest")
        return {"phase_a_evidence": evidence}, selection

    reports = args.scaling_reports_dir
    protocol_path = ROOT / prior.selection_protocol
    selection_report = reports / "selection.json"
    arm_report = reports / f"{prior.selected_arm}.json"
    if file_sha256(protocol_path) != prior.selection_protocol_sha256:
        raise ValueError("scaling protocol bytes differ from the frozen evaluator")
    if file_sha256(selection_report) != prior.selection_report_sha256:
        raise ValueError("scaling selection report bytes differ from the frozen evaluator")
    if file_sha256(arm_report) != prior.arm_report_sha256:
        raise ValueError("scaling arm report bytes differ from the frozen evaluator")
    for declared, path in (
        (prior.architecture_config_sha256, ROOT / prior.architecture_config),
        (prior.training_config_sha256, ROOT / prior.training_config),
    ):
        if file_sha256(path) != declared:
            raise ValueError(f"frozen flow-prior config bytes changed: {path}")
    evidence = validate_selected_flow_prior(
        protocol_path,
        selection_report,
        arm_report,
        expected_arm=prior.selected_arm,
        training_experiment_id=prior.training_experiment_id,
        checkpoint_path=cfg.flow_checkpoint_path,
        run_meta_path=prior.run_meta_path,
        best_pointer_path=prior.best_pointer_path,
        role=(
            "capacity_control"
            if prior.provenance_kind == "concept_independent_scaling_capacity_control"
            else "selected"
        ),
    )
    if (
        evidence["parameters"] != prior.parameters
        or evidence["training_dataset"] != prior.training_dataset
        or evidence["training_dataset_artifact_identity"]["sha256"]["statistics"]
        != prior.training_statistics_sha256
    ):
        raise ValueError("selected flow prior differs from the frozen evaluator declaration")
    checkpoint_selection = evidence.pop("checkpoint_selection")
    if (
        checkpoint_selection["run_meta_sha256"] != prior.run_meta_sha256
        or checkpoint_selection["best_pointer_sha256"] != prior.best_pointer_sha256
    ):
        raise ValueError("flow-prior checkpoint sidecars differ from the frozen evaluator")
    return {"flow_prior_evidence": evidence}, checkpoint_selection


def _static_provenance(args: argparse.Namespace) -> tuple[EvaluatorConfig, dict]:
    cfg = load_evaluator_config(args.config)
    evidence, selection = _flow_prior_provenance(args, cfg)
    if selection["checkpoint_sha256"] != cfg.flow_checkpoint_sha256:
        raise ValueError("frozen flow checkpoint differs from evaluator manifest")
    if selection["checkpoint_step"] != cfg.flow_checkpoint_step:
        raise ValueError("frozen flow checkpoint step differs from evaluator manifest")
    if file_sha256(args.sae_dir / cfg.phase_b.sae.config_filename) != cfg.phase_b.sae.config_sha256:
        raise ValueError("frozen SAE config bytes changed")
    if (
        file_sha256(args.sae_dir / cfg.phase_b.sae.weights_filename)
        != cfg.phase_b.sae.weights_sha256
    ):
        raise ValueError("frozen SAE weights bytes changed")
    _baseline_inputs(cfg, args.baseline_dir)
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_RESOLVED_NAME,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )
    artifacts = _tree_artifacts(snapshot)
    provenance = {
        "source_revision": source_revision(),
        "evaluator_config_sha256": file_sha256(args.config),
        "parent_protocol_sha256": cfg.parent_protocol_sha256,
        "uv_lock_sha256": file_sha256(ROOT / "uv.lock"),
        "flow_checkpoint_sha256": cfg.flow_checkpoint_sha256,
        "sae_config_sha256": cfg.phase_b.sae.config_sha256,
        "sae_weights_sha256": cfg.phase_b.sae.weights_sha256,
        "gpt2_revision": MODEL_REVISION,
        "gpt2_snapshot": str(snapshot.resolve()),
        "gpt2_model_artifacts": artifacts["model"],
        "gpt2_model_artifact_sha256": _artifact_digest(artifacts["model"]),
        "gpt2_tokenizer_artifacts": artifacts["tokenizer"],
        "gpt2_tokenizer_artifact_sha256": _artifact_digest(artifacts["tokenizer"]),
        "baseline_artifacts": {
            spec.method: {
                "raw": spec.raw_sha256,
                "meta": spec.meta_sha256,
            }
            for spec in cfg.baselines
        },
        **evidence,
        "checkpoint_selection": selection,
    }
    if cfg.flow_prior is None:
        provenance["phase_a_report_sha256"] = file_sha256(args.phase_a_report)
    else:
        provenance["flow_prior"] = asdict(cfg.flow_prior) | {
            "run_meta_path": str(cfg.flow_prior.run_meta_path),
            "best_pointer_path": str(cfg.flow_prior.best_pointer_path),
        }
    return cfg, provenance


def _freeze_release(args: argparse.Namespace) -> None:
    cfg, provenance = _static_provenance(args)
    revision = str(provenance["source_revision"])
    digest = revision.removeprefix("snapshot-sha256:").removeprefix("git:")
    release_id = f"{cfg.experiment_id}_{digest}"
    output = cfg.server_output_root / release_id
    if output.exists():
        raise FileExistsError(f"release output already exists: {output}")
    output.mkdir(parents=True)
    manifest = {
        "status": "frozen_before_full_dev_generation",
        "research_status": "NOT_EVALUATED",
        "experiment_id": cfg.experiment_id,
        "release_id": release_id,
        "absolute_output_directory": str(output.resolve()),
        "created_utc": datetime.now(UTC).isoformat(),
        "scientific_selection_performed": False,
        "all_flow_arms": [asdict(arm) for arm in cfg.arms],
        "metric_versions": cfg.metric_versions,
        "schema_version": cfg.schema_version,
        "bootstrap": {
            "seed": cfg.bootstrap_seed,
            "resamples": cfg.bootstrap_resamples,
            "confidence": cfg.bootstrap_confidence,
            "unit": "vector_mean",
        },
        "paths": {
            "baseline_dir": str(args.baseline_dir.resolve()),
            "sae_dir": str(args.sae_dir.resolve()),
            "phase_a_report": str(args.phase_a_report.resolve()),
        },
        "provenance": provenance,
    }
    _write_new_json(output / "release_manifest.json", manifest)
    print(output / "release_manifest.json")


def _load_release(args: argparse.Namespace) -> tuple[EvaluatorConfig, dict, Path]:
    cfg, current = _static_provenance(args)
    manifest = json.loads(args.release_manifest.read_text())
    output = Path(manifest["absolute_output_directory"])
    if args.release_manifest.resolve() != (output / "release_manifest.json").resolve():
        raise ValueError("release manifest is not inside its frozen absolute output directory")
    if manifest.get("status") != "frozen_before_full_dev_generation":
        raise ValueError("release manifest is not frozen for generation")
    if manifest.get("provenance") != current:
        raise ValueError("current source/artifact provenance differs from the frozen release")
    if manifest.get("all_flow_arms") != [asdict(arm) for arm in cfg.arms]:
        raise ValueError("release flow-arm matrix changed")
    if manifest.get("metric_versions") != cfg.metric_versions:
        raise ValueError("release metric versions changed")
    return cfg, manifest, output


def _load_runtime(args: argparse.Namespace, cfg: EvaluatorConfig):
    device = resolve_device(args.device)
    flow, metadata, _ = load_flow_checkpoint(cfg.flow_checkpoint_path, device)
    # The steering boundary is the 768-wide residual activation; the prior's own
    # internal d_model may be wider. Comparing d_model here would silently reject
    # every wide prior, so the checked quantity is the activation width.
    if (
        metadata.get("step") != cfg.flow_checkpoint_step
        or flow.cfg.activation_width != cfg.phase_b.d_model
    ):
        raise ValueError("loaded flow checkpoint payload differs from the frozen evaluator")
    if cfg.flow_prior is not None:
        # The normalizer buffers travel inside the checkpoint, so binding the
        # payload to the exact activation statistics artifact proves the
        # standardization used at steering time is the one the prior trained under.
        identity = metadata.get("dataset_artifact_identity", {})
        if (
            metadata.get("dataset") != cfg.flow_prior.training_dataset
            or metadata.get("experiment_id") != cfg.flow_prior.training_experiment_id
            or identity.get("sha256", {}).get("statistics")
            != cfg.flow_prior.training_statistics_sha256
        ):
            raise ValueError("loaded flow checkpoint carries different activation statistics")
    sae = load_frozen_sae(cfg.phase_b.sae, args.sae_dir, device=device)
    model = load_model(str(device))
    if model.cfg.model_name != MODEL_RESOLVED_NAME or cfg.phase_b.hook not in model.hook_dict:
        raise ValueError("loaded GPT-2 interface differs from the frozen evaluator")
    return device, flow, sae, model


def _metrics_for(
    cfg: EvaluatorConfig,
    model,
    sae,
    vector,
    prompts: list[str],
    continuations: list[str],
) -> list[dict]:  # noqa: ANN001
    feature_ids = [vector.feature, *cfg.control_features[vector.name]]
    model_rows = score_continuations(
        model, sae, cfg.phase_b.hook, prompts, continuations, feature_ids, batch_size=16
    )
    out = []
    for continuation, model_row in zip(continuations, model_rows, strict=True):
        text_row = text_metrics(continuation, vector.lexicon)
        controls = model_row.sae_feature_means[1:]
        values = {
            **asdict(text_row),
            "nll": model_row.nll,
            "sae_act_target": model_row.sae_feature_means[0],
            "sae_act_control_mean": sum(controls) / len(controls),
            "sae_act_control_max": max(controls),
        }
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("clean evaluator produced a non-finite metric")
        out.append(
            {
                "metrics": values,
                "continuation_token_ids": list(model_row.continuation_token_ids),
                "prompt_token_count": model_row.prompt_token_count,
            }
        )
    return out


def _base_row(cfg: EvaluatorConfig, manifest: dict, cell: BaselineContinuation) -> dict:
    return {
        "schema_version": cfg.schema_version,
        "experiment_id": cfg.experiment_id,
        "release_id": manifest["release_id"],
        "source_revision": manifest["provenance"]["source_revision"],
        "vector": cell.vector,
        "feature": cell.feature,
        "split": "dev",
        "alpha": cell.alpha,
        "alpha_hex": cell.alpha.hex(),
        "alpha_hat": cell.alpha_hat,
        "is_stress": cell.is_stress,
        "prompt_id": cell.prompt_id,
        "generation_seed": cell.generation_seed,
        "continuation": cell.continuation,
        "metric_versions": cfg.metric_versions,
    }


def _publish_rows(final: Path, rows: list[dict], meta: dict) -> None:
    if final.exists() or final.with_suffix(".meta.json").exists():
        raise FileExistsError(f"immutable evaluator output already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    attempt = final.with_name(f".{final.name}.{uuid4().hex}.partial")
    try:
        with attempt.open("xb") as handle:
            for row in rows:
                handle.write(_json_bytes(row))
            handle.flush()
            os.fsync(handle.fileno())
        receipt = {
            **meta,
            "status": "complete",
            "row_count": len(rows),
            "sha256": file_sha256(attempt),
            "completed_utc": datetime.now(UTC).isoformat(),
        }
        os.rename(attempt, final)
        _write_new_json(final.with_suffix(".meta.json"), receipt)
    except BaseException as error:
        failure = final.with_name(f"{final.name}.{uuid4().hex}.FAILED.json")
        _write_new_json(
            failure,
            {
                "status": "INTERRUPTED",
                "error_type": type(error).__name__,
                "error": str(error),
                "partial": str(attempt),
                "recorded_utc": datetime.now(UTC).isoformat(),
            },
        )
        raise


def _rescore_baselines(args: argparse.Namespace) -> None:
    cfg, manifest, output = _load_release(args)
    imported = _baseline_inputs(cfg, args.baseline_dir)
    _, _, sae, model = _load_runtime(args, cfg)
    vectors = {vector.name: vector for vector in cfg.phase_b.vectors}
    specs = {spec.method: spec for spec in cfg.baselines}
    for method, source_rows in imported.items():
        final = output / "baselines" / f"{method}.rescored.jsonl"
        if final.exists() and final.with_suffix(".meta.json").exists():
            print(f"skip complete {final}")
            continue
        rows = []
        for vector_name, vector in vectors.items():
            selected = [row for row in source_rows if row.vector == vector_name]
            prompts = [cfg.phase_b.prompts[row.prompt_id] for row in selected]
            continuations = [row.continuation for row in selected]
            scored = _metrics_for(cfg, model, sae, vector, prompts, continuations)
            for cell, metrics in zip(selected, scored, strict=True):
                rows.append(
                    {
                        **_base_row(cfg, manifest, cell),
                        "method": method,
                        "arm_id": method,
                        "t_start": None,
                        "nfe": None,
                        **metrics,
                        "geometry": None,
                        "noise": None,
                        "source_baseline": {
                            "raw_sha256": specs[method].raw_sha256,
                            "meta_sha256": specs[method].meta_sha256,
                        },
                    }
                )
        if len(rows) != 2_880:
            raise ValueError(f"rescored {method} row count {len(rows)} != 2880")
        _publish_rows(
            final,
            rows,
            {"method": method, "release_id": manifest["release_id"], "schema": cfg.schema_version},
        )
        print(final)


def _generate_condition(
    cfg: EvaluatorConfig,
    model,
    flow,
    sae,
    arm: FlowArm,
    vector,
    alpha: float,
    alpha_hat: float,
    is_stress: bool,
    seed: int,
    *,
    max_new_tokens: int,
    prompt_ids: tuple[int, ...],
) -> list[dict]:  # noqa: ANN001
    prompts = [cfg.phase_b.prompts[index] for index in prompt_ids]
    tokens, attention_mask, prepend_bos = prepare_generation_batch(model, prompts)
    cells = tuple(FlowNoiseCell(vector.name, alpha, prompt_id, seed) for prompt_id in prompt_ids)
    direction = sae.decoder_directions([vector.feature])[0].to(model.cfg.device)
    session = FlowGenerationSession(
        flow,
        direction,
        alpha=alpha,
        t_start=arm.t_start,
        nfe=arm.nfe,
        cells=cells,
        prompt_width=tokens.shape[1],
        max_new_tokens=max_new_tokens,
        off_distribution_norm=cfg.phase_b.off_distribution_norm,
        prompt_attention_mask=attention_mask,
        noise_namespace=cfg.phase_b.noise_namespace,
    )
    torch.manual_seed(seed)
    if tokens.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    def apply(activation: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001, ARG001
        return session.apply(activation)

    with model.hooks(fwd_hooks=[(cfg.phase_b.hook, apply)]):
        generated = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            stop_at_eos=False,
            do_sample=True,
            temperature=cfg.phase_b.temperature,
            top_p=cfg.phase_b.top_p,
            top_k=cfg.phase_b.top_k if cfg.phase_b.top_k > 0 else None,
            freq_penalty=cfg.phase_b.freq_penalty,
            use_past_kv_cache=True,
            prepend_bos=prepend_bos,
            return_type="tokens",
            verbose=False,
        )
    continuation_tokens = generated[:, tokens.shape[1] :]
    continuations = model.tokenizer.batch_decode(continuation_tokens, skip_special_tokens=True)
    metrics = _metrics_for(cfg, model, sae, vector, prompts, continuations)
    receipts = session.cell_receipts()
    if tuple(item["prompt_id"] for item in receipts) != prompt_ids:
        raise ValueError("per-prompt flow receipts lost cell order")
    rows = []
    for prompt_id, continuation, metric, receipt in zip(
        prompt_ids, continuations, metrics, receipts, strict=True
    ):
        cell = BaselineContinuation(
            vector=vector.name,
            feature=vector.feature,
            split="dev",
            alpha=alpha,
            alpha_hat=alpha_hat,
            is_stress=is_stress,
            prompt_id=prompt_id,
            generation_seed=seed,
            continuation=continuation,
        )
        rows.append({"cell": cell, "metric": metric, "receipt": receipt})
    return rows


def _run_arm(
    args: argparse.Namespace,
    cfg: EvaluatorConfig,
    manifest: dict,
    output: Path,
    arm: FlowArm,
    runtime,
) -> None:  # noqa: ANN001, E501
    _, flow, sae, model = runtime
    final = output / "flow" / f"{arm.arm_id}.jsonl"
    if final.exists() and final.with_suffix(".meta.json").exists():
        print(f"skip complete {final}")
        return
    rows = []
    alpha_grid = tuple((value, False) for value in cfg.phase_b.alpha_hat) + tuple(
        (value, True) for value in cfg.phase_b.alpha_hat_stress
    )
    prompt_ids = tuple(range(len(cfg.phase_b.prompts)))
    for vector in cfg.phase_b.vectors:
        for alpha_hat, is_stress in alpha_grid:
            alpha = alpha_hat * cfg.phase_b.activation_norm_mean
            for seed in cfg.phase_b.generation_seeds:
                generated = _generate_condition(
                    cfg,
                    model,
                    flow,
                    sae,
                    arm,
                    vector,
                    alpha,
                    alpha_hat,
                    is_stress,
                    seed,
                    max_new_tokens=cfg.phase_b.max_new_tokens,
                    prompt_ids=prompt_ids,
                )
                for item in generated:
                    cell, metric, receipt = item["cell"], item["metric"], item["receipt"]
                    rows.append(
                        {
                            **_base_row(cfg, manifest, cell),
                            "method": "flow",
                            "arm_id": arm.arm_id,
                            "t_start": arm.t_start,
                            "nfe": arm.nfe,
                            **metric,
                            "geometry": receipt["geometry"],
                            "noise": {
                                "namespace": cfg.phase_b.noise_namespace,
                                "position_scheme": "prompt_relative_v1",
                                "positions": receipt["epsilon_positions"],
                                "epsilon_sha256": receipt["epsilon_sha256"],
                                "padding_positions_evaluated": 0,
                                "guarded_positions": receipt["guarded_positions"],
                            },
                            "source_baseline": None,
                        }
                    )
                if len(rows) % 300 == 0:
                    print(f"{arm.arm_id}: {len(rows)}/2880", flush=True)
    if len(rows) != 2_880:
        raise ValueError(f"{arm.arm_id} row count {len(rows)} != 2880")
    _publish_rows(
        final,
        rows,
        {
            "method": "flow",
            "arm": asdict(arm),
            "release_id": manifest["release_id"],
            "schema": cfg.schema_version,
        },
    )
    print(final)


def _run_all(args: argparse.Namespace) -> None:
    cfg, manifest, output = _load_release(args)
    runtime = _load_runtime(args, cfg)
    for arm in cfg.arms:
        _run_arm(args, cfg, manifest, output, arm, runtime)


def _smoke(args: argparse.Namespace) -> None:
    cfg, provenance = _static_provenance(args)
    runtime = _load_runtime(args, cfg)
    _, flow, sae, model = runtime
    vector = cfg.phase_b.vectors[0]
    arm = cfg.arms[0]
    alpha_hat = 0.1
    first = _generate_condition(
        cfg,
        model,
        flow,
        sae,
        arm,
        vector,
        alpha_hat * cfg.phase_b.activation_norm_mean,
        alpha_hat,
        False,
        0,
        max_new_tokens=2,
        prompt_ids=(0, 1),
    )
    second = _generate_condition(
        cfg,
        model,
        flow,
        sae,
        arm,
        vector,
        alpha_hat * cfg.phase_b.activation_norm_mean,
        alpha_hat,
        False,
        0,
        max_new_tokens=2,
        prompt_ids=(0, 1),
    )
    serial = lambda rows: [  # noqa: E731
        {
            "prompt_id": item["cell"].prompt_id,
            "continuation": item["cell"].continuation,
            "metric": item["metric"],
            "receipt": item["receipt"],
        }
        for item in rows
    ]
    if serial(first) != serial(second):
        raise ValueError("multi-prompt evaluator smoke is not deterministic")
    report = {
        "status": "complete",
        "research_status": "NOT_EVALUATED",
        "purpose": "operational_only_no_scientific_selection",
        "source_revision": source_revision(),
        "provenance": provenance,
        "scientific_selection_performed": False,
        "held_out_accessed": False,
        "smoke": serial(first),
    }
    _write_new_json(args.smoke_output, report)
    print(args.smoke_output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase-a-config", type=Path, default=DEFAULT_PHASE_A)
    parser.add_argument("--phase-a-report", type=Path, default=DEFAULT_PHASE_A_REPORT)
    parser.add_argument("--sae-dir", type=Path, default=DEFAULT_SAE)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--scaling-reports-dir", type=Path, default=DEFAULT_SCALING_REPORTS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--smoke-output", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze-release")
    sub.add_parser("validate-release")
    sub.add_parser("rescore-baselines")
    sub.add_parser("smoke")
    sub.add_parser("run-all")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "freeze-release":
        _freeze_release(args)
    elif args.command == "validate-release":
        _, _, output = _load_release(args)
        print(f"VALID {output}")
    elif args.command == "rescore-baselines":
        _rescore_baselines(args)
    elif args.command == "smoke":
        if args.smoke_output is None:
            raise ValueError("smoke requires --smoke-output")
        _smoke(args)
    elif args.command == "run-all":
        _run_all(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise
