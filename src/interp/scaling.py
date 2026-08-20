"""Frozen protocol and concept-independent comparison for the 2x2 cheap-scaling study.

This module reuses the Phase-A measurement semantics verbatim: it only builds the
frozen ``PhaseAConfig`` for one trained arm and adds the cross-arm selection rule
and the shared validation flow-loss diagnostic. No steering vector, DEV direction,
held-out artifact, Phase-B result, or LLM judge may enter here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from .activations import ActivationDataset, file_sha256
from .flow_core import flow_matching_loss, sample_flow_batch
from .phase_a import PhaseAConfig, paired_bootstrap_mean_ci, validate_frozen_checkpoint

# Every frozen version of the protocol. v1 is retained because it was used for an
# aborted run whose artifacts are archived; v2 raises the matched training budget
# from 100k to 250k steps and is the active protocol.
APPROVED_SCALING_PROTOCOL_SHA256S: dict[str, str] = {
    "flow_scaling_2x2_v1": "4532a19e4c51b8297385fe13a89d8d1ce396530240d409502016f900fbb78e45",
    "flow_scaling_2x2_v2": "4e11e4d382188680d28996c2dc0b1484e0721720a6c2a2811a4d0c4a32a1e09c",
}


@dataclass(frozen=True)
class ScalingArm:
    arm_id: str
    training_config: Path
    flow_core_config: Path
    parameters: int
    dataset: str
    unique_activation_tokens: int


@dataclass(frozen=True)
class ScalingProtocol:
    experiment_id: str
    arms: tuple[ScalingArm, ...]
    max_parameters: int
    total_activation_presentations: int
    optimizer_steps: int
    batch_size: int
    validation_artifact: str
    validation_split: str
    per_seq: int
    val_fraction: float
    split_seed: int
    split_fingerprint: str
    flow_loss_seed: int
    flow_loss_batch_size: int
    t_bins: tuple[float, ...]
    n_sequences: int
    ctx: int
    hook: str
    lm_batch_size: int
    noise_seed: int
    identity_nfe: int
    t_starts: tuple[float, ...]
    nfes: tuple[int, ...]
    primary_t_start: float
    primary_nfe: int
    bootstrap_seed: int
    bootstrap_resamples: int
    bootstrap_confidence: float
    tie_breakers: tuple[str, ...]
    raw: dict

    def arm(self, arm_id: str) -> ScalingArm:
        for arm in self.arms:
            if arm.arm_id == arm_id:
                return arm
        raise ValueError(f"unknown arm {arm_id!r}; expected one of {[a.arm_id for a in self.arms]}")


def load_scaling_protocol(path: Path) -> ScalingProtocol:
    """Load the preregistered 2x2 protocol and reject any semantic drift."""

    observed = file_sha256(path)
    raw = yaml.safe_load(path.read_text())
    experiment_id = str(raw.get("experiment_id"))
    frozen = APPROVED_SCALING_PROTOCOL_SHA256S.get(experiment_id)
    if frozen is None:
        raise ValueError(
            f"unknown scaling protocol {experiment_id!r}; approved: "
            f"{sorted(APPROVED_SCALING_PROTOCOL_SHA256S)}"
        )
    if observed != frozen:
        raise ValueError(f"scaling protocol SHA256 {observed} != frozen {frozen}")
    if (
        raw.get("version") != 1
        or raw.get("status") != "frozen_before_collection_and_training"
        or raw.get("experiment_class") != "concept_independent_capacity_data_scaling"
    ):
        raise ValueError("scaling protocol identity changed")

    design = raw["design"]
    if design.get("kind") != "2x2" or design.get("method_change") != "capacity_only":
        raise ValueError("the scaling design must remain a capacity-only 2x2")
    protected = raw["protected_data"]
    expected_protected = {
        "dev_directions",
        "held_out",
        "steering_vectors",
        "phase_b",
        "llm_judges",
    }
    if set(protected) != expected_protected or set(protected.values()) != {"forbidden"}:
        raise ValueError("protected-data contract changed")

    arms = tuple(
        ScalingArm(
            arm_id=str(item["id"]),
            training_config=Path(item["training_config"]),
            flow_core_config=Path(item["flow_core_config"]),
            parameters=int(item["parameters"]),
            dataset=str(item["dataset"]),
            unique_activation_tokens=int(item["unique_activation_tokens"]),
        )
        for item in raw["arms"]
    )
    if tuple(arm.arm_id for arm in arms) != (
        "narrow16m_fw4m",
        "narrow16m_fw32m",
        "wide60m_fw4m",
        "wide60m_fw32m",
    ):
        raise ValueError("the frozen 2x2 arm set changed")
    capacity = raw["capacity_bound"]
    if int(capacity["max_parameters"]) != 60_407_808:
        raise ValueError("the cheapness capacity bound changed")
    if any(arm.parameters > int(capacity["max_parameters"]) for arm in arms):
        raise ValueError("an arm exceeds the frozen capacity bound")

    matching = raw["compute_matching"]
    if matching.get("matched_quantity") != "total_activation_presentations":
        raise ValueError("compute matching must control total activation presentations")
    presentations = int(matching["total_activation_presentations"])
    steps = int(matching["optimizer_steps"])
    batch_size = int(matching["batch_size"])
    if steps * batch_size != presentations:
        raise ValueError("declared presentations disagree with steps x batch size")

    validation = raw["validation"]
    if validation.get("fixed_across_arms") is not True:
        raise ValueError("the validation set must be fixed across arms")
    if validation.get("disjoint_from_training_documents") is not True:
        raise ValueError("the validation set must be document-disjoint from training")
    flow_loss = validation["flow_loss"]

    phase_a = raw["phase_a"]
    t_starts = tuple(float(value) for value in phase_a["t_start"])
    nfes = tuple(int(value) for value in phase_a["nfe"])
    if t_starts != (0.10, 0.25, 0.50) or nfes != (1, 3, 5):
        raise ValueError("the concept-independent inference grid drifted")
    if phase_a.get("skip_bos") is not True:
        raise ValueError("Phase A must protect the untrained BOS activation")
    if phase_a.get("matched_epsilon_across_arms") is not True:
        raise ValueError("checkpoint comparisons require matched epsilon")
    if int(phase_a["n_sequences"]) != 256:
        raise ValueError("the frozen validation-sequence count changed")

    rule = raw["selection_rule"]
    if rule.get("frozen_before_results") is not True:
        raise ValueError("the selection rule must be frozen before results")
    if rule.get("primary") != "mean_reconstructed_delta_lm_at_primary_cell":
        raise ValueError("the primary selection metric changed")
    if rule.get("primary_direction") != "lower_is_better":
        raise ValueError("the primary selection direction changed")
    forbidden = set(rule["forbidden_inputs"])
    if not {"steering metrics", "Phase-B results", "held-out directions"} <= forbidden:
        raise ValueError("the selection rule no longer forbids steering-derived inputs")

    return ScalingProtocol(
        experiment_id=str(raw["experiment_id"]),
        arms=arms,
        max_parameters=int(capacity["max_parameters"]),
        total_activation_presentations=presentations,
        optimizer_steps=steps,
        batch_size=batch_size,
        validation_artifact=str(validation["artifact"]),
        validation_split=str(validation["split"]),
        per_seq=int(validation["per_seq"]),
        val_fraction=float(validation["val_fraction"]),
        split_seed=int(validation["split_seed"]),
        split_fingerprint=str(validation["split_fingerprint"]),
        flow_loss_seed=int(flow_loss["seed"]),
        flow_loss_batch_size=int(flow_loss["batch_size"]),
        t_bins=tuple(float(value) for value in flow_loss["t_bins"]),
        n_sequences=int(phase_a["n_sequences"]),
        ctx=int(phase_a["ctx"]),
        hook=str(phase_a["hook"]),
        lm_batch_size=int(phase_a["lm_batch_size"]),
        noise_seed=int(phase_a["noise_seed"]),
        identity_nfe=int(phase_a["identity_nfe"]),
        t_starts=t_starts,
        nfes=nfes,
        primary_t_start=float(phase_a["primary_t_start"]),
        primary_nfe=int(phase_a["primary_nfe"]),
        bootstrap_seed=int(phase_a["bootstrap_seed"]),
        bootstrap_resamples=int(phase_a["bootstrap_resamples"]),
        bootstrap_confidence=float(phase_a["bootstrap_confidence"]),
        tie_breakers=tuple(str(value) for value in rule["tie_breakers"]),
        raw=raw,
    )


def arm_phase_a_config(
    protocol: ScalingProtocol,
    arm: ScalingArm,
    *,
    checkpoint_path: Path,
    run_meta_path: Path,
    best_pointer_path: Path,
    checkpoint_sha256: str,
    run_meta_sha256: str,
    best_pointer_sha256: str,
    checkpoint_step: int,
    history_entries: int,
    training_experiment_id: str,
    validation_artifact_sha256: dict[str, str],
    validation_token_cache_sha256: str,
    dataset_repository: str,
    dataset_config: str,
    dataset_revision: str,
) -> PhaseAConfig:
    """Bind one trained arm to the frozen Phase-A measurement on the shared validation set.

    The checkpoint digests are observed, not preregistered: they cannot exist
    before the run. What is preregistered is the protocol, the grid, and the
    selection rule.
    """

    return PhaseAConfig(
        experiment_id=f"{protocol.experiment_id}_{arm.arm_id}",
        experiment_class="concept_independent_capacity_data_scaling",
        checkpoint_path=checkpoint_path,
        run_meta_path=run_meta_path,
        best_pointer_path=best_pointer_path,
        checkpoint_sha256=checkpoint_sha256,
        run_meta_sha256=run_meta_sha256,
        best_pointer_sha256=best_pointer_sha256,
        checkpoint_step=checkpoint_step,
        training_experiment_id=training_experiment_id,
        expected_history_entries=history_entries,
        selection_metric="val_flow_mse",
        selection_mode="min",
        dataset_name=protocol.validation_artifact,
        activation_split=protocol.validation_split,
        internal_split="validation_only",
        dataset_repository=dataset_repository,
        dataset_config=dataset_config,
        dataset_revision=dataset_revision,
        tokenizer="gpt2",
        model_name="gpt2-small",
        resolved_model_name="gpt2",
        model_revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
        hook=protocol.hook,
        ctx=protocol.ctx,
        per_seq=protocol.per_seq,
        val_fraction=protocol.val_fraction,
        split_seed=protocol.split_seed,
        split_fingerprint=protocol.split_fingerprint,
        token_cache_sha256=validation_token_cache_sha256,
        artifact_sha256=dict(validation_artifact_sha256),
        n_sequences=protocol.n_sequences,
        sequence_selection="first_sorted_internal_validation_sequences",
        lm_batch_size=protocol.lm_batch_size,
        noise_seed=protocol.noise_seed,
        bootstrap_seed=protocol.bootstrap_seed,
        bootstrap_resamples=protocol.bootstrap_resamples,
        bootstrap_confidence=protocol.bootstrap_confidence,
        t_starts=protocol.t_starts,
        nfes=protocol.nfes,
        skip_bos=True,
        identity_nfe=protocol.identity_nfe,
        primary_t_start=protocol.primary_t_start,
        primary_nfe=protocol.primary_nfe,
        steering_vectors="forbidden",
        dev_directions="forbidden",
        held_out="forbidden",
        raw=protocol.raw,
    )


def validate_selected_flow_prior(
    protocol_path: Path,
    selection_report_path: Path,
    arm_report_path: Path,
    *,
    expected_arm: str,
    training_experiment_id: str,
    checkpoint_path: Path,
    run_meta_path: Path,
    best_pointer_path: Path,
    role: str = "selected",
) -> dict:
    """Prove that a checkpoint is an arm of the frozen concept-independent 2x2.

    This is the provenance bridge a Phase-B rerun must cross before it may load a
    prior that did not come from the original Phase-A gate. It refuses any report
    that recorded steering, DEV, held-out, Phase-B, or LLM-judge access, and it
    reuses ``validate_frozen_checkpoint`` verbatim for the checkpoint itself.

    ``role="selected"`` requires the arm the frozen rule chose. ``role=
    "capacity_control"`` requires a *different* arm of the same frozen run, which
    is how a sibling arm may be evaluated as a controlled contrast without any
    steering evidence entering the choice. Both roles verify the arm report is the
    exact one the selection consumed, so neither can smuggle in an arm that was
    never part of the preregistered 2x2.
    """

    if role not in ("selected", "capacity_control"):
        raise ValueError(f"unsupported flow-prior role {role!r}")

    protocol = load_scaling_protocol(protocol_path)
    protocol_sha256 = file_sha256(protocol_path)
    selection_sha256 = file_sha256(selection_report_path)
    arm_sha256 = file_sha256(arm_report_path)
    selection = json.loads(selection_report_path.read_text())
    report = json.loads(arm_report_path.read_text())

    for name, document in (("selection", selection), ("arm", report)):
        if document.get("status") != "complete":
            raise ValueError(f"scaling {name} report is not complete")
        if document.get("experiment_id") != protocol.experiment_id:
            raise ValueError(f"scaling {name} report belongs to another experiment")
        if document.get("protocol", {}).get("sha256") != protocol_sha256:
            raise ValueError(f"scaling {name} report was produced under a different protocol")
        if any(document["protected_data"].values()):
            raise ValueError(f"scaling {name} report records protected-data access")

    rule = selection["selection"]
    if rule.get("steering_metrics_used") is not False or rule.get("phase_b_used") is not False:
        raise ValueError("the scaling selection was not concept independent")
    if role == "selected":
        if rule.get("selected_arm") != expected_arm:
            raise ValueError(
                f"frozen rule selected {rule.get('selected_arm')!r}, not {expected_arm!r}"
            )
    elif rule.get("selected_arm") == expected_arm:
        raise ValueError(
            f"{expected_arm!r} is the selected arm and cannot also be its own capacity control"
        )
    if report.get("arm") != expected_arm:
        raise ValueError("the arm report is not the selected arm")
    rows = {row["arm"]: row for row in selection["arms"]}
    if rows[expected_arm]["report_sha256"] != arm_sha256:
        raise ValueError("the arm report is not the one the selection consumed")

    arm = protocol.arm(expected_arm)
    checkpoint = report["checkpoint_selection"]
    validation = report["validation_artifact"]["validation_report"]
    phase_a_config = arm_phase_a_config(
        protocol,
        arm,
        checkpoint_path=checkpoint_path,
        run_meta_path=run_meta_path,
        best_pointer_path=best_pointer_path,
        checkpoint_sha256=checkpoint["checkpoint_sha256"],
        run_meta_sha256=checkpoint["run_meta_sha256"],
        best_pointer_sha256=checkpoint["best_pointer_sha256"],
        checkpoint_step=int(checkpoint["checkpoint_step"]),
        history_entries=int(checkpoint["history_entries"]),
        training_experiment_id=training_experiment_id,
        validation_artifact_sha256=validation["sha256"],
        validation_token_cache_sha256=validation["token_cache_sha256"],
        dataset_repository=validation["dataset_repository"],
        dataset_config=validation["dataset_config"],
        dataset_revision=validation["dataset_revision"],
    )
    selection_receipt = validate_frozen_checkpoint(checkpoint_path, phase_a_config)
    return {
        "provenance_kind": "concept_independent_scaling_selection",
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha256},
        "selection_report_sha256": selection_sha256,
        "arm_report_sha256": arm_sha256,
        "selected_arm": expected_arm,
        "decided_by": rule["decided_by"],
        "parameters": arm.parameters,
        "training_dataset": arm.dataset,
        "unique_activation_tokens": arm.unique_activation_tokens,
        "training_dataset_artifact_identity": report["training_identity"][
            "training_dataset_artifact_identity"
        ],
        "checkpoint_selection": selection_receipt,
        "steering_metrics_used": False,
        "phase_b_used": False,
    }


@torch.no_grad()
def validation_flow_loss(
    model,  # noqa: ANN001
    dataset: ActivationDataset,
    indices: np.ndarray,
    protocol: ScalingProtocol,
    device: torch.device,
) -> dict:
    """Flow MSE on the shared validation rows, with identical times and noise per arm."""

    ordered = np.sort(np.asarray(indices, dtype=np.int64))
    batch_size = protocol.flow_loss_batch_size
    n_batches = len(ordered) // batch_size
    if n_batches < 1:
        raise ValueError("validation flow loss requires at least one full batch")
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(protocol.flow_loss_seed)
    bins = list(zip(protocol.t_bins, protocol.t_bins[1:], strict=False))
    losses: list[float] = []
    controls: list[float] = []
    cosines: list[float] = []
    by_bin: dict[str, list[float]] = {f"{left:.2f}-{right:.2f}": [] for left, right in bins}

    for batch_index in range(n_batches):
        rows = ordered[batch_index * batch_size : (batch_index + 1) * batch_size]
        values = np.array(dataset.array[rows], dtype=np.float32, copy=True)
        x0 = model.normalizer.normalize(torch.from_numpy(values).to(device))
        batch = sample_flow_batch(x0, generator=generator)
        prediction = model(batch.x_t, batch.t)
        losses.append(float(flow_matching_loss(prediction, batch.target_velocity)))
        controls.append(float(batch.target_velocity.square().mean()))
        cosines.append(
            float(torch.cosine_similarity(prediction, batch.target_velocity, dim=-1).mean())
        )
        for (left, right), key in zip(bins, by_bin, strict=True):
            time = left + (right - left) * torch.rand(
                (x0.shape[0], 1), generator=generator, device=device, dtype=x0.dtype
            )
            noise = torch.randn(x0.shape, generator=generator, device=device, dtype=x0.dtype)
            state = (1.0 - time) * x0 + time * noise
            by_bin[key].append(float(flow_matching_loss(model(state, time), noise - x0)))
    model.train(was_training)
    return {
        "val_flow_mse": float(np.mean(losses)),
        "zero_predictor_mse": float(np.mean(controls)),
        "val_cosine_velocity": float(np.mean(cosines)),
        "val_flow_mse_by_bin": {key: float(np.mean(value)) for key, value in by_bin.items()},
        "n_rows": int(n_batches * batch_size),
        "batches": n_batches,
        "seed": protocol.flow_loss_seed,
        "device": str(device),
    }


def _tie_break(candidates: list[dict], tie_breakers: tuple[str, ...]) -> dict:
    ranked = list(candidates)
    for rule in tie_breakers:
        if rule == "lower validation flow MSE on resid7_fw_val_1024k_v1":
            key = "val_flow_mse"
        elif rule == "fewer parameters":
            key = "parameters"
        elif rule == "fewer unique activation tokens":
            key = "unique_activation_tokens"
        else:
            raise ValueError(f"unsupported frozen tie-breaker {rule!r}")
        best = min(float(item[key]) for item in ranked)
        ranked = [item for item in ranked if float(item[key]) == best]
        if len(ranked) == 1:
            return {**ranked[0], "decided_by": rule}
    if len(ranked) != 1:
        raise ValueError("the frozen tie-breakers did not resolve a unique arm")
    return {**ranked[0], "decided_by": tie_breakers[-1]}


def apply_selection_rule(arms: list[dict], protocol: ScalingProtocol) -> dict:
    """Apply the frozen concept-independent selection rule to completed arm reports.

    Each entry needs ``arm_id``, ``primary_effects`` (per-sequence reconstructed
    delta LM at the primary cell), ``val_flow_mse``, ``parameters``, and
    ``unique_activation_tokens``.
    """

    if not arms:
        raise ValueError("selection requires at least one completed arm")
    values = {}
    for entry in arms:
        effects = np.asarray(entry["primary_effects"], dtype=np.float64)
        if effects.ndim != 1 or len(effects) != protocol.n_sequences:
            raise ValueError(
                f"arm {entry['arm_id']!r} must report exactly {protocol.n_sequences} "
                "per-sequence primary values"
            )
        if not np.isfinite(effects).all():
            raise ValueError(f"arm {entry['arm_id']!r} reported non-finite primary values")
        values[entry["arm_id"]] = effects

    means = {arm_id: float(effects.mean()) for arm_id, effects in values.items()}
    leader = min(means, key=lambda arm_id: means[arm_id])
    comparisons = {}
    tied = [leader]
    for arm_id, effects in values.items():
        if arm_id == leader:
            continue
        difference = values[leader] - effects
        bootstrap = paired_bootstrap_mean_ci(
            difference,
            seed=protocol.bootstrap_seed,
            n_resamples=protocol.bootstrap_resamples,
            confidence=protocol.bootstrap_confidence,
        )
        resolved = bootstrap["ci_upper"] < 0.0
        comparisons[f"{leader}_minus_{arm_id}"] = {**bootstrap, "leader_resolved_better": resolved}
        if not resolved:
            tied.append(arm_id)

    by_id = {entry["arm_id"]: entry for entry in arms}
    if len(tied) == 1:
        selected = {**by_id[leader], "decided_by": "primary paired bootstrap"}
    else:
        selected = _tie_break(
            [
                {
                    "arm_id": arm_id,
                    "val_flow_mse": by_id[arm_id]["val_flow_mse"],
                    "parameters": by_id[arm_id]["parameters"],
                    "unique_activation_tokens": by_id[arm_id]["unique_activation_tokens"],
                }
                for arm_id in tied
            ],
            protocol.tie_breakers,
        )
    return {
        "rule": "frozen concept-independent scaling selection rule v1",
        "primary_cell": f"t{protocol.primary_t_start:.2f}_nfe{protocol.primary_nfe}",
        "primary_means": means,
        "leader": leader,
        "tied_with_leader": sorted(tied),
        "comparisons": comparisons,
        "selected_arm": selected["arm_id"],
        "decided_by": selected["decided_by"],
        "steering_metrics_used": False,
        "phase_b_used": False,
    }
