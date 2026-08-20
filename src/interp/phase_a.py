"""Frozen, concept-independent Phase A reconstruction specification and analysis."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml

from .activations import file_sha256, make_split
from .flow_sampling import flow_correct, partial_noise
from .functional import sequence_delta_lm, sequence_lm_losses

APPROVED_PHASE_A_CONFIG_SHA256 = (
    "a762c27be901d6015a06794c695ff8893b107c93861e2599e944dabec4855b5e"
)


@dataclass(frozen=True)
class PhaseAConfig:
    experiment_id: str
    experiment_class: str
    checkpoint_path: Path
    run_meta_path: Path
    best_pointer_path: Path
    checkpoint_sha256: str
    run_meta_sha256: str
    best_pointer_sha256: str
    checkpoint_step: int
    training_experiment_id: str
    expected_history_entries: int
    selection_metric: str
    selection_mode: str
    dataset_name: str
    activation_split: str
    internal_split: str
    dataset_repository: str
    dataset_config: str
    dataset_revision: str
    tokenizer: str
    model_name: str
    resolved_model_name: str
    model_revision: str
    hook: str
    ctx: int
    per_seq: int
    val_fraction: float
    split_seed: int
    split_fingerprint: str
    token_cache_sha256: str
    artifact_sha256: dict[str, str]
    n_sequences: int
    sequence_selection: str
    lm_batch_size: int
    noise_seed: int
    bootstrap_seed: int
    bootstrap_resamples: int
    bootstrap_confidence: float
    t_starts: tuple[float, ...]
    nfes: tuple[int, ...]
    skip_bos: bool
    identity_nfe: int
    primary_t_start: float
    primary_nfe: int
    steering_vectors: str
    dev_directions: str
    held_out: str
    raw: dict


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a full SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a full SHA-256 digest") from error
    return value


def load_phase_a_config(path: Path) -> PhaseAConfig:
    """Load the human-approved Phase A specification and reject semantic drift."""

    observed_config_sha256 = file_sha256(path)
    if observed_config_sha256 != APPROVED_PHASE_A_CONFIG_SHA256:
        raise ValueError(
            f"approved config SHA-256 {observed_config_sha256} != frozen "
            f"{APPROVED_PHASE_A_CONFIG_SHA256}"
        )
    raw = yaml.safe_load(path.read_text())
    if raw.get("version") != 1 or raw.get("status") != "frozen":
        raise ValueError("Phase A config must be frozen at version 1")
    if raw.get("experiment_class") != "dev_method_development":
        raise ValueError("Phase A must remain DEV method development")

    checkpoint = raw["checkpoint"]
    data = raw["data"]
    evaluation = raw["evaluation"]
    gate = raw["gate"]
    protected = raw["protected_data"]

    if checkpoint.get("selection_metric") != "val_flow_mse":
        raise ValueError("checkpoint selection must use concept-independent val_flow_mse")
    if checkpoint.get("selection_mode") != "min":
        raise ValueError("checkpoint selection must minimize val_flow_mse")
    if checkpoint.get("training_experiment_id") != "clean_flow_100k_v1":
        raise ValueError("Phase A must use the completed clean 100k training run")
    if int(checkpoint.get("expected_history_entries", 0)) != 200:
        raise ValueError("Phase A must bind the complete 200-entry validation history")
    step = int(checkpoint["step"])
    checkpoint_path = Path(checkpoint["path"])
    if checkpoint_path.name != f"best_step_{step:06d}.pt":
        raise ValueError("checkpoint filename and frozen step disagree")

    expected_data = {
        "activation_split": "train",
        "internal_split": "validation_only",
        "hook": "blocks.7.hook_resid_pre",
        "ctx": 128,
        "per_seq": 127,
    }
    for field, expected in expected_data.items():
        if data.get(field) != expected:
            raise ValueError(f"data.{field} must remain {expected!r}")

    t_starts = tuple(float(value) for value in evaluation["t_start"])
    nfes = tuple(int(value) for value in evaluation["nfe"])
    if t_starts != (0.10, 0.25, 0.50) or nfes != (1, 3, 5):
        raise ValueError("Phase A inference grid drifted from the approved grid")
    if evaluation.get("skip_bos") is not True:
        raise ValueError("Phase A must protect the untrained BOS activation")
    if int(evaluation["n_sequences"]) != 256:
        raise ValueError("Phase A must use exactly 256 validation sequences")
    if evaluation.get("sequence_selection") != "first_sorted_internal_validation_sequences":
        raise ValueError("Phase A validation-sequence selection rule drifted")
    if int(evaluation["lm_batch_size"]) < 1:
        raise ValueError("lm_batch_size must be positive")
    if int(evaluation["bootstrap_resamples"]) < 1:
        raise ValueError("bootstrap_resamples must be positive")
    confidence = float(evaluation["bootstrap_confidence"])
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap_confidence must lie strictly inside (0, 1)")

    primary = (float(gate["primary_t_start"]), int(gate["primary_nfe"]))
    if primary != (0.50, 1):
        raise ValueError("Phase A primary cell drifted from t_start=0.50, NFE=1")
    expected_gate = {
        "paired_effect": "reconstructed_delta_lm_minus_corrupted_delta_lm",
        "unit": "validation_sequence",
        "require_mean_below_zero": True,
        "require_ci_upper_below_zero": True,
        "require_positive_corruption_each_t_start": True,
        "require_exact_identity_delta_zero": True,
        "require_zero_identity_flow_evaluations": True,
        "require_finite_outputs_and_geometry": True,
        "diagnostic_cells_cannot_select": True,
    }
    for field, expected in expected_gate.items():
        if gate.get(field) != expected:
            raise ValueError(f"gate.{field} must remain {expected!r}")
    for field in ("steering_vectors", "dev_directions", "held_out"):
        if protected.get(field) != "forbidden":
            raise ValueError(f"protected_data.{field} must be forbidden")

    artifact_sha256 = dict(data["artifact_sha256"])
    if set(artifact_sha256) != {"array", "metadata", "statistics"}:
        raise ValueError("data.artifact_sha256 must identify array, metadata, and statistics")
    for name, digest in artifact_sha256.items():
        _require_sha256(digest, f"data.artifact_sha256.{name}")

    return PhaseAConfig(
        experiment_id=str(raw["experiment_id"]),
        experiment_class=str(raw["experiment_class"]),
        checkpoint_path=checkpoint_path,
        run_meta_path=Path(checkpoint["run_meta_path"]),
        best_pointer_path=Path(checkpoint["best_pointer_path"]),
        checkpoint_sha256=_require_sha256(checkpoint["sha256"], "checkpoint.sha256"),
        run_meta_sha256=_require_sha256(
            checkpoint["run_meta_sha256"], "checkpoint.run_meta_sha256"
        ),
        best_pointer_sha256=_require_sha256(
            checkpoint["best_pointer_sha256"], "checkpoint.best_pointer_sha256"
        ),
        checkpoint_step=step,
        training_experiment_id=str(checkpoint["training_experiment_id"]),
        expected_history_entries=int(checkpoint["expected_history_entries"]),
        selection_metric=str(checkpoint["selection_metric"]),
        selection_mode=str(checkpoint["selection_mode"]),
        dataset_name=str(data["dataset"]),
        activation_split=str(data["activation_split"]),
        internal_split=str(data["internal_split"]),
        dataset_repository=str(data["repository"]),
        dataset_config=str(data["repository_config"]),
        dataset_revision=str(data["repository_revision"]),
        tokenizer=str(data["tokenizer"]),
        model_name=str(data["model"]),
        resolved_model_name=str(data["resolved_model_name"]),
        model_revision=str(data["model_revision"]),
        hook=str(data["hook"]),
        ctx=int(data["ctx"]),
        per_seq=int(data["per_seq"]),
        val_fraction=float(data["val_fraction"]),
        split_seed=int(data["split_seed"]),
        split_fingerprint=str(data["split_fingerprint"]),
        token_cache_sha256=_require_sha256(
            data["token_cache_sha256"], "data.token_cache_sha256"
        ),
        artifact_sha256=artifact_sha256,
        n_sequences=int(evaluation["n_sequences"]),
        sequence_selection=str(evaluation["sequence_selection"]),
        lm_batch_size=int(evaluation["lm_batch_size"]),
        noise_seed=int(evaluation["noise_seed"]),
        bootstrap_seed=int(evaluation["bootstrap_seed"]),
        bootstrap_resamples=int(evaluation["bootstrap_resamples"]),
        bootstrap_confidence=confidence,
        t_starts=t_starts,
        nfes=nfes,
        skip_bos=bool(evaluation["skip_bos"]),
        identity_nfe=int(evaluation["identity_nfe"]),
        primary_t_start=primary[0],
        primary_nfe=primary[1],
        steering_vectors=str(protected["steering_vectors"]),
        dev_directions=str(protected["dev_directions"]),
        held_out=str(protected["held_out"]),
        raw=raw,
    )


def validate_frozen_checkpoint(
    checkpoint_path: Path,
    cfg: PhaseAConfig,
) -> dict:
    """Prove that the frozen file is the concept-independent history minimum."""

    expected_name = f"best_step_{cfg.checkpoint_step:06d}.pt"
    if checkpoint_path != cfg.checkpoint_path or checkpoint_path.name != expected_name:
        raise ValueError("checkpoint path does not match the frozen checkpoint")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"frozen checkpoint is absent: {checkpoint_path}")
    observed_sha256 = file_sha256(checkpoint_path)
    if observed_sha256 != cfg.checkpoint_sha256:
        raise ValueError(
            f"checkpoint SHA-256 {observed_sha256} != frozen {cfg.checkpoint_sha256}"
        )
    sidecar_specs = (
        (cfg.run_meta_path, cfg.run_meta_sha256, "run metadata"),
        (cfg.best_pointer_path, cfg.best_pointer_sha256, "best pointer"),
    )
    for path, expected_sha256, label in sidecar_specs:
        if not path.is_file():
            raise FileNotFoundError(f"frozen {label} sidecar is absent: {path}")
        sidecar_sha256 = file_sha256(path)
        if sidecar_sha256 != expected_sha256:
            raise ValueError(
                f"{label} sidecar SHA-256 {sidecar_sha256} != frozen {expected_sha256}"
            )
    run_meta = json.loads(cfg.run_meta_path.read_text())
    best = json.loads(cfg.best_pointer_path.read_text())
    if not isinstance(run_meta, dict) or not isinstance(best, dict):
        raise ValueError("checkpoint selection sidecars must contain JSON objects")
    if run_meta.get("status") != "complete":
        raise ValueError("checkpoint selection run metadata must be complete")
    if run_meta.get("experiment_id") != cfg.training_experiment_id:
        raise ValueError("checkpoint selection run has the wrong experiment identity")
    if run_meta.get("held_out_accessed") is not False:
        raise ValueError("checkpoint metadata does not prove held-out remained untouched")
    if run_meta.get("selection_metric") != cfg.selection_metric:
        raise ValueError("checkpoint was not selected by concept-independent val_flow_mse")
    if (
        best.get("selection_metric") != cfg.selection_metric
        or best.get("selection_mode") != cfg.selection_mode
    ):
        raise ValueError("best pointer was not selected by concept-independent val_flow_mse")
    if (
        run_meta.get("best_checkpoint") != checkpoint_path.name
        or best.get("checkpoint") != checkpoint_path.name
    ):
        raise ValueError("best checkpoint pointer does not match the frozen checkpoint")

    history = run_meta.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("checkpoint history must be nonempty")
    if len(history) != cfg.expected_history_entries:
        raise ValueError(
            f"checkpoint history entries {len(history)} != frozen "
            f"{cfg.expected_history_entries}"
        )
    pairs: list[tuple[float, int]] = []
    for row in history:
        value = float(row[cfg.selection_metric])
        step = int(row["step"])
        if not math.isfinite(value):
            raise ValueError("checkpoint history contains non-finite validation loss")
        pairs.append((value, step))
    minimum_value = min(value for value, _ in pairs)
    minimum_steps = [step for value, step in pairs if value == minimum_value]
    if minimum_steps != [cfg.checkpoint_step]:
        if cfg.checkpoint_step in minimum_steps:
            raise ValueError("frozen checkpoint must be the unique history minimum")
        raise ValueError("frozen checkpoint is not the history minimum")
    if float(run_meta.get("best_val_flow_mse")) != minimum_value:
        raise ValueError("run metadata best value is not the history minimum")
    if float(best.get("value")) != minimum_value:
        raise ValueError("best pointer value is not the history minimum")

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": observed_sha256,
        "checkpoint_step": cfg.checkpoint_step,
        "run_meta_sha256": cfg.run_meta_sha256,
        "best_pointer_sha256": cfg.best_pointer_sha256,
        "run_status": "complete",
        "selection_metric": cfg.selection_metric,
        "selection_mode": cfg.selection_mode,
        "selection_value": minimum_value,
        "history_entries": len(history),
        "held_out_accessed": False,
    }


def validate_checkpoint_payload(metadata: dict, cfg: PhaseAConfig) -> dict:
    """Bind loaded weights to the frozen step, split, and exact activation artifacts."""

    expected = {
        "step": cfg.checkpoint_step,
        "dataset": cfg.dataset_name,
        "split_fingerprint": cfg.split_fingerprint,
        "dataset_artifact_identity": {
            "sha256": cfg.artifact_sha256,
            "token_cache_sha256": cfg.token_cache_sha256,
        },
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                f"checkpoint payload {field}={metadata.get(field)!r} != frozen {value!r}"
            )
    return expected


def matched_gaussian_noise(
    sequence_ids: np.ndarray,
    *,
    positions: int,
    d_model: int,
    seed: int,
) -> torch.Tensor:
    """Generate one deterministic Gaussian tensor for each immutable sequence ID."""

    ids = np.asarray(sequence_ids)
    if (
        ids.ndim != 1
        or len(ids) == 0
        or not np.issubdtype(ids.dtype, np.integer)
        or np.any(ids < 0)
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("sequence_ids must be a nonempty vector of unique nonnegative integers")
    if positions < 1 or d_model < 1:
        raise ValueError("positions and d_model must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    rows = []
    for sequence_id in ids:
        digest = hashlib.sha256(f"{seed}:{int(sequence_id)}".encode()).digest()
        sequence_seed = int.from_bytes(digest[:16], byteorder="little", signed=False)
        generator = np.random.Generator(np.random.PCG64(sequence_seed))
        rows.append(generator.standard_normal((positions, d_model), dtype=np.float32))
    return torch.from_numpy(np.stack(rows))


def select_validation_sequence_ids(n_tokens: int, cfg: PhaseAConfig) -> np.ndarray:
    """Select the frozen first sorted document IDs from the internal validation split."""

    if cfg.internal_split != "validation_only":
        raise ValueError("Phase A sequence selection requires the internal validation split")
    if cfg.sequence_selection != "first_sorted_internal_validation_sequences":
        raise ValueError("unsupported Phase A sequence-selection rule")
    split = make_split(n_tokens, cfg.per_seq, cfg.val_fraction, cfg.split_seed)
    if split.fingerprint() != cfg.split_fingerprint:
        raise ValueError(
            f"split fingerprint {split.fingerprint()} != frozen {cfg.split_fingerprint}"
        )
    validation_ids = np.unique(split.val // cfg.per_seq)
    if len(validation_ids) < cfg.n_sequences:
        raise ValueError("internal validation split has too few sequences for Phase A")
    selected = np.asarray(validation_ids[: cfg.n_sequences], dtype=np.int64)
    training_ids = np.unique(split.train // cfg.per_seq)
    if np.intersect1d(selected, training_ids).size:
        raise RuntimeError("Phase A validation sequence selection overlaps training")
    return selected


def paired_bootstrap_mean_ci(
    effects: np.ndarray,
    *,
    seed: int,
    n_resamples: int,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Bootstrap a paired mean over validation-sequence effects, never tokens."""

    values = np.asarray(effects, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("paired effects must be a nonempty one-dimensional sequence vector")
    if not np.isfinite(values).all():
        raise ValueError("paired effects must be finite")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples < 1:
        raise ValueError("bootstrap n_resamples must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie strictly inside (0, 1)")

    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return {
        "mean": float(values.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": float(confidence),
        "n_units": int(len(values)),
        "n_resamples": int(n_resamples),
        "unit": "validation_sequence",
    }


def reconstruction_geometry(
    clean: torch.Tensor, reconstructed: torch.Tensor
) -> dict[str, float | int]:
    """Return finite tokenwise relative-L2 and cosine reconstruction summaries."""

    if clean.ndim != 2 or clean.shape != reconstructed.shape or clean.shape[0] == 0:
        raise ValueError("geometry expects same-shaped nonempty [activation, d_model] tensors")
    if (
        not clean.is_floating_point()
        or not reconstructed.is_floating_point()
        or not torch.isfinite(clean).all()
        or not torch.isfinite(reconstructed).all()
    ):
        raise ValueError("geometry inputs must be finite and floating point")
    norms = clean.norm(dim=-1)
    if not bool((norms > 0).all()):
        raise ValueError("relative geometry requires nonzero clean activation norms")
    relative = (reconstructed - clean).norm(dim=-1) / norms
    cosine = torch.nn.functional.cosine_similarity(reconstructed, clean, dim=-1)
    if not torch.isfinite(relative).all() or not torch.isfinite(cosine).all():
        raise ValueError("reconstruction geometry must be finite")
    return {
        "mean_relative_l2": float(relative.double().mean()),
        "mean_cosine": float(cosine.double().mean()),
        "n_activations": int(clean.shape[0]),
    }


def _finite_nested(values: object) -> bool:
    if isinstance(values, dict):
        return all(_finite_nested(value) for value in values.values())
    if isinstance(values, (np.ndarray, torch.Tensor)):
        array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values
        return bool(np.isfinite(array).all())
    if isinstance(values, (float, int, np.number)) and not isinstance(values, bool):
        return math.isfinite(float(values))
    return True


def assess_phase_a_gate(
    *,
    primary_effects: np.ndarray,
    bootstrap: dict,
    identity_deltas: np.ndarray,
    identity_flow_evaluations: int,
    corruption_deltas: dict[float, np.ndarray],
    all_functional_values: dict[str, np.ndarray],
    all_geometry: dict[str, dict[str, float]],
) -> dict:
    """Apply only the approved primary paired gate and invariant safety controls."""

    effects = np.asarray(primary_effects, dtype=np.float64)
    identity = np.asarray(identity_deltas, dtype=np.float64)
    if effects.ndim != 1 or len(effects) == 0:
        raise ValueError("primary effects must be a nonempty validation-sequence vector")
    if identity.shape != effects.shape:
        raise ValueError("identity deltas must use the same validation-sequence unit")
    if bootstrap.get("unit") != "validation_sequence" or bootstrap.get("n_units") != len(effects):
        raise ValueError("bootstrap must use the paired validation-sequence unit")
    observed_mean = float(effects.mean())
    if not math.isclose(
        float(bootstrap.get("mean", float("nan"))), observed_mean, rel_tol=1e-12, abs_tol=1e-15
    ):
        raise ValueError("bootstrap mean does not match the primary paired effects")

    corruption_positive = bool(corruption_deltas) and all(
        np.asarray(values).ndim == 1
        and len(values) > 0
        and float(np.asarray(values, dtype=np.float64).mean()) > 0.0
        for values in corruption_deltas.values()
    )
    finite = _finite_nested(
        {
            "primary_effects": effects,
            "bootstrap": bootstrap,
            "identity_deltas": identity,
            "corruption_deltas": corruption_deltas,
            "all_functional_values": all_functional_values,
            "all_geometry": all_geometry,
        }
    )
    criteria = {
        "primary_mean_below_zero": math.isfinite(observed_mean) and observed_mean < 0.0,
        "primary_ci_upper_below_zero": math.isfinite(
            float(bootstrap.get("ci_upper", float("nan")))
        )
        and float(bootstrap["ci_upper"]) < 0.0,
        "identity_delta_exact_zero": bool(np.equal(identity, 0.0).all()),
        "identity_zero_flow_evaluations": identity_flow_evaluations == 0,
        "positive_aggregate_corruption_each_t_start": corruption_positive,
        "finite_outputs_and_geometry": finite,
    }
    passed = all(criteria.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "research_status": "SUPPORTED" if passed else "NOT_SUPPORTED",
        "primary_effect_definition": "delta_lm_reconstructed_minus_delta_lm_corrupted",
        "criteria": criteria,
        "primary_effect": bootstrap,
    }


class _CountingVelocity:
    def __init__(self, model) -> None:  # noqa: ANN001
        self.model = model
        self.normalizer = model.normalizer
        self.evaluations = 0

    def __call__(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        self.evaluations += 1
        return self.model(state, time)


class _ConditionTransform:
    def __init__(
        self,
        flow_model,
        noise: torch.Tensor,
        *,
        t_start: float,
        nfe: int,
        integrate: bool,
    ) -> None:  # noqa: ANN001
        self.flow = _CountingVelocity(flow_model)
        self.noise = noise
        self.t_start = t_start
        self.nfe = nfe
        self.integrate = integrate
        self.sequence_offset = 0
        self.relative_l2_sum = 0.0
        self.cosine_sum = 0.0
        self.n_activations = 0

    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.ndim != 3:
            raise ValueError("Phase A hook transform expects [sequence, position, d_model]")
        batch = activation.shape[0]
        following = self.sequence_offset + batch
        if following > self.noise.shape[0] or activation.shape[1:] != self.noise.shape[1:]:
            raise ValueError("matched noise shape/order does not match the validation batch")
        noise = self.noise[self.sequence_offset : following].to(
            device=activation.device, dtype=activation.dtype
        )
        self.sequence_offset = following
        flat = activation.reshape(-1, activation.shape[-1])
        flat_noise = noise.reshape_as(flat)
        if self.integrate:
            transformed = flow_correct(
                self.flow,
                flat,
                noise=flat_noise,
                t_start=self.t_start,
                nfe=self.nfe,
            )
        else:
            normalized = self.flow.normalizer.normalize(flat)
            transformed = self.flow.normalizer.denormalize(
                partial_noise(normalized, flat_noise, t_start=self.t_start)
            )
        if not torch.isfinite(transformed).all():
            raise ValueError("Phase A transform produced a non-finite activation")
        clean_norm = flat.norm(dim=-1)
        if not bool((clean_norm > 0).all()):
            raise ValueError("Phase A geometry requires nonzero clean activation norms")
        relative = (transformed - flat).norm(dim=-1) / clean_norm
        cosine = torch.nn.functional.cosine_similarity(transformed, flat, dim=-1)
        if not torch.isfinite(relative).all() or not torch.isfinite(cosine).all():
            raise ValueError("Phase A online geometry became non-finite")
        self.relative_l2_sum += float(relative.double().sum())
        self.cosine_sum += float(cosine.double().sum())
        self.n_activations += flat.shape[0]
        return transformed.reshape_as(activation)

    def geometry(self) -> dict[str, float | int]:
        if self.sequence_offset != self.noise.shape[0]:
            raise ValueError("not every validation sequence consumed its matched noise")
        if self.n_activations == 0:
            raise ValueError("Phase A transform saw no activations")
        return {
            "mean_relative_l2": self.relative_l2_sum / self.n_activations,
            "mean_cosine": self.cosine_sum / self.n_activations,
            "n_activations": self.n_activations,
        }


def _condition_result(
    flow_model,
    language_model,
    tokens: torch.Tensor,
    clean_losses: torch.Tensor,
    noise: torch.Tensor,
    cfg: PhaseAConfig,
    *,
    t_start: float,
    nfe: int,
    integrate: bool,
) -> dict:  # noqa: ANN001
    transform = _ConditionTransform(
        flow_model, noise, t_start=t_start, nfe=nfe, integrate=integrate
    )
    result = sequence_delta_lm(
        language_model,
        tokens,
        transform,
        hook=cfg.hook,
        skip_bos=cfg.skip_bos,
        batch_size=cfg.lm_batch_size,
        clean=clean_losses,
    )
    expected_evaluations = (
        0
        if not integrate or t_start == 0.0
        else math.ceil(tokens.shape[0] / cfg.lm_batch_size) * nfe
    )
    if transform.flow.evaluations != expected_evaluations:
        raise RuntimeError(
            f"flow evaluation count {transform.flow.evaluations} != expected "
            f"{expected_evaluations}"
        )
    deltas = result["delta"].numpy()
    return {
        "t_start": t_start,
        "nfe": nfe if integrate else 0,
        "delta_lm_per_sequence": deltas.tolist(),
        "mean_delta_lm": float(deltas.mean()),
        "flow_evaluations": transform.flow.evaluations,
        "geometry": transform.geometry(),
    }


@torch.no_grad()
def evaluate_phase_a(
    flow_model,
    language_model,
    tokens: torch.Tensor,
    sequence_ids: np.ndarray,
    cfg: PhaseAConfig,
) -> dict:  # noqa: ANN001
    """Evaluate the frozen validation-only reconstruction grid and approved gate."""

    ids = np.asarray(sequence_ids)
    if tokens.ndim != 2 or tokens.shape[0] != cfg.n_sequences:
        raise ValueError("tokens must contain exactly the frozen number of validation sequences")
    if ids.ndim != 1 or len(ids) != cfg.n_sequences:
        raise ValueError("sequence_ids must identify every frozen validation sequence")
    if not hasattr(flow_model, "normalizer"):
        raise ValueError("flow model must carry its train-split normalizer")
    d_model = int(flow_model.normalizer.mean.shape[0])
    noise = matched_gaussian_noise(
        ids,
        positions=tokens.shape[1] - 1,
        d_model=d_model,
        seed=cfg.noise_seed,
    )
    noise_sha256 = hashlib.sha256(noise.contiguous().numpy().tobytes()).hexdigest()
    clean = sequence_lm_losses(
        language_model,
        tokens,
        hook=cfg.hook,
        skip_bos=cfg.skip_bos,
        batch_size=cfg.lm_batch_size,
    )
    identity = _condition_result(
        flow_model,
        language_model,
        tokens,
        clean,
        noise,
        cfg,
        t_start=0.0,
        nfe=cfg.identity_nfe,
        integrate=True,
    )

    corruptions = {}
    cells = {}
    for t_start in cfg.t_starts:
        corruption_key = f"t{t_start:.2f}"
        corruptions[corruption_key] = _condition_result(
            flow_model,
            language_model,
            tokens,
            clean,
            noise,
            cfg,
            t_start=t_start,
            nfe=0,
            integrate=False,
        )
        for nfe in cfg.nfes:
            key = f"t{t_start:.2f}_nfe{nfe}"
            cells[key] = _condition_result(
                flow_model,
                language_model,
                tokens,
                clean,
                noise,
                cfg,
                t_start=t_start,
                nfe=nfe,
                integrate=True,
            )

    primary_key = f"t{cfg.primary_t_start:.2f}_nfe{cfg.primary_nfe}"
    primary_corruption_key = f"t{cfg.primary_t_start:.2f}"
    primary_effects = np.asarray(
        cells[primary_key]["delta_lm_per_sequence"], dtype=np.float64
    ) - np.asarray(
        corruptions[primary_corruption_key]["delta_lm_per_sequence"], dtype=np.float64
    )
    bootstrap = paired_bootstrap_mean_ci(
        primary_effects,
        seed=cfg.bootstrap_seed,
        n_resamples=cfg.bootstrap_resamples,
        confidence=cfg.bootstrap_confidence,
    )
    corruption_deltas = {
        t_start: np.asarray(
            corruptions[f"t{t_start:.2f}"]["delta_lm_per_sequence"], dtype=np.float64
        )
        for t_start in cfg.t_starts
    }
    all_functional_values = {
        "clean": clean.numpy(),
        "identity": np.asarray(identity["delta_lm_per_sequence"]),
        **{
            f"corruption_{key}": np.asarray(value["delta_lm_per_sequence"])
            for key, value in corruptions.items()
        },
        **{
            f"cell_{key}": np.asarray(value["delta_lm_per_sequence"])
            for key, value in cells.items()
        },
    }
    all_geometry = {
        "identity": identity["geometry"],
        **{f"corruption_{key}": value["geometry"] for key, value in corruptions.items()},
        **{f"cell_{key}": value["geometry"] for key, value in cells.items()},
    }
    gate = assess_phase_a_gate(
        primary_effects=primary_effects,
        bootstrap=bootstrap,
        identity_deltas=np.asarray(identity["delta_lm_per_sequence"]),
        identity_flow_evaluations=identity["flow_evaluations"],
        corruption_deltas=corruption_deltas,
        all_functional_values=all_functional_values,
        all_geometry=all_geometry,
    )
    gate["primary_cell"] = primary_key
    gate["primary_effect"] = {
        **bootstrap,
        "definition": "delta_lm_reconstructed_minus_delta_lm_corrupted",
        "values": primary_effects.tolist(),
    }
    diagnostic_cells = sorted(key for key in cells if key != primary_key)
    return {
        "phase": "A_functional_reconstruction",
        "experiment_id": cfg.experiment_id,
        "research_status": gate["research_status"],
        "validation_sequence_ids": [int(value) for value in ids],
        "clean_lm_loss_per_sequence": clean.numpy().tolist(),
        "mean_clean_lm_loss": float(clean.numpy().mean()),
        "noise": {
            "seed": cfg.noise_seed,
            "sha256": noise_sha256,
            "shape": list(noise.shape),
            "construction": "sha256(base_seed:validation_sequence_id)+PCG64",
            "reused_across_t_start": True,
            "reused_across_nfe": True,
        },
        "identity": identity,
        "corruptions": corruptions,
        "cells": cells,
        "gate": gate,
        "diagnostic_only_cells": diagnostic_cells,
        "selection_prohibition": (
            "Diagnostic cells cannot select a checkpoint, t_start, NFE, or redefine the gate."
        ),
    }


def write_immutable_json(path: Path, value: dict) -> None:
    """Write one JSON artifact atomically with exclusive-create overwrite protection."""

    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite immutable output: {path}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_failure_receipt(
    output: Path,
    error: BaseException,
    *,
    status: str,
    command: list[str],
    started_utc: str,
) -> Path:
    """Write an immutable operational failure/interruption receipt beside an output."""

    if status not in {"INVALID", "INTERRUPTED"}:
        raise ValueError("failure receipt status must be INVALID or INTERRUPTED")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    receipt = output.with_name(f"{output.name}.{status}_{stamp}_{uuid4().hex}.json")
    finished_utc = datetime.now(UTC).isoformat()
    write_immutable_json(
        receipt,
        {
            **lifecycle_record(
                status=status,
                command=command,
                started_utc=started_utc,
                finished_utc=finished_utc,
                termination_reason=type(error).__name__,
            ),
            "error": repr(error),
            "intended_output": str(output),
        },
    )
    return receipt


def lifecycle_record(
    *,
    status: str,
    command: list[str],
    started_utc: str,
    finished_utc: str,
    termination_reason: str | None = None,
) -> dict:
    """Return the minimum reproducible lifecycle fields for one evaluation attempt."""

    if status not in {"complete", "INVALID", "INTERRUPTED"}:
        raise ValueError("unsupported evaluation lifecycle status")
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("evaluation command must be a nonempty string list")
    if not started_utc or not finished_utc:
        raise ValueError("evaluation lifecycle requires start and finish timestamps")
    reason = "completed" if status == "complete" else termination_reason
    if not reason:
        raise ValueError("failed/interrupted evaluation requires a termination reason")
    return {
        "status": status,
        "termination_reason": reason,
        "command": command,
        "command_shell": shlex.join(command),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
    }
