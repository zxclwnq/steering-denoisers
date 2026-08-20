"""Concept-independent training and validation for the clean flow matcher."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml
from torch import nn
from tqdm import tqdm

from .activations import (
    ActivationDataset,
    PageCacheGovernor,
    make_split,
    split_stats,
    validate_activation_metadata,
)
from .conditional_flow import (
    ConditionalFlowBatch,
    ConditionalFlowMatcher,
    ConditionEncoderConfig,
    TrainingDirectionPool,
    load_conditional_flow_config,
    load_training_direction_pool,
    sample_conditional_flow_batch,
)
from .flow_core import (
    ActivationNormalizer,
    FlowMatcher,
    FlowModelConfig,
    flow_matching_loss,
    load_flow_config,
    sample_flow_batch,
)
from .model import MODEL_NAME, MODEL_RESOLVED_NAME, MODEL_REVISION, STEERING_HOOK
from .provenance import source_revision
from .steering_denoiser import (
    STEERING_CORRUPTION_SPEC,
    STEERING_DENOISER_OBJECTIVE,
    SteeringCorruptionSpec,
    SteeringDenoiseBatch,
    SteeringDenoiser,
    sample_steering_corruption_batch,
)
from .tangent_flow import (
    ISOTROPIC_OBJECTIVE,
    TANGENT_OBJECTIVES,
    TangentFlowBatch,
    coordinate,
    sample_tangent_flow_batch,
    tangent_path_states,
    tangent_project,
)

# Every objective this trainer can run. The tangent ones are flows on a
# constraint-preserving path; the denoiser is not a flow at all.
NON_DEFAULT_OBJECTIVES = (*TANGENT_OBJECTIVES, STEERING_DENOISER_OBJECTIVE)
ALL_OBJECTIVES = (ISOTROPIC_OBJECTIVE, *NON_DEFAULT_OBJECTIVES)


@dataclass(frozen=True)
class FlowObjectiveSpec:
    """Which corruption geometry the run trains on.

    ``None`` on a ``FlowTrainingConfig`` means the frozen isotropic objective,
    whose code path, RNG consumption, and config fingerprint are unchanged.
    """

    type: str
    output_projection: bool

    def __post_init__(self) -> None:
        if self.type not in NON_DEFAULT_OBJECTIVES:
            raise ValueError(
                f"the non-default flow objectives are {NON_DEFAULT_OBJECTIVES!r}, "
                f"got {self.type!r}"
            )
        if self.type == STEERING_DENOISER_OBJECTIVE and self.output_projection:
            raise ValueError(
                "the steering denoiser predicts a residual, not a velocity; there "
                "is nothing to project, so output_projection must be false"
            )


@dataclass(frozen=True)
class ConditionalTrainingSpec:
    """The conditional variant's extra ingredients: encoder and direction pool.

    ``None`` on a ``FlowTrainingConfig`` means the frozen unconditional trainer,
    whose code path, RNG consumption, and config fingerprint are unchanged.
    """

    condition: ConditionEncoderConfig
    coordinate: str
    direction_pool: str


@dataclass(frozen=True)
class SteeringCorruptionTrainingSpec:
    """Post-stop experiment B's extra ingredient: which directions corrupt, how hard.

    The denoiser needs the training-only pool but no condition encoder, so its
    pool lives here rather than in ``conditioning``. That keeps ``conditioning``
    exactly what it has always been -- the conditional flow's encoder -- and
    keeps every historical config fingerprint unchanged.
    """

    direction_pool: str
    spec: SteeringCorruptionSpec = STEERING_CORRUPTION_SPEC


@dataclass(frozen=True)
class FlowTrainingConfig:
    experiment_id: str
    experiment_class: str
    held_out: str
    dataset_name: str
    dataset_repository: str
    dataset_config: str
    dataset_revision: str
    tokenizer: str
    per_seq: int
    val_fraction: float
    split_seed: int
    split_fingerprint: str
    norm_eps: float
    model: FlowModelConfig
    training_seed: int
    noise_seed: int
    steps: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    grad_clip: float
    eval_every: int
    eval_batches: int
    eval_seed: int
    t_bins: tuple[float, ...]
    selection_metric: str
    save_steps: tuple[int, ...]
    keep: tuple[str, ...]
    raw: dict
    conditioning: ConditionalTrainingSpec | None = None
    flow_objective: FlowObjectiveSpec | None = None
    steering_corruption: SteeringCorruptionTrainingSpec | None = None

    @property
    def status(self) -> str:
        return str(self.raw.get("status"))

    @property
    def objective_type(self) -> str:
        return ISOTROPIC_OBJECTIVE if self.flow_objective is None else self.flow_objective.type

    @property
    def direction_pool_path(self) -> str | None:
        """Where the training-only directions come from, whatever uses them."""

        if self.steering_corruption is not None:
            return self.steering_corruption.direction_pool
        if self.conditioning is not None:
            return self.conditioning.direction_pool
        return None


def load_training_config(path: Path) -> FlowTrainingConfig:
    """Load the authorized 100k DEV training specification and reject semantic drift."""

    raw = yaml.safe_load(path.read_text())
    # "prepared" is schema-valid but not human-approved; train_flow refuses to run it.
    if raw.get("version") != 1 or raw.get("status") not in {"authorized", "prepared"}:
        raise ValueError("expected an authorized or prepared training config at version 1")
    data = raw["data"]
    normalization = raw["normalization"]
    training = raw["training"]
    validation = raw["validation"]
    protected = raw["protected_data"]
    if raw.get("experiment_class") not in {
        "dev_method_development",
        "concept_independent_capacity_data_scaling",
        "conditional_prior_method_development",
        "tangent_prior_method_development",
        # docs/POST_STOP_PROTOCOL_2026-08-19.md: bounded additional checks run
        # after the branch stop rule fired, labelled so no artifact of this pass
        # can be mistaken for a preregistered arm of the closed branches.
        "post_stop_method_development",
    }:
        raise ValueError("flow training must remain concept-independent method development")
    if protected.get("held_out") != "forbidden":
        raise ValueError("held-out access must be forbidden during flow training")
    if "flow_objective" in raw and protected.get("dev") != "forbidden":
        # Tangent training must declare both exclusions explicitly. Historical
        # isotropic configs predate the `dev` key and keep their existing
        # contract, so their fingerprints and loadability are untouched.
        raise ValueError(
            "tangent training must declare protected_data.dev = forbidden; "
            f"got {protected.get('dev')!r}"
        )
    if data.get("split") != "train" or not data.get("bos_dropped"):
        raise ValueError("training requires the BOS-dropped training activation dataset")
    if data.get("steering_vectors_used") is not False:
        raise ValueError("flow training data must contain no steering vectors")
    if normalization.get("statistics_from") != "train_split_only":
        raise ValueError("normalization must use train-split-only statistics")
    if normalization.get("accumulation_dtype") != "float64":
        raise ValueError("normalization statistics must accumulate in float64")
    if training.get("optimizer") != "adamw" or training.get("schedule") != "cosine":
        raise ValueError("the authorized optimizer is AdamW with a cosine schedule")
    if training.get("dtype") != "float32":
        raise ValueError("the authorized training dtype is float32")
    if validation.get("selection_metric") != "val_flow_mse":
        raise ValueError("checkpoint selection must use val_flow_mse only")
    if validation.get("selection_mode") != "min":
        raise ValueError("val_flow_mse checkpoint selection must minimize")
    t_bins = tuple(float(value) for value in validation["t_bins"])
    if t_bins[0] != 0.0 or t_bins[-1] != 1.0 or any(
        right <= left for left, right in pairwise(t_bins)
    ):
        raise ValueError("validation t_bins must increase strictly from zero to one")
    steps = int(training["steps"])
    keep = tuple(str(item) for item in raw["checkpoints"]["keep"])
    if keep != ("best", "last", "configured_steps"):
        raise ValueError(
            "the only implemented retention policy is [best, last, configured_steps]"
        )
    save_steps = tuple(int(step) for step in raw["checkpoints"]["save_steps"])
    if any(step <= 0 or step > steps or step % int(training["eval_every"]) for step in save_steps):
        raise ValueError("every configured save step must be an in-range evaluation step")
    project_root = path.resolve().parents[1]
    model_config_path = project_root / raw["flow_core_config"]
    conditioning = None
    if "conditioning" in raw:
        conditional = load_conditional_flow_config(model_config_path)
        pool = raw["conditioning"]
        if pool.get("direction_pool_split") != "training_only":
            raise ValueError("conditional training may only use a training_only pool")
        if pool.get("steering_vector_conditioning") is not False:
            raise ValueError(
                "conditioning must come from the training-only pool, never from "
                "DEV or held-out steering vectors"
            )
        conditioning = ConditionalTrainingSpec(
            condition=conditional.condition,
            coordinate=conditional.coordinate,
            direction_pool=str(pool["direction_pool"]),
        )
    steering_corruption = None
    if "steering_corruption" in raw:
        section = raw["steering_corruption"]
        if section.get("direction_pool_split") != "training_only":
            raise ValueError("steering corruption may only use a training_only pool")
        if section.get("steering_vectors_used") is not False:
            raise ValueError(
                "the denoiser's corrupting directions come from the training-only "
                "pool, never from DEV or held-out steering vectors"
            )
        declared = SteeringCorruptionSpec(
            version=str(section["spec_version"]),
            distribution=str(section["distribution"]),
            delta_max=float(section["delta_max"]),
            lambda_grid=tuple(float(value) for value in section["lambda_grid"]),
        )
        if declared != STEERING_CORRUPTION_SPEC:
            raise ValueError(
                "the steering corruption distribution is frozen in "
                "interp.steering_denoiser.STEERING_CORRUPTION_SPEC; a config may "
                "restate it but may not redefine it"
            )
        steering_corruption = SteeringCorruptionTrainingSpec(
            direction_pool=str(section["direction_pool"]), spec=declared
        )

    flow_objective = None
    if "flow_objective" in raw:
        objective = raw["flow_objective"]
        denoising = objective.get("type") == STEERING_DENOISER_OBJECTIVE
        if denoising and steering_corruption is None:
            raise ValueError(
                "the steering denoiser requires a steering_corruption section "
                "naming its training-only direction pool"
            )
        if denoising and conditioning is not None:
            raise ValueError(
                "the steering denoiser is unconditional; it must not declare a "
                "conditioning section"
            )
        if not denoising and conditioning is None:
            raise ValueError(
                "the tangent objective conditions on (direction, coordinate) and "
                "requires a conditioning section"
            )
        flow_objective = FlowObjectiveSpec(
            type=str(objective["type"]),
            output_projection=bool(objective["output_projection"]),
        )
    return FlowTrainingConfig(
        experiment_id=str(raw["experiment_id"]),
        experiment_class=str(raw["experiment_class"]),
        held_out=str(protected["held_out"]),
        dataset_name=str(data["dataset"]),
        dataset_repository=str(data["repository"]),
        dataset_config=str(data["repository_config"]),
        dataset_revision=str(data["repository_revision"]),
        tokenizer=str(data["tokenizer"]),
        per_seq=int(data["per_seq"]),
        val_fraction=float(data["val_fraction"]),
        split_seed=int(data["split_seed"]),
        split_fingerprint=str(data["split_fingerprint"]),
        norm_eps=float(normalization["eps"]),
        model=load_flow_config(model_config_path),
        training_seed=int(training["seed"]),
        noise_seed=int(training["noise_seed"]),
        steps=steps,
        batch_size=int(training["batch_size"]),
        lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
        warmup_steps=int(training["warmup_steps"]),
        grad_clip=float(training["grad_clip"]),
        eval_every=int(training["eval_every"]),
        eval_batches=int(validation["batches"]),
        eval_seed=int(validation["seed"]),
        t_bins=t_bins,
        selection_metric=str(validation["selection_metric"]),
        save_steps=save_steps,
        keep=keep,
        raw=raw,
        conditioning=conditioning,
        flow_objective=flow_objective,
        steering_corruption=steering_corruption,
    )


def _fetch(
    dataset: ActivationDataset, indices: np.ndarray, device: torch.device
) -> torch.Tensor:
    values = np.array(dataset.array[indices], dtype=np.float32, copy=True)
    return torch.from_numpy(values).to(device)


# Format 1 stored mean/std (as state-dict buffers) but not the normalizer eps, so
# every format-1 checkpoint implicitly used the ActivationNormalizer default. Every
# run that produced one configured normalization.eps = 1e-5, so reading them back
# with this constant reproduces the training-time standardization exactly.
LEGACY_NORMALIZER_EPS = 1e-5

CHECKPOINT_FORMAT_VERSION = 2


def normalizer_identity(normalizer: ActivationNormalizer) -> dict[str, object]:
    """Tamper-evident identity of the standardization a checkpoint was trained under."""

    digest = hashlib.sha256()
    for buffer in (normalizer.mean, normalizer.std):
        digest.update(buffer.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())
    digest.update(repr(float(normalizer.eps)).encode())
    return {
        "width": int(normalizer.mean.shape[0]),
        "eps": float(normalizer.eps),
        "digest": digest.hexdigest(),
    }


def save_flow_checkpoint(
    model: FlowMatcher | SteeringDenoiser,
    path: Path,
    *,
    metadata: dict,
    training_state: dict | None = None,
    flow_objective: str = ISOTROPIC_OBJECTIVE,
) -> None:
    """Save model, normalizer buffers and eps, architecture, metadata, and resume state."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    if flow_objective not in ALL_OBJECTIVES:
        raise ValueError(f"unknown flow objective {flow_objective!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    conditional = isinstance(model, ConditionalFlowMatcher)
    denoiser = isinstance(model, SteeringDenoiser)
    if (flow_objective == STEERING_DENOISER_OBJECTIVE) != denoiser:
        raise ValueError(
            "the steering-denoising objective and the SteeringDenoiser class must "
            "be used together; neither may carry the other's label"
        )
    if flow_objective in TANGENT_OBJECTIVES and not conditional:
        raise ValueError("the tangent objective requires a conditional flow model")
    kind = "conditional_flow" if conditional else "flow"
    if flow_objective in TANGENT_OBJECTIVES:
        kind = "tangent_conditional_flow"
    if denoiser:
        kind = "steering_denoiser"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "kind": kind,
            "flow_objective": flow_objective,
            "model_config": asdict(model.cfg),
            "condition_config": asdict(model.cond_cfg) if conditional else None,
            # Standardization is part of the model: eps belongs with mean and std.
            "normalizer_eps": float(model.normalizer.eps),
            "state_dict": model.state_dict(),
            "metadata": metadata,
            "training_state": training_state,
        },
        path,
    )


def checkpoint_objective(path: Path, device: torch.device | str = "cpu") -> str:
    """Read only which corruption geometry a checkpoint was trained on.

    Evaluators use this to pick their frozen plan from the checkpoint itself
    rather than from an operator flag, so a model can never be scored on a grid
    it was not trained for. Checkpoints written before the tangent branch carry
    no objective key and are read as ``isotropic``, which is what they are.
    """

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return str(checkpoint.get("flow_objective", ISOTROPIC_OBJECTIVE))


def load_flow_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
    *,
    expected_objective: str | None = None,
) -> tuple[FlowMatcher | SteeringDenoiser, dict, dict | None]:
    """Load either flow variant; format-1 checkpoints keep their implicit eps.

    ``expected_objective`` rejects a checkpoint trained on a different corruption
    geometry.  Checkpoints written before the tangent branch carry no objective
    key and are read as ``isotropic``, which is what they are.
    """

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    version = checkpoint.get("format_version")
    if version not in (1, CHECKPOINT_FORMAT_VERSION):
        raise ValueError("unsupported clean-flow checkpoint format")
    objective = str(checkpoint.get("flow_objective", ISOTROPIC_OBJECTIVE))
    if expected_objective is not None and objective != expected_objective:
        raise ValueError(
            f"checkpoint {path} was trained on the {objective!r} objective, "
            f"but {expected_objective!r} was required"
        )
    cfg = FlowModelConfig(**checkpoint["model_config"])
    width = cfg.activation_width
    eps = float(checkpoint.get("normalizer_eps", LEGACY_NORMALIZER_EPS))
    normalizer = ActivationNormalizer(torch.zeros(width), torch.ones(width), eps)
    condition_config = checkpoint.get("condition_config")
    if objective == STEERING_DENOISER_OBJECTIVE:
        model: FlowMatcher | SteeringDenoiser = SteeringDenoiser(cfg, normalizer)
    elif condition_config is None:
        model = FlowMatcher(cfg, normalizer)
    else:
        model = ConditionalFlowMatcher(
            cfg, ConditionEncoderConfig(**condition_config), normalizer
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint["metadata"], checkpoint.get("training_state")


def _sample_batch(
    model: FlowMatcher,
    h: torch.Tensor,
    *,
    pool: TrainingDirectionPool | None,
    generator: torch.Generator,
    objective: str = ISOTROPIC_OBJECTIVE,
    spec: SteeringCorruptionSpec = STEERING_CORRUPTION_SPEC,
):
    """One flow batch of the configured variant, from raw activations."""

    if objective == STEERING_DENOISER_OBJECTIVE:
        if pool is None:
            raise ValueError("the steering denoiser requires a training-only direction pool")
        return sample_steering_corruption_batch(
            h, normalizer=model.normalizer, pool=pool, generator=generator, spec=spec
        )
    if objective in TANGENT_OBJECTIVES:
        if pool is None:
            raise ValueError("the tangent objective requires a training-only direction pool")
        return sample_tangent_flow_batch(
            h,
            normalizer=model.normalizer,
            pool=pool,
            generator=generator,
            objective=objective,
        )
    if pool is None:
        return sample_flow_batch(model.normalizer.normalize(h), generator=generator)
    return sample_conditional_flow_batch(
        h, normalizer=model.normalizer, pool=pool, generator=generator
    )


def _predict(
    model: FlowMatcher,
    x_t: torch.Tensor,
    t: torch.Tensor,
    batch,
    *,
    output_projection: bool = True,
) -> torch.Tensor:
    """Evaluate the model, carrying the batch's condition when there is one.

    For a tangent batch the returned velocity is analytically projected into
    ``v_x``-perp, so the velocity entering the loss is exactly the one used at
    inference.  ``output_projection=False`` trains the unconstrained variant.
    """

    if isinstance(batch, SteeringDenoiseBatch):
        # The denoiser predicts a residual and is told nothing about the
        # corruption; there is no velocity here and nothing to project.
        return model(x_t)
    if isinstance(batch, TangentFlowBatch):
        raw = model(x_t, t, batch.v_x, batch.c_x)
        return tangent_project(raw, batch.v_x) if output_projection else raw
    if isinstance(batch, ConditionalFlowBatch):
        return model(x_t, t, batch.v_x, batch.c_x)
    return model(x_t, t)


def _bin_state(
    batch,
    x0: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
    *,
    normalizer: ActivationNormalizer | None = None,
    spec: SteeringCorruptionSpec = STEERING_CORRUPTION_SPEC,
):
    """State and target for one validation bin, in the batch's own geometry.

    For the flow objectives a bin is a slice of flow time. The steering denoiser
    has no time, so its bins are read as slices of **corruption magnitude**:
    ``|delta| = time * delta_max``, with the sign taken deterministically from
    the drawn noise. The resulting curve -- reconstruction error against how hard
    the activation was steered -- is the denoiser's analogue of loss by time bin,
    and is concept-independent in exactly the same way.
    """

    if isinstance(batch, SteeringDenoiseBatch):
        if normalizer is None:
            raise ValueError("denoiser validation bins need the model's normalizer")
        scale = (normalizer.std + normalizer.eps).to(device=x0.device, dtype=x0.dtype)
        sign = torch.where(noise[:, :1] >= 0.0, 1.0, -1.0)
        delta = sign * time * spec.delta_max
        # normalize(h + delta v) == x0 + delta * v / (std + eps)
        state = x0 + delta * (batch.v / scale)
        return state, x0 - state
    if isinstance(batch, TangentFlowBatch):
        _, state, target = tangent_path_states(
            x0, batch.v_x, batch.c_x, noise, time, objective=batch.objective
        )
        return state, target
    return (1.0 - time) * x0 + time * noise, noise - x0


@torch.no_grad()
def evaluate_flow(
    model: FlowMatcher,
    dataset: ActivationDataset,
    validation_indices: np.ndarray,
    cfg: FlowTrainingConfig,
    device: torch.device,
    pool: TrainingDirectionPool | None = None,
) -> dict:
    """Run deterministic concept-independent validation, including equal-coverage bins."""

    if len(validation_indices) == 0:
        raise ValueError("validation indices must be nonempty")
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(cfg.eval_seed)
    index_rng = np.random.default_rng(cfg.eval_seed)
    picks = index_rng.integers(
        0, len(validation_indices), size=(cfg.eval_batches, cfg.batch_size)
    )
    projection = cfg.flow_objective is None or cfg.flow_objective.output_projection
    losses: list[float] = []
    controls: list[float] = []
    cosines: list[float] = []
    raw_parallel: list[float] = []
    bin_pairs = tuple(pairwise(cfg.t_bins))
    by_bin: dict[str, list[float]] = {
        f"{left:.2f}-{right:.2f}": [] for left, right in bin_pairs
    }
    zero_by_bin: dict[str, list[float]] = {key: [] for key in by_bin}

    for batch_index in range(cfg.eval_batches):
        h = _fetch(dataset, validation_indices[picks[batch_index]], device)
        batch = _sample_batch(
            model, h, pool=pool, generator=generator, objective=cfg.objective_type
        )
        x0 = model.normalizer.normalize(h)
        prediction = _predict(model, batch.x_t, batch.t, batch, output_projection=projection)
        losses.append(float(flow_matching_loss(prediction, batch.target_velocity)))
        controls.append(float(batch.target_velocity.square().mean()))
        cosines.append(
            float(torch.cosine_similarity(prediction, batch.target_velocity, dim=-1).mean())
        )
        if isinstance(batch, TangentFlowBatch):
            # Diagnostic only: the parallel component the network still emits
            # before analytic projection removes it.
            unprojected = model(batch.x_t, batch.t, batch.v_x, batch.c_x)
            raw_parallel.append(float(coordinate(unprojected, batch.v_x).abs().mean()))
        for (left, right), key in zip(bin_pairs, by_bin, strict=True):
            time = left + (right - left) * torch.rand(
                (x0.shape[0], 1), generator=generator, device=device, dtype=x0.dtype
            )
            noise = torch.randn(
                x0.shape, generator=generator, device=device, dtype=x0.dtype
            )
            state, target = _bin_state(
                batch, x0, noise, time, normalizer=model.normalizer
            )
            by_bin[key].append(
                float(
                    flow_matching_loss(
                        _predict(model, state, time, batch, output_projection=projection),
                        target,
                    )
                )
            )
            zero_by_bin[key].append(float(target.square().mean()))
    model.train(was_training)
    report = {
        "val_flow_mse": float(np.mean(losses)),
        "zero_predictor_mse": float(np.mean(controls)),
        "val_cosine_velocity": float(np.mean(cosines)),
        "val_flow_mse_by_bin": {key: float(np.mean(values)) for key, values in by_bin.items()},
        "zero_predictor_mse_by_bin": {
            key: float(np.mean(values)) for key, values in zero_by_bin.items()
        },
    }
    if raw_parallel:
        report["val_raw_parallel_velocity_mean"] = float(np.mean(raw_parallel))
    return report


def _learning_rate_multiplier(step: int, cfg: FlowTrainingConfig) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.steps - cfg.warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _training_state(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    flow_generator: torch.Generator,
    batch_rng: np.random.Generator,
    history: list[dict],
    best: float,
    best_checkpoint: str | None,
) -> dict:
    return {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "flow_generator_state": flow_generator.get_state(),
        "batch_rng_state": batch_rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "history": history,
        "best": best,
        "best_checkpoint": best_checkpoint,
    }


def _used_config(cfg: FlowTrainingConfig) -> dict:
    values = asdict(cfg)
    values.pop("raw")
    # Unconditional runs keep the fingerprint they had before conditioning existed,
    # and isotropic runs the one they had before the tangent objective existed, so
    # historical config fingerprints stay comparable.
    if values.get("conditioning") is None:
        values.pop("conditioning", None)
    if values.get("flow_objective") is None:
        values.pop("flow_objective", None)
    # Likewise: runs that do not corrupt by steering keep the fingerprint they had
    # before post-stop experiment B existed.
    if values.get("steering_corruption") is None:
        values.pop("steering_corruption", None)
    return values


def _config_fingerprint(used_config: dict) -> str:
    encoded = json.dumps(used_config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validated_dataset_identity(dataset: ActivationDataset, cfg: FlowTrainingConfig) -> dict:
    report = dataset.meta.get("full_validation_report")
    if not isinstance(report, dict):
        raise ValueError("training requires the attached full activation validation report")
    expected_report = {
        "status": "VALID",
        "name": cfg.dataset_name,
        "split_fingerprint": cfg.split_fingerprint,
        "token_cache_sha256": dataset.meta["token_cache_sha256"],
    }
    for field, expected in expected_report.items():
        if report.get(field) != expected:
            raise ValueError(
                f"activation validation report {field}={report.get(field)!r} "
                f"!= expected {expected!r}"
            )
    artifact_hashes = report.get("sha256")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != {
        "array",
        "metadata",
        "statistics",
    }:
        raise ValueError("activation validation report lacks the exact artifact SHA set")
    if any(not isinstance(value, str) or len(value) != 64 for value in artifact_hashes.values()):
        raise ValueError("activation validation report contains an invalid SHA-256 digest")
    return {
        "sha256": dict(artifact_hashes),
        "token_cache_sha256": report["token_cache_sha256"],
    }


def _write_new_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite run metadata: {path}")
    path.write_text(json.dumps(value, indent=2) + "\n")


def _build_model(
    cfg: FlowTrainingConfig, normalizer: ActivationNormalizer
) -> FlowMatcher | SteeringDenoiser:
    if cfg.objective_type == STEERING_DENOISER_OBJECTIVE:
        return SteeringDenoiser(cfg.model, normalizer)
    if cfg.conditioning is None:
        return FlowMatcher(cfg.model, normalizer)
    return ConditionalFlowMatcher(cfg.model, cfg.conditioning.condition, normalizer)


def _environment(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def train_flow(
    dataset: ActivationDataset,
    cfg: FlowTrainingConfig,
    run_dir: Path,
    *,
    device: torch.device | str,
    progress: bool = True,
    resume_checkpoint: Path | None = None,
) -> dict:
    """Train one immutable run and select checkpoints by validation flow MSE only."""

    if cfg.status != "authorized":
        raise PermissionError(
            f"config {cfg.experiment_id} has status {cfg.status!r}; a training run "
            "requires an explicitly human-authorized config"
        )
    if resume_checkpoint is None and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory {run_dir} already contains artifacts")
    if resume_checkpoint is not None:
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_checkpoint}")
        if resume_checkpoint.resolve().parent != run_dir.resolve():
            raise ValueError("resume checkpoint must belong to the requested run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    split = make_split(len(dataset), cfg.per_seq, cfg.val_fraction, cfg.split_seed)
    if split.fingerprint() != cfg.split_fingerprint:
        raise ValueError(
            f"split fingerprint {split.fingerprint()} != configured {cfg.split_fingerprint}"
        )
    validate_activation_metadata(
        dataset,
        expected_name=cfg.dataset_name,
        expected_split="train",
        expected_model=MODEL_NAME,
        expected_resolved_model_name=MODEL_RESOLVED_NAME,
        expected_model_revision=MODEL_REVISION,
        expected_hook=STEERING_HOOK,
        expected_ctx=cfg.per_seq + 1,
        expected_d_model=cfg.model.activation_width,
        expected_dataset_repository=cfg.dataset_repository,
        expected_dataset_config=cfg.dataset_config,
        expected_dataset_revision=cfg.dataset_revision,
        expected_tokenizer=cfg.tokenizer,
    )
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")

    mean, std = split_stats(dataset, split.train)
    expected_normalizer = ActivationNormalizer(
        torch.from_numpy(mean), torch.from_numpy(std), cfg.norm_eps
    )
    dataset_artifact_identity = _validated_dataset_identity(dataset, cfg)
    pool = None
    pool_identity = None
    pool_source = cfg.direction_pool_path
    if pool_source is not None:
        # The config stores a repo-relative manifest path, so the recorded config
        # fingerprint stays machine-independent.
        manifest = Path(pool_source)
        if not manifest.is_absolute():
            manifest = Path(__file__).resolve().parents[2] / manifest
        pool = load_training_direction_pool(manifest).to(
            device=resolved_device, dtype=torch.float32
        )
        if pool.directions.shape[1] != cfg.model.activation_width:
            raise ValueError(
                f"direction pool width {pool.directions.shape[1]} != activation width "
                f"{cfg.model.activation_width}"
            )
        pool_identity = pool.identity()
    if cfg.flow_objective is not None and pool is None:
        raise ValueError(
            f"the {cfg.objective_type} objective requires a training-only direction pool"
        )
    revision = source_revision()
    environment = _environment(resolved_device)
    used_config = _used_config(cfg)
    config_fingerprint = _config_fingerprint(used_config)
    objective_identity = {
        "flow_objective": cfg.objective_type,
        "condition_type": None if cfg.conditioning is None else cfg.conditioning.condition.kind,
        "steering_corruption": (
            None if cfg.steering_corruption is None else asdict(cfg.steering_corruption.spec)
        ),
        "tangent_output_projection": (
            None if cfg.flow_objective is None else cfg.flow_objective.output_projection
        ),
        "normalizer": normalizer_identity(expected_normalizer),
    }

    if resume_checkpoint is None:
        torch.manual_seed(cfg.training_seed)
        if resolved_device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.training_seed)
        model = _build_model(cfg, expected_normalizer).to(resolved_device).float().train()
        history: list[dict] = []
        best = float("inf")
        best_checkpoint: str | None = None
        start_step = 0
        checkpoint_state = None
    else:
        model, checkpoint_meta, checkpoint_state = load_flow_checkpoint(
            resume_checkpoint, resolved_device, expected_objective=cfg.objective_type
        )
        if checkpoint_state is None:
            raise ValueError("resume checkpoint has no training state")
        recorded_objective = checkpoint_meta.get("objective_identity")
        if recorded_objective is None:
            # Checkpoints written before the tangent branch record no objective
            # identity. They are isotropic by construction, and their normalizer
            # is verified against the train-split statistics below.
            if cfg.objective_type != ISOTROPIC_OBJECTIVE:
                raise ValueError(
                    f"resume checkpoint predates flow objectives and is isotropic, "
                    f"but the config requests {cfg.objective_type!r}"
                )
        elif recorded_objective != objective_identity:
            raise ValueError(
                f"resume checkpoint objective identity {recorded_objective!r} "
                f"!= expected {objective_identity!r}"
            )
        required_identity = {
            "experiment_id": cfg.experiment_id,
            "dataset": dataset.meta["name"],
            "split_fingerprint": split.fingerprint(),
            "source_revision": revision,
            "config_fingerprint": config_fingerprint,
            "dataset_artifact_identity": dataset_artifact_identity,
            "direction_pool": pool_identity,
        }
        for field, expected in required_identity.items():
            if checkpoint_meta.get(field) != expected:
                raise ValueError(
                    f"resume checkpoint {field}={checkpoint_meta.get(field)!r} "
                    f"!= expected {expected!r}"
                )
        if model.cfg != cfg.model:
            raise ValueError("resume checkpoint architecture differs from the config")
        if not torch.equal(
            model.normalizer.mean.cpu(), expected_normalizer.mean
        ) or not torch.equal(model.normalizer.std.cpu(), expected_normalizer.std):
            raise ValueError("resume checkpoint normalization differs from train-split stats")
        start_step = int(checkpoint_meta["step"])
        if not 0 < start_step < cfg.steps:
            raise ValueError(f"resume step {start_step} is outside (0, {cfg.steps})")
        history = list(checkpoint_state["history"])
        if not history or int(history[-1]["step"]) != start_step:
            raise ValueError("resume history does not end at the checkpoint step")
        best = float(checkpoint_state["best"])
        best_checkpoint = checkpoint_state["best_checkpoint"]
        model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _learning_rate_multiplier(step, cfg)
    )
    flow_generator = torch.Generator(device=resolved_device).manual_seed(cfg.noise_seed)
    batch_rng = np.random.default_rng(cfg.training_seed)
    page_cache = PageCacheGovernor()
    if checkpoint_state is not None:
        optimizer.load_state_dict(checkpoint_state["optimizer"])
        scheduler.load_state_dict(checkpoint_state["scheduler"])
        flow_generator.set_state(checkpoint_state["flow_generator_state"].cpu())
        batch_rng.bit_generator.state = checkpoint_state["batch_rng_state"]
        torch.set_rng_state(checkpoint_state["torch_rng_state"].cpu())
        cuda_rng_state = checkpoint_state.get("cuda_rng_state_all")
        if resolved_device.type == "cuda" and cuda_rng_state is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state])
    running = {
        "experiment_id": cfg.experiment_id,
        "status": "RUNNING",
        "source_revision": revision,
        "config_fingerprint": config_fingerprint,
        "dataset_artifact_identity": dataset_artifact_identity,
        "direction_pool": pool_identity,
        "objective_identity": objective_identity,
        "dataset": dataset.meta,
        "steps": cfg.steps,
        "used_config": used_config,
        "environment": environment,
        "held_out_accessed": False,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
    }
    event = (
        "status_RUNNING.json"
        if resume_checkpoint is None
        else f"status_RESUMED_from_{start_step:06d}_{uuid4().hex}.json"
    )
    _write_new_json(run_dir / event, running)

    last_completed_step = start_step
    try:
        iterator = tqdm(
            range(start_step, cfg.steps), desc=cfg.experiment_id, disable=not progress
        )
        for step_index in iterator:
            selected = split.train[
                batch_rng.integers(0, len(split.train), size=cfg.batch_size)
            ]
            h = _fetch(dataset, selected, resolved_device)
            # Soft page-cache cap. Reads nothing and reorders nothing; only drops clean
            # mmap pages once the cgroup charge approaches its limit, so a 45.8 GiB
            # artifact can be sampled for the whole run without the container dying.
            page_cache.step(dataset.array)
            batch = _sample_batch(
                model, h, pool=pool, generator=flow_generator, objective=cfg.objective_type
            )
            loss = flow_matching_loss(
                _predict(
                    model,
                    batch.x_t,
                    batch.t,
                    batch,
                    output_projection=(
                        cfg.flow_objective is None or cfg.flow_objective.output_projection
                    ),
                ),
                batch.target_velocity,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {step_index + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {step_index + 1}")
            optimizer.step()
            scheduler.step()

            step = step_index + 1
            last_completed_step = step
            if step % cfg.eval_every == 0 or step == cfg.steps:
                evaluation = evaluate_flow(
                    model, dataset, split.val, cfg, resolved_device, pool
                )
                row = {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                    "lr": float(scheduler.get_last_lr()[0]),
                    **evaluation,
                }
                history.append(row)
                iterator.set_postfix(
                    train=f"{row['train_loss']:.4f}", val=f"{row['val_flow_mse']:.4f}"
                )
                checkpoint_meta = {
                    "experiment_id": cfg.experiment_id,
                    "dataset": dataset.meta["name"],
                    "split_fingerprint": split.fingerprint(),
                    "source_revision": revision,
                    "config_fingerprint": config_fingerprint,
                    "dataset_artifact_identity": dataset_artifact_identity,
                    "direction_pool": pool_identity,
                    "objective_identity": objective_identity,
                    **row,
                }
                if row[cfg.selection_metric] < best:
                    best = row[cfg.selection_metric]
                    superseded = best_checkpoint
                    best_checkpoint = f"best_step_{step:06d}.pt"
                    state = _training_state(
                        optimizer,
                        scheduler,
                        flow_generator,
                        batch_rng,
                        history,
                        best,
                        best_checkpoint,
                    )
                    save_flow_checkpoint(
                        model,
                        run_dir / best_checkpoint,
                        metadata=checkpoint_meta,
                        training_state=state,
                        flow_objective=cfg.objective_type,
                    )
                    # keep == [best, last, configured_steps]: the new best is on disk
                    # before the superseded one is removed, so a best checkpoint always
                    # exists. Configured step checkpoints use a different filename and
                    # are never touched here.
                    if superseded is not None and superseded != best_checkpoint:
                        (run_dir / superseded).unlink(missing_ok=True)
                if step in cfg.save_steps:
                    state = _training_state(
                        optimizer,
                        scheduler,
                        flow_generator,
                        batch_rng,
                        history,
                        best,
                        best_checkpoint,
                    )
                    save_flow_checkpoint(
                        model,
                        run_dir / f"step_{step:06d}.pt",
                        metadata=checkpoint_meta,
                        training_state=state,
                        flow_objective=cfg.objective_type,
                    )

        metadata = {
            "experiment_id": cfg.experiment_id,
            "experiment_class": cfg.experiment_class,
            "status": "complete",
            "source_revision": revision,
            "config_fingerprint": config_fingerprint,
            "direction_pool": pool_identity,
            "objective_identity": objective_identity,
            "dataset_artifact_identity": dataset_artifact_identity,
            "used_config": used_config,
            "environment": environment,
            "dataset": dataset.meta["name"],
            "dataset_provenance": dataset.meta,
            "steps": cfg.steps,
            "batch_size": cfg.batch_size,
            "training_seed": cfg.training_seed,
            "noise_seed": cfg.noise_seed,
            "split_seed": cfg.split_seed,
            "split_fingerprint": split.fingerprint(),
            "n_train_tokens": int(len(split.train)),
            "n_val_tokens": int(len(split.val)),
            "selection_metric": cfg.selection_metric,
            "best_val_flow_mse": best,
            "best_checkpoint": best_checkpoint,
            "held_out_accessed": False,
            "history": history,
        }
        final_checkpoint_meta = {
            "experiment_id": cfg.experiment_id,
            "dataset": dataset.meta["name"],
            "split_fingerprint": split.fingerprint(),
            "source_revision": revision,
            "config_fingerprint": config_fingerprint,
            "dataset_artifact_identity": dataset_artifact_identity,
            "direction_pool": pool_identity,
            "objective_identity": objective_identity,
            "step": cfg.steps,
            **(history[-1] if history else {}),
        }
        save_flow_checkpoint(
            model,
            run_dir / "last.pt",
            metadata=final_checkpoint_meta,
            flow_objective=cfg.objective_type,
            training_state=_training_state(
                optimizer,
                scheduler,
                flow_generator,
                batch_rng,
                history,
                best,
                best_checkpoint,
            ),
        )
        _write_new_json(run_dir / "meta.json", metadata)
        if best_checkpoint is None:
            raise RuntimeError("training completed without selecting a best checkpoint")
        _write_new_json(
            run_dir / "best.json",
            {
                "selection_metric": cfg.selection_metric,
                "selection_mode": "min",
                "value": best,
                "checkpoint": best_checkpoint,
            },
        )
        _write_new_json(run_dir / "status_complete.json", {**running, "status": "complete"})
        return metadata
    except KeyboardInterrupt as error:
        failure = {**running, "status": "INTERRUPTED", "error": repr(error)}
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        _write_new_json(
            run_dir / f"status_INTERRUPTED_after_{last_completed_step:06d}_{stamp}.json",
            failure,
        )
        raise
    except BaseException as error:
        failure = {**running, "status": "INVALID", "error": repr(error)}
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        _write_new_json(
            run_dir / f"status_INVALID_after_{last_completed_step:06d}_{stamp}.json",
            failure,
        )
        raise
