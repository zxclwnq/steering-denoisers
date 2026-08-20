"""Fail-closed manifest and provenance boundary for the clean Phase B evaluator."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformer_lens.utilities.tokenize_utils import get_attention_mask

from .activations import file_sha256
from .phase_b import APPROVED_PHASE_B_CONFIG_SHA256, PhaseBConfig, load_phase_b_config

# Every frozen evaluator release. The DEV design (vectors, prompts, seeds, alpha
# grid, arms, noise identity, statistics, matching, metrics, baselines) is byte
# identical across entries; only the flow prior under test and its output root
# differ. A new entry is the only permitted way to evaluate a different prior.
APPROVED_EVALUATOR_CONFIG_SHA256S: dict[str, str] = {
    "clean_flow_phase_b_dev_v1": (
        "960fe0b2596ed72cb50a246b2f93ad8b985c2c0ea4ef0fb234b3581dda5f08c2"
    ),
    "clean_flow_phase_b_dev_wide60m_v1": (
        "83bcc7344cfdded76a24965f6f7d0ae329ab2b58b32a90fd1e250ca910520dbd"
    ),
    # Capacity control for the wide release: same corpus, unique data, budget, and
    # activation statistics, narrow architecture. Authorized as a follow-up after
    # the wide result; it does not reselect or modify that release.
    "clean_flow_phase_b_dev_narrow16m_fw32m_v1": (
        "59ce46d033e9b310398e67db77c056b54554f3127ddb8e3a600691d0cacbfbc5"
    ),
}
APPROVED_EVALUATOR_CONFIG_SHA256 = APPROVED_EVALUATOR_CONFIG_SHA256S["clean_flow_phase_b_dev_v1"]
_SERVER_OUTPUT_ROOTS = {
    "clean_flow_phase_b_dev_v1": "/workspace/results/clean_flow_phase_b_dev_v1",
    "clean_flow_phase_b_dev_wide60m_v1": ("/workspace/results/clean_flow_phase_b_dev_wide60m_v1"),
    "clean_flow_phase_b_dev_narrow16m_fw32m_v1": (
        "/workspace/results/clean_flow_phase_b_dev_narrow16m_fw32m_v1"
    ),
}
_NARROW_CHECKPOINT_PATH = "/workspace/checkpoints/clean_flow_100k_v1_a439b2d7/best_step_099500.pt"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_METRIC_VERSIONS = {
    "text": "clean_text_metrics_v1",
    "nll": "clean_gpt2_conditional_nll_v1",
    "sae": "clean_frozen_sae_continuation_mean_v1",
    "geometry": "clean_flow_geometry_v2",
}
_ARM_IDENTITY = (
    ("flow_t010_nfe1", 0.10, 1),
    ("flow_t010_nfe3", 0.10, 3),
    ("flow_t010_nfe5", 0.10, 5),
    ("flow_t025_nfe1", 0.25, 1),
    ("flow_t025_nfe3", 0.25, 3),
    ("flow_t025_nfe5", 0.25, 5),
    ("flow_t050_nfe1", 0.50, 1),
    ("flow_t050_nfe3", 0.50, 3),
    ("flow_t050_nfe5", 0.50, 5),
)


@dataclass(frozen=True)
class BaselineArtifactSpec:
    method: str
    raw_filename: str
    raw_sha256: str
    meta_filename: str
    meta_sha256: str
    source_method: str
    denoiser_checkpoint_sha256: str | None = None
    kappa: float | None = None


@dataclass(frozen=True)
class FlowArm:
    arm_id: str
    t_start: float
    nfe: int


@dataclass(frozen=True)
class FlowPriorSpec:
    """Identity of a flow prior selected outside the original Phase-A gate.

    Present only for evaluator releases whose checkpoint does not come from
    ``configs/flow_phase_a_100k_v1.yaml``. It binds the checkpoint to the frozen
    concept-independent selection that produced it, so a Phase-B rerun can never
    silently adopt a prior chosen by steering evidence.
    """

    provenance_kind: str
    selection_protocol: str
    selection_protocol_sha256: str
    selection_report_sha256: str
    arm_report_sha256: str
    selected_arm: str
    parameters: int
    activation_width: int
    architecture_config: str
    architecture_config_sha256: str
    training_config: str
    training_config_sha256: str
    training_dataset: str
    training_statistics_sha256: str
    run_meta_path: Path
    run_meta_sha256: str
    best_pointer_path: Path
    best_pointer_sha256: str
    history_entries: int
    training_experiment_id: str
    comparison_release_id: str


@dataclass(frozen=True, order=True)
class CellIdentity:
    vector: str
    feature: int
    alpha: float
    alpha_hex: str
    alpha_hat: float
    is_stress: bool
    prompt_id: int
    generation_seed: int


@dataclass(frozen=True)
class BaselineContinuation:
    vector: str
    feature: int
    split: str
    alpha: float
    alpha_hat: float
    is_stress: bool
    prompt_id: int
    generation_seed: int
    continuation: str


@dataclass(frozen=True)
class EvaluatorConfig:
    experiment_id: str
    phase_b: PhaseBConfig
    parent_protocol_sha256: str
    flow_checkpoint_path: Path
    flow_checkpoint_sha256: str
    flow_checkpoint_step: int
    flow_prior: FlowPriorSpec | None
    baselines: tuple[BaselineArtifactSpec, ...]
    metric_versions: dict[str, str]
    control_features: dict[str, tuple[int, ...]]
    arms: tuple[FlowArm, ...]
    schema_version: str
    output_layout_version: str
    server_output_root: Path
    bootstrap_seed: int
    bootstrap_resamples: int
    bootstrap_confidence: float
    repetition_threshold: float
    raw: dict


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a full SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value


def _load_baselines(raw: object) -> tuple[BaselineArtifactSpec, ...]:
    if not isinstance(raw, list):
        raise ValueError("baselines must be a list")
    out = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("baseline entry must be a mapping")
        out.append(
            BaselineArtifactSpec(
                method=str(item["method"]),
                raw_filename=str(item["raw_filename"]),
                raw_sha256=_require_sha(item["raw_sha256"], "baseline raw SHA"),
                meta_filename=str(item["meta_filename"]),
                meta_sha256=_require_sha(item["meta_sha256"], "baseline meta SHA"),
                source_method=str(item["source_method"]),
                denoiser_checkpoint_sha256=(
                    _require_sha(item["denoiser_checkpoint_sha256"], "denoiser SHA")
                    if "denoiser_checkpoint_sha256" in item
                    else None
                ),
                kappa=float(item["kappa"]) if "kappa" in item else None,
            )
        )
    if tuple(item.method for item in out) != ("additive", "naive", "shrinkage_k080"):
        raise ValueError("baseline method set or order changed")
    if tuple(item.source_method for item in out) != ("additive", "naive", "shrinkage"):
        raise ValueError("baseline source method identity changed")
    if out[2].kappa != 0.8 or out[0].kappa is not None or out[1].kappa is not None:
        raise ValueError("frozen shrinkage control changed")
    return tuple(out)


def _load_flow_prior(raw: object, experiment_id: str) -> FlowPriorSpec:
    if not isinstance(raw, dict):
        raise ValueError("flow_prior must be a mapping")
    # Either the arm the frozen 2x2 rule selected, or a sibling arm from the same
    # frozen 2x2 run kept as a controlled contrast. Both are concept-independent
    # provenance; neither may be chosen by steering evidence, which the next check
    # enforces. No other origin is accepted.
    if raw.get("provenance_kind") not in (
        "concept_independent_scaling_selection",
        "concept_independent_scaling_capacity_control",
    ):
        raise ValueError(
            "a non-Phase-A flow prior must come from a concept-independent scaling selection"
        )
    if raw.get("selected_by_steering_evidence") is not False:
        raise ValueError("flow_prior must state that steering evidence did not select it")
    spec = FlowPriorSpec(
        provenance_kind=str(raw["provenance_kind"]),
        selection_protocol=str(raw["selection_protocol"]),
        selection_protocol_sha256=_require_sha(
            raw["selection_protocol_sha256"], "scaling protocol SHA"
        ),
        selection_report_sha256=_require_sha(
            raw["selection_report_sha256"], "scaling selection report SHA"
        ),
        arm_report_sha256=_require_sha(raw["arm_report_sha256"], "scaling arm report SHA"),
        selected_arm=str(raw["selected_arm"]),
        parameters=int(raw["parameters"]),
        activation_width=int(raw["activation_width"]),
        architecture_config=str(raw["architecture_config"]),
        architecture_config_sha256=_require_sha(
            raw["architecture_config_sha256"], "flow architecture config SHA"
        ),
        training_config=str(raw["training_config"]),
        training_config_sha256=_require_sha(
            raw["training_config_sha256"], "flow training config SHA"
        ),
        training_dataset=str(raw["training_dataset"]),
        training_statistics_sha256=_require_sha(
            raw["training_statistics_sha256"], "activation statistics SHA"
        ),
        run_meta_path=Path(str(raw["run_meta_path"])),
        run_meta_sha256=_require_sha(raw["run_meta_sha256"], "run metadata SHA"),
        best_pointer_path=Path(str(raw["best_pointer_path"])),
        best_pointer_sha256=_require_sha(raw["best_pointer_sha256"], "best pointer SHA"),
        history_entries=int(raw["history_entries"]),
        training_experiment_id=str(raw["training_experiment_id"]),
        comparison_release_id=str(raw["comparison_release_id"]),
    )
    if spec.activation_width != 768:
        raise ValueError("the flow prior must keep the frozen 768-wide activation boundary")
    if spec.parameters > 60_407_808:
        raise ValueError("the flow prior exceeds the frozen cheapness capacity bound")
    if not spec.comparison_release_id.startswith("clean_flow_phase_b_dev_v1_"):
        raise ValueError("a rerun must name the original narrow Phase-B release as its comparison")
    if experiment_id == "clean_flow_phase_b_dev_v1":
        raise ValueError("the original narrow release cannot declare a substitute flow prior")
    return spec


def load_evaluator_config(path: Path) -> EvaluatorConfig:
    """Load only an approved full-DEV evaluator release."""

    observed = file_sha256(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("evaluator config must contain a mapping")
    experiment_id = raw.get("experiment_id")
    approved = APPROVED_EVALUATOR_CONFIG_SHA256S.get(str(experiment_id))
    if approved is None:
        raise ValueError(
            f"unknown evaluator release {experiment_id!r}; approved: "
            f"{sorted(APPROVED_EVALUATOR_CONFIG_SHA256S)}"
        )
    if observed != approved:
        raise ValueError(f"evaluator config SHA-256 {observed} != approved {approved}")
    if (
        raw.get("version") != 1
        or raw.get("status") != "frozen_before_full_dev_generation"
        or raw.get("experiment_class") != "dev_method_development"
        or raw.get("split") != "dev"
        or raw.get("held_out_access") != "forbidden"
    ):
        raise ValueError("evaluator must remain the frozen DEV-only experiment")

    parent = raw["parent_protocol"]
    if parent != {
        "path": "configs/flow_phase_b_dev_v1.yaml",
        "sha256": APPROVED_PHASE_B_CONFIG_SHA256,
    }:
        raise ValueError("parent Phase B protocol identity changed")
    phase_b = load_phase_b_config(_PROJECT_ROOT / parent["path"])

    flow = raw["flow_checkpoint"]
    flow_prior = (
        _load_flow_prior(raw["flow_prior"], str(experiment_id)) if "flow_prior" in raw else None
    )
    if flow_prior is None:
        if (
            flow.get("path") != _NARROW_CHECKPOINT_PATH
            or int(flow.get("step")) != phase_b.checkpoint_step
            or flow.get("sha256") != phase_b.checkpoint_sha256
        ):
            raise ValueError("flow checkpoint differs from the frozen Phase A selection")
    elif (
        flow.get("path") == _NARROW_CHECKPOINT_PATH
        or flow.get("sha256") == phase_b.checkpoint_sha256
        or int(flow.get("step")) == phase_b.checkpoint_step
    ):
        raise ValueError(
            "a substitute flow prior must not reuse the narrow Phase-A checkpoint identity"
        )

    metric_versions = raw["metric_versions"]
    if metric_versions != _METRIC_VERSIONS:
        raise ValueError("clean evaluator metric versions changed")

    controls_raw = raw["control_features"]
    if not isinstance(controls_raw, dict):
        raise ValueError("control_features must be a mapping")
    controls = {
        str(name): tuple(int(item) for item in values) for name, values in controls_raw.items()
    }
    dev_features = {vector.feature for vector in phase_b.vectors}
    if set(controls) != {vector.name for vector in phase_b.vectors}:
        raise ValueError("control feature vectors differ from the frozen DEV set")
    for name, values in controls.items():
        if (
            len(values) != 32
            or len(set(values)) != 32
            or min(values) < 0
            or max(values) >= phase_b.sae.d_sae
            or set(values) & dev_features
        ):
            raise ValueError(f"invalid frozen SAE controls for {name}")

    arms = tuple(
        FlowArm(str(item["arm_id"]), float(item["t_start"]), int(item["nfe"]))
        for item in raw["arms"]
    )
    if tuple((item.arm_id, item.t_start, item.nfe) for item in arms) != _ARM_IDENTITY:
        raise ValueError("frozen nine-arm flow matrix changed")

    noise = raw["noise_identity"]
    if noise != {
        "namespace": "flow_noise_v2",
        "position_scheme": "prompt_relative_v1",
        "fields": [
            "vector",
            "exact_alpha_hex",
            "prompt_id",
            "generation_seed",
            "token_position",
        ],
        "independent_of": [
            "t_start",
            "nfe",
            "arm_id",
            "batch_order",
            "padding_length",
            "dataframe_row_order",
        ],
        "padding_has_identity": False,
    }:
        raise ValueError("matched flow-noise contract changed")

    statistics = raw["statistics"]
    if statistics != {
        "bootstrap_seed": 20_260_813,
        "bootstrap_resamples": 10_000,
        "bootstrap_confidence": 0.95,
        "bootstrap_unit": "vector_mean",
        "shared_resample_matrix": True,
        "repetition_threshold": 0.027785714285713504,
        "require_vector_signs": True,
        "require_leave_one_vector_out": True,
    }:
        raise ValueError("frozen vector-aware statistical contract changed")
    if raw["matching"] != {
        "identity": ["vector", "prompt_id", "generation_seed"],
        "coordinate": "realised_projection",
        "interpolation": "linear_bracketing",
        "extrapolation": "forbidden",
        "clipping": "forbidden",
        "record_brackets_and_weight": True,
    }:
        raise ValueError("scalar-attenuation matching contract changed")
    if (
        raw.get("schema_version") != "clean_phase_b_row_v1"
        or raw.get("output_layout_version") != "clean_phase_b_outputs_v1"
        or raw.get("server_output_root") != _SERVER_OUTPUT_ROOTS[str(experiment_id)]
        or raw.get("release_id_rule") != "experiment_id_plus_full_source_snapshot"
        or raw.get("immutable_outputs") is not True
    ):
        raise ValueError("row schema or immutable output layout changed")

    return EvaluatorConfig(
        experiment_id=raw["experiment_id"],
        phase_b=phase_b,
        parent_protocol_sha256=parent["sha256"],
        flow_checkpoint_path=Path(flow["path"]),
        flow_checkpoint_sha256=_require_sha(flow["sha256"], "flow checkpoint SHA"),
        flow_checkpoint_step=int(flow["step"]),
        flow_prior=flow_prior,
        baselines=_load_baselines(raw["baselines"]),
        metric_versions=dict(metric_versions),
        control_features=controls,
        arms=arms,
        schema_version=raw["schema_version"],
        output_layout_version=raw["output_layout_version"],
        server_output_root=Path(raw["server_output_root"]),
        bootstrap_seed=int(statistics["bootstrap_seed"]),
        bootstrap_resamples=int(statistics["bootstrap_resamples"]),
        bootstrap_confidence=float(statistics["bootstrap_confidence"]),
        repetition_threshold=float(statistics["repetition_threshold"]),
        raw=raw,
    )


def expected_cells(config: EvaluatorConfig) -> tuple[CellIdentity, ...]:
    """Enumerate the exact baseline/arm continuation cells in canonical order."""

    alpha_grid = tuple((value, False) for value in config.phase_b.alpha_hat) + tuple(
        (value, True) for value in config.phase_b.alpha_hat_stress
    )
    return tuple(
        CellIdentity(
            vector=vector.name,
            feature=vector.feature,
            alpha=alpha_hat * config.phase_b.activation_norm_mean,
            alpha_hex=(alpha_hat * config.phase_b.activation_norm_mean).hex(),
            alpha_hat=alpha_hat,
            is_stress=is_stress,
            prompt_id=prompt_id,
            generation_seed=seed,
        )
        for vector in config.phase_b.vectors
        for alpha_hat, is_stress in alpha_grid
        for seed in config.phase_b.generation_seeds
        for prompt_id in range(len(config.phase_b.prompts))
    )


def parse_baseline_row(row: object, expected: CellIdentity) -> BaselineContinuation:
    """Copy only frozen cell identity and text from an untrusted legacy row."""

    if not isinstance(row, dict):
        raise ValueError("baseline row must be a JSON object")
    if row.get("vector") != expected.vector:
        raise ValueError("baseline vector identity changed")
    if row.get("feature") != expected.feature:
        raise ValueError("baseline feature identity changed")
    if row.get("split") != "dev":
        raise ValueError("baseline row must be DEV only")
    alpha = row.get("alpha")
    if (
        not isinstance(alpha, (int, float))
        or isinstance(alpha, bool)
        or not math.isfinite(float(alpha))
        or float(alpha).hex() != expected.alpha_hex
    ):
        raise ValueError("baseline alpha identity changed")
    alpha_hat = row.get("alpha_hat")
    if (
        not isinstance(alpha_hat, (int, float))
        or isinstance(alpha_hat, bool)
        or not math.isfinite(float(alpha_hat))
        or float(alpha_hat) != expected.alpha_hat
    ):
        raise ValueError("baseline alpha_hat identity changed")
    if row.get("is_stress") is not expected.is_stress:
        raise ValueError("baseline stress identity changed")
    if row.get("prompt_idx") != expected.prompt_id:
        raise ValueError("baseline prompt identity changed")
    if row.get("seed") != expected.generation_seed:
        raise ValueError("baseline seed identity changed")
    continuation = row.get("continuation")
    if not isinstance(continuation, str):
        raise ValueError("baseline continuation must be text")
    return BaselineContinuation(
        vector=expected.vector,
        feature=expected.feature,
        split="dev",
        alpha=expected.alpha,
        alpha_hat=expected.alpha_hat,
        is_stress=expected.is_stress,
        prompt_id=expected.prompt_id,
        generation_seed=expected.generation_seed,
        continuation=continuation,
    )


def _baseline_key(row: object) -> tuple[str, str, int, int]:
    if not isinstance(row, dict):
        raise ValueError("baseline row must be a JSON object")
    alpha = row.get("alpha")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise ValueError("baseline alpha must be numeric")
    return (
        str(row.get("vector")),
        float(alpha).hex(),
        int(row.get("prompt_idx")),
        int(row.get("seed")),
    )


def validate_and_import_baseline(
    spec: BaselineArtifactSpec,
    raw_path: Path,
    meta_path: Path,
    config: EvaluatorConfig,
) -> tuple[BaselineContinuation, ...]:
    """Hash, validate, and import a frozen baseline without trusting its metrics."""

    if file_sha256(raw_path) != spec.raw_sha256:
        raise ValueError(f"{spec.method} baseline raw SHA-256 changed")
    if file_sha256(meta_path) != spec.meta_sha256:
        raise ValueError(f"{spec.method} baseline meta SHA-256 changed")
    meta = json.loads(meta_path.read_text())
    if (
        not isinstance(meta, dict)
        or meta.get("method") != spec.source_method
        or meta.get("model") != "gpt2"
        or meta.get("sae_hook") != config.phase_b.hook
        or meta.get("sae_release") != config.phase_b.sae.release
        or tuple(meta.get("prompts", ())) != config.phase_b.prompts
        or tuple(meta.get("seeds", ())) != config.phase_b.generation_seeds
        or tuple(float(value) for value in meta.get("alpha_hat", ())) != config.phase_b.alpha_hat
        or tuple(float(value) for value in meta.get("alpha_hat_stress", ()))
        != config.phase_b.alpha_hat_stress
        or int(meta.get("max_new_tokens", -1)) != config.phase_b.max_new_tokens
        or float(meta.get("temperature", -1)) != config.phase_b.temperature
        or float(meta.get("top_p", -1)) != config.phase_b.top_p
        or int(meta.get("top_k", -1)) != config.phase_b.top_k
        or float(meta.get("freq_penalty", -1)) != config.phase_b.freq_penalty
        or float(meta.get("activation_norm_mean", -1)) != config.phase_b.activation_norm_mean
    ):
        raise ValueError(f"{spec.method} baseline metadata differs from the frozen protocol")
    if spec.method == "naive" and meta.get("denoiser_sha256") != spec.denoiser_checkpoint_sha256:
        raise ValueError("naive baseline denoiser provenance changed")
    if spec.method == "shrinkage_k080" and float(meta.get("kappa", -1)) != spec.kappa:
        raise ValueError("shrinkage baseline kappa changed")

    expected = {
        (cell.vector, cell.alpha_hex, cell.prompt_id, cell.generation_seed): cell
        for cell in expected_cells(config)
    }
    observed: dict[tuple[str, str, int, int], BaselineContinuation] = {}
    with raw_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
                key = _baseline_key(row)
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid baseline row {line_number}") from error
            if key not in expected:
                raise ValueError(f"unexpected baseline cell at row {line_number}")
            if key in observed:
                raise ValueError(f"duplicate baseline cell at row {line_number}")
            observed[key] = parse_baseline_row(row, expected[key])
    if set(observed) != set(expected):
        raise ValueError(
            f"{spec.method} baseline coverage differs: missing={len(set(expected) - set(observed))}"
        )
    return tuple(observed[key] for key in expected)


def validate_cross_baseline_coverage(
    baselines: dict[str, tuple[BaselineContinuation, ...]],
) -> None:
    """Require exact continuation-cell coverage across every reused baseline."""

    if set(baselines) != {"additive", "naive", "shrinkage_k080"}:
        raise ValueError("clean evidence path requires all three frozen baselines")

    def keys(rows: tuple[BaselineContinuation, ...]) -> set[tuple[str, str, int, int]]:
        return {(row.vector, row.alpha.hex(), row.prompt_id, row.generation_seed) for row in rows}

    reference = keys(baselines["additive"])
    for method, rows in baselines.items():
        if keys(rows) != reference:
            raise ValueError(f"{method} baseline covers different continuation cells")


def prepare_generation_batch(  # noqa: ANN001
    model, prompts: list[str]
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Create the frozen left-padded batch with exactly one attended BOS per row."""

    if not prompts or not all(isinstance(prompt, str) and prompt for prompt in prompts):
        raise ValueError("generation prompts must be nonempty text")
    tokens = model.to_tokens(prompts, prepend_bos=True)
    if tokens.ndim != 2 or tokens.shape[0] != len(prompts):
        raise ValueError("generation tokenization returned an unexpected batch shape")
    prepend_bos = True
    mask = get_attention_mask(model.tokenizer, tokens, prepend_bos).bool()
    bos = model.tokenizer.bos_token_id
    if bos is None or not bool((mask.sum(dim=1) >= 2).all()):
        raise ValueError("every generation prompt needs BOS plus at least one token")
    for row in range(tokens.shape[0]):
        attended = tokens[row, mask[row]]
        if int(attended[0]) != bos or int((attended == bos).sum()) != 1:
            raise ValueError("each generation row must contain exactly one attended leading BOS")
    return tokens, mask, prepend_bos
