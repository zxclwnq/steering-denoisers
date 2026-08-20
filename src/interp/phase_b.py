"""Fail-closed DEV manifest and exact frozen-SAE access for Phase B."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file
from transformer_lens.utilities.tokenize_utils import get_attention_mask

from .activations import file_sha256
from .phase_a import load_phase_a_config

APPROVED_PHASE_B_CONFIG_SHA256 = (
    "4d2f0d5a2152f206e261b94107078fa06f253e863681db7e7ca79751b38322d0"
)
_PHASE_A_CONFIG_SHA256 = "a762c27be901d6015a06794c695ff8893b107c93861e2599e944dabec4855b5e"
_PHASE_A_REPORT_SHA256 = "c8a0ee7602a79e0c6ac4ff094b53b3a740b2ca1c79e418994356eb651ddf2047"
_DEV_IDENTITY = (
    ("allegations", 20428, 0, "references to allegations"),
    (
        "dungeon",
        11356,
        9,
        'mentions of "dungeon" and related terms in various contexts',
    ),
    ("locations_addresses", 18660, 17, "specific locations and addresses"),
    (
        "illicit_drugs",
        22059,
        19,
        "mentions of illicit drugs, particularly heroin, and their effects",
    ),
    (
        "law_enforcement_officials",
        4513,
        22,
        "mentions of law enforcement officials, particularly sheriffs",
    ),
    (
        "same_sex_marriage",
        18752,
        24,
        "references to same-sex marriage and related discussions",
    ),
    ("borders", 15946, 27, "references to borders and border-related topics"),
    ("sports_awards", 14241, 32, "references to award-winning players in sports"),
)
_PROMPTS = (
    "The first thing to say is that",
    "Yesterday afternoon, I went to",
    "In my opinion, the most important",
    "She looked around the room and",
    "It started when someone decided to",
    "According to a recent report, the",
    "Here is what happened next:",
    "He had never seen anything quite like",
    "One question that keeps coming up is",
    "Over the past few years, people have",
)
_T_STARTS = (0.10, 0.25, 0.50)
_NFES = (1, 3, 5)
_ALPHA_HAT = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0)
_ALPHA_HAT_STRESS = (1.5, 2.0)


@dataclass(frozen=True)
class DevVectorSpec:
    name: str
    feature: int
    rank: int
    split: str
    desc: str
    lexicon: tuple[str, ...]


@dataclass(frozen=True)
class FrozenSAESpec:
    release: str
    repo_id: str
    revision: str
    neuronpedia_id: str
    model_name: str
    hook: str
    d_in: int
    d_sae: int
    config_filename: str
    weights_filename: str
    config_sha256: str
    weights_sha256: str


@dataclass(frozen=True)
class SmokeSpec:
    vector: str
    prompt_id: int
    generation_seed: int
    alpha_hat: float
    t_start: float
    nfe: int
    max_new_tokens: int


@dataclass(frozen=True)
class PhaseBConfig:
    experiment_id: str
    experiment_class: str
    checkpoint_sha256: str
    checkpoint_step: int
    phase_a_config_sha256: str
    phase_a_report_sha256: str
    model_name: str
    resolved_model_name: str
    model_revision: str
    hook: str
    d_model: int
    sae: FrozenSAESpec
    vectors: tuple[DevVectorSpec, ...]
    prompts: tuple[str, ...]
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    freq_penalty: float
    generation_seeds: tuple[int, ...]
    activation_norm_mean: float
    alpha_hat: tuple[float, ...]
    alpha_hat_stress: tuple[float, ...]
    t_starts: tuple[float, ...]
    nfes: tuple[int, ...]
    off_distribution_norm: float
    noise_namespace: str
    smoke: SmokeSpec
    raw: dict


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a full SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value


def load_phase_b_config(path: Path) -> PhaseBConfig:
    """Load the single frozen DEV protocol; there is no split-selection API."""

    observed_sha256 = file_sha256(path)
    if observed_sha256 != APPROVED_PHASE_B_CONFIG_SHA256:
        raise ValueError(
            f"Phase B config SHA-256 {observed_sha256} != approved "
            f"{APPROVED_PHASE_B_CONFIG_SHA256}"
        )
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Phase B config must contain a mapping")
    if (
        raw.get("version") != 1
        or raw.get("status") != "frozen_dev_protocol"
        or raw.get("phase") != "B_dev_steering"
        or raw.get("split") != "dev"
        or raw.get("final_evaluation_access") != "forbidden"
        or raw.get("experiment_id") != "clean_flow_phase_b_dev_v1"
        or raw.get("experiment_class") != "dev_method_development"
    ):
        raise ValueError("Phase B must use the frozen DEV-only protocol")

    phase_a = raw["phase_a"]
    expected_phase_a = {
        "config_sha256": _PHASE_A_CONFIG_SHA256,
        "report_sha256": _PHASE_A_REPORT_SHA256,
        "gate": "PASS",
        "checkpoint_step": 99_500,
        "checkpoint_sha256": (
            "9d1d3cb66b9eaab1cbc89edab121d5cfa318271d7502e2ce42230432faad30d2"
        ),
        "selection_metric": "val_flow_mse",
        "selection_mode": "min",
    }
    for field, expected in expected_phase_a.items():
        if phase_a.get(field) != expected:
            raise ValueError(f"phase_a.{field} differs from the frozen Phase A evidence")

    model = raw["model"]
    if model != {
        "name": "gpt2-small",
        "resolved_name": "gpt2",
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "hook": "blocks.7.hook_resid_pre",
        "d_model": 768,
    }:
        raise ValueError("canonical GPT-2 interface changed")

    sae_raw = raw["sae"]
    expected_sae = {
        "release": "gpt2-small-res-jb",
        "repo_id": "jbloom/GPT2-Small-SAEs-Reformatted",
        "revision": "57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9",
        "neuronpedia_id": "gpt2-small/7-res-jb",
        "model_name": "gpt2-small",
        "hook": "blocks.7.hook_resid_pre",
        "d_in": 768,
        "d_sae": 24576,
        "config_filename": "cfg.json",
        "weights_filename": "sae_weights.safetensors",
        "config_sha256": (
            "93d39f5eefeb5c254bf45c871fddc9527619d3626eeb0bd015e5f7330945f88e"
        ),
        "weights_sha256": (
            "47bfb75008fdd7ebf068044c0c3a212606aaa3f5dc05f1d1a7cffe502002c0b6"
        ),
        "encoder": "relu_centered_by_b_dec",
        "direction": "unit_normalized_W_dec_row",
    }
    if sae_raw != expected_sae:
        raise ValueError("frozen SAE identity or mathematical interface changed")
    sae = FrozenSAESpec(**{key: sae_raw[key] for key in FrozenSAESpec.__dataclass_fields__})

    vectors = tuple(
        DevVectorSpec(
            name=str(item["name"]),
            feature=int(item["feature"]),
            rank=int(item["rank"]),
            split=str(item["split"]),
            desc=str(item["desc"]),
            lexicon=tuple(str(word) for word in item["lexicon"]),
        )
        for item in raw["vectors"]
    )
    identity = tuple((item.name, item.feature, item.rank, item.desc) for item in vectors)
    if identity != _DEV_IDENTITY or any(item.split != "dev" for item in vectors):
        raise ValueError("Phase B vector manifest is not the exact frozen DEV set")
    if any(not item.lexicon or len(set(item.lexicon)) != len(item.lexicon) for item in vectors):
        raise ValueError("DEV lexicons must be nonempty and contain unique entries")

    flow = raw["flow"]
    t_starts = tuple(float(value) for value in flow["t_start"])
    nfes = tuple(int(value) for value in flow["nfe"])
    if (
        flow.get("kind") != "sdedit_partial_noising"
        or flow.get("integrator") != "reverse_explicit_euler"
        or t_starts != _T_STARTS
        or nfes != _NFES
        or float(flow.get("off_distribution_norm")) != 600.0
    ):
        raise ValueError("frozen flow inference grid or algorithm changed")

    noise = raw["noise"]
    if noise != {
        "namespace": "flow_noise_v2",
        "identity": [
            "vector",
            "exact_alpha",
            "prompt_id",
            "generation_seed",
            "token_position",
        ],
        "independent_of": [
            "method",
            "t_start",
            "nfe",
            "batch_order",
            "hook_call_grouping",
        ],
        "draw_at_guarded_positions": True,
    }:
        raise ValueError("matched flow-noise identity changed")

    generation = raw["generation"]
    prompts = tuple(str(prompt) for prompt in generation["prompts"])
    alpha_hat = tuple(float(value) for value in generation["alpha_hat"])
    stress = tuple(float(value) for value in generation["alpha_hat_stress"])
    seeds = tuple(int(value) for value in generation["seeds"])
    decoding = generation["decoding"]
    if (
        generation.get("source_protocol_sha256")
        != "0fe8b88e4ac4ea480067d818a3faaed44cb38a310f5897183bcd003006423a81"
        or prompts != _PROMPTS
        or alpha_hat != _ALPHA_HAT
        or stress != _ALPHA_HAT_STRESS
        or seeds != (0, 1, 2)
        or float(generation["activation_norm_mean"]) != 88.76
        or decoding
        != {
            "max_new_tokens": 48,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 0,
            "freq_penalty": 0.0,
        }
    ):
        raise ValueError("frozen generation protocol changed")

    smoke_raw = raw["smoke"]
    if smoke_raw != {
        "purpose": "operational_only_no_scientific_selection",
        "vector": "allegations",
        "prompt_id": 0,
        "generation_seed": 0,
        "alpha_hat": 0.1,
        "t_start": 0.10,
        "nfe": 1,
        "max_new_tokens": 2,
    }:
        raise ValueError("predetermined Phase B smoke cell changed")
    smoke = SmokeSpec(
        vector=smoke_raw["vector"],
        prompt_id=int(smoke_raw["prompt_id"]),
        generation_seed=int(smoke_raw["generation_seed"]),
        alpha_hat=float(smoke_raw["alpha_hat"]),
        t_start=float(smoke_raw["t_start"]),
        nfe=int(smoke_raw["nfe"]),
        max_new_tokens=int(smoke_raw["max_new_tokens"]),
    )

    return PhaseBConfig(
        experiment_id=str(raw["experiment_id"]),
        experiment_class=str(raw["experiment_class"]),
        checkpoint_sha256=_require_sha(phase_a["checkpoint_sha256"], "checkpoint SHA"),
        checkpoint_step=int(phase_a["checkpoint_step"]),
        phase_a_config_sha256=_require_sha(phase_a["config_sha256"], "Phase A config SHA"),
        phase_a_report_sha256=_require_sha(phase_a["report_sha256"], "Phase A report SHA"),
        model_name=model["name"],
        resolved_model_name=model["resolved_name"],
        model_revision=model["revision"],
        hook=model["hook"],
        d_model=int(model["d_model"]),
        sae=sae,
        vectors=vectors,
        prompts=prompts,
        max_new_tokens=int(decoding["max_new_tokens"]),
        temperature=float(decoding["temperature"]),
        top_p=float(decoding["top_p"]),
        top_k=int(decoding["top_k"]),
        freq_penalty=float(decoding["freq_penalty"]),
        generation_seeds=seeds,
        activation_norm_mean=float(generation["activation_norm_mean"]),
        alpha_hat=alpha_hat,
        alpha_hat_stress=stress,
        t_starts=t_starts,
        nfes=nfes,
        off_distribution_norm=float(flow["off_distribution_norm"]),
        noise_namespace=noise["namespace"],
        smoke=smoke,
        raw=raw,
    )


@dataclass(frozen=True)
class FrozenSAE:
    """The exact standard ReLU SAE operations Phase B actually consumes."""

    spec: FrozenSAESpec
    w_enc: torch.Tensor
    w_dec: torch.Tensor
    b_enc: torch.Tensor
    b_dec: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return self.w_enc, self.w_dec, self.b_enc, self.b_dec

    def _feature_ids(self, feature_ids: list[int]) -> torch.Tensor:
        if not feature_ids or len(set(feature_ids)) != len(feature_ids):
            raise ValueError("feature_ids must be nonempty and unique")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in feature_ids):
            raise ValueError("feature_ids must contain integers")
        if min(feature_ids) < 0 or max(feature_ids) >= self.spec.d_sae:
            raise ValueError("feature id is outside the frozen SAE width")
        return torch.tensor(feature_ids, device=self.w_enc.device)

    def decoder_directions(self, feature_ids: list[int]) -> torch.Tensor:
        ids = self._feature_ids(feature_ids)
        return torch.nn.functional.normalize(self.w_dec[ids], dim=-1)

    def encode_features(self, h: torch.Tensor, feature_ids: list[int]) -> torch.Tensor:
        if h.shape[-1:] != (self.spec.d_in,) or not h.is_floating_point():
            raise ValueError("SAE input must be floating point with the frozen activation width")
        if not torch.isfinite(h).all():
            raise ValueError("SAE input must be finite")
        ids = self._feature_ids(feature_ids)
        value = h.to(device=self.w_enc.device, dtype=self.w_enc.dtype)
        return torch.relu((value - self.b_dec) @ self.w_enc[:, ids] + self.b_enc[ids])


def load_frozen_sae(
    spec: FrozenSAESpec, directory: Path, *, device: torch.device | str
) -> FrozenSAE:
    """Load only exact, SHA-bound SAE files from an explicit local directory."""

    config_path = directory / spec.config_filename
    weights_path = directory / spec.weights_filename
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"frozen SAE files are absent below {directory}")
    if file_sha256(config_path) != spec.config_sha256:
        raise ValueError("frozen SAE config SHA-256 mismatch")
    if file_sha256(weights_path) != spec.weights_sha256:
        raise ValueError("frozen SAE weights SHA-256 mismatch")

    metadata = json.loads(config_path.read_text())
    if (
        metadata.get("model_name") != spec.model_name
        or metadata.get("hook_point") != spec.hook
        or int(metadata.get("d_in", -1)) != spec.d_in
        or int(metadata.get("d_sae", -1)) != spec.d_sae
    ):
        raise ValueError("frozen SAE metadata does not match the Phase B manifest")

    state = load_file(str(weights_path), device=str(torch.device(device)))
    expected_shapes = {
        "W_enc": (spec.d_in, spec.d_sae),
        "W_dec": (spec.d_sae, spec.d_in),
        "b_enc": (spec.d_sae,),
        "b_dec": (spec.d_in,),
    }
    if set(state) != set(expected_shapes):
        raise ValueError("frozen SAE weights contain an unexpected tensor set")
    for name, shape in expected_shapes.items():
        value = state[name]
        if tuple(value.shape) != shape or not value.is_floating_point():
            raise ValueError(f"SAE tensor {name} has the wrong shape or dtype")
        if not torch.isfinite(value).all():
            raise ValueError(f"SAE tensor {name} contains non-finite values")
        value.requires_grad_(False)

    sae = FrozenSAE(
        spec=spec,
        w_enc=state["W_enc"],
        w_dec=state["W_dec"],
        b_enc=state["b_enc"],
        b_dec=state["b_dec"],
    )
    norms = sae.w_dec.norm(dim=-1)
    if not torch.isfinite(norms).all() or not bool((norms > 0).all()):
        raise ValueError("SAE decoder rows must have finite nonzero norms")
    if not math.isfinite(float(norms.mean())):
        raise ValueError("SAE decoder norm summary is non-finite")
    return sae


def validate_phase_a_evidence(
    cfg: PhaseBConfig,
    phase_a_config_path: Path,
    phase_a_report_path: Path,
) -> dict[str, str | int | bool]:
    """Bind Phase B to the exact successful, concept-independent Phase A gate."""

    if file_sha256(phase_a_config_path) != cfg.phase_a_config_sha256:
        raise ValueError("Phase A config SHA-256 differs from the Phase B freeze")
    phase_a = load_phase_a_config(phase_a_config_path)
    if (
        phase_a.checkpoint_sha256 != cfg.checkpoint_sha256
        or phase_a.checkpoint_step != cfg.checkpoint_step
        or phase_a.selection_metric != "val_flow_mse"
        or phase_a.selection_mode != "min"
    ):
        raise ValueError("Phase A checkpoint selection differs from the Phase B freeze")
    observed_report_sha = file_sha256(phase_a_report_path)
    if observed_report_sha != cfg.phase_a_report_sha256:
        raise ValueError("Phase A report SHA-256 differs from the Phase B freeze")
    report = json.loads(phase_a_report_path.read_text())
    selection = report.get("checkpoint_selection", {})
    protected = report.get("protected_data", {})
    gate = report.get("evaluation", {}).get("gate", {})
    if report.get("status") != "complete" or report.get("research_status") != "SUPPORTED":
        raise ValueError("Phase A evidence is not a completed supported result")
    if gate.get("status") != "PASS":
        raise ValueError("Phase A gate did not pass")
    if report.get("config", {}).get("sha256") != cfg.phase_a_config_sha256:
        raise ValueError("Phase A report embeds a different config SHA-256")
    if (
        selection.get("checkpoint_sha256") != cfg.checkpoint_sha256
        or selection.get("checkpoint_step") != cfg.checkpoint_step
        or selection.get("selection_metric") != "val_flow_mse"
        or selection.get("selection_mode") != "min"
        or selection.get("held_out_accessed") is not False
    ):
        raise ValueError("Phase A report does not prove the frozen concept-independent selection")
    protected_fields = (
        "steering_vectors_loaded",
        "dev_directions_loaded",
        "held_out_accessed",
    )
    if any(protected.get(field) is not False for field in protected_fields):
        raise ValueError("Phase A report does not prove protected steering data stayed unloaded")
    return {
        "gate": "PASS",
        "checkpoint_step": cfg.checkpoint_step,
        "checkpoint_sha256": cfg.checkpoint_sha256,
        "phase_a_config_sha256": cfg.phase_a_config_sha256,
        "phase_a_report_sha256": observed_report_sha,
        "protected_data_loaded": False,
    }


def prepare_smoke_input(model, prompt: str) -> tuple[torch.Tensor, bool]:  # noqa: ANN001
    """Tokenize one prompt with exactly one BOS that TransformerLens will attend to."""

    tokens = model.to_tokens(prompt, prepend_bos=True)
    tokenizer = model.tokenizer
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 2:
        raise ValueError("smoke prompt must tokenize to one nonempty sequence plus BOS")
    bos = tokenizer.bos_token_id
    if bos is None or int(tokens[0, 0]) != bos or int((tokens == bos).sum()) != 1:
        raise ValueError("smoke input must contain exactly one leading BOS token")
    # For a pre-tokenized tensor this flag does not add another BOS. It tells TL's
    # attention-mask builder that the leading pad/BOS-shared ID is a real attended BOS.
    prepend_bos = True
    mask = get_attention_mask(tokenizer, tokens, prepend_bos)
    if not bool(mask.all()):
        raise ValueError("single-prompt smoke input unexpectedly contains a masked token")
    return tokens, prepend_bos


def operational_smoke_claims() -> dict[str, str | bool | list[str]]:
    """Label what the smoke measures without implying a scientific steering result."""

    return {
        "purpose": "operational_only_no_scientific_selection",
        "scientific_selection_performed": False,
        "scientific_metrics_computed": [],
        "operational_diagnostics": [
            "deterministic_repeat",
            "flow_network_evaluation_count",
            "finite_steering_geometry",
        ],
    }
