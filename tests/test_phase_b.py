"""Frozen DEV assets and the canonical SAE boundary for Phase B."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

import interp.phase_b as phase_b
from interp.phase_b import load_frozen_sae, load_phase_b_config, operational_smoke_claims

ROOT = Path(__file__).resolve().parents[1]
DEV_NAMES = (
    "allegations",
    "dungeon",
    "locations_addresses",
    "illicit_drugs",
    "law_enforcement_officials",
    "same_sex_marriage",
    "borders",
    "sports_awards",
)
DEV_FEATURES = (20428, 11356, 18660, 22059, 4513, 18752, 15946, 14241)


def test_smoke_claims_distinguish_operational_geometry_from_scientific_metrics() -> None:
    assert operational_smoke_claims() == {
        "purpose": "operational_only_no_scientific_selection",
        "scientific_selection_performed": False,
        "scientific_metrics_computed": [],
        "operational_diagnostics": [
            "deterministic_repeat",
            "flow_network_evaluation_count",
            "finite_steering_geometry",
        ],
    }


def test_phase_b_manifest_freezes_only_dev_assets_and_the_approved_smoke() -> None:
    assert phase_b.APPROVED_PHASE_B_CONFIG_SHA256 == (
        "4d2f0d5a2152f206e261b94107078fa06f253e863681db7e7ca79751b38322d0"
    )
    cfg = load_phase_b_config(ROOT / "configs" / "flow_phase_b_dev_v1.yaml")

    assert cfg.experiment_id == "clean_flow_phase_b_dev_v1"
    assert cfg.experiment_class == "dev_method_development"
    assert tuple(vector.name for vector in cfg.vectors) == DEV_NAMES
    assert tuple(vector.feature for vector in cfg.vectors) == DEV_FEATURES
    assert all(vector.split == "dev" for vector in cfg.vectors)
    assert cfg.vectors[1].desc == 'mentions of "dungeon" and related terms in various contexts'
    assert cfg.t_starts == (0.10, 0.25, 0.50)
    assert cfg.nfes == (1, 3, 5)
    assert cfg.alpha_hat == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0)
    assert cfg.alpha_hat_stress == (1.5, 2.0)
    assert cfg.activation_norm_mean == 88.76
    assert cfg.off_distribution_norm == 600.0
    assert cfg.noise_namespace == "flow_noise_v2"
    assert cfg.checkpoint_sha256 == (
        "9d1d3cb66b9eaab1cbc89edab121d5cfa318271d7502e2ce42230432faad30d2"
    )
    assert cfg.phase_a_config_sha256 == (
        "a762c27be901d6015a06794c695ff8893b107c93861e2599e944dabec4855b5e"
    )
    assert cfg.phase_a_report_sha256 == (
        "c8a0ee7602a79e0c6ac4ff094b53b3a740b2ca1c79e418994356eb651ddf2047"
    )
    assert cfg.sae.repo_id == "jbloom/GPT2-Small-SAEs-Reformatted"
    assert cfg.sae.revision == "57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9"
    assert cfg.sae.hook == "blocks.7.hook_resid_pre"
    assert cfg.sae.config_sha256 == (
        "93d39f5eefeb5c254bf45c871fddc9527619d3626eeb0bd015e5f7330945f88e"
    )
    assert cfg.sae.weights_sha256 == (
        "47bfb75008fdd7ebf068044c0c3a212606aaa3f5dc05f1d1a7cffe502002c0b6"
    )
    assert cfg.smoke.vector == "allegations"
    assert cfg.smoke.prompt_id == 0
    assert cfg.smoke.generation_seed == 0
    assert cfg.smoke.alpha_hat == 0.1
    assert cfg.smoke.t_start == 0.1
    assert cfg.smoke.nfe == 1
    assert cfg.smoke.max_new_tokens == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_vector",
        "non_dev",
        "wrong_grid",
        "duplicate",
        "description",
        "lexicon",
        "phase_a_report_sha",
        "experiment_id",
    ],
)
def test_phase_b_manifest_rejects_semantic_mutations(
    tmp_path: Path, mutation: str
) -> None:
    source = ROOT / "configs" / "flow_phase_b_dev_v1.yaml"
    raw = yaml.safe_load(source.read_text())
    if mutation == "extra_vector":
        raw["vectors"].append(
            {
                "name": "unapproved",
                "feature": 1,
                "rank": 99,
                "split": "dev",
                "desc": "unapproved",
                "lexicon": ["unapproved"],
            }
        )
    elif mutation == "non_dev":
        raw["vectors"][0]["split"] = "final"
    elif mutation == "wrong_grid":
        raw["flow"]["nfe"] = [1, 3]
    elif mutation == "duplicate":
        raw["vectors"][1]["name"] = raw["vectors"][0]["name"]
    elif mutation == "description":
        raw["vectors"][1]["desc"] = "changed after freeze"
    elif mutation == "lexicon":
        raw["vectors"][0]["lexicon"][0] = "changed_after_freeze"
    elif mutation == "phase_a_report_sha":
        raw["phase_a"]["report_sha256"] = "0" * 64
    elif mutation == "experiment_id":
        raw["experiment_id"] = "substituted_experiment"
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    with pytest.raises(ValueError):
        load_phase_b_config(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_sae_matches_hand_computed_relu_encoder_and_decoder_rows(
    tmp_path: Path,
) -> None:
    base = load_phase_b_config(ROOT / "configs" / "flow_phase_b_dev_v1.yaml").sae
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_name": "gpt2-small",
                "hook_point": "blocks.7.hook_resid_pre",
                "d_in": 2,
                "d_sae": 3,
            }
        )
    )
    weights_path = tmp_path / "sae_weights.safetensors"
    save_file(
        {
            "W_enc": torch.tensor([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]]),
            "W_dec": torch.tensor([[3.0, 4.0], [0.0, 2.0], [-5.0, 0.0]]),
            "b_enc": torch.tensor([0.0, 1.0, -1.0]),
            "b_dec": torch.tensor([1.0, 2.0]),
        },
        weights_path,
    )
    spec = replace(
        base,
        d_in=2,
        d_sae=3,
        config_sha256=_sha256(cfg_path),
        weights_sha256=_sha256(weights_path),
    )

    sae = load_frozen_sae(spec, tmp_path, device="cpu")

    assert torch.equal(
        sae.encode_features(torch.tensor([[2.0, 3.0]]), [0, 1, 2]),
        torch.tensor([[1.0, 3.0, 0.0]]),
    )
    assert torch.equal(
        sae.decoder_directions([0, 1, 2]),
        torch.tensor([[0.6, 0.8], [0.0, 1.0], [-1.0, 0.0]]),
    )
    assert all(not tensor.requires_grad for tensor in sae.tensors())


@pytest.mark.parametrize("mutation", ["config_hash", "weight_hash", "shape", "nonfinite"])
def test_frozen_sae_rejects_unbound_or_malformed_artifacts(
    tmp_path: Path, mutation: str
) -> None:
    base = load_phase_b_config(ROOT / "configs" / "flow_phase_b_dev_v1.yaml").sae
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_name": "gpt2-small",
                "hook_point": "blocks.7.hook_resid_pre",
                "d_in": 2,
                "d_sae": 3,
            }
        )
    )
    weights_path = tmp_path / "sae_weights.safetensors"
    w_enc = torch.ones(2, 3)
    if mutation == "shape":
        w_enc = torch.ones(2, 2)
    elif mutation == "nonfinite":
        w_enc[0, 0] = float("nan")
    save_file(
        {
            "W_enc": w_enc,
            "W_dec": torch.ones(3, 2),
            "b_enc": torch.zeros(3),
            "b_dec": torch.zeros(2),
        },
        weights_path,
    )
    spec = replace(
        base,
        d_in=2,
        d_sae=3,
        config_sha256=_sha256(cfg_path),
        weights_sha256=_sha256(weights_path),
    )
    if mutation == "config_hash":
        spec = replace(spec, config_sha256="0" * 64)
    elif mutation == "weight_hash":
        spec = replace(spec, weights_sha256="0" * 64)

    with pytest.raises(ValueError):
        load_frozen_sae(spec, tmp_path, device="cpu")
