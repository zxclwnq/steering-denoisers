from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest
import yaml

from interp.prior_diagnostic import (
    assert_exact_file_sha256,
    load_prior_diagnostic_config,
    nfe_marginals,
    summarize_phase_a,
    wide_glp_parameter_count,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "flow_prior_diagnostic_v1.yaml"


def _cell(delta: float, relative_l2: float, cosine: float) -> dict:
    return {
        "mean_delta_lm": delta,
        "geometry": {
            "mean_relative_l2": relative_l2,
            "mean_cosine": cosine,
            "n_activations": 10,
        },
    }


def test_frozen_diagnostic_config_has_precommitted_grid_and_protections() -> None:
    cfg = load_prior_diagnostic_config(CONFIG)

    assert [item.step for item in cfg.checkpoints] == [10_000, 30_000, 50_000, 70_000, 99_500]
    assert [item.sha256 for item in cfg.checkpoints] == [
        "e886209dcfe02d94f51e6d2b4539780d928aa08ea832aae44358a54486b5e6f8",
        "733914e2b07ecc29296e5854afd4d788562d5590c9c6aa2bac1098fd1c6b7048",
        "cc42174db60bafadd98fe3929093c657b03a8ab2a8ac68e04052329ce72dbe50",
        "d2e50b36c99db81a0bae502df50c66b63ace283341554c1fd61f0f645b6a5d9e",
        "9d1d3cb66b9eaab1cbc89edab121d5cfa318271d7502e2ce42230432faad30d2",
    ]
    assert cfg.t_starts == (0.10, 0.25, 0.50)
    assert cfg.checkpoint_nfes == (1, 3, 5)
    assert cfg.nfe_sweep_step == 99_500
    assert cfg.nfe_sweep == (1, 3, 5, 10, 20)
    assert all(value == "forbidden" for value in cfg.protected_data.values())


def test_loader_rejects_mutated_frozen_config(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text())
    raw["evaluation"]["nfe_sweep"] = [1, 3, 5, 10, 30]
    changed = tmp_path / CONFIG.name
    changed.write_text(yaml.safe_dump(raw, sort_keys=False))

    with pytest.raises(ValueError, match="approved diagnostic config SHA256"):
        load_prior_diagnostic_config(changed)


def test_phase_a_summary_uses_aggregate_paired_damage_definition() -> None:
    report = {
        "corruptions": {"t0.50": _cell(2.0, 0.8, 0.5)},
        "cells": {"t0.50_nfe1": _cell(0.5, 0.2, 0.9)},
    }

    summary = summarize_phase_a(report)["t0.50_nfe1"]

    assert summary == {
        "t_start": 0.5,
        "nfe": 1,
        "corruption_delta_lm": 2.0,
        "reconstructed_delta_lm": 0.5,
        "recovered_damage": 1.5,
        "recovered_fraction": 0.75,
        "mean_relative_l2": 0.2,
        "mean_cosine": 0.9,
        "n_activations": 10,
    }


def test_nfe_marginals_report_positive_oriented_improvements() -> None:
    report = {
        "corruptions": {"t0.50": _cell(2.0, 0.8, 0.5)},
        "cells": {
            "t0.50_nfe1": _cell(0.8, 0.4, 0.7),
            "t0.50_nfe3": _cell(0.6, 0.3, 0.8),
            "t0.50_nfe5": _cell(0.5, 0.25, 0.85),
            "t0.50_nfe10": _cell(0.45, 0.23, 0.87),
            "t0.50_nfe20": _cell(0.44, 0.22, 0.88),
        },
    }

    got = nfe_marginals(report)["t0.50"]["1->3"]

    assert got["delta_lm_reduction"] == pytest.approx(0.2)
    assert got["recovered_damage_gain"] == pytest.approx(0.2)
    assert got["relative_l2_reduction"] == pytest.approx(0.1)
    assert got["cosine_gain"] == pytest.approx(0.1)


def test_wide_glp_parameter_count_is_exact() -> None:
    assert (
        wide_glp_parameter_count(
            activation_dim=768,
            d_model=1536,
            d_mlp=3072,
            n_blocks=3,
            time_dim=256,
            time_hidden=768,
        )
        == 60_407_808
    )


def test_diagnostic_cli_has_no_protected_scientific_imports() -> None:
    path = ROOT / "scripts" / "diagnose_flow_prior.py"
    tree = ast.parse(path.read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    source = path.read_text().lower()

    assert not any("phase_b" in name or "flow_steering" in name for name in imported)
    assert "load_frozen_sae" not in source
    assert "load_phase_b_config" not in source
    assert not any("judge" in name or "sae" in name for name in imported)


def test_exact_artifact_hash_validation_fails_loudly(tmp_path: Path) -> None:
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"frozen checkpoint")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    assert_exact_file_sha256(artifact, digest)
    artifact.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="SHA256"):
        assert_exact_file_sha256(artifact, digest)


def test_config_file_sha_is_stable_literal() -> None:
    assert len(hashlib.sha256(CONFIG.read_bytes()).hexdigest()) == 64
