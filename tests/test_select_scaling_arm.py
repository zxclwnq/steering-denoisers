"""Tests for the frozen cross-arm selection stage."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from interp.activations import file_sha256
from interp.scaling import load_scaling_protocol

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "configs" / "flow_scaling_2x2_v2.yaml"


def _module():
    spec = importlib.util.spec_from_file_location(
        "select_scaling_arm", REPO / "scripts" / "select_scaling_arm.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(protocol, arm, effects: np.ndarray, val_flow_mse: float, noise: str = "n" * 64) -> dict:
    cells = {
        f"t{t:.2f}_nfe{nfe}": {
            "mean_delta_lm": float(effects.mean()) + 0.01 * nfe,
            "geometry": {"mean_relative_l2": 0.32, "mean_cosine": 0.94, "n_activations": 32512},
        }
        for t in protocol.t_starts
        for nfe in protocol.nfes
    }
    return {
        "status": "complete",
        "experiment_id": protocol.experiment_id,
        "arm": arm.arm_id,
        "protocol": {"sha256": file_sha256(PROTOCOL)},
        "protected_data": {
            "steering_vectors_loaded": False,
            "dev_directions_loaded": False,
            "held_out_accessed": False,
            "phase_b_loaded": False,
            "llm_judge_used": False,
        },
        "arm_spec": {
            "parameters": arm.parameters,
            "dataset": arm.dataset,
            "total_activation_presentations": protocol.total_activation_presentations,
        },
        "checkpoint_selection": {"checkpoint_step": 250_000, "checkpoint_sha256": "c" * 64},
        "validation_artifact": {"validation_report": {"sha256": {"array": "a" * 64}}},
        "validation_flow_loss": {
            "val_flow_mse": val_flow_mse,
            "val_flow_mse_by_bin": {"0.00-0.10": 0.5},
            "val_cosine_velocity": 0.71,
        },
        "phase_a": {
            "mean_clean_lm_loss": 3.27,
            "identity": {"mean_delta_lm": 0.0, "flow_evaluations": 0},
            "corruptions": {
                f"t{t:.2f}": {"mean_delta_lm": 1.32} for t in protocol.t_starts
            },
            "cells": cells,
            "noise": {"sha256": noise},
        },
        "selection_inputs": {
            "primary_effects": effects.tolist(),
            "mean_reconstructed_delta_lm": float(effects.mean()),
            "val_flow_mse": val_flow_mse,
            "parameters": arm.parameters,
            "unique_activation_tokens": arm.unique_activation_tokens,
        },
    }


def _write_arms(tmp_path: Path, protocol, winner: str | None = None, noise=None) -> Path:
    rng = np.random.default_rng(0)
    directory = tmp_path / "reports"
    directory.mkdir()
    for index, arm in enumerate(protocol.arms):
        effects = rng.normal(1.0, 0.05, size=protocol.n_sequences)
        if arm.arm_id == winner:
            effects = effects - 0.5
        report = _report(
            protocol,
            arm,
            effects,
            val_flow_mse=1.0 - 0.001 * index,
            noise=(noise[index] if noise else "n" * 64),
        )
        (directory / f"{arm.arm_id}.json").write_text(json.dumps(report))
    return directory


def test_selection_stage_picks_the_resolved_winner(tmp_path: Path) -> None:
    module = _module()
    protocol = load_scaling_protocol(PROTOCOL)
    directory = _write_arms(tmp_path, protocol, winner="wide60m_fw32m")

    entries = module.collect_arm_reports(protocol, directory, file_sha256(PROTOCOL))
    assert set(entries) == {arm.arm_id for arm in protocol.arms}

    args = type("Args", (), {"config": PROTOCOL, "reports_dir": directory, "output": None})()
    result = module._run(args)

    assert result["selection"]["selected_arm"] == "wide60m_fw32m"
    assert result["selection"]["steering_metrics_used"] is False
    assert len(result["arms"]) == 4
    assert result["compute_matching"]["total_activation_presentations"] == 256_000_000
    assert result["matched_epsilon_sha256"] == "n" * 64
    assert all(not value for value in result["protected_data"].values())


def test_selection_stage_rejects_unmatched_epsilon(tmp_path: Path) -> None:
    module = _module()
    protocol = load_scaling_protocol(PROTOCOL)
    directory = _write_arms(
        tmp_path, protocol, noise=["n" * 64, "n" * 64, "m" * 64, "n" * 64]
    )
    args = type("Args", (), {"config": PROTOCOL, "reports_dir": directory, "output": None})()

    with pytest.raises(ValueError, match="matched epsilon"):
        module._run(args)


def test_selection_stage_rejects_a_protected_data_receipt(tmp_path: Path) -> None:
    module = _module()
    protocol = load_scaling_protocol(PROTOCOL)
    directory = _write_arms(tmp_path, protocol)
    path = directory / "wide60m_fw4m.json"
    leaked = json.loads(path.read_text())
    leaked["protected_data"]["held_out_accessed"] = True
    path.write_text(json.dumps(leaked))

    with pytest.raises(ValueError, match="protected-data access"):
        module.collect_arm_reports(protocol, directory, file_sha256(PROTOCOL))


def test_selection_stage_requires_every_arm(tmp_path: Path) -> None:
    module = _module()
    protocol = load_scaling_protocol(PROTOCOL)
    directory = _write_arms(tmp_path, protocol)
    (directory / "narrow16m_fw32m.json").unlink()

    with pytest.raises(FileNotFoundError, match="narrow16m_fw32m"):
        module.collect_arm_reports(protocol, directory, file_sha256(PROTOCOL))
