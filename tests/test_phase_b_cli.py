"""The Phase B smoke command cannot become a tuning surface."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformer_lens.utilities.tokenize_utils import get_attention_mask

from interp.phase_b import (
    load_phase_b_config,
    prepare_smoke_input,
    validate_phase_a_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def test_pretokenized_smoke_has_exactly_one_attended_bos() -> None:
    class Tokenizer:
        bos_token_id = 50256
        eos_token_id = 50256
        pad_token_id = 50256
        padding_side = "left"

    class Model:
        tokenizer = Tokenizer()

        @staticmethod
        def to_tokens(prompt: str, *, prepend_bos: bool) -> torch.Tensor:
            assert prompt == "frozen prompt"
            assert prepend_bos is True
            return torch.tensor([[50256, 123, 456]])

    tokens, prepend_bos = prepare_smoke_input(Model(), "frozen prompt")

    assert int((tokens == Model.tokenizer.bos_token_id).sum()) == 1
    assert prepend_bos is True
    assert get_attention_mask(Model.tokenizer, tokens, prepend_bos).tolist() == [[1, 1, 1]]
    assert get_attention_mask(Model.tokenizer, tokens, False).tolist() == [[0, 1, 1]]


def test_phase_b_smoke_cli_exposes_only_artifact_locations_and_device() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/smoke_flow_steering.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for option in (
        "--config",
        "--phase-a-config",
        "--phase-a-report",
        "--sae-dir",
        "--output",
        "--device",
    ):
        assert option in result.stdout
    for forbidden in (
        "--checkpoint",
        "--vector",
        "--prompt",
        "--alpha",
        "--t-start",
        "--nfe",
        "--seed",
        "--max-new-tokens",
    ):
        assert forbidden not in result.stdout


def _write_report(path: Path, *, gate: str = "PASS", dev_loaded: bool = False) -> None:
    phase_a_config = ROOT / "configs" / "flow_phase_a_100k_v1.yaml"
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "research_status": "SUPPORTED",
                "config": {
                    "sha256": hashlib.sha256(phase_a_config.read_bytes()).hexdigest()
                },
                "checkpoint_selection": {
                    "checkpoint_sha256": (
                        "9d1d3cb66b9eaab1cbc89edab121d5cfa318271d7502e2ce42230432faad30d2"
                    ),
                    "checkpoint_step": 99500,
                    "selection_metric": "val_flow_mse",
                    "selection_mode": "min",
                    "held_out_accessed": False,
                },
                "protected_data": {
                    "steering_vectors_loaded": False,
                    "dev_directions_loaded": dev_loaded,
                    "held_out_accessed": False,
                },
                "evaluation": {"gate": {"status": gate}},
            },
            sort_keys=True,
        )
        + "\n"
    )


def test_phase_b_requires_complete_passing_phase_a_evidence(tmp_path: Path) -> None:
    report_path = tmp_path / "phase_a.json"
    _write_report(report_path)
    cfg = load_phase_b_config(ROOT / "configs" / "flow_phase_b_dev_v1.yaml")
    cfg = replace(
        cfg,
        phase_a_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
    )

    receipt = validate_phase_a_evidence(
        cfg,
        ROOT / "configs" / "flow_phase_a_100k_v1.yaml",
        report_path,
    )

    assert receipt["gate"] == "PASS"
    assert receipt["checkpoint_step"] == 99500
    assert receipt["checkpoint_sha256"] == cfg.checkpoint_sha256
    assert receipt["phase_a_report_sha256"] == cfg.phase_a_report_sha256
    assert receipt["protected_data_loaded"] is False


@pytest.mark.parametrize("mutation", ["failed_gate", "dev_loaded", "tampered_after_freeze"])
def test_phase_b_rejects_failed_or_tampered_phase_a_evidence(
    tmp_path: Path, mutation: str
) -> None:
    report_path = tmp_path / "phase_a.json"
    _write_report(
        report_path,
        gate="FAIL" if mutation == "failed_gate" else "PASS",
        dev_loaded=mutation == "dev_loaded",
    )
    cfg = load_phase_b_config(ROOT / "configs" / "flow_phase_b_dev_v1.yaml")
    cfg = replace(
        cfg,
        phase_a_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
    )
    if mutation == "tampered_after_freeze":
        report_path.write_text(report_path.read_text() + "\n")

    with pytest.raises(ValueError):
        validate_phase_a_evidence(
            cfg,
            ROOT / "configs" / "flow_phase_a_100k_v1.yaml",
            report_path,
        )
