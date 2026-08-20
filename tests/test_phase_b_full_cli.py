"""The full Phase B evaluator exposes logistics, never scientific tuning knobs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from interp.phase_b_evaluator import prepare_generation_batch

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_generation_batch_has_one_attended_bos_and_explicit_padding() -> None:
    class Tokenizer:
        bos_token_id = 99
        eos_token_id = 99
        pad_token_id = 99
        padding_side = "left"

    class Model:
        tokenizer = Tokenizer()

        @staticmethod
        def to_tokens(prompts: list[str], *, prepend_bos: bool) -> torch.Tensor:
            assert prompts == ["long", "short"]
            assert prepend_bos is True
            return torch.tensor([[99, 1, 2], [99, 99, 3]])

    tokens, mask, prepend_bos = prepare_generation_batch(Model(), ["long", "short"])

    assert tokens.tolist() == [[99, 1, 2], [99, 99, 3]]
    assert mask.tolist() == [[True, True, True], [False, True, True]]
    assert prepend_bos is True
    assert mask.sum(dim=1).tolist() == [3, 2]


def test_full_evaluator_cli_has_no_scientific_overrides() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_flow_phase_b.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for command in ("validate-release", "rescore-baselines", "smoke", "run-all"):
        assert command in result.stdout
    for forbidden in (
        "--vector",
        "--alpha",
        "--t-start",
        "--nfe",
        "--seed",
        "--prompt",
        "--split",
        "--temperature",
        "--top-p",
        "--max-new-tokens",
        "--best-arm",
    ):
        assert forbidden not in result.stdout
