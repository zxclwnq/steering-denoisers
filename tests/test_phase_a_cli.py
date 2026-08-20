"""The Phase A command surface stays narrow enough to preserve the frozen protocol."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from interp.phase_a import write_failure_receipt

ROOT = Path(__file__).resolve().parents[1]


def test_phase_a_cli_exposes_only_artifact_locations_and_device() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/eval_flow_reconstruction.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--activation-dir" in result.stdout
    assert "--token-cache-dir" in result.stdout
    assert "--output" in result.stdout
    assert "--device" in result.stdout
    assert "--checkpoint" not in result.stdout
    assert "--n-sequences" not in result.stdout
    assert "--t-start" not in result.stdout
    assert "--nfe" not in result.stdout


def test_interruption_receipt_records_status_reason_and_exact_command(tmp_path: Path) -> None:
    command = ["/workspace/project/.venv/bin/python", "scripts/eval_flow_reconstruction.py"]

    path = write_failure_receipt(
        tmp_path / "intended.json",
        KeyboardInterrupt("manual stop"),
        status="INTERRUPTED",
        command=command,
        started_utc="2026-08-13T17:00:00+00:00",
    )

    receipt = json.loads(path.read_text())
    assert receipt["status"] == "INTERRUPTED"
    assert receipt["termination_reason"] == "KeyboardInterrupt"
    assert receipt["command"] == command
    assert receipt["started_utc"] == "2026-08-13T17:00:00+00:00"
    assert receipt["finished_utc"]
