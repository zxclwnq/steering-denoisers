"""Apply the frozen concept-independent selection rule to the completed 2x2 arms.

Reads only the immutable per-arm reports. No steering metric, no Phase-B row, no
DEV or held-out artifact, and no LLM judge may enter this stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from interp.activations import file_sha256
from interp.phase_a import (
    lifecycle_record,
    write_failure_receipt,
    write_immutable_json,
)
from interp.provenance import source_revision
from interp.scaling import apply_selection_rule, load_scaling_protocol


def collect_arm_reports(protocol, reports_dir: Path, protocol_sha256: str) -> dict:  # noqa: ANN001
    """Load and validate one immutable report per frozen arm."""

    reports = {}
    for arm in protocol.arms:
        path = reports_dir / f"{arm.arm_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"arm report is missing: {path}")
        report = json.loads(path.read_text())
        if report.get("status") != "complete":
            raise ValueError(f"arm report {path} is not complete")
        if report.get("experiment_id") != protocol.experiment_id:
            raise ValueError(f"arm report {path} belongs to another experiment")
        if report.get("arm") != arm.arm_id:
            raise ValueError(f"arm report {path} identifies arm {report.get('arm')!r}")
        if report.get("protocol", {}).get("sha256") != protocol_sha256:
            raise ValueError(f"arm report {path} was produced under a different protocol")
        protected = report["protected_data"]
        if any(protected.values()):
            raise ValueError(f"arm report {path} records protected-data access")
        spec = report["arm_spec"]
        if spec["parameters"] != arm.parameters or spec["dataset"] != arm.dataset:
            raise ValueError(f"arm report {path} does not match the frozen arm definition")
        if spec["total_activation_presentations"] != protocol.total_activation_presentations:
            raise ValueError(f"arm report {path} is not compute matched")
        reports[arm.arm_id] = {"path": path, "sha256": file_sha256(path), "report": report}
    return reports


def _summary_row(arm, entry: dict) -> dict:  # noqa: ANN001
    report = entry["report"]
    phase_a = report["phase_a"]
    selection = report["selection_inputs"]
    cells = {
        key: {
            "mean_delta_lm": value["mean_delta_lm"],
            "mean_relative_l2": value["geometry"]["mean_relative_l2"],
            "mean_cosine": value["geometry"]["mean_cosine"],
        }
        for key, value in phase_a["cells"].items()
    }
    corruptions = {key: value["mean_delta_lm"] for key, value in phase_a["corruptions"].items()}
    recovered = {
        key: corruptions[key.split("_")[0]] - value["mean_delta_lm"]
        for key, value in cells.items()
    }
    return {
        "arm": arm.arm_id,
        "parameters": arm.parameters,
        "unique_activation_tokens": arm.unique_activation_tokens,
        "dataset": arm.dataset,
        "report_sha256": entry["sha256"],
        "checkpoint_step": report["checkpoint_selection"]["checkpoint_step"],
        "checkpoint_sha256": report["checkpoint_selection"]["checkpoint_sha256"],
        "primary_mean_reconstructed_delta_lm": selection["mean_reconstructed_delta_lm"],
        "val_flow_mse": selection["val_flow_mse"],
        "val_flow_mse_by_bin": report["validation_flow_loss"]["val_flow_mse_by_bin"],
        "val_cosine_velocity": report["validation_flow_loss"]["val_cosine_velocity"],
        "mean_clean_lm_loss": phase_a["mean_clean_lm_loss"],
        "corruption_delta_lm": corruptions,
        "cells": cells,
        "recovered_damage": recovered,
        "identity_delta_lm": phase_a["identity"]["mean_delta_lm"],
        "identity_flow_evaluations": phase_a["identity"]["flow_evaluations"],
        "noise_sha256": phase_a["noise"]["sha256"],
    }


def _run(args: argparse.Namespace) -> dict:
    protocol = load_scaling_protocol(args.config)
    protocol_sha256 = file_sha256(args.config)
    entries = collect_arm_reports(protocol, args.reports_dir, protocol_sha256)

    noise_hashes = {
        arm_id: entry["report"]["phase_a"]["noise"]["sha256"] for arm_id, entry in entries.items()
    }
    if len(set(noise_hashes.values())) != 1:
        raise ValueError(f"arms did not share matched epsilon: {noise_hashes}")
    validation_arrays = {
        arm_id: entry["report"]["validation_artifact"]["validation_report"]["sha256"]["array"]
        for arm_id, entry in entries.items()
    }
    if len(set(validation_arrays.values())) != 1:
        raise ValueError(f"arms used different validation artifacts: {validation_arrays}")

    selection = apply_selection_rule(
        [
            {
                "arm_id": arm_id,
                "primary_effects": entry["report"]["selection_inputs"]["primary_effects"],
                "val_flow_mse": entry["report"]["selection_inputs"]["val_flow_mse"],
                "parameters": entry["report"]["selection_inputs"]["parameters"],
                "unique_activation_tokens": entry["report"]["selection_inputs"][
                    "unique_activation_tokens"
                ],
            }
            for arm_id, entry in entries.items()
        ],
        protocol,
    )
    rows = [_summary_row(arm, entries[arm.arm_id]) for arm in protocol.arms]
    return {
        "status": "complete",
        "stage": "concept_independent_arm_selection",
        "experiment_id": protocol.experiment_id,
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "protocol": {"path": str(args.config), "sha256": protocol_sha256},
        "matched_epsilon_sha256": next(iter(set(noise_hashes.values()))),
        "validation_array_sha256": next(iter(set(validation_arrays.values()))),
        "compute_matching": {
            "total_activation_presentations": protocol.total_activation_presentations,
            "optimizer_steps": protocol.optimizer_steps,
            "batch_size": protocol.batch_size,
        },
        "arms": rows,
        "selection": selection,
        "protected_data": {
            "steering_vectors_loaded": False,
            "dev_directions_loaded": False,
            "held_out_accessed": False,
            "phase_b_loaded": False,
            "llm_judge_used": False,
        },
        "next_stage_note": (
            "Phase B may only be consulted after this selection. The optional "
            "NFE 10/20 recheck is diagnostic and cannot change the selected arm."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--reports-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    command = [sys.executable, *sys.argv]
    started_utc = datetime.now(UTC).isoformat()

    try:
        report = _run(args)
        report.update(
            lifecycle_record(
                status="complete",
                command=command,
                started_utc=started_utc,
                finished_utc=datetime.now(UTC).isoformat(),
            )
        )
        write_immutable_json(args.output, report)
    except KeyboardInterrupt as error:
        write_failure_receipt(
            args.output, error, status="INTERRUPTED", command=command, started_utc=started_utc
        )
        raise
    except Exception as error:
        write_failure_receipt(
            args.output, error, status="INVALID", command=command, started_utc=started_utc
        )
        raise

    print(f"{'arm':<18}{'params':>12}{'unique':>12}{'primary dLM':>14}{'val_flow_mse':>14}")
    for row in report["arms"]:
        print(
            f"{row['arm']:<18}{row['parameters']:>12,}{row['unique_activation_tokens']:>12,}"
            f"{row['primary_mean_reconstructed_delta_lm']:>14.6f}{row['val_flow_mse']:>14.6f}"
        )
    selection = report["selection"]
    print(f"\nleader: {selection['leader']}   tied: {selection['tied_with_leader']}")
    print(f"selected: {selection['selected_arm']} (decided by {selection['decided_by']})")
    print(f"wrote immutable selection report: {args.output}")


if __name__ == "__main__":
    main()
