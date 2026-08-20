#!/usr/bin/env python3
"""Versioned analysis of completed Phase-B DEV releases, and their narrow/wide comparison.

`analyze` recomputes every headline quantity from one release's immutable raw
rows. `compare` pairs two analyses over the eight frozen steering directions.
Neither subcommand regenerates continuations, touches held-out data, or calls an
LLM judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from interp.activations import file_sha256
from interp.phase_a import lifecycle_record, write_failure_receipt, write_immutable_json
from interp.phase_b_analysis import (
    ANALYSIS_VERSION,
    ROW_METRICS,
    descriptive,
    equal_alpha_paired,
    load_release_rows,
    matched_projection,
    nfe_effects,
    paired_release_difference,
    resample_matrix,
)
from interp.phase_b_evaluator import load_evaluator_config
from interp.provenance import source_revision

ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, records: list[dict]) -> str:
    if path.exists():
        raise FileExistsError(f"immutable analysis sidecar already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for record in records:
            text = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write(f"{text}\n".encode())
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def _analyze(args: argparse.Namespace) -> dict:
    cfg = load_evaluator_config(args.config)
    manifest_path = args.release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment_id") != cfg.experiment_id:
        raise ValueError("release manifest does not belong to this evaluator config")
    release_id = manifest["release_id"]
    vectors = tuple(vector.name for vector in cfg.phase_b.vectors)
    matrix = resample_matrix(vectors, seed=cfg.bootstrap_seed, n_resamples=cfg.bootstrap_resamples)

    artifacts: dict[str, dict[str, str]] = {}
    baselines = {}
    for spec in cfg.baselines:
        path = args.release_dir / "baselines" / f"{spec.method}.rescored.jsonl"
        baselines[spec.method] = load_release_rows(
            path,
            release_id=release_id,
            schema_version=cfg.schema_version,
            metric_versions=cfg.metric_versions,
            expected_method=spec.method,
            expected_arm=spec.method,
        )
        artifacts[spec.method] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "meta_sha256": file_sha256(path.with_suffix(".meta.json")),
        }
    arms = {}
    for arm in cfg.arms:
        path = args.release_dir / "flow" / f"{arm.arm_id}.jsonl"
        arms[arm.arm_id] = load_release_rows(
            path,
            release_id=release_id,
            schema_version=cfg.schema_version,
            metric_versions=cfg.metric_versions,
            expected_method="flow",
            expected_arm=arm.arm_id,
        )
        artifacts[arm.arm_id] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "meta_sha256": file_sha256(path.with_suffix(".meta.json")),
        }

    epsilon = {}
    for arm_id, rows in arms.items():
        epsilon[arm_id] = {
            (row["vector"], row["alpha_hex"], row["prompt_id"], row["generation_seed"]): row[
                "noise"
            ]["epsilon_sha256"]
            for row in rows
        }
    reference = epsilon[cfg.arms[0].arm_id]
    if any(value != reference for value in epsilon.values()):
        raise ValueError("flow arms did not share matched epsilon per continuation cell")
    # Epsilon is keyed only by (namespace, vector, exact alpha, prompt, seed, token
    # position) and is drawn at the 768-wide activation boundary, so a rerun with a
    # different prior must reproduce this digest exactly. Comparing it across
    # releases is what makes the narrow/wide difference a matched-noise comparison
    # rather than two independent samples.
    epsilon_digest = hashlib.sha256(
        json.dumps(
            sorted((list(key), value) for key, value in reference.items()),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    padding = {
        arm_id: sum(int(row["noise"]["padding_positions_evaluated"]) for row in rows)
        for arm_id, rows in arms.items()
    }
    if set(padding.values()) != {0}:
        raise ValueError("a flow arm evaluated padding positions")

    records: list[dict] = []
    analysis: dict[str, dict] = {}
    for arm in cfg.arms:
        rows = arms[arm.arm_id]
        entry: dict[str, object] = {"descriptive": descriptive(rows)}
        entry["equal_alpha_vs_additive"] = equal_alpha_paired(
            rows, baselines["additive"], vectors, matrix, confidence=cfg.bootstrap_confidence
        )
        for method, scale in (("additive", 1.0), ("shrinkage_k080", 0.8)):
            summary, arm_records = matched_projection(
                rows,
                baselines[method],
                vectors,
                matrix,
                coordinate_scale=scale,
                repetition_threshold=cfg.repetition_threshold,
                confidence=cfg.bootstrap_confidence,
            )
            entry[f"matched_projection_vs_{method}"] = summary
            records.extend({**record, "baseline": method} for record in arm_records)
        analysis[arm.arm_id] = entry

    brackets_sha256 = _write_jsonl(args.brackets, records)
    return {
        "status": "complete",
        "stage": "phase_b_dev_analysis",
        "analysis_version": ANALYSIS_VERSION,
        "experiment_id": cfg.experiment_id,
        "release_id": release_id,
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "evaluator_config": {"path": str(args.config), "sha256": file_sha256(args.config)},
        "release_manifest_sha256": file_sha256(manifest_path),
        "flow_checkpoint_sha256": cfg.flow_checkpoint_sha256,
        "flow_prior": (
            None
            if cfg.flow_prior is None
            else {
                "selected_arm": cfg.flow_prior.selected_arm,
                "parameters": cfg.flow_prior.parameters,
                "training_dataset": cfg.flow_prior.training_dataset,
                "comparison_release_id": cfg.flow_prior.comparison_release_id,
            }
        ),
        "raw_artifacts": artifacts,
        "vectors": list(vectors),
        "metrics": list(ROW_METRICS),
        "statistics": {
            "bootstrap_seed": cfg.bootstrap_seed,
            "bootstrap_resamples": cfg.bootstrap_resamples,
            "bootstrap_confidence": cfg.bootstrap_confidence,
            "unit": "vector_mean",
            "shared_resample_matrix": True,
            "repetition_threshold": cfg.repetition_threshold,
        },
        "matched_epsilon_across_arms": True,
        "epsilon_cell_digest": epsilon_digest,
        "padding_positions_evaluated": padding,
        "baselines": {
            method: {"descriptive": descriptive(rows)} for method, rows in baselines.items()
        },
        "arms": analysis,
        "nfe_effects": nfe_effects(arms, vectors, matrix, confidence=cfg.bootstrap_confidence),
        "brackets": {"path": str(args.brackets), "sha256": brackets_sha256, "rows": len(records)},
        "protected_data": {
            "held_out_accessed": False,
            "llm_judge_used": False,
            "continuations_regenerated": False,
        },
    }


COMPARISON_FAMILIES = (
    "equal_alpha_vs_additive",
    "matched_projection_vs_additive",
    "matched_projection_vs_shrinkage_k080",
)


def _compare(args: argparse.Namespace) -> dict:
    narrow = json.loads(args.narrow.read_text())
    wide = json.loads(args.wide.read_text())
    for name, document in (("narrow", narrow), ("wide", wide)):
        if document.get("status") != "complete" or document.get("analysis_version") != (
            ANALYSIS_VERSION
        ):
            raise ValueError(f"{name} analysis is not a completed {ANALYSIS_VERSION} report")
        if any(document["protected_data"].values()):
            raise ValueError(f"{name} analysis records protected-data access")
    if narrow["statistics"] != wide["statistics"]:
        raise ValueError("the two analyses used different statistical contracts")
    if narrow["vectors"] != wide["vectors"]:
        raise ValueError("the two analyses used different steering directions")
    if wide["flow_prior"] is None:
        raise ValueError("the wide analysis must name the prior it substituted")
    # Two releases form a legitimate pair either when the substituted prior names
    # the other side directly as its comparison, or when both sides are siblings
    # that name the same third release. The sibling case is how a capacity control
    # is compared against the arm the frozen rule selected: both were frozen
    # against the original narrow Phase-B release, neither against each other.
    narrow_prior = narrow.get("flow_prior")
    direct = wide["flow_prior"]["comparison_release_id"] == narrow["release_id"]
    siblings = (
        narrow_prior is not None
        and narrow_prior["comparison_release_id"] == wide["flow_prior"]["comparison_release_id"]
    )
    if not (direct or siblings):
        raise ValueError(
            "the two releases are neither a declared comparison pair nor siblings "
            "of one common comparison release"
        )
    if narrow["release_id"] == wide["release_id"]:
        raise ValueError("narrow and wide analyses are the same release")
    if narrow["epsilon_cell_digest"] != wide["epsilon_cell_digest"]:
        raise ValueError(
            "narrow and wide releases did not draw the same epsilon per continuation cell"
        )
    if narrow["flow_checkpoint_sha256"] == wide["flow_checkpoint_sha256"]:
        raise ValueError("narrow and wide analyses used the same flow checkpoint")

    vectors = tuple(narrow["vectors"])
    statistics = narrow["statistics"]
    matrix = resample_matrix(
        vectors,
        seed=int(statistics["bootstrap_seed"]),
        n_resamples=int(statistics["bootstrap_resamples"]),
    )
    confidence = float(statistics["bootstrap_confidence"])

    arms = {}
    for arm_id in narrow["arms"]:
        entry: dict[str, object] = {}
        for family in COMPARISON_FAMILIES:
            entry[family] = {
                metric: paired_release_difference(
                    wide["arms"][arm_id][family][metric],
                    narrow["arms"][arm_id][family][metric],
                    vectors,
                    matrix,
                    confidence=confidence,
                )
                for metric in ROW_METRICS
            }
            if family.startswith("matched_projection"):
                entry[f"{family}_support"] = {
                    "narrow": narrow["arms"][arm_id][family]["counts"],
                    "wide": wide["arms"][arm_id][family]["counts"],
                }
        entry["descriptive"] = {
            key: {
                "narrow": narrow["arms"][arm_id]["descriptive"][key],
                "wide": wide["arms"][arm_id]["descriptive"][key],
                "delta": wide["arms"][arm_id]["descriptive"][key]
                - narrow["arms"][arm_id]["descriptive"][key],
            }
            for key in narrow["arms"][arm_id]["descriptive"]
            if key in wide["arms"][arm_id]["descriptive"]
            and isinstance(narrow["arms"][arm_id]["descriptive"][key], (int, float))
        }
        arms[arm_id] = entry

    primary = arms["flow_t050_nfe1"]["matched_projection_vs_additive"]["nll"]
    best = min(
        (
            (arm_id, entry["matched_projection_vs_additive"]["nll"])
            for arm_id, entry in arms.items()
        ),
        key=lambda item: item[1]["wide_mean"],
    )
    return {
        "status": "complete",
        "stage": "phase_b_narrow_to_wide_comparison",
        "analysis_version": ANALYSIS_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "narrow": {
            "release_id": narrow["release_id"],
            "analysis_sha256": file_sha256(args.narrow),
            "flow_checkpoint_sha256": narrow["flow_checkpoint_sha256"],
        },
        "wide": {
            "release_id": wide["release_id"],
            "analysis_sha256": file_sha256(args.wide),
            "flow_checkpoint_sha256": wide["flow_checkpoint_sha256"],
            "flow_prior": wide["flow_prior"],
        },
        "vectors": list(vectors),
        "statistics": statistics,
        "baseline_metric_agreement": {
            method: {
                key: {
                    "narrow": narrow["baselines"][method]["descriptive"][key],
                    "wide": wide["baselines"][method]["descriptive"][key],
                }
                for key in ROW_METRICS
            }
            for method in narrow["baselines"]
        },
        "arms": arms,
        "hypotheses": {
            "H_capacity_transfer": {
                "statement": (
                    "matched-projection NLL penalty is smaller for the wide prior than "
                    "for the narrow prior"
                ),
                "primary_cell": "flow_t050_nfe1",
                "primary": primary,
                "resolved_improvement": primary["improved"],
                "best_wide_arm_by_matched_nll": {
                    "arm_id": best[0],
                    "wide_mean": best[1]["wide_mean"],
                    "narrow_mean": best[1]["narrow_mean"],
                    "difference": best[1]["mean"],
                    "ci_lower": best[1]["ci_lower"],
                    "ci_upper": best[1]["ci_upper"],
                },
            },
            "H_pareto": {
                "statement": (
                    "at matched realised steering strength some wide arm beats both "
                    "additive and shrinkage on NLL"
                ),
                "arms_below_zero_vs_additive": [
                    arm_id
                    for arm_id, entry in arms.items()
                    if entry["matched_projection_vs_additive"]["nll"]["wide_mean"] < 0.0
                ],
                "arms_resolved_below_zero_vs_additive": [
                    arm_id
                    for arm_id, entry in arms.items()
                    if wide["arms"][arm_id]["matched_projection_vs_additive"]["nll"]["ci_upper"]
                    < 0.0
                ],
                "arms_resolved_below_zero_vs_shrinkage": [
                    arm_id
                    for arm_id, entry in arms.items()
                    if wide["arms"][arm_id]["matched_projection_vs_shrinkage_k080"]["nll"][
                        "ci_upper"
                    ]
                    < 0.0
                ],
            },
            "H_nfe": {
                "statement": "NFE 1 remains at least as good as NFE 3 and 5 under steering",
                "narrow": narrow["nfe_effects"],
                "wide": wide["nfe_effects"],
            },
        },
        "protected_data": {
            "held_out_accessed": False,
            "llm_judge_used": False,
            "continuations_regenerated": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--config", required=True, type=Path)
    analyze.add_argument("--release-dir", required=True, type=Path)
    analyze.add_argument("--brackets", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("--narrow", required=True, type=Path)
    compare.add_argument("--wide", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    command = [sys.executable, *sys.argv]
    started_utc = datetime.now(UTC).isoformat()
    try:
        report = _analyze(args) if args.command == "analyze" else _compare(args)
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
    print(f"wrote immutable analysis: {args.output}")


if __name__ == "__main__":
    main()
