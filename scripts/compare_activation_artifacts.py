"""Compare a rebuilt activation artifact against the preserved historical record.

The 32M FineWeb array was lost with the previous GPU worker; only its metadata,
statistics, and validation report survive locally. This compares a rebuild against that
record field by field, so a SHA mismatch can be judged rather than waved away.

    uv run python scripts/compare_activation_artifacts.py \
        --name resid7_fw_train_32000k_v1 \
        --historical results/remote_pull_20260814/fineweb_activations \
        --rebuilt results/rebuild_20260815/fineweb_activations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Fields that must be bit-identical for the rebuild to be the same experiment.
INVARIANTS = (
    "shape",
    "dtype",
    "artifact_split",
    "model",
    "hook",
    "corpus",
    "dataset_repository",
    "dataset_config",
    "dataset_revision",
    "tokenizer",
    "token_cache_sha256",
    "split_fingerprint",
    "n_train_tokens",
    "n_val_tokens",
)

# Float summaries: equality here is evidence the difference is at fp16-rounding scale.
SUMMARIES = (
    "bos_distance_min",
    "first_position_mean_norm",
    "other_position_mean_norm",
    "mean_error_max_abs",
    "std_error_max_abs",
    "train_mean_abs_max",
    "train_std_min",
    "train_std_max",
)

META_SUMMARIES = (
    "valid_activations",
    "raw_tokens",
    "n_seqs",
    "bos_activations_discarded",
    "padding_activations_discarded",
    "written_tokens",
    "mean_norm",
    "mean_abs_mean",
    "mean_std",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--rebuilt", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    def load(directory: Path, suffix: str) -> dict:
        return json.loads((directory / f"{args.name}{suffix}").read_text())

    old_validation = load(args.historical, "_validation.json")
    new_validation = load(args.rebuilt, "_validation.json")
    old_meta = load(args.historical, ".json")
    new_meta = load(args.rebuilt, ".json")

    report: dict[str, object] = {
        "name": args.name,
        "sha256_historical": old_validation["sha256"],
        "sha256_rebuilt": new_validation["sha256"],
        "array_sha256_matches": (
            old_validation["sha256"]["array"] == new_validation["sha256"]["array"]
        ),
        "statistics_sha256_matches": (
            old_validation["sha256"]["statistics"] == new_validation["sha256"]["statistics"]
        ),
        "status_rebuilt": new_validation.get("status"),
    }

    invariants = {}
    for field in INVARIANTS:
        old_value, new_value = old_validation.get(field), new_validation.get(field)
        invariants[field] = {"historical": old_value, "rebuilt": new_value,
                             "match": old_value == new_value}
    report["invariants"] = invariants
    report["all_invariants_match"] = all(item["match"] for item in invariants.values())

    summaries = {}
    for field in SUMMARIES:
        old_value, new_value = old_validation.get(field), new_validation.get(field)
        if isinstance(old_value, int | float) and isinstance(new_value, int | float):
            summaries[field] = {
                "historical": old_value,
                "rebuilt": new_value,
                "abs_diff": abs(new_value - old_value),
                "bit_identical": old_value == new_value,
            }
    for field in META_SUMMARIES:
        old_value, new_value = old_meta.get(field), new_meta.get(field)
        if isinstance(old_value, int | float) and isinstance(new_value, int | float):
            summaries[field] = {
                "historical": old_value,
                "rebuilt": new_value,
                "abs_diff": abs(new_value - old_value),
                "bit_identical": old_value == new_value,
            }
    report["summaries"] = summaries

    old_stats = np.load(args.historical / f"{args.name}_stats.npz")
    new_stats = np.load(args.rebuilt / f"{args.name}_stats.npz")
    arrays = {}
    for key in sorted(set(old_stats.files) & set(new_stats.files)):
        old_array = np.asarray(old_stats[key], dtype=np.float64)
        new_array = np.asarray(new_stats[key], dtype=np.float64)
        if old_array.shape != new_array.shape:
            arrays[key] = {"shape_historical": old_array.shape,
                           "shape_rebuilt": new_array.shape, "match": False}
            continue
        difference = np.abs(new_array - old_array)
        scale = np.maximum(np.abs(old_array), 1e-12)
        arrays[key] = {
            "shape": list(old_array.shape),
            "max_abs_diff": float(difference.max()),
            "mean_abs_diff": float(difference.mean()),
            "max_rel_diff": float((difference / scale).max()),
            "bit_identical": bool(np.array_equal(old_array, new_array)),
            "n_differing_entries": int((difference > 0).sum()),
        }
    report["statistics_arrays"] = arrays

    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
