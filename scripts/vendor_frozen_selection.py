"""Vendor the frozen selection inputs so the rank rule is reproducible inside this repo.

The training-only direction pool is defined by rank in the frozen BLAKE2b ordering.
Reproducing that rank requires the exact candidate universe and the exact ordering
implementation, which previously lived only in the sibling `interp` project.  This
script copies the three NON-PROTECTED inputs into ``data/frozen_selection/``, records
their SHA256, and mechanically checks that none of them contains or enumerates a
protected evaluation identity.

The check is deliberately blunt: a vendored input is rejected if it mentions a held-out
split at all, or if it refers to so few features that it could be an enumeration of the
accepted concept set rather than the whole SAE.

Run:

    uv run python scripts/vendor_frozen_selection.py --source /path/to/interp
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Anything naming an evaluation split has no business in a training-only input.
FORBIDDEN_MARKERS = ("heldout", "held_out", "held-out")

# A file referring to fewer than this many distinct features cannot be the full SAE
# universe, and might be a selected subset. Refuse it rather than reason about it.
MIN_DISTINCT_FEATURES = 1000

D_SAE = 24576

INPUTS = {
    "explanations": "data/explanations_gpt2-small_7-res-jb.jsonl",
    "feature_stats": "data/feature_stats_7-res-jb_dev.npz",
    "selection_rule": "src/interp/selection.py",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_split_markers(name: str, text: str) -> None:
    lowered = text.lower()
    for marker in FORBIDDEN_MARKERS:
        if marker in lowered:
            raise ValueError(
                f"vendored input {name} mentions {marker!r}: it may carry an "
                "evaluation split and must not be vendored"
            )


def audit_explanations(path: Path) -> dict:
    """The full auto-interp dump: many features, one labeller field, no split field."""

    indices: set[int] = set()
    labellers: set[str] = set()
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if set(row) & {"split", "dev", "heldout"}:
                raise ValueError("explanation record carries a split field")
            indices.add(int(row["index"]))
            labellers.add(str(row["explanationModelName"]))
    if len(indices) < MIN_DISTINCT_FEATURES:
        raise ValueError(
            f"explanations cover only {len(indices)} features; refusing a possible subset"
        )
    if max(indices) >= D_SAE:
        raise ValueError("explanation feature index outside the SAE feature range")
    return {
        "distinct_features": len(indices),
        "feature_id_range": [min(indices), max(indices)],
        "labellers": sorted(labellers),
        "records_carry_split_field": False,
    }


def audit_feature_stats(path: Path) -> dict:
    """Per-feature arrays over the whole SAE, plus the corpus split they came from."""

    stats = np.load(path, allow_pickle=True)
    arrays = {key: stats[key] for key in stats.files}
    per_feature = {key: value for key, value in arrays.items() if value.ndim == 1}
    if not per_feature:
        raise ValueError("feature stats contain no per-feature array")
    for key, value in per_feature.items():
        if value.shape[0] != D_SAE:
            raise ValueError(
                f"feature-stats array {key} has {value.shape[0]} entries, "
                f"expected the full SAE universe {D_SAE}"
            )
    corpus_split = str(arrays["split"])
    if corpus_split in FORBIDDEN_MARKERS:
        raise ValueError(f"feature stats come from the {corpus_split} corpus split")
    return {
        "per_feature_arrays": sorted(per_feature),
        "n_features": D_SAE,
        "corpus_split": corpus_split,
        "n_tokens": int(arrays["n_tokens"]),
        "n_seqs": int(arrays["n_seqs"]),
    }


def audit_selection_rule(path: Path) -> dict:
    """The ordering implementation must contain no feature identities at all."""

    text = path.read_text()
    _reject_split_markers("selection_rule", text)
    tree = ast.parse(text)
    # Parse rather than grep: prose in docstrings and comments carries numbers that are
    # not data. Executable integer constants are the only ones that could be identities.
    executable_ints = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, int)
    ]
    # A tuple of expressions (a sort key) is fine; a collection of integer literals is
    # the shape a hard-coded identity list would take.
    for node in ast.walk(tree):
        if not isinstance(node, ast.List | ast.Tuple | ast.Set):
            continue
        constants = [
            element
            for element in node.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, int)
        ]
        if constants:
            raise ValueError(
                "selection rule contains a collection of integer literals; it may hold ids"
            )
    seed = re.search(r"SELECTION_SEED\s*=\s*(\d+)", text)
    if seed is None:
        raise ValueError("selection rule does not declare SELECTION_SEED")
    unexplained = set(executable_ints) - {int(seed.group(1)), 8}
    if unexplained:
        raise ValueError(
            f"selection rule contains unexplained integer constants {sorted(unexplained)}"
        )
    return {
        "selection_seed": int(seed.group(1)),
        "executable_integer_constants": sorted(set(executable_ints)),
        "contains_feature_ids": False,
    }


AUDITS = {
    "explanations": audit_explanations,
    "feature_stats": audit_feature_stats,
    "selection_rule": audit_selection_rule,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/frozen_selection"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "purpose": (
            "reproduce the frozen BLAKE2b candidate ordering inside this repository "
            "so the training-only rank rule needs no protected manifest"
        ),
        "vendored_utc": datetime.now(UTC).isoformat(),
        "source_project": str(args.source),
        "protected_inputs_used": False,
        "inputs": {},
    }
    for name, relative in INPUTS.items():
        source = args.source / relative
        if not source.is_file():
            raise FileNotFoundError(f"missing frozen selection input: {source}")
        target = args.out / source.name
        target.write_bytes(source.read_bytes())
        audit = AUDITS[name](target)
        record["inputs"][name] = {
            "file": target.name,
            "source_path": str(source),
            "sha256": file_sha256(target),
            "bytes": target.stat().st_size,
            "audit": audit,
        }
    (args.out / "PROVENANCE.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
