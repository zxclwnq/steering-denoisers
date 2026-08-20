"""Read-only full validation and hashing of a generated activation artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from interp.activations import (
    activation_paths,
    artifact_hashes,
    file_sha256,
    load_activations,
    make_split,
    release_mapped_pages,
    split_stats,
    validate_activation_metadata,
)
from interp.data import CTX, corpus_for
from interp.model import (
    MODEL_NAME,
    MODEL_RESOLVED_NAME,
    MODEL_REVISION,
    STEERING_HOOK,
    load_model,
    resolve_device,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--activation-dir", required=True, type=Path)
    parser.add_argument("--token-cache-dir", required=True, type=Path)
    parser.add_argument("--corpus", default="openwebtext")
    parser.add_argument("--expected-split", required=True, choices=("train", "dev", "val"))
    parser.add_argument("--bos-probe", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split-seed", type=int, default=20_260_807)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    corpus = corpus_for(args.corpus)
    dataset = load_activations(args.name, args.activation_dir)
    if args.bos_probe < 1:
        raise ValueError("bos-probe must be positive")
    device = resolve_device(args.device)
    model = load_model(str(device), dtype=torch.float32)
    validate_activation_metadata(
        dataset,
        expected_name=args.name,
        expected_split=args.expected_split,
        expected_model=MODEL_NAME,
        expected_resolved_model_name=MODEL_RESOLVED_NAME,
        expected_model_revision=MODEL_REVISION,
        expected_hook=STEERING_HOOK,
        expected_ctx=CTX,
        expected_d_model=model.cfg.d_model,
        expected_dataset_repository=corpus.repository,
        expected_dataset_config=corpus.config,
        expected_dataset_revision=corpus.revision,
        expected_tokenizer=model.tokenizer.name_or_path,
    )
    array_path, meta_path, stats_path = activation_paths(args.name, args.activation_dir)
    expected_bytes = dataset.array.size * dataset.array.dtype.itemsize
    if abs(array_path.stat().st_size - expected_bytes) >= 4096:
        raise ValueError("activation file size does not agree with shape and dtype")
    token_cache_path = args.token_cache_dir / dataset.meta["token_cache_file"]
    if not token_cache_path.is_file():
        raise FileNotFoundError(f"recorded token cache does not exist: {token_cache_path}")
    observed_token_hash = file_sha256(token_cache_path)
    if observed_token_hash != dataset.meta["token_cache_sha256"]:
        raise ValueError("recorded token-cache hash does not match the token cache")

    per_sequence = int(dataset.meta["ctx"]) - 1
    probe_sequences = min(args.bos_probe, int(dataset.meta["n_seqs"]))
    with torch.no_grad():
        ids = torch.tensor([[model.tokenizer.bos_token_id]], device=device)
        _, cache = model.run_with_cache(ids, names_filter=[STEERING_HOOK])
        bos_activation = cache[STEERING_HOOK][0, 0].float().cpu().numpy()
    probe = np.asarray(
        dataset.array[: probe_sequences * per_sequence], dtype=np.float32
    )
    bos_distance_min = float(np.linalg.norm(probe - bos_activation, axis=1).min())
    if bos_distance_min <= 1.0:
        raise ValueError(
            "stored activations contain a BOS activation after the claimed BOS drop"
        )
    norms = np.linalg.norm(probe, axis=1).reshape(probe_sequences, per_sequence)
    first_position_norm = float(norms[:, 0].mean())
    other_position_norm = float(norms[:, 1:].mean())
    if first_position_norm >= 3.0 * other_position_norm:
        raise ValueError("periodic sequence-start norm spike indicates retained BOS")

    observed_total = np.zeros(dataset.d_model, dtype=np.float64)
    observed_squares = np.zeros(dataset.d_model, dtype=np.float64)
    for start in range(0, len(dataset), 262_144):
        values = np.asarray(dataset.array[start : start + 262_144], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite activation values in chunk starting {start}")
        observed_total += values.sum(axis=0)
        observed_squares += np.square(values).sum(axis=0)
        del values
        release_mapped_pages(dataset.array, array_path)
    observed_mean = observed_total / len(dataset)
    observed_std = np.sqrt(
        np.maximum(observed_squares / len(dataset) - np.square(observed_mean), 0.0)
    )
    mean_error = float(np.abs(observed_mean - dataset.mean).max())
    std_error = float(np.abs(observed_std - dataset.std).max())
    if mean_error >= 0.01 or std_error >= 0.01:
        raise ValueError(
            "stored statistics disagree with full recomputation: "
            f"mean {mean_error}, std {std_error}"
        )

    split = make_split(len(dataset), per_sequence, args.val_fraction, args.split_seed)
    train_mean, train_std = split_stats(dataset, split.train)
    report = {
        "status": "VALID",
        "name": args.name,
        "shape": list(dataset.array.shape),
        "dtype": str(dataset.array.dtype),
        "artifact_split": dataset.meta["split"],
        "model": dataset.meta["model"],
        "hook": dataset.meta["hook"],
        "corpus": dataset.meta.get("corpus", "openwebtext"),
        "dataset_repository": dataset.meta["dataset_repository"],
        "dataset_config": dataset.meta["dataset_config"],
        "dataset_document_range": dataset.meta.get("dataset_document_range"),
        "token_manifest": dataset.meta.get("token_manifest"),
        "dataset_revision": dataset.meta["dataset_revision"],
        "tokenizer": dataset.meta["tokenizer"],
        "token_cache_sha256": observed_token_hash,
        "bos_distance_min": bos_distance_min,
        "first_position_mean_norm": first_position_norm,
        "other_position_mean_norm": other_position_norm,
        "mean_error_max_abs": mean_error,
        "std_error_max_abs": std_error,
        "split_fingerprint": split.fingerprint(),
        "n_train_tokens": int(len(split.train)),
        "n_val_tokens": int(len(split.val)),
        "train_mean_abs_max": float(np.abs(train_mean).max()),
        "train_std_min": float(train_std.min()),
        "train_std_max": float(train_std.max()),
        "sha256": artifact_hashes(args.name, args.activation_dir),
    }
    report_path = args.activation_dir / f"{args.name}_validation.json"
    if report_path.exists():
        raise FileExistsError(f"validation report already exists: {report_path}")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
