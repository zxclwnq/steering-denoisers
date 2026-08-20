"""Collect a new immutable BOS-safe activation array on the execution worker."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from interp.activations import (
    DTYPE,
    activation_paths,
    collection_status,
    file_sha256,
    release_mapped_pages,
)
from interp.data import (
    CTX,
    corpus_for,
    load_tokens,
    token_cache_path,
    token_manifest_path,
    tokenizer_identity,
)
from interp.model import MODEL_NAME, MODEL_REVISION, STEERING_HOOK, load_model, resolve_device
from interp.provenance import source_revision

BATCH_SIZE = 8

# The 32M artifact is a 48 GB sequential memmap write. Flushing only at the end
# leaves the whole file as dirty page cache, which a memory-limited container
# counts against its own limit and OOM-kills. Writing back every ~6 GB and then
# dropping the already-written pages keeps resident memory flat.
#
# INFRASTRUCTURE ONLY: this changes *when* the kernel writes bytes out, never
# which bytes are written, in what order, or with what statistics. The array,
# metadata and statistics SHA256s are unaffected, so artifacts collected with and
# without it are byte-identical.
FLUSH_EVERY_ROWS = 4_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="openwebtext")
    parser.add_argument("--split", required=True, choices=("train", "dev", "val"))
    parser.add_argument("--n-tokens", required=True, type=int)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--token-cache-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.n_tokens < 1 or args.batch_size < 1:
        raise ValueError("n_tokens and batch_size must be positive")
    array_path, meta_path, stats_path = activation_paths(args.name, args.output_dir)
    existing = [path for path in (array_path, meta_path, stats_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite activation artifacts: {existing}")

    corpus = corpus_for(args.corpus)
    device = resolve_device(args.device)
    model = load_model(str(device), dtype=torch.float32)
    per_sequence = CTX - 1
    n_sequences = -(-args.n_tokens // per_sequence)
    n_tokens = n_sequences * per_sequence
    tokens = load_tokens(
        args.split,
        n_sequences,
        model.tokenizer,
        ctx=CTX,
        cache_dir=args.token_cache_dir,
        corpus=corpus,
    )
    token_path = token_cache_path(
        args.split, n_sequences, CTX, model.tokenizer, args.token_cache_dir, corpus
    )
    manifest_path = token_manifest_path(token_path)
    token_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "name": args.name,
        "status": "RUNNING",
        "split": args.split,
        "corpus": corpus.key,
        "dataset_repository": corpus.repository,
        "dataset_config": corpus.config,
        "dataset_revision": corpus.revision,
        "dataset_document_range": list(corpus.range(args.split)),
        "token_manifest": token_manifest,
        "tokenizer": tokenizer_identity(model.tokenizer),
        "tokenizer_bos_token_id": model.tokenizer.bos_token_id,
        "token_cache_file": token_path.name,
        "token_cache_sha256": file_sha256(token_path),
        "hook": STEERING_HOOK,
        "model": MODEL_NAME,
        "resolved_model_name": model.cfg.model_name,
        "model_revision": MODEL_REVISION,
        "compute_dtype": "float32",
        "storage_dtype": DTYPE.__name__,
        "shape": [n_tokens, model.cfg.d_model],
        "ctx": CTX,
        "bos_dropped": True,
        "n_seqs": n_sequences,
        # Every cached sequence is exactly CTX real tokens beginning with one BOS,
        # so collection has no padding positions and drops exactly one BOS per row.
        "raw_tokens": n_sequences * CTX,
        "bos_activations_discarded": n_sequences,
        "padding_activations_discarded": 0,
        "valid_activations": n_tokens,
        "source_revision": source_revision(),
        "started_utc": datetime.now(UTC).isoformat(),
        "steering_vectors_used": None,
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    written = 0
    try:
        output = np.lib.format.open_memmap(
            array_path,
            mode="w+",
            dtype=DTYPE,
            shape=(n_tokens, model.cfg.d_model),
        )
        total = np.zeros(model.cfg.d_model, dtype=np.float64)
        squares = np.zeros(model.cfg.d_model, dtype=np.float64)
        norm_sum = 0.0
        last_flush = 0
        for start in tqdm(range(0, n_sequences, args.batch_size), desc=args.name):
            token_batch = tokens[start : start + args.batch_size].to(device)
            _, cache = model.run_with_cache(token_batch, names_filter=[STEERING_HOOK])
            activations = cache[STEERING_HOOK][:, 1:, :].reshape(-1, model.cfg.d_model).float()
            if not torch.isfinite(activations).all():
                raise FloatingPointError(f"non-finite activation in sequence batch {start}")
            values = activations.cpu().numpy().astype(np.float64)
            total += values.sum(axis=0)
            squares += np.square(values).sum(axis=0)
            norm_sum += float(np.linalg.norm(values, axis=1).sum())
            output[written : written + len(values)] = values.astype(DTYPE)
            written += len(values)
            if written - last_flush >= FLUSH_EVERY_ROWS:
                output.flush()
                release_mapped_pages(output, array_path)
                last_flush = written
        if written != n_tokens:
            raise RuntimeError(f"wrote {written}/{n_tokens} activations")
        output.flush()
        mean = total / n_tokens
        variance = np.maximum(squares / n_tokens - np.square(mean), 0.0)
        std = np.sqrt(variance)
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
            raise FloatingPointError("computed invalid activation statistics")
        np.savez(stats_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
        metadata.update(
            mean_norm=norm_sum / n_tokens,
            mean_abs_mean=float(np.abs(mean).mean()),
            mean_std=float(std.mean()),
        )
        metadata = collection_status(
            metadata, status="complete", written_tokens=written, error=None
        )
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2))
    except KeyboardInterrupt as error:
        metadata = collection_status(
            metadata, status="INTERRUPTED", written_tokens=written, error=error
        )
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        raise
    except BaseException as error:
        metadata = collection_status(
            metadata, status="INVALID", written_tokens=written, error=error
        )
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
