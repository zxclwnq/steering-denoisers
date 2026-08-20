"""Immutable activation-array format and document-disjoint train/validation splits."""

from __future__ import annotations

import contextlib
import hashlib
import json
import mmap
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ACT_DIR = Path("activations")
DTYPE = np.float16


def collection_status(
    metadata: dict,
    *,
    status: str,
    written_tokens: int,
    error: BaseException | None,
) -> dict:
    """Return an auditable terminal collection record without hiding failures."""

    if status not in {"complete", "INTERRUPTED", "INVALID"}:
        raise ValueError(f"unsupported terminal collection status {status!r}")
    if written_tokens < 0:
        raise ValueError("written_tokens cannot be negative")
    if status == "complete" and error is not None:
        raise ValueError("a complete collection cannot carry an error")
    if status != "complete" and error is None:
        raise ValueError("a failed or interrupted collection must carry an error")
    result = {
        **metadata,
        "status": status,
        "written_tokens": written_tokens,
        "finished_utc": datetime.now(UTC).isoformat(),
        "termination_reason": "completed" if error is None else type(error).__name__,
    }
    if error is not None:
        result["error"] = repr(error)
    else:
        result.pop("error", None)
    return result


@dataclass(frozen=True)
class ActivationDataset:
    array: np.ndarray
    meta: dict
    mean: np.ndarray
    std: np.ndarray

    def __len__(self) -> int:
        return int(self.array.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.array.shape[1])


@dataclass(frozen=True)
class ActivationSplit:
    train: np.ndarray
    val: np.ndarray
    per_seq: int
    seed: int
    val_fraction: float

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for indices in (self.train, self.val):
            digest.update(np.ascontiguousarray(indices, dtype=np.int64).tobytes())
        return digest.hexdigest()[:16]


def activation_paths(name: str, root: Path | None = None) -> tuple[Path, Path, Path]:
    directory = ACT_DIR if root is None else root
    return (
        directory / f"{name}.npy",
        directory / f"{name}.json",
        directory / f"{name}_stats.npz",
    )


# Hashing the 46 GiB array pulls the whole file into page cache, which counts toward
# the worker's 56 GiB cgroup limit and was the steepest climb of the whole validation
# (measured 50.6 -> 53.8 GiB in 40 s). These pages are read through ordinary buffered
# I/O and are not mapped, so plain fadvise does evict them.
_HASH_RELEASE_EVERY_BYTES = 512 * 1024 * 1024


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Stream a SHA256 without letting the file accumulate in page cache.

    Releasing pages changes only what the kernel caches, never which bytes are read
    or hashed, so the digest is identical with and without it.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        unreleased = 0
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            unreleased += len(chunk)
            if unreleased >= _HASH_RELEASE_EVERY_BYTES:
                with contextlib.suppress(AttributeError, OSError):
                    os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                unreleased = 0
    return digest.hexdigest()


def artifact_hashes(name: str, root: Path | None = None) -> dict[str, str]:
    """Hash the array, metadata, and stored statistics as immutable training inputs."""

    array_path, meta_path, stats_path = activation_paths(name, root)
    return {
        "array": file_sha256(array_path),
        "metadata": file_sha256(meta_path),
        "statistics": file_sha256(stats_path),
    }


def load_validation_report(
    name: str,
    root: Path,
    *,
    expected_split_fingerprint: str,
    verify_hashes: bool,
) -> dict:
    """Load a full-scan report and optionally re-hash inputs immediately before use."""

    path = root / f"{name}_validation.json"
    if not path.exists():
        raise FileNotFoundError(f"activation validation report does not exist: {path}")
    report = json.loads(path.read_text())
    if report.get("status") != "VALID" or report.get("name") != name:
        raise ValueError(f"activation validation report is not VALID for {name!r}")
    if report.get("split_fingerprint") != expected_split_fingerprint:
        raise ValueError(
            "validation split fingerprint "
            f"{report.get('split_fingerprint')} != expected {expected_split_fingerprint}"
        )
    if verify_hashes:
        observed = artifact_hashes(name, root)
        if observed != report.get("sha256"):
            raise ValueError(
                f"activation artifact hash mismatch: observed {observed}, "
                f"validated {report.get('sha256')}"
            )
    return report


def make_split(
    n_tokens: int, per_seq: int, val_fraction: float, seed: int
) -> ActivationSplit:
    """Split whole source documents, then expand their token indices."""

    if n_tokens % per_seq:
        raise ValueError(f"{n_tokens} tokens is not a whole number of {per_seq}-token sequences")
    n_sequences = n_tokens // per_seq
    n_validation = int(round(n_sequences * val_fraction))
    if not 0 < n_validation < n_sequences:
        raise ValueError("val_fraction must leave at least one train and validation sequence")
    order = np.random.default_rng(seed).permutation(n_sequences)
    val_sequences = np.sort(order[:n_validation])
    train_sequences = np.sort(order[n_validation:])
    offsets = np.arange(per_seq, dtype=np.int64)

    def expand(sequences: np.ndarray) -> np.ndarray:
        return (sequences[:, None].astype(np.int64) * per_seq + offsets).ravel()

    return ActivationSplit(
        train=expand(train_sequences),
        val=expand(val_sequences),
        per_seq=per_seq,
        seed=seed,
        val_fraction=val_fraction,
    )


def split_stats(
    dataset: ActivationDataset, indices: np.ndarray, chunk: int = 262_144
) -> tuple[np.ndarray, np.ndarray]:
    """Compute population mean/std in float64 over only the requested split."""

    if len(indices) == 0 or chunk < 1:
        raise ValueError("statistics require nonempty indices and a positive chunk")
    total = np.zeros(dataset.d_model, dtype=np.float64)
    squares = np.zeros(dataset.d_model, dtype=np.float64)
    for start in range(0, len(indices), chunk):
        values = np.asarray(dataset.array[indices[start : start + chunk]], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("activation array contains non-finite values")
        total += values.sum(axis=0)
        squares += np.square(values).sum(axis=0)
        del values
        release_mapped_pages(dataset.array)
    mean = total / len(indices)
    variance = np.maximum(squares / len(indices) - np.square(mean), 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def validate_activation_metadata(
    dataset: ActivationDataset,
    *,
    expected_name: str,
    expected_split: str,
    expected_model: str,
    expected_resolved_model_name: str,
    expected_model_revision: str,
    expected_hook: str,
    expected_ctx: int,
    expected_d_model: int,
    expected_dataset_repository: str,
    expected_dataset_config: str,
    expected_dataset_revision: str,
    expected_tokenizer: str,
) -> None:
    """Reject an activation artifact unless its scientific identity is exact."""

    meta = dataset.meta
    expected_fields = {
        "name": expected_name,
        "split": expected_split,
        "model": expected_model,
        "resolved_model_name": expected_resolved_model_name,
        "model_revision": expected_model_revision,
        "hook": expected_hook,
        "ctx": expected_ctx,
        "dataset_repository": expected_dataset_repository,
        "dataset_config": expected_dataset_config,
        "dataset_revision": expected_dataset_revision,
        "tokenizer": expected_tokenizer,
        "compute_dtype": "float32",
        "storage_dtype": DTYPE.__name__,
        "bos_dropped": True,
        "steering_vectors_used": None,
    }
    for field, expected in expected_fields.items():
        if meta.get(field) != expected:
            raise ValueError(
                f"activation metadata {field}={meta.get(field)!r} != expected {expected!r}"
            )
    token_cache_hash = meta.get("token_cache_sha256")
    if not isinstance(token_cache_hash, str) or len(token_cache_hash) != 64:
        raise ValueError("activation metadata token_cache_sha256 is not a SHA-256 digest")
    if dataset.d_model != expected_d_model:
        raise ValueError(
            f"activation d_model={dataset.d_model} != expected {expected_d_model}"
        )
    if meta.get("shape") != list(dataset.array.shape):
        raise ValueError("activation metadata shape does not match the array")
    per_sequence = expected_ctx - 1
    if len(dataset) % per_sequence:
        raise ValueError("activation count is not a whole number of source sequences")
    expected_n_sequences = len(dataset) // per_sequence
    if meta.get("n_seqs") != expected_n_sequences:
        raise ValueError(
            f"activation metadata n_seqs={meta.get('n_seqs')!r} "
            f"!= expected {expected_n_sequences}"
        )


def _advise_random_access(array: np.ndarray) -> None:
    """Tell the kernel this memory map is read by scattered rows, not sequentially.

    Training gathers ~1000 random rows per step from a file far larger than RAM. With
    the default 128 KB readahead the kernel faults in ~85 KB for every 1.5 KB row, so
    the run becomes disk-bound (measured: 3.7 it/s, GPU idle). MADV_RANDOM disables that
    readahead and measured 2.2-2.9x more rows/s on the real artifact.

    This changes only how pages are fetched. The bytes returned, the sampling order, and
    every tensor downstream are unaffected, so it cannot alter a result. Best effort: a
    platform without madvise silently keeps the default behaviour.
    """

    handle = getattr(array, "_mmap", None)
    advice = getattr(mmap, "MADV_RANDOM", None)
    if handle is None or advice is None:
        return
    with contextlib.suppress(OSError, ValueError):  # platform-dependent behaviour
        handle.madvise(advice)


def release_mapped_pages(array: np.ndarray, path: Path | None = None) -> None:
    """Drop already-processed pages of a memory-mapped artifact from memory.

    The 46 GiB artifacts are written and re-read inside a 56 GiB cgroup, where page
    cache counts toward memory.current. Both halves are required: fadvise cannot evict
    a page that is still mapped into the process, and every page these scans touch
    stays mapped, so fadvise alone frees nothing. madvise(MADV_DONTNEED) on the mapping
    is what releases them; for a shared file-backed mapping the pages are re-read from
    the file if touched again, so a writer must msync first and a reader loses nothing.

    Changes only when the kernel holds bytes in memory, never which bytes are read,
    written, or summed. Best effort: a platform without either call keeps the default.
    """

    handle = getattr(array, "_mmap", None)
    advice = getattr(mmap, "MADV_DONTNEED", None)
    if handle is not None and advice is not None:
        with contextlib.suppress(OSError, ValueError):  # platform-dependent
            handle.madvise(advice)

    if path is None:  # np.memmap records the file it is backed by
        filename = getattr(array, "filename", None)
        if filename is None:
            return
        path = Path(filename)

    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass
    finally:
        os.close(fd)


CGROUP_MEMORY_CURRENT = Path("/sys/fs/cgroup/memory.current")

# The worker's cgroup limit is 56.2 GiB. Training reads a 45.8 GiB artifact by random
# row for 35-45 minutes, so its page cache grows toward that limit even though every
# page is clean. This threshold is a signal to drop clean pages, NOT an error and NOT
# a kill switch: crossing it is expected and the run continues untouched.
PAGE_CACHE_SOFT_LIMIT_BYTES = 51 * 1024**3


def cgroup_memory_current(path: Path | None = None) -> int | None:
    """Current cgroup v2 memory charge in bytes, or None outside a cgroup v2 container.

    The path is resolved at call time, not bound as a default, so it stays overridable.
    """

    try:
        return int((path or CGROUP_MEMORY_CURRENT).read_text().strip())
    except (OSError, ValueError):
        return None


@dataclass
class PageCacheGovernor:
    """Keeps a memory-mapped artifact's page cache under a soft limit during long runs.

    Reads nothing, orders nothing and samples nothing: it only asks the kernel to drop
    clean pages of a mapping that has already been read. MADV_RANDOM stays in force for
    access; dropped pages are re-faulted from the file on the next touch, so every byte
    a caller subsequently reads is identical. Throughput is traded for completion by
    design.
    """

    threshold_bytes: int = PAGE_CACHE_SOFT_LIMIT_BYTES
    check_every: int = 50
    releases: int = 0
    checks: int = 0
    last_before: int | None = None
    last_after: int | None = None
    _calls: int = 0

    def step(self, array: np.ndarray) -> bool:
        """Call once per batch. Returns True when this call released page cache."""

        self._calls += 1
        if self._calls % self.check_every:
            return False
        self.checks += 1
        before = cgroup_memory_current()
        if before is None or before < self.threshold_bytes:
            return False
        release_mapped_pages(array)
        self.releases += 1
        self.last_before = before
        self.last_after = cgroup_memory_current()
        return True


def load_activations(name: str, root: Path | None = None) -> ActivationDataset:
    """Open a complete activation artifact as a read-only memory map."""

    array_path, meta_path, stats_path = activation_paths(name, root)
    for path in (array_path, meta_path, stats_path):
        if not path.exists():
            raise FileNotFoundError(f"required activation artifact {path} does not exist")
    meta = json.loads(meta_path.read_text())
    if meta.get("status") != "complete":
        raise ValueError(f"activation artifact {name!r} is marked {meta.get('status')!r}")
    array = np.load(array_path, mmap_mode="r")
    _advise_random_access(array)
    if array.dtype != DTYPE:
        raise ValueError(f"activation dtype {array.dtype} != required {DTYPE}")
    if tuple(meta.get("shape", ())) != array.shape:
        raise ValueError(f"activation shape {array.shape} != recorded {meta.get('shape')}")
    with np.load(stats_path) as stats:
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
    if mean.shape != (array.shape[1],) or std.shape != mean.shape:
        raise ValueError("activation statistics do not match the activation width")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("activation statistics must be finite with strictly positive std")
    return ActivationDataset(array=array, meta=meta, mean=mean, std=std)
