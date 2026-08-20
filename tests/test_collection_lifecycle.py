"""Failure-status tests for immutable activation collection artifacts."""

from __future__ import annotations

from datetime import UTC, datetime

from interp.activations import collection_status


def test_collection_status_records_failure_reason_and_progress() -> None:
    started = {"name": "tiny", "status": "RUNNING", "started_utc": "earlier"}

    failed = collection_status(
        started,
        status="INVALID",
        written_tokens=127,
        error=RuntimeError("disk failed"),
    )

    assert failed["status"] == "INVALID"
    assert failed["written_tokens"] == 127
    assert failed["termination_reason"] == "RuntimeError"
    assert failed["error"] == "RuntimeError('disk failed')"
    assert datetime.fromisoformat(failed["finished_utc"]).tzinfo == UTC


def test_complete_collection_status_has_no_error() -> None:
    complete = collection_status(
        {"status": "RUNNING"}, status="complete", written_tokens=254, error=None
    )

    assert complete["status"] == "complete"
    assert complete["written_tokens"] == 254
    assert complete["termination_reason"] == "completed"
    assert "error" not in complete



class _StubMapping:
    """Stands in for mmap.mmap, whose madvise is read-only and cannot be patched."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[int] = []
        self.error = error

    def madvise(self, advice: int, *_rest: int) -> None:
        self.calls.append(advice)
        if self.error is not None:
            raise self.error


class _StubArray:
    def __init__(self, mapping: object) -> None:
        self._mmap = mapping


def test_page_release_advises_the_mapping_not_only_the_file(tmp_path) -> None:
    """Writing and re-reading a 46 GiB artifact inside a 56 GiB cgroup OOM-killed the
    container because fadvise cannot evict pages that are still mapped. Releasing
    memory REQUIRES madvise on the mapping; a fadvise-only version frees nothing."""

    import mmap as mmap_mod

    from interp.activations import release_mapped_pages

    mapping = _StubMapping()
    path = tmp_path / "array.bin"
    path.write_bytes(b"\x00" * 4096)

    release_mapped_pages(_StubArray(mapping), path)

    assert mapping.calls == [mmap_mod.MADV_DONTNEED], "mapping was not advised DONTNEED"


def test_page_release_survives_a_platform_without_madvise(tmp_path) -> None:
    """Best effort: a failing or absent madvise must degrade, not crash."""

    from interp.activations import release_mapped_pages

    path = tmp_path / "array.bin"
    path.write_bytes(b"\x00" * 4096)

    release_mapped_pages(_StubArray(_StubMapping(OSError("unsupported"))), path)
    release_mapped_pages(_StubArray(None), path)


def test_page_release_preserves_every_written_byte(tmp_path) -> None:
    """Dropping pages must never lose data: a shared file-backed mapping re-reads
    from the file, and the writer msyncs before releasing."""

    import numpy as np

    from interp.activations import release_mapped_pages

    path = tmp_path / "array.dat"
    array = np.memmap(path, mode="w+", dtype=np.float16, shape=(1024, 8))
    array[:] = 3.5
    array.flush()

    release_mapped_pages(array, path)

    assert np.array_equal(np.asarray(array), np.full((1024, 8), 3.5, dtype=np.float16))
    reread = np.memmap(path, mode="r", dtype=np.float16, shape=(1024, 8))
    assert np.array_equal(np.asarray(reread), np.full((1024, 8), 3.5, dtype=np.float16))


def test_both_scan_paths_release_pages() -> None:
    """Collector and validator must both release; the validator half was the second
    instance of the same bug and is what the read-back phase needs."""

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    collector = (root / "scripts" / "collect_activations.py").read_text()
    validator = (root / "scripts" / "validate_activations.py").read_text()

    assert "release_mapped_pages(output, array_path)" in collector
    assert "release_mapped_pages(dataset.array, array_path)" in validator


def test_file_sha256_is_unchanged_by_page_release(tmp_path) -> None:
    """Hashing the 46 GiB array drove the cgroup to 53.8 GiB, so file_sha256 now drops
    its page cache as it streams. The digest must be bit-identical regardless."""

    import hashlib
    import os

    from interp.activations import _HASH_RELEASE_EVERY_BYTES, file_sha256

    path = tmp_path / "big.bin"
    path.write_bytes(os.urandom(1024) * ((_HASH_RELEASE_EVERY_BYTES * 2) // 1024 + 3))

    assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_full_artifact_scan_releases_pages() -> None:
    """Three separate paths read the whole 46 GiB artifact. Missing any one of them
    refills the page cache and pushes the cgroup back toward its limit; split_stats
    was the one overlooked after the first two were fixed."""

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    activations = (root / "src" / "interp" / "activations.py").read_text()
    collector = (root / "scripts" / "collect_activations.py").read_text()
    validator = (root / "scripts" / "validate_activations.py").read_text()

    assert "release_mapped_pages(output, array_path)" in collector, "write loop"
    assert "release_mapped_pages(dataset.array, array_path)" in validator, "rescan loop"
    assert "release_mapped_pages(dataset.array)" in activations, "split_stats loop"
    assert "_HASH_RELEASE_EVERY_BYTES" in activations, "sha256 stream"


def test_release_derives_the_path_from_the_mapping(tmp_path) -> None:
    """split_stats has no path in scope, so the helper must find it on the memmap."""

    import numpy as np

    from interp.activations import release_mapped_pages

    path = tmp_path / "derived.dat"
    array = np.memmap(path, mode="w+", dtype=np.float16, shape=(512, 8))
    array[:] = 7.0
    array.flush()

    release_mapped_pages(array)  # no path argument

    assert np.array_equal(np.asarray(array), np.full((512, 8), 7.0, dtype=np.float16))


def test_governor_only_acts_above_the_threshold_and_never_raises(tmp_path) -> None:
    """Crossing the soft limit is a signal to drop pages, not an error: the governor
    must return a plain bool and leave the caller running either way."""

    import numpy as np

    from interp.activations import PageCacheGovernor

    array = np.memmap(tmp_path / "a.dat", mode="w+", dtype=np.float16, shape=(64, 4))
    array[:] = 1.0
    array.flush()

    low = tmp_path / "low"
    low.write_text("1000\n")
    high = tmp_path / "high"
    high.write_text(str(80 * 1024**3))

    import interp.activations as module

    original = module.CGROUP_MEMORY_CURRENT
    try:
        module.CGROUP_MEMORY_CURRENT = low
        quiet = PageCacheGovernor(check_every=1)
        assert quiet.step(array) is False
        assert quiet.releases == 0

        module.CGROUP_MEMORY_CURRENT = high
        busy = PageCacheGovernor(check_every=1)
        assert busy.step(array) is True
        assert busy.releases == 1
    finally:
        module.CGROUP_MEMORY_CURRENT = original


def test_governor_checks_only_every_nth_batch(tmp_path) -> None:
    """Reading the cgroup file every step would add a syscall per batch."""

    import numpy as np

    import interp.activations as module
    from interp.activations import PageCacheGovernor

    array = np.memmap(tmp_path / "b.dat", mode="w+", dtype=np.float16, shape=(8, 4))
    counter = tmp_path / "cur"
    counter.write_text("1000\n")
    original = module.CGROUP_MEMORY_CURRENT
    try:
        module.CGROUP_MEMORY_CURRENT = counter
        governor = PageCacheGovernor(check_every=10)
        for _ in range(29):
            governor.step(array)
        assert governor.checks == 2
    finally:
        module.CGROUP_MEMORY_CURRENT = original


def test_governor_absent_cgroup_is_a_no_op(tmp_path) -> None:
    """Outside a cgroup v2 container the trainer must behave exactly as before."""

    import numpy as np

    import interp.activations as module
    from interp.activations import PageCacheGovernor, cgroup_memory_current

    array = np.memmap(tmp_path / "c.dat", mode="w+", dtype=np.float16, shape=(8, 4))
    original = module.CGROUP_MEMORY_CURRENT
    try:
        module.CGROUP_MEMORY_CURRENT = tmp_path / "does_not_exist"
        assert cgroup_memory_current(tmp_path / "does_not_exist") is None
        governor = PageCacheGovernor(check_every=1)
        assert governor.step(array) is False
    finally:
        module.CGROUP_MEMORY_CURRENT = original


def test_training_loop_governs_without_touching_the_sampler() -> None:
    """The batch indices must come from batch_rng alone. The governor call has to sit
    after the fetch and take no part in selection."""

    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src" / "interp" / "train_flow.py").read_text()
    selection = source.index("selected = split.train[")
    fetch = source.index("h = _fetch(dataset, selected, resolved_device)", selection)
    governor = source.index("page_cache.step(dataset.array)", selection)
    assert selection < fetch < governor, "governor must run after selection and fetch"
    between = source[selection:governor]
    assert "page_cache" not in between.replace("page_cache.step(dataset.array)", "")
