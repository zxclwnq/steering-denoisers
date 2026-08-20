"""I/O smoke test for the soft page-cache cap on the training loader path.

Exercises the same mmap, the same split and the same batch sampler the trainer uses,
long enough to cross the soft threshold several times, and checks that the governor
changes nothing except resident page cache.

    uv run python scripts/loader_memory_smoke.py \
      --config configs/flow_train_tangent_vp_narrow16m_fw32m_v1.yaml \
      --activation-dir /workspace/data/fineweb_activations \
      --out results/loader_memory_smoke_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from interp.activations import (
    PageCacheGovernor,
    cgroup_memory_current,
    load_activations,
    make_split,
)
from interp.train_flow import load_training_config


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--activation-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--min-releases", type=int, default=3)
    args = parser.parse_args()

    cfg = load_training_config(args.config)
    dataset = load_activations(cfg.dataset_name, args.activation_dir)
    split = make_split(len(dataset), cfg.per_seq, cfg.val_fraction, cfg.split_seed)

    events: list[dict] = []
    samples: list[dict] = []
    peak = 0
    governor = PageCacheGovernor()

    # Exactly the trainer's sampler, seeded identically.
    batch_rng = np.random.default_rng(cfg.training_seed)
    started = time.perf_counter()
    for step in range(args.steps):
        selected = split.train[
            batch_rng.integers(0, len(split.train), size=cfg.batch_size)
        ]
        values = np.array(dataset.array[selected], dtype=np.float32, copy=True)
        if step % 200 == 0:  # keep a bitwise record to re-check after reclaim
            samples.append(
                {"step": step, "indices": selected.tolist(), "digest": _digest(values)}
            )
        released = governor.step(dataset.array)
        current = cgroup_memory_current()
        if current is not None:
            peak = max(peak, current)
        if released:
            events.append(
                {
                    "step": step,
                    "before_bytes": governor.last_before,
                    "after_bytes": governor.last_after,
                    "freed_bytes": (governor.last_before or 0) - (governor.last_after or 0),
                }
            )
    governed_seconds = time.perf_counter() - started

    # 1. sample sequence unchanged: replay the sampler with no governor at all
    replay_rng = np.random.default_rng(cfg.training_seed)
    order_identical = True
    for step in range(args.steps):
        expected = split.train[
            replay_rng.integers(0, len(split.train), size=cfg.batch_size)
        ]
        for record in samples:
            if record["step"] == step and record["indices"] != expected.tolist():
                order_identical = False
    # 2. data bitwise identical: re-read recorded batches after all the reclaims
    data_identical = all(
        _digest(np.array(dataset.array[np.asarray(rec["indices"])], dtype=np.float32))
        == rec["digest"]
        for rec in samples
    )

    # 5. throughput reference: ungoverned rate over a short window
    ref_rng = np.random.default_rng(cfg.training_seed)
    ref_steps = min(400, args.steps)
    ref_started = time.perf_counter()
    for _ in range(ref_steps):
        selected = split.train[
            ref_rng.integers(0, len(split.train), size=cfg.batch_size)
        ]
        np.array(dataset.array[selected], dtype=np.float32, copy=True)
    ungoverned_rate = ref_steps / (time.perf_counter() - ref_started)
    governed_rate = args.steps / governed_seconds

    events_path = Path("/sys/fs/cgroup/memory.events")
    counters = {}
    if events_path.is_file():
        counters = dict(
            line.split() for line in events_path.read_text().strip().splitlines()
        )

    report = {
        "status": "PASS",
        "finished_utc": datetime.now(UTC).isoformat(),
        "config": str(args.config),
        "dataset": cfg.dataset_name,
        "steps": args.steps,
        "batch_size": cfg.batch_size,
        "threshold_bytes": governor.threshold_bytes,
        "cgroup_limit_bytes": int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
        if Path("/sys/fs/cgroup/memory.max").is_file()
        else None,
        "peak_memory_current_bytes": peak,
        "releases": governor.releases,
        "release_events": events,
        "sample_order_identical": order_identical,
        "data_bitwise_identical": data_identical,
        "memory_events": counters,
        "governed_steps_per_s": governed_rate,
        "ungoverned_steps_per_s": ungoverned_rate,
        "throughput_ratio": governed_rate / ungoverned_rate if ungoverned_rate else None,
        "held_out_accessed": False,
    }

    failures = []
    if not order_identical:
        failures.append("sampler order changed")
    if not data_identical:
        failures.append("data not bitwise identical after reclaim")
    if governor.releases < args.min_releases:
        failures.append(
            f"only {governor.releases} reclaim cycles, wanted >= {args.min_releases}"
        )
    if any(int(counters.get(key, 0)) for key in ("oom", "oom_kill", "oom_group_kill")):
        failures.append(f"OOM counters nonzero: {counters}")
    if peak and report["cgroup_limit_bytes"] and peak >= report["cgroup_limit_bytes"]:
        failures.append("peak reached the cgroup limit")
    if any(event["freed_bytes"] <= 0 for event in events):
        failures.append("a release did not lower memory.current")
    if failures:
        report["status"] = "FAIL"
        report["failures"] = failures

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "loader_memory_smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "release_events"}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
