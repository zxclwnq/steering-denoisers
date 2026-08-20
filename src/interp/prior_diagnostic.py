"""Pure contracts and summaries for the concept-independent flow-prior diagnostic."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

APPROVED_PRIOR_DIAGNOSTIC_CONFIG_SHA256 = (
    "fdca9001117c7df9038e6e740c78712cdbe3e55b4fbc959db3ed4f820231c53d"
)
_CELL = re.compile(r"^t(?P<t>\d+\.\d+)_nfe(?P<nfe>\d+)$")


@dataclass(frozen=True)
class DiagnosticCheckpoint:
    step: int
    path: Path
    sha256: str


@dataclass(frozen=True)
class PriorDiagnosticConfig:
    experiment_id: str
    parent_phase_a_path: Path
    parent_phase_a_sha256: str
    run_meta_path: Path
    run_meta_sha256: str
    best_pointer_path: Path
    best_pointer_sha256: str
    expected_history_entries: int
    checkpoints: tuple[DiagnosticCheckpoint, ...]
    t_starts: tuple[float, ...]
    checkpoint_nfes: tuple[int, ...]
    nfe_sweep_step: int
    nfe_sweep: tuple[int, ...]
    protected_data: dict[str, str]
    output_root: Path
    wide_capacity_design: dict[str, int | str]
    raw: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_exact_file_sha256(path: Path, expected: str) -> None:
    """Fail loudly unless one immutable artifact has the expected full digest."""

    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{path} SHA256 {observed} != frozen {expected}")


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a full SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value


def load_prior_diagnostic_config(path: Path) -> PriorDiagnosticConfig:
    """Load the single preregistered diagnostic config and reject semantic drift."""

    observed = _sha256(path)
    if observed != APPROVED_PRIOR_DIAGNOSTIC_CONFIG_SHA256:
        raise ValueError(
            f"approved diagnostic config SHA256 is "
            f"{APPROVED_PRIOR_DIAGNOSTIC_CONFIG_SHA256}, got {observed}"
        )
    raw = yaml.safe_load(path.read_text())
    if (
        raw.get("version") != 1
        or raw.get("status") != "frozen_before_diagnostic_run"
        or raw.get("experiment_class") != "concept_independent_dev_diagnostic"
    ):
        raise ValueError("diagnostic config identity changed")

    parent = raw["parent_phase_a"]
    run = raw["training_run"]
    grid = raw["checkpoint_grid"]
    evaluation = raw["evaluation"]
    checkpoints = tuple(
        DiagnosticCheckpoint(
            step=int(item["step"]),
            path=Path(item["path"]),
            sha256=_require_sha256(item["sha256"], f"checkpoint[{index}].sha256"),
        )
        for index, item in enumerate(grid["checkpoints"])
    )
    if tuple(item.step for item in checkpoints) != (10_000, 30_000, 50_000, 70_000, 99_500):
        raise ValueError("checkpoint grid changed")
    protected = dict(raw["protected_data"])
    if not protected or set(protected.values()) != {"forbidden"}:
        raise ValueError("protected-data contract changed")

    return PriorDiagnosticConfig(
        experiment_id=str(raw["experiment_id"]),
        parent_phase_a_path=Path(parent["path"]),
        parent_phase_a_sha256=_require_sha256(parent["sha256"], "parent_phase_a.sha256"),
        run_meta_path=Path(run["meta_path"]),
        run_meta_sha256=_require_sha256(run["meta_sha256"], "training_run.meta_sha256"),
        best_pointer_path=Path(run["best_pointer_path"]),
        best_pointer_sha256=_require_sha256(
            run["best_pointer_sha256"], "training_run.best_pointer_sha256"
        ),
        expected_history_entries=int(run["expected_history_entries"]),
        checkpoints=checkpoints,
        t_starts=tuple(float(value) for value in evaluation["t_start"]),
        checkpoint_nfes=tuple(int(value) for value in evaluation["checkpoint_nfe"]),
        nfe_sweep_step=int(evaluation["nfe_sweep_checkpoint_step"]),
        nfe_sweep=tuple(int(value) for value in evaluation["nfe_sweep"]),
        protected_data=protected,
        output_root=Path(raw["output"]["root"]),
        wide_capacity_design=dict(raw["wide_capacity_design"]),
        raw=raw,
    )


def summarize_phase_a(report: dict) -> dict[str, dict[str, float | int]]:
    """Summarize functional and geometric recovery for every Phase-A cell."""

    result: dict[str, dict[str, float | int]] = {}
    for key, cell in report["cells"].items():
        match = _CELL.fullmatch(key)
        if match is None:
            raise ValueError(f"invalid Phase-A cell key {key!r}")
        t_start = float(match.group("t"))
        nfe = int(match.group("nfe"))
        corruption = report["corruptions"][f"t{t_start:.2f}"]
        corrupted_delta = float(corruption["mean_delta_lm"])
        reconstructed_delta = float(cell["mean_delta_lm"])
        recovered = corrupted_delta - reconstructed_delta
        if not math.isfinite(corrupted_delta) or corrupted_delta <= 0:
            raise ValueError("aggregate corruption delta-LM must be finite and positive")
        geometry = cell["geometry"]
        values: dict[str, float | int] = {
            "t_start": t_start,
            "nfe": nfe,
            "corruption_delta_lm": corrupted_delta,
            "reconstructed_delta_lm": reconstructed_delta,
            "recovered_damage": recovered,
            "recovered_fraction": recovered / corrupted_delta,
            "mean_relative_l2": float(geometry["mean_relative_l2"]),
            "mean_cosine": float(geometry["mean_cosine"]),
            "n_activations": int(geometry["n_activations"]),
        }
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("Phase-A summary became non-finite")
        result[key] = values
    return result


def nfe_marginals(report: dict) -> dict[str, dict[str, dict[str, float]]]:
    """Return positive-oriented improvements between adjacent NFE settings."""

    summaries = summarize_phase_a(report)
    grouped: dict[float, dict[int, dict[str, float | int]]] = {}
    for item in summaries.values():
        grouped.setdefault(float(item["t_start"]), {})[int(item["nfe"])] = item

    result: dict[str, dict[str, dict[str, float]]] = {}
    for t_start, by_nfe in sorted(grouped.items()):
        transitions: dict[str, dict[str, float]] = {}
        nfes = sorted(by_nfe)
        for old_nfe, new_nfe in zip(nfes, nfes[1:], strict=False):
            old = by_nfe[old_nfe]
            new = by_nfe[new_nfe]
            transitions[f"{old_nfe}->{new_nfe}"] = {
                "delta_lm_reduction": float(old["reconstructed_delta_lm"])
                - float(new["reconstructed_delta_lm"]),
                "recovered_damage_gain": float(new["recovered_damage"])
                - float(old["recovered_damage"]),
                "relative_l2_reduction": float(old["mean_relative_l2"])
                - float(new["mean_relative_l2"]),
                "cosine_gain": float(new["mean_cosine"]) - float(old["mean_cosine"]),
            }
        result[f"t{t_start:.2f}"] = transitions
    return result


def wide_glp_parameter_count(
    *,
    activation_dim: int,
    d_model: int,
    d_mlp: int,
    n_blocks: int,
    time_dim: int,
    time_hidden: int,
) -> int:
    """Count the proposed 768-in/out, wider internal GLP exactly."""

    if min(activation_dim, d_model, d_mlp, n_blocks, time_dim, time_hidden) <= 0:
        raise ValueError("all model dimensions must be positive")
    input_projection = activation_dim * d_model + d_model
    time_embedding = time_dim * time_hidden + time_hidden
    time_embedding += time_hidden * d_model + d_model
    block = 2 * d_model
    block += 3 * (d_model * d_mlp + d_mlp)
    block += d_mlp * d_model + d_model
    output_norm = 2 * d_model
    output_projection = d_model * activation_dim + activation_dim
    return (
        input_projection
        + time_embedding
        + n_blocks * block
        + output_norm
        + output_projection
    )
