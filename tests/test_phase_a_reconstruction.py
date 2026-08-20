"""Phase A reconstruction is a scientific gate, so its frozen inputs are executable."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from interp.flow_core import ActivationNormalizer
from interp.phase_a import (
    assess_phase_a_gate,
    evaluate_phase_a,
    lifecycle_record,
    load_phase_a_config,
    matched_gaussian_noise,
    paired_bootstrap_mean_ci,
    reconstruction_geometry,
    select_validation_sequence_ids,
    validate_checkpoint_payload,
    validate_frozen_checkpoint,
    write_immutable_json,
)

ROOT = Path(__file__).resolve().parents[1]


def test_approved_phase_a_config_freezes_checkpoint_data_grid_and_gate() -> None:
    cfg = load_phase_a_config(ROOT / "configs" / "flow_phase_a_100k_v1.yaml")

    assert cfg.experiment_id == "clean_flow_phase_a_100k_v1"
    assert cfg.experiment_class == "dev_method_development"
    assert cfg.checkpoint_path == Path(
        "/workspace/checkpoints/clean_flow_100k_v1_a439b2d7/best_step_099500.pt"
    )
    assert cfg.checkpoint_sha256 == (
        "9d1d3cb66b9eaab1cbc89edab121d5cfa318271d7502e2ce42230432faad30d2"
    )
    assert cfg.checkpoint_step == 99_500
    assert cfg.training_experiment_id == "clean_flow_100k_v1"
    assert cfg.run_meta_sha256 == (
        "220d63c16a1996be5badd7719916df603ba86981c3ea1b08df93de2f583bed11"
    )
    assert cfg.best_pointer_sha256 == (
        "094973b776613903a78778054b042fcfdf1839a9fb277dbe39c9d1291690c975"
    )
    assert cfg.expected_history_entries == 200
    assert cfg.selection_metric == "val_flow_mse"
    assert cfg.selection_mode == "min"
    assert cfg.dataset_name == "resid7_train_4000k_v2"
    assert cfg.activation_split == "train"
    assert cfg.internal_split == "validation_only"
    assert cfg.split_fingerprint == "c34aec678a328131"
    assert cfg.n_sequences == 256
    assert cfg.sequence_selection == "first_sorted_internal_validation_sequences"
    assert cfg.lm_batch_size == 8
    assert cfg.noise_seed == 0
    assert cfg.bootstrap_seed == 20_260_813
    assert cfg.bootstrap_resamples == 10_000
    assert cfg.t_starts == (0.10, 0.25, 0.50)
    assert cfg.nfes == (1, 3, 5)
    assert (cfg.primary_t_start, cfg.primary_nfe) == (0.50, 1)
    assert cfg.skip_bos is True
    assert cfg.steering_vectors == "forbidden"
    assert cfg.dev_directions == "forbidden"
    assert cfg.held_out == "forbidden"


def test_phase_a_loader_rejects_a_semantically_changed_frozen_config(
    tmp_path: Path,
) -> None:
    source = ROOT / "configs" / "flow_phase_a_100k_v1.yaml"
    changed = tmp_path / source.name
    changed.write_text(source.read_text().replace("noise_seed: 0", "noise_seed: 1"))

    with pytest.raises(ValueError, match="approved config SHA-256"):
        load_phase_a_config(changed)


def _selection_fixture(tmp_path: Path):
    checkpoint = tmp_path / "best_step_099500.pt"
    checkpoint.write_bytes(b"frozen-checkpoint-fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    base = load_phase_a_config(ROOT / "configs" / "flow_phase_a_100k_v1.yaml")
    cfg = replace(base, checkpoint_path=checkpoint, checkpoint_sha256=digest)
    history = [
        {"step": 99_000, "val_flow_mse": 1.0},
        {"step": 99_500, "val_flow_mse": 0.9},
        {"step": 100_000, "val_flow_mse": 0.95},
    ]
    run_meta = {
        "experiment_id": "clean_flow_100k_v1",
        "status": "complete",
        "selection_metric": "val_flow_mse",
        "best_checkpoint": checkpoint.name,
        "best_val_flow_mse": 0.9,
        "held_out_accessed": False,
        "history": history,
    }
    best = {
        "selection_metric": "val_flow_mse",
        "selection_mode": "min",
        "value": 0.9,
        "checkpoint": checkpoint.name,
    }
    run_meta_path = tmp_path / "meta.json"
    best_path = tmp_path / "best.json"
    run_meta_path.write_text(json.dumps(run_meta, indent=2) + "\n")
    best_path.write_text(json.dumps(best, indent=2) + "\n")
    cfg = replace(
        cfg,
        run_meta_path=run_meta_path,
        best_pointer_path=best_path,
        run_meta_sha256=hashlib.sha256(run_meta_path.read_bytes()).hexdigest(),
        best_pointer_sha256=hashlib.sha256(best_path.read_bytes()).hexdigest(),
        expected_history_entries=3,
    )
    return cfg, run_meta, best, checkpoint


def _rewrite_selection_sidecars(
    cfg, run_meta: dict, best: dict, *, freeze: bool = True  # noqa: ANN001
):
    cfg.run_meta_path.write_text(json.dumps(run_meta, indent=2) + "\n")
    cfg.best_pointer_path.write_text(json.dumps(best, indent=2) + "\n")
    if not freeze:
        return cfg
    return replace(
        cfg,
        run_meta_sha256=hashlib.sha256(cfg.run_meta_path.read_bytes()).hexdigest(),
        best_pointer_sha256=hashlib.sha256(cfg.best_pointer_path.read_bytes()).hexdigest(),
    )


def test_checkpoint_validation_proves_concept_independent_history_minimum(
    tmp_path: Path,
) -> None:
    cfg, run_meta, best, checkpoint = _selection_fixture(tmp_path)

    receipt = validate_frozen_checkpoint(checkpoint, cfg)

    assert receipt == {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": cfg.checkpoint_sha256,
        "checkpoint_step": 99_500,
        "run_meta_sha256": cfg.run_meta_sha256,
        "best_pointer_sha256": cfg.best_pointer_sha256,
        "run_status": "complete",
        "selection_metric": "val_flow_mse",
        "selection_mode": "min",
        "selection_value": 0.9,
        "history_entries": 3,
        "held_out_accessed": False,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("steering_metric", "concept-independent"),
        ("wrong_best_step", "checkpoint"),
        ("wrong_history_minimum", "minimum"),
        ("held_out", "held-out"),
        ("wrong_sha", "SHA-256"),
        ("incomplete", "complete"),
        ("truncated_history", "history entries"),
        ("tied_minimum", "unique"),
        ("tampered_sidecar", "sidecar SHA-256"),
    ],
)
def test_checkpoint_validation_rejects_nonfrozen_selection(
    tmp_path: Path, mutation: str, message: str
) -> None:
    cfg, run_meta, best, checkpoint = _selection_fixture(tmp_path)
    if mutation == "steering_metric":
        run_meta["selection_metric"] = "steering_score"
    elif mutation == "wrong_best_step":
        best["checkpoint"] = "best_step_100000.pt"
    elif mutation == "wrong_history_minimum":
        run_meta["history"][0]["val_flow_mse"] = 0.8
    elif mutation == "held_out":
        run_meta["held_out_accessed"] = True
    elif mutation == "wrong_sha":
        cfg = replace(cfg, checkpoint_sha256="0" * 64)
    elif mutation == "incomplete":
        run_meta["status"] = "RUNNING"
    elif mutation == "truncated_history":
        run_meta["history"] = run_meta["history"][:-1]
    elif mutation == "tied_minimum":
        run_meta["history"][2]["val_flow_mse"] = 0.9
    elif mutation == "tampered_sidecar":
        run_meta["history"][0]["train_loss"] = 123.0
    else:  # pragma: no cover - table is exhaustive
        raise AssertionError(mutation)

    if mutation not in {"wrong_sha", "tampered_sidecar"}:
        cfg = _rewrite_selection_sidecars(cfg, run_meta, best)
    elif mutation == "tampered_sidecar":
        _rewrite_selection_sidecars(cfg, run_meta, best, freeze=False)

    with pytest.raises(ValueError, match=message):
        validate_frozen_checkpoint(checkpoint, cfg)


def test_matched_noise_is_deterministic_finite_and_has_no_arm_inputs() -> None:
    sequence_ids = np.array([5, 11, 29], dtype=np.int64)

    first = matched_gaussian_noise(sequence_ids, positions=4, d_model=3, seed=7)
    second = matched_gaussian_noise(sequence_ids, positions=4, d_model=3, seed=7)

    assert first.shape == (3, 4, 3)
    assert first.dtype == torch.float32
    assert first.device.type == "cpu"
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert set(inspect.signature(matched_gaussian_noise).parameters) == {
        "sequence_ids",
        "positions",
        "d_model",
        "seed",
    }


def test_matched_noise_follows_sequence_identity_not_input_order() -> None:
    ordered_ids = np.array([5, 11, 29], dtype=np.int64)
    permuted_ids = np.array([29, 5, 11], dtype=np.int64)

    ordered = matched_gaussian_noise(ordered_ids, positions=2, d_model=4, seed=13)
    permuted = matched_gaussian_noise(permuted_ids, positions=2, d_model=4, seed=13)

    assert torch.equal(permuted[0], ordered[2])
    assert torch.equal(permuted[1], ordered[0])
    assert torch.equal(permuted[2], ordered[1])


@pytest.mark.parametrize(
    "sequence_ids",
    [
        np.array([1, 1]),
        np.array([-1, 2]),
        np.array([[1, 2]]),
        np.array([1.5, 2.5]),
        np.array([], dtype=np.int64),
    ],
)
def test_matched_noise_rejects_ambiguous_sequence_identity(
    sequence_ids: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="sequence_ids"):
        matched_gaussian_noise(sequence_ids, positions=2, d_model=3, seed=0)


def test_paired_bootstrap_resamples_sequence_effects_and_is_deterministic() -> None:
    effects = np.full(8, -0.25, dtype=np.float64)

    first = paired_bootstrap_mean_ci(effects, seed=17, n_resamples=100, confidence=0.95)
    second = paired_bootstrap_mean_ci(effects, seed=17, n_resamples=100, confidence=0.95)

    assert first == second
    assert first == {
        "mean": -0.25,
        "ci_lower": -0.25,
        "ci_upper": -0.25,
        "confidence": 0.95,
        "n_units": 8,
        "n_resamples": 100,
        "unit": "validation_sequence",
    }


def test_reconstruction_geometry_matches_hand_calculated_vectors() -> None:
    clean = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
    reconstructed = torch.tensor([[0.0, 4.0], [1.0, 0.0]])

    got = reconstruction_geometry(clean, reconstructed)

    assert got["mean_relative_l2"] == pytest.approx((3.0 / 5.0) / 2.0)
    assert got["mean_cosine"] == pytest.approx((0.8 + 1.0) / 2.0)
    assert got["n_activations"] == 2


def _passing_gate_inputs() -> dict:
    primary_effects = np.full(8, -0.25, dtype=np.float64)
    bootstrap = paired_bootstrap_mean_ci(
        primary_effects, seed=17, n_resamples=100, confidence=0.95
    )
    return {
        "primary_effects": primary_effects,
        "bootstrap": bootstrap,
        "identity_deltas": np.zeros(8, dtype=np.float64),
        "identity_flow_evaluations": 0,
        "corruption_deltas": {
            0.10: np.array([0.01, 0.02]),
            0.25: np.array([0.1, 0.2]),
            0.50: np.array([1.0, 1.2]),
        },
        "all_functional_values": {
            "diagnostic_huge_positive": np.array([10_000.0]),
            "diagnostic_huge_negative": np.array([-10_000.0]),
        },
        "all_geometry": {
            "t0.50_nfe1": {"mean_relative_l2": 0.2, "mean_cosine": 0.9}
        },
    }


def test_gate_passes_only_from_primary_paired_effect_and_safety_controls() -> None:
    result = assess_phase_a_gate(**_passing_gate_inputs())

    assert result["status"] == "PASS"
    assert result["research_status"] == "SUPPORTED"
    assert all(result["criteria"].values())
    assert result["primary_effect_definition"] == (
        "delta_lm_reconstructed_minus_delta_lm_corrupted"
    )


@pytest.mark.parametrize(
    ("mutation", "failed_criterion"),
    [
        ("positive_mean", "primary_mean_below_zero"),
        ("ci_crosses_zero", "primary_ci_upper_below_zero"),
        ("near_zero_identity", "identity_delta_exact_zero"),
        ("identity_forwards", "identity_zero_flow_evaluations"),
        ("nonpositive_corruption", "positive_aggregate_corruption_each_t_start"),
        ("nonfinite_functional", "finite_outputs_and_geometry"),
        ("nonfinite_geometry", "finite_outputs_and_geometry"),
    ],
)
def test_gate_fails_without_redefining_the_primary_test(
    mutation: str, failed_criterion: str
) -> None:
    inputs = _passing_gate_inputs()
    if mutation == "positive_mean":
        inputs["primary_effects"] = np.full(8, 0.25)
        inputs["bootstrap"] = paired_bootstrap_mean_ci(
            inputs["primary_effects"], seed=17, n_resamples=100
        )
    elif mutation == "ci_crosses_zero":
        inputs["bootstrap"] = {**inputs["bootstrap"], "ci_upper": 0.01}
    elif mutation == "near_zero_identity":
        inputs["identity_deltas"][0] = 1e-15
    elif mutation == "identity_forwards":
        inputs["identity_flow_evaluations"] = 1
    elif mutation == "nonpositive_corruption":
        inputs["corruption_deltas"][0.25] = np.array([-0.2, 0.1])
    elif mutation == "nonfinite_functional":
        inputs["all_functional_values"]["diagnostic_huge_positive"][0] = np.nan
    elif mutation == "nonfinite_geometry":
        inputs["all_geometry"]["t0.50_nfe1"]["mean_cosine"] = np.inf

    result = assess_phase_a_gate(**inputs)

    assert result["status"] == "FAIL"
    assert result["research_status"] == "NOT_SUPPORTED"
    assert result["criteria"][failed_criterion] is False


def test_bootstrap_rejects_token_shaped_or_nonfinite_effects() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        paired_bootstrap_mean_ci(np.zeros((2, 3)), seed=1, n_resamples=10)
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap_mean_ci(np.array([0.0, np.nan]), seed=1, n_resamples=10)


class TinyHookedLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(device="cpu")
        self.embedding = nn.Embedding(7, 3)
        self.unembed = nn.Linear(3, 7, bias=False)
        self._hooks = []
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0],
                    ]
                )
            )
            self.unembed.weight.copy_(self.embedding.weight)

    @contextmanager
    def hooks(self, *, fwd_hooks):  # noqa: ANN001
        previous = self._hooks
        self._hooks = list(fwd_hooks)
        try:
            yield self
        finally:
            self._hooks = previous

    def forward(self, tokens: torch.Tensor, *, return_type: str) -> torch.Tensor:
        assert return_type == "logits"
        activation = self.embedding(tokens)
        for _, hook_fn in self._hooks:
            activation = hook_fn(activation, None)
        return self.unembed(activation)


class RecordingZeroFlow:
    def __init__(self) -> None:
        self.normalizer = ActivationNormalizer(torch.zeros(3), torch.ones(3), eps=0.0)
        self.states: list[torch.Tensor] = []
        self.times: list[torch.Tensor] = []

    def __call__(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        self.states.append(state.detach().clone())
        self.times.append(time.detach().clone())
        return torch.zeros_like(state)


EVAL_TOKENS = torch.tensor(
    [
        [0, 1, 2, 3, 4],
        [0, 2, 3, 4, 5],
        [0, 3, 4, 5, 6],
    ],
    dtype=torch.long,
)


def test_evaluator_reuses_one_noise_tensor_and_counts_exact_flow_evaluations() -> None:
    cfg = replace(
        load_phase_a_config(ROOT / "configs" / "flow_phase_a_100k_v1.yaml"),
        n_sequences=3,
        lm_batch_size=2,
        bootstrap_resamples=100,
    )
    flow = RecordingZeroFlow()
    language_model = TinyHookedLM()
    sequence_ids = np.array([2, 7, 9], dtype=np.int64)

    report = evaluate_phase_a(flow, language_model, EVAL_TOKENS, sequence_ids, cfg)

    assert report["validation_sequence_ids"] == [2, 7, 9]
    assert report["noise"]["reused_across_t_start"] is True
    assert report["noise"]["reused_across_nfe"] is True
    assert len(report["noise"]["sha256"]) == 64
    assert report["identity"]["flow_evaluations"] == 0
    assert report["identity"]["mean_delta_lm"] == 0.0
    assert report["identity"]["delta_lm_per_sequence"] == [0.0, 0.0, 0.0]
    for t_start in cfg.t_starts:
        for nfe in cfg.nfes:
            assert (
                report["cells"][f"t{t_start:.2f}_nfe{nfe}"]["flow_evaluations"]
                == 2 * nfe
            )
    assert len(flow.states) == 54

    clean = language_model.embedding(EVAL_TOKENS[:2])[:, 1:, :].reshape(-1, 3)
    first_state_indices = [0, 2, 8, 18, 20, 26, 36, 38, 44]
    recovered_noises = []
    for state_index, t_start in zip(
        first_state_indices,
        [0.10, 0.10, 0.10, 0.25, 0.25, 0.25, 0.50, 0.50, 0.50],
        strict=True,
    ):
        recovered_noises.append((flow.states[state_index] - (1.0 - t_start) * clean) / t_start)
    assert all(
        torch.allclose(recovered_noises[0], value, rtol=1e-5, atol=1e-6)
        for value in recovered_noises[1:]
    )


def test_evaluator_primary_effect_is_reconstructed_minus_corrupted_per_sequence() -> None:
    cfg = replace(
        load_phase_a_config(ROOT / "configs" / "flow_phase_a_100k_v1.yaml"),
        n_sequences=3,
        lm_batch_size=3,
        bootstrap_resamples=100,
    )

    report = evaluate_phase_a(
        RecordingZeroFlow(), TinyHookedLM(), EVAL_TOKENS, np.array([2, 7, 9]), cfg
    )

    primary = report["cells"]["t0.50_nfe1"]["delta_lm_per_sequence"]
    corrupted = report["corruptions"]["t0.50"]["delta_lm_per_sequence"]
    expected = np.asarray(primary) - np.asarray(corrupted)
    assert report["gate"]["primary_effect"]["values"] == expected.tolist()
    assert report["gate"]["primary_effect"]["definition"] == (
        "delta_lm_reconstructed_minus_delta_lm_corrupted"
    )
    assert report["gate"]["status"] == "FAIL"


def test_immutable_json_writer_never_reuses_an_output_path(tmp_path: Path) -> None:
    output = tmp_path / "phase_a.json"

    write_immutable_json(output, {"status": "complete", "gate": "FAIL"})

    assert output.read_text().endswith("\n")
    with pytest.raises(FileExistsError, match="overwrite"):
        write_immutable_json(output, {"status": "complete", "gate": "PASS"})


def test_immutable_json_writer_does_not_leave_a_partial_final_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "phase_a.json"

    def fail_link(source, destination):  # noqa: ANN001, ARG001
        raise OSError("simulated atomic-publish failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="simulated"):
        write_immutable_json(output, {"status": "complete"})

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_completed_lifecycle_record_has_exact_command_and_timestamps() -> None:
    command = ["/workspace/project/.venv/bin/python", "scripts/eval_flow_reconstruction.py"]

    record = lifecycle_record(
        status="complete",
        command=command,
        started_utc="2026-08-13T17:00:00+00:00",
        finished_utc="2026-08-13T17:10:00+00:00",
    )

    assert record == {
        "status": "complete",
        "termination_reason": "completed",
        "command": command,
        "command_shell": (
            "/workspace/project/.venv/bin/python scripts/eval_flow_reconstruction.py"
        ),
        "started_utc": "2026-08-13T17:00:00+00:00",
        "finished_utc": "2026-08-13T17:10:00+00:00",
    }


def test_sequence_selection_uses_only_sorted_document_level_validation_ids() -> None:
    from interp.activations import make_split

    split = make_split(40, per_seq=2, val_fraction=0.25, seed=19)
    cfg = replace(
        load_phase_a_config(ROOT / "configs" / "flow_phase_a_100k_v1.yaml"),
        per_seq=2,
        val_fraction=0.25,
        split_seed=19,
        split_fingerprint=split.fingerprint(),
        n_sequences=3,
    )

    selected = select_validation_sequence_ids(40, cfg)

    validation_ids = np.unique(split.val // 2)
    training_ids = np.unique(split.train // 2)
    assert selected.tolist() == validation_ids[:3].tolist()
    assert np.intersect1d(selected, training_ids).size == 0


def _checkpoint_payload(cfg) -> dict:  # noqa: ANN001
    return {
        "step": cfg.checkpoint_step,
        "dataset": cfg.dataset_name,
        "split_fingerprint": cfg.split_fingerprint,
        "dataset_artifact_identity": {
            "sha256": cfg.artifact_sha256,
            "token_cache_sha256": cfg.token_cache_sha256,
        },
    }


def test_checkpoint_payload_is_bound_to_the_frozen_activation_artifact() -> None:
    cfg = load_phase_a_config(ROOT / "configs" / "flow_phase_a_100k_v1.yaml")

    receipt = validate_checkpoint_payload(_checkpoint_payload(cfg), cfg)

    assert receipt["step"] == 99_500
    assert receipt["dataset"] == "resid7_train_4000k_v2"
    assert receipt["dataset_artifact_identity"]["sha256"] == cfg.artifact_sha256


@pytest.mark.parametrize("field", ["step", "dataset", "split_fingerprint", "artifact"])
def test_checkpoint_payload_rejects_substituted_scientific_identity(field: str) -> None:
    cfg = load_phase_a_config(ROOT / "configs" / "flow_phase_a_100k_v1.yaml")
    payload = _checkpoint_payload(cfg)
    if field == "step":
        payload["step"] = 100_000
    elif field == "dataset":
        payload["dataset"] = "resid7_dev_262k_v2"
    elif field == "split_fingerprint":
        payload["split_fingerprint"] = "0" * 16
    elif field == "artifact":
        payload["dataset_artifact_identity"] = {
            **payload["dataset_artifact_identity"],
            "sha256": {**cfg.artifact_sha256, "array": "0" * 64},
        }

    with pytest.raises(ValueError, match="checkpoint payload"):
        validate_checkpoint_payload(payload, cfg)
