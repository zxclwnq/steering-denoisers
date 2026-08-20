"""Provenance gates the T1/T2 evaluators must pass before a formal verdict.

Each test here corresponds to one adversarial-review finding: a claim the
result metadata used to make that nothing actually checked.

Everything is synthetic and on disk in tmp_path. No real checkpoint, artifact,
or protected file is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from interp.activations import DTYPE, file_sha256, make_split
from interp.conditional_flow import (
    MIN_TRAINING_RANK,
    load_training_direction_pool,
    save_direction_pool,
)
from interp.model import MODEL_NAME, MODEL_RESOLVED_NAME, MODEL_REVISION, STEERING_HOOK
from interp.tangent_eval import (
    TANGENT_RECONSTRUCTION_SPEC,
    VP_TANGENT_RECONSTRUCTION_SPEC,
    load_validated_evaluation_bundle,
    require_fresh_output_dir,
    unselected_checkpoint_receipt,
    verify_direction_pool,
    verify_selected_checkpoint,
    verify_t1_pass_receipt,
    write_t1_receipt,
)
from interp.tangent_flow import (
    ISOTROPIC_OBJECTIVE,
    TANGENT_OBJECTIVE,
    VP_TANGENT_OBJECTIVE,
)

PER_SEQ = 7
CTX = PER_SEQ + 1
N_SEQS = 40
D_MODEL = 4
VAL_FRACTION = 0.25
SPLIT_SEED = 11

OBJECTIVE_IDENTITY = {
    "flow_objective": TANGENT_OBJECTIVE,
    "condition_type": "direction_coordinate_film",
    "tangent_output_projection": True,
    "normalizer": {"width": D_MODEL, "eps": 1e-5, "digest": "n" * 64},
}


# --------------------------------------------------------------------------
# P2: checkpoint-selection provenance
# --------------------------------------------------------------------------


def _run_dir(tmp_path: Path, *, best: str = "best_step_000100.pt") -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / best).write_bytes(b"synthetic checkpoint bytes")
    (run / "best.json").write_text(
        json.dumps(
            {
                "selection_metric": "val_flow_mse",
                "selection_mode": "min",
                "value": 0.42,
                "checkpoint": best,
            }
        )
    )
    (run / "meta.json").write_text(
        json.dumps(
            {
                "experiment_id": "tangent_run",
                "config_fingerprint": "f" * 64,
                "source_revision": "snapshot-sha256:" + "0" * 64,
                "selection_metric": "val_flow_mse",
                "best_checkpoint": best,
            }
        )
    )
    return run


def _checkpoint_meta(**overrides: object) -> dict:
    meta = {
        "experiment_id": "tangent_run",
        "config_fingerprint": "f" * 64,
        "step": 100,
        "objective_identity": OBJECTIVE_IDENTITY,
    }
    meta.update(overrides)
    return meta


def test_the_run_selected_checkpoint_is_accepted(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    receipt = verify_selected_checkpoint(
        run / "best_step_000100.pt", _checkpoint_meta(), run_dir=run
    )
    assert receipt["verified"] is True
    assert receipt["selection_metric"] == "val_flow_mse"
    assert receipt["selection_is_concept_independent"] is True
    assert receipt["checkpoint_sha256"] == file_sha256(run / "best_step_000100.pt")


def test_an_arbitrary_non_selected_checkpoint_is_rejected(tmp_path: Path) -> None:
    """P2 regression: a CLI path is an assertion, not evidence."""

    run = _run_dir(tmp_path)
    other = run / "step_000050.pt"
    other.write_bytes(b"a different checkpoint")

    with pytest.raises(ValueError, match="not the run's selected checkpoint"):
        verify_selected_checkpoint(other, _checkpoint_meta(), run_dir=run)


def test_a_checkpoint_from_another_run_is_rejected(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    with pytest.raises(ValueError, match="!= run"):
        verify_selected_checkpoint(
            run / "best_step_000100.pt",
            _checkpoint_meta(config_fingerprint="e" * 64),
            run_dir=run,
        )


def test_a_run_selected_on_another_metric_is_rejected(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    best = json.loads((run / "best.json").read_text())
    best["selection_metric"] = "steering_delta_nll"
    (run / "best.json").write_text(json.dumps(best))

    with pytest.raises(ValueError, match="concept-independence claim does not hold"):
        verify_selected_checkpoint(
            run / "best_step_000100.pt", _checkpoint_meta(), run_dir=run
        )


def test_a_bare_checkpoint_without_a_run_directory_is_rejected(tmp_path: Path) -> None:
    lonely = tmp_path / "somewhere" / "best_step_000100.pt"
    lonely.parent.mkdir()
    lonely.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="cannot be issued from a bare checkpoint"):
        verify_selected_checkpoint(
            lonely, _checkpoint_meta(), run_dir=tmp_path / "somewhere"
        )


def test_an_isotropic_checkpoint_cannot_pass_the_tangent_selection_check(
    tmp_path: Path,
) -> None:
    run = _run_dir(tmp_path)
    isotropic = {**OBJECTIVE_IDENTITY, "flow_objective": ISOTROPIC_OBJECTIVE}
    with pytest.raises(ValueError, match="checkpoint objective"):
        verify_selected_checkpoint(
            run / "best_step_000100.pt",
            _checkpoint_meta(objective_identity=isotropic),
            run_dir=run,
        )


def test_unselected_receipt_is_marked_ineligible(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    receipt = unselected_checkpoint_receipt(
        run / "best_step_000100.pt", _checkpoint_meta(), "operator override"
    )
    assert receipt["verified"] is False
    assert receipt["formal_verdict_eligible"] is False
    assert receipt["selection_is_concept_independent"] is False


# --------------------------------------------------------------------------
# P3: direction-pool identity
# --------------------------------------------------------------------------


def _pool_file(path: Path, seed: int, *, rows: int = 6, width: int = D_MODEL) -> Path:
    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn(rows, width, generator=generator)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    save_direction_pool(
        path,
        directions,
        ranks=tuple(range(MIN_TRAINING_RANK, MIN_TRAINING_RANK + rows)),
        source="synthetic_test_fixture",
        selection="blake2b_priority_rank",
        selection_seed=20260807,
    )
    return path


def test_the_training_pool_is_accepted(tmp_path: Path) -> None:
    pool = load_training_direction_pool(_pool_file(tmp_path / "pool.pt", seed=1))
    meta = _checkpoint_meta(direction_pool=pool.identity())
    receipt = verify_direction_pool(meta, pool)
    assert receipt["verified"] is True
    assert receipt["digest"] == pool.identity()["digest"]


def test_a_same_shaped_pool_with_a_different_digest_is_rejected(tmp_path: Path) -> None:
    """P3 regression: same shape is not same pool."""

    trained = load_training_direction_pool(_pool_file(tmp_path / "a.pt", seed=1))
    other = load_training_direction_pool(_pool_file(tmp_path / "b.pt", seed=2))
    assert trained.directions.shape == other.directions.shape
    assert trained.identity()["digest"] != other.identity()["digest"]

    meta = _checkpoint_meta(direction_pool=trained.identity())
    with pytest.raises(ValueError, match="does not match the pool the checkpoint"):
        verify_direction_pool(meta, other)


def test_a_checkpoint_without_a_recorded_pool_is_rejected(tmp_path: Path) -> None:
    pool = load_training_direction_pool(_pool_file(tmp_path / "pool.pt", seed=1))
    with pytest.raises(ValueError, match="records no direction pool"):
        verify_direction_pool(_checkpoint_meta(), pool)


# --------------------------------------------------------------------------
# P4: activation / token artifact bundle
# --------------------------------------------------------------------------


def _artifact(tmp_path: Path, *, token_seed: int = 0) -> tuple[Path, Path, str]:
    """A complete, self-consistent synthetic validation artifact plus token cache."""

    activation_dir = tmp_path / "activations"
    cache_dir = tmp_path / "tokens"
    activation_dir.mkdir()
    cache_dir.mkdir()
    name = "synthetic_val_v1"

    rows = N_SEQS * PER_SEQ
    array = np.random.default_rng(0).normal(size=(rows, D_MODEL)).astype(DTYPE)
    np.save(activation_dir / f"{name}.npy", array)
    np.savez(
        activation_dir / f"{name}_stats.npz",
        mean=array.astype(np.float32).mean(0),
        std=array.astype(np.float32).std(0) + 1.0,
    )

    tokens = np.random.default_rng(token_seed).integers(
        0, 50000, size=(N_SEQS, CTX), dtype=np.int64
    )
    cache_file = f"val_{N_SEQS}x{CTX}.npy"
    np.save(cache_dir / cache_file, tokens)
    cache_sha = file_sha256(cache_dir / cache_file)

    meta = {
        "name": name,
        "status": "complete",
        "split": "val",
        "hook": STEERING_HOOK,
        "model": MODEL_NAME,
        "resolved_model_name": MODEL_RESOLVED_NAME,
        "model_revision": MODEL_REVISION,
        "compute_dtype": "float32",
        "storage_dtype": DTYPE.__name__,
        "shape": list(array.shape),
        "ctx": CTX,
        "bos_dropped": True,
        "n_seqs": N_SEQS,
        "steering_vectors_used": None,
        "dataset_repository": "HuggingFaceFW/fineweb",
        "dataset_config": "sample-10BT",
        "dataset_revision": "9bb295ddab0e05d785b879661af7260fed5140fc",
        "tokenizer": "gpt2",
        "token_cache_file": cache_file,
        "token_cache_sha256": cache_sha,
    }
    (activation_dir / f"{name}.json").write_text(json.dumps(meta))

    split = make_split(rows, PER_SEQ, VAL_FRACTION, SPLIT_SEED)
    (activation_dir / f"{name}_validation.json").write_text(
        json.dumps(
            {
                "status": "VALID",
                "name": name,
                "split_fingerprint": split.fingerprint(),
                "token_cache_sha256": cache_sha,
                "sha256": {
                    "array": file_sha256(activation_dir / f"{name}.npy"),
                    "metadata": file_sha256(activation_dir / f"{name}.json"),
                    "statistics": file_sha256(activation_dir / f"{name}_stats.npz"),
                },
            }
        )
    )
    return activation_dir, cache_dir, name


def _load(activation_dir: Path, cache_dir: Path, name: str):  # noqa: ANN202
    return load_validated_evaluation_bundle(
        name, activation_dir, cache_dir,
        per_seq=PER_SEQ, val_fraction=VAL_FRACTION, split_seed=SPLIT_SEED,
        d_model=D_MODEL,
    )


def test_a_consistent_bundle_loads_and_records_its_identity(tmp_path: Path) -> None:
    bundle = _load(*_artifact(tmp_path))
    assert bundle.tokens.shape == (N_SEQS, CTX)
    assert len(bundle.dataset) == N_SEQS * PER_SEQ
    identity = bundle.identity
    assert identity["model_revision"] == MODEL_REVISION
    assert identity["hook"] == STEERING_HOOK
    assert identity["bos_dropped"] is True
    assert identity["ctx"] == CTX
    assert identity["n_seqs"] == N_SEQS
    assert identity["validation_report_status"] == "VALID"
    assert len(identity["token_cache_sha256"]) == 64


def test_a_same_shape_but_different_token_cache_is_rejected(tmp_path: Path) -> None:
    """P4 regression: shape compatibility must not be enough to pair files."""

    activation_dir, cache_dir, name = _artifact(tmp_path)
    meta = json.loads((activation_dir / f"{name}.json").read_text())
    # Same shape, same dtype, different content: a plausible wrong pairing.
    impostor = np.random.default_rng(999).integers(
        0, 50000, size=(N_SEQS, CTX), dtype=np.int64
    )
    np.save(cache_dir / meta["token_cache_file"], impostor)

    with pytest.raises(ValueError, match="do not belong together"):
        _load(activation_dir, cache_dir, name)


def test_a_tampered_activation_array_is_rejected(tmp_path: Path) -> None:
    activation_dir, cache_dir, name = _artifact(tmp_path)
    array = np.load(activation_dir / f"{name}.npy")
    array[0, 0] = 12.5
    np.save(activation_dir / f"{name}.npy", array)

    with pytest.raises(ValueError, match="hash mismatch"):
        _load(activation_dir, cache_dir, name)


def test_a_wrong_split_is_rejected(tmp_path: Path) -> None:
    activation_dir, cache_dir, name = _artifact(tmp_path)
    meta = json.loads((activation_dir / f"{name}.json").read_text())
    meta["split"] = "train"
    (activation_dir / f"{name}.json").write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="artifact split"):
        _load(activation_dir, cache_dir, name)


def test_a_missing_token_cache_is_rejected(tmp_path: Path) -> None:
    activation_dir, cache_dir, name = _artifact(tmp_path)
    meta = json.loads((activation_dir / f"{name}.json").read_text())
    (cache_dir / meta["token_cache_file"]).unlink()

    with pytest.raises(FileNotFoundError, match="token cache does not exist"):
        _load(activation_dir, cache_dir, name)


def test_a_mismatched_activation_width_is_rejected(tmp_path: Path) -> None:
    activation_dir, cache_dir, name = _artifact(tmp_path)
    with pytest.raises(ValueError, match="d_model"):
        load_validated_evaluation_bundle(
            name, activation_dir, cache_dir,
            per_seq=PER_SEQ, val_fraction=VAL_FRACTION, split_seed=SPLIT_SEED,
            d_model=D_MODEL + 1,
        )


def test_a_wrong_hook_is_rejected(tmp_path: Path) -> None:
    activation_dir, cache_dir, name = _artifact(tmp_path)
    with pytest.raises(ValueError, match="hook"):
        load_validated_evaluation_bundle(
            name, activation_dir, cache_dir, hook="blocks.3.hook_resid_pre",
            per_seq=PER_SEQ, val_fraction=VAL_FRACTION, split_seed=SPLIT_SEED,
            d_model=D_MODEL,
        )


# --------------------------------------------------------------------------
# P8: T2 requires a formal T1 PASS receipt
# --------------------------------------------------------------------------


def _t1_payload(verdict: str = "PASS", *, eligible: bool = True) -> dict:
    return {
        "experiment": "tangent_reconstruction_t1_v1",
        "source_revision": "snapshot-sha256:" + "0" * 64,
        "t1_gate": {
            "verdict": verdict,
            "primary_cell": TANGENT_RECONSTRUCTION_SPEC.primary_cell(),
            "primary_t_start": TANGENT_RECONSTRUCTION_SPEC.primary_t_start,
            "primary_nfe": TANGENT_RECONSTRUCTION_SPEC.primary_nfe,
            "formal_verdict_eligible": eligible,
        },
        "checkpoint_selection": {
            "checkpoint_sha256": "c" * 64,
            "objective_identity": OBJECTIVE_IDENTITY,
            "config_fingerprint": "f" * 64,
        },
        "direction_pool": {"digest": "d" * 64},
    }


def test_a_passing_receipt_authorizes_t2(tmp_path: Path) -> None:
    path = tmp_path / "t1_receipt.json"
    write_t1_receipt(path, _t1_payload())
    verified = verify_t1_pass_receipt(
        path,
        checkpoint_sha256="c" * 64,
        pool_identity={"digest": "d" * 64},
        objective_identity=OBJECTIVE_IDENTITY,
    )
    assert verified["verified"] is True
    assert verified["verdict"] == "PASS"


def test_t2_refuses_without_any_receipt(tmp_path: Path) -> None:
    """P8 regression: T2 has no standing without a formal T1 PASS."""

    with pytest.raises(FileNotFoundError, match="requires a formal T1 PASS receipt"):
        verify_t1_pass_receipt(
            tmp_path / "absent.json",
            checkpoint_sha256="c" * 64,
            pool_identity={"digest": "d" * 64},
            objective_identity=OBJECTIVE_IDENTITY,
        )


def test_t2_refuses_on_a_failing_t1(tmp_path: Path) -> None:
    path = tmp_path / "t1_receipt.json"
    write_t1_receipt(path, _t1_payload("FAIL"))
    with pytest.raises(ValueError, match="not PASS; the protocol stops"):
        verify_t1_pass_receipt(
            path, checkpoint_sha256="c" * 64,
            pool_identity={"digest": "d" * 64},
            objective_identity=OBJECTIVE_IDENTITY,
        )


def test_t2_refuses_on_a_diagnostic_only_t1(tmp_path: Path) -> None:
    path = tmp_path / "t1_receipt.json"
    write_t1_receipt(path, _t1_payload("DIAGNOSTIC_ONLY", eligible=False))
    with pytest.raises(ValueError, match="diagnostic-only"):
        verify_t1_pass_receipt(
            path, checkpoint_sha256="c" * 64,
            pool_identity={"digest": "d" * 64},
            objective_identity=OBJECTIVE_IDENTITY,
        )


def test_t2_refuses_a_receipt_for_a_different_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "t1_receipt.json"
    write_t1_receipt(path, _t1_payload())
    with pytest.raises(ValueError, match="differs from the checkpoint T1 passed"):
        verify_t1_pass_receipt(
            path, checkpoint_sha256="9" * 64,
            pool_identity={"digest": "d" * 64},
            objective_identity=OBJECTIVE_IDENTITY,
        )


def test_t2_refuses_a_receipt_for_a_different_pool(tmp_path: Path) -> None:
    path = tmp_path / "t1_receipt.json"
    write_t1_receipt(path, _t1_payload())
    with pytest.raises(ValueError, match="pool differs from the pool T1 used"):
        verify_t1_pass_receipt(
            path, checkpoint_sha256="c" * 64,
            pool_identity={"digest": "9" * 64},
            objective_identity=OBJECTIVE_IDENTITY,
        )


def test_t2_refuses_a_receipt_from_a_stale_primary_cell(tmp_path: Path) -> None:
    payload = _t1_payload()
    payload["t1_gate"]["primary_cell"] = "t0.75_nfe5_tangent"
    path = tmp_path / "t1_receipt.json"
    write_t1_receipt(path, payload)
    with pytest.raises(ValueError, match="frozen gate is"):
        verify_t1_pass_receipt(
            path, checkpoint_sha256="c" * 64,
            pool_identity={"digest": "d" * 64},
            objective_identity=OBJECTIVE_IDENTITY,
        )


VP_OBJECTIVE_IDENTITY = {**OBJECTIVE_IDENTITY, "flow_objective": VP_TANGENT_OBJECTIVE}


def _vp_t1_payload() -> dict:
    payload = _t1_payload()
    payload["experiment"] = VP_TANGENT_RECONSTRUCTION_SPEC.version
    payload["t1_gate"]["primary_cell"] = VP_TANGENT_RECONSTRUCTION_SPEC.primary_cell()
    payload["t1_gate"]["primary_t_start"] = VP_TANGENT_RECONSTRUCTION_SPEC.primary_t_start
    payload["checkpoint_selection"]["objective_identity"] = VP_OBJECTIVE_IDENTITY
    return payload


def test_a_vp_receipt_authorizes_the_vp_arm(tmp_path: Path) -> None:
    """Post-stop A: the VP arm's own T1 PASS must authorize the VP T2.

    Its cell key and objective are the matched-severity image of the frozen
    linear ones, so verifying against the linear defaults rejects a valid gate
    and makes experiment A unrunnable.
    """

    path = tmp_path / "t1_receipt.json"
    write_t1_receipt(path, _vp_t1_payload())
    verified = verify_t1_pass_receipt(
        path,
        checkpoint_sha256="c" * 64,
        pool_identity={"digest": "d" * 64},
        objective_identity=VP_OBJECTIVE_IDENTITY,
        spec=VP_TANGENT_RECONSTRUCTION_SPEC,
    )
    assert verified["verified"] is True
    assert verified["primary_cell"] == "t0.50_nfe1_vp_tangent"


def test_t2_refuses_a_receipt_from_the_other_corruption_path(tmp_path: Path) -> None:
    """The arms must not authorize each other: same model, different path."""

    path = tmp_path / "t1_receipt.json"
    write_t1_receipt(path, _t1_payload())
    with pytest.raises(ValueError, match="frozen gate is"):
        verify_t1_pass_receipt(
            path, checkpoint_sha256="c" * 64,
            pool_identity={"digest": "d" * 64},
            objective_identity=OBJECTIVE_IDENTITY,
            spec=VP_TANGENT_RECONSTRUCTION_SPEC,
        )


def test_t2_refuses_a_vp_receipt_whose_objective_is_the_linear_path(tmp_path: Path) -> None:
    """A cell key alone must not authorize: the recorded objective is checked too.

    `objective_identity=None` isolates this check from the identity-equality one
    that follows it, so the objective gate is shown to be arm-aware on its own.
    """

    payload = _vp_t1_payload()
    payload["checkpoint_selection"]["objective_identity"] = OBJECTIVE_IDENTITY
    path = tmp_path / "t1_receipt.json"
    write_t1_receipt(path, payload)
    with pytest.raises(ValueError, match="not for a .* checkpoint"):
        verify_t1_pass_receipt(
            path, checkpoint_sha256="c" * 64,
            pool_identity={"digest": "d" * 64},
            objective_identity=None,
            spec=VP_TANGENT_RECONSTRUCTION_SPEC,
        )


# --------------------------------------------------------------------------
# P10: result immutability
# --------------------------------------------------------------------------


def test_a_fresh_output_directory_is_created(tmp_path: Path) -> None:
    target = tmp_path / "results" / "t1"
    require_fresh_output_dir(target)
    assert target.is_dir()
    # An empty existing directory is fine.
    require_fresh_output_dir(target)


def test_an_existing_result_cannot_be_silently_overwritten(tmp_path: Path) -> None:
    """P10 regression: scientific artifacts are not build outputs."""

    target = tmp_path / "results"
    target.mkdir()
    (target / "tangent_reconstruction.json").write_text("{}")

    with pytest.raises(FileExistsError, match="refusing to overwrite a scientific result"):
        require_fresh_output_dir(target)

    # ...unless the operator explicitly asks for a discardable debug run.
    require_fresh_output_dir(target, overwrite_debug=True)
