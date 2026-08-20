"""Fail-closed tests for the clean full-DEV Phase B evaluator manifest."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "flow_phase_b_evaluator_v1.yaml"
LEGACY_RAW = Path(
    os.environ.get(
        "PHASE_B_BASELINE_DIR", "/home/oleg/projects/tink/interp/results/raw"
    )
)


def _evaluator_module():
    try:
        return importlib.import_module("interp.phase_b_evaluator")
    except ModuleNotFoundError:
        pytest.fail("clean Phase B evaluator module is not implemented")


def test_exact_evaluator_manifest_loads_and_enumerates_frozen_matrix() -> None:
    module = _evaluator_module()
    cfg = module.load_evaluator_config(CONFIG)

    assert cfg.experiment_id == "clean_flow_phase_b_dev_v1"
    assert cfg.schema_version == "clean_phase_b_row_v1"
    assert cfg.output_layout_version == "clean_phase_b_outputs_v1"
    assert cfg.bootstrap_seed == 20_260_813
    assert cfg.bootstrap_resamples == 10_000
    assert cfg.metric_versions == {
        "text": "clean_text_metrics_v1",
        "nll": "clean_gpt2_conditional_nll_v1",
        "sae": "clean_frozen_sae_continuation_mean_v1",
        "geometry": "clean_flow_geometry_v2",
    }
    assert tuple(arm.arm_id for arm in cfg.arms) == (
        "flow_t010_nfe1",
        "flow_t010_nfe3",
        "flow_t010_nfe5",
        "flow_t025_nfe1",
        "flow_t025_nfe3",
        "flow_t025_nfe5",
        "flow_t050_nfe1",
        "flow_t050_nfe3",
        "flow_t050_nfe5",
    )
    cells = module.expected_cells(cfg)
    assert len(cells) == 2_880
    assert len(set(cells)) == 2_880
    assert {cell.vector for cell in cells} == {
        "allegations",
        "dungeon",
        "locations_addresses",
        "illicit_drugs",
        "law_enforcement_officials",
        "same_sex_marriage",
        "borders",
        "sports_awards",
    }


def test_evaluator_manifest_binds_every_baseline_artifact_sha() -> None:
    cfg = _evaluator_module().load_evaluator_config(CONFIG)

    assert {
        spec.method: (spec.raw_sha256, spec.meta_sha256) for spec in cfg.baselines
    } == {
        "additive": (
            "ba6a164f7cbd428f65679c67d3f101dafbed31262b07abc78c653d9350f80058",
            "06904ea3015cc4afa3ea77a176f97a746a88a7f7e09a0a363d351996fcf2405a",
        ),
        "naive": (
            "f382d61303018334ae3001c286eb22ae6426228ad061c49b305ae2e2b543a831",
            "093f7133d42a89ee69c45ee5c32534ebc74f59e8ec7cf1d2a9ac0c2cbc02d40e",
        ),
        "shrinkage_k080": (
            "b9537df71f7bbf5daa1418fe9f2198516e3390cafbea3444a0f4aa50c410d097",
            "69116464b4268104219c4778d9b18a079a44006c37070197b960d2a1e636fb5f",
        ),
    }


def test_evaluator_controls_are_exact_and_do_not_include_dev_directions() -> None:
    cfg = _evaluator_module().load_evaluator_config(CONFIG)
    dev_features = {vector.feature for vector in cfg.phase_b.vectors}

    assert set(cfg.control_features) == {vector.name for vector in cfg.phase_b.vectors}
    assert all(len(values) == 32 for values in cfg.control_features.values())
    assert all(len(set(values)) == 32 for values in cfg.control_features.values())
    assert all(not (set(values) & dev_features) for values in cfg.control_features.values())
    assert cfg.control_features["allegations"][:4] == (19237, 8326, 17561, 7329)
    assert cfg.control_features["sports_awards"][-4:] == (15046, 22600, 17909, 22481)


def test_baseline_row_import_allow_lists_only_identity_and_text() -> None:
    module = _evaluator_module()
    cfg = module.load_evaluator_config(CONFIG)
    cell = module.expected_cells(cfg)[0]
    base = {
        "vector": cell.vector,
        "feature": cell.feature,
        "split": "dev",
        "alpha": cell.alpha,
        "alpha_hat": cell.alpha_hat,
        "is_stress": cell.is_stress,
        "prompt_idx": cell.prompt_id,
        "seed": cell.generation_seed,
        "continuation": " fixed text",
    }

    first = module.parse_baseline_row(
        {**base, "nll": -999.0, "lexicon_score": 1.0}, cell
    )
    second = module.parse_baseline_row(
        {**base, "nll": 999.0, "lexicon_score": 0.0, "legacy_aggregate": "junk"}, cell
    )

    assert first == second
    assert first.continuation == " fixed text"
    assert not hasattr(first, "nll")
    assert not hasattr(first, "lexicon_score")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split", "heldout", "DEV"),
        ("feature", 999, "feature"),
        ("alpha", 0.25, "alpha"),
        ("alpha_hat", 0.25, "alpha_hat"),
        ("is_stress", True, "stress"),
        ("prompt_idx", 9, "prompt"),
        ("seed", 7, "seed"),
        ("continuation", 123, "continuation"),
    ],
)
def test_baseline_row_import_rejects_identity_or_text_mutation(
    field: str, value: object, message: str
) -> None:
    module = _evaluator_module()
    cfg = module.load_evaluator_config(CONFIG)
    cell = module.expected_cells(cfg)[0]
    row = {
        "vector": cell.vector,
        "feature": cell.feature,
        "split": "dev",
        "alpha": cell.alpha,
        "alpha_hat": cell.alpha_hat,
        "is_stress": cell.is_stress,
        "prompt_idx": cell.prompt_id,
        "seed": cell.generation_seed,
        "continuation": " text",
    }
    row[field] = value

    with pytest.raises(ValueError, match=message):
        module.parse_baseline_row(row, cell)


def test_all_frozen_baselines_validate_with_exact_identical_coverage() -> None:
    module = _evaluator_module()
    cfg = module.load_evaluator_config(CONFIG)
    imported = {
        spec.method: module.validate_and_import_baseline(
            spec,
            LEGACY_RAW / spec.raw_filename,
            LEGACY_RAW / spec.meta_filename,
            cfg,
        )
        for spec in cfg.baselines
    }

    assert {method: len(rows) for method, rows in imported.items()} == {
        "additive": 2_880,
        "naive": 2_880,
        "shrinkage_k080": 2_880,
    }
    module.validate_cross_baseline_coverage(imported)


def test_baseline_validation_rejects_changed_raw_bytes(tmp_path: Path) -> None:
    module = _evaluator_module()
    cfg = module.load_evaluator_config(CONFIG)
    spec = cfg.baselines[0]
    changed = tmp_path / spec.raw_filename
    changed.write_bytes((LEGACY_RAW / spec.raw_filename).read_bytes() + b"\n")

    with pytest.raises(ValueError, match="raw SHA"):
        module.validate_and_import_baseline(
            spec,
            changed,
            LEGACY_RAW / spec.meta_filename,
            cfg,
        )


def test_baseline_validation_rejects_changed_meta_bytes(tmp_path: Path) -> None:
    module = _evaluator_module()
    cfg = module.load_evaluator_config(CONFIG)
    spec = cfg.baselines[0]
    changed = tmp_path / spec.meta_filename
    meta = json.loads((LEGACY_RAW / spec.meta_filename).read_text())
    meta["split"] = "heldout"
    changed.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="meta SHA"):
        module.validate_and_import_baseline(
            spec,
            LEGACY_RAW / spec.raw_filename,
            changed,
            cfg,
        )
