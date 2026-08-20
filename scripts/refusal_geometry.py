"""Experiment D, phase 2: the Experiment C diagnostic on a known causal direction.

Runs only after `scripts/refusal_causal_validation.py` has issued
CAUSAL_CONTROL_PASS on this exact direction, and refuses otherwise: geometry on a
direction that has not been shown to act on behaviour would answer nothing.

The statistics are `interp.curvature`, unchanged --- the same secants, the same
shuffle null, the same split-half ceiling that produced Experiment C. Two things
necessarily differ, and both are consequences of the setting rather than choices
made to get a nicer number:

* C had 32 directions and clustered its bootstrap over them. D has exactly one
  causal direction, so its interval comes from resampling *prompts*, and the
  random axes supply an empirical null the single value is placed against.
* the bin-occupancy floor is lower, because the harmful split holds 572 prompts
  rather than 262144 activation rows. The split-half ceiling is reported beside
  every estimate precisely so that a thin bin cannot pass as curvature.

The harmful/harmless confound is handled by three analyses fixed in
`docs/EXPERIMENT_D_PROTOCOL.md` before any result: class-balanced (primary),
harmful-only, harmless-only.

    uv run python scripts/refusal_geometry.py \
        --validation results/refusal_causal_validation_d_v1 \
        --direction data/refusal_direction/gemma-2b-it_direction.pt \
        --metadata data/refusal_direction/gemma-2b-it_direction_metadata.json \
        --splits data/refusal_direction/splits \
        --out-dir results/refusal_geometry_d_v1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interp.curvature import (
    CURVATURE_SPEC,
    CurvatureSpec,
    _direction_shortfall,
    bin_indices,
    bin_means,
    direction_curvature,
    secant_geometry,
    spec_payload,
)
from interp.provenance import source_revision
from interp.refusal_control import (
    REFUSAL_CONTROL_SPEC,
    capture_residual_pre,
    load_refusal_direction,
    tokenize_instructions,
)
from interp.tangent_eval import require_fresh_output_dir

SPEC = REFUSAL_CONTROL_SPEC

# The D geometry plan. Same cuts, same seeds, same statistics as C. Two values
# differ and are frozen here, before any D geometry number existed:
#   min_bin_rows  256 -> 64   the harmful split has 572 prompts, so six bins hold
#                             ~95 each; 64 admits them while still refusing a
#                             badly undersampled bin. The reliability ceiling,
#                             not this floor, is what licenses a curvature claim.
#   n_random_directions 32 -> 128  D compares ONE direction against the null, so
#                             the null needs resolution; the statistic is
#                             unchanged, only the number of draws from it.
D_CURVATURE_SPEC = replace(
    CURVATURE_SPEC,
    version="refusal_curvature_d_v1",
    min_bin_rows=64,
    n_random_directions=128,
)

CLASS_HARMLESS, CLASS_HARMFUL = 0, 1


def load_split(splits_dir: Path, name: str) -> list[str]:
    rows = json.loads((Path(splits_dir) / f"{name}.json").read_text())
    return [row["instruction"] for row in rows]


@torch.no_grad()
def collect_activations(
    model, tokenizer, instructions: list[str], layer: int, position: int
) -> np.ndarray:
    """The residual stream entering ``layer`` at the direction's own token position.

    One row per prompt. Left padding keeps ``position`` counted from the true end
    of every prompt rather than from padding.
    """

    rows = []
    for start in range(0, len(instructions), SPEC.scoring_batch_size):
        batch = instructions[start : start + SPEC.scoring_batch_size]
        tokens = tokenize_instructions(tokenizer, batch)
        store: list[torch.Tensor] = []
        with capture_residual_pre(model, layer, store):
            model(
                input_ids=tokens.input_ids.to(model.device),
                attention_mask=tokens.attention_mask.to(model.device),
            )
        rows.append(store[0][:, position, :].to(torch.float64).numpy())
        print(f"  collected {sum(r.shape[0] for r in rows)}/{len(instructions)}", flush=True)
    return np.concatenate(rows, axis=0)


def random_unit_directions(width: int, spec: CurvatureSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.random_direction_seed)
    raw = rng.normal(size=(spec.n_random_directions, width))
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def _statistics(record: dict) -> dict[str, float]:
    """The three headline numbers of one curvature record."""

    cos_secants = np.asarray(record["cos_consecutive_secants"], dtype=np.float64)
    cos_direction = np.asarray(record["cos_secant_direction"], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return {
            "mean_cos_consecutive_secants": float(np.nanmean(cos_secants)),
            "mean_cos_secant_direction": float(np.nanmean(cos_direction)),
            "shortfall_below_ceiling": _direction_shortfall(record),
        }


def prompt_bootstrap(
    activations: np.ndarray,
    direction: np.ndarray,
    classes: np.ndarray | None,
    spec: CurvatureSpec,
) -> dict:
    """Interval for a SINGLE direction, by resampling prompts.

    C could resample directions because it had 32 of them. D has one causal
    direction, so the sampling variability that matters is over prompts.
    """

    unit = direction / np.linalg.norm(direction)
    n = activations.shape[0]
    rng = np.random.default_rng(spec.bootstrap_seed)
    per_rung: list[np.ndarray] = []
    secant_cos: list[np.ndarray] = []
    for _ in range(spec.bootstrap_resamples):
        rows = rng.integers(0, n, size=n)
        values = activations[rows]
        coordinate = values @ unit
        assignment = bin_indices(coordinate, spec)
        labels = None if classes is None else classes[rows]
        mu, _ = bin_means(values, assignment, spec, classes=labels)
        geometry = secant_geometry(mu, unit)
        per_rung.append(np.asarray(geometry["cos_secant_direction"], dtype=np.float64))
        secant_cos.append(
            np.asarray(geometry["cos_consecutive_secants"], dtype=np.float64)
        )
    tail = (1.0 - spec.confidence) / 2.0

    def interval(draws: np.ndarray) -> list[dict]:
        out = []
        for index in range(draws.shape[1]):
            column = draws[:, index]
            column = column[np.isfinite(column)]
            if column.size < 2:
                out.append({"rung": index, "mean": float("nan"), "usable": False})
                continue
            out.append({
                "rung": index,
                "mean": float(column.mean()),
                "ci_lower": float(np.quantile(column, tail)),
                "ci_upper": float(np.quantile(column, 1.0 - tail)),
                "usable": True,
            })
        return out

    stacked_direction = np.stack(per_rung)
    stacked_secants = np.stack(secant_cos)
    with np.errstate(invalid="ignore"):
        pooled_direction = np.nanmean(stacked_direction, axis=1)
        pooled_secants = np.nanmean(stacked_secants, axis=1)
    pooled_direction = pooled_direction[np.isfinite(pooled_direction)]
    pooled_secants = pooled_secants[np.isfinite(pooled_secants)]

    def pooled(draws: np.ndarray) -> dict:
        if draws.size < 2:
            return {"mean": float("nan"), "usable": False}
        return {
            "mean": float(draws.mean()),
            "ci_lower": float(np.quantile(draws, tail)),
            "ci_upper": float(np.quantile(draws, 1.0 - tail)),
            "usable": True,
        }

    return {
        "resampling": "prompt_bootstrap_single_direction",
        "n_resamples": spec.bootstrap_resamples,
        "confidence": spec.confidence,
        "cos_secant_direction_by_rung": interval(stacked_direction),
        "cos_consecutive_secants_by_rung": interval(stacked_secants),
        "pooled_cos_secant_direction": pooled(pooled_direction),
        "pooled_cos_consecutive_secants": pooled(pooled_secants),
    }


def empirical_null(
    value: float, null_values: np.ndarray, *, spec: CurvatureSpec
) -> dict:
    """Where one number falls in the distribution of the random axes' numbers."""

    null_values = np.asarray(null_values, dtype=np.float64)
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size < 2 or not np.isfinite(value):
        return {"usable": False, "n_random": int(null_values.size)}
    tail = (1.0 - spec.confidence) / 2.0
    above = float((null_values < value).mean())
    return {
        "usable": True,
        "value": float(value),
        "random_mean": float(null_values.mean()),
        "random_ci_lower": float(np.quantile(null_values, tail)),
        "random_ci_upper": float(np.quantile(null_values, 1.0 - tail)),
        "random_min": float(null_values.min()),
        "random_max": float(null_values.max()),
        "n_random": int(null_values.size),
        "fraction_of_random_axes_below": above,
        "outside_random_range": bool(value > null_values.max() or value < null_values.min()),
        "above_random_interval": bool(value > np.quantile(null_values, 1.0 - tail)),
        "below_random_interval": bool(value < np.quantile(null_values, tail)),
    }


def analyse(
    name: str,
    activations: np.ndarray,
    classes: np.ndarray | None,
    unit: np.ndarray,
    spec: CurvatureSpec,
) -> dict:
    """One of the three analyses: refusal direction, random null, and comparison."""

    sequence = np.arange(activations.shape[0])
    record = direction_curvature(
        activations, unit, sequence, spec=spec, seed_offset=0, classes=classes
    )
    stats = _statistics(record)
    randoms = [
        direction_curvature(
            activations, axis, sequence, spec=spec, seed_offset=1000 + index,
            classes=classes,
        )
        for index, axis in enumerate(random_unit_directions(activations.shape[1], spec))
    ]
    random_stats = {
        key: np.array([_statistics(r)[key] for r in randoms])
        for key in stats
    }
    usable_bins = int(np.isfinite(np.asarray(record["bin_coordinate_means"])).sum())
    usable_secants = int(
        np.isfinite(np.asarray(record["cos_secant_direction"], dtype=np.float64)).sum()
    )
    return {
        "analysis": name,
        "n_rows": int(activations.shape[0]),
        "class_balanced": classes is not None,
        "usable_bins": usable_bins,
        "usable_secants": usable_secants,
        "sufficient_for_a_claim": bool(usable_secants >= 2),
        "refusal_direction": {
            "record": record,
            "statistics": stats,
            "bootstrap": prompt_bootstrap(activations, unit, classes, spec),
        },
        "random_axes": {
            "n_directions": len(randoms),
            "statistics_mean": {k: float(np.nanmean(v)) for k, v in random_stats.items()},
            "by_direction": {k: v.tolist() for k, v in random_stats.items()},
        },
        "refusal_vs_random": {
            key: empirical_null(stats[key], random_stats[key], spec=spec)
            for key in stats
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True, type=Path,
                        help="the CAUSAL_CONTROL_PASS result directory from phase 1")
    parser.add_argument("--direction", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-path", default=SPEC.model_path)
    parser.add_argument("--overwrite-debug-mode", action="store_true")
    args = parser.parse_args()

    require_fresh_output_dir(args.out_dir, overwrite_debug=args.overwrite_debug_mode)

    validation = json.loads((args.validation / "causal_validation.json").read_text())
    verdict = validation["verdict"]
    if verdict["verdict"] != "CAUSAL_CONTROL_PASS":
        raise SystemExit(
            f"causal validation says {verdict['verdict']!r}: Experiment D stops at "
            "phase 1 and no geometry is computed on an unvalidated direction"
        )
    direction = load_refusal_direction(args.direction, args.metadata, spec=SPEC)
    if validation["direction"]["sha256"] != direction.sha256:
        raise ValueError(
            "the validated direction and the geometry direction are not the same "
            "vector; the control would not transfer"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    model.requires_grad_(False)

    harmful = load_split(args.splits, "harmful_test")
    harmless = load_split(args.splits, "harmless_test")[: SPEC.geometry_max_prompts_per_class]
    print(f"collecting harmful ({len(harmful)})...", flush=True)
    harmful_acts = collect_activations(
        model, tokenizer, harmful, direction.layer, direction.position
    )
    print(f"collecting harmless ({len(harmless)})...", flush=True)
    harmless_acts = collect_activations(
        model, tokenizer, harmless, direction.layer, direction.position
    )

    activations = np.concatenate([harmless_acts, harmful_acts], axis=0)
    classes = np.array(
        [CLASS_HARMLESS] * harmless_acts.shape[0] + [CLASS_HARMFUL] * harmful_acts.shape[0]
    )
    unit = direction.unit.numpy()

    analyses = {
        "d_main_class_balanced": analyse(
            "d_main_class_balanced", activations, classes, unit, D_CURVATURE_SPEC
        ),
        "d_harmful_only": analyse(
            "d_harmful_only", harmful_acts, None, unit, D_CURVATURE_SPEC
        ),
        "d_harmless_only": analyse(
            "d_harmless_only", harmless_acts, None, unit, D_CURVATURE_SPEC
        ),
    }

    payload = {
        "experiment": D_CURVATURE_SPEC.version,
        "phase": "geometry",
        "class": "post_stop_method_development",
        "question": (
            "does the Experiment C diagnostic behave differently on an "
            "independently validated causal linear direction?"
        ),
        "protocol": "docs/EXPERIMENT_D_PROTOCOL.md",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "debug_mode": bool(args.overwrite_debug_mode),
        "model_path": args.model_path,
        "activation_site": (
            f"residual stream entering decoder block {direction.layer}, token "
            f"position {direction.position} (the direction's own site)"
        ),
        "curvature_spec": spec_payload(D_CURVATURE_SPEC),
        "curvature_spec_deviations_from_c": {
            "min_bin_rows": {"c": CURVATURE_SPEC.min_bin_rows,
                             "d": D_CURVATURE_SPEC.min_bin_rows},
            "n_random_directions": {"c": CURVATURE_SPEC.n_random_directions,
                                    "d": D_CURVATURE_SPEC.n_random_directions},
            "reason": "prompt-level population and a single causal direction; "
                      "frozen before any D geometry number existed",
        },
        "refusal_control_spec": {
            "layer": direction.layer, "position": direction.position,
        },
        "direction": direction.receipt,
        "causal_validation": {
            "source": str(args.validation),
            "verdict": verdict["verdict"],
            "ablation_refusal_drop_pp": verdict["ablation_refusal_drop_pp"],
            "addition_refusal_rise_pp": verdict["addition_refusal_rise_pp"],
        },
        "n_harmful": int(harmful_acts.shape[0]),
        "n_harmless": int(harmless_acts.shape[0]),
        "analyses": analyses,
        "dev_vectors_accessed": False,
        "held_out_accessed": False,
        "llm_judge_used": False,
        "trained_anything": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "refusal_geometry.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(
        args.out_dir / "raw_activations.npz",
        harmful=harmful_acts.astype(np.float32),
        harmless=harmless_acts.astype(np.float32),
        unit_direction=unit,
    )
    for name, analysis in analyses.items():
        print(f"\n=== {name} ===")
        print(f"  usable secants: {analysis['usable_secants']}")
        print(f"  refusal: {json.dumps(analysis['refusal_direction']['statistics'])}")
        print(f"  random : {json.dumps(analysis['random_axes']['statistics_mean'])}")


if __name__ == "__main__":
    main()
