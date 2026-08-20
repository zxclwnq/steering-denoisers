"""Experiment D, phase 3: issue one of the four preregistered outcomes.

The geometry run stores the pooled comparisons and the raw activations; the
outcome rule also needs the *profile* of the alignment --- how it behaves at the
natural centre versus the tails --- against the same random axes. That is
recomputed here from `raw_activations.npz`, so no forward pass and no model are
needed and nothing about the measurement changes.

The decision itself is `interp.refusal_control.geometry_interpretation`, whose
rule was frozen with `docs/EXPERIMENT_D_PROTOCOL.md` before any D number existed.
This script only assembles its inputs.

    uv run python scripts/refusal_geometry_interpret.py \
        --geometry results/refusal_geometry_d_v1 \
        --out-dir results/refusal_interpretation_d_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from interp.curvature import bin_indices, bin_means, secant_geometry
from interp.provenance import source_revision
from interp.refusal_control import geometry_interpretation
from interp.tangent_eval import require_fresh_output_dir

# Six bins give five secants; secant 2 crosses the median and is the natural
# centre, secant 4 is the high-coordinate tail. Same convention as Experiment C.
CENTRAL_SECANT = 2
UPPER_TAIL_SECANT = 4


def _module_spec(geometry: dict):
    """Rebuild the frozen D curvature spec from what the run recorded."""

    from dataclasses import replace

    from interp.curvature import CURVATURE_SPEC

    recorded = geometry["curvature_spec"]
    return replace(
        CURVATURE_SPEC,
        version=recorded["version"],
        min_bin_rows=recorded["min_bin_rows"],
        n_random_directions=recorded["n_random_directions"],
    )


def alignment_profile(
    activations: np.ndarray, direction: np.ndarray, spec
) -> np.ndarray:
    """``cos(d_k, r)`` per rung, for one direction, on one population."""

    unit = direction / np.linalg.norm(direction)
    assignment = bin_indices(activations @ unit, spec)
    mu, _ = bin_means(activations, assignment, spec)
    return np.asarray(secant_geometry(mu, unit)["cos_secant_direction"], dtype=np.float64)


def random_axes(width: int, spec) -> np.ndarray:
    rng = np.random.default_rng(spec.random_direction_seed)
    raw = rng.normal(size=(spec.n_random_directions, width))
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def empirical_null(value: float, null_values: np.ndarray, spec) -> dict:
    null_values = np.asarray(null_values, dtype=np.float64)
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size < 2 or not np.isfinite(value):
        return {"usable": False, "n_random": int(null_values.size)}
    tail = (1.0 - spec.confidence) / 2.0
    return {
        "usable": True,
        "value": float(value),
        "random_mean": float(null_values.mean()),
        "random_ci_lower": float(np.quantile(null_values, tail)),
        "random_ci_upper": float(np.quantile(null_values, 1.0 - tail)),
        "random_min": float(null_values.min()),
        "random_max": float(null_values.max()),
        "n_random": int(null_values.size),
        "outside_random_range": bool(
            value > null_values.max() or value < null_values.min()
        ),
        "above_random_interval": bool(value > np.quantile(null_values, 1.0 - tail)),
        "below_random_interval": bool(value < np.quantile(null_values, tail)),
    }


def tail_minus_central(activations: np.ndarray, direction: np.ndarray, spec) -> dict:
    """Paired bootstrap on the alignment profile: does it fall away from the centre?

    Paired by construction --- each resample supplies both rungs --- so the
    contrast is not built from prompts that were never compared with each other.
    """

    unit = direction / np.linalg.norm(direction)
    n = activations.shape[0]
    rng = np.random.default_rng(spec.bootstrap_seed)
    draws = []
    for _ in range(spec.bootstrap_resamples):
        rows = rng.integers(0, n, size=n)
        profile = alignment_profile(activations[rows], unit, spec)
        draws.append(profile[UPPER_TAIL_SECANT] - profile[CENTRAL_SECANT])
    values = np.asarray(draws, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return {"usable": False, "reason": "the tail or central rung is not estimable"}
    tail = (1.0 - spec.confidence) / 2.0
    return {
        "usable": True,
        "mean": float(values.mean()),
        "ci_lower": float(np.quantile(values, tail)),
        "ci_upper": float(np.quantile(values, 1.0 - tail)),
        "confidence": float(spec.confidence),
        "n_resamples": int(values.size),
        "resampling": "paired_prompt_bootstrap",
    }


def variance_control(activations: np.ndarray, unit: np.ndarray, spec, n_components: int = 8):
    """Is the alignment about causality, or just about carrying variance?

    Binning on a coordinate guarantees consecutive bin means differ along it, and
    how visible that displacement is depends on how much variance the direction
    carries. The refusal direction carries far more than a random axis, so the
    random-axis comparison alone cannot separate "causal" from "high-variance".

    The control is the leading principal components of the same activations:
    directions chosen by variance alone, with no causal validation of any kind,
    run through the identical pipeline. If they align as well, alignment tracks
    variance and says nothing about causal efficacy.
    """

    centred = activations - activations.mean(axis=0, keepdims=True)
    # economy SVD: we only need the leading right singular vectors
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    eigenvalues = (singular**2) / max(centred.shape[0] - 1, 1)
    variance_along_r = float(np.var(centred @ unit))
    total = float(np.sum(np.var(centred, axis=0)))

    components = []
    for index in range(min(n_components, vt.shape[0])):
        axis = vt[index]
        profile = alignment_profile(activations, axis, spec)
        with np.errstate(invalid="ignore"):
            pooled = float(np.nanmean(profile))
        components.append({
            "component": index,
            "variance": float(eigenvalues[index]),
            "variance_ratio_to_refusal": float(eigenvalues[index] / variance_along_r),
            "pooled_cos_secant_direction": None if not np.isfinite(pooled) else pooled,
            "central_cos_secant_direction": (
                None if not np.isfinite(profile[CENTRAL_SECANT])
                else float(profile[CENTRAL_SECANT])
            ),
        })
    usable = [c for c in components if c["pooled_cos_secant_direction"] is not None]
    refusal_profile = alignment_profile(activations, unit, spec)
    with np.errstate(invalid="ignore"):
        refusal_pooled = float(np.nanmean(refusal_profile))
    best = max((c["pooled_cos_secant_direction"] for c in usable), default=float("nan"))
    return {
        "variance_along_refusal_direction": variance_along_r,
        "variance_share_of_total": variance_along_r / total if total else None,
        "leading_component_variance": float(eigenvalues[0]),
        "refusal_pooled_cos_secant_direction": (
            None if not np.isfinite(refusal_pooled) else refusal_pooled
        ),
        "principal_components": components,
        "best_component_pooled_alignment": None if not np.isfinite(best) else float(best),
        "refusal_exceeds_every_component": bool(
            np.isfinite(refusal_pooled) and np.isfinite(best) and refusal_pooled > best
        ),
        "note": (
            "principal components are chosen by variance alone and have no "
            "validated causal effect; they are the control for the possibility "
            "that alignment merely tracks variance"
        ),
    }


def interpret(name: str, activations: np.ndarray, unit: np.ndarray, analysis: dict, spec):
    profile = alignment_profile(activations, unit, spec)
    axes = random_axes(activations.shape[1], spec)
    random_profiles = np.stack([alignment_profile(activations, axis, spec) for axis in axes])

    central = empirical_null(
        profile[CENTRAL_SECANT], random_profiles[:, CENTRAL_SECANT], spec
    )
    contrast = tail_minus_central(activations, unit, spec)
    outcome = geometry_interpretation(
        pooled_alignment_vs_random=analysis["refusal_vs_random"]["mean_cos_secant_direction"],
        central_alignment_vs_random=central,
        tail_minus_central=contrast,
        curvature_vs_random=analysis["refusal_vs_random"]["mean_cos_consecutive_secants"],
        shortfall_vs_random=analysis["refusal_vs_random"]["shortfall_below_ceiling"],
    )
    return {
        "analysis": name,
        "n_rows": int(activations.shape[0]),
        "usable_secants": analysis["usable_secants"],
        "alignment_profile": [None if not np.isfinite(x) else float(x) for x in profile],
        "random_alignment_profile_mean": [
            None if not np.isfinite(x) else float(x)
            for x in np.nanmean(random_profiles, axis=0)
        ],
        "central_rung_vs_random": central,
        "tail_minus_central": contrast,
        "pooled_alignment_vs_random": analysis["refusal_vs_random"][
            "mean_cos_secant_direction"
        ],
        "curvature_vs_random": analysis["refusal_vs_random"]["mean_cos_consecutive_secants"],
        "shortfall_vs_random": analysis["refusal_vs_random"]["shortfall_below_ceiling"],
        "reliability_ceiling_available": bool(
            analysis["refusal_vs_random"]["shortfall_below_ceiling"].get("usable")
        ),
        "variance_control": variance_control(activations, unit, spec),
        "outcome": outcome,
    }


def _variance_conclusion(results: dict) -> dict:
    """What the variance control does to the preregistered label.

    The frozen rule compares the causal direction against *random* axes. This
    control was added after seeing that it fired, and it is post-hoc for that
    reason; it is recorded here rather than used to rewrite the label, because
    changing a preregistered rule after seeing its output is the failure the rule
    exists to prevent. What it does establish is whether the inference the label
    was written to license still holds.
    """

    checked = {
        name: result["variance_control"]
        for name, result in results.items()
        if "variance_control" in result
    }
    if not checked:
        return {"available": False}
    exceeds = {
        name: control["refusal_exceeds_every_component"]
        for name, control in checked.items()
    }
    stands_out = any(exceeds.values())
    return {
        "available": True,
        "class": "post_hoc",
        "question": (
            "is the alignment about causality, or about how much variance the "
            "direction carries?"
        ),
        "refusal_exceeds_every_principal_component": exceeds,
        "variance_ratio_to_random_axis": {
            name: control["variance_along_refusal_direction"]
            for name, control in checked.items()
        },
        "conclusion": (
            "the causal direction aligns with the natural trajectory better than "
            "every variance-selected principal component, so the alignment is not "
            "explained by variance alone"
            if stands_out
            else "principal components -- selected by variance alone, with no "
            "causal validation whatsoever -- align at least as well as the causal "
            "direction. The random-axis reference in the frozen rule is therefore "
            "not an adequate control for this statistic: random unit axes in a "
            "high-dimensional residual stream carry very little variance, so any "
            "structured direction beats them. D provides NO evidence that causal "
            "directions are geometrically special"
        ),
        "effect_on_the_preregistered_label": (
            "the label stands as the frozen rule's output and is not rewritten; "
            "but the inference it was written to license -- that good causal "
            "intervention axes are geometrically distinguishable -- is not "
            "supported once variance is controlled"
            if not stands_out
            else "the label and the inference it licenses both survive the control"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--overwrite-debug-mode", action="store_true")
    args = parser.parse_args()

    require_fresh_output_dir(args.out_dir, overwrite_debug=args.overwrite_debug_mode)
    geometry = json.loads((args.geometry / "refusal_geometry.json").read_text())
    raw = np.load(args.geometry / "raw_activations.npz")
    spec = _module_spec(geometry)
    unit = np.asarray(raw["unit_direction"], dtype=np.float64)
    populations = {
        "d_harmful_only": np.asarray(raw["harmful"], dtype=np.float64),
        "d_harmless_only": np.asarray(raw["harmless"], dtype=np.float64),
    }

    results = {}
    for name, activations in populations.items():
        analysis = geometry["analyses"][name]
        if not analysis["sufficient_for_a_claim"]:
            results[name] = {
                "analysis": name,
                "usable_secants": analysis["usable_secants"],
                "outcome": {"outcome": None, "why": "too few usable secants to interpret"},
            }
            continue
        results[name] = interpret(name, activations, unit, analysis, spec)

    main_analysis = geometry["analyses"]["d_main_class_balanced"]
    payload = {
        "experiment": "refusal_geometry_interpretation_d_v1",
        "phase": "interpretation",
        "class": "post_stop_method_development",
        "protocol": "docs/EXPERIMENT_D_PROTOCOL.md",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "geometry_source": str(args.geometry),
        "causal_validation": geometry["causal_validation"],
        "d_main_class_balanced": {
            "usable_secants": main_analysis["usable_secants"],
            "sufficient_for_a_claim": main_analysis["sufficient_for_a_claim"],
            "note": (
                "the primary class-balanced analysis is UNDEFINED: the refusal "
                "coordinate separates harmful from harmless so completely that no "
                "coordinate bin holds enough of both classes to balance. This was "
                "anticipated in the protocol; the within-class analyses carry the "
                "interpretation and the reliability ceiling is stated with them"
            ),
        },
        "analyses": results,
        "post_hoc_variance_control": _variance_conclusion(results),
        "recomputes_no_activation": True,
        "dev_vectors_accessed": False,
        "held_out_accessed": False,
        "llm_judge_used": False,
        "trained_anything": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "refusal_interpretation.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    for name, result in results.items():
        print(f"\n=== {name} ===")
        print(f"  outcome: {result['outcome']['outcome']}")
        print(f"  alignment profile: {result.get('alignment_profile')}")
        if "central_rung_vs_random" in result:
            print(f"  central vs random: {json.dumps(result['central_rung_vs_random'])}")
            print(f"  tail - central   : {json.dumps(result['tail_minus_central'])}")


if __name__ == "__main__":
    main()
