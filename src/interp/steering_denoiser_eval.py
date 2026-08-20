"""Matched-strength evaluation of the steering-corruption denoiser (post-stop B).

The whole point of this file is one guard rail: **a quality win at a smaller
realised concept strength is not a quality win.** A denoiser trained on
``z = h + delta v`` can trivially score well against additive steering at equal
*nominal* alpha by simply removing the steering. So the decisive arm here is not
additive-at-nominal-alpha but

    scalar shrinkage evaluated at the denoiser's own realised alpha_eff,
    row by row.

Both arms then carry the same amount of concept, and any remaining difference in
language-model quality is attributable to *how* the activation was moved rather
than to *how far* along ``v`` it ended up.

Everything selectable is frozen in :data:`STEERING_DENOISER_EVAL_SPEC` before any
quality number is computed. The sequence, direction and quantile plan is
`natural_support_v1`, reused verbatim so this experiment sits on the same prompts,
seeds and clustering as the closed branch's T2.

Concept-independent inputs throughout: frozen validation activations, training-only
pool directions, no DEV steering vector, no held-out data, no LLM judge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .natural_support import NATURAL_SUPPORT_SPEC, NaturalSupportSpec
from .steering_denoiser import STEERING_CORRUPTION_SPEC, SteeringCorruptionSpec
from .tangent_eval import (
    clustered_bootstrap_mean_ci,
    equal_weight_pooled_effect,
    hierarchical_quantile_bootstrap,
)

# Frozen before any Experiment B quality number existed.
STOP_RULE_B = (
    "If the steering-trained denoiser's apparent quality gain disappears once the "
    "comparison is matched on realised concept strength, the method is an "
    "attenuator and experiment B is negative. Do not rescue it by selecting a "
    "favourable lambda, widening the delta distribution, growing the model, or "
    "reporting nominal-alpha comparisons as the headline."
)


@dataclass(frozen=True)
class SteeringDenoiserEvalSpec:
    """Frozen inference grid, decision rule, and validity tolerances.

    The direction/sequence/quantile plan is inherited from
    :data:`interp.natural_support.NATURAL_SUPPORT_SPEC` rather than restated, so
    experiment B is evaluated on exactly the prompts, directions, seeds and
    bootstrap clustering the frozen T2 used.
    """

    version: str = "steering_denoiser_matched_strength_v1"
    plan: NaturalSupportSpec = NATURAL_SUPPORT_SPEC
    corruption: SteeringCorruptionSpec = STEERING_CORRUPTION_SPEC
    # THE experiment-level decision: one lambda, pooled over the five target
    # quantiles with equal weight, so there is exactly one primary statistic.
    # lambda = 1.00 is the method as stated -- the full correction D(z). The
    # partial lambdas are diagnostics that trace the strength/quality curve; none
    # of them may be promoted to primary after a result is seen.
    primary_lambda: float = 1.00
    # A matched-strength claim is void if the two arms did not actually land on
    # the same realised strength. Expressed relative to the natural coordinate
    # spread (sd 7.52), 1e-3 is ~1.3e-4 sigma.
    strength_match_tolerance: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.primary_lambda not in self.corruption.lambda_grid:
            raise ValueError("the primary lambda must be evaluated by the frozen grid")
        if self.strength_match_tolerance <= 0.0:
            raise ValueError("the strength-match tolerance must be positive")

    def primary_cell(self) -> str:
        """The one cell key the Experiment B verdict is read from."""

        return f"pooled_lambda{self.primary_lambda:.2f}_vs_matched_shrinkage"

    def arm_label(self, lam: float) -> str:
        return f"denoise_lambda{lam:.2f}"

    def shrinkage_label(self, lam: float) -> str:
        return f"matched_shrinkage_for_lambda{lam:.2f}"


STEERING_DENOISER_EVAL_SPEC = SteeringDenoiserEvalSpec()


def spec_payload(spec: SteeringDenoiserEvalSpec = STEERING_DENOISER_EVAL_SPEC) -> dict:
    return asdict(spec)


# --------------------------------------------------------------------------
# the alpha ladder
# --------------------------------------------------------------------------


def alpha_ladder(
    coordinate_stats: list[dict[str, float | int]],
    spec: SteeringDenoiserEvalSpec = STEERING_DENOISER_EVAL_SPEC,
) -> dict[str, np.ndarray]:
    """Per-direction steering strengths, read off each direction's own support.

    ``alpha_q = quantile_q(<h, v>) - quantile_50(<h, v>)`` over the frozen
    reference rows. This puts every requested displacement inside the range the
    activations actually occupy -- the same construction the natural-support
    experiments used -- instead of an arbitrary alpha grid, and it makes the
    ladder directly comparable to the frozen T2 clamp displacements.

    ``p50`` is identically zero and is kept: it is the alpha = 0 control, where a
    correct denoiser must do nothing.
    """

    if not coordinate_stats:
        raise ValueError("need per-direction coordinate statistics")
    ladder: dict[str, np.ndarray] = {}
    # Key format matches interp.natural_support.natural_coordinate_stats exactly.
    median_key = "p50"
    for quantile in spec.plan.target_quantiles:
        key = f"p{int(round(quantile * 100)):02d}"
        values = []
        for entry in coordinate_stats:
            if key not in entry or median_key not in entry:
                raise ValueError(
                    f"coordinate statistics lack {key!r}/{median_key!r}; the frozen "
                    "recorded quantiles must cover the target quantiles"
                )
            values.append(float(entry[key]) - float(entry[median_key]))
        ladder[key] = np.asarray(values, dtype=np.float64)
    if not np.allclose(ladder["p50"], 0.0):
        raise ValueError("the p50 rung must be the alpha = 0 control")
    return ladder


# --------------------------------------------------------------------------
# validity: the two arms must really carry the same concept strength
# --------------------------------------------------------------------------


def assert_strength_match(
    denoised: dict[str, np.ndarray],
    shrunk: dict[str, np.ndarray],
    *,
    tolerance: float = STEERING_DENOISER_EVAL_SPEC.strength_match_tolerance,
) -> dict[str, float | int | bool]:
    """Refuse to report a matched-strength cell whose arms drifted apart.

    Without this, "matched strength" is a claim in a docstring rather than a
    property of the data, and the entire experiment reduces to the nominal-alpha
    comparison it was built to avoid.
    """

    left = np.asarray(denoised["realised_alpha"], dtype=np.float64)
    right = np.asarray(shrunk["realised_alpha"], dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("the two arms must report one realised strength per row")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("realised strengths must be finite")
    worst = float(np.abs(left - right).max())
    if worst > tolerance:
        raise ValueError(
            f"matched-strength arms differ by up to {worst:.3e} in realised alpha, "
            f"above the frozen tolerance {tolerance:g}; this cell is not a "
            "matched comparison and must not be reported as one"
        )
    return {
        "max_arm_strength_difference": worst,
        "tolerance": float(tolerance),
        "n_rows": int(left.size),
        "matched": True,
    }


# --------------------------------------------------------------------------
# per-arm summary
# --------------------------------------------------------------------------


def strength_summary(
    rows: dict[str, np.ndarray],
    nll: np.ndarray,
    baseline_nll: np.ndarray,
    clean_nll: np.ndarray,
    assignment: np.ndarray,
    *,
    spec: SteeringDenoiserEvalSpec = STEERING_DENOISER_EVAL_SPEC,
) -> dict:
    """One arm's quality and the concept strength it actually delivered.

    ``baseline_nll`` is the arm this one is paired against -- matched shrinkage
    for a denoiser arm, additive steering for the nominal-alpha context table.
    Both the paired effect and the raw strengths are always reported, so a reader
    can see attenuation directly rather than inferring it.
    """

    paired = np.asarray(nll, dtype=np.float64) - np.asarray(baseline_nll, dtype=np.float64)
    requested = np.asarray(rows["requested_alpha"], dtype=np.float64)
    realised = np.asarray(rows["realised_alpha"], dtype=np.float64)
    nonzero = np.abs(requested) > 1e-9
    summary: dict[str, object] = {
        "mean_nll": float(np.mean(nll)),
        "mean_clean_nll": float(np.mean(clean_nll)),
        "delta_lm_vs_clean": float(np.mean(nll) - np.mean(clean_nll)),
        "paired_delta_nll_vs_baseline": clustered_bootstrap_mean_ci(
            paired,
            assignment,
            seed=spec.plan.bootstrap_seed,
            n_resamples=spec.plan.bootstrap_resamples,
            confidence=spec.plan.confidence,
        ),
        "requested_alpha_mean": float(requested.mean()),
        "realised_alpha_mean": float(realised.mean()),
        "attenuation_mean": float((requested - realised).mean()),
        "orthogonal_correction_norm_mean": float(
            np.mean(rows["orthogonal_correction_norm"])
        ),
        "parallel_correction_norm_mean": float(np.mean(rows["parallel_correction_norm"])),
        "n_rows": int(realised.size),
    }
    # The single number that says whether the model repaired or removed. Undefined
    # at the alpha = 0 rung, where it is omitted rather than faked.
    if nonzero.any():
        summary["retained_strength_fraction"] = float(
            np.mean(realised[nonzero] / requested[nonzero])
        )
    else:
        summary["retained_strength_fraction"] = None
        summary["retained_strength_note"] = (
            "alpha = 0 control: no strength was requested, so a retained fraction "
            "would be meaningless"
        )
    return summary


# --------------------------------------------------------------------------
# the experiment-level verdict
# --------------------------------------------------------------------------


def steering_denoiser_verdict(
    effects_by_quantile: dict[str, np.ndarray],
    assignment: np.ndarray,
    *,
    spec: SteeringDenoiserEvalSpec = STEERING_DENOISER_EVAL_SPEC,
    formal_eligible: bool = True,
    ineligible_reason: str | None = None,
) -> dict[str, object]:
    """The frozen Experiment B decision, read from the primary lambda only.

    ``effects_by_quantile[q]`` is the paired per-sequence effect

        NLL(denoiser at the primary lambda) - NLL(shrinkage at the same alpha_eff)

    A negative pooled effect with an interval excluding zero is a *candidate*
    positive: genuine repair beyond what steering less would have bought. Anything
    else closes the hypothesis. The additional controls the protocol requires
    before a candidate may be called a result are not evaluated here; this
    function issues the primary statistic and nothing more.
    """

    pooled = equal_weight_pooled_effect(effects_by_quantile, assignment)
    interval = hierarchical_quantile_bootstrap(
        effects_by_quantile,
        assignment,
        seed=spec.plan.bootstrap_seed,
        n_resamples=spec.plan.bootstrap_resamples,
        confidence=spec.plan.confidence,
    )
    improves = float(interval["ci_upper"]) < 0.0
    verdict: dict[str, object] = {
        "gate": "B",
        "primary_cell": spec.primary_cell(),
        "primary_lambda": spec.primary_lambda,
        "comparison": (
            "denoiser vs scalar shrinkage at the SAME realised concept strength; "
            "a nominal-alpha comparison is context only and never the decision"
        ),
        "pooled_effect": pooled,
        "interval": interval,
        "stop_rule": STOP_RULE_B,
        "formal_verdict_eligible": bool(formal_eligible),
    }
    if not formal_eligible:
        verdict["verdict"] = None
        verdict["ineligible_reason"] = ineligible_reason or "not eligible for a formal verdict"
        return verdict
    verdict["verdict"] = "CANDIDATE_POSITIVE" if improves else "NEGATIVE"
    verdict["interpretation"] = (
        "the denoiser improves LM quality over steering the same amount less; "
        "the protocol's remaining controls must now be run before this is called "
        "a result"
        if improves
        else "matched on realised concept strength, the denoiser does not improve "
        "LM quality over simple shrinkage; the apparent gain, if any, is "
        "attenuation"
    )
    return verdict
