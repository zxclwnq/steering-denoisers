"""Evaluation harness shared by the tangent branch's T1 and T2 experiments.

T1 asks whether the tangent-trained flow solves *its own* matched task: recover a
tangent-corrupted validation activation. T2 asks the downstream question: at a
semantic coordinate held exactly fixed by a hard clamp, does the tangent flow
lower LM NLL relative to the clamp alone.

Both are concept-independent: frozen validation activations, training-only pool
directions, no DEV or held-out data, no LLM judge. Nothing here trains anything.

The stop rule these evaluators serve is in docs/TANGENT_FLOW_PROTOCOL.md and is
restated in :data:`STOP_RULE` so a reader of a result file cannot miss it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .activations import (
    ActivationDataset,
    ActivationSplit,
    file_sha256,
    load_activations,
    load_validation_report,
    make_split,
    validate_activation_metadata,
)
from .conditional_flow import (
    ConditionalFlowMatcher,
    TrainingDirectionPool,
    clamp_seed,
    standardized_hyperplane,
)
from .model import MODEL_NAME, MODEL_RESOLVED_NAME, MODEL_REVISION, STEERING_HOOK
from .phase_a import paired_bootstrap_mean_ci
from .tangent_flow import (
    TANGENT_OBJECTIVE,
    VP_TANGENT_OBJECTIVE,
    matched_vp_time,
    tangent_path_states,
)

FROZEN_SELECTION_METRIC = "val_flow_mse"

STOP_RULE = (
    "If the tangent-trained flow clearly solves T1 (matched tangent reconstruction) "
    "but still does not improve hard-clamp NLL at a fixed coordinate in T2, "
    "generative naturalization stops entirely. Do not rescue with more parameters, "
    "more NFE, more generic data, another inference projection trick, or LLM judges."
)


@dataclass(frozen=True)
class TangentReconstructionSpec:
    """Frozen T1 sampling and analysis plan. Changing a field makes a new experiment.

    The direction/sequence selection fields are named to match
    :class:`interp.natural_support.NaturalSupportSpec` so the same frozen,
    already-tested selection helpers apply to both experiments.
    """

    version: str = "tangent_reconstruction_t1_v1"
    n_directions: int = 32
    direction_seed: int = 20260907
    n_sequences: int = 64
    sequence_seed: int = 20260908
    noise_seed: int = 20260909
    bootstrap_seed: int = 20260910
    bootstrap_resamples: int = 2000
    confidence: float = 0.95
    # The diagnostic grid. Its ORDER carries no scientific meaning: the formal
    # gate reads primary_t_start / primary_nfe below, never t_start[0].
    t_start: tuple[float, ...] = (0.25, 0.50, 0.75)
    nfe: tuple[int, ...] = (1, 3, 5)
    # The single frozen cell the T1 PASS/FAIL verdict is computed on. Named
    # explicitly so reordering a tuple can never move the scientific decision.
    primary_t_start: float = 0.50
    primary_nfe: int = 1
    # Rows used only to recompute the validation tangent-flow MSE.
    mse_rows: int = 65536
    mse_row_seed: int = 20260911
    mse_batches: int = 16
    mse_batch_size: int = 1024
    mse_seed: int = 20260912
    # Which corruption path this spec evaluates, and the arm name its cells are
    # keyed by. The frozen linear-path spec keeps "tangent" so every historical
    # cell key, result column and receipt is byte-identical.
    objective: str = TANGENT_OBJECTIVE
    arm_label: str = "tangent"

    def __post_init__(self) -> None:
        if self.n_sequences % self.n_directions:
            raise ValueError("n_sequences must be a whole multiple of n_directions")
        if not all(0.0 < t <= 1.0 for t in self.t_start):
            raise ValueError("T1 corruption times must lie inside (0, 1]")
        if not all(n >= 1 for n in self.nfe):
            raise ValueError("NFE settings must be positive")
        if self.primary_t_start not in self.t_start:
            raise ValueError("the primary t_start must be evaluated by the grid")
        if self.primary_nfe not in self.nfe:
            raise ValueError("the primary NFE must be evaluated by the grid")

    def primary_cell(self) -> str:
        """The one cell key the formal T1 verdict is read from."""

        return f"t{self.primary_t_start:.2f}_nfe{self.primary_nfe}_{self.arm_label}"

    def corruption_cell(self) -> str:
        """The corrupted control matching the primary cell."""

        return f"t{self.primary_t_start:.2f}_corrupted"


TANGENT_RECONSTRUCTION_SPEC = TangentReconstructionSpec()

# Post-stop experiment A (docs/POST_STOP_PROTOCOL_2026-08-19.md section 2). Same
# plan, same seeds, same directions, same sequences; only the corruption path and
# the time grid differ. The grid is the MATCHED-SEVERITY image of the frozen
# linear grid, so the two runs are compared at equal orthogonal noise-to-signal
# rather than at equal t.
VP_TANGENT_RECONSTRUCTION_SPEC = TangentReconstructionSpec(
    version="vp_tangent_reconstruction_a_t1_v1",
    t_start=tuple(matched_vp_time(t) for t in TangentReconstructionSpec.t_start),
    primary_t_start=matched_vp_time(TangentReconstructionSpec.primary_t_start),
    objective=VP_TANGENT_OBJECTIVE,
    arm_label="vp_tangent",
)


@dataclass(frozen=True)
class TangentNaturalizationSpec:
    """Frozen T2 inference grid and the single experiment-level decision rule.

    The sequence/direction/target plan is `natural_support_v1`, reused verbatim
    so the hard-clamp baseline stays comparable to
    `results/constrained_naturalization_v1/`. Only the inference grid and the
    decision rule are named here.
    """

    version: str = "tangent_naturalization_t2_v1"
    t_start: tuple[float, ...] = (0.10, 0.25, 0.50)
    nfe: tuple[int, ...] = (1, 3)
    # THE experiment-level decision. One frozen operating point, pooled across
    # the five natural-support target quantiles, so there is exactly one primary
    # statistic and no multiplicity to control. Every other cell is diagnostic.
    #
    # t_start 0.10 / NFE 1 is chosen because it is simultaneously the cheapest
    # point in the grid and the one most likely to help: the method must beat a
    # clamp that already costs only +0.003 to +0.054 nats, and larger t_start
    # buys correction by destroying more of the activation. Choosing the cheap
    # point is the cheap-prior motivation of the whole programme, not a guess at
    # where the effect will be largest.
    primary_t_start: float = 0.10
    primary_nfe: int = 1
    primary_pools_quantiles: bool = True
    objective: str = TANGENT_OBJECTIVE
    arm_label: str = "tangent_flow"

    def __post_init__(self) -> None:
        if self.primary_t_start not in self.t_start:
            raise ValueError("the primary t_start must be evaluated by the grid")
        if self.primary_nfe not in self.nfe:
            raise ValueError("the primary NFE must be evaluated by the grid")

    def primary_cell(self) -> str:
        return f"pooled_t{self.primary_t_start:.2f}_nfe{self.primary_nfe}_{self.arm_label}"


TANGENT_NATURALIZATION_SPEC = TangentNaturalizationSpec()

# Post-stop experiment A, T2 half: the same frozen decision rule at the
# matched-severity image of the frozen inference grid.
VP_TANGENT_NATURALIZATION_SPEC = TangentNaturalizationSpec(
    version="vp_tangent_naturalization_a_t2_v1",
    t_start=tuple(matched_vp_time(t) for t in TangentNaturalizationSpec.t_start),
    primary_t_start=matched_vp_time(TangentNaturalizationSpec.primary_t_start),
    objective=VP_TANGENT_OBJECTIVE,
    arm_label="vp_tangent_flow",
)

# Dispatch tables: the spec is chosen by the checkpoint's own recorded objective,
# never by an operator flag, so a model can never be evaluated on the wrong path.
RECONSTRUCTION_SPECS = {
    TANGENT_OBJECTIVE: TANGENT_RECONSTRUCTION_SPEC,
    VP_TANGENT_OBJECTIVE: VP_TANGENT_RECONSTRUCTION_SPEC,
}
NATURALIZATION_SPECS = {
    TANGENT_OBJECTIVE: TANGENT_NATURALIZATION_SPEC,
    VP_TANGENT_OBJECTIVE: VP_TANGENT_NATURALIZATION_SPEC,
}


def reconstruction_spec_for(objective: str) -> TangentReconstructionSpec:
    """The frozen T1 plan for a checkpoint's recorded corruption path."""

    try:
        return RECONSTRUCTION_SPECS[objective]
    except KeyError:
        raise ValueError(f"no frozen T1 plan for objective {objective!r}") from None


def naturalization_spec_for(objective: str) -> TangentNaturalizationSpec:
    """The frozen T2 plan for a checkpoint's recorded corruption path."""

    try:
        return NATURALIZATION_SPECS[objective]
    except KeyError:
        raise ValueError(f"no frozen T2 plan for objective {objective!r}") from None


def spec_payload(spec: TangentReconstructionSpec = TANGENT_RECONSTRUCTION_SPEC) -> dict:
    return asdict(spec)


# --------------------------------------------------------------------------
# direction-clustered bootstrap
# --------------------------------------------------------------------------

# The formal bootstrap unit for every T1 and T2 gate in this branch.
BOOTSTRAP_UNIT = "direction_cluster_then_sequence"


def clustered_bootstrap_mean_ci(
    effects: np.ndarray,
    clusters: np.ndarray,
    *,
    seed: int,
    n_resamples: int,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Hierarchical bootstrap: resample directions, then sequences within them.

    The evaluation plan assigns several validation sequences to each direction,
    so sequences are **not** independent: sequences sharing a direction share
    that direction's idiosyncrasies. A flat sequence-level bootstrap treats
    ``n_sequences`` as the effective sample size when the real one is closer to
    ``n_directions``, which understates the interval.

    Procedure, matching the standard two-stage cluster bootstrap:

        for each resample:
            draw len(directions) directions with replacement
            for each drawn direction, draw its own sequences with replacement
            take the mean over the pooled draws

    This is the canonical interval for formal gates. A sequence-level interval
    may be reported alongside for continuity with historical results, but the
    two are not interchangeable and must never be compared as if they were.
    """

    values = np.asarray(effects, dtype=np.float64)
    groups = np.asarray(clusters)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("effects must be a nonempty one-dimensional vector")
    if groups.shape != values.shape:
        raise ValueError("one cluster label is required per effect")
    if not np.isfinite(values).all():
        raise ValueError("effects must be finite")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples < 1:
        raise ValueError("bootstrap n_resamples must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie strictly inside (0, 1)")

    unique = np.unique(groups)
    members = [np.flatnonzero(groups == label) for label in unique]
    generator = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for draw in range(n_resamples):
        picked = generator.integers(0, len(unique), size=len(unique))
        pooled: list[np.ndarray] = []
        for index in picked:
            rows = members[index]
            pooled.append(generator.choice(rows, size=len(rows), replace=True))
        means[draw] = values[np.concatenate(pooled)].mean()
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return {
        "mean": float(values.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": float(confidence),
        "n_units": int(len(unique)),
        "n_observations": int(values.size),
        "n_resamples": int(n_resamples),
        "unit": BOOTSTRAP_UNIT,
        "seed": int(seed),
    }


# --------------------------------------------------------------------------
# matched corruption / reconstruction pair
# --------------------------------------------------------------------------


def _broadcast_hyperplane(
    model: ConditionalFlowMatcher,
    h: torch.Tensor,
    direction: torch.Tensor,
    c_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    v_x, c_x = standardized_hyperplane(model.normalizer, direction, c_target)
    if v_x.shape[0] == 1 and h.shape[0] != 1:
        v_x = v_x.expand(h.shape[0], v_x.shape[1])
    if c_x.shape[0] == 1 and h.shape[0] != 1:
        c_x = c_x.expand(h.shape[0], 1)
    return (
        v_x.to(device=h.device, dtype=h.dtype).contiguous(),
        c_x.to(device=h.device, dtype=h.dtype).contiguous(),
    )


@torch.no_grad()
def tangent_corrupted_activation(
    model: ConditionalFlowMatcher,
    h: torch.Tensor,
    direction: torch.Tensor,
    c_target: torch.Tensor,
    *,
    noise: torch.Tensor,
    t_start: float,
    objective: str = TANGENT_OBJECTIVE,
) -> torch.Tensor:
    """The exact raw state ``clamp_then_tangent_flow`` starts its integration from.

    Using this as the corrupted control guarantees the control and the
    reconstruction are the same trajectory with and without the model, rather
    than two independently drawn corruptions.
    """

    seed = clamp_seed(h, direction, c_target)
    if t_start == 0.0:
        return seed
    v_x, c_x = _broadcast_hyperplane(model, h, direction, c_target)
    _, x_t, _ = tangent_path_states(
        model.normalizer.normalize(seed), v_x, c_x, noise, t_start, objective=objective
    )
    return model.normalizer.denormalize(x_t)


@torch.no_grad()
def natural_coordinate(h: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Each activation's own coordinate ``<h, v>``, the T1 conditioning target."""

    return (h * direction).sum(dim=-1, keepdim=True)


# --------------------------------------------------------------------------
# per-row geometry
# --------------------------------------------------------------------------


@torch.no_grad()
def tangent_geometry(
    clean: torch.Tensor,
    produced: torch.Tensor,
    direction: torch.Tensor,
    c_target: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Per-row geometry of one arm against the clean activation.

    ``tangent_error`` is the norm of the component of ``produced - clean``
    orthogonal to ``v``: the part the flow is allowed to change, and therefore
    the only part a reconstruction claim can be about.
    """

    delta = produced - clean
    parallel = (delta * direction).sum(dim=-1, keepdim=True)
    orthogonal = delta - parallel * direction
    realised = (produced * direction).sum(dim=-1, keepdim=True)
    clean_norm = clean.norm(dim=-1)

    def numpy(value: torch.Tensor) -> np.ndarray:
        return value.double().cpu().numpy()

    return {
        "relative_l2_to_clean": numpy(delta.norm(dim=-1) / clean_norm),
        "cosine_to_clean": numpy(
            torch.nn.functional.cosine_similarity(produced, clean, dim=-1)
        ),
        "tangent_error": numpy(orthogonal.norm(dim=-1)),
        "parallel_error": numpy(parallel.abs().squeeze(-1)),
        "coordinate_abs_error": numpy((realised - c_target).abs().squeeze(-1)),
        "c_target": numpy(c_target.squeeze(-1)),
        "c_realised": numpy(realised.squeeze(-1)),
        "c0_clean": numpy((clean * direction).sum(dim=-1)),
    }


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Join per-hook-batch row records into one record per field."""

    if not parts:
        raise ValueError("nothing to concatenate")
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------


def reconstruction_summary(
    rows: dict[str, np.ndarray],
    nll: np.ndarray,
    clean_nll: np.ndarray,
    corrupted_nll: np.ndarray,
    assignment: np.ndarray,
    *,
    spec: TangentReconstructionSpec = TANGENT_RECONSTRUCTION_SPEC,
) -> dict:
    """T1 summary: did the flow recover the damage its own corruption caused?

    ``recovered_fraction`` is the share of the corruption's LM damage removed.
    It is undefined when the corruption did no damage, so the raw deltas are
    always reported alongside it and the fraction is omitted rather than faked.

    ``assignment`` gives each validation sequence's direction, which is the
    bootstrap cluster. The canonical interval is the clustered one; the
    sequence-level interval is reported alongside for continuity with historical
    results and is explicitly labelled as not comparable to it.
    """

    delta_lm = float(nll.mean() - clean_nll.mean())
    corruption_delta = float(corrupted_nll.mean() - clean_nll.mean())
    paired = nll - corrupted_nll
    summary: dict[str, object] = {
        "mean_nll": float(nll.mean()),
        "mean_clean_nll": float(clean_nll.mean()),
        "delta_lm_vs_clean": delta_lm,
        "corruption_delta_lm": corruption_delta,
        "recovered_damage": corruption_delta - delta_lm,
        "paired_delta_nll_vs_corrupted": clustered_bootstrap_mean_ci(
            paired,
            assignment,
            seed=spec.bootstrap_seed,
            n_resamples=spec.bootstrap_resamples,
            confidence=spec.confidence,
        ),
        "paired_delta_nll_vs_corrupted_sequence_level": {
            **paired_bootstrap_mean_ci(
                paired,
                seed=spec.bootstrap_seed,
                n_resamples=spec.bootstrap_resamples,
                confidence=spec.confidence,
            ),
            "note": (
                "non-canonical: ignores direction clustering, reported only for "
                "continuity with historical sequence-level intervals; not "
                "comparable to the clustered CI and never used by a gate"
            ),
        },
        "mean_relative_l2": float(rows["relative_l2_to_clean"].mean()),
        "mean_cosine": float(rows["cosine_to_clean"].mean()),
        "mean_tangent_error": float(rows["tangent_error"].mean()),
        "mean_parallel_error": float(rows["parallel_error"].mean()),
        "coordinate_abs_error_mean": float(rows["coordinate_abs_error"].mean()),
        "coordinate_abs_error_max": float(rows["coordinate_abs_error"].max()),
        "n_rows": int(rows["tangent_error"].size),
    }
    if corruption_delta > 0.0:
        summary["recovered_fraction"] = (corruption_delta - delta_lm) / corruption_delta
    else:
        summary["recovered_fraction"] = None
        summary["recovered_fraction_note"] = (
            "corruption caused no measurable LM damage; a recovered fraction "
            "would be meaningless here"
        )
    return summary


def naturalization_summary(
    rows: dict[str, np.ndarray],
    nll: np.ndarray,
    clamp_nll: np.ndarray,
    assignment: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    confidence: float,
) -> dict:
    """T2 summary. The primary quantity is ``NLL_tangent - NLL_clamp``, negative = useful.

    Every supporting field exists to stop that number being believed for the
    wrong reason: a coordinate error above float noise means the arms are no
    longer at the same semantic coordinate and the comparison is void.
    """

    paired = nll - clamp_nll
    by_direction = {
        int(index): float(paired[assignment == index].mean())
        for index in np.unique(assignment)
    }
    values = np.array(list(by_direction.values()), dtype=np.float64)
    lovo = [float(paired[assignment != index].mean()) for index in np.unique(assignment)]
    return {
        "mean_nll": float(nll.mean()),
        "primary_paired_delta_nll_vs_clamp": clustered_bootstrap_mean_ci(
            paired,
            assignment,
            seed=bootstrap_seed,
            n_resamples=bootstrap_resamples,
            confidence=confidence,
        ),
        "paired_delta_nll_vs_clamp_sequence_level": {
            **paired_bootstrap_mean_ci(
                paired,
                seed=bootstrap_seed,
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            "note": (
                "non-canonical: ignores direction clustering, reported only for "
                "continuity with historical sequence-level intervals; not "
                "comparable to the clustered CI and never used by a gate"
            ),
        },
        "coordinate_abs_error_mean": float(rows["coordinate_abs_error"].mean()),
        "coordinate_abs_error_max": float(rows["coordinate_abs_error"].max()),
        "orthogonal_correction_norm_mean": float(rows["orthogonal_correction_norm"].mean()),
        "relative_l2_to_clean_mean": float(rows["relative_l2_to_clean"].mean()),
        "cosine_to_clean_mean": float(rows["cosine_to_clean"].mean()),
        "per_direction_paired_delta_nll": by_direction,
        "fraction_directions_negative": float((values < 0).mean()),
        "lovo_paired_delta_nll_min": float(np.min(lovo)),
        "lovo_paired_delta_nll_max": float(np.max(lovo)),
    }


# --------------------------------------------------------------------------
# validity gates
# --------------------------------------------------------------------------


def assert_coordinate_match(
    clamp_rows: dict[str, np.ndarray],
    flow_rows: dict[str, np.ndarray],
    *,
    tolerance: float = 1e-3,
) -> dict[str, float]:
    """Refuse to report a T2 cell whose two arms are not at the same coordinate.

    Without this, the flow arm could "win" by attenuating the coordinate, which
    is precisely the failure mode the closed branch already produced.
    """

    difference = np.abs(flow_rows["c_realised"] - clamp_rows["c_realised"])
    worst = float(difference.max())
    if worst > tolerance:
        raise ValueError(
            f"clamp and tangent-flow arms differ in realised coordinate by up to "
            f"{worst:.3e} (> {tolerance:.1e}); the paired NLL comparison is not "
            "at a fixed semantic coordinate and must not be reported"
        )
    return {
        "max_arm_coordinate_difference": worst,
        "mean_arm_coordinate_difference": float(difference.mean()),
        "tolerance": float(tolerance),
    }


# Frozen T1 materiality threshold. Human-approved 2026-08-16, before any real T1
# result existed. It is a pragmatic preregistered convention, NOT a theoretically
# derived constant -- there is no derivation behind 0.25, and none is claimed.
#
# Changing it after a T1 result has been observed would convert a preregistered
# gate into a post-hoc one and invalidate the T1 interpretation. If a future
# experiment needs a different threshold, that is a new protocol version with a
# new experiment id, decided before its results are seen.
T1_MIN_RECOVERED_FRACTION = 0.25


def t1_verdict(
    arms: dict[str, dict],
    *,
    spec: TangentReconstructionSpec = TANGENT_RECONSTRUCTION_SPEC,
    min_recovered_fraction: float = T1_MIN_RECOVERED_FRACTION,
    formal_eligible: bool = True,
    ineligible_reason: str | None = None,
) -> dict[str, object]:
    """Frozen T1 gate, read from the explicitly named primary cell.

    The cell is ``spec.primary_cell()``, built from ``primary_t_start`` and
    ``primary_nfe``. It is never taken from a tuple position, so reordering the
    diagnostic grid cannot move the scientific decision.

    ``formal_eligible=False`` (an unselected checkpoint, an unvalidated bundle)
    produces a diagnostic-only result: no PASS/FAIL is issued at all, rather
    than a verdict a reader might mistake for the formal one.
    """

    cell = spec.primary_cell()
    corruption_cell = spec.corruption_cell()
    if cell not in arms or corruption_cell not in arms:
        raise KeyError(
            f"T1 primary cell {cell!r} or its control {corruption_cell!r} is absent; "
            "the frozen gate cannot be evaluated"
        )
    tangent = arms[cell]
    corrupted_delta_lm = float(arms[corruption_cell]["delta_lm_vs_clean"])

    recovered = tangent.get("recovered_fraction")
    ci = tangent["paired_delta_nll_vs_corrupted"]
    improves = float(ci["ci_upper"]) < 0.0
    passes = (
        corrupted_delta_lm > 0.0
        and recovered is not None
        and float(recovered) >= min_recovered_fraction
        and improves
    )
    verdict: dict[str, object] = {
        "gate": "T1",
        "primary_cell": cell,
        "primary_t_start": spec.primary_t_start,
        "primary_nfe": spec.primary_nfe,
        "corruption_control_cell": corruption_cell,
        "corruption_delta_lm": corrupted_delta_lm,
        "min_recovered_fraction": float(min_recovered_fraction),
        "threshold_status": (
            "frozen 2026-08-16 before any T1 result; pragmatic preregistered "
            "convention, not a theoretically derived constant"
        ),
        "bootstrap_unit": ci.get("unit"),
        "recovered_fraction": recovered,
        "paired_ci_excludes_zero_and_is_negative": improves,
        "formal_verdict_eligible": bool(formal_eligible),
    }
    if not formal_eligible:
        verdict["verdict"] = "DIAGNOSTIC_ONLY"
        verdict["would_have_been"] = "PASS" if passes else "FAIL"
        verdict["ineligible_reason"] = (
            ineligible_reason or "the run was not eligible for a formal verdict"
        )
        verdict["consequence"] = (
            "this result carries no formal T1 verdict and cannot authorize T2"
        )
        return verdict
    verdict["verdict"] = "PASS" if passes else "FAIL"
    verdict["consequence"] = (
        "T2 is worth running"
        if passes
        else "stop and diagnose implementation/training before any steering-like "
        "evaluation; do not proceed to T2"
    )
    return verdict


def t2_cell_report(cell: dict) -> dict[str, object]:
    """Per-cell diagnostic view. **This is not the experiment-level decision.**

    Every non-primary cell gets one of these. Reading a PASS here as "T2 passed"
    is exactly the multiplicity error the experiment-level rule exists to
    prevent: with 5 quantiles x 3 t_start x 2 NFE there are 30 cells, and some
    will look favourable by chance alone.
    """

    ci = cell["primary_paired_delta_nll_vs_clamp"]
    negative = float(ci["ci_upper"]) < 0.0
    homogeneous = float(cell["fraction_directions_negative"]) > 0.80
    lovo_stable = float(cell["lovo_paired_delta_nll_max"]) < 0.0
    return {
        "scope": "single_cell_diagnostic",
        "not_the_experiment_verdict": True,
        "paired_delta_nll_mean": float(ci["mean"]),
        "bootstrap_unit": ci.get("unit"),
        "ci_excludes_zero_and_is_negative": negative,
        "fraction_directions_negative": float(cell["fraction_directions_negative"]),
        "direction_signs_homogeneous": homogeneous,
        "lovo_stable": lovo_stable,
        "cell_favourable": negative and homogeneous and lovo_stable,
    }


def t2_experiment_verdict(
    pooled_cell: dict,
    *,
    spec: TangentNaturalizationSpec = TANGENT_NATURALIZATION_SPEC,
    formal_eligible: bool = True,
    ineligible_reason: str | None = None,
) -> dict[str, object]:
    """The single experiment-level T2 decision.

    Computed on ONE frozen operating point (``spec.primary_t_start`` /
    ``spec.primary_nfe``), pooled across the five natural-support target
    quantiles. One statistic, one decision, no multiplicity correction needed
    because no selection over cells takes place.

    Explicitly **not** the rule: "T2 passes if any grid cell passes".
    """

    ci = pooled_cell["primary_paired_delta_nll_vs_clamp"]
    negative = float(ci["ci_upper"]) < 0.0
    homogeneous = float(pooled_cell["fraction_directions_negative"]) > 0.80
    lovo_stable = float(pooled_cell["lovo_paired_delta_nll_max"]) < 0.0
    coordinate_ok = (
        float(pooled_cell["coordinate_abs_error_max"]) < 1e-2
        and pooled_cell.get("coordinate_match", {}).get("max_arm_coordinate_difference", 0.0)
        < 1e-3
    )
    passes = negative and homogeneous and lovo_stable and coordinate_ok
    verdict: dict[str, object] = {
        "gate": "T2",
        "scope": "experiment_level",
        "rule": (
            "single frozen operating point (t_start "
            f"{spec.primary_t_start:.2f}, NFE {spec.primary_nfe}) pooled across "
            "the natural-support target quantiles; NOT any-cell-passes"
        ),
        "primary_cell": spec.primary_cell(),
        "primary_t_start": spec.primary_t_start,
        "primary_nfe": spec.primary_nfe,
        "pools_target_quantiles": spec.primary_pools_quantiles,
        "paired_delta_nll_mean": float(ci["mean"]),
        "paired_delta_nll_ci": [float(ci["ci_lower"]), float(ci["ci_upper"])],
        "bootstrap_unit": ci.get("unit"),
        "ci_excludes_zero_and_is_negative": negative,
        "fraction_directions_negative": float(pooled_cell["fraction_directions_negative"]),
        "direction_signs_homogeneous": homogeneous,
        "lovo_stable": lovo_stable,
        "coordinate_held_fixed": coordinate_ok,
        "formal_verdict_eligible": bool(formal_eligible),
    }
    if not formal_eligible:
        verdict["verdict"] = "DIAGNOSTIC_ONLY"
        verdict["would_have_been"] = "PASS" if passes else "FAIL"
        verdict["ineligible_reason"] = (
            ineligible_reason or "the run was not eligible for a formal verdict"
        )
        verdict["consequence"] = "this result carries no formal T2 verdict"
        return verdict
    verdict["verdict"] = "PASS" if passes else "FAIL"
    verdict["consequence"] = (
        "a 60M confirmation or eventual DEV experiment may be justified"
        if passes
        else STOP_RULE
    )
    return verdict


# --------------------------------------------------------------------------
# checkpoint-selection provenance
# --------------------------------------------------------------------------


def verify_selected_checkpoint(
    checkpoint_path: Path,
    checkpoint_meta: dict,
    *,
    run_dir: Path,
    expected_objective: str = TANGENT_OBJECTIVE,
    selection_metric: str = FROZEN_SELECTION_METRIC,
) -> dict[str, object]:
    """Prove the supplied checkpoint IS the run's concept-independent selection.

    A CLI path is an assertion by whoever typed it. This turns the claim
    "selected by ``val_flow_mse``, concept-independently" into something the
    filesystem has to agree with:

    * the run directory's ``best.json`` names this exact filename;
    * ``best.json`` selected on the frozen metric, minimizing;
    * the run's ``meta.json`` agrees on both the metric and the best checkpoint;
    * the checkpoint's own embedded metadata carries the same experiment id and
      config fingerprint as the run;
    * the objective identity is the tangent one.

    Raises on any mismatch. The returned receipt is recorded verbatim in the
    result file, including the checkpoint SHA-256.
    """

    checkpoint_path = Path(checkpoint_path)
    run_dir = Path(run_dir)
    best_pointer = run_dir / "best.json"
    run_meta_path = run_dir / "meta.json"
    for path in (best_pointer, run_meta_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"checkpoint-selection provenance requires {path}; a formal T1 "
                "verdict cannot be issued from a bare checkpoint file"
            )
    best = json.loads(best_pointer.read_text())
    run_meta = json.loads(run_meta_path.read_text())

    if best.get("selection_metric") != selection_metric:
        raise ValueError(
            f"run selected on {best.get('selection_metric')!r}, not the frozen "
            f"{selection_metric!r}; the concept-independence claim does not hold"
        )
    if best.get("selection_mode") != "min":
        raise ValueError(f"selection mode {best.get('selection_mode')!r} is not 'min'")
    if run_meta.get("selection_metric") != selection_metric:
        raise ValueError("run metadata and best pointer disagree on the selection metric")
    if best.get("checkpoint") != run_meta.get("best_checkpoint"):
        raise ValueError("run metadata and best pointer name different checkpoints")
    if checkpoint_path.name != best.get("checkpoint"):
        raise ValueError(
            f"checkpoint {checkpoint_path.name!r} is not the run's selected checkpoint "
            f"{best.get('checkpoint')!r}; pass --allow-unselected-checkpoint to record "
            "it as a diagnostic result with no formal verdict"
        )
    if checkpoint_path.resolve().parent != run_dir.resolve():
        raise ValueError("the checkpoint does not belong to the supplied run directory")

    for field in ("experiment_id", "config_fingerprint"):
        if checkpoint_meta.get(field) != run_meta.get(field):
            raise ValueError(
                f"checkpoint {field}={checkpoint_meta.get(field)!r} != run "
                f"{field}={run_meta.get(field)!r}"
            )
    objective = (checkpoint_meta.get("objective_identity") or {}).get("flow_objective")
    if objective != expected_objective:
        raise ValueError(
            f"checkpoint objective {objective!r} != expected {expected_objective!r}"
        )

    return {
        "verified": True,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "run_dir": str(run_dir),
        "experiment_id": run_meta.get("experiment_id"),
        "config_fingerprint": run_meta.get("config_fingerprint"),
        "source_revision": run_meta.get("source_revision"),
        "selection_metric": selection_metric,
        "selection_mode": "min",
        "selection_value": best.get("value"),
        "selection_is_concept_independent": True,
        "objective_identity": checkpoint_meta.get("objective_identity"),
        "step": checkpoint_meta.get("step"),
    }


def unselected_checkpoint_receipt(
    checkpoint_path: Path, checkpoint_meta: dict, reason: str
) -> dict[str, object]:
    """Receipt for a deliberately non-selected checkpoint: diagnostic use only."""

    checkpoint_path = Path(checkpoint_path)
    return {
        "verified": False,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "selection_is_concept_independent": False,
        "objective_identity": checkpoint_meta.get("objective_identity"),
        "step": checkpoint_meta.get("step"),
        "formal_verdict_eligible": False,
        "reason": reason,
    }


# --------------------------------------------------------------------------
# direction-pool compatibility
# --------------------------------------------------------------------------


def verify_direction_pool(
    checkpoint_meta: dict, pool: TrainingDirectionPool
) -> dict[str, object]:
    """Require the evaluation pool to be the one the checkpoint trained on.

    Two same-shaped pools with different contents are silently interchangeable
    without this check, and the conditioning the model learned would be
    evaluated against directions it never saw.
    """

    trained = checkpoint_meta.get("direction_pool")
    if not isinstance(trained, dict):
        raise ValueError(
            "checkpoint records no direction pool; it cannot be evaluated against one"
        )
    observed = pool.identity()
    if trained.get("digest") != observed.get("digest"):
        raise ValueError(
            f"direction pool digest {observed.get('digest')} does not match the "
            f"pool the checkpoint trained on ({trained.get('digest')})"
        )
    differing = sorted(
        key for key in set(trained) | set(observed) if trained.get(key) != observed.get(key)
    )
    if differing:
        raise ValueError(
            f"direction pool identity differs from the training pool in {differing}"
        )
    return {"verified": True, **observed}


# --------------------------------------------------------------------------
# activation / token artifact bundle
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationBundle:
    """A fully validated (activations, tokens, split) triple.

    T1 and T2 consume this rather than opening loosely related files, so a
    shape-compatible but scientifically unrelated array and token cache cannot
    be paired by accident.
    """

    dataset: ActivationDataset
    tokens: torch.Tensor
    split: ActivationSplit
    identity: dict[str, object]

    @property
    def activations(self) -> np.ndarray:
        return self.dataset.array

    @property
    def meta(self) -> dict:
        return self.dataset.meta


def load_validated_evaluation_bundle(
    name: str,
    activation_dir: Path,
    token_cache_dir: Path,
    *,
    hook: str = STEERING_HOOK,
    per_seq: int,
    val_fraction: float,
    split_seed: int,
    d_model: int = 768,
    expected_split: str = "val",
) -> EvaluationBundle:
    """Load and fully validate the artifacts an evaluator is about to use.

    Applies the same identity checks training applies, plus the pairing checks
    an evaluator specifically needs:

    * the full-scan validation report is VALID and its artifact SHA-256 set is
      re-verified against the bytes on disk right now;
    * split identity and fingerprint agree with the report;
    * GPT-2 name, resolved name and revision; tokenizer; hook; context length;
      activation width; BOS-dropped invariant; FineWeb repository/config/revision;
    * the token cache named by the metadata hashes to the recorded digest, and
      the report agrees with the metadata on that digest;
    * token cache shape matches ``n_seqs x ctx``;
    * the activation count is exactly ``n_seqs * (ctx - 1)``.
    """

    activation_dir = Path(activation_dir)
    token_cache_dir = Path(token_cache_dir)
    dataset = load_activations(name, activation_dir)
    meta = dataset.meta
    if meta.get("split") != expected_split:
        raise ValueError(
            f"artifact split {meta.get('split')!r} != expected {expected_split!r}"
        )

    split = make_split(len(dataset), per_seq, val_fraction, split_seed)
    report = load_validation_report(
        name,
        activation_dir,
        expected_split_fingerprint=split.fingerprint(),
        verify_hashes=True,
    )
    validate_activation_metadata(
        dataset,
        expected_name=name,
        expected_split=expected_split,
        expected_model=MODEL_NAME,
        expected_resolved_model_name=MODEL_RESOLVED_NAME,
        expected_model_revision=MODEL_REVISION,
        expected_hook=hook,
        expected_ctx=per_seq + 1,
        expected_d_model=d_model,
        expected_dataset_repository=meta["dataset_repository"],
        expected_dataset_config=meta["dataset_config"],
        expected_dataset_revision=meta["dataset_revision"],
        expected_tokenizer=meta["tokenizer"],
    )

    cache_name = meta.get("token_cache_file")
    if not isinstance(cache_name, str) or not cache_name:
        raise ValueError("activation metadata names no token cache file")
    cache_path = token_cache_dir / cache_name
    if not cache_path.is_file():
        raise FileNotFoundError(f"token cache does not exist: {cache_path}")
    observed_cache_sha = file_sha256(cache_path)
    if observed_cache_sha != meta.get("token_cache_sha256"):
        raise ValueError(
            f"token cache {cache_path.name} hashes to {observed_cache_sha}, but the "
            f"activation artifact was built from {meta.get('token_cache_sha256')}; "
            "these files do not belong together"
        )
    if report.get("token_cache_sha256") != meta.get("token_cache_sha256"):
        raise ValueError("validation report and metadata disagree on the token cache")

    tokens = torch.from_numpy(np.load(cache_path))
    expected_tokens = (int(meta["n_seqs"]), int(meta["ctx"]))
    if tuple(tokens.shape) != expected_tokens:
        raise ValueError(
            f"token cache shape {tuple(tokens.shape)} != expected {expected_tokens}"
        )
    expected_rows = int(meta["n_seqs"]) * per_seq
    if len(dataset) != expected_rows:
        raise ValueError(
            f"activation count {len(dataset)} != n_seqs*{per_seq} = {expected_rows}; "
            "activations and tokens are not aligned"
        )

    identity = {
        "name": name,
        "artifact_sha256": report.get("sha256"),
        "token_cache_file": cache_name,
        "token_cache_sha256": observed_cache_sha,
        "split_fingerprint": split.fingerprint(),
        "split": expected_split,
        "hook": hook,
        "model": meta.get("model"),
        "resolved_model_name": meta.get("resolved_model_name"),
        "model_revision": meta.get("model_revision"),
        "tokenizer": meta.get("tokenizer"),
        "ctx": int(meta["ctx"]),
        "per_seq": per_seq,
        "d_model": dataset.d_model,
        "n_seqs": int(meta["n_seqs"]),
        "n_activations": len(dataset),
        "bos_dropped": meta.get("bos_dropped"),
        "dataset_repository": meta.get("dataset_repository"),
        "dataset_config": meta.get("dataset_config"),
        "dataset_revision": meta.get("dataset_revision"),
        "validation_report_status": report.get("status"),
    }
    return EvaluationBundle(dataset=dataset, tokens=tokens, split=split, identity=identity)


# --------------------------------------------------------------------------
# T1 -> T2 receipt, and result immutability
# --------------------------------------------------------------------------


def write_t1_receipt(path: Path, payload: dict) -> dict[str, object]:
    """Distil a completed T1 result into the receipt T2 requires."""

    receipt = {
        "receipt": "tangent_t1_v1",
        "experiment": payload.get("experiment"),
        "verdict": payload["t1_gate"]["verdict"],
        "primary_cell": payload["t1_gate"]["primary_cell"],
        "primary_t_start": payload["t1_gate"]["primary_t_start"],
        "primary_nfe": payload["t1_gate"]["primary_nfe"],
        "formal_verdict_eligible": payload["t1_gate"]["formal_verdict_eligible"],
        "checkpoint_sha256": payload["checkpoint_selection"]["checkpoint_sha256"],
        "objective_identity": payload["checkpoint_selection"].get("objective_identity"),
        "direction_pool_digest": payload["direction_pool"]["digest"],
        "config_fingerprint": payload["checkpoint_selection"].get("config_fingerprint"),
        "source_revision": payload.get("source_revision"),
    }
    Path(path).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def verify_t1_pass_receipt(
    path: Path,
    *,
    checkpoint_sha256: str,
    pool_identity: dict,
    objective_identity: dict | None,
    spec: TangentReconstructionSpec = TANGENT_RECONSTRUCTION_SPEC,
) -> dict[str, object]:
    """T2 refuses to run without a formal T1 PASS on the very same artifacts.

    T2 is only meaningful once the model has been shown to solve its own matched
    task. Without this, a T2 number could be produced from a model that failed
    T1 -- or from a different checkpoint entirely -- and read as if the staged
    protocol had been followed.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"T2 requires a formal T1 PASS receipt; none found at {path}. "
            "Run T1 first: an unvalidated T2 number has no protocol standing."
        )
    receipt = json.loads(path.read_text())
    if receipt.get("receipt") != "tangent_t1_v1":
        raise ValueError(f"{path} is not a tangent T1 receipt")
    if not receipt.get("formal_verdict_eligible"):
        raise ValueError("the T1 receipt is diagnostic-only and cannot authorize T2")
    if receipt.get("verdict") != "PASS":
        raise ValueError(
            f"T1 verdict is {receipt.get('verdict')!r}, not PASS; the protocol stops "
            "at T1 and T2 must not run"
        )
    if receipt.get("primary_cell") != spec.primary_cell():
        raise ValueError(
            f"T1 receipt used primary cell {receipt.get('primary_cell')!r}, but the "
            f"frozen gate is {spec.primary_cell()!r}"
        )
    if receipt.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(
            "T2 checkpoint SHA-256 differs from the checkpoint T1 passed on; the "
            "staged protocol requires the same model"
        )
    if receipt.get("direction_pool_digest") != pool_identity.get("digest"):
        raise ValueError("T2 direction pool differs from the pool T1 used")
    recorded = receipt.get("objective_identity") or {}
    if recorded.get("flow_objective") != spec.objective:
        raise ValueError(
            f"the T1 receipt is not for a {spec.objective!r} checkpoint"
        )
    if objective_identity is not None and recorded != objective_identity:
        raise ValueError("T2 checkpoint objective identity differs from T1's")
    return {"verified": True, **receipt}


def require_fresh_output_dir(path: Path, *, overwrite_debug: bool = False) -> None:
    """Result directories are immutable: refuse to write over an existing one.

    Scientific artifacts are not build outputs. Silently replacing a previous
    result destroys the record of what was observed before a change, which is
    the specific failure docs/RESEARCH_GOVERNANCE.md section 4 exists to prevent.
    """

    path = Path(path)
    if path.exists() and any(path.iterdir()) and not overwrite_debug:
        raise FileExistsError(
            f"result directory {path} already contains artifacts; refusing to "
            "overwrite a scientific result. Write to a new directory, or pass "
            "--overwrite-debug-mode to mark the run as a discardable debug run."
        )
    path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# T2 formal aggregation: equal quantile weight, hierarchical resampling
# --------------------------------------------------------------------------
#
# Specified before any T2 result was observed (human instruction, 2026-08-16).
#
# The formal T2 effect is NOT a mean over concatenated rows. Each of the five
# natural-support target quantiles gets equal weight:
#
#     d_pooled = (1/5) * sum_q mean_i d_{q,i}
#
# and the bootstrap keeps the five quantile observations of a drawn sequence
# together, so dependence across quantiles (same sequence, same direction, same
# noise) is preserved rather than dissolved by row-level resampling.

TARGET_QUANTILE_KEYS = ("q50", "q75", "q90", "q95", "q99")


def equal_weight_pooled_effect(
    effects_by_quantile: dict[str, np.ndarray], assignment: np.ndarray
) -> dict[str, object]:
    """Point estimates for the formal T2 statistic, with per-direction and LOVO.

    ``effects_by_quantile[q]`` is the paired per-sequence effect
    ``NLL_tangent - NLL_clamp`` for target quantile ``q``, aligned so index ``i``
    is the same validation sequence in every quantile. ``assignment`` gives each
    sequence's direction.
    """

    keys = sorted(effects_by_quantile)
    if not keys:
        raise ValueError("no target quantiles survived the validity checks")
    lengths = {len(effects_by_quantile[k]) for k in keys}
    if len(lengths) != 1:
        raise ValueError("every quantile must supply one effect per sequence")
    n = lengths.pop()
    groups = np.asarray(assignment)
    if groups.shape != (n,):
        raise ValueError("assignment must give one direction per sequence")
    stacked = np.stack([np.asarray(effects_by_quantile[k], dtype=np.float64) for k in keys])
    if not np.isfinite(stacked).all():
        raise ValueError("paired effects must be finite")

    per_quantile = {k: float(stacked[i].mean()) for i, k in enumerate(keys)}
    pooled = float(np.mean(list(per_quantile.values())))

    directions = np.unique(groups)
    by_direction: dict[int, float] = {}
    for d in directions:
        rows = groups == d
        by_direction[int(d)] = float(np.mean([stacked[i][rows].mean() for i in range(len(keys))]))
    lovo = []
    for d in directions:
        rows = groups != d
        lovo.append(float(np.mean([stacked[i][rows].mean() for i in range(len(keys))])))
    signs = np.array(list(by_direction.values()), dtype=np.float64)
    return {
        "pooled_mean": pooled,
        "per_quantile_mean": per_quantile,
        "quantiles_used": keys,
        "n_quantiles": len(keys),
        "n_sequences": int(n),
        "n_directions": int(len(directions)),
        "n_paired_rows": int(stacked.size),
        "per_direction_pooled_effect": by_direction,
        "fraction_directions_negative": float((signs < 0).mean()),
        "lovo_min": float(np.min(lovo)),
        "lovo_max": float(np.max(lovo)),
        "weighting": "equal_quantile_weight",
    }


def hierarchical_quantile_bootstrap(
    effects_by_quantile: dict[str, np.ndarray],
    assignment: np.ndarray,
    *,
    seed: int,
    n_resamples: int,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Canonical T2 interval: direction -> sequence, quantile rows kept together.

    Per replicate: resample direction clusters with replacement; within each
    drawn direction resample its sequences with replacement; carry all five
    target-quantile observations of every drawn sequence; take each quantile's
    mean over the drawn multiset; equal-weight the five quantile means.

    A sequence-only bootstrap is not admissible for the formal gate: it would
    treat sequences sharing a direction as independent and would also break the
    dependence between the five quantiles evaluated on the same sequence.
    """

    keys = sorted(effects_by_quantile)
    stacked = np.stack([np.asarray(effects_by_quantile[k], dtype=np.float64) for k in keys])
    groups = np.asarray(assignment)
    if not np.isfinite(stacked).all():
        raise ValueError("paired effects must be finite")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples < 1:
        raise ValueError("bootstrap n_resamples must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie strictly inside (0, 1)")

    directions = np.unique(groups)
    members = [np.flatnonzero(groups == d) for d in directions]
    generator = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        picked = generator.integers(0, len(directions), size=len(directions))
        sequences = np.concatenate(
            [generator.choice(members[i], size=len(members[i]), replace=True) for i in picked]
        )
        # One index vector, applied identically to all five quantiles: the drawn
        # sequence brings its whole quantile row-set with it.
        draws[r] = float(stacked[:, sequences].mean(axis=1).mean())
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    point = float(stacked.mean(axis=1).mean())
    return {
        "mean": point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": float(confidence),
        "n_units": int(len(directions)),
        "n_sequences": int(stacked.shape[1]),
        "n_quantiles": int(stacked.shape[0]),
        "n_observations": int(stacked.size),
        "n_resamples": int(n_resamples),
        "unit": BOOTSTRAP_UNIT,
        "weighting": "equal_quantile_weight",
        "quantile_rows_kept_together": True,
        "seed": int(seed),
    }


def t2_pooled_cell(
    effects_by_quantile: dict[str, np.ndarray],
    assignment: np.ndarray,
    geometry: dict[str, float],
    *,
    seed: int,
    n_resamples: int,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Assemble the single formal T2 cell that ``t2_experiment_verdict`` judges."""

    point = equal_weight_pooled_effect(effects_by_quantile, assignment)
    ci = hierarchical_quantile_bootstrap(
        effects_by_quantile, assignment,
        seed=seed, n_resamples=n_resamples, confidence=confidence,
    )
    return {
        "arm": "clamp_plus_tangent_flow",
        "scope": "pooled_equal_quantile_weight",
        "primary_paired_delta_nll_vs_clamp": ci,
        "per_quantile_paired_delta_nll": point["per_quantile_mean"],
        "quantiles_used": point["quantiles_used"],
        "per_direction_paired_delta_nll": point["per_direction_pooled_effect"],
        "fraction_directions_negative": point["fraction_directions_negative"],
        "lovo_paired_delta_nll_min": point["lovo_min"],
        "lovo_paired_delta_nll_max": point["lovo_max"],
        "n_sequences": point["n_sequences"],
        "n_directions": point["n_directions"],
        "n_paired_rows": point["n_paired_rows"],
        **geometry,
    }


def write_t2_receipt(path: Path, payload: dict) -> dict[str, object]:
    """Distil a completed T2 result into an immutable formal receipt.

    Mirrors :func:`write_t1_receipt`. This is the artifact that records what the
    decisive experiment concluded, including the stop rule it triggers.
    """

    verdict = payload["t2_experiment_verdict"]
    cell = payload["arms"][verdict["primary_cell"]]
    ci = cell["primary_paired_delta_nll_vs_clamp"]
    receipt = {
        "receipt": "tangent_t2_v1",
        "experiment": payload.get("experiment"),
        "verdict": verdict["verdict"],
        "formal_verdict_eligible": verdict["formal_verdict_eligible"],
        "decision_rule": verdict["rule"],
        "primary_cell": verdict["primary_cell"],
        "primary_t_start": verdict["primary_t_start"],
        "primary_nfe": verdict["primary_nfe"],
        "pooled_paired_delta_nll": ci["mean"],
        "pooled_ci": [ci["ci_lower"], ci["ci_upper"]],
        "bootstrap": {
            "unit": ci["unit"],
            "weighting": ci["weighting"],
            "quantile_rows_kept_together": ci["quantile_rows_kept_together"],
            "n_units": ci["n_units"],
            "n_sequences": ci["n_sequences"],
            "n_quantiles": ci["n_quantiles"],
            "n_observations": ci["n_observations"],
            "n_resamples": ci["n_resamples"],
            "seed": ci["seed"],
            "confidence": ci["confidence"],
        },
        "per_quantile_paired_delta_nll": cell["per_quantile_paired_delta_nll"],
        "fraction_directions_negative": cell["fraction_directions_negative"],
        "lovo_min": cell["lovo_paired_delta_nll_min"],
        "lovo_max": cell["lovo_paired_delta_nll_max"],
        "coordinate_abs_error_max": cell["coordinate_abs_error_max"],
        "coordinate_match": cell["coordinate_match"],
        "orthogonal_correction_norm_mean": cell["orthogonal_correction_norm_mean"],
        "parallel_correction_norm_mean": cell["parallel_correction_norm_mean"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "objective_identity": payload.get("checkpoint_objective_identity"),
        "direction_pool_digest": payload["direction_pool"]["digest"],
        "t1_receipt": payload["t1_receipt"],
        "validation_artifact": payload["validation_artifact"],
        "source_revision": payload.get("source_revision"),
        "stop_rule": payload["stop_rule"],
        "stop_rule_fired": verdict["verdict"] == "FAIL",
        "dev_vectors_accessed": payload["dev_vectors_accessed"],
        "held_out_accessed": payload["held_out_accessed"],
        "llm_judge_used": payload["llm_judge_used"],
        "trained_anything": payload["trained_anything"],
    }
    Path(path).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
