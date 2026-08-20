# Branch Closure: Generic / Conditional Flow Naturalization

## Status

**CLOSED — SCIENTIFICALLY COMPLETE**

Closed 2026-08-15 by explicit human instruction. This is not an inconclusive
branch. It reached a decisive negative result on its central mechanism, plus
several durable positive findings that survive the closure.

No further training, DEV evaluation, or inference-trick rescue is authorized for
this branch.

---

# 1. What the branch covered

1. simple activation denoiser;
2. protected / parallel correction variants;
3. unconditional cheap flow prior;
4. capacity/data-scaled unconditional flow;
5. conditional direction-coordinate flow;
6. natural-support controllability;
7. clamp-seed conditional flow;
8. tangent-noise inference;
9. projected / hard-constrained conditional flow.

---

# 2. Final conclusion

> **The learned generic/conditional activation prior does not provide useful
> orthogonal naturalization of steering interventions.**

The strongest and final test is the projected constrained arm. There

    <h_out, v> = c_target

is enforced analytically after every Euler step, so the parallel channel is
closed and the attenuation explanation is structurally unavailable. The flow may
modify only the remaining degrees of freedom.

Those orthogonal corrections systematically **worsen** LM quality relative to an
ordinary hard clamp, and the penalty grows with the magnitude of the orthogonal
correction:

| t_start | orthogonal correction norm | paired ΔNLL vs hard clamp |
|---|---|---|
| 0.10 | 6.82 | +0.0061 |
| 0.25 | 15.30 | +0.0448 |
| 0.50 | 25.59 | +0.2511 |

Therefore the mechanism

    hard semantic constraint
    + generic learned activation-prior correction
    -> improved naturalness

is **not supported** by the current experiments.

Evidence: `results/constrained_naturalization_v1/REPORT.md`. Across 90 flow
cells, 2 had mean ΔNLL < 0 and exactly 1 had a confidence interval below zero —
the unconstrained arm at the most extreme target, explained entirely by
coordinate slip (see §3E).

---

# 3. Positive findings that must be preserved

The branch is not summarized as "flow failed". These stand on their own.

## A. Generic denoising works

The flow prior genuinely learns natural activation reconstruction, and scaling
capacity and data substantially improved Phase-A reconstruction.

That improvement did **not** transfer to steering. The durable lesson:

    better modelling of natural / noise-corrupted activations
    !=
    better correction of structured steering interventions.

## B. Conditional coordinate control works in-distribution

The direction-coordinate conditional model does not ignore its condition. Inside
natural coordinate support it is a strong controller: requested-vs-realised
coordinate slope **0.906** (95% CI 0.867–0.947) at high flow time, with 32 of 32
directions showing positive slope and a leave-one-direction-out pooled range of
0.9242–0.9359.

The earlier C1 failure under `alpha_hat` was primarily an **out-of-distribution
coordinate problem**: requested shifts were roughly **2.5σ–25σ** relative to the
natural coordinate spread (median std 3.602), because `alpha = alpha_hat * 88.76`
scales to activation norm rather than to coordinate spread.

Evidence: `results/natural_support_v1/REPORT.md`,
`results/conditional_c1_precheck_v1/`.

## C. Clean-seed controllability has a reconstruction tradeoff

Strong controllability appears only at high `t_start`, where the source
activation has already been heavily destroyed. At t≈0.90 even the self-condition
arm — which requests the *natural* coordinate and therefore performs no steering
— costs about **+4.08 nats ΔLM** against a clean baseline of 3.770. The operating
point is unusable regardless of how well it controls.

## D. Hard clamp is an unexpectedly strong baseline

Within natural coordinate support, the hard clamp alone costs roughly **+0.003 to
+0.054 nats** across the tested quantiles while satisfying the requested
coordinate exactly (coordinate error ~1e−6). This is the single most useful
practical finding of the branch, and it is the bar any generative method must
clear.

## E. The final constrained test rejects orthogonal naturalization

Projected conditional flow preserves the requested coordinate exactly, has zero
parallel correction by construction, uses only orthogonal correction, is worse
than hard clamp across the primary grid, degrades in proportion to correction
magnitude, and fails homogeneously across directions (only 22–31% of directions
improve; LOVO ranges entirely positive).

The occasional apparent improvement in unconstrained SDEdit is explained by
coordinate slip. At the one winning cell the ordering is perfectly monotone in
how much slip each arm permits:

| arm | ΔNLL | coordinate error | parallel correction |
|---|---|---|---|
| sdedit | −0.00454 | 1.4426 | 1.4426 |
| tangent | +0.00513 | 0.6711 | 0.6711 |
| projected | +0.00943 | **0.0000** | **0.0000** |

This is attenuation and **must not be counted as naturalization**.

---

# 4. Protocol and analysis notes

## Degeneracy gating ambiguity

A pre-existing row-level vs cell-level degeneracy-gating ambiguity was present.
Both interpretations were evaluated and the substantive conclusions were
invariant to the choice.

Row-level gating is the **prospective** rule for future work. Do not
retroactively claim it was unambiguously preregistered.

## Natural-support classifier threshold

The numerical "25% of clean loss = catastrophic ΔLM" threshold used in
`interp.natural_support.classify` is **not preregistered**. The original
requirement was qualitative ("quality cost is not catastrophic"), and the
threshold was implemented after the qualitative clause was found missing from
the shipped classifier.

The conclusion does not depend on it: the high-`t` reconstruction cost is
descriptively very large (+4.08 nats on a 3.77 baseline, +108%), so any
reasonable threshold gives the same answer. Do not present the number as
preregistered.

## Protected-data exposure

Protected steering identities from the sibling project were exposed while this
branch was being run.

This does not invalidate the concept-independent training or diagnostic results:
the direction-pool construction and every diagnostic in this branch were
mechanical, used the training-only pool with `excluded_splits: [dev, held_out]`,
and recorded `held_out_accessed: false`.

It does mean that a held-out evaluation, if one is ever run, must be carried out
without reference to anything produced under that exposure.

---

# 5. Explicitly not recommended

Do not pursue any of the following as continuation of this branch:

* more NFE for the current prior;
* generic flow priors larger than ~60M;
* more generic Gaussian / noise-only data scaling;
* another conditional-flow run with the same objective;
* training with inconsistent supervision, i.e. condition `<h,v> + delta` against
  target `h` (the target does not satisfy the condition — withdrawn in
  `results/conditional_c1_precheck_v1/FINDINGS.md`);
* rescuing the method through final hard projection alone;
* interpreting coordinate attenuation as naturalization;
* spending LLM-judge budget on this branch.

No additional DEV sweep is needed. `configs/flow_phase_b_conditional_dev_v1.yaml`
remains `draft_not_frozen` and is superseded by this closure.

---

# 6. Artifact index

Local canonical paths, hashes in `results/BRANCH_CLOSE_MANIFEST.sha256`.

| artifact | path |
|---|---|
| selected checkpoint | `results/remote_conditional_v1/conditional_flow_60m_fw32m_v1/best_step_249500.pt` |
| checkpoint ladder + optimizer state | same directory, `step_*.pt`, `last.pt` |
| run status / used_config / provenance | `.../status_complete.json`, `.../meta.json` |
| training log | `results/remote_conditional_v1/conditional_flow_60m_fw32m_v1.log` |
| training config | `configs/flow_train_conditional_60m_v1.yaml`, `configs/flow_core_conditional_60m_v1.yaml` |
| direction pool | `data/direction_pools/training_only_rank256_v1.pt` |
| condition-use diagnostic | `results/condition_use_v1/` |
| C1 precheck | `results/conditional_c1_precheck_v1/` |
| natural-support controllability | `results/natural_support_v1/` |
| constrained naturalization | `results/constrained_naturalization_v1/` |

Checkpoint SHA256:
`83324cfab50eb7d055ac69f864bf7972f56e8a45bcdecaaf7d9eff7841933f76`

---

# 7. Successor

`docs/PROPOSAL_CONSTRAINT_PRESERVING_TANGENT_FLOW.md` — proposal only, nothing
implemented or trained.
