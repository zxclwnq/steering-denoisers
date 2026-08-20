# Post-Stop Protocol — Three Bounded Checks (2026-08-19)

## Status

**FROZEN 2026-08-19, before any result of experiments A, B or C existed.**

Authorized by explicit human instruction on 2026-08-19. This is a *bounded
post-stop pass*: three named checks, then the research branch closes for good.

---

# 0. Relationship to the closed branches

This document does **not** reopen, revise, or reinterpret:

* `docs/BRANCH_CLOSURE_GENERIC_FLOW_NATURALIZATION.md`;
* `docs/BRANCH_CLOSURE_CONSTRAINT_PRESERVING_TANGENT_FLOW.md`;
* `docs/TANGENT_FLOW_PROTOCOL.md`.

The T1 PASS / T2 FAIL verdict of 2026-08-16 stands exactly as recorded. Nothing
measured under this protocol may be back-dated into it, and no result here may
be described as having been preregistered by it.

Every arm defined below is labelled **post-stop / additional**. Results are
written to new directories with new spec versions. No historical result
directory, config, column name, or method label is overwritten or renamed.

The stop rule in `docs/TANGENT_FLOW_PROTOCOL.md` §6 forbade rescuing the closed
branch with more parameters, more NFE, more generic data, another inference
projection trick, or LLM judges. This pass does none of those. It asks three
questions the stop rule did not answer, at fixed capacity, fixed data, fixed
NFE, and with no judge.

---

# 1. Shared invariants (all three experiments)

Held constant, identical to the closed tangent branch:

| item | value |
|---|---|
| language model | GPT-2 small, revision-pinned |
| intervention site | `blocks.7.hook_resid_pre` |
| direction pool | `data/direction_pools/training_only_rank256_v1.pt`, training-only, rank floor 256 |
| DEV protocol | `natural_support_v1` sequence/direction/target plan |
| prompts / seeds | `NaturalSupportSpec` seeds, unchanged |
| primary quality metric | clean GPT-2 conditional continuation NLL |
| concept metric | realised coordinate `<h, v>`; frozen lexicon / SAE metrics where already defined |
| uncertainty | clustered bootstrap `direction_cluster_then_sequence`, 2000 resamples, equal quantile weight |
| controls | additive, scalar shrinkage, hard clamp |

Forbidden throughout:

* any access to `configs/protected/` or held-out directions;
* any DEV steering vector in training or checkpoint selection;
* any LLM judge;
* any metric change after a result is seen;
* any new 60M model before a new 16M/cheap arm shows a positive result;
* any post-hoc search of the diagnostic grid for a favourable operating point.

Checkpoint selection for every trained model in this pass is
**concept-independent**: validation reconstruction loss on the artifact's own
held-back 5% document split, minimized, nothing else.

---

# 2. Experiment A — variance-preserving tangent path

## A.1 Hypothesis

The tangent path used by the closed branch,

    x_t = c v + (1 - t) x_perp + t eps_perp,

preserves `<x_t, v> = c` but **shrinks the orthogonal scale**. If `x_perp` and
`eps_perp` have equal variance and are independent,

    Var[(1 - t) x_perp + t eps_perp] = (1 - t)^2 + t^2,

which is 1/2 at `t = 0.5`. Mid-trajectory states are therefore off the natural
activation shell in norm, not only in direction. A model trained on them may be
learning to denoise an artificially shrunken distribution.

## A.2 The path

With `theta = (pi/2) t`:

    x_t = c v + cos(theta) x_perp + sin(theta) eps_perp
    u*  = (pi/2) ( -sin(theta) x_perp + cos(theta) eps_perp )

Both invariants hold analytically:

* `<x_t, v> = c` for every `t`, because both moving terms are v-orthogonal;
* `<u*, v> = 0`;
* `cos^2 + sin^2 = 1`, so the orthogonal scale is preserved under the standard
  assumption `Var[x_perp] = Var[eps_perp] = 1` per orthogonal coordinate.

Endpoints: `t = 0` gives `x_0`; `t = 1` gives `c v + eps_perp`.

Predicted velocity is analytically projected exactly as before:

    u_used = u_raw - <u_raw, v> v

Inference always projects. There is no switch.

## A.3 Matched severity — the fair comparison rule

Comparing the two paths at equal `t` compares different signal/noise ratios.
Define the orthogonal noise-to-signal ratio:

    linear:              r_lin(t) = t / (1 - t)
    variance-preserving: r_vp(t)  = tan(theta) = tan((pi/2) t)

Matched severity means equal `r`. Therefore

    t_vp = (2/pi) arctan( t_lin / (1 - t_lin) )

**Every primary claim in Experiment A is made at matched `r`, not at matched
`t`.** The matched grid is computed by `interp.tangent_flow.matched_vp_time`
and frozen here:

| `t_lin` | `t_vp` | role |
|---|---|---|
| 0.10 | 0.070447 | T2 primary operating point |
| 0.25 | 0.204833 | T1/T2 diagnostic |
| 0.50 | 0.500000 | T1 primary operating point (fixed point of the map) |
| 0.75 | 0.795167 | T1 diagnostic |

## A.4 Training

Exactly the frozen `flow_train_tangent_narrow16m_fw32m_v1` recipe with **one**
field changed — `flow_objective.type` — plus the identical `noise_seed`
`20260816`.

The seed is deliberately *not* changed. The variance-preserving sampler consumes
its generator in the same order as the linear one (pool direction, then `t`,
then `eps`), so the same seed yields the **same directions, the same `t`, and
the same `eps`** as the frozen linear tangent run. This is a genuinely paired
corruption stream: the only difference between the two training runs is the path
formula. That makes A an objective-only control, which the closed branch
explicitly could not claim.

Controlled and identical: architecture (`flow_core_conditional_narrow16m_v1`),
capacity, corpus (`resid7_fw_train_32000k_v1`), split, direction pool, batch
size 1024, AdamW, lr 3e-4, cosine schedule, 250000 steps, warmup 500, grad clip
1.0, float32, eval every 500, checkpoint schedule.

Selection metric: `val_flow_mse` under the variance-preserving objective,
minimized. Concept-independent. DEV is never consulted.

## A.5 Evaluation

**A-T1 (reconstruction analogue).** The frozen T1 machinery, run at the matched
`t_vp` grid: corrupted ΔNLL, reconstructed ΔNLL, recovered fraction, validation
MSE. Reported beside the frozen linear-path T1 numbers.

Note recorded in advance: `val_flow_mse` is **not** comparable across the two
objectives — the two targets have different scales (`u*_vp` carries a `pi/2`
factor). The comparable quantities are the functional ones: recovered fraction
and paired ΔNLL, both measured through GPT-2.

**A-T2 (the decision).** Post-clamp correction on the `natural_support_v1` plan
at p50/p75/p90/p95/p99, comparing three arms at the matched primary point
(`t_vp = 0.070447`, NFE 1):

1. hard clamp;
2. old linear-tangent 16M model (frozen result, re-used, not re-run);
3. new variance-preserving 16M model.

Primary statistic: pooled paired `dNLL_VP-clamp`, equal quantile weight,
clustered bootstrap, identical to the frozen T2 statistic.

## A.6 Verdict rule (frozen before results)

* `dNLL_VP-clamp >= 0`, or CI containing zero → **hypothesis closed**. The
  orthogonal-scale defect was not what was wrong. Report and stop.
* `dNLL_VP-clamp < 0` with CI excluding zero → **candidate positive**. Before it
  may be called a result, all four must hold: coordinate preservation to ≤1e-3;
  per-direction consistency (>80% of directions negative); no attenuation
  (parallel correction ≈ 0 by construction, verified); LOVO interval never
  crossing zero. Only then is it reported as positive.

A reconstruction improvement with clamp still winning is **not** a positive
result; it is the hypothesis being closed.

---

# 3. Experiment B — denoiser trained on steering-like corruption

## B.1 Hypothesis

Every prior in this programme was trained on Gaussian corruption. Cold Diffusion
and SPAR suggest training the restoration model on the *actual* degradation:

    z = h + delta v,   D(z) -> h

## B.2 The expected failure mode, named in advance

**The model will simply learn to undo the steering.** A comparison at equal
nominal `alpha` is therefore *not* decisive and is reported only as context. The
decisive comparison is at **equal realised concept strength** (§B.5).

## B.3 Corruption distribution — frozen before any DEV result

    v      ~ uniform over the training-only pool (rank floor 256)
    delta  ~ Uniform(-32, +32)   in raw activation units
    z      = h + delta v
    target = h

Justification, computed from the already-published frozen T2 rows only
(`results/tangent_t2_v1/raw_rows.npz`, hard-clamp arm), i.e. from displacements
that are already public within this project and involve no DEV steering vector:
the natural coordinate has sd 7.52; clamp displacements over p50…p99 have sd
6.18, `|delta|` 99th percentile 21.8 and observed maximum 32.0. `Uniform(-32,
32)` covers the entire working range with symmetric support and includes
`delta = 0` (the identity case). No other distribution is tried.

## B.4 Model and training

Cheap residual MLP, not a flow:

    D(z) = z + f_theta(z)        in standardized coordinates

`f_theta` is the frozen narrow ~16M `flow_core_v1` trunk, **unconditional**: it
sees only `z`. It is not told `v`, not told `delta`, and not told `t`. Training
loss is MSE against the clean `x_0`. Same corpus, split, batch size, optimizer,
schedule and budget as Experiment A.

Selection metric: validation denoising MSE, minimized. Concept-independent. No
DEV direction enters training or selection.

## B.5 Inference and the decisive control

Full correction `D(z)`, and partial correction over a grid frozen here:

    h'_lambda = z + lambda ( D(z) - z ),   lambda in {0.25, 0.50, 0.75, 1.00}

No single lambda is selected post hoc; all four are reported.

For every evaluated point, record the realised steering strength

    alpha_eff = <h' - h_clean, v>

and compare **not** against additive steering at nominal alpha but against
additive and scalar-shrinkage controls at the **same realised `alpha_eff`**. The
matched-strength curve carries four arms:

1. additive;
2. scalar shrinkage;
3. old Gaussian denoiser (frozen historical result);
4. steering-trained denoiser (this experiment).

## B.6 Verdict rule (frozen before results)

* Apparent gain disappears once matched on `alpha_eff` → **attenuation**. The
  method removes steering rather than repairing it. Negative; report and stop.
* Gain survives matching → **candidate positive**. Then and only then: unseen
  DEV directions, SAE metric, degeneration gate, clustered bootstrap, and
  per-direction consistency, all before it is called a result.

---

# 4. Experiment C — curvature diagnostic

## C.1 Hypothesis

The programme assumes concept strength `≈ <h, v>` for one fixed `v`. If natural
activations lie along a *curved* trajectory, `v` is only a local tangent, strong
additive steering leaves the natural manifold, and orthogonal correction defined
relative to a single fixed `v` need not match the real geometry.

This is a **diagnostic**. It trains nothing and proposes no new intervention.

## C.2 Procedure

For each training-only direction `v`:

1. take natural activations `h` from the frozen validation artifact;
2. compute `c = <h, v>`;
3. bin by quantiles of `c` at p10/p25/p50/p75/p90;
4. per bin, `mu_k = E[h | c in bin_k]`;
5. secants `d_k = mu_{k+1} - mu_k`, split into `d_par` and `d_perp`;
6. report

       r_k        = ||d_perp|| / ||d||
       cos(d_k, v)
       cos(d_k, d_{k+1})

A near-linear concept trajectory keeps `d_k` close to `v` and to each other.
Systematic rotation with increasing `c` is evidence of curvature.

## C.3 Controls (mandatory)

* shuffled coordinate/bin labels;
* random unit directions;
* bootstrap over sequences;
* many training-only directions (32, the frozen `natural_support_v1` count), not
  one favourable example.

Optional if cheap: linear probe vs small quadratic probe on held-out contexts.
A stable nonlinear gain is *supporting* evidence only, never the main curvature
test.

## C.4 Reporting rule (frozen before results)

* Strong curvature → add the limitation to Discussion, with one figure (concept
  quantile × local direction angle × orthogonal drift). Do **not** attempt
  nonlinear steering in this pass.
* No curvature → state plainly that no strong evidence of curvature was found in
  the natural range, and do **not** use nonlinear geometry as an explanation of
  the negative results.

---

# 5. Priority and stopping

Order: **A, then B, then C.**

If A or B produces a positive result, its controls (§A.6, §B.6) are exhausted
before C is broadened. After these three checks the research branch closes.

---

# 6. Deliverables

Per experiment, as specified by the authorizing instruction: exact path/config/
checkpoint (A); architecture and corruption distribution (B); dataset size and
direction count (C); the frozen comparisons; and an explicit verdict.

Afterwards: update figures and report from the obtained results only, `make
report`, `make verify`, `make test`, `make lint`, and confirm held-out was not
used.
