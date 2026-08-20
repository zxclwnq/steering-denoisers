# Phase B-C Protocol: DEV Steering with a Conditional Flow Prior

## Status

SUPERSEDED — BRANCH CLOSED 2026-08-15

See `docs/BRANCH_CLOSURE_GENERIC_FLOW_NATURALIZATION.md`. This protocol was never
frozen and never executed. It is retained for the record only; the branch it
belonged to reached a decisive negative result by other, cheaper means.

Original status line follows.

NEW PROTOCOL — DRAFT, EXECUTION BLOCKED BY ITS OWN C1 STOP RULE

The concept-independent C1 pre-check (`results/conditional_c1_precheck_v1/`) was
run before requesting freeze approval and **C1 fails** for `seed_mode="clean"`:
realised displacement saturates near 8 units regardless of the requested alpha.
Per section 13, C2 through C4 are therefore uninterpretable and the DEV sweep
must not run on checkpoint `conditional_flow_60m_fw32m_v1`. See that directory's
FINDINGS.md for the mechanism and the suggested next training experiment.

The protocol below remains the correct design for a checkpoint that passes C1.

This is **not** a modification of `docs/PHASE_B_FLOW_STEERING_PROTOCOL.md`. That
protocol remains frozen and untouched, and its results remain valid for the
unconditional SDEdit method. This document defines a scientifically distinct
experiment for the conditional flow prior trained as
`conditional_flow_60m_fw32m_v1`.

Freezing requires explicit human approval. Until then no DEV generation may run.

---

# 1. Why a new protocol is required

The frozen Phase B protocol defines the intervention as

    h_s = h + alpha * v   ->  standardize -> partial noise -> reverse Euler

with a velocity field `f(x_t, t)`. The conditional model's velocity field is
`f(x_t, t, v_x, c_x)` and cannot be evaluated without a condition. There is no
substitution of the conditional checkpoint into the frozen pipeline that leaves
the frozen protocol's meaning intact, so a new protocol version is the only
result-preserving option (RESEARCH_GOVERNANCE.md section 4).

---

# 2. Main question

Does requesting a coordinate through the learned conditional prior produce a
better concept-quality trade-off than adding the same displacement additively?

The intervention under test replaces the additive steer entirely. The
requested coordinate *is* the steer.

The alternative explanations this protocol must distinguish are the same two
that governed frozen Phase B:

1. the conditional model merely attenuates the requested displacement;
2. the conditional model merely reproduces scalar shrinkage.

---

# 3. Frozen checkpoint

    experiment_id       conditional_flow_60m_fw32m_v1
    checkpoint          best_step_249500.pt
    checkpoint_sha256   83324cfab50eb7d055ac69f864bf7972f56e8a45bcdecaaf7d9eff7841933f76
    selection_metric    val_flow_mse (0.8269649088382721)
    config_fingerprint  e5279a77f79f84a289a759d999889f5184e2fbcc2f3e22c1ad3a02d4ba36d950

Selected by validation flow MSE only. No steering result influenced selection.
Training used the training-only direction pool (23354 directions, rank >= 256,
`excluded_splits: [dev, held_out]`), so no DEV direction entered training.

Do not continue training before this experiment unless explicitly authorized.

---

# 4. Intervention

For clean residual activation `h` at a guarded position, unit direction `v`, and
steering magnitude `alpha`:

    c_nat      = <h, v>                          # natural coordinate, raw space
    c_target   = c_nat + alpha                   # requested coordinate
    v_x, c_x   = standardized_hyperplane(normalizer, v, c_target)
    x          = normalize(h)                    # CLEAN h, never h + alpha*v
    x_t        = (1 - t_s) * x + t_s * epsilon
    reverse Euler from t_s to 0 with f(x, t, v_x, c_x)
    h_tilde    = denormalize(x_K)

`alpha` keeps the frozen relative convention `alpha = alpha_hat * 88.76`, so
nominal steering strength is directly comparable to the additive baseline.

## 4.1 Consequences that must be tested, not assumed

**`t_start = 0` is a null intervention at every alpha.** The condition enters
only through the model call; with no model call there is no steering, so
`h_tilde = h` regardless of alpha. This differs from additive steering, where
`t_start = 0` still leaves `h + alpha*v`. Encode as a test.

**`alpha = 0` is not the identity.** At `alpha = 0` the model is asked to
reproduce the natural coordinate, but it still partially noises and
reverse-integrates, so `h_tilde != h` in general. The `alpha = 0` arm therefore
measures the reconstruction cost of the prior itself and is a required control,
not a no-op. Encode as a test.

---

# 5. Inference grid

    t_start in {0.50, 0.75, 0.90}
    NFE     in {1, 3, 5}

Nine arms, matching the frozen protocol's cardinality.

## 5.1 Why this grid differs from frozen Phase B

The frozen grid is `{0.10, 0.25, 0.50}`. The concept-independent condition-use
diagnostic (`results/condition_use_v1/`, frozen spec `condition_use_v1`,
computed on frozen validation activations and training-only pool directions)
measured relative swap sensitivity to the requested coordinate:

    t = 0.50   0.026
    t = 0.75   0.077
    t = 0.90   0.219
    t = 1.00   0.157

Below t = 0.5 the conditional model effectively ignores the condition, because
`x_t` still carries the true coordinate and the condition is redundant. A grid
of `{0.10, 0.25, 0.50}` would therefore measure a null for mechanical reasons.

`t_start = 0.50` is retained as the overlap point with the frozen grid so the
two methods share one comparable operating point.

This grid choice is derived from concept-independent evidence recorded **before
any DEV steering result was observed**. It is preregistered, not post-hoc. It
must not be widened after seeing DEV results.

---

# 6. Matched noise

Reuse the frozen matched-noise mechanics unchanged
(`interp.flow_steering.matched_flow_noise`, namespace `flow_noise_v2`).

Cell identity: vector, exact alpha, prompt_id, generation_seed, token_position.

Epsilon must NOT depend on: method label, t_start, NFE, dataframe row, batch
order, hook-call grouping.

The same epsilon is reused across all t_start and NFE for a matched cell, so
only the interpolation coefficient changes.

---

# 7. DEV vectors

Frozen DEV set only, reused verbatim from `configs/flow_phase_b_dev_v1.yaml`:
allegations, dungeon, locations_addresses, illicit_drugs,
law_enforcement_officials, same_sex_marriage, borders, sports_awards.

Held-out directions: forbidden. No loading, no enumeration, no metrics.

---

# 8. Steering grid

    alpha_hat in {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0}

Do not enlarge because conditional results are weak.

---

# 9. Required baselines

1. **Additive** — `h + alpha*v`. Reuse existing frozen Phase B rows; do not regenerate.
2. **Scalar shrinkage** — `h + 0.8*alpha*v`. Reuse existing frozen rows.
3. **Unconditional flow** — the nine frozen arms from `results/phase_b_wide60m_v1/`,
   the same-data same-budget prior named as this experiment's baseline. Reuse; do not rerun.
4. **Conditional flow** — the nine arms defined here.

Baselines 1-3 already exist. Only arm 4 consumes new compute.

---

# 10. Metrics

Unchanged from frozen Phase B, so numbers stay comparable:

* primary quality: clean GPT-2 conditional continuation NLL;
* frozen repetition/degeneration gate; distinct-1/2/3;
* concept: frozen lexicon score, target SAE activation, unrelated SAE controls;
* no LLM fluency judge.

---

# 11. Mandatory realised-steering analysis

    Delta_cond = <h_tilde - h, v>
    r_retain   = Delta_cond / alpha        (alpha != 0)

Report by vector, alpha, t_start, NFE.

For the conditional method `r_retain` is the central mechanistic quantity: it
measures whether a *requested* coordinate is actually realised. A conditional
method that requests alpha and realises 0.3*alpha is an attenuator with extra
steps.

Correction geometry uses the additive steer as the shared reference point so
the numbers stay comparable to frozen Phase B:

    c            = h_tilde - (h + alpha*v)
    c_parallel   = (c . v) v
    c_orthogonal = c - c_parallel

Report |c|, |c_parallel|, |c_orthogonal|, cos(c, v).

---

# 12. Critical controls

**Attenuation control.** Compare conditional flow against additive steering at
matched realised projection, not matched nominal alpha. If conditional flow
loses its quality advantage once realised projection is matched, the correct
interpretation is attenuation.

**Shrinkage control.** At matched realised projection, conditional flow must beat
`h + 0.8*alpha*v` to claim useful nonlinear correction.

**Unconditional-flow control.** At matched realised projection and matched
t_start = 0.50, conditional flow must beat the unconditional SDEdit arm to claim
that conditioning contributed anything. This control is what makes the
experiment about *conditioning* rather than about flow priors in general.

---

# 13. Hypotheses

**C1** — the requested coordinate is realised: `r_retain` is materially above
zero and increases with t_start.

**C2** — at matched realised projection, conditional flow improves NLL or
degeneration relative to additive steering.

**C3** — at matched realised projection, conditional flow beats scalar shrinkage.

**C4** — at matched realised projection and t_start = 0.50, conditional flow
beats the unconditional flow arm. This isolates the contribution of conditioning.

**C5** — NFE saturation: at fixed t_start, NFE 1 ≈ 3 ≈ 5.

**C6** — t_start controls intervention strength, with higher t_start giving more
generative rewriting and, per section 5.1, more condition adherence.

C1 failing makes C2-C4 uninterpretable and should stop the experiment.

---

# 14. Statistical comparison

Matched at the experimental-cell level. Respect vector structure; eight DEV
directions is the scientific sample size, not the row count. Report paired
effects, bootstrap CIs, vector-wise sign consistency.

---

# 15. Success and failure

**Strong success** — C1 and C2 and C4 hold, surviving the attenuation control.

**Moderate success** — a small vector-consistent frontier improvement on cheap
metrics that survives auditing.

**Mechanistic success without behavioural improvement** — conditioning produces
clearly different activation geometry from shrinkage and from unconditional
flow, explaining the mechanism even with a flat Pareto frontier.

**Negative result** — the conditional model attenuates, or conditioning adds
nothing over the unconditional prior at matched strength. This is an acceptable
and publishable outcome. Preserve it; do not widen the grid in response.

---

# 16. Held-out policy

DEV only. No held-out evaluation, for any reason, at any point in this protocol.
