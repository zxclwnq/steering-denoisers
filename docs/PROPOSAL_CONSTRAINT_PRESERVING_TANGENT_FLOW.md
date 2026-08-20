# Proposal: Constraint-Preserving Tangent Flow

## Status

**PROPOSAL ONLY — NOTHING IMPLEMENTED, NOTHING TRAINED, NOTHING LAUNCHED**

Requires explicit human approval before any code or compute. Supersedes
`docs/BRANCH_CLOSURE_GENERIC_FLOW_NATURALIZATION.md` as the active research
direction, but shares none of its trained artifacts.

---

# 1. Hypothesis

> The previous prior was trained on the wrong corruption geometry. If the desired
> semantic coordinate is meant to remain fixed, the model should be trained
> directly on corruption and denoising trajectories that live in the tangent
> subspace of that constraint.

This is a **new objective and training distribution**, not another inference
trick applied to the closed branch's checkpoint.

---

# 2. Why this is scientifically different

The completed model was trained on ordinary isotropic flow corruption

    x_t = (1 - t) x0 + t eps

and only at *inference* was asked to perform constrained/tangent correction. The
projected arm therefore ran the model on a state distribution it had never been
trained on: every `x_t` it saw at inference lay exactly on a hyperplane, while
every `x_t` it saw in training did not.

Projected inference on the closed branch was consequently a **train/test geometry
mismatch**. The next branch removes that mismatch by training on exactly the
geometry used at inference.

This is **not** evidence that the new method will work. It is the cleanest
remaining hypothesis, and it is falsifiable cheaply.

---

# 3. Core training objective

For a clean standardized activation `x0` and a **training-only** direction `v`
(unit norm, canonicalized), let the conditioned coordinate be

    c = <x0, v>

Decompose

    x0 = c v + x0_perp,        x0_perp = x0 - c v

Sample Gaussian noise and remove its parallel component

    eps_perp = eps - <eps, v> v

Define the constrained flow path

    x_t = c v + (1 - t) x0_perp + t eps_perp
        = (1 - t) x0 + t (eps_perp + c v)

so that

    <x_t, v> = c        for every t

The velocity target is tangent:

    u_target = eps_perp - x0_perp

and the model predicts a tangent velocity `u_pred ≈ u_target`.

Note this path is the same closed form already derived and unit-tested for
inference in `src/interp/constrained_flow.py`; the proposal is to make it the
*training* distribution as well.

## 3.1 Design choice: enforce tangency analytically or learn it?

The output can optionally be projected

    u_pred <- u_pred - <u_pred, v> v

Arguments for enforcing analytically:

* removes an entire error mode the model would otherwise spend capacity on;
* guarantees the constraint holds exactly at every step regardless of model
  quality, so a negative result cannot be blamed on leakage along `v`;
* costs nothing at inference.

Arguments for learning it:

* if the model cannot learn tangency on its own, that is diagnostic information
  about whether the conditioning representation is adequate;
* analytic projection could mask a model that has not actually learned the
  constrained geometry, producing a flattering training loss.

**Recommendation:** train *without* analytic output projection, and measure the
residual parallel component as a diagnostic. Apply analytic projection only at
inference, where correctness matters more than diagnosis. This keeps the
training signal honest while making the inference-time constraint exact. The
decision should be revisited if training tangency proves unstable.

---

# 4. Cost constraints

The branch must preserve the original cheap-prior motivation.

* do not exceed ~60M parameters;
* start with the cheaper architecture (the existing ~16M) unless there is a
  concrete reason not to;
* inference target NFE <= 3, preferably NFE = 1;
* training-only directions exclusively;
* no DEV or held-out leakage at any stage;
* no LLM judge during method development.

Do not propose scaling to GLP-sized models. The closed branch already showed
that scaling the generic prior improved reconstruction without improving
steering.

---

# 5. Staged experiment

Do not train the 60M model first.

## Stage T0 — mathematical / synthetic validation

No real data, no GPU beyond trivial. Implement and test:

* hyperplane preservation of the constrained path at every `t`;
* tangent noise construction;
* tangent velocity target correctness;
* tangent model output (residual parallel component measured);
* exact NFE accounting;
* synthetic overfit: a tiny model must drive the constrained loss to ~0 on a
  small fixed batch, proving the objective is learnable at all.

Much of the geometry is already implemented and unit-tested in
`src/interp/constrained_flow.py` and `tests/test_constrained_flow.py`; T0 should
reuse it rather than reimplement.

**Gate:** all invariants pass and the synthetic overfit succeeds.

## Stage T1 — cheap real-data tangent reconstruction

Train the small (~16M) architecture on the tangent objective. Concept-independent
only: FineWeb activations, training-only direction pool.

Evaluate whether the tangent-trained flow improves reconstruction of
**tangent-corrupted** validation activations relative to the current isotropic
prior evaluated on the same matched task.

**Gate:** if the tangent-trained model cannot beat the isotropic prior on its own
matched training task, **stop the branch**. A model that cannot solve the task it
was trained for will not solve the harder downstream one.

## Stage T2 — natural-support hard-clamp naturalization

Reuse the frozen `natural_support_v1` plan: training-only validation directions,
natural-support target coordinates, same sequences and seeds.

Primary comparison, coordinate held exactly fixed in both arms:

    hard clamp
        vs
    hard clamp + tangent-trained flow

Primary mechanistic quantity:

    paired ΔNLL = NLL_flow - NLL_clamp

matched per validation sequence, bootstrapped over sequences.

This is directly comparable to `results/constrained_naturalization_v1/`, which
provides the hard-clamp baseline (+0.003 to +0.054 nats) and the isotropic-prior
failure numbers the new method must beat.

**Gate:** only a convincing negative paired ΔNLL, with CI excluding zero and
homogeneous direction signs, justifies a 60M confirmation run or any DEV
consideration.

---

# 6. Hard stop condition

The branch must stop, and generative-naturalization work must end entirely, if
the tangent-trained flow:

* **clearly improves** matched tangent reconstruction (T1 passes),

but

* **still does not improve** hard-clamp NLL at fixed coordinate (T2 fails).

Interpretation in that case:

> Even learning the correct constrained corruption geometry does not produce
> useful steering repair.

Do not respond by increasing model size. Do not respond by adding NFE. That
combination of results would be a clean, informative, publishable negative for
the whole generative-naturalization family, and it should be reported as such.

---

# 7. What this proposal does not authorize

* no training;
* no DEV or held-out evaluation;
* no LLM-judge spend;
* no implementation before human approval of this document.

---

# 8. Open questions for the human before approval

1. Is the ~16M architecture acceptable for T1, or is there a reason to start at
   60M despite the cost constraint?
2. Should tangency be enforced analytically during training (§3.1 recommends
   not) — this changes what a T1 pass means.
3. Should the conditioned direction `v` be sampled per-example from the
   training-only pool each step, or fixed per batch? This affects how much
   direction diversity the model sees per unit compute and is a real design
   choice, not an implementation detail.
4. Given the closed branch's result that hard clamp costs only +0.003 to +0.054
   nats, is the remaining headroom large enough to be worth the compute at all?
   The honest framing is that the method is competing against an already very
   cheap baseline.
