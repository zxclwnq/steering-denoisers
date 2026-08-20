# Research Governance

## Purpose

This document defines the scientific governance rules this project runs under.

The goal is not to prevent exploration. The goal is to preserve a clear distinction between:

* preregistered/frozen tests;
* DEV method development;
* exploratory analysis;
* final held-out evaluation.

An experiment can fail scientifically and still be a successful research result. The
narrative must never be optimized at the expense of experimental validity.

---

# 1. Authority hierarchy

When rules conflict, this order applies:

1. the frozen experiment protocol for the active experiment;
2. this governance document;
3. implementation convenience.

A code comment or stale script must not override a frozen protocol.

---

# 2. Protected data

## Held-out directions

Held-out directions are a final evaluation resource.

Default policy:

**NO ACCESS.**

Forbidden before final authorization:

* generation;
* metric computation;
* debugging;
* visualization;
* prompt inspection tied to held-out results;
* partial sweeps;
* "quick sanity checks";
* checkpoint selection;
* choosing `t_start`, NFE, alpha grid, architecture, or metric thresholds.

If code cannot be tested without loading held-out metadata, redesign the test.

Accidental held-out access is a critical finding.

## Where the identities live

The held-out identities are kept in a separate protected location, outside this public
repository, and are deliberately absent from every project document. Reading,
enumerating or summarizing them counts as protected-data access under the policy above,
not as a lesser category: knowing which directions are held out is by itself enough to
bias a development decision.

Held-out directions were never evaluated in this project. Every artifact records
`held_out_accessed: false`.

---

# 3. Experiment classes

Every nontrivial experiment should be identifiable as one of:

## Frozen confirmatory

Protocol and analysis rule fixed before result inspection.

Changes after observing results invalidate the original confirmatory interpretation.

## DEV method development

Hyperparameters and methods may be explored on DEV, but changes must remain documented.

Do not later describe DEV discoveries as preregistered.

## Exploratory

Used for hypothesis generation and mechanistic investigation.

Exploratory findings may motivate a new frozen test.

## Final held-out

One frozen method/configuration is evaluated after all development decisions are complete.

No iterative tuning on held-out.

---

# 4. Result-preserving changes

After results exist, the following require a new experiment/config version if they can change outcomes:

* metric definition;
* filtering or degeneration rule;
* alpha grid;
* prompt set;
* seeds;
* vector selection;
* checkpoint;
* architecture;
* inference algorithm;
* corruption distribution;
* `t_start`;
* NFE;
* method implementation;
* model precision when numerically relevant.

Pure formatting/reporting fixes may update existing reports if raw values are unchanged.

Bug fixes must preserve and label the invalid historical result rather than silently replacing it.

---

# 5. Model-selection independence

Whenever possible, concept-agnostic activation models should be selected using concept-independent validation data.

For the flow matcher, preferred model-selection evidence includes:

* validation flow loss;
* loss by time bin;
* clean/partially corrupted activation reconstruction;
* functional ΔLM reconstruction diagnostics.

Do not select a flow checkpoint because it steers a DEV feature better unless a new protocol explicitly allows that.

---

# 6. Required controls for steering claims

A claimed steering improvement should survive appropriate controls.

At minimum consider:

## Additive baseline

[
h' = h + \alpha v.
]

## Realised projection

Measure actual displacement along (v), not only nominal alpha.

## Scalar attenuation / shrinkage

Check whether a simple smaller effective alpha reproduces the quality improvement.

## Unrelated SAE controls

Target activation increases should not merely reflect broad activation inflation.

## Degeneration

Points failing the frozen degeneration gate do not define the valid primary Pareto frontier.

---

# 7. Flow-specific scientific contract

Current flow model:

[
x_0=(h-\mu)/\sigma
]

[
x_t=(1-t)x_0+t\epsilon,
\qquad
t\sim U[0,1],
\qquad
\epsilon\sim\mathcal N(0,I)
]

Target velocity:

[
u^*=\epsilon-x_0.
]

The model predicts:

[
f_\theta(x_t,t)\approx u^*.
]

For steering:

[
h_s=h+\alpha v
]

then standardize, partially noise, reverse-integrate, and denormalize.

Do not accidentally revert this method into the old direct reconstruction denoiser.

### Phase A result

Functional reconstruction is successful.

NFE 1/3/5 is approximately saturated.

Therefore Phase B should explicitly test whether this NFE saturation persists under steering rather than assuming multi-step integration is beneficial.

`t_start` is currently the primary meaningful flow-control axis.

---

# 8. Severity levels

## Critical

Examples:

* held-out leakage;
* wrong data split;
* wrong hook location;
* mathematically incorrect intervention;
* metric implementation invalidating headline result;
* nondeterministic pairing presented as paired;
* overwritten or falsified provenance.

Blocks experiment execution.

## High

Likely to materially change experimental outcome or interpretation.

Blocks expensive execution until resolved.

## Medium

Important robustness/reproducibility problem but unlikely to reverse the main result alone.

May block final evaluation.

## Low

Minor maintainability/reporting issue without material scientific effect.

Does not block experimentation unless accumulated.

---

# 9. Review gates

Explicit human approval is required before:

* first final held-out evaluation;
* changing protected data policy;
* changing a frozen primary metric after seeing results;
* extending compute substantially because current results are disappointing;
* adding new LLM-judge spending beyond an already approved protocol;
* deleting irreplaceable experimental artifacts;
* redefining the headline hypothesis after observing its result.

Cheap debugging and smoke tests do not require a human gate if they obey the data policy.

---

# 10. Negative results

Negative results must be preserved.

A negative result must not be answered by immediately:

* changing the metric;
* widening the search;
* deleting failed configurations;
* extending training indefinitely;
* selecting only favourable vectors;
* changing the hypothesis.

First characterize what failed and why.

A mechanism explaining a negative result may be more valuable than a weak positive result.
