# Phase B Protocol: DEV Steering with Cheap Flow Matching

## Status

FROZEN DEV PROTOCOL — EXECUTED

The protocol was frozen before generation. The full Phase B sweep was gated on the
from-scratch flow implementation independently passing the Phase-A reconstruction
criterion; it did, and the sweep then ran under the protocol below unchanged.

---

# 1. Main question

Does SDEdit-style correction using the learned cheap flow prior move the valid concept–quality steering Pareto frontier outward?

More specifically:

> Can flow correction preserve more semantic steering effect at a given language-model quality than additive steering or scalar attenuation?

The important alternative explanation is:

> Flow merely reduces effective alpha.

Phase B must explicitly distinguish these.

---

# 2. Frozen flow checkpoint

Use the Phase A flow model selected without steering-vector performance.

Do not choose or extend a checkpoint based on Phase B steering results.

Checkpoint path and SHA must be recorded in the resulting Phase B provenance.

Do not continue training before Phase B unless explicitly authorized.

---

# 3. Flow steering transformation

Start with clean residual activation

[
h.
]

Standard additive steering:

[
h_s=h+\alpha v.
]

Standardize the steered activation:

[
x_s
===

\frac{h_s-\mu}{\sigma}.
]

Partial flow corruption:

[
x_{t_s}
=======

(1-t_s)x_s+t_s\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
]

This is a convex combination.

Reverse integrate from

[
t=t_s
]

to

[
t=0.
]

For NFE (K):

[
t_i
===

t_s\left(1-\frac{i}{K}\right),
\qquad
i=0,\ldots,K.
]

Explicit Euler:

[
x_{i+1}
=======

x_i
+
(t_{i+1}-t_i)
f_\theta(x_i,t_i).
]

Denormalize:

[
\tilde h
========

\sigma\odot x_K+\mu.
]

Inject (\tilde h) back into GPT-2.

---

# 4. Frozen inference grid

Evaluate:

[
t_{\rm start}
\in
{0.10,0.25,0.50}
]

crossed with:

[
NFE
\in
{1,3,5}.
]

Total:

9 flow inference arms.

Do not remove NFE=3/5 even though Phase A suggests saturation.

Their near-equivalence is now a testable prediction.

Do not add:

* NFE 10;
* NFE 20;
* additional `t_start`;
* adaptive `t_start`.

Any such experiment would be a separate later branch.

---

# 5. Phase A-derived NFE prediction

Phase A strongly predicts:

[
NFE=1
\approx
NFE=3
\approx
NFE=5
]

at fixed `t_start`.

Phase B must test this explicitly under steering.

If confirmed, the correct conclusion is:

> The cheap model's useful steering correction is effectively captured by one large flow evaluation.

If steering uniquely benefits from multiple steps despite reconstruction saturation, that is a scientifically interesting dissociation and should be highlighted.

---

# 6. Matched flow noise

The initial Gaussian sample is a new source of randomness.

It must be paired.

For one underlying experimental cell, the Gaussian draw must be deterministic.

Stable cell identity should include the relevant immutable quantities, for example:

* vector ID;
* alpha;
* prompt ID;
* generation seed;
* generated token position when correction occurs.

The Gaussian seed must NOT depend on:

* NFE;
* method label;
* dataframe row;
* batch ordering.

Prefer using the same underlying

[
\epsilon
]

across all `t_start` and NFE settings for a matched cell.

Different `t_start` values should alter only the interpolation coefficient.

This permits low-variance comparisons.

Add explicit tests.

---

# 7. DEV vectors

Use only the frozen DEV set:

* allegations;
* dungeon;
* locations_addresses;
* illicit_drugs;
* law_enforcement_officials;
* same_sex_marriage;
* borders;
* sports_awards.

Do not load or evaluate held-out features.

---

# 8. Steering grid

Reuse the already frozen relative-alpha convention:

[
\hat\alpha=\alpha/88.76.
]

Primary grid:

[
\hat\alpha
\in
{
0,
0.1,
0.2,
0.3,
0.4,
0.5,
0.6,
0.7,
0.85,
1.0
}.
]

Stress alpha points should only be included if already part of the frozen Phase B generation protocol.

Do not silently enlarge the alpha search because flow results are weak.

---

# 9. Required baselines

Phase B comparison should include at minimum:

## Additive

[
h'=h+\alpha v.
]

## Frozen old naive denoiser

Use the already frozen one-step Gaussian-denoiser configuration.

Do not retune.

## Scalar shrinkage

Important control:

[
h'
==

h+\kappa\alpha v.
]

The previously informative control is approximately:

[
\kappa=0.8.
]

Do not perform a new large shrinkage sweep.

## Flow

All nine:

[
3\ t_{\rm start}
\times
3\ NFE.
]

---

# 10. Quality metrics

Primary:

## Clean GPT-2 conditional continuation NLL

[
L
=

-\frac1{|y|}
\sum_t
\log
p_{\rm GPT2}(y_t\mid x,y_{<t}).
]

Lower is better.

Also compute:

* frozen repetition/degeneration gate;
* distinct-1;
* distinct-2;
* distinct-3.

Do not use LLM fluency.

---

# 11. Concept metrics

Compute:

* frozen lexicon score;
* target SAE activation;
* unrelated SAE controls.

Do not initially spend LLM-judge budget on all flow arms.

Cheap metrics are screening tools.

A matched semantic concept audit is only justified if a small number of candidates survive.

---

# 12. Mandatory realised-steering analysis

For a unit steering direction (v), additive displacement is:

[
\Delta_{\rm add}
================

# \langle h_s-h,v\rangle

\alpha.
]

After flow:

[
\Delta_{\rm flow}
=================

\langle\tilde h-h,v\rangle.
]

Define retained steering fraction:

[
r_{\rm retain}
==============

\frac{\Delta_{\rm flow}}{\alpha}
]

for (\alpha\ne0).

Report it by:

* vector;
* alpha;
* `t_start`;
* NFE.

This is a primary mechanistic diagnostic.

---

# 13. Flow correction geometry

Define:

[
c
=

\tilde h-h_s.
]

Parallel correction:

[
c_\parallel
===========

(c\cdot v)v.
]

Orthogonal correction:

[
c_\perp
=======

c-c_\parallel.
]

Report:

* (|c|);
* (|c_\parallel|);
* (|c_\perp|);
* (\cos(c,v));
* retained steering fraction.

Compare these quantities to the old one-step denoiser.

---

# 14. Critical scalar-attenuation control

A flow quality improvement is not sufficient.

Suppose flow transforms:

[
\alpha
\rightarrow
\alpha_{\rm eff}.
]

Then compare against ordinary additive steering at approximately the same:

* realised projection;
* SAE concept strength;
* lexicon concept strength where possible.

The central test is:

> Does flow still reduce NLL / degeneration after effective steering strength is matched?

If no, the correct interpretation is attenuation, not generative correction.

---

# 15. Primary hypotheses

## H1: functional steering correction

At least one flow configuration improves quality relative to equal-nominal-alpha additive steering.

This is weak evidence only.

## H2: genuine Pareto improvement

At least one flow configuration improves quality at matched concept or matched realised steering strength.

This is the central hypothesis.

## H3: NFE saturation generalizes

At fixed `t_start`:

[
NFE=1\approx3\approx5.
]

## H4: `t_start` controls intervention strength

Increasing `t_start` is expected to increase the amount of generative rewriting.

A plausible outcome is:

[
t_{\rm start}\uparrow
\Rightarrow
r_{\rm retain}\downarrow
]

and quality improves.

This is a hypothesis, not an assumption.

## H5: flow differs from scalar shrinkage

At matched realised steering strength, flow should outperform simple shrinkage if the learned prior provides genuinely useful nonlinear correction.

---

# 16. NFE analysis

Do not average NFE arms before testing them.

For each fixed `t_start`, perform matched comparisons:

* 1 vs 3;
* 1 vs 5;
* 3 vs 5.

Primary quantities:

* NLL;
* target SAE;
* lexicon;
* repetition/degeneration;
* retained steering fraction.

If all differences are negligible, later reporting may collapse NFE for presentation, but the original result must remain visible.

---

# 17. Pareto analysis

Construct valid nondegenerate frontiers for:

* additive;
* naive denoiser;
* shrinkage;
* each flow configuration.

At minimum:

[
\text{lexicon}
\quad\text{vs}\quad
\text{NLL}
]

and:

[
\text{target SAE}
\quad\text{vs}\quad
\text{NLL}.
]

Also examine:

[
\text{realised projection}
\quad\text{vs}\quad
\text{NLL}.
]

Degenerate samples do not define the primary valid frontier.

---

# 18. Statistical comparison

Prefer matched comparisons at the experimental-cell level.

Respect vector structure.

Do not treat thousands of token/generation rows as fully independent when the scientific unit includes a small number of steering directions.

Report:

* paired effects where applicable;
* bootstrap confidence intervals;
* vector-wise sign consistency;
* leave-one-vector-out sensitivity for headline comparisons when useful.

Avoid pseudo-replication.

---

# 19. LLM semantic audit gate

Do not automatically judge all flow outputs.

A candidate may proceed to matched semantic auditing only if:

1. it is valid/nondegenerate;
2. it shows a meaningful cheap-metric Pareto signal or a scientifically interesting dissociation;
3. the candidate selection rule is recorded before semantic judge results are observed.

Use the existing concept-inclusion rubric.

Use both established concept judges.

Do not use the rejected fluency judge.

---

# 20. What counts as success

Strong success:

A flow configuration improves NLL / degeneration at matched semantic concept strength and cannot be explained by reduced effective alpha.

Moderate success:

A small, vector-consistent frontier improvement appears on cheap metrics and survives semantic auditing.

Mechanistic success without behavioural improvement:

Flow produces clearly different activation geometry from scalar shrinkage, yielding an informative explanation even if the Pareto frontier does not move.

Negative result:

Flow behaves primarily as attenuation, or provides no advantage over additive/shrinkage.

A clean negative result is acceptable.

---

# 21. What happens after Phase B

Do not decide before results.

Possible next branches:

## If flow shows a real but incomplete advantage

Consider:

* longer training;
* modest capacity increase;
* improved small-(t) accuracy.

## If `t_start` matters but fixed values are inefficient

Consider adaptive:

[
t_{\rm start}=f(\hat\alpha)
]

with:

[
t_{\rm start}(0)=0.
]

## If flow behaves like attenuation

Consider structured corruption mismatch:

Train on perturbations aligned with training-only SAE directions rather than pure Gaussian directions.

This must be a new experiment.

## If flow clearly fails

Do not spend compute indefinitely.

Preserve the result and move to another method family.

---

# 22. Held-out policy

Phase B is DEV only.

No held-out evaluation.

After method development is completely finished:

1. choose final method;
2. freeze architecture;
3. freeze checkpoint;
4. freeze `t_start`;
5. freeze NFE;
6. freeze alpha grid;
7. freeze metrics;
8. freeze analysis;
9. record commit and hashes;
10. obtain explicit human approval.

Only then perform one final held-out evaluation.
