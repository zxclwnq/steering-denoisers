# Metrics and Statistical Analysis Contract

## Purpose

This document defines the canonical metrics, comparison units, statistical conventions, and interpretation rules for this repository.

The objective is to prevent silent metric drift, pseudo-replication, invalid matched comparisons, and over-interpretation of proxy metrics.

If implementation code disagrees with this document or with a more specific frozen experiment protocol, stop and resolve the discrepancy before producing headline results.

---

# 1. General principle

Activation steering is a multi-objective problem.

We care simultaneously about:

1. **concept expression**;
2. **language-model quality**;
3. **degeneration**;
4. **mechanistic intervention strength**.

No single scalar metric is sufficient.

The central object is therefore a concept–quality trade-off / Pareto frontier.

A method is interesting only if it improves one objective without merely worsening or weakening another in a trivial way.

---

# 2. Experimental cell

A comparison is only "paired" when the compared methods act on the same underlying experimental cell.

For current steering experiments, a cell is defined by stable quantities such as:

* steering vector ID;
* prompt ID;
* nominal alpha;
* generation seed;
* any additional frozen sampling identifier required by the generation protocol.

For flow methods, the matched flow-noise draw must also derive deterministically from this cell identity.

Method name must not affect the identity of the underlying cell.

Row position in a dataframe must never define randomness.

---

# 3. Clean-model continuation NLL

## Definition

Let:

* (x) be the original prompt;
* (y=(y_1,\ldots,y_T)) be the generated continuation.

The primary language-model quality metric is the mean conditional negative log-likelihood of the continuation under the **clean, unmodified GPT-2 model**:

[
L_{\rm NLL}(x,y)
================

-\frac{1}{T}
\sum_{t=1}^{T}
\log
p_{\rm clean}
(y_t\mid x,y_{<t}).
]

Equivalent interpretation:

For one-hot next-token targets, this is mean autoregressive cross-entropy over continuation tokens.

Lower is better.

Units are natural-log units / nats per continuation token unless explicitly changed.

---

# 4. What NLL does and does not measure

NLL is a functional compatibility metric.

It measures how probable the generated continuation appears under the original clean GPT-2 distribution.

It is useful because interventions that move the model into badly damaged states often produce high-NLL continuations.

However:

**NLL is not a direct semantic fluency oracle.**

Low NLL does not prove:

* target concept preservation;
* factual correctness;
* semantic coherence;
* absence of generic/high-probability blandness.

High NLL does not automatically imply total semantic failure.

Therefore NLL must be paired with concept and degeneration metrics.

---

# 5. BOS/tokenization invariant for NLL

A historical double-BOS bug affected continuation scoring.

The canonical continuation-NLL implementation must satisfy:

* exactly one BOS where the tokenizer/model convention requires one;
* no manually duplicated BOS;
* prompt tokens define conditioning context;
* loss reduction is performed only over continuation tokens;
* prompt tokens must not accidentally contribute to the continuation NLL.

When modifying the implementation, compare against an independent explicit token-by-token or shifted-logit implementation.

A metric implementation that shares the same helper logic is not an independent validation.

---

# 6. Clean LM reference

Current clean reference continuation loss is approximately:

[
L_{\rm clean}\approx3.2723.
]

This number is a diagnostic reference, not a universal constant.

Any experiment using different:

* prompts;
* generation length;
* tokenizer semantics;
* model checkpoint;
* filtering;

may produce a different clean value.

Do not hard-code this value into metric logic.

---

# 7. Functional reconstruction ΔLM

For activation reconstruction experiments, define:

[
\Delta L_{\rm LM}
=================

## L_{\rm modified}

L_{\rm clean}.
]

Where:

* (L_{\rm clean}) is the loss with original clean activations;
* (L_{\rm modified}) is the loss after corrupted/reconstructed activations are inserted.

Interpretation:

[
\Delta L_{\rm LM}=0
]

means no functional LM-loss damage relative to the clean activation.

Smaller positive values are better.

Negative values are possible in principle and should not be automatically clipped.

---

# 8. Recovered functional damage

For diagnostic reconstruction experiments, a useful descriptive quantity is:

[
R
=

\frac{
\Delta L_{\rm corrupted}
------------------------

\Delta L_{\rm reconstructed}
}{
\Delta L_{\rm corrupted}
}.
]

This is the fraction of the corruption-induced LM-loss increase removed by reconstruction.

Use only when:

[
\Delta L_{\rm corrupted}>0.
]

This quantity is descriptive.

Do not compare recovery percentages across fundamentally different corruption distributions without strong justification.

Example:

* additive Gaussian corruption;
* flow convex-combination corruption;

are not directly equivalent corruption scales.

---

# 9. Repetition / degeneration metric

A frozen repeated-pattern metric is used as a degeneration gate.

Current global threshold:

[
T_{\rm rep}\approx0.027786.
]

A sample or cell exceeding the frozen threshold is marked degenerate according to the existing implementation/protocol.

Do not retune this threshold because a new method produces inconvenient results.

The threshold was calibrated against clean/unsteered behaviour.

---

# 10. Degeneration and Pareto analysis

Degenerate observations may be shown in diagnostic plots.

However:

**degenerate observations do not define the primary valid concept–quality Pareto frontier.**

If a method obtains very high concept score only after crossing the degeneration gate, report this as a stress-regime effect rather than primary success.

---

# 11. Distinct-n diagnostics

Additional diversity diagnostics:

* distinct-1;
* distinct-2;
* distinct-3.

These are secondary.

They can help distinguish:

* repetition collapse;
* vocabulary narrowing;
* locally diverse but semantically broken text.

Do not substitute them for NLL or semantic concept evaluation.

---

# 12. Frozen lexicon concept score

Each concept has a frozen lexicon defined before behavioural evaluation.

The score measures the presence/frequency of lexicon-linked concept expression according to the established implementation.

Use the existing implementation and frozen lexicons.

Do not expand or edit a lexicon after seeing method outputs unless creating a new explicitly post-hoc protocol.

---

# 13. Known lexicon failure modes

Lexicon score is useful but incomplete.

Observed examples:

## locations_addresses

Lexicon response is relatively strong while target SAE response is weaker.

Interpretation:

common address/location-related words can cause the lexicon to over-report concept expression.

## sports_awards

Lexicon response is weak while target SAE response is strong.

Interpretation:

semantic/mechanistic steering can occur without frequent use of frozen lexicon words.

Therefore:

[
\text{lexicon score}
\neq
\text{semantic concept strength}.
]

---

# 14. Target SAE metric

The target SAE feature associated with the steering direction is monitored during evaluation.

This provides a mechanistic proxy for reactivation of the feature.

Higher target SAE activation can indicate stronger feature-level expression.

However:

[
\text{high target SAE activation}
\not\Rightarrow
\text{high semantic concept quality}.
]

Clamping experiments demonstrated this failure mode.

---

# 15. Unrelated SAE controls

Unrelated SAE features are used as controls.

Purpose:

* detect broad activation inflation;
* detect nonspecific feature excitation;
* distinguish target-specific response from general activation pathology.

A method that increases both target and unrelated features indiscriminately should not be interpreted as clean concept amplification.

---

# 16. Semantic LLM concept judge

The secondary semantic judge rubric is:

## 0 — absent

Target concept is not meaningfully expressed.

## 1 — weak / indirect

The continuation has a weak, peripheral, ambiguous, or indirect relation to the target concept.

## 2 — clear / substantive

The continuation clearly and meaningfully expresses the target concept.

No rationale is required in the primary structured output.

---

# 17. Concept judges

The established primary semantic judges are:

* Qwen;
* Luna.

Matched-cell concept auditing showed good ordinal agreement between them.

A third ChatGPT audit was used during calibration but is not the default primary pair for future protocol execution unless explicitly requested.

---

# 18. LLM fluency judge is rejected

Do not use the previous absolute or pairwise LLM fluency judge as a primary quality metric.

Reason:

Absolute fluency scoring showed judge-specific categorical cutpoints.

Pairwise scoring showed:

* position bias;
* swap inconsistency;
* unstable tie behaviour.

Therefore the frozen conclusion is:

**LLM fluency calibration was inconclusive and rejected for primary evaluation.**

Primary quality remains:

* clean GPT-2 continuation NLL;
* frozen degeneration/repetition gate.

---

# 19. Matched semantic auditing

Semantic comparisons should use the same underlying experimental cells for all methods being compared.

Do not independently sample 50 examples per method and then call differences "paired".

A previous unpaired audit produced an apparent large parallel-only advantage that disappeared after proper matched-cell evaluation.

This is a canonical warning against unpaired semantic comparisons.

---

# 20. Nominal alpha

Nominal additive steering:

[
h_s=h+\alpha v.
]

Normalized steering strength:

[
\hat\alpha
==========

\frac{\alpha}{E|h|}.
]

Current corrected mean activation norm:

[
E|h|\approx88.76.
]

Primary alpha grid:

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

Stress settings:

[
\hat\alpha
\in
{1.5,2.0}.
]

---

# 21. Realised steering projection

Nominal alpha is not sufficient after learned correction.

For unit steering direction (v):

[
\Delta_v
========

\langle h'-h,v\rangle.
]

For pure additive steering:

[
\Delta_v=\alpha.
]

After a correction method, this may differ substantially from alpha.

Define retained steering fraction:

[
r_{\rm retain}
==============

\frac{\Delta_v}{\alpha},
\qquad
\alpha\ne0.
]

This is a mandatory mechanistic metric for methods that alter the steered activation.

---

# 22. Correction decomposition

Given additive steered activation:

[
h_s=h+\alpha v
]

and corrected activation:

[
\tilde h,
]

define correction:

[
c=\tilde h-h_s.
]

Parallel component:

[
c_\parallel
===========

(c\cdot v)v
]

for unit (v).

Orthogonal component:

[
c_\perp
=======

c-c_\parallel.
]

Useful diagnostics:

[
|c|,
]

[
|c_\parallel|,
]

[
|c_\perp|,
]

[
\cos(c,v),
]

[
r_{\rm retain}.
]

---

# 23. Scalar attenuation control

Every quality-improving correction must be compared against the possibility:

[
\alpha
\rightarrow
\alpha_{\rm eff}.
]

Simple shrinkage:

[
h'
==

h+\kappa\alpha v.
]

A learned method does not establish a new mechanism merely because:

[
L_{\rm NLL}
]

improves at the same nominal alpha.

The relevant question is whether it improves quality after matching:

* realised target projection;
* target SAE strength;
* lexicon strength;
* semantic concept strength where feasible.

---

# 24. Pareto dominance

For concept score (C) where higher is better and quality loss (Q) where lower is better, point (A) dominates point (B) if:

[
C_A\ge C_B
]

and

[
Q_A\le Q_B,
]

with at least one strict inequality.

Primary frontier construction excludes points failing the frozen degeneration gate.

Do not call a method "better" because it dominates only one arbitrarily selected alpha of another method.

Compare frontiers.

---

# 25. Primary Pareto views

At minimum construct:

[
\text{lexicon}
\quad\text{vs}\quad
L_{\rm NLL},
]

[
\text{target SAE}
\quad\text{vs}\quad
L_{\rm NLL}.
]

For mechanistic analysis also use:

[
\text{realised projection along }v
\quad\text{vs}\quad
L_{\rm NLL}.
]

If semantic LLM scores are available:

[
\text{semantic concept score}
\quad\text{vs}\quad
L_{\rm NLL}.
]

---

# 26. Statistical unit

There are many generated samples but only a small number of steering directions.

Do not treat every generated row as an independent scientific replicate for a claim about generalization across concepts.

The analysis should remain **vector-aware**.

Prompt/seed repetitions improve precision within a vector but do not create dozens of independent concepts.

---

# 27. Paired comparisons

When the same experimental cell exists under methods (A) and (B), define:

[
d_i
===

## m_i^{(A)}

m_i^{(B)}.
]

Use these paired differences whenever possible.

Do not use an unpaired bootstrap when exact pairing exists unless there is a documented reason.

---

# 28. Bootstrap confidence intervals

Use bootstrap confidence intervals for effect summaries when appropriate.

For headline comparisons, bootstrap should preserve the scientific grouping structure.

Preferred options:

* hierarchical/vector-aware bootstrap;
* bootstrap over vectors with within-vector aggregation;
* paired cell bootstrap plus vector robustness analysis.

Avoid naïvely bootstrapping thousands of rows as independent observations if the claim is concept-general.

Record:

* bootstrap unit;
* number of resamples;
* seed;
* confidence level.

---

# 29. Vector-wise sign consistency

For each steering direction (v_j), compute the method effect:

[
\Delta_j.
]

Report counts such as:

* positive;
* tie;
* negative.

Example:

`5 / 0 / 3`

means:

* 5 vectors favour method A;
* 0 ties;
* 3 favour method B.

This is an important robustness summary when (n_{\rm vectors}=8).

---

# 30. Leave-one-vector-out analysis

For important aggregate findings, compute the headline effect after removing each vector in turn.

Report:

[
{\Delta_{-j}}_{j=1}^{8}.
]

If an aggregate claim disappears or reverses when one vector is removed, state this explicitly.

Do not hide concept dependence behind a pooled mean.

---

# 31. Sign tests / small-n inference

With only 8 DEV directions, sign-based inference may be informative when vector-level effects are consistently directional.

Do not overstate exact p-values from small (n).

Prefer reporting:

* direction counts;
* effect magnitudes;
* bootstrap intervals;
* leave-one-vector-out stability.

---

# 32. Correlations

Correlations between proxy metrics must be computed at a clearly stated level.

Possible levels:

## Cell level

All matched generated cells.

## Within-vector

Compute association after conditioning/grouping by steering direction.

## Vector aggregate

One summary value per concept direction.

These answer different questions.

Do not report a six-method aggregate correlation and infer that two metrics are globally related or unrelated.

Previous misleading aggregate interpretations were retracted after proper matched-cell analysis.

---

# 33. Known matched correlations

In the established matched semantic dataset, concept-related metrics were substantially correlated rather than independent.

Representative observed relationships included:

* lexicon vs Qwen semantic judge: substantial positive correlation;
* lexicon vs SAE: substantial positive correlation;
* Qwen vs Luna: strong positive correlation.

Therefore do not describe lexicon/SAE/LLM concept metrics as "orthogonal" without a new analysis demonstrating this.

They are imperfect and non-equivalent, but meaningfully related.

---

# 34. Multiple hypotheses

The project contains several mechanistic hypotheses.

Do not silently search many comparisons and report only the best one as confirmatory.

Each result should be labelled:

* frozen / preregistered;
* DEV method development;
* exploratory;
* post-hoc.

For exploratory sweeps, exact family-wise multiple-testing correction is not always necessary, but the search process must be transparent.

---

# 35. Candidate selection before LLM judging

If cheap metrics are used to choose methods for expensive semantic judging:

1. define the candidate-selection rule;
2. freeze the selected candidates;
3. only then unblind/obtain semantic judge results.

Do not inspect semantic scores and then redefine the candidate set.

---

# 36. Alpha=0 semantics

Different methods may have different semantics at nominal alpha zero.

Examples:

## Additive

[
\alpha=0
]

is identity.

## Direct denoiser

[
D(h)
]

may alter the clean activation.

## Fixed-(t) flow/SDEdit

may alter the activation even when steering alpha is zero.

Therefore always state whether alpha=0 is:

* exact no-op;
* clean correction;
* stochastic reconstruction;
* clamping ablation.

Do not pool these rows as a single "unsteered baseline".

For flow implementation, an explicit `t_start=0` path should be exact identity with zero flow forwards.

---

# 37. Phase B-specific statistical requirements

For the active flow Phase B experiment:

At fixed `t_start`, compare:

[
NFE=1,;3,;5
]

using matched cells.

Phase A predicts near-equivalence.

Report paired differences in:

* NLL;
* lexicon;
* target SAE;
* retained steering fraction;
* degeneration.

Do not collapse the NFE arms before reporting this test.

---

# 38. Phase B `t_start` analysis

Primary flow axis:

[
t_{\rm start}
\in
{0.10,0.25,0.50}.
]

Analyze whether:

[
t_{\rm start}\uparrow
]

causes:

* greater quality recovery;
* lower retained steering;
* lower/higher concept metrics;
* more or less degeneration.

The key question is whether any quality improvement remains after matching effective steering strength.

---

# 39. Reporting negative results

A confidence interval overlapping zero is not automatically evidence of equality.

Use language such as:

* "not resolved";
* "not supported";
* "consistent with no detectable effect at current precision".

Use stronger equivalence language only when an equivalence margin/protocol supports it.

Likewise, a statistically resolved tiny effect may be scientifically unimportant.

Report magnitude as well as uncertainty.

---

# 40. Canonical reporting template

For an important method comparison report:

1. method definitions;
2. exact matched sample count;
3. valid/nondegenerate sample count;
4. mean/median effect;
5. confidence interval;
6. vector-wise signs;
7. leave-one-vector-out range if headline;
8. realised-steering difference;
9. degeneration difference;
10. interpretation;
11. alternative explanation;
12. whether the result was frozen or exploratory.

---

# 41. Forbidden analysis shortcuts

Do not:

* use row count as independent (n) when vector dependence matters;
* compare independently sampled method subsets as paired;
* retune degeneration thresholds per method;
* select alpha per vector using held-out behaviour;
* report only nominal alpha after a correction changes realised steering;
* equate SAE activation with semantic inclusion;
* use LLM fluency as established truth;
* average away large vector-specific reversals;
* silently remove failed seeds;
* discard degenerate outputs without reporting their frequency;
* define the hypothesis after seeing the strongest metric.

---

# 42. Scientific success criterion

The strongest evidence for a useful correction method is:

[
\text{better quality}
]

at matched:

[
\text{semantic concept expression}
]

and/or matched:

[
\text{realised steering strength},
]

with:

* no increase in degeneration;
* vector-level robustness;
* reproducible matched comparisons;
* no protected-data leakage.

Anything weaker should be described accordingly.
