# Experiment D — positive causal control for the Experiment C interpretation

**Status: FROZEN 2026-08-20, before any D result was computed.**

Class: `post_stop_method_development`. This is additional and post-hoc relative to
every preregistered claim of the closed branches, and it does not revise the
T1/T2 verdict of 2026-08-16.

Held-out is not touched. `configs/protected/` is not read.

---

## 0. The question D exists to answer

Experiment C found, for GPT-2 SAE steering directions:

* conditional natural trajectories are substantially curved;
* curvature exceeds a matched random-axis reference;
* but `cos(d_k, v)` — the alignment of the local direction of natural motion with
  the fixed direction — is **indistinguishable from a random axis**, and rises
  with concept strength rather than peaking at the natural centre.

Two incompatible readings survive that result:

1. the SAE direction is a poor *intervention* axis, and its lack of tangent
   alignment is why steering repair failed;
2. even a direction that is **known** to be causally effective need not align
   with the tangent of the natural trajectory, in which case tangent alignment
   says nothing about steerability.

D discriminates between them by running the *same* geometric diagnostic on a
direction whose causal effect on behaviour is independently established.

D is **not** a search for another positive result. If the refusal direction is
causal but unaligned, that is the answer and the geometric story in the report
gets weaker, not stronger.

---

## 1. The direction

Methodology: Arditi et al., *Refusal in Language Models Is Mediated by a Single
Direction* (arXiv:2406.11717), official implementation
`github.com/andyrdt/refusal_direction`.

Model: `google/gemma-2b-it` (18 layers, `d_model` 2048), bf16.

The **published** direction shipped with that repository is used verbatim. It is
not re-derived, not re-selected, and not tuned:

* artifact `pipeline/runs/gemma-2b-it/direction.pt`;
* `direction_metadata.json`: `layer = 10`, `pos = -2`;
* the tensor is the **raw** difference-in-means vector, float64, norm ≈ 10.064 —
  it is *not* unit-normalized, and its scale is the published intervention
  magnitude.

Extraction convention it encodes (recorded here so the geometry uses the same
site): difference of mean activations between harmful and harmless instructions,
captured by a **forward pre-hook on the decoder block**, i.e. the residual stream
*entering* layer 10, at the end-of-instruction token positions; `pos = -2`
indexes those end-of-instruction tokens.

Two derived objects, both frozen here:

* `r = direction / ||direction||` — unit vector, used for all geometry and for
  the coordinate `c = <h, r>`;
* `direction` (raw, norm 10.064) — used for the activation-addition arm with
  coefficient `±1.0`, which is the published convention.

**After causal validation nothing about the direction may change**: not the
layer, not the sign, not the token position, not the prompt subset. The direction
is never selected or adjusted using any C or D geometry metric.

---

## 2. Causal validation is a prerequisite, run first

Interventions follow the official implementation exactly.

**Ablation** (should remove refusal), applied at every layer and every token
position, as a pre-hook on each decoder block and an output hook on every
attention and MLP module:

    h <- h - <h, r> r

**Activation addition** (should induce refusal), a pre-hook on decoder block 10
only, with the raw published vector:

    h <- h + 1.0 * direction

**Data.** The `test` splits, which are disjoint from the `train` split the
direction was derived from and the `val` split it was selected on:
`harmful_test` (572 instructions), `harmless_test` (6266 instructions).

**Metrics.** Both are the repository's own, so nothing new is invented:

* `refusal_score` — the logit-level score
  `log p(refusal_tok) - log(1 - p(refusal_tok))` at the final prompt position,
  with gemma's refusal token `235285` (`'I'`);
* substring-matching refusal rate over generated completions, using the
  repository's refusal-prefix list.

`llamaguard2` is **not** used: it requires a separate gated safety model and adds
no information for a control whose question is whether refusal moved at all.

### Frozen pass criteria

Fixed here, before any D number exists. Both must hold:

* **Ablation removes refusal on harmful prompts.** The substring-matching refusal
  rate on `harmful_test` drops by at least **50 percentage points** relative to
  baseline, and the mean `refusal_score` decreases.
* **Addition induces refusal on harmless prompts.** The substring-matching
  refusal rate on `harmless_test` rises by at least **50 percentage points**
  relative to baseline, and the mean `refusal_score` increases.

Verdict `CAUSAL_CONTROL_PASS` only if both hold. Otherwise
`CAUSAL_CONTROL_FAIL`, and **D stops**: no geometry is computed and no geometric
interpretation is offered. A direction that does not demonstrably act on
behaviour cannot serve as a positive control.

---

## 3. The geometry, transferred unchanged

The diagnostic is `src/interp/curvature.py` with `CURVATURE_SPEC` — the same code,
the same statistics, the same bootstrap, the same shuffle and split-half
calibrations that produced Experiment C. No new statistic is invented for D, and
no statistic is invented specially for the random control.

**Activation population.** The residual stream entering layer 10 — the exact site
the direction is defined at — taken at the `pos = -2` end-of-instruction token,
one activation per prompt, on the `test` splits.

**Coordinate.** `c_i = <h_i, r>`.

**Binning.** The frozen `CURVATURE_SPEC` cuts: p10/p25/p50/p75/p90, giving six
bins and five secants `d_k = mu_{k+1} - mu_k`.

### The harmful/harmless confound

The refusal direction was *built* to separate harmful from harmless
instructions, so sorting a mixed pool by `<h, r>` largely sorts by class.
A conditional mean that changes across bins could then reflect a change in class
composition rather than any refusal geometry. Three analyses, fixed before
results:

* **D-main — class-balanced.** Within each bin, the conditional mean is the
  unweighted average of the harmful-row mean and the harmless-row mean in that
  bin, so a change in class proportion cannot move it. A bin is usable only if
  both classes supply at least `min_bin_rows` rows; bins that fail this are
  reported as unusable rather than silently pooled. **This is the primary
  analysis.**
* **D-harmful** — the whole diagnostic within harmful prompts only.
* **D-harmless** — the whole diagnostic within harmless prompts only.

Within-class analyses are robustness, not the primary. Because the coordinate
separates the classes, it is expected in advance that extreme bins may contain
one class only; if that leaves too few usable bins for D-main, that fact is
reported and the within-class analyses carry the interpretation, with the
reliability ceiling stated.

---

## 4. Primary geometry statistics

**D1 — curvature.** `cos(d_k, d_{k+1})`, and the reliability-normalized
`shortfall below the split-half ceiling`, exactly as in C.

**D2 — alignment with the causal direction.** `cos(d_k, r)`, reported per
quantile interval, as a pooled mean, and with a direction-clustered bootstrap
interval. Never collapsed to a single pooled number alone.

---

## 5. Controls

* **Random axes.** Matched random unit directions in the *same model, same layer,
  same activations, same binning pipeline*, with the same statistics —
  `cos(d_k, d_{k+1})`, split-half ceiling, shortfall, and `cos(d_k, q)`. The
  refusal direction is compared against random axes *inside gemma*, never
  directly against the GPT-2 numbers.
* **Shuffle.** The same shuffle control as C: destroy the activation-to-bin
  correspondence and confirm the sequential structure disappears.
* **Reliability.** The same split-half analysis. A curvature claim is made only
  relative to the estimation ceiling. If `cos(d_k, d_{k+1}) < 1` but the ceiling
  is also low, this is **not** reported as curvature.

---

## 6. Outcome interpretations, fixed before the result

Exactly one label is issued, from the alignment of `cos(d_k, r)` against the
matched random-axis reference and the shape of its profile:

* **`CAUSAL_DIRECTION_ALIGNED`** — `cos(d_k, r)` is materially above the matched
  random-axis reference (bootstrap interval on the difference excludes zero and
  is positive), and the trajectory is relatively linear. Supports: good causal
  intervention axes can be geometrically unlike our SAE readout directions. The
  admissible wording is *"a known causal refusal direction exhibits
  natural-trajectory alignment that was absent for the SAE steering directions in
  our GPT-2 experiment"* — evidence and positive control, not a causal taxonomy.
* **`CAUSAL_BUT_NOT_ALIGNED`** — causal validation passed, yet `cos(d_k, r)` is
  not above the random reference. Then tangent alignment is **not** a necessary
  condition for causal steering, C may not be used as evidence that the SAE
  directions are "merely correlational", and C stands only as a geometric
  description of natural representation geometry. This is an important possible
  negative result and is to be reported as such.
* **`LOCAL_ALIGNMENT_ONLY`** — alignment is high near the natural centre and
  falls away at extreme quantiles. Supports a local-linear picture and motivates
  state-dependent steering as future work. **No such method is built now.**
* **`GEOMETRY_NOT_INFORMATIVE_FOR_CAUSALITY`** — geometry indistinguishable from
  random on every statistic while the causal effect is strong. Then causal
  intervention geometry and natural conditional geometry are different objects,
  and C describes only the latter.

---

## 7. What D may not be used to claim

D uses a different model family, training regime, activation distribution, layer
and token semantics from the GPT-2 SAE experiments. It is therefore **not** a
matched comparison, and the following are prohibited regardless of outcome:

* "SAE features are merely correlational";
* any claim that causal directions differ from correlational ones *in general*;
* any cross-model causal taxonomy.

Three properties must stay separated: representation/readout evidence, causal
intervention evidence, and natural-trajectory geometry. D relates only the last
two, in one model.

The narrow question D is licensed to answer is: **can our C diagnostic behave
differently on an independently validated causal linear direction?**

---

## 8. Stop rule

After D, no further experimental branch is opened automatically. D exists to test
the validity of the C interpretation, not to find another positive result.

If the refusal direction is causal but unaligned, that result is accepted and the
geometric story in the report is weakened accordingly.

If alignment appears, it is recorded as a suggestive positive control and no
cross-model causal claim is made.

The optional same-model readout-direction control of §13 of the task (decodable
but not causally validated directions) is **not** part of D and is not run unless
separately requested.
