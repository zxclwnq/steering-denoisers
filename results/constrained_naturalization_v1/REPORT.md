# Constrained Naturalization: the prior preserves the constraint but cannot naturalize

Diagnostic `constrained_naturalization_v1`. Checkpoint
`conditional_flow_60m_fw32m_v1` / `best_step_249500.pt`. Frozen validation
activations, training-only pool directions, targets reused from the frozen
`natural_support_v1` quantile plan. No DEV, no held-out, no LLM judge. Nothing
trained.

## Headline

**Category B.** The constrained sampler works exactly as derived — the projected
arm holds the requested coordinate to 0.0000 with zero parallel correction — and
the learned prior still makes the language model *worse* than an ordinary hard
clamp at every one of the 90 flow cells that preserve the constraint.

The one cell in 90 whose confidence interval falls below zero is the
*unconstrained* arm at the most extreme target, and it buys its improvement by
letting the requested coordinate slip. That is attenuation, not naturalization.

## 1. Constrained-noising derivation

In standardized coordinates the constraint is `<x, v_x> = c_x`, `||v_x|| = 1`.
Ordinary forward noising violates it twice:

    x_t = (1-t) x_0 + t eps
    <x_t, v_x> = (1-t) c_x + t <eps, v_x>

the coordinate is shrunk by `(1-t)`, and Gaussian mass is injected along `v_x`.

Split `x_0 = c_x v_x + x_0_perp`, noise only the orthogonal part, pin the
parallel part:

    x_t = c_x v_x + (1-t)(x_0 - c_x v_x) + t eps_perp
        = (1-t) x_0 + t (eps_perp + c_x v_x)

giving `<x_t, v_x> = (1-t) c_x + t c_x = c_x` for every `t`.

The closed form is ordinary interpolation toward a modified endpoint

    eps' = eps_perp + c_x v_x = eps - (<eps, v_x> - c_x) v_x

which is exactly the affine projection of `eps` onto the constraint hyperplane.
So constrained noising is ordinary SDEdit with the Gaussian endpoint projected;
nothing else changes.

Reverse integration still leaves the hyperplane because the learned velocity has
an unconstrained parallel component. The projected arm restores it analytically
after each Euler step:

    x <- x + (c_x - <x, v_x>) v_x

Projections are never counted as network evaluations.

## 2. Files changed

* `src/interp/constrained_flow.py` — new. Projection, constrained endpoint,
  constrained noising, projected reverse Euler, four arms, correction
  decomposition.
* `tests/test_constrained_flow.py` — new, 28 geometry tests.
* `scripts/constrained_naturalization.py` — new runner.
* `results/constrained_naturalization_v1/` — results, raw rows, this report.

Nothing existing was modified. No training code touched.

## 3. Geometry tests — 28/28 pass

| invariant | result |
|---|---|
| `<h_clamp, v> = c_target` | passes, atol 1e-5 |
| raw ↔ standardized constraint equivalence under the normalizer | passes, atol 1e-4 |
| ordinary SDEdit *breaks* the hyperplane (premise) | confirmed broken |
| constrained noising holds `<x_t,v_x> = c_x` at t ∈ {0, .10, .25, .50, .90, 1.0} | passes, atol 1e-4 |
| constrained endpoint equals the affine projection of eps | passes, atol 1e-5 |
| constrained noising still moves the orthogonal subspace | passes |
| projected Euler holds the constraint after *every* step | max residual < 1e-4 |
| unprojected integration leaves the hyperplane | confirmed drifts |
| projection is idempotent | passes |
| NFE accounting: projections never counted | passes for nfe ∈ {1,3,5} × 3 arms |
| sign canonicalization: (v,c) ≡ (−v,−c) | passes, atol 1e-3 |
| projected arm's parallel correction ≈ 0 | < 1e-3 |

## 4. Hard-clamp baseline (arm A, NFE = 0)

Clean LM loss 3.7702 nats.

| target | NLL | ΔLM vs clean | coordinate error | rel L2 | cosine |
|---|---|---|---|---|---|
| p50 | 3.7734 | +0.0032 | 5.7e−07 | 0.031 | 0.9990 |
| p75 | 3.7736 | +0.0034 | 5.9e−07 | 0.037 | 0.9989 |
| p90 | 3.7779 | +0.0077 | 6.7e−07 | 0.057 | 0.9978 |
| p95 | 3.7849 | +0.0147 | 8.1e−07 | 0.078 | 0.9962 |
| p99 | 3.8239 | +0.0537 | 1.2e−06 | 0.141 | 0.9887 |

The hard clamp is essentially free and numerically exact. This is the bar.

## 5. Flow arms vs hard clamp (paired ΔNLL, negative = better)

Target p90, all t_start × NFE:

| arm | t | nfe | ΔNLL | 95% CI | coord err | ‖Δ∥‖ | ‖Δ⊥‖ |
|---|---|---|---|---|---|---|---|
| sdedit | 0.10 | 1 | +0.0031 | [−0.0001, +0.0062] | 0.703 | 0.703 | 6.743 |
| tangent | 0.10 | 1 | +0.0040 | [+0.0008, +0.0070] | 0.463 | 0.463 | 6.756 |
| projected | 0.10 | 1 | +0.0061 | [+0.0027, +0.0096] | **0.000** | **0.000** | 6.817 |
| sdedit | 0.25 | 1 | +0.0325 | [+0.0244, +0.0409] | 2.468 | 2.468 | 14.859 |
| tangent | 0.25 | 1 | +0.0340 | [+0.0260, +0.0422] | 1.972 | 1.972 | 14.896 |
| projected | 0.25 | 1 | +0.0448 | [+0.0338, +0.0578] | **0.000** | **0.000** | 15.296 |
| sdedit | 0.50 | 1 | +0.2286 | [+0.2020, +0.2541] | 4.028 | 4.028 | 24.804 |
| tangent | 0.50 | 1 | +0.2306 | [+0.2045, +0.2564] | 3.487 | 3.487 | 24.888 |
| projected | 0.50 | 1 | +0.2511 | [+0.2235, +0.2787] | **0.000** | **0.000** | 25.592 |

NFE 3 is uniformly slightly worse than NFE 1 at every t_start.

**Across all 90 flow cells: 2 have mean ΔNLL < 0, and exactly 1 has a CI entirely
below zero.**

## 6. Coordinate error

The projected arm holds the coordinate at exactly 0.0000 everywhere, UP and
DOWN alike. The tangent arm preserves it through forward noising but drifts
during unprojected integration (0.33–3.49). The sdedit arm drifts most
(0.36–4.03), growing with `t_start`.

## 7. Parallel / orthogonal decomposition — the central mechanistic test

By construction `‖Δ∥‖ = 0.000` for the projected arm, so the *only* channel it
can use is `Δ⊥`. It uses that channel heavily — orthogonal correction norm 6.8
at t=0.10 rising to 25.6 at t=0.50 — and the LM gets monotonically worse as it
does:

| t_start | ‖Δ⊥‖ (projected) | ΔNLL vs clamp |
|---|---|---|
| 0.10 | 6.82 | +0.0061 |
| 0.25 | 15.30 | +0.0448 |
| 0.50 | 25.59 | +0.2511 |

Damage scales with the size of the orthogonal edit. The prior's orthogonal
corrections are not merely useless — they are actively harmful, and more of them
is strictly worse.

**There is no evidence for useful orthogonal naturalization at fixed semantic
coordinate. The evidence points the other way.**

## 8. Per-direction signs and LOVO

At t=0.10, nfe=1 the fraction of the 32 directions with negative ΔNLL is
0.22–0.31 for the constrained arms: roughly 70–78% of directions are made worse.
Leave-one-direction-out ranges are entirely positive, e.g. projected p90
[+0.00512, +0.00672]. The failure is uniform across directions, not driven by
outliers — the same robustness that would have supported a positive result
supports this negative one.

## 9. Target quantile and UP/DOWN

ΔNLL is positive at every target quantile for the constrained arms, with the
penalty growing from p50 (+0.0055 projected) to p99 (+0.0094 projected). UP and
DOWN behave alike: the projected arm holds 0.0000 error in both; coordinate slip
in the unconstrained arms appears in whichever direction the request is large.

## 10. Category: **B — flow preserves the constraint but does not naturalize**

Coordinate control is exact where enforced. ΔNLL versus hard clamp is positive
essentially everywhere, with confidence intervals excluding zero. Per the
decision rule this is a strong negative result for the current learned-prior
mechanism: the generative prior cannot find useful orthogonal corrections even
when the intended semantic displacement is explicitly protected.

### The single exception is a category-C artifact

`q99_t0.10_nfe1_sdedit`: ΔNLL −0.00454, CI [−0.00811, −0.00105]. Its constrained
siblings at the identical cell, with identical noise:

| arm | ΔNLL | coordinate error | ‖Δ∥‖ |
|---|---|---|---|
| sdedit | **−0.00454** | 1.4426 | 1.4426 |
| tangent | +0.00513 | 0.6711 | 0.6711 |
| projected | +0.00943 | **0.0000** | **0.0000** |

Perfectly monotone: the more coordinate slip an arm permits, the better its NLL.
The improvement is bought by not performing the requested intervention. It is
attenuation, and it must not be called a success. Its magnitude is −0.005 nats
on a 3.77 baseline (0.12%).

Both readings agree on the decision. Not D: no finite or geometry invariant
failed anywhere.

## 11. Is there evidence for useful orthogonal naturalization at fixed coordinate?

**No, and the result is stronger than a null.** With the coordinate held exactly
and the parallel channel closed to zero, the prior's orthogonal edits degrade
the LM in proportion to their size, uniformly across 32 directions, at all five
target quantiles and all three flow times. The mechanism we hoped for —
hard semantic constraint plus generative orthogonal correction — does not exist
in this checkpoint.

The hard clamp itself remains an excellent intervention: exact coordinate control
at +0.003 to +0.054 nats. Nothing the learned prior does improves on it.

## Not measured

* Target/unrelated SAE activation: the pool artifact carries directions and
  ranks but no SAE feature ids, so features cannot be indexed without rebuilding
  the selection. Category A was already excluded on ΔNLL, so this would not
  change the classification.
* Lexicon: exists only for DEV vectors.
* Repetition/degeneration: this diagnostic substitutes activations and measures
  NLL, it does not generate text.
* Autoregressive generation: substitution at all non-BOS positions of validation
  sequences only.
