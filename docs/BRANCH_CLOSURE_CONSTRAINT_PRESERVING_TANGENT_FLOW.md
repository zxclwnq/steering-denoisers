# Branch Closure: Constraint-Preserving Tangent Flow

## Status

**CLOSED — 2026-08-16. The predefined stop condition fired.**

**Generative naturalization is finished as a research direction.**

This document is final. It is not reopened, revised, or reinterpreted. It closes
not only the tangent branch but the whole generative-naturalization family, per
the stop rule frozen in `docs/TANGENT_FLOW_PROTOCOL.md` §6 before any T2 number
existed.

---

# 1. The result

> **T1 PASSED. T2 FAILED. That is exactly the combination the branch declared in
> advance would end the programme.**

The tangent-trained flow learned its own matched task well — it recovers 77% of
the LM damage caused by tangent corruption. That ability did **not** transfer
into useful post-clamp naturalization. At a semantic coordinate held exactly
fixed, its orthogonal corrections make language-model quality **worse** than a
plain hard clamp.

## Formal T2 statistic

Frozen primary operating point `t_start = 0.10, NFE = 1`, pooled over the five
`natural_support_v1` target quantiles with equal quantile weight:

    pooled paired dNLL = +0.006184 nats
    canonical 95% CI   = [+0.001631, +0.010788]

Positive, with the interval excluding zero. Negative would have been useful; the
effect is the wrong sign and statistically distinguishable from zero.

Bootstrap: `direction_cluster_then_sequence`, equal quantile weight, each drawn
sequence carrying all five of its quantile observations, 32 direction clusters,
64 sequences, 320 paired observations, 2000 resamples, seed 20260906.

Robustness clauses, all failed: only 25% of directions negative (needs >80%);
LOVO range [+0.005244, +0.006816], never crossing zero.

---

# 2. Why this is a real negative and not a plumbing artifact

The coordinate was held exactly. The failure cannot be attributed to leakage,
attenuation, or a broken invariant:

| quantity | value |
|---|---|
| mean parallel correction `\|Δh_∥\|` | 5.6e-07 |
| mean orthogonal correction `\|Δh_⊥\|` | 7.085 |
| ratio parallel / orthogonal | 7.9e-08 |
| max coordinate error | 5.7e-06 |
| max arm coordinate difference | 9.5e-06 (tolerance 1e-3) |
| max pre-projection drift | 2.9e-06 |

The two arms sit on the same semantic coordinate to eight significant figures.
The tangent flow moved **only** orthogonal degrees of freedom, by a large amount
(‖Δh_⊥‖ ≈ 7.1, relative L2 0.113), and every one of those moves cost quality.
Attenuation — the failure mode of the closed isotropic branch — was structurally
impossible here and did not occur.

## The mechanism: correction magnitude *is* the penalty

Pooled across quantiles, NFE = 1:

| t_start | ‖Δh_⊥‖ | pooled ΔNLL |
|---|---|---|
| 0.10 | 7.09 | +0.006184 |
| 0.25 | 16.81 | +0.054130 |
| 0.50 | 29.58 | +0.346458 |

The relationship is monotone and steepening: more orthogonal correction, worse
LM quality, super-linearly. The prior's notion of "more natural" is orthogonal —
in the literal and the figurative sense — to what the language model needs.

---

# 3. The hypothesis this kills

The branch existed to test one specific explanation for the closed
generic/conditional branch's failure:

> The previous priors were trained on isotropic corruption but asked at inference
> to preserve a semantic coordinate. Train/test corruption geometry may therefore
> be mismatched.

**That explanation is now refuted.** The tangent model was trained directly on
the constraint-preserving geometry it was evaluated under. The mismatch was
removed. The outcome did not change.

## Side-by-side with the closed branch, same frozen plan

`t_start = 0.10, NFE = 1`, paired ΔNLL vs hard clamp:

| target | closed branch (isotropic, projected) | this branch (tangent-trained) |
|---|---|---|
| p50 | +0.005493 (‖Δh_⊥‖ 6.75) | +0.004702 (‖Δh_⊥‖ 7.02) |
| p75 | +0.005790 (6.76) | +0.005042 (7.03) |
| p90 | +0.006119 (6.82) | +0.005572 (7.06) |
| p95 | +0.006488 (6.88) | +0.006363 (7.09) |
| p99 | +0.009429 (7.11) | +0.009241 (7.23) |

Training on matched geometry bought a few thousandths of a nat and did not change
the sign. The two methods are, for practical purposes, the same failure.

The hard-clamp baseline reproduced **exactly** across the two runs — ΔLM
+0.00318 / +0.00341 / +0.00767 / +0.01470 / +0.05369 at p50…p99 in both — despite
different sessions, a regenerated activation artifact and a different GPU. The
comparison is sound.

---

# 4. What was durably established

Preserve these; do not collapse the branch into "tangent flow failed".

* **The tangent geometry is correct and implementable.** `<x_t, v> = c` holds at
  every `t` and across every Euler step to ~1e-6, with the numerical safeguard
  demonstrably not doing the work (identical results with it disabled).
* **The objective is learnable.** T1 recovered fraction 0.7730 at the frozen
  primary cell, paired ΔNLL −1.012611, CI [−1.072443, −0.949261]. Validation
  tangent MSE 0.9662 against a 1.9951 zero-predictor.
* **Multi-step integration buys nothing.** NFE 1 → 3 → 5 is flat in T1 and T2
  alike, confirming the cheap-prior operating regime was never the limitation.
* **Hard clamp remains a very strong baseline** at +0.003 to +0.054 nats with
  exact coordinate satisfaction. Nothing in this programme beat it.
* **A caution recorded during T1:** an unmatched 60M isotropic prior scored
  *better* on the tangent reconstruction task (recovered fraction 0.8371 vs
  0.7730) than the 16M tangent model. Confounded by 3.7x capacity, so it isolates
  nothing — but it was an early sign that matched corruption geometry was not
  buying what the branch assumed.

---

# 5. What must not happen next

The stop rule named these in advance, before any T2 number was visible. They
remain forbidden without a new, separately justified research programme:

* no larger model (60M confirmation or otherwise);
* no additional NFE;
* no further generic training data;
* no alternative corruption scheme;
* no auxiliary losses;
* no LLM judge;
* no post-hoc search of the diagnostic grid for a favourable cell.

The diagnostic grid was searched for completeness and reported in full: **0 of 30
cells favourable.** There is no cell to rescue.

A matched 16M isotropic control would isolate the causal contribution of tangent
training. Under the stop rule it is **not** authorized, because the question it
answers — *why* the tangent objective did not help — no longer changes any
decision. It is recorded here as the one experiment that would add understanding,
should the programme ever be revisited for publication.

---

# 6. Interpretation

> Even after removing the train/inference corruption-geometry mismatch — the last
> remaining explanation for the earlier failures — a learned activation prior does
> not produce useful orthogonal naturalization of a hard-clamped activation. The
> prior can reconstruct natural activations, and can do so under an exactly
> preserved semantic constraint, yet every orthogonal edit it makes costs language
> model quality in proportion to its size.

This is a clean, informative negative for the whole generative-naturalization
family, and should be reported as one. The programme asked a well-posed question,
built the controls to answer it, pre-committed to the decision rule, and got a
clear answer.

---

# 7. Provenance

| item | value |
|---|---|
| checkpoint | `best_step_249000.pt` |
| checkpoint SHA256 | `066afb601418da89f79b003c97b37227a9aa7702a442ad5f3fb0ab68a4199d4c` |
| selection | `val_flow_mse` minimized = 0.9680510, concept-independent, verified |
| config fingerprint | `e4af61135b0205cdcd6f196a61d5af464f0369b29e9bcaa471fd0764e7f85499` |
| training source revision | `snapshot-sha256:93eb1617f2105badacf4690856a4ec9f4d4d9106fa54308f48cccd41ef0c6aab` |
| T2 source revision | `snapshot-sha256:234752a0245f4ab73b72d09e7f3d5d1f500bf57ff74a209babbe8931cf4a126c` |
| direction pool | `45241c49814abe71ed7106e1a0fcbbe7d8aad40b215621674ec72ee7356d7a2c` (training-only, rank floor 256) |
| validation artifact | `resid7_fw_val_1024k_v1`, split fingerprint `5725541eaecf437c` |
| T1 receipt | PASS, `t0.50_nfe1_tangent`, formally eligible |

The T2 source revision differs from T1's because the formal pooled statistic
(equal quantile weight, quantile rows held together under resampling) was
specified and implemented **after** T1 and **before** any T2 result existed.

Access receipts: `dev_vectors_accessed: false`, `held_out_accessed: false`,
`llm_judge_used: false`, `trained_anything: false`.

## Artifacts

    results/tangent_t1_v1/     T1 run, evaluation, receipt, checkpoint, MANIFEST.sha256
    results/tangent_t2_v1/     T2 result, raw rows, receipt, T1 receipt used
