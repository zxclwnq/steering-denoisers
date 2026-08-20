# Natural-Support Controllability: the conditional prior *is* a coordinate controller

Diagnostic `natural_support_v1`. Checkpoint `conditional_flow_60m_fw32m_v1` /
`best_step_249500.pt`. Frozen validation activations, training-only pool
directions. No DEV vectors, no held-out data, no LLM judge. Nothing was trained.

## Headline

**The C1 failure was extrapolation, not absence of controllability.**

Inside natural support the model tracks a requested coordinate almost
one-to-one: calibration slope **0.906** (95% CI 0.867–0.947) at `t_start = 0.90`,
90.8% of rows moving in the requested direction, all 32 directions positive.

But controllability only appears at a flow time where the prior destroys the
language model: `t_start = 0.90` costs **+4.08 nats** on a clean baseline of
3.77, *before any steering is requested*. At the one usable quality point
(`t_start = 0.50`, +0.23 nats) the slope is 0.055.

That is a quality–controllability dissociation, not a conditioning failure.

## 1. Scale — why C1 failed

Natural coordinate distribution over 262144 reference activations, 32 directions:

| statistic | median across directions | range |
|---|---|---|
| std | 3.602 | 2.899 – 5.970 |
| p01 | −5.616 | −36.620 – +1.804 |
| p50 | −0.476 | −24.055 – +9.295 |
| p90 | +4.945 | −16.130 – +15.862 |
| p99 | +12.340 | −8.101 – +28.758 |

The natural p01→p99 span is roughly 18 units. The C1 pre-check requested
displacements of 8.9 to 88.8 units, i.e. **2.5σ to 25σ**, because `alpha_hat` is
scaled to the activation *norm* (88.76) rather than to coordinate spread (~3.6).
The ~8-unit saturation ceiling C1 found is ≈2.2σ — the edge of natural support.
The old scale asked the model for coordinates it had never seen.

## 2. Calibration by flow time (nfe = 1, correct condition)

| t_start | slope | 95% CI | Pearson | Spearman | frac. correct direction | ΔLM (self arm) |
|---|---|---|---|---|---|---|
| 0.50 | 0.055 | 0.041 – 0.072 | — | — | 0.569 | +0.233 |
| 0.75 | 0.212 | 0.161 – 0.271 | — | — | 0.712 | +1.146 |
| 0.90 | 0.906 | 0.867 – 0.947 | — | — | 0.908 | +4.077 |

Monotonic across target quantiles at 0.75 and 0.90; not at 0.50.

## 3. Quantile table (t_start = 0.90, nfe = 1)

| target | requested coord | realised coord | requested disp | realised disp | control fraction | ΔLM |
|---|---|---|---|---|---|---|
| p50 | −1.059 | −0.864 | −0.781 | −0.586 | 0.750 | 4.219 |
| p75 | +1.317 | +0.724 | +1.697 | +1.104 | 0.650 | 4.232 |
| p90 | +4.074 | +2.479 | +4.440 | +2.845 | 0.641 | 4.380 |
| p95 | +6.159 | +3.523 | +6.518 | +3.881 | 0.596 | 4.519 |
| p99 | +11.971 | +5.643 | +12.280 | +5.952 | 0.485 | 4.868 |

Control fraction declines as the request grows: the same saturation as C1, but
now visible *inside* support and far milder.

At `t_start = 0.50` realised displacement never exceeds 0.20 regardless of
request — no control at usable quality.

## 4. UP vs DOWN

At `t_start = 0.90`, split by sign of the request:

| target | direction | n | requested | realised | control fraction | correct sign |
|---|---|---|---|---|---|---|
| p50 | up | 3501 | +2.395 | +2.369 | 0.989 | 0.954 |
| p50 | down | 3833 | −3.682 | −3.286 | 0.892 | 0.937 |
| p90 | up | 7098 | +5.356 | +3.663 | 0.684 | 0.888 |
| p90 | down | 808 | −3.609 | −4.343 | 1.203 | 0.953 |
| p99 | up | 8020 | +12.453 | +6.096 | 0.489 | 0.907 |
| p99 | down | 89 | −3.291 | −7.003 | 2.128 | 0.989 |

Both directions are controlled. For displacements within about ±3 units (~1σ)
control is essentially calibrated (0.89–0.99). DOWN control overshoots at p99,
but n=89 there and the requested magnitudes are small; treat as noise.

At `t_start = 0.50` both directions are near zero (0.02–0.12).

## 5. Correct vs shuffled condition (target p90, nfe = 1)

| t_start | arm | slope | mean abs coord error | realised disp | ΔLM |
|---|---|---|---|---|---|
| 0.50 | correct | 0.080 | 5.027 | 0.102 | 0.235 |
| 0.50 | shuffled | 0.039 | 8.465 | 0.100 | 0.236 |
| 0.75 | correct | 0.279 | 4.494 | 0.557 | 1.166 |
| 0.75 | shuffled | 0.154 | 7.302 | 0.486 | 1.169 |
| 0.90 | correct | 0.930 | 2.050 | 2.845 | 4.380 |
| 0.90 | shuffled | 0.484 | 4.766 | 2.128 | 4.543 |

Correct conditioning beats shuffled at every flow time, on both slope and
coordinate error. The gap widens with `t_start`, mirroring the frozen
condition-use diagnostic.

Caveat on the shuffled arm: it shares `c0` with the correct arm, so its
requested delta is correlated with the correct one through `−c0`. Its nonzero
slope is partly that shared term, not evidence that a wrong target is followed.
The correct-vs-shuffled *gap* is the sound comparison.

## 6. ΔLM trade-off — the blocking result

Clean GPT-2 loss on these 64 validation sequences: **3.770 nats**.

| t_start | self-arm ΔLM (no steering requested) | as % of clean | marginal ΔLM of control |
|---|---|---|---|
| 0.50 | +0.233 | +6% | +0.002 |
| 0.75 | +1.146 | +30% | +0.067 |
| 0.90 | +4.077 | +108% | +0.791 |

The decomposition matters. At `t_start = 0.90` almost the entire cost is the
prior's own reconstruction, not the act of steering: asking for the *natural*
coordinate already costs +4.077. Controlling on top adds +0.79.

So control is cheap; the operating point where control exists is ruinous. The
defect is the prior's reconstruction fidelity at high flow time (relative L2
0.707, cosine 0.702 at t=0.90), not the conditioning mechanism.

## 7. Direction robustness

At `t_start = 0.90`, target p90: 32 of 32 directions have positive slope, median
0.907, range 0.809–0.951, quartiles [0.884, 0.907, 0.928]. Leave-one-direction-out
pooled slope stays inside 0.9242–0.9359. The effect is not carried by any
direction.

## 8. Category

**A on every controllability clause; blocked by A's quality clause.**

The specified rule admits category A only if "quality cost is not catastrophic
inside natural support". At the only `t_start` where controllability holds, LM
loss more than doubles, so A cannot be claimed. B ("control is weak") does not
describe slope 0.906 and control fraction 0.99 at p50, and C is plainly wrong.
The classifier records this as `A_controllability_quality_blocked` rather than
force-fitting a letter.

Reading it by clause: the *conditioning objective* succeeds; the *prior* fails.

### Correction to the shipped classifier

The first run printed plain "A" for `t_start = 0.90` because the implemented
`classify` omitted the quality clause that the specified rule contains. The
clause is now implemented and tested; `decision_corrected.json` holds the
corrected labels. The raw arm data in `natural_support_controllability.json` is
unchanged — only the label was wrong, and the original output is preserved in
that file's `decision` block.

## 9. Was C1 OOD targets or lack of controllability?

**Out-of-distribution target coordinates**, with high confidence.

Inside support the model realises 89–99% of requests within ~1σ. C1 requested
2.5σ–25σ. The saturation ceiling C1 measured (~8 units) sits at the boundary of
natural support, exactly where in-support control begins to degrade (control
fraction 0.49 at p99, a +12.3 unit request). Same curve, sampled far outside its
usable range.

The secondary cause is parameterization: `alpha = alpha_hat * 88.76` scales to
activation norm, which has no fixed relationship to a direction's coordinate
spread (2.9–6.0 across directions here). One `alpha_hat` means very different
things for different directions.

## 10. Recommended next step

**No retraining.** Per the spec's category-A branch, and because the conditioning
objective is not what failed.

1. **Reparameterize steering strength in natural-coordinate units.** Replace
   `alpha_hat * 88.76` with a per-direction z-score or quantile request:
   `c_target = quantile_v(q)` or `c_target = c_nat + z * std_v`. This is a
   config and analysis change, not a training change, and it makes one strength
   setting mean the same thing across directions.

2. **Attack reconstruction fidelity at high flow time, not the conditioning
   objective.** Control needs `t_start ≈ 0.90`; the prior is unusable there
   (+4.08 nats, cosine 0.70). Any path forward has to make high-`t` reconstruction
   cheap in ΔLM. Whether that is capacity, training distribution, or a better
   integrator is an open question this diagnostic does not answer.

3. **Do not run DEV Phase B** until a configuration exists with both slope
   materially above zero and ΔLM within a few percent of clean. No point on the
   current grid satisfies both.

### Explicitly not recommended

The earlier suggestion in `results/conditional_c1_precheck_v1/FINDINGS.md` — train
with `c_target = <h,v> + delta` against a clean target `h` — is **withdrawn**. It
is internally inconsistent supervision: the target activation does not satisfy
the supplied condition. It has been struck from that file.

## Limitations

* Substitution at all non-BOS positions of validation sequences, not
  autoregressive generation. Generation compounds errors differently.
* Training-only pool directions, not DEV SAE directions. Transfer assumed.
* The unconditional wide60m control (control 4, marked optional) was not run;
  the `self` arm provides the same-model regression-to-natural baseline.
* nfe=3 not run; nfe=1 only.
* 64 sequences, 32 directions. Bootstrap resamples sequences, not rows.
