# C1 Pre-check: the conditional prior does not realise off-distribution coordinates

Diagnostic: `conditional_c1_precheck_v1`
Checkpoint: `conditional_flow_60m_fw32m_v1` / `best_step_249500.pt`
Data: frozen `resid7_fw_val_1024k_v1`, 4096 rows, 64 training-only pool directions.
DEV vectors accessed: no. Held-out accessed: no.

## Result

C1 of `docs/PHASE_B_CONDITIONAL_PROTOCOL.md` **fails** under `seed_mode="clean"`,
the arm in which the requested coordinate reaches the model only through the
condition.

Absolute realised displacement along `v` (`r_retain * alpha`, nfe=1):

| alpha_hat | requested | t=0.50 | t=0.75 | t=0.90 |
|-----------|-----------|--------|--------|--------|
| 0.1       |      8.88 |   0.18 |   0.89 |   4.41 |
| 0.3       |     26.63 |   0.38 |   1.78 |   7.86 |
| 0.5       |     44.38 |   0.50 |   2.26 |   8.27 |
| 1.0       |     88.76 |   0.82 |   2.80 |   6.95 |

The realised displacement **saturates at roughly 8 units regardless of what is
requested**, while the request grows from 8.9 to 88.8. Retained fraction
therefore collapses from 0.50 to 0.08 as alpha grows. At `t_start = 0.50`, the
operating point shared with frozen Phase B, there is effectively no steering at
all (0.18 to 0.82 units realised).

## This is not a plumbing failure

Three independent checks say the implementation is sound and the model is the
limitation:

1. `seed_mode="clamp"` does move the coordinate (`r_retain` up to 0.69), so the
   condition, sampler, and geometry paths work.
2. The frozen condition-use diagnostic (`results/condition_use_v1/`)
   independently found the condition is read, mechanically passing at every
   checkpoint, but only weakly and only at high flow time.
3. The saturation is smooth and monotone in `t_start`, matching the
   t-dependence that diagnostic measured (relative swap sensitivity 0.026 at
   t=0.5 rising to 0.219 at t=0.9).

## Mechanism

`configs/flow_train_conditional_60m_v1.yaml` sets

    conditioning.condition_source: natural_coordinate

so every condition seen in training was the *natural* coordinate `<h, v>` of a
real activation. The model never saw a requested coordinate displaced off the
data distribution. At inference, `c_nat + 88.76` is far outside the conditioning
distribution it was fit on, and the model falls back to reconstructing the
natural coordinate. The learned conditioning is a weak attractor with a bounded
pull, not a coordinate controller.

This is the training-distribution mismatch that
`docs/PHASE_B_FLOW_STEERING_PROTOCOL.md` section 21 anticipated as the
"structured corruption mismatch" branch, arriving through the conditioning
channel rather than the corruption channel.

## Consequence for Phase B

Per the stop rule in `PHASE_B_CONDITIONAL_PROTOCOL.md` section 13, C1 failing
makes C2 through C4 uninterpretable. **The DEV sweep should not run on this
checkpoint.** Spending it would produce a null that reflects the conditioning
distribution rather than the scientific question.

`seed_mode="clamp"` remains runnable, but its `r_retain < 1` throughout means it
behaves as an attenuator of an additive steer. That is the explanation the
research program already rejected as sufficient on the old denoiser branch, so
it should not be promoted to a headline method without the matched-projection
controls in section 12.

## Limitations

* Static single-position probe on validation activations, not autoregressive
  generation; errors may compound differently under generation.
* Training-only pool directions, not DEV SAE directions. The model saw neither,
  but transfer is assumed, not demonstrated.
* `r_retain` means different things across seed modes: creation of displacement
  under `clean`, survival of an existing displacement under `clamp`. They are
  not directly comparable.

## WITHDRAWN recommendation

This section previously proposed retraining with condition augmentation,
`c = <h,v> + delta` against a clean target `h`. **That recommendation is
withdrawn.** It is internally inconsistent supervision: the target activation
does not satisfy the supplied condition, so the model would be trained to
predict an activation that contradicts its own conditioning input. Do not
implement it.

## Superseded interpretation

The natural-support controllability diagnostic
(`results/natural_support_v1/REPORT.md`) shows the failure documented above is
**extrapolation**, not absence of controllability. Inside the natural coordinate
distribution the model tracks requested coordinates with calibration slope 0.906
at `t_start = 0.90`. The requests in this pre-check were 2.5σ to 25σ outside
natural support, because `alpha_hat` scales to activation norm (88.76) rather
than to coordinate spread (~3.6).

The measurements in this file stand. The mechanism stated above — that the model
regresses to the natural coordinate — is correct but incomplete: it does so only
for targets outside the support it was trained on.
