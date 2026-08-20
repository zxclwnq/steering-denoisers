# Proposal: structured-corruption activation prior

**Status: PROPOSED, NOT FROZEN, NOT APPROVED, NOT LAUNCHED.**

This document exists so that the Gaussian-flow branch can be closed with a
successor design on the table. Nothing here may be trained until the human
freezes it. Several open questions in §9 must be answered first.

---

## 1. Hypothesis

The Gaussian-flow branch produced a sharp dissociation:

* capacity materially improves reconstruction of **isotropic Gaussian**
  corruption (Phase A: reconstructed ΔLM `0.3041 → 0.2222` at fixed corpus, data,
  and budget — a 26.9% improvement with a resolved CI);
* the same capacity increase moves **steering** correction by a statistically
  unresolved ~2% (Phase B: matched-projection ΔNLL `1.1667 → 1.1460`).

The proposed explanation:

> The prior is trained to invert isotropic noise, but inference asks it to invert
> a rank-1, semantically meaningful, large-norm displacement `+alpha v`. That
> displacement is not in the training corruption distribution, so a better prior
> over natural activations buys nothing on it.

Two mechanistic observations support this over the alternatives:

* the correction is dominated by its **orthogonal** component (at `t=0.50, NFE=1`
  the wide prior applies `‖c_⊥‖ = 29.3` against `‖c_∥‖ = 19.9`) — the model pushes
  the activation back toward the data manifold in directions unrelated to `v`;
* retained fraction falls monotonically with `t_start` (`0.92 → 0.71 → 0.43`),
  which is the signature of attenuation, not of targeted correction.

The hypothesis is **falsifiable**: if a prior trained on structured corruption
still loses at matched realised projection, corruption mismatch is not the
limiting factor either, and the whole "cheap activation prior as steering
corrector" family should be closed rather than scaled further.

---

## 2. The single variable

Everything is held identical to the current 60M wide prior except the corruption
distribution. That is what makes the existing `wide60m_fw32m` run a directly
comparable control rather than a loose reference:

| held fixed | value |
|---|---|
| architecture | `configs/flow_core_wide_60m_v1.yaml`, 60,407,808 parameters |
| activation data | `resid7_fw_train_32000k_v1` (FineWeb, 32M unique tokens) |
| standardization | the same train-split float64 statistics, SHA `2e3081f1…` |
| optimizer, LR, batch, schedule, seeds | the frozen v2 recipe, 250k steps |
| hook, model, tokenizer, revisions | unchanged |
| Phase-A / Phase-B measurement stack | unchanged |

The one change is what the model is asked to undo.

---

## 3. Corruption distribution

Training draws a mixture. With probability `p_struct` the corruption is
structured, otherwise it is the current isotropic Gaussian. Keeping a Gaussian
component is not decoration: the concept-independent Phase-A diagnostic and the
"is this still a valid activation prior" check both depend on it, and it prevents
the model from collapsing into a device that only erases rank-1 spikes.

In standardized coordinates, for a unit SAE decoder direction `d` from the
training-only pool:

```
structured:  x_c = x_0 + (alpha / sigma) ⊙ d ,  alpha = alpha_hat * 88.76
isotropic:   x_c = (1-t) x_0 + t * epsilon        (the current path)
```

`alpha_hat ~ U[0, 1.2]`. The upper bound sits just above the frozen Phase-B
primary grid maximum of `1.0` and below the stress points `1.5 / 2.0`, so the
training strength distribution covers the evaluation range without being tuned to
it. `88.76` is the frozen activation-norm constant already used by Phase B, so
training and inference strengths are expressed in the same units.

Proposed `p_struct = 0.5`, with `{0.25, 0.5}` as the **only** tuning axis, decided
on concept-independent validation (§6) and never on DEV steering.

---

## 4. What the model predicts

Keep the rectified-flow objective and the velocity parameterization. The model
predicts the displacement back toward the clean state:

```
f_theta(x_c, s)  ≈  x_0 - x_c
```

where `s` is the corruption-strength embedding replacing the current time
embedding (identical sinusoidal module, reinterpreted argument: `s = t` for the
isotropic branch, `s = alpha_hat / 1.2` for the structured branch).

This is deliberately **not** the old one-step Gaussian denoiser. That branch is a
completed negative result and must not be revived under a new name. The
differences are explicit: the corruption family is structured rather than
isotropic, the target is a displacement rather than a direct reconstruction, and
the strength conditioning is exposed to the network. The proposal must carry a
new experiment ID and must never reuse `naive`/denoiser labels.

**Inference becomes cheaper, not more expensive.** At inference the steered
activation `h_s = h + alpha v` is itself a sample from the structured corruption
family, with a direction the model has never seen. There is no partial-noising
step to choose: standardize, evaluate at the known `alpha_hat`, step, denormalize.
`NFE = 1` is the natural setting; `NFE ∈ {1, 2, 3}` is the frozen grid, and
nothing above 3 is permitted.

---

## 5. Training-only direction pool, and the leakage guard

This is the part that most needs human sign-off, because it is where leakage
would enter.

The frozen selection procedure orders SAE features by a deterministic BLAKE2b
priority under `seed = 20260807` and accepts 16 concept directions, 8 DEV and 8
held-out. The DEV entries occupy ranks 0, 9, 17, 19, 22, 24, 27, 32.

**The training-only pool is defined by a rank threshold, so that no evaluation
direction is ever enumerated:**

```
training_only_pool = { features with BLAKE2b priority rank >= R_min }
R_min = 256
```

Two mechanical assertions are required at config-freeze time, and both must be
tests, not prose:

1. `max(rank of any accepted concept direction) < 64` — establishes the margin;
2. `min(rank in training_only_pool) >= 256` — establishes disjointness.

Together these prove the training pool is disjoint from both DEV and held-out
*without the config, the code, or the logs ever containing an evaluation feature
ID*. No held-out direction is loaded, embedded, scored, or measured at any point.

The pool is split further into `train_directions` and `val_directions` so that §6
can measure structured reconstruction on directions unseen during training but
still not DEV.

---

## 6. Concept-independent validation

Model selection must not touch DEV. Three concept-independent signals, all
computed on the frozen FineWeb validation artifact:

1. **isotropic reconstruction** — the existing Phase-A functional ΔLM grid. The
   structured prior must not have destroyed its ability to denoise natural
   corruption; a large regression here means it became a direction-specific
   eraser.
2. **structured reconstruction on `val_directions`** — the same functional ΔLM
   measurement with the corruption drawn along pool directions never trained on.
   This is the generalization signal, and it is concept-independent because
   `val_directions` are not DEV.
3. **validation flow/displacement loss** by strength bin, as now.

Checkpoint selection uses (3) with (1) as a guard, exactly as the current
protocol does. The selection rule must be written down before any result is seen.

---

## 7. Generalization to unseen DEV directions

The scientific claim is *transfer*: correction learned on one family of semantic
directions must work on directions never trained on. The test is the existing
frozen Phase-B DEV design, unchanged — same eight DEV vectors, same prompts,
seeds, alpha grid, baselines, matched epsilon, degeneracy rule, bracketing, and
bootstrap.

Held-out stays untouched. It is spent only once, after this branch produces a
method worth a final evaluation, with explicit human approval.

---

## 8. Baselines and the control that decides the result

Reuse the three frozen baselines verbatim: additive, the old naive denoiser, and
scalar shrinkage `kappa = 0.8`. Add two:

* **the current wide Gaussian prior** (`wide60m_fw32m`) — the direct control for
  the single variable, already generated and analyzed;
* **matched-strength scalar attenuation** — the frozen matched-realised-projection
  comparison, which is what the method must actually beat.

The last one is not optional. A model trained to remove rank-1 spikes has a
trivial degenerate solution: subtract a fixed fraction of the perturbation. That
solution *is* shrinkage. The headline quantity is therefore matched-projection
ΔNLL against additive **and** against shrinkage, with vector signs and
leave-one-vector-out, exactly as now. An equal-nominal-alpha improvement is not
success.

---

## 9. Open questions for the human before freezing

1. Is `R_min = 256` acceptable, and may the two rank assertions read the frozen
   ordering far enough to compute `max(rank of accepted)`?
2. `p_struct` — accept `{0.25, 0.5}` as the only tuning axis, or fix `0.5`?
3. Does the change from a noise-time embedding to a strength embedding count as a
   new method family requiring a new protocol document, or an amendment?
4. Budget: 250k steps at 60M is ~1.5 h on the RTX 4090 per arm. With two
   `p_struct` values plus the Phase-A/Phase-B stack, roughly 5–6 GPU hours.
5. Should the structured prior also be trained at 16M for a capacity replication,
   or is one capacity enough for a first falsification test?

---

## 10. Stopping condition

If the structured prior loses at matched realised projection against both
additive and shrinkage, with vector-consistent signs, the corruption-mismatch
hypothesis is falsified. In that case do not scale, do not deepen, and do not add
a third corruption family. Record the negative result and change method family.
