# C6 — covariance controls for the Experiment C curvature result

**Status: FROZEN 2026-08-20, before any C6 number was computed.**

Class: `post_hoc covariance controls`. `preregistered: false`. This does not
rewrite any frozen C artifact, trains nothing, and touches no held-out data.

---

## 0. Why this exists

Experiment D found that `cos(d_k, v)` — the alignment of the local direction of
natural motion with a fixed direction — is dominated by how much variance that
direction carries. A validated causal direction beat random unit axes by a wide
margin, and principal components with no causal validation beat it in turn. So
comparing a direction against ordinary random unit axes is not a valid test of
tangentness or of intervention quality.

The same confound reaches into C. For a multivariate Gaussian,

    E[h | v'h = c] = mu + (Sigma v / v'Sigma v) (c - v'mu)

so the natural *linear* direction of the conditional trajectory is not `v` at
all — it is `Sigma v`. A residual stream is strongly anisotropic, so a large part
of what C measured could be second-order covariance geometry rather than any
nonlinearity.

C6 asks one question: **how much of the C result survives a control for
covariance geometry and finite-sample conditioning?**

---

## 1. Population, frozen from C

Identical to `scripts/curvature_diagnostic.py`, so nothing about the measured
object changes:

* artifact `resid7_fw_val_1024k_v1`;
* rows: `CURVATURE_SPEC.n_rows` = 262144, drawn with `row_seed` = 20260914;
* directions: the same 32 drawn with `direction_seed` = 20260913;
* sequence unit: `row // 127`, the resampling and splitting unit throughout;
* binning: `CURVATURE_SPEC` cuts p10/p25/p50/p75/p90, six bins, five secants,
  `min_bin_rows` = 256.

New seeds introduced by C6, fixed here:

| purpose | seed |
|---|---|
| train/test sequence split (C6.2) | 20260920 |
| random candidate pool (C6.4) | 20260921 |
| Gaussian surrogate sampling (C6.3) | 20260922 |

---

## 2. C6.1 — the covariance-predicted linear direction

Estimate the empirical covariance `Sigma` on the C rows. `d_model` is 768, so a
dense 768x768 estimate is used directly; no approximation is needed.

For each concept direction `v`:

    b_v = Sigma v / (v' Sigma v)          t_v = Sigma v / ||Sigma v||

Report `cos(d_k, t_v)` per quantile rung and pooled, beside the existing
`cos(d_k, v)`.

**Descriptive only.** A high `cos(d_k, Sigma v)` is *not* evidence of causalness
or of anything else about intervention. The only question is how well the
conditional trajectory follows the linear prediction covariance geometry makes.

---

## 3. C6.2 — residualize the best linear conditional model (the main test)

Coefficients are fit on one half and evaluated on the other. Fitting and
evaluating on the same rows would let the linear model absorb noise and would
understate any residual structure.

Sequences are split into halves with seed 20260920. On the TRAIN half, for each
`v`, fit the multivariate linear conditional model

    h = a_v + b_v c,    c = v'h

by least squares. `c` is standardized using TRAIN-half statistics for numerical
conditioning; the standardization is recorded.

On the TEST half compute residuals `r_i = h_i - (a_v + b_v c_i)`, then run the
**same frozen binning** on the same coordinate `c` (which residualization does
not change) to obtain residual conditional means `rho_k`.

Reported:

* `R_k = ||rho_k||`, and its size relative to `||mu_{k+1} - mu_k||`;
* the residual secants `e_k = rho_{k+1} - rho_k`, **only** where the residual
  vectors are large enough for an angle to be numerically meaningful; a cosine
  between two near-zero vectors is noise and is reported as unusable rather than
  as a number.

**The primary scalar test** avoids that instability entirely. On the TEST half,
compare held-out prediction error of the linear model against a fixed
low-capacity quadratic:

    h ≈ a + b1 c + b2 c^2

    Delta_MSE = MSE_linear - MSE_quadratic

positive means the quadratic predicts better. Bootstrap over TEST-half
sequences. Degree 2 only; no higher degree without a separate reason.

---

## 4. C6.3 — covariance-matched Gaussian surrogate

Sample `h_synth ~ N(mu_hat, Sigma_hat)` with the empirical mean and covariance of
the same rows, the same row count and dimensionality, by eigendecomposition of
`Sigma_hat`. **Not** an isotropic Gaussian. The reproduction accuracy of the
covariance spectrum is recorded.

Run the *exact* C pipeline on the surrogate for every real concept direction.
The surrogate's population conditional mean is linear by construction, so any
curvature it shows is produced by finite-sample estimation and by the pipeline
itself.

Primary comparison: real `shortfall_below_ceiling` minus surrogate
`shortfall_below_ceiling`, with a direction-clustered bootstrap interval.

**Known asymmetry, stated in advance:** surrogate rows are drawn independently,
while real rows within a sequence are correlated. The surrogate therefore has
more effective independence at the same row count. This is why the *shortfall*
below the split-half ceiling is primary rather than the raw secant cosine: the
ceiling absorbs the noise level, so the two sides stay comparable. Raw cosines
are reported too, with this caveat attached.

---

## 5. C6.4 — covariance-matched random directions

The old unmatched random-axis control is kept as a historical control and is no
longer the primary null.

Candidate pool: 20000 random unit directions, seed 20260921. For each candidate
`q` compute

    s_q^2 = q' Sigma q            a_q = q' Sigma q / ||Sigma q||

`s^2` is the projected variance; `a` locates the direction relative to the
covariance spectrum. Both are standardized across the candidate pool (`log s^2`
and `a`, z-scored with pool mean and sd), and each concept direction is matched
to its **nearest candidate in that two-dimensional standardized space, without
replacement**, giving 32 distinct matched nulls.

Achieved balance is reported: per-variable mean and max absolute difference,
before and after matching. The rule is fixed here and is not adjusted after
seeing any result.

The matched nulls run the exact C pipeline. Primary comparison:

    concept curvature - covariance-matched random curvature

with a clustered bootstrap interval. `cos(d_k, v)` is **not** the primary
comparison here.

---

## 6. C6.5 — the rising profile

C reported an alignment profile that rises toward the extreme quantiles, and D
reproduced the same shape on a strongly causal direction. C6 asks whether it
survives covariance control.

The existing tail/centre definition is reused unchanged (secant 4 minus secant
2). Reported:

    Delta_profile = (tail - centre)_concept - (tail - centre)_matched_null

with a bootstrap interval. If it is indistinguishable from zero, the rising
profile is not concept-specific geometry.

---

## 7. C6.6 — PCA sanity control (secondary)

Run the C statistics on PC1..PC8 of the same activations: projected variance,
consecutive-secant curvature, `cos(d_k, PC_j)`, reliability. These are chosen by
variance alone. The purpose is to show how the pipeline behaves on directions
selected for variance and nothing else, not to compare their causalness.

---

## 8. Verdict, fixed before the result

Exactly one:

* **`CURVATURE_BEYOND_COVARIANCE`** — real curvature exceeds the
  covariance-matched Gaussian *and* the covariance-matched random directions,
  the held-out quadratic improves on the linear model, and residual conditional
  means keep systematic structure. Then C strengthens to: *conditional means of
  natural activations vary nonlinearly with the chosen SAE steering coordinate,
  beyond what finite-sample noise and second-order covariance geometry explain.*
* **`CURVATURE_NOT_CONCEPT_SPECIFIC_AFTER_MATCHING`** — the concept-versus-random
  excess disappears under matching, but the conditional mean is still nonlinear.
  Then: *natural conditional trajectories are nonlinear, but the apparent excess
  curvature of SAE directions relative to ordinary random axes is substantially
  explained by covariance anisotropy.*
* **`CURVATURE_EXPLAINED_BY_COVARIANCE`** — most of the curvature disappears
  under the linear and covariance controls. Then C is weakened honestly: *the
  original curvature statistic was largely explained by anisotropic covariance
  geometry and finite-sample conditioning effects.* No attempt is made to rescue
  the original wording.

## 9. Claims that remain prohibited regardless of outcome

D already showed these do not follow, and C6 cannot restore them:

* curvature causes steering failure;
* `v` is or is not causal;
* `v` is or is not a valid intervention direction;
* SAE features are correlational;
* tangent alignment predicts steerability.

`cos(d_k, v)` may be shown as a descriptive statistic only, and must carry the
note that it is confounded by activation covariance.

## 10. Stop

No new experiment after C6.
