# Constraint-Preserving Tangent Flow — Design and Protocol

## Status

**BRANCH CLOSED — 2026-08-16. The predefined stop condition fired.**

T0 complete, T1 **PASS**, T2 **FAIL**. Read
`docs/BRANCH_CLOSURE_CONSTRAINT_PRESERVING_TANGENT_FLOW.md` first; it is the
authoritative record of the outcome. This document is retained as the frozen
protocol the experiment was run under, and its gates are preserved exactly as
they were specified in advance.

    T1 recovered fraction 0.7730  (threshold 0.25)   -> PASS
    T2 pooled dNLL +0.006184 nats, CI [+0.001631, +0.010788] -> FAIL

Generative naturalization is finished. Nothing further in this branch is
authorized: no larger model, no more NFE, no more data, no other corruption
scheme, no auxiliary losses, no LLM judge, no post-hoc cell search.

Supersedes `docs/PROPOSAL_CONSTRAINT_PRESERVING_TANGENT_FLOW.md` as the active
specification for this branch. That proposal remains the record of what was
proposed and why; where the two differ, §3.1 below states the difference and its
reason.

---

# 1. What is closed, and what is new

## Closed: generic and conditional isotropic priors

`docs/BRANCH_CLOSURE_GENERIC_FLOW_NATURALIZATION.md` is final and is not
reopened, revised, or reinterpreted here. Its result stands as written:

> The learned generic/conditional activation prior does not provide useful
> orthogonal naturalization of steering interventions.

Its durable positive findings also stand: the prior does learn genuine natural
activation reconstruction; conditional coordinate control works in-distribution
(slope 0.906); the earlier C1 failure was an out-of-distribution coordinate
problem; and **hard clamp is a very strong baseline** at ~+0.003 to +0.054 nats
with exact coordinate satisfaction.

No number, arm label, config, or result directory from that branch is modified
by this one. This branch shares none of its trained artifacts.

## New: the corruption-geometry mismatch hypothesis

The closed branch trained on isotropic corruption

    x_t = (1 - t) x0 + t eps

and only at *inference* asked the model to correct an activation while a
semantic coordinate was held fixed. In its decisive projected arm, every `x_t`
the model saw lay exactly on a constraint hyperplane, while every `x_t` it saw in
training did not. Projected inference was therefore run on a state distribution
the model had never been trained on.

> **The only remaining hypothesis in the generative-naturalization family is that
> the previous priors failed because of train/inference corruption-geometry
> mismatch, not because orthogonal naturalization is impossible.**

This branch removes the mismatch by training directly on the tangent geometry
used at inference. This is a **new objective and training distribution**, not
another inference trick applied to a closed checkpoint.

It is not evidence that the new method will work. It is the cleanest remaining
hypothesis, it is cheap, and it is falsifiable.

---

# 2. Mathematical core

All states are in standardized activation space,
`x = (h - mu) / (std + eps)`, using the existing normalizer.

For a clean standardized activation `x0` and a canonical standardized unit
direction `v` with coordinate

    c = <x0, v>

decompose

    x0_par  = c v
    x0_perp = x0 - c v

sample `eps ~ N(0, I)` and remove its parallel component

    eps_perp = eps - <eps, v> v

The **tangent flow path** is

    x_t = c v + (1 - t) x0_perp + t eps_perp
        = (1 - t) x0 + t (eps_perp + c v)

which satisfies, for every `t`:

    <x_t, v> = (1 - t) c + t c = c

The **velocity target** is

    u* = eps_perp - x0_perp

which satisfies `<u*, v> = 0`.

Implemented in `interp.tangent_flow.tangent_flow_states`, directly from these
equations rather than by routing through the isotropic `sample_flow_batch`. It
is cross-checked numerically against the independently derived
`interp.constrained_flow.constrained_partial_noise`.

## 2.1 The raw-space constraint

The raw constraint `<h, v> = c_target` is carried into standardized space by the
existing exact hyperplane transform (`standardized_hyperplane`), unchanged:

    q = scale * v ; v_x = q / ||q|| ; c_x = (c_target - <mu, v>) / ||q||

sign-canonicalized so `(v, c)` and `(-v, -c)` give one representation. No new
transform math was written for this branch.

---

# 3. Model and tangency enforcement

The architecture is the **existing** `ConditionalFlowMatcher`:
`f_theta(x_t, t, v_x, c_x)`, with the existing direction-coordinate FiLM
condition encoder. No new modules. The tangent branch changes the corruption
geometry, not the network.

* T1 architecture: the narrow ~16M class
  (`configs/flow_core_conditional_narrow16m_v1.yaml`).
* The ~60M variant (`configs/flow_core_conditional_60m_v1.yaml`) is instantiable
  with **no code change** if T1 and T2 ever justify a confirmation run.

## 3.1 Tangency is enforced analytically — APPROVED

**The override of proposal §3.1 was approved by the human on 2026-08-16.**

The network predicts an unconstrained vector. The velocity actually used for the
**training loss**, for **Euler integration**, and for **reconstruction** is

    u_tangent = u - <u, v> v

The raw parallel component is recorded as a diagnostic
(`val_raw_parallel_velocity_mean` in training,
`raw_parallel_velocity_norm_mean` at inference) and never moves the activation.

**This differs from the proposal.** `PROPOSAL_CONSTRAINT_PRESERVING_TANGENT_FLOW.md`
§3.1 recommended training *without* analytic projection so that residual tangency
error would be diagnostic. The task specification overrode that, and the override
is the right call for this branch's purpose: the decisive question is whether
orthogonal correction helps at a *fixed* coordinate, and a negative result must
not be attributable to coordinate leakage along `v`. The diagnostic value is
retained by recording the raw parallel component rather than by letting it act.

The semantic-coordinate invariant is **part of the method**, not something the
network is required to learn. `output_projection: true` is the primary setting
for both T1 and T2.

The learn-it variant remains supported in code without changes
(`flow_objective.output_projection: false`) as a **possible later ablation
only**. It must **not** be run as part of primary T1 or T2 without separate
human authorization. Using it makes a different experiment with a different
config fingerprint and a different checkpoint identity, so the two can never be
confused in the record.

---

# 4. Data and protected-data policy

* Directions come **only** from `TrainingDirectionPool`
  (`data/direction_pools/training_only_rank256_v1.pt`, rank floor 256).
* No API in this branch accepts an arbitrary DEV or held-out feature ID.
* `configs/protected/` is never read.
* No DEV steering vector influences training, checkpoint selection, T1, or T2.
* No LLM judge is used anywhere in this branch.

Checkpoint selection is `val_flow_mse` on the run's own held-back document split,
computed on the tangent objective with training-only directions. Steering
performance of any kind is barred from checkpoint selection.

---

# 5. Staged experiment

## T0 — mathematical and synthetic validation (**complete**)

CPU only. No real data, no GPU. See §8 for what was verified.

**Gate: passed.** All geometry invariants hold, the objective overfits a fixed
batch to the floor, and a tiny tangent model measurably reconstructs the
orthogonal state on a structured synthetic distribution.

## T1 — cheap real-data tangent reconstruction (**prepared, not run**)

Train the ~16M conditional architecture on the tangent objective:
`configs/flow_train_tangent_narrow16m_fw32m_v1.yaml`, FineWeb32M activations,
training-only pool.

Then run `scripts/eval_tangent_reconstruction.py`: tangent-corrupt frozen
validation activations at each activation's own natural coordinate, reconstruct,
and compare against

* the **corrupted control** — by construction the exact state the reconstruction
  arm integrates from, not an independently drawn corruption;
* the **clean identity**;
* optionally the closed branch's isotropic conditional prior, run through the
  identical tangent inference path (rejected automatically if its width or
  standardization differ, since that would not be a matched comparison).

Metrics: ΔLM, recovered damage and fraction, relative L2, cosine, tangent
(orthogonal) reconstruction error, exact coordinate drift, NFE {1, 3, 5},
validation tangent-flow MSE.

### T1 materiality threshold — FROZEN

    recovered_fraction >= 0.25

**Approved and frozen by the human on 2026-08-16, before any real T1 result
existed.** Encoded as `interp.tangent_eval.T1_MIN_RECOVERED_FRACTION` and pinned
by a test.

`0.25` is a **pragmatic preregistered convention, not a theoretically derived
constant.** There is no derivation behind it and none is claimed. It exists so
the pass/fail decision is fixed before the number it judges.

**It must not be changed after a T1 result is observed.** Doing so would convert
a preregistered gate into a post-hoc one and would invalidate the T1
interpretation. A different threshold requires a new protocol version with a new
experiment id, decided before its results are seen.

### T1 primary cell — FROZEN

    T1 primary = t_start 0.50, NFE 1

Named explicitly as `primary_t_start` / `primary_nfe` on
`TangentReconstructionSpec`, **never** taken from a tuple position. Reordering
the diagnostic grid cannot move the scientific decision, and a test asserts
this. Every other `(t_start, NFE)` cell is diagnostic.

**Gate (`interp.tangent_eval.t1_verdict`):** at the primary cell, recovered
fraction ≥ 0.25 with the direction-clustered bootstrap CI against the corrupted
control excluding zero and negative.

A run that is not eligible for a formal verdict — an unverified checkpoint, or
`--overwrite-debug-mode` — returns `DIAGNOSTIC_ONLY` rather than PASS/FAIL, and
its receipt cannot authorize T2.

* **T1 PASS** → T2 is worth running.
* **T1 FAIL** → stop and diagnose implementation or training. Do **not** proceed
  to any steering-like evaluation. A model that cannot solve the task it was
  trained for will not solve the harder downstream one.

## T2 — natural-support hard-clamp naturalization (**prepared, not run**)

`scripts/tangent_naturalization.py`. Reuses the frozen `natural_support_v1` plan
verbatim — same directions, sequences, target quantiles `{p50, p75, p90, p95,
p99}`, and seeds — so the hard-clamp baseline is directly comparable to
`results/constrained_naturalization_v1/`.

Two arms, at the identical semantic coordinate:

| Arm | Intervention | NFE |
|---|---|---|
| A | `h_clamp = h + (c_target - <h, v>) v` | 0 |
| B | the same clamp, then the tangent-trained flow | 1 or 3 |

Primary statistic:

    delta_NLL = NLL_tangent_flow - NLL_clamp        (negative = useful)

paired per validation sequence, bootstrapped over sequences.

Supporting: coordinate error, orthogonal correction norm, relative L2, cosine,
UP/DOWN split, target quantile, per-direction signs, deterministic bootstrap,
LOVO.

Primary inference grid, deliberately cheap: `t_start ∈ {0.10, 0.25, 0.50}`,
`NFE ∈ {1, 3}`. Widening it is a protocol change, not a tuning decision.

**Attenuation is closed off by construction and by gate.** The velocity is
analytically tangent, so arm B cannot move the coordinate; and
`assert_coordinate_match` refuses to report any cell whose two arms differ in
realised coordinate by more than 1e-3. An arm that "wins" by shrinking the
coordinate is not a result — that is exactly what the closed branch produced.

### T2 experiment-level decision rule — FROZEN

There is **exactly one** T2 decision, and it is **not** "any grid cell passes".
With 5 quantiles x 3 t_start x 2 NFE = 30 cells, some will look favourable by
chance, so no cell-level result is an experiment result.

    T2 primary = t_start 0.10, NFE 1, pooled across all five target quantiles

One frozen operating point, one pooled statistic, no multiplicity to correct.
`t_start 0.10 / NFE 1` is simultaneously the cheapest point in the grid and the
one most likely to help — the method must beat a clamp that already costs only
+0.003 to +0.054 nats, and larger `t_start` buys correction by destroying more
of the activation. Choosing the cheap point is the cheap-prior motivation of the
programme, not a guess about where the effect will be largest.

**Gate (`interp.tangent_eval.t2_experiment_verdict`):** on the pooled primary
cell, the direction-clustered paired ΔNLL CI excludes zero and is negative,
>80% of directions negative, LOVO maximum still negative, and the two arms'
realised coordinates agree to within 1e-3.

Per-cell entries are written to `t2_cell_diagnostics` via `t2_cell_report`,
which deliberately emits no `verdict` key at all.

### Formal bootstrap unit — FROZEN

    direction_cluster_then_sequence

Several validation sequences share each direction, so sequences are not
independent. Every formal gate uses a two-stage cluster bootstrap: resample
directions with replacement, then resample each drawn direction's sequences with
replacement. 2000 resamples, seed 20260906 (T2) / 20260910 (T1), 95%.

A sequence-level interval is still recorded beside it for continuity with
historical results, explicitly labelled non-canonical. **The two are not
comparable and must never be presented as if they were.**

---

# 6. Hard scientific stop rule

> If the tangent-trained flow **clearly solves T1** but **still does not improve
> hard-clamp NLL at a fixed coordinate in T2**, generative naturalization stops
> entirely.

Interpretation in that case:

> Even learning the correct constrained corruption geometry does not produce
> useful steering repair.

Do **not** respond with:

* more parameters (>60M, or 60M as a "rescue");
* more NFE;
* more generic training data;
* another inference projection trick;
* an LLM judge.

That combination of results is a clean, informative, publishable negative for the
whole generative-naturalization family and must be reported as one. The rule is
encoded in `interp.tangent_eval.STOP_RULE` and copied into every result file this
branch writes, so it cannot be lost between sessions.

The honest framing to keep in view: hard clamp already costs only +0.003 to
+0.054 nats. The remaining headroom is small, and this method is competing
against an already very cheap baseline.

---

# 7. Implementation map

| Concern | Location |
|---|---|
| tangent path, velocity target, projection | `src/interp/tangent_flow.py` |
| training batch sampler | `sample_tangent_flow_batch` |
| inference (clamp → tangent SDEdit) | `clamp_then_tangent_flow` |
| trainer integration | `src/interp/train_flow.py` (`FlowObjectiveSpec`) |
| T1/T2 analysis, gates, verdicts | `src/interp/tangent_eval.py` |
| ~16M conditional architecture | `configs/flow_core_conditional_narrow16m_v1.yaml` |
| prepared T1 training config | `configs/flow_train_tangent_narrow16m_fw32m_v1.yaml` |
| T1 evaluator | `scripts/eval_tangent_reconstruction.py` |
| T2 evaluator | `scripts/tangent_naturalization.py` |

## Compatibility guarantees

* The unconditional and isotropic-conditional trainers are byte-identical in
  code path, RNG consumption, and config fingerprint. `flow_objective` is
  omitted from the fingerprint when absent, exactly as `conditioning` is.
* Checkpoints written before this branch carry no objective key and are read as
  `isotropic` — which is what they are.
* A tangent checkpoint gets kind `tangent_conditional_flow` and cannot be loaded
  where `isotropic` is required, or vice versa.
* Resume rejects a mismatched objective, output-projection flag, direction pool,
  or normalizer.

## Evaluator provenance gates

None of the following is a claim in prose; each is enforced before an evaluator
computes anything, and its receipt is written into the result file.

| Gate | Mechanism | Failure |
|---|---|---|
| Checkpoint IS the run's selection | `verify_selected_checkpoint` reads the run's `best.json` + `meta.json`: filename, frozen metric `val_flow_mse`, mode `min`, experiment id, config fingerprint, tangent objective | raises; `--allow-unselected-checkpoint` downgrades the run to `DIAGNOSTIC_ONLY` |
| Pool is the training pool | `verify_direction_pool` compares the full canonical identity, digest first, against `checkpoint.direction_pool` | raises |
| Artifacts belong together | `load_validated_evaluation_bundle`: VALID report with SHA re-verified now, split fingerprint, GPT-2 name/resolved name/revision, tokenizer, hook, ctx, width, BOS-dropped, FineWeb repo/config/revision, token-cache SHA, token shape, `n_activations == n_seqs * per_seq` | raises |
| T2 needs a T1 PASS | `verify_t1_pass_receipt`: verdict PASS, formally eligible, same primary cell, same checkpoint SHA, same pool digest, tangent objective | raises; T2 cannot run |
| Results are immutable | `require_fresh_output_dir` | raises unless `--overwrite-debug-mode`, which itself forces `DIAGNOSTIC_ONLY` |

### Isotropic reference is not a control

The optional `--isotropic-checkpoint` arm is recorded as
`diagnostic_unmatched_reference`. Matching activation width does **not** make a
historical prior an objective-only control: capacity, training budget, data and
noise stream all differ. The result file records its parameter count, model
config, pool, and config fingerprint, plus an explicit `interpretation_limit`,
and it is excluded from the formal T1 gate. A causal objective-only comparison
would need a separately trained matched 16M isotropic control — not authorized.

### Seeds are not paired with any historical run

The tangent config uses `noise_seed: 20260816`; the historical FW32M isotropic
arm used `20260812`. Restoring the historical seed would not produce paired
noise: the tangent sampler draws and consumes the generator differently, so
identical seeds give different corruption streams. A distinct seed is kept
precisely so no pairing is implied, and the config lists `noise_seed` under
`uncontrolled_variables`. **No objective-only paired claim may be made against
any historical run.**

## Source provenance — resolved 2026-08-16

This tree is not a Git repository, and **no canonical Git history exists to
restore**: there is no `.git` in the tree or any parent directory, and all 81
`source_revision` values recorded across every historical result and checkpoint
in this project — including the entire closed branch — are `snapshot-sha256:`.
Not one is `git:`.

Decision: **use the existing deterministic snapshot provenance.** Do not
`git init` merely to manufacture a commit history. Beyond being artificial, a
fresh repository would make this branch's provenance *less* comparable to the
runs it must be compared against, all of which are snapshot-hashed.

`source_revision()` falls back to a deterministic SHA-256 over an **explicit
allowlist**: `pyproject.toml`, `uv.lock`, and `src/**/*.py`, `scripts/**/*.py`,
`configs/**/*.yaml`. That is a genuine content hash of the exact source that
ran, not a placeholder. T1 is **not** blocked on Git.

### Provenance never reads protected data

The earlier implementation used a broad recursive `configs/**/*.yaml` glob,
which would have opened and hashed any `.yaml` placed under `configs/protected/`
— a run could have read held-out bytes while computing the hash that accompanies
its own `held_out_accessed: false`.

Directories named in `PROTECTED_DIRECTORY_NAMES` are now **pruned during
traversal**, before any file is opened, with a second filter before the read
loop. Tests prove both that protected content cannot change the revision and
that protected files are never opened (`Path.read_bytes` is instrumented).

Measured when the fix landed: **0 files pruned** from the real tree. No `.yaml`
exists under `configs/protected/` today, so no historical run ever hashed
protected bytes. The leak was a latent vector, not an executed one, and no past
artifact's `held_out_accessed: false` is retroactively false.

### Dirty trees cannot masquerade as clean commits

If a canonical Git repository ever exists:

    git:<commit>                        clean tree
    git:<commit>+dirty:<snapshot>       uncommitted changes

Dirtiness is detected with a pathspec restricted to the same allowlist and
excluding `configs/protected`, so protected filenames are never even listed.

## Checkpoint metadata

Every tangent checkpoint records, under `objective_identity`:

* `flow_objective` — `tangent_constraint_preserving`
* `condition_type` — `direction_coordinate_film`
* `tangent_output_projection` — true/false
* `normalizer` — width, eps, and a SHA-256 digest of mean and std

alongside the existing `direction_pool` digest, `config_fingerprint`,
`dataset_artifact_identity`, `source_revision`, and seeds.

---

# 8. T0 evidence

Verified on CPU, in `tests/test_tangent_flow.py`, `tests/test_tangent_eval.py`,
`tests/test_tangent_scripts.py`, `tests/test_tangent_training.py`, and
`tests/test_tangent_synthetic.py`:

**Geometry.** `<x_t, v> = c` at `t ∈ {0, .01, .1, .25, .5, .9, 1}`;
`<eps_perp, v> = 0`; `<u*, v> = 0`; the projected model velocity is tangent while
the raw one is not; the path matches the independently derived constrained
implementation; endpoints are `x0` and `eps_perp + c v`.

**Integration.** The coordinate is fixed across every Euler step at NFE 1/3/5
**with the numerical safeguard switched off**, so tangency alone — not the
safeguard — preserves the coordinate. With the safeguard on, pre-projection
drift stays below 1e-4 and the result is unchanged; projections are never counted
as network evaluations.

**Raw/standardized equivalence.** The raw hard-clamp coordinate equals the
standardized hyperplane representation to tolerance.

**Sign invariance.** `(v, c)` and `(-v, -c)` give identical geometry and matched
trajectories.

**Batch semantics.** Per-row directions give per-row constraints.

**`t_start = 0`.** Exact hard clamp, zero network evaluations, zero orthogonal
correction.

**Checkpoints.** Tangent checkpoints round-trip; loading one as an incompatible
objective raises; unconditional models cannot carry the tangent objective; resume
reproduces an uninterrupted run and rejects mismatched objective state.

**Hook plumbing.** Both evaluators' substitution hooks give identical results
under any hook batching, and the T2 clamp arm is the exact clamp.

**Synthetic overfit.** A tiny tangent model drives the constrained loss on a
fixed batch to below 2% of the zero-predictor MSE (the overfit floor is zero).
On a structured rank-2 distribution it recovers ~30% of the orthogonal damage at
`t_start = 0.5, NFE 1` and ~45% at `t_start = 0.25` — genuine orthogonal
reconstruction, not coordinate copying, which is free here by construction.

**Matched-objective sanity (§9 of the task spec).** Same architecture, same
budget, evaluated on tangent-corrupted data: the tangent-trained model fits the
tangent velocity target better (MSE 0.793 vs 0.805, zero predictor 1.774).
Reconstruction quality came out **level** between the two (~0.27 recovered at
`t_start = 0.5` for both). That is recorded as observed. On this toy the two
geometries are nearly equivalent, so this fixture is an implementation check
only and is **not** evidence about T1 in either direction.

---

# 9. What this document does not authorize

* no training run;
* no GPU launch;
* no DEV or held-out evaluation;
* no LLM-judge spend;
* no change to the closed branch's results or labels.
