# Phase-B DEV Rerun with the Concept-Independently Selected Wide Prior

Status: **frozen before generation** — this document records the design and the
provenance chain. Results are appended only after the run completes, in a clearly
separated section.

Experiment ID: `clean_flow_phase_b_dev_wide60m_v1`
Evaluator config: `configs/flow_phase_b_evaluator_wide60m_v1.yaml`
Config SHA256: `83bcc7344cfdded76a24965f6f7d0ae329ab2b58b32a90fd1e250ca910520dbd`

---

## 1. The question

Phase B with the narrow 16.1M prior was negative: at matched realised steering
strength the flow correction was worse than both additive steering and scalar
shrinkage, and its useful effect looked like attenuation.

The 2x2 capacity/data experiment then showed a large, resolved concept-independent
improvement from width: at `t_start=0.50, NFE=1` the reconstructed ΔLM fell from
0.3075 (narrow, 4M) to 0.2222 (wide, 32M), while 8x more unique activation data
changed nothing at either capacity.

This rerun asks exactly one question:

> Does the wide prior's concept-independent Phase-A improvement transfer into a
> better steering correction at matched realised strength?

The scientific question, the protocol, the grid, and the analysis rule are
**not** changed in response to the earlier negative result.

---

## 2. What is substituted, and what is not

Substituted: the flow prior, and nothing else.

Everything below `baselines:` in the evaluator config is byte-identical to
`configs/flow_phase_b_evaluator_v1.yaml` except the output root and the
`release_requires` provenance list. This is enforced by
`tests/test_phase_b_analysis.py::test_wide_release_keeps_the_narrow_dev_design_byte_identical`,
which diffs the two files and asserts the exact set of changed lines.

Unchanged: the eight frozen DEV directions; ten prompts; seeds 0/1/2; 48 new
tokens; temperature 1.0, top-p 0.95, top-k 0; the primary `alpha_hat` grid
`{0, .1, .2, .3, .4, .5, .6, .7, .85, 1.0}` and stress points `{1.5, 2.0}`; the
three rescored baselines (additive, frozen naive denoiser, shrinkage k=0.8); the
256 SAE control features per direction; the metric versions; the matched-noise
identity; the repetition threshold; the bootstrap contract; and the
realised-projection matching contract.

The flow grid stays frozen at `t_start ∈ {0.10, 0.25, 0.50} × NFE ∈ {1, 3, 5}`.
NFE 3 and 5 are retained deliberately. The concept-independent capacity
diagnostic again found NFE 1 best in all four 2x2 arms, but dropping 3/5 here
would destroy comparability with the narrow release, which is the whole point of
this run. NFE 10/20 are not added.

---

## 3. Frozen provenance

### Flow prior

| Item | Value |
|---|---|
| selected arm | `wide60m_fw32m` |
| selection rule | frozen concept-independent scaling selection rule v1 |
| decided by | lower validation flow MSE on `resid7_fw_val_1024k_v1` |
| steering metrics used in selection | **false** |
| Phase-B evidence used in selection | **false** |
| checkpoint | `/workspace/checkpoints/flow_scaling_wide60m_fw32m_v2/best_step_249500.pt` |
| checkpoint step | 249500 |
| checkpoint SHA256 | `68482e6837d9d72e04dbdc06728ffe18f0c1d81e7e5e9ed807bbe26ac36affb1` |
| run metadata SHA256 | `2487ea15b8a37ca4bc7263343d5e70462d9bd79b299d0c4e23f624e9ed53d3da` |
| best-pointer SHA256 | `270feec12eb60a9348f2c6e7a0f0b54be72dc71f5be17ea86a1223e22863e8e9` |
| history entries | 500 |
| parameters | 60,407,808 |
| activation boundary | 768 in, 768 out (internal `d_model` 1536) |

### Selection chain

| Item | SHA256 |
|---|---|
| `configs/flow_scaling_2x2_v2.yaml` | `4e11e4d382188680d28996c2dc0b1484e0721720a6c2a2811a4d0c4a32a1e09c` |
| 2x2 selection report | `4da315b87cb2051c171ea47d828a8311508f3718ea5cfc62b46060a862013099` |
| `wide60m_fw32m` arm report | `8c42e5b8fba9321bb7aef001f58989e406e24f0d59152c60d6d414ede94ac37e` |
| `configs/flow_core_wide_60m_v1.yaml` | `4a0112c05054a944123d46e319edb93e542a98ca926afd87974004e55bde1295` |
| `configs/flow_train_scaling_wide60m_fw32m_v2.yaml` | `1eec7872dc91a60053c387380713786800f06fbaef94e1981286da6aaeea72e2` |

`interp.scaling.validate_selected_flow_prior` walks this chain at every evaluator
invocation and refuses to proceed unless the selection report is complete, names
`wide60m_fw32m`, records `steering_metrics_used=false` and `phase_b_used=false`,
and consumed exactly the arm report whose bytes are declared here. It then hands
the checkpoint to the already-tested `validate_frozen_checkpoint`, which proves
the file is the unique concept-independent minimum of its own 500-entry history.

### Activation statistics

The normalizer mean/std are buffers inside the checkpoint. The runtime loader
additionally binds the loaded payload to `dataset = resid7_fw_train_32000k_v1`,
`experiment_id = flow_scaling_wide60m_fw32m_v2`, and statistics artifact SHA256
`2e3081f14122c3837d080f44c60a9ee7d5f5ed24ab8ccb332c0b8f73a73cda3f`, so the
standardization applied at steering time provably belongs to this prior and not
to the OpenWebText statistics the narrow prior used.

### Frozen baselines (reused text, recomputed metrics)

| Artifact | Raw SHA256 |
|---|---|
| `cmp1_additive_dev.jsonl` | `ba6a164f7cbd428f65679c67d3f101dafbed31262b07abc78c653d9350f80058` |
| `cmp1_naive_dev.jsonl` | `f382d61303018334ae3001c286eb22ae6426228ad061c49b305ae2e2b543a831` |
| `ct_shrink_k080_dev.jsonl` | `b9537df71f7bbf5daa1418fe9f2198516e3390cafbea3444a0f4aa50c410d097` |

Only immutable cell identity and continuation text cross into the evidence path;
every metric is recomputed by the same clean scorer that scores the flow rows.

### Other frozen identities

* GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, hook `blocks.7.hook_resid_pre`.
* SAE `jbloom/GPT2-Small-SAEs-Reformatted` revision `57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9`,
  config SHA `93d39f5e…`, weights SHA `47bfb750…`, both verified byte-for-byte on the worker.
* Bootstrap: seed 20260813, 10,000 resamples, 95%, unit = steering direction, one shared resample matrix.
* Repetition/degeneration threshold: `0.027785714285713504`.
* Comparison release: `clean_flow_phase_b_dev_v1_e965eb211254b0c24d84475782091beb30d11f99fd328ee49f26a6c403d26a4f`.

---

## 4. The analysis layer

As of 2026-08-13 no frozen Phase-B analyzer existed, and the matched-projection
tables produced until then were provisional. This rerun adds one:
`src/interp/phase_b_analysis.py` (`clean_phase_b_analysis_v1`) plus
`scripts/analyze_flow_phase_b.py`.

It recomputes everything from the raw rows and enforces:

* the statistical unit is the steering direction, never the generation row;
* one resample matrix shared by every comparison;
* linear bracketing on realised projection inside `(vector, prompt, seed)`
  families, no extrapolation, no clipping, every bracket and weight written to a
  companion JSONL;
* a comparison point is dropped when the flow row **or** either bracketing
  baseline row exceeds the repetition threshold;
* stress alphas never enter the primary grid;
* all nine arms must share epsilon per continuation cell, and the two releases
  must share the same epsilon digest before they may be compared.

### Reproduction of the narrow release

Running the analyzer on the untouched narrow raw rows reproduces the provisional
numbers **exactly** for: all twelve descriptive rows (NLL, lexicon, target SAE,
repetition), all nine equal-alpha paired NLL means and vector sign splits, all
nine geometry rows, and the matched-projection *unsupported* counts.

It does **not** reproduce the provisional matched-projection supported/degenerate
split (for `flow_t010_nfe1`: 1771/538 here versus 1919/390 there). No variant of
the degeneracy filter was found that yields the provisional numbers. The rule used
here is the literal reading of the frozen contract and is recorded in every
matched-projection block as `degeneracy_rule`. The provisional matched-projection
counts are therefore superseded; the descriptive, equal-alpha, geometry, and NFE
tables stand.

---

## 5. Known confound, stated once and repeated with every claim

The wide prior differs from the narrow prior in **four** ways at once, because
the 2x2 was all-FineWeb by design:

| | narrow (Phase B v1) | wide (this rerun) |
|---|---|---|
| parameters | 16,147,200 | 60,407,808 |
| corpus | OpenWebText | FineWeb `sample-10BT` |
| unique activation tokens | 4,000,119 | 32,000,063 |
| optimizer steps | 100,000 | 250,000 |

So a narrow→wide difference here is **not** a clean capacity effect. The correct
phrasing is "the concept-independently selected wide prior" versus "the original
narrow prior".

The 2x2 does bound the data contribution: 8x more unique tokens moved the
concept-independent primary metric by ~1% with a CI crossing zero at both
capacities, so capacity is the dominant term in the Phase-A improvement. Nothing
in the 2x2 bounds the corpus or budget contribution to a *steering* result.

The clean control is a second Phase-B run on `narrow16m_fw32m` — same corpus,
same budget, same unique data, narrow capacity. It is proposed as the follow-up,
not run here.

---

## 6. Hypotheses, fixed before results

**H-capacity-transfer.** Matched-projection ΔNLL is smaller for the wide prior
than for the narrow prior. Primary cell `t=0.50, NFE=1`; resolved when the paired
95% CI over the eight directions lies entirely below zero.

**H-Pareto.** At least one wide arm achieves ΔNLL < 0 against both additive and
shrinkage at matched realised projection, with a resolved CI.

**H-NFE.** The `NFE=1 ≈ 3 ≈ 5`-or-worse pattern reproduces under steering.

Success is not redefined after seeing results. An equal-nominal-alpha NLL
improvement alone is **not** success: it is what pure attenuation produces.

---

## 7. Outcome interpretation, fixed before results

**A.** Wide materially improves Phase B and reaches or approaches the baselines →
capacity/compute limitation plausibly explains a substantial part of the original
negative result.

**B.** Wide materially improves Phase A but Phase B barely moves → strong evidence
of a train/test corruption mismatch: the prior models natural corruption better,
and that does not transfer to structured `+alpha v` interventions.

**C.** Wide improves Phase B partially but still loses at matched strength →
capacity matters, but this scale is insufficient for a genuine Pareto gain.

---

## 8. LLM semantic judge

Not run automatically. It is justified only if a plausible semantic loophole
survives the cheap metrics — for example materially higher target SAE or lexicon
at worse NLL. In that case the audit is proposed as a separate small frozen
matched design, with its candidate selection rule recorded before any judge
output is observed. The rejected fluency rubric is not used.

---

## 9. Pre-launch verification

The independent pre-launch review normally required before a run of this size was
**waived on 2026-08-14**, and is recorded here as waived rather than as passed. The
checks below were run in its place; each is empirical rather than an assertion.

**Wide checkpoint is the one actually used.** The smoke provenance records
checkpoint SHA `68482e68…` and the full selection chain. Its geometry differs
measurably from the narrow smoke at the identical cell (retained 0.8161 vs
0.8674, parallel correction 1.66 vs 1.29), so a stale narrow prior cannot be
silently in place.

**Standardization statistics belong to the wide prior.** The checkpoint's
normalizer buffers were compared against the `resid7_fw_train_32000k_v1`
statistics artifact: maximum per-dimension disagreement is `3.4e-4 σ`
(mean `8.2e-5 σ`), which is the expected effect of the frozen
`statistics_from: train_split_only` recipe computing over the 95% train rows
rather than all rows. For contrast, applying an unstandardized (unit) normalizer
would misplace dimensions by up to `13 σ`. The runtime additionally binds the
loaded payload to the statistics artifact SHA, the dataset name, and the training
experiment ID.

**Baseline metrics are identical across the two releases.** Per-cell digests over
`(metrics, continuation)` for all three baselines are byte-identical between the
narrow release and this one, computed on two different machines under two
different source revisions:

| baseline | digest (both releases) |
|---|---|
| additive | `53835074fb5f6ac67f5051bfab47ba0e98c87b2d64cef6476e22b8a539104707` |
| naive | `aff70d83359c3d784879643b9d4ee9c9bd1bcc89cad33f5904ae3cb4c800fc6c` |
| shrinkage k=0.8 | `b6e7c7c4331197141ddfa5ae74f938179f916eb112b8b275b6d42d49949a8e51` |

This is the strongest available evidence that the clean scorer is deterministic,
that no legacy metric entered the evidence path, and that the two runs share a
genuine common reference.

**Matched epsilon.** `_position_seed` depends only on namespace, direction name,
exact alpha hex, prompt ID, generation seed, and token position — the prior is
not an input, and epsilon is drawn at the 768-wide activation boundary shared by
both models. The two smokes produced identical epsilon digests. The analyzer
additionally refuses to compare two releases whose per-cell epsilon digest
differs; the narrow release's digest is
`4cdfc25a589d018124dd74c7c06c69aa731df0df075c984306f0a8b900c0278f`.

**Held-out.** No module on the evaluator, analyzer, or scaling path contains any
code that opens, enumerates, or resolves a held-out artifact. The only directions
loaded are the eight DEV entries in the frozen parent protocol.

**Padding and evaluation counts.** The smoke recorded 2 hook calls, 2 flow network
evaluations at NFE 1, and 0 padding positions evaluated. The analyzer fails closed
if any arm reports a nonzero padding count.

**Tests.** 234 tests pass locally and on the worker (1 skipped remotely: the 2x2
selection report is not present in the worker checkout). One pre-existing
tolerance issue was fixed: the wide-model row-permutation invariance test asserted
`atol=1e-6`, but batched float32 matmul reordering gives up to `1.3e-6` on an
eight-thread CPU. The tolerance moved to `1e-5` and a contrast assertion was added
that perturbs one row by `+50` and requires the other rows to stay put — so the
looser tolerance cannot hide real cross-token mixing.

---

## 10. Execution

Smoke first (one direction, one alpha, two prompts, one flow cell, two tokens),
then the full DEV rerun. Held-out is not touched at any point.

```bash
# on the worker, under tmux
/workspace/logs/run_wide_phase_b.sh 2>&1 | tee /workspace/logs/wide_phase_b.log
```

Local artifacts land under `results/phase_b_wide60m_v1/`.
