# Cheap Scaling Experiment: Capacity x Unique FineWeb Data (2x2)

Status: **prepared, smoke-validated, not launched.** The full collection and the
four training runs require explicit human approval.

This document is the map for the next concept-independent experiment. The frozen
scientific content lives in `configs/flow_scaling_2x2_v2.yaml`
(SHA256 `4e11e4d382188680d28996c2dc0b1484e0721720a6c2a2811a4d0c4a32a1e09c`, pinned in
`interp.scaling.APPROVED_SCALING_PROTOCOL_SHA256S`).

**v1 -> v2:** the matched training budget rose from 100,000 to 250,000 steps at
human request. Everything else — arms, capacity, data, validation set, inference
grid, selection rule — is identical, and the change applies to all four arms
equally. Under v1 one arm (`narrow16m_fw4m`, 100k steps) was trained and
evaluated (reconstructed ΔLM 0.318806, val flow MSE 0.998907) before the budget
change; that result is superseded and exploratory, is not a v2 cell, and must not
be compared against v2 arms. `configs/flow_scaling_2x2_v1.yaml` and its four v1
training configs are left untouched as history.

DEV steering, held-out directions, Phase B, and LLM judges are untouched
throughout and are declared `forbidden` in every config added here.

---

## 1. Question

The 16.5M flow prior reconstructs partially corrupted activations well (Phase A)
but produced a clean negative Phase-B steering result, and the prior diagnostic
classified it as *approximately saturated at its current scale*, with capacity
limitation and unique-data limitation unresolved.

This experiment separates the two, while staying cheap:

|                | 4M unique FineWeb | 32M unique FineWeb |
| -------------- | ----------------- | ------------------ |
| 16M narrow     | `narrow16m_fw4m`  | `narrow16m_fw32m`  |
| 60M wide       | `wide60m_fw4m`    | `wide60m_fw32m`    |

Estimands: capacity main effect, unique-data main effect, and their interaction,
all measured concept-independently.

### One deliberate deviation from the task description

The task listed the 4M/narrow cell as "existing". The existing 4M run
(`clean_flow_100k_v1`) is **OpenWebText**, so reusing it would confound corpus
with the data-scale factor in every comparison down the left column. A fresh
FineWeb 4M narrow arm costs about 15 minutes of GPU time and makes the 2x2
internally clean, so all four cells are new and all four use FineWeb.
`clean_flow_100k_v1` is retained as a **corpus reference** against
`narrow16m_fw4m`: same architecture, same recipe, same token count, different
corpus. That is a free extra measurement, not a cell of the design.

---

## 2. FineWeb source

| Field | Value |
| --- | --- |
| Repository | `HuggingFaceFW/fineweb` |
| Config | `sample-10BT` |
| Revision | `9bb295ddab0e05d785b879661af7260fed5140fc` |
| Access | `datasets` streaming, `split="train"`, pinned revision |
| Filtering | none; no concept, no steering metadata, no FineWeb-Edu |

Standard FineWeb rather than FineWeb-Edu, matching the reference GLP data source
(`docs/REFERENCE_NOTES_GLP.md` §14). The pin lives in `interp.data.FINEWEB` and
`configs/fineweb_activations_v1.yaml`; a test asserts the two agree.

### Sampling method

Sampling is by **document index range over the pinned stream**, which is
deterministic and requires no RNG:

| Split | Document range | Purpose |
| --- | --- | --- |
| `val` | `[0, 40000)` | frozen concept-independent validation |
| `train` | `[100000, 20000000)` | flow training only |

* The ranges are disjoint, with a 60,000-document gap, so no validation document
  can drift into training if document acceptance rates change.
* Every training artifact takes the **first N accepted documents** of the same
  stream, so `resid7_fw_train_4000k_v1` is a strict prefix (subset) of
  `resid7_fw_train_32000k_v1`. The data axis is a nested ladder, not two
  unrelated samples.
* One document contributes exactly one 128-token window, so no activation is
  repeated inside a nominal "unique" dataset.
* Documents with fewer than 127 text tokens are **skipped, never padded**.
  Measured acceptance in the smoke: 278 documents consumed, 22 skipped, 256 kept.
* Adding a later 128M artifact needs no redesign: extend the same prefix.

### Tokenization and collection invariants

Unchanged from the validated GPT-2 path: pinned `gpt2` at revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, hook `blocks.7.hook_resid_pre`,
`ctx=128`, exactly one BOS at position 0, BOS activation dropped, no padding
positions, no steering hook, `model.eval()`, float32 compute, float16 storage.

Each collection records: documents consumed, documents skipped short, documents
kept, raw text tokens seen, raw tokens, BOS activations discarded, padding
activations discarded (always 0), valid activations, tokenizer/model revisions,
corpus revision, document range, token-cache SHA256, and the source snapshot.
Token-level provenance is written to a `*.manifest.json` beside the token cache
and copied into the activation metadata.

---

## 3. Activation artifacts

| Artifact | Split | Sequences | Activations | float16 bytes | Split fingerprint |
| --- | --- | ---: | ---: | ---: | --- |
| `resid7_fw_val_1024k_v1` | val | 8,064 | 1,024,128 | 1,573,060,608 | `5725541eaecf437c` |
| `resid7_fw_train_4000k_v1` | train | 31,497 | 4,000,119 | 6,144,182,784 | `c34aec678a328131` |
| `resid7_fw_train_32000k_v1` | train | 251,969 | 32,000,063 | 49,152,096,768 | `02dad171587ee40f` |

Total activation bytes: **56,869,340,160 (56.87 GB)**, plus about 149 MB of token
caches. Worker free space measured at preparation time: **127 GB**. The 4M token
count is deliberately identical to the historical OpenWebText artifact so the
corpus reference is exact.

Storage decisions:

* float16 storage, float32 collection, **float64** statistics accumulation.
* Per-artifact per-dimension mean/std recomputed from that artifact's own train
  split. The 4M statistics are **not** reused for the 32M artifact.
* Whole-file SHA256 of array, metadata, and statistics is kept even at 49 GB: the
  validator already performs a full float64 rescan of the array, so a chunk
  manifest would add a provenance scheme without adding evidence. Hashing cost is
  a few minutes, and it is recorded in the immutable validation report.

### Validation split

`resid7_fw_val_1024k_v1` is collected from the validation document range and is
disjoint from every training artifact. Phase A consumes the **first 256 sorted
sequences of its internal validation split** (`split_seed=20260807`,
`val_fraction=0.05`, 403 internal validation sequences available), which is the
existing frozen Phase-A selection rule reused verbatim. The same rows feed the
shared validation flow-loss diagnostic. The set is identical for all four arms
and does not change between capacity comparisons.

In-run `val_flow_mse` for checkpoint selection still comes from each training
artifact's own held-back 5% document split. That quantity is only used *inside*
an arm; every cross-arm number in the report comes from the frozen FineWeb
validation artifact.

---

## 4. Wide 60M-class model

`configs/flow_core_wide_60m_v1.yaml`:

```yaml
activation_dim: 768
d_model: 1536
d_mlp: 3072
n_blocks: 3
time_dim: 256
time_hidden: 768   # time out_dim 1536 = d_model
```

**Exact parameter count: 60,407,808** — asserted three ways in
`tests/test_scaling_2x2.py`: the constructed module, the analytic
`flow_core.flow_parameter_count`, and the independent
`prior_diagnostic.wide_glp_parameter_count`. Measured on the worker as
60,407,808.

`configs/flow_core_v1.yaml` is untouched and still yields exactly 16,147,200.

The only code change is that `FlowModelConfig` gained an optional
`activation_dim` that decouples the 768-wide activation boundary from the
internal `d_model`. `activation_dim: None` means "same as `d_model`", so existing
narrow checkpoints and configs are bit-for-bit unaffected (a regression test
loads a pre-`activation_dim` checkpoint payload). Objective, time convention,
standardization, sampler, and block structure are unchanged. This is a
capacity-only change, not a new method.

Cheapness bound: 60,407,808 parameters is the frozen maximum. No 6/12/24-block
scaling, nothing above 100M, no attention, no ensembles; the protocol loader
rejects an arm that exceeds the bound.

---

## 5. Training recipe

**The recipe is unchanged from `clean_flow_100k_v1` in all four arms**: AdamW,
`lr=3e-4`, weight decay `0.01`, cosine over a 100,000-step horizon, 500 warmup
steps, batch 1024, float32, gradient clipping 1.0, `seed=0`,
`noise_seed=20260812`, evaluation every 500 steps.

Recommendation and rationale: the reference GLP recipe (`lr≈5e-5`, batch 4096,
bf16, ~1 pass over ~1B activations) is a different optimization regime. Moving
toward it *now* would change capacity, unique data, and optimization in the same
experiment, and would also break comparability with the frozen narrow baseline.
The first controlled comparison therefore keeps the validated recipe. If the 2x2
shows a capacity effect, a separate single-variable optimizer experiment is the
right follow-up. A test asserts all four arms share one recipe tuple.

### Compute matching

Matched quantity: **total activation presentations = 256,000,000**
(250,000 steps x batch 1024) with identical schedule horizon and
identical optimizer step count. Only the number of dataset passes differs, which
is exactly the unique-data factor under test.

| Arm | Unique activations | Presentations | Dataset passes | Optimizer steps | Expected runtime |
| --- | ---: | ---: | ---: | ---: | --- |
| `narrow16m_fw4m` | 4,000,119 | 256,000,000 | 67.37 | 250,000 | ~33 min |
| `narrow16m_fw32m` | 32,000,063 | 256,000,000 | 8.42 | 250,000 | ~33 min |
| `wide60m_fw4m` | 4,000,119 | 256,000,000 | 67.37 | 250,000 | 85–120 min |
| `wide60m_fw32m` | 32,000,063 | 256,000,000 | 8.42 | 250,000 | 85–120 min |

Wall-clock is deliberately **not** matched: the wide model costs about 3.7x per
step, and equalizing wall-clock would silently shorten wide training. Narrow
runtime uses the measured 114 steps/s of the historical 100k run. Wide runtime
brackets the two measurements from the smoke: 53.3 steps/s on a resident batch
and 28.2 steps/s in the tiny end-to-end run including memmap fetch and frequent
evaluation.

Trade-off, stated explicitly: matched presentations means the wide arms receive
more FLOPs than the narrow arms. The alternative — matched FLOPs — would give the
wide model roughly 27k steps and confound capacity with training length against
the frozen 100k baseline. Matched presentations is the cleaner design here.

---

## 6. Evaluation and selection rule (frozen before any result)

Every arm is evaluated by `scripts/evaluate_scaling_arm.py` on the frozen FineWeb
validation artifact, with epsilon matched across arms (deterministic per
validation sequence ID, seed 0):

* Phase-A grid `t_start ∈ {0.10, 0.25, 0.50}` x `NFE ∈ {1, 3, 5}` plus the exact
  `t_start=0` identity control (must be ΔLM 0 with 0 flow evaluations).
* Reported per cell: corrupted ΔLM, reconstructed ΔLM, recovered damage,
  relative L2, cosine.
* Validation flow loss overall and by `t` bin, on the same rows with the same
  times and noise for every arm (seed 20260814).
* `NFE ∈ {10, 20}` may be rechecked **after** selection, diagnostic only.

Selection rule, frozen in the protocol:

1. **Primary**: lowest mean reconstructed ΔLM at `t_start=0.50, NFE=1` over the
   256 frozen validation sequences.
2. An arm beats another only if the paired bootstrap CI (10,000 resamples, seed
   20260813, unit = validation sequence) of the per-sequence difference excludes
   zero. Otherwise the arms are tied.
3. Tie-breakers, in order: lower validation flow MSE, then fewer parameters, then
   fewer unique activation tokens — the cheap model wins an unresolved contest.
4. Steering metrics, Phase-B rows, DEV directions, held-out directions, and LLM
   judges may not enter the rule. Diagnostic cells cannot select an arm.

Phase B is not to be consulted until an arm has been selected by this rule.

---

## 7. Smoke results (already run on the worker)

Receipt: `/workspace/results/flow_scaling_smoke_v1.json`, `status=complete`,
`research_status=NOT_EVALUATED` (operational only; not evidence).

FineWeb collection smoke — artifact `smoke_fw_256seq_v1`, 32,512 activations:

* pinned repository/config/revision recorded in metadata and token manifest;
* 278 documents consumed, 22 skipped short, 256 kept, 193,863 raw text tokens;
* raw tokens 32,768, BOS discarded 256, padding discarded 0, valid 32,512;
* float16 storage, float64 statistics, mean/std recomputation error `2.8e-5` /
  `3.3e-5`;
* BOS-probe minimum distance 3018.61 (no BOS activation retained);
* deterministic replay: identical array SHA256 `4341a0475ef686ef…` on a second
  independent collection;
* mean activation norm 88.48, against 88.90 for the OpenWebText artifact;
* validation report `VALID`.

Wide-model smoke, batch 1024 on the RTX 4090:

* exact parameter count 60,407,808;
* finite forward, backward, gradient norm, and optimizer updates;
* fixed-batch overfit: loss 2.3310 → 1.2942 over 100 steps;
* 53.3 steps/s resident, peak CUDA allocation 1,293,988,352 bytes (1.29 GB);
* checkpoint save/load round-trip reproduces outputs and parameter count.

End-to-end smoke (FineWeb → activations → 200-step wide training → checkpoint →
Phase A, no steering):

* training 28.2 steps/s, peak CUDA 1,774,121,984 bytes (1.77 GB);
* val flow MSE 1.7506 → 1.5378 across the four evaluations;
* Phase-A identity ΔLM exactly 0 with 0 flow-network evaluations;
* corruption ΔLM 0.0195 / 0.1847 / 1.3194 at `t_start` 0.10 / 0.25 / 0.50;
* reconstruction at NFE 1 already below corruption: 0.0034 / 0.0728 / 0.5549.

---

## 8. Commands

Every command below runs on a single CUDA device. `HF_HOME` points at whatever
directory holds the model and dataset cache. The activation directory and the
run directory are passed explicitly so that the same commands work regardless of
where the bulk artifacts live.

Collection, ~1-1.5 h wall clock, dominated by tokenization:

```bash
uv run python scripts/collect_activations.py \
  --corpus fineweb --split val --n-tokens 1024128 \
  --name resid7_fw_val_1024k_v1 \
  --output-dir "$ACT_DIR" --token-cache-dir "$TOK_DIR" --device cuda

uv run python scripts/collect_activations.py \
  --corpus fineweb --split train --n-tokens 4000119 \
  --name resid7_fw_train_4000k_v1 \
  --output-dir "$ACT_DIR" --token-cache-dir "$TOK_DIR" --device cuda

uv run python scripts/collect_activations.py \
  --corpus fineweb --split train --n-tokens 32000063 \
  --name resid7_fw_train_32000k_v1 \
  --output-dir "$ACT_DIR" --token-cache-dir "$TOK_DIR" --device cuda
```

Every artifact is validated before any training run touches it:

```bash
for name in resid7_fw_val_1024k_v1:val \
            resid7_fw_train_4000k_v1:train \
            resid7_fw_train_32000k_v1:train; do
  uv run python scripts/validate_activations.py \
    --corpus fineweb --name "${name%%:*}" --expected-split "${name##*:}" \
    --activation-dir "$ACT_DIR" --token-cache-dir "$TOK_DIR" --device cuda
done
```

The four training runs:

```bash
for arm in narrow16m_fw4m narrow16m_fw32m wide60m_fw4m wide60m_fw32m; do
  uv run python scripts/train_flow.py \
    --config "configs/flow_train_scaling_${arm}_v2.yaml" \
    --activation-dir "$ACT_DIR" \
    --run-dir "checkpoints/flow_scaling_${arm}_v2" \
    --device cuda
done
```

Concept-independent evaluation of each arm:

```bash
for arm in narrow16m_fw4m narrow16m_fw32m wide60m_fw4m wide60m_fw32m; do
  uv run python scripts/evaluate_scaling_arm.py \
    --config configs/flow_scaling_2x2_v2.yaml \
    --arm "$arm" \
    --training-experiment-id "flow_scaling_${arm}_v2" \
    --run-dir "checkpoints/flow_scaling_${arm}_v2" \
    --activation-dir "$ACT_DIR" --token-cache-dir "$TOK_DIR" \
    --output "results/flow_scaling_2x2_v2/${arm}.json" \
    --device cuda
done
```

The smoke pipeline is `scripts/smoke_scaling_pipeline.py`; rerunning it needs a
fresh `--work-dir` and `--output`, because the outputs are immutable.

---

## 9. Expected artifact and result paths

```text
/workspace/data/fineweb_token_cache/train_251969x128_<fingerprint>.npy (+ .manifest.json)
/workspace/data/fineweb_activations/resid7_fw_{val_1024k,train_4000k,train_32000k}_v1.{npy,json}
/workspace/data/fineweb_activations/resid7_fw_*_v1_stats.npz
/workspace/data/fineweb_activations/resid7_fw_*_v1_validation.json
/workspace/checkpoints/flow_scaling_<arm>_v2/{meta.json,best.json,best_step_*.pt,step_*.pt,last.pt}
/workspace/results/flow_scaling_2x2_v2/<arm>.json
/workspace/results/flow_scaling_smoke_v1.json
```

`configs/flow_scaling_2x2_artifacts_v1.yaml` is the human-readable index of these
identities and must be filled in as each step completes.

---

## 10. Cost

GPU work at the v2 budget: about 0.4 h collection (measured), 1.1 h narrow
training, 3–4 h wide training, and a few minutes of evaluation — roughly
**5–6 h of RTX 4090 time**. Tokenization for the 32M cache measured at about
14 min, well under the earlier estimate. At typical OctaSpace
4090 pricing of about $0.3–0.5/h this is on the order of **$1–2**, dominated by
however long the worker stays alive rather than by the runs themselves. Treat the
tokenization estimate as unmeasured.

---

## 11. Checkpoint retention (added with v2)

`train_flow` now implements the declared `keep: [best, last, configured_steps]`
policy: a superseded best checkpoint is deleted once its replacement is on disk,
and the config loader rejects any other declared policy. Before this fix every
validation improvement left a full optimizer-state checkpoint behind (~195 files
per run at 100k steps, ~500 at 250k), which is what filled the first worker's
disk mid-experiment. Checkpoint writing consumes no RNG, so the fix cannot change
training dynamics. Retained per run: 5 configured step checkpoints + `last.pt` +
1 best, about 1.3 GB narrow and 5.1 GB wide.
