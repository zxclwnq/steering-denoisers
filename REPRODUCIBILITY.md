# Reproducibility and artifact provenance

Identifiers for the artifacts behind `report/report.pdf`. The report itself points
here rather than carrying this table, so that it reads as a research report.

## Rebuilding the report

    make report    # figures, tables, PDF
    make verify    # every numeric claim against its artifact (121 assertions)
    make test      # 731 tests
    make lint

None of these need a GPU. They read only the small machine-readable artifacts
tracked in `results/`, which is the complete set required to reproduce every
number in the report.

## Key artifacts

| artifact | identifier |
|---|---|
| checkpoint, model with orthogonal noise | `tangent-flow-16m-linear/model.pt`, SHA256 `066afb601418da89f79b003c97b37227a9aa7702a442ad5f3fb0ab68a4199d4c` |
| checkpoint selection | `val_flow_mse = 0.9680510`, minimized, using no steering metric |
| config fingerprint | `e4af61135b0205cdcd6f196a61d5af464f0369b29e9bcaa471fd0764e7f85499` |
| direction pool | `45241c49814abe71ed7106e1a0fcbbe7d8aad40b215621674ec72ee7356d7a2c` (training-only pool, rank threshold 256) |
| early Gaussian denoiser | SHA256 `963b4dda162d60f1064b47979843c6ca99b733a1837196abc758432e7770c583` (weights not preserved) |
| validation artifact | `resid7_fw_val_1024k_v1`, split fingerprint `5725541eaecf437c` |
| SAE | `jbloom/GPT2-Small-SAEs-Reformatted`, revision `57d08a4f…`, weights SHA256 `47bfb750…` |

## Published weights

All six trained models: <https://huggingface.co/qweclownq/steering-denoisers>

| directory in that repo | SHA256 of `model.pt` |
|---|---|
| `tangent-flow-16m-linear/` | `066afb601418da89f79b003c97b37227a9aa7702a442ad5f3fb0ab68a4199d4c` |
| `tangent-flow-16m-variance-preserving/` | `1047d063a8434804b68f28b808f60680e5d985b393b58d40cc2a570d87a25fcc` |
| `steering-corruption-denoiser-16m/` | `8b95467b1e8cd1978005482860de91b36ac825bbaffa1ba39d8ecb15e566c3c2` |
| `flow-prior-16m/` | `70f8999d7537a76023b9432eea1a4ef98dd28d43b42c3e86adaa004e4d64b298` |
| `flow-prior-60m/` | `68482e6837d9d72e04dbdc06728ffe18f0c1d81e7e5e9ed807bbe26ac36affb1` |
| `conditional-flow-60m/` | `83324cfab50eb7d055ac69f864bf7972f56e8a45bcdecaaf7d9eff7841933f76` |

The first three hashes were recorded independently by the evaluation artifacts
before the files were retrieved, and match byte-for-byte.

`tangent-flow-16m-linear` is also published on its own, with the T1 and T2
receipts alongside it, at <https://huggingface.co/qweclownq/tangent-flow-16m>.

## Artifacts not distributed through git

Weights, activation dumps and raw generation rows are too large for a git
repository. They are not needed to rebuild the report or to check any of its
numbers; they are needed only to re-run an experiment end to end.

| artifact | size | how to obtain |
|---|---|---|
| the six trained models | 2.1 GB | `huggingface_hub.snapshot_download("qweclownq/steering-denoisers")` |
| FineWeb activations `resid7_fw_train_32000k_v1` | 49.2 GB float16 | recollect with `scripts/collect_activations.py` |
| FineWeb activations `resid7_fw_train_4000k_v1` | 6.1 GB float16 | the same, at the smaller token count |
| FineWeb activations `resid7_fw_val_1024k_v1` | 1.6 GB float16 | the frozen validation artifact, split fingerprint `5725541eaecf437c` |
| training-only direction pool | 69 MB | rebuild with `scripts/build_training_direction_pool.py` |
| raw Phase-B generation rows beyond the four tracked JSONL files | 0.5 GB | regenerate with `scripts/evaluate_flow_phase_b.py` under `configs/flow_phase_b_evaluator_*.yaml` |
| intermediate training checkpoints every 10k steps | 5.1 GB | not preserved; only the selected step is published |

Every token count, sequence count, shape, split seed and split fingerprint for the
activation artifacts is recorded in `configs/fineweb_activations_v1.yaml`, and the
exact collection commands are in `docs/CHEAP_SCALING_EXPERIMENT_2X2.md` §8.

Recollection is deterministic given the seeds and token counts in the configs, and
`scripts/validate_activations.py` checks a recollected artifact against the
recorded split fingerprint before any training run may use it.

## The one set of numbers not recomputed from a result artifact

The weights and training config of the early Gaussian denoiser
(SHA `963b4dda…`) are not preserved, so its four reconstruction values come from
the transcribed experimental record in
`results/gaussian_denoiser_v1/reconstruction_record.json` rather than from a
rerun. Its behaviour *under steering* is recomputed in full from available
artifacts. The build script parses the record back and checks it against the
values printed in the report, so the two cannot drift apart.

## Access receipts

`held_out_accessed: false` in **every** artifact of the project.
`dev_vectors_accessed: false` for every evaluation that does not use concept
metrics. `llm_judge_used: false` throughout. No training was performed while
preparing the report.

The held-out direction identities are not part of this repository. They were
never evaluated, and no development decision was made with knowledge of them.

## Frozen protocols and machine-readable outcomes

Protocol documents, exact seeds, per-experiment configs and the machine-readable
outcome labels each analysis emitted live under `docs/`, `configs/` and
`results/`. They are deliberately not reproduced in the report.

## Framework choice

All interventions go through TransformerLens. The SAE was trained on activations
in that basis, and `from_pretrained` applies `fold_ln` and
`center_writing_weights` by default, so the residual stream is numerically not
the same as the input to `transformer.h[7]` in plain HuggingFace. Moving to a
faster stack would require reproducing that transform separately, or the concept
metric would silently be computed in a different basis.
