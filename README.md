# Cheap learned priors for activation steering

Can the damage that activation steering does to a language model be reduced while
the target concept is still expressed? This project tests a learned activation
correction against plain `h + αv` in GPT-2 small.

**📄 Final report: [`report/report.pdf`](report/report.pdf)** (23 pages, in Russian) ·
[LaTeX source](report/report.tex) · [artifact inventory](report/INVENTORY.md)

**🤗 Weights: [qweclownq/steering-denoisers](https://huggingface.co/qweclownq/steering-denoisers)** ·
final model on its own: [qweclownq/tangent-flow-16m](https://huggingface.co/qweclownq/tangent-flow-16m)

---

## Main result

> A model's ability to reconstruct natural activations does not by itself transfer
> to repairing steering interventions. After successively removing steering
> attenuation, prior weakness, the absence of conditioning, and the mismatch
> between training and inference geometry, no useful orthogonal naturalization
> appears. A plain **hard clamp** remains the stronger baseline — it costs only
> +0.003 to +0.054 nats while satisfying the coordinate exactly.

The decisive experiment: the tangent flow model solves its own task convincingly
(T1: 77.3 % of the damage recovered), yet with the semantic coordinate held fixed
it makes quality worse than a hard clamp — T2: ΔNLL = **+0.006184**, 95 % CI
[+0.001631, +0.010788], 0 of 30 diagnostic configurations favourable.

The negative result follows a preregistered stop rule fixed before any data
existed.

---

## Installation

```bash
uv sync                 # environment from uv.lock
uv run pytest -q        # 731 tests
uv run ruff check src/ scripts/ tests/
```

Requires Python 3.12+. No GPU is needed to rebuild the report.

---

## Build the report (one command, no GPU)

```bash
make report    # figures and tables from artifacts, then XeLaTeX → BibTeX → XeLaTeX ×2
make verify    # every number in the report against its artifact (121 assertions)
```

`scripts/build_report_figures.py` reads **only** immutable artifacts from
`results/`, trains nothing and generates nothing, and produces `report/figures/`
(14 figures as PNG and PDF), `report/tables/` (4 tables as CSV and Markdown) and
`report/data/` (the aggregated series behind each figure, as JSON).

`make report` needs XeLaTeX and the `cm-unicode` fonts from TeX Live: Latin Modern
has no Cyrillic coverage, so the text font is loaded from an explicit path set by
`\cmupath` in the preamble of `report/report.tex`. On another system that one line
is all you need to change.

---

## What is in this repository

| Path | Contents |
|:---|:---|
| `report/` | report source, built PDF, figures, tables, the series behind each figure |
| `src/interp/` | flow matching, tangent flow, steering evaluation, metrics |
| `scripts/` | training, evaluation, diagnostics, report build, verifier |
| `tests/` | 731 tests, mostly on scientific invariants rather than code shape |
| `configs/` | the configuration of every run, each with a fingerprint |
| `docs/` | frozen protocols, branch-closure records, governance |
| `audits/` | scripts for one-off methodological audits, kept verbatim |
| `results/` | the small machine-readable artifacts the report is built from |

Large artifacts — weights, activation dumps, raw generations — are not kept in
git. The weights are published on HuggingFace; everything else is listed in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) together with the commands to
regenerate it.

---

## Weights

All six models: **<https://huggingface.co/qweclownq/steering-denoisers>**

| Model | Directory in the HF repo |
|:---|:---|
| Tangent flow 16M, linear path (the final model) | `tangent-flow-16m-linear/` |
| Tangent flow 16M, variance-preserving path | `tangent-flow-16m-variance-preserving/` |
| Denoiser trained on steering-like corruption | `steering-corruption-denoiser-16m/` |
| Isotropic flow prior, 16M | `flow-prior-16m/` |
| Isotropic flow prior, 60M | `flow-prior-60m/` |
| Conditional flow, 60M | `conditional-flow-60m/` |

The final model is also published separately, together with the T1 and T2
receipts: [`qweclownq/tangent-flow-16m`](https://huggingface.co/qweclownq/tangent-flow-16m).

The weights of the early Gaussian denoiser (`963b4dda…7770c583`) were not
preserved. Its behaviour **under steering** is recomputed in full from the Phase B
artifacts (the `naive` arm); the four numbers describing its denoising ability are
transcribed from the experimental record and labelled as transcribed —
`results/gaussian_denoiser_v1/reconstruction_record.json`.

---

## Reproducing individual parts

Everything below needs a GPU and collected activations. Nothing here has to run in
order to read the report; the full list of artifacts, hashes, seeds and commands is
in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

**T1 — tangent reconstruction:**

```bash
uv run python scripts/eval_tangent_reconstruction.py \
  --checkpoint <tangent-flow-16m-linear/model.pt> \
  --run-dir <training run directory> \
  --activation-dir <activations> --token-cache-dir <token cache> \
  --name resid7_fw_val_1024k_v1 \
  --pool data/direction_pools/training_only_rank256_v1.pt \
  --out-dir results/<new directory>
```

**T2 — hard clamp versus tangent flow** (requires a receipt that T1 passed):

```bash
uv run python scripts/tangent_naturalization.py \
  --checkpoint <tangent-flow-16m-linear/model.pt> \
  --t1-receipt <t1_receipt.json> \
  --activation-dir <activations> --token-cache-dir <token cache> \
  --name resid7_fw_val_1024k_v1 \
  --pool data/direction_pools/training_only_rank256_v1.pt \
  --out-dir results/<new directory>
```

Result directories are immutable: writing into a non-empty directory is refused
unless `--overwrite-debug-mode` is passed explicitly, which marks the run as a
debug run.

**Training** — `scripts/train_flow.py` with a configuration from `configs/`.
Configurations whose status is `prepared` refuse to run.

---

## Protected data

Eight held-out directions were **never used in any artifact of this project** —
every one carries the receipt `held_out_accessed: false`. Their identities are not
part of the public repository. The source-provenance function
(`interp.provenance`) excludes the protected directory from its walk before
opening any file, which is covered by tests.

The rules are described in [`docs/RESEARCH_GOVERNANCE.md`](docs/RESEARCH_GOVERNANCE.md).

---

## Code layout

```
src/interp/
  flow_core.py         base flow matching (path, velocity target, architecture)
  conditional_flow.py  conditioning on (direction, coordinate), direction pool
  constrained_flow.py  geometry at a fixed coordinate (independent derivation)
  tangent_flow.py      constraint-preserving tangent flow
  tangent_eval.py      T1/T2 gates, clustered bootstrap, provenance checks
  train_flow.py        one trainer for every variant
  phase_b*.py          steering evaluation on DEV directions
  natural_support.py   frozen evaluation plan over the natural range
```

The tangent geometry is deliberately derived twice — in `tangent_flow.py` and in
`constrained_flow.py` — with an equivalence test between them. That is an
independent check on the central formulas; the two should not be merged into one
function.
