# Vendored artifacts — Arditi et al. refusal direction

Source: `https://github.com/andyrdt/refusal_direction`, commit
`9d852fae1a9121c78b29142de733cb1340770cc3`, MIT licence.

Paper: Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda,
*Refusal in Language Models Is Mediated by a Single Direction*, arXiv:2406.11717.

Nothing here was re-derived, re-selected or modified. Files are byte-identical
copies of the upstream artifacts.

| file | upstream path | SHA256 |
|---|---|---|
| `gemma-2b-it_direction.pt` | `pipeline/runs/gemma-2b-it/direction.pt` | `7ec71901fe89520fb9ad3c5800a06284453993cdff5222b3f8f304fd6229b6e9` |
| `gemma-2b-it_direction_metadata.json` | `pipeline/runs/gemma-2b-it/direction_metadata.json` | — |
| `splits/harmful_test.json` | `dataset/splits/harmful_test.json` | `5e12ae102c3791dee083a69ab6269a78e033411c629bc3f66f75d2fde196d9ef` |
| `splits/harmless_test.json` | `dataset/splits/harmless_test.json` | `1b5930ce5e855ada758b3116ce7c4aaea9b9d05f8cdd77b385511d4c84173b19` |

## The direction tensor

* shape `(2048,)`, dtype `float64`, norm `10.064353277286578`;
* it is the **raw** difference-in-means vector, deliberately **not**
  unit-normalized — its scale is the published activation-addition magnitude;
* metadata: `layer = 10`, `pos = -2`.

`pos = -2` indexes the end-of-instruction token positions, i.e. the second-to-last
token of the gemma chat template suffix that follows the instruction. The
activation site is the residual stream **entering** decoder block 10 (a forward
pre-hook on `model.model.layers[10]`).

## Splits

`harmful_test` (572 instructions) and `harmless_test` (6266 instructions) are
disjoint from the `train` split the direction was derived from and the `val`
split it was selected on. Only these test splits are used by Experiment D.

The `train`/`val` splits are deliberately **not** vendored, so they cannot be
used by accident.
