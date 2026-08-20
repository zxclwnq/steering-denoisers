# Audit: matched-projection degeneracy granularity

**Status:** closed. The historical rule was recovered exactly. The frozen text is
genuinely ambiguous on this point, so this note records a *disambiguation*, not a
bug fix, and states which reading is canonical going forward.

**Scope:** matched-projection support/degeneracy accounting only. No generation
was rerun. No raw artifact was modified.

---

## 1. The discrepancy

The provisional 2026-08-13 analysis reported, for the narrow release at
`flow_t010_nfe1` versus additive:

| | supported | unsupported | degenerate |
|---|---:|---:|---:|
| provisional analysis | 1,919 | 91 | 390 |
| clean analyzer `clean_phase_b_analysis_v1` | 1,771 | 91 | 538 |

The unsupported count is identical, so bracketing, the extrapolation ban, and the
coordinate definition already agreed. Only the degeneracy classification differed:
148 rows moved from supported to degenerate.

## 2. The old rule, recovered

Seven candidate readings were run against the untouched narrow raw rows. Exactly
one reproduces the provisional numbers:

| candidate reading | supported | unsupported | degenerate |
|---|---:|---:|---:|
| clean analyzer: flow **row** and both bracketing baseline **rows** | 1,771 | 91 | 538 |
| flow row only | 2,011 | 91 | 298 |
| flow row + lower bracket only | 1,889 | 91 | 420 |
| flow row + baseline `(vector, alpha)` **cell** means | 1,835 | 91 | 474 |
| **flow cell mean + both bracketing baseline cell means** | **1,919** | **91** | **390** |
| flow cell mean only | 1,979 | 91 | 330 |
| clean rule with `>=` instead of `>` | 1,771 | 91 | 538 |

The historical rule was therefore **cell-level**: a `(vector, alpha)` cell — 30
rows, ten prompts by three seeds — is degenerate when its *mean* repetition rate
exceeds `T_rep = 0.027785714285713504`, and the test is applied to the flow cell
and to both bracketing baseline cells.

Two independent confirmations that this is the right reconstruction, not a
coincidence of counts:

* under the cell rule the narrow `flow_t010_nfe1` matched-projection NLL
  difference is `0.0581`, against the provisional `0.058126`;
* the narrow `flow_t050_nfe1` value is `1.1616`, against the provisional `1.161636`.

The provisional wording matches: "Flow and both bracketing baseline **vector/alpha
cells** must be nondegenerate under the frozen repetition threshold."

## 3. The frozen contract is ambiguous here

`docs/METRICS_AND_STATISTICS.md` §9 says:

> A **sample or cell** exceeding the frozen threshold is marked degenerate
> according to the existing implementation/protocol.

That clause permits both granularities and defers to the implementation. The
evaluator config's `matching:` block fixes the identity, coordinate, interpolation,
extrapolation ban, clipping ban, and bracket recording, but says nothing about
degeneracy granularity. So neither implementation violates the frozen contract.

This is an actual inconsistency in the frozen protocol, and it is being recorded
as such rather than resolved silently.

## 4. Canonical rule going forward: row level

The clean analyzer's row-level rule is canonical, on one stated principle:

> the degeneracy gate must be applied at the same granularity as the comparison
> it gates.

Matched projection is a per-row interpolation producing a per-row difference. A
cell-mean gate admits individual generations that fail the gate whenever their
29 neighbours are clean, which is precisely what §10 of the metrics document
forbids: "degenerate **observations** do not define the primary valid
concept–quality Pareto frontier." `RESEARCH_GOVERNANCE.md` §6 uses the same
granularity: "**Points** failing the frozen degeneration gate do not define the
valid primary Pareto frontier."

The choice was made after both sets of numbers were visible, which is why the
principle is stated explicitly and why §5 records that the choice is not
outcome-favourable. The rule is recorded in every matched-projection block as
`degeneracy_rule`, so no future reader has to infer it.

## 5. Materiality: none

Point estimates of matched-projection ΔNLL versus additive under both readings,
all nine arms, both releases:

| arm | narrow row | narrow cell | wide row | wide cell | max shift |
|---|---:|---:|---:|---:|---:|
| t=.10 NFE1 | 0.0410 | 0.0581 | 0.0458 | 0.0518 | 0.0171 |
| t=.10 NFE3 | 0.0449 | 0.0619 | 0.0172 | 0.0330 | 0.0170 |
| t=.10 NFE5 | 0.0455 | 0.0603 | 0.0116 | 0.0271 | 0.0155 |
| t=.25 NFE1 | 0.3112 | 0.2773 | 0.3251 | 0.2850 | 0.0401 |
| t=.25 NFE3 | 0.2831 | 0.2668 | 0.2543 | 0.2218 | 0.0325 |
| t=.25 NFE5 | 0.2866 | 0.2610 | 0.2560 | 0.2279 | 0.0281 |
| t=.50 NFE1 | 1.1667 | 1.1616 | 1.1460 | 1.1473 | 0.0050 |
| t=.50 NFE3 | 1.0992 | 1.1038 | 1.0363 | 1.0360 | 0.0046 |
| t=.50 NFE5 | 1.0820 | 1.0814 | 1.0239 | 0.9992 | 0.0248 |

Largest absolute shift: **0.0401 nats**, at `t=.25 NFE1`, against a penalty of
`+0.31`. The rule choice is not outcome-favourable: it makes the flow arms look
*worse* at `t=.10` and *better* at `t=.25`, with no consistent direction.

Primary cell, with the frozen vector bootstrap:

| release | rule | mean | 95% CI | supported | signs + |
|---|---|---:|---|---:|---:|
| narrow | row | 1.1667 | [1.0166, 1.3236] | 1,823 | 8/8 |
| narrow | cell | 1.1616 | [0.9903, 1.3453] | 1,966 | 8/8 |
| wide | row | 1.1460 | [0.9883, 1.3076] | 1,796 | 8/8 |
| wide | cell | 1.1473 | [0.9452, 1.3465] | 1,967 | 8/8 |

Narrow→wide paired difference at the primary cell: `-0.0206` `[-0.0464, +0.0131]`
under the row rule, `-0.0143` `[-0.0616, +0.0351]` under the cell rule. Both
unresolved.

No headline changes: no sign flip, no CI crossing zero in either direction, no arm
moving below zero, H-Pareto rejected under both readings.

## 6. Disposition

* The provisional **supported/degenerate counts** and matched-projection table
  are **superseded** by `clean_phase_b_analysis_v1` — not because they were wrong
  under their own rule, but because a single granularity must be canonical.
* The provisional descriptive, equal-alpha, geometry, and NFE tables **stand**; the
  clean analyzer reproduces them exactly.
* Both narrow and wide releases are analyzed under the row rule, so every
  narrow↔wide comparison is internally consistent.
* Raw artifacts were not touched. Any future reader can recompute either reading
  from the immutable rows. The two scripts are kept verbatim at
  `audits/degeneracy_rule_recovery_2026_08_14.py` (§2) and
  `audits/degeneracy_rule_materiality_2026_08_14.py` (§5). They live outside
  `src/`, `scripts/`, and `configs/` so that running the audit cannot change any
  release's source-revision identity.
