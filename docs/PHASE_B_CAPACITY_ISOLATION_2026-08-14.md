# Phase B capacity isolation: narrow16m_fw32m vs wide60m_fw32m

**Date:** 2026-08-14
**Class:** DEV method development, frozen protocol reused verbatim.
**Held-out:** not accessed. **LLM judge:** not used. **Continuations:** not regenerated.

Comparison artifact: `results/remote_phase_b_wide60m/narrow16m_vs_wide60m.json`
SHA256 `0f055de8ac0a1988127ed17e2589e7c71c26d4a71f1c47065d1d8a3e760fbfe5`
source_revision `snapshot-sha256:7e84eddf75c7ae6dcd75e970e9d86ebf61867aba95185c1fd4a9342d83bfed8d`

---

## 1. Artifact preservation

| release | local path | status |
|---|---|---|
| wide60m_fw32m | `results/phase_b_wide60m_v1/…_148d23a6…/` | 24 raw artifacts + metas verified, brackets + manifest verified |
| narrow16m_fw32m | `results/phase_b_narrow16m_fw32m_v1/…_c64e524d…/` | 12 raw artifacts verified, 13/13 analysis-referenced files verified |

Config `configs/flow_phase_b_evaluator_narrow16m_fw32m_v1.yaml` SHA256
`59ce46d033e9b310398e67db77c056b54554f3127ddb8e3a600691d0cacbfbc5`, matching the
value recorded inside `analysis.json`.

Logs, driver script, and smoke output pulled to `_logs/` and `_smoke/`.
Remote copies under `/workspace/results/` were **not** deleted.

## 2. Degeneracy audit

Closed. See `docs/DEGENERACY_RULE_AUDIT_2026-08-14.md`. Historical rule recovered
exactly (cell-mean); row-level made canonical on a stated principle; max effect
0.0401 nats, no sign flip, no CI crossing, no headline change. Both releases in
this document are analyzed under the row rule.

## 3. Provenance of the capacity control

`narrow16m_fw32m` and `wide60m_fw32m` are siblings of the same frozen 2×2 run.

| | narrow16m_fw32m | wide60m_fw32m |
|---|---|---|
| parameters | 16,147,200 | 60,407,808 |
| architecture config | `flow_core_v1.yaml` | `flow_core_wide_60m_v1.yaml` |
| checkpoint SHA256 | `70f8999d…` | `68482e68…` |
| training dataset | `resid7_fw_train_32000k_v1` | identical |
| statistics SHA256 | `2e3081f1…` | identical |
| split fingerprint | `02dad171587ee40f` | identical |
| steps / presentations | 250k / 256M | identical |
| `selected_by_steering_evidence` | false | false |

The three baselines (additive, naive, shrinkage 0.8) are **bit-identical** across
the two releases on every metric — the strongest available check that the only
moving part is the flow prior. Matched epsilon confirmed identical at the smoke
cell. `n_rows` 2400 in both, delta 0.

## 4. Phase A (concept-independent), fixed data

| | narrow16m | wide60m | Δ |
|---|---:|---:|---:|
| val_flow_mse | 0.980932 | 0.838363 | −14.53% |
| reconstructed ΔLM @ t=.50/NFE1 | 0.304079 | 0.222219 | **−26.92%** |
| recovered fraction | 0.771453 | 0.832979 | +7.98% |
| relative L2 | 0.322929 | 0.278678 | −13.70% |
| cosine | 0.945445 | 0.959427 | +1.48% |

Paired bootstrap wide − narrow on ΔLM: **−0.081860 [−0.087774, −0.076017]**, resolved.
Corruption ΔLM identical (1.330485) and identity ΔLM 0.0 with 0 flow evaluations for both.

## 5. Phase B, matched realised projection

Positive = flow **worse** than the baseline. Δ = wide − narrow, paired vector bootstrap
(seed 20260813, 10,000 resamples, shared resample matrix).

### vs additive

| arm | narrow | wide | Δ(w−n) | 95% CI | signs+ | LOVO range | supp n / w |
|---|---:|---:|---:|---|---:|---|---:|
| t=.10 NFE1 | 0.0470 | 0.0458 | −0.0012 | [−0.0458, +0.0398] | 4/8 | [−0.012, +0.015] | 1759 / 1752 |
| t=.10 NFE3 | 0.0443 | 0.0172 | −0.0271 | [−0.0462, −0.0102] | 1/8 | [−0.032, −0.020] | 1752 / 1761 |
| t=.10 NFE5 | 0.0385 | 0.0116 | −0.0270 | [−0.0451, −0.0084] | 1/8 | [−0.034, −0.021] | 1749 / 1758 |
| t=.25 NFE1 | 0.3369 | 0.3251 | −0.0118 | [−0.0995, +0.0596] | 5/8 | [−0.025, +0.023] | 1780 / 1820 |
| t=.25 NFE3 | 0.3074 | 0.2543 | −0.0531 | [−0.1008, −0.0117] | 3/8 | [−0.063, −0.033] | 1751 / 1776 |
| t=.25 NFE5 | 0.3087 | 0.2560 | −0.0528 | [−0.0960, −0.0066] | 1/8 | [−0.070, −0.037] | 1752 / 1756 |
| **t=.50 NFE1** | **1.3874** | **1.1460** | **−0.2414** | **[−0.3012, −0.1805]** | **0/8** | [−0.260, −0.219] | 1852 / 1796 |
| t=.50 NFE3 | 1.2352 | 1.0363 | −0.1989 | [−0.2823, −0.1115] | 1/8 | [−0.231, −0.175] | 1831 / 1813 |
| t=.50 NFE5 | 1.2192 | 1.0239 | −0.1953 | [−0.3057, −0.0565] | 1/8 | [−0.256, −0.167] | 1822 / 1798 |

### vs shrinkage κ=0.8

| arm | narrow | wide | Δ(w−n) | 95% CI | signs+ |
|---|---:|---:|---:|---|---:|
| t=.10 NFE1 | 0.0516 | 0.0329 | −0.0186 | [−0.0622, +0.0179] | 5/8 |
| t=.10 NFE3 | 0.0403 | 0.0145 | −0.0258 | [−0.0516, −0.0053] | 1/8 |
| t=.10 NFE5 | 0.0319 | 0.0030 | −0.0289 | [−0.0506, −0.0100] | 1/8 |
| t=.25 NFE1 | 0.3022 | 0.2730 | −0.0292 | [−0.1005, +0.0342] | 4/8 |
| t=.25 NFE3 | 0.2949 | 0.2303 | −0.0645 | [−0.1036, −0.0327] | 0/8 |
| t=.25 NFE5 | 0.2896 | 0.2402 | −0.0495 | [−0.0843, −0.0123] | 1/8 |
| t=.50 NFE1 | 1.3751 | 1.1375 | −0.2376 | [−0.3059, −0.1699] | 0/8 |
| t=.50 NFE3 | 1.2251 | 1.0284 | −0.1966 | [−0.2681, −0.1285] | 0/8 |
| t=.50 NFE5 | 1.2067 | 0.9821 | −0.2246 | [−0.2994, −0.1503] | 0/8 |

**Every arm of both models is above zero against both baselines.**
`H_pareto`: `arms_below_zero_vs_additive = []`. No candidate, resolved or otherwise.

### Concept metrics and controls (matched projection, flow − additive)

| arm | lexicon n→w | Δ CI | SAE target n→w | Δ CI | ctrl-mean Δ | ctrl-max Δ |
|---|---|---|---|---|---|---|
| t=.10 NFE1 | +0.00003→−0.00005 | [−0.0011,+0.0009] | +0.0101→+0.0059 | [−0.026,+0.015] | −0.00037 | −0.0091 |
| t=.25 NFE1 | +0.00191→+0.00453 | [+0.0012,+0.0044] | +0.0171→+0.0498 | [+0.009,+0.060] | −0.00007 | −0.0021 |
| t=.50 NFE1 | +0.00754→+0.01065 | [−0.0002,+0.0065] | +0.1332→+0.1361 | [−0.053,+0.075] | −0.00005 | −0.0029 |

Unrelated-SAE controls move by ≤ 4e−4 (mean) throughout and mostly *toward* zero for
wide, so nothing here is broad activation inflation.

### Geometry (narrow → wide)

| arm | retained frac | realised proj | ‖c‖ | ‖c∥‖ | ‖c⊥‖ | cos |
|---|---|---|---|---|---|---|
| t=.10 NFE1 | 0.947→0.924 | 39.52→38.92 | 7.79→8.15 | 1.78→2.38 | 7.51→7.68 | −0.22→−0.28 |
| t=.25 NFE1 | 0.760→0.710 | 33.60→32.18 | 20.10→20.30 | 7.73→9.15 | 18.12→17.48 | −0.37→−0.43 |
| t=.50 NFE1 | 0.444→0.426 | 21.33→21.43 | 38.80→36.71 | 20.05→19.94 | 31.81→29.29 | −0.48→−0.51 |

At the primary cell the wide prior does **less orthogonal damage** (31.81 → 29.29,
−7.9%) at essentially the same realised projection (21.33 → 21.43). That, not extra
attenuation, is where its matched-projection gain comes from: retained fraction is
*lower* for wide at every arm, and matched projection already controls for realised
strength. So the capacity effect on steering is real and not a relabelled shrinkage.

## 6. Capacity transfer: does the Phase-A gain reach Phase B?

Yes, resolvedly, and in the right direction — but at a fraction of the size and
nowhere near the goal.

| | narrow | wide | relative reduction |
|---|---:|---:|---:|
| Phase A reconstructed ΔLM | 0.3041 | 0.2222 | **−26.9%** |
| Phase B matched-projection penalty, primary cell | 1.3874 | 1.1460 | **−17.4%** |

Transfer ratio ≈ **0.65**. Both quantities are LM-degradation in nats with an absolute
zero floor, so the ratio is meaningful, with the caveat that Phase A measures distance
to the clean activation and Phase B measures distance to the additive baseline. It is
reported as an order-of-magnitude statement, not a calibrated constant.

The extrapolation that matters: 3.74× parameters buys 0.2414 nats, i.e. ≈0.127 nats per
parameter-doubling at the primary cell. Reaching additive parity from 1.146 would need
≈9 further doublings — of order 10<sup>10</sup> parameters for a "cheap" activation prior on
GPT-2-small residuals. Two points make this a crude log-linear extrapolation, but no
plausible refinement of the fit rescues three orders of magnitude.

The one arm where extrapolation looks survivable, `t=.10 NFE5` (wide already at +0.0116
vs additive and +0.0030 vs shrinkage), is the arm where the prior barely acts: retained
fraction 0.948, correction norm 7.7 against a realised projection of 39.6. Converging to
the baseline by converging to the identity is not a method.

## 7. NFE dissociation, and whether it is capacity-dependent

Matched-projection NLL, NFE_k − NFE_1, negative = more integration steps help.

| contrast | narrow | 95% CI | wide | 95% CI |
|---|---:|---|---:|---|
| t=.10 NFE3−1 | +0.0100 | [−0.0009,+0.0199] | +0.0017 | [−0.0163,+0.0211] |
| t=.10 NFE5−1 | +0.0122 | [+0.0041,+0.0206] | +0.0009 | [−0.0237,+0.0270] |
| t=.25 NFE3−1 | +0.0292 | [+0.0061,+0.0510] | +0.0037 | [−0.0267,+0.0426] |
| t=.25 NFE5−1 | +0.0408 | [+0.0037,+0.0721] | +0.0222 | [−0.0157,+0.0688] |
| t=.50 NFE3−1 | −0.0754 | [−0.1127,−0.0379] | −0.0537 | [−0.0914,−0.0106] |
| t=.50 NFE5−1 | −0.0679 | [−0.1202,−0.0124] | −0.0588 | [−0.1146,−0.0025] |

**The t=0.50 dissociation is not capacity-dependent.** It is present and resolved in
both models, with overlapping intervals and near-identical magnitude. Extra integration
steps help only at t=0.50 and hurt or do nothing below it, in both.

What extra steps buy at t=0.50 is visible on the concept side:

| contrast, SAE target | narrow | wide |
|---|---:|---:|
| t=.50 NFE3−1 | −0.0234 [−0.051,+0.003] | −0.0708 [−0.147,−0.017] |
| t=.50 NFE5−1 | −0.0210 [−0.039,−0.005] | −0.0807 [−0.169,−0.020] |

More steps at t=0.50 lower NLL while lowering target SAE activation, and the wide model
does more of both. That is the attenuation signature, and it means the t=0.50 NFE effect
should not be read as "iterative integration recovers the concept". `t_start` remains the
dominant control axis; NFE is a second-order attenuation knob. Realised projection is
reported as geometry only and is not treated as concept strength.

## 8. Branch classification

The offered options do not fit exactly, and forcing one would misreport the result.

* **A (capacity explains the failure)** — rejected. The wide prior loses to additive at
  every arm, 0/8 favourable vector signs at the primary cell.
* **B (capacity improves reconstruction but not steering)** — closest, but **factually
  wrong on its second half**. Capacity improves steering correction resolvedly at 7 of 9
  arms, and the mechanism is legitimate (less orthogonal damage at matched projection),
  not an artefact.
* **C (data dominates capacity)** — not tested by this experiment; data was held fixed by
  construction. The earlier 2×2 remains the only evidence on that axis.
* **D (insufficient)** — rejected. Both hypotheses are resolved and the controls are clean.

**Recorded classification: B′ — capacity transfers into steering, in the right direction,
at ~65% of its Phase-A relative effect, from a starting penalty large enough that the
transfer is irrelevant to the scientific question.** The method is not limited by the
prior's quality as a model of natural activations.

## 9. Should Gaussian-flow scaling stop?

**Yes. Stop it.**

The branch has now answered its own question. The prior got measurably better as a model
of activations (Phase A, −26.9%), that improvement did transfer to steering correction
(Phase B, −17.4%, resolved), and the method still lost to `h + αv` by 1.15 nats and to a
one-line scalar shrinkage by 1.14 nats. Scaling is not the binding constraint; a further
3.74× would move the primary cell to roughly 0.95 nats, still an order of magnitude of
doublings from parity. No further Gaussian-flow capacity or data arm should be run.

Per the standing instruction for this task: no model beyond ~60M, no deeper 100M+ prior
as a rescue attempt.

## 10. LLM judging

Not warranted and not run. `H_pareto` returned an empty candidate set — there is no arm
whose concept/quality trade-off is ambiguous under the current metrics, so there is
nothing for a semantic audit to disambiguate.

## 11. Recommended next experiment

`docs/STRUCTURED_CORRUPTION_PROPOSAL.md` — **PROPOSED, NOT FROZEN, NOT LAUNCHED.**

The result above is what makes it the right successor rather than a consolation prize:
the prior's failure is not a quality failure, so the remaining candidate explanation is
that the training corruption family (isotropic Gaussian) does not contain the object
inference is asked to invert (a rank-1 semantic displacement). The proposal changes
exactly that one variable, keeps `wide60m_fw32m` as a direct control, and carries its own
stopping condition: if it loses at matched realised projection against both additive and
shrinkage, the "cheap activation prior as steering corrector" family is closed.

Five open questions in §9 of that document need human answers before it can be frozen.

**Not launched. Requires explicit approval.**
