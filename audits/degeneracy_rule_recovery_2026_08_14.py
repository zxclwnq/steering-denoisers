"""Audit: which degeneracy reading reproduces the provisional 1919/91/390 split?

Tries candidate historical interpretations against the untouched narrow raw rows.
Nothing here becomes canonical; the point is provenance, not a better number.
"""
import json, collections
from pathlib import Path

REL = Path("results/remote_clean_flow_phase_b_dev_v1_e965eb211254b0c24d84475782091beb30d11f99fd328ee49f26a6c403d26a4f")
THR = 0.027785714285713504

def rows(p):
    return [json.loads(l) for l in open(p)]

flow = [r for r in rows(REL/"flow"/"flow_t010_nfe1.jsonl") if not r["is_stress"]]
add  = [r for r in rows(REL/"baselines"/"additive.rescored.jsonl") if not r["is_stress"]]

# baseline curve: coordinate = alpha (additive realises exactly alpha)
curves = collections.defaultdict(list)
for r in add:
    curves[(r["vector"], r["prompt_id"], r["generation_seed"])].append(r)
for k in curves:
    curves[k].sort(key=lambda r: r["alpha"])

# cell-level (vector, alpha) mean repetition for the baseline
cell_rep = collections.defaultdict(list)
for r in add:
    cell_rep[(r["vector"], r["alpha_hex"])].append(r["metrics"]["repetition_rate"])
cell_mean = {k: sum(v)/len(v) for k, v in cell_rep.items()}
flow_cell_rep = collections.defaultdict(list)
for r in flow:
    flow_cell_rep[(r["vector"], r["alpha_hex"])].append(r["metrics"]["repetition_rate"])
flow_cell_mean = {k: sum(v)/len(v) for k, v in flow_cell_rep.items()}

def bracket(row):
    pts = curves[(row["vector"], row["prompt_id"], row["generation_seed"])]
    t = row["geometry"]["realized_projection_mean"]
    for a, b in zip(pts, pts[1:]):
        if a["alpha"] <= t <= b["alpha"]:
            return a, b
    return None

VARIANTS = {
    "clean analyzer (flow row + both bracket rows)":
        lambda r, lo, hi: r["metrics"]["repetition_rate"] > THR or
                          lo["metrics"]["repetition_rate"] > THR or
                          hi["metrics"]["repetition_rate"] > THR,
    "flow row only":
        lambda r, lo, hi: r["metrics"]["repetition_rate"] > THR,
    "flow row + lower bracket only":
        lambda r, lo, hi: r["metrics"]["repetition_rate"] > THR or
                          lo["metrics"]["repetition_rate"] > THR,
    "flow row + baseline (vector,alpha) cell means":
        lambda r, lo, hi: r["metrics"]["repetition_rate"] > THR or
                          cell_mean[(lo["vector"], lo["alpha_hex"])] > THR or
                          cell_mean[(hi["vector"], hi["alpha_hex"])] > THR,
    "flow (vector,alpha) cell mean + baseline cell means":
        lambda r, lo, hi: flow_cell_mean[(r["vector"], r["alpha_hex"])] > THR or
                          cell_mean[(lo["vector"], lo["alpha_hex"])] > THR or
                          cell_mean[(hi["vector"], hi["alpha_hex"])] > THR,
    "flow (vector,alpha) cell mean only":
        lambda r, lo, hi: flow_cell_mean[(r["vector"], r["alpha_hex"])] > THR,
    "clean rule but >= instead of >":
        lambda r, lo, hi: r["metrics"]["repetition_rate"] >= THR or
                          lo["metrics"]["repetition_rate"] >= THR or
                          hi["metrics"]["repetition_rate"] >= THR,
}

print(f"target (provisional): supported 1919  unsupported 91  degenerate 390\n")
print(f"{'variant':<52}{'sup':>7}{'unsup':>7}{'degen':>7}  match")
for name, is_degen in VARIANTS.items():
    c = collections.Counter()
    for r in flow:
        br = bracket(r)
        # order matters: the clean analyzer tests flow degeneracy before bracketing
        if br is None:
            if is_degen(r, r, r) and name in ("flow row only", "flow (vector,alpha) cell mean only"):
                c["degenerate"] += 1
            else:
                c["unsupported"] += 1
            continue
        lo, hi = br
        if is_degen(r, lo, hi):
            c["degenerate"] += 1
        else:
            c["supported"] += 1
    hit = (c["supported"], c["unsupported"], c["degenerate"]) == (1919, 91, 390)
    print(f"{name:<52}{c['supported']:>7}{c['unsupported']:>7}{c['degenerate']:>7}  {'<== MATCH' if hit else ''}")

# how the clean analyzer orders the two rejections
c = collections.Counter()
for r in flow:
    if r["metrics"]["repetition_rate"] > THR:
        c["degenerate_flow"] += 1; continue
    br = bracket(r)
    if br is None:
        c["outside_bracket"] += 1; continue
    lo, hi = br
    if lo["metrics"]["repetition_rate"] > THR or hi["metrics"]["repetition_rate"] > THR:
        c["degenerate_bracket"] += 1
    else:
        c["supported"] += 1
print("\nclean analyzer decomposition:", dict(c))
