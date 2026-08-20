"""Does the row-vs-cell degeneracy reading move any headline? Point estimates and
the primary-cell bootstrap, both releases, all nine arms, versus additive."""
import collections, json
from pathlib import Path
import numpy as np

from interp.phase_b_analysis import resample_matrix

THR = 0.027785714285713504
NARROW = Path("results/remote_clean_flow_phase_b_dev_v1_e965eb211254b0c24d84475782091beb30d11f99fd328ee49f26a6c403d26a4f")
WIDE = Path("results/phase_b_wide60m_v1/clean_flow_phase_b_dev_wide60m_v1_148d23a632e7b50e6534ef7b388987c70bca76dc16498910cfe11a4083e978f0")
ARMS = ["flow_t010_nfe1","flow_t010_nfe3","flow_t010_nfe5","flow_t025_nfe1","flow_t025_nfe3",
        "flow_t025_nfe5","flow_t050_nfe1","flow_t050_nfe3","flow_t050_nfe5"]
VECTORS = ("allegations","dungeon","locations_addresses","illicit_drugs",
           "law_enforcement_officials","same_sex_marriage","borders","sports_awards")

def rows(p): return [json.loads(l) for l in open(p)]

def analyse(rel, arm, rule):
    flow = [r for r in rows(rel/"flow"/f"{arm}.jsonl") if not r["is_stress"]]
    add = [r for r in rows(rel/"baselines"/"additive.rescored.jsonl") if not r["is_stress"]]
    curves = collections.defaultdict(list)
    for r in add:
        curves[(r["vector"], r["prompt_id"], r["generation_seed"])].append(r)
    for k in curves: curves[k].sort(key=lambda r: r["alpha"])
    cmean = collections.defaultdict(list)
    for r in add: cmean[(r["vector"], r["alpha_hex"])].append(r["metrics"]["repetition_rate"])
    cmean = {k: sum(v)/len(v) for k, v in cmean.items()}
    fmean = collections.defaultdict(list)
    for r in flow: fmean[(r["vector"], r["alpha_hex"])].append(r["metrics"]["repetition_rate"])
    fmean = {k: sum(v)/len(v) for k, v in fmean.items()}

    per_vec = collections.defaultdict(list)
    n_sup = 0
    for r in flow:
        pts = curves[(r["vector"], r["prompt_id"], r["generation_seed"])]
        t = r["geometry"]["realized_projection_mean"]
        br = next(((a, b) for a, b in zip(pts, pts[1:]) if a["alpha"] <= t <= b["alpha"]), None)
        if br is None: continue
        lo, hi = br
        if rule == "row":
            bad = (r["metrics"]["repetition_rate"] > THR or
                   lo["metrics"]["repetition_rate"] > THR or hi["metrics"]["repetition_rate"] > THR)
        else:
            bad = (fmean[(r["vector"], r["alpha_hex"])] > THR or
                   cmean[(lo["vector"], lo["alpha_hex"])] > THR or
                   cmean[(hi["vector"], hi["alpha_hex"])] > THR)
        if bad: continue
        span = hi["alpha"] - lo["alpha"]
        w = 0.0 if span == 0 else (t - lo["alpha"]) / span
        matched = (1 - w) * lo["metrics"]["nll"] + w * hi["metrics"]["nll"]
        per_vec[r["vector"]].append(r["metrics"]["nll"] - matched)
        n_sup += 1
    means = np.array([np.mean(per_vec[v]) for v in VECTORS])
    return float(means.mean()), means, n_sup

matrix = resample_matrix(VECTORS, seed=20260813, n_resamples=10000)

def ci(means):
    draws = means[matrix].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

print("matched-projection NLL vs additive (>0 = flow worse)\n")
print(f"{'arm':<16}{'narrow row':>12}{'narrow cell':>13}{'wide row':>12}{'wide cell':>12}{'max shift':>11}")
worst = 0.0
for arm in ARMS:
    nr, _, _ = analyse(NARROW, arm, "row")
    nc, _, _ = analyse(NARROW, arm, "cell")
    wr, _, _ = analyse(WIDE, arm, "row")
    wc, _, _ = analyse(WIDE, arm, "cell")
    shift = max(abs(nr-nc), abs(wr-wc)); worst = max(worst, shift)
    print(f"{arm:<16}{nr:>12.4f}{nc:>13.4f}{wr:>12.4f}{wc:>12.4f}{shift:>11.4f}")
print(f"\nlargest absolute shift from the rule choice: {worst:.4f} nats")

print("\nprimary cell t=0.50 NFE=1, with CIs:")
for name, rel in (("narrow", NARROW), ("wide", WIDE)):
    for rule in ("row", "cell"):
        m, means, n = analyse(rel, "flow_t050_nfe1", rule)
        lo, hi = ci(means)
        print(f"  {name:<7}{rule:<6} mean {m:.4f}  CI [{lo:.4f}, {hi:.4f}]  supported {n}  signs+ {int((means>0).sum())}/8")

print("\nnarrow->wide paired difference at the primary cell:")
for rule in ("row", "cell"):
    _, nm, _ = analyse(NARROW, "flow_t050_nfe1", rule)
    _, wm, _ = analyse(WIDE, "flow_t050_nfe1", rule)
    d = wm - nm
    lo, hi = ci(d)
    print(f"  {rule:<6} mean {d.mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  resolved {'yes' if hi < 0 else 'no'}")
