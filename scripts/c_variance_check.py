"""Do the Experiment C directions carry more variance than a random axis?

D found that alignment with the natural trajectory tracks how much variance a
direction carries. C compared its SAE directions against random axes and found no
difference. If the SAE directions carry about as much variance as a random axis,
then C's null is a variance statement, not a statement about those directions.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from interp.activations import load_activations
from interp.conditional_flow import load_training_direction_pool
from interp.curvature import CURVATURE_SPEC as S

ds = load_activations("resid7_fw_val_1024k_v1", Path("/workspace/data/fineweb_activations"))
rng = np.random.default_rng(S.row_seed)
rows = np.sort(rng.choice(ds.array.shape[0], size=S.n_rows, replace=False))
X = np.array(ds.array[rows], dtype=np.float64)
Xc = X - X.mean(0)
total = float(np.sum(np.var(Xc, axis=0)))

pool = load_training_direction_pool(Path("data/direction_pools/training_only_rank256_v1.pt"))
picked = np.sort(
    np.random.default_rng(S.direction_seed).choice(
        len(pool), size=S.n_directions, replace=False
    )
)
V = pool.directions[picked].double().cpu().numpy()
V /= np.linalg.norm(V, axis=1, keepdims=True)

qr = np.random.default_rng(S.random_direction_seed)
Q = qr.normal(size=(S.n_random_directions, X.shape[1]))
Q /= np.linalg.norm(Q, axis=1, keepdims=True)

var_v = np.var(Xc @ V.T, axis=0)
var_q = np.var(Xc @ Q.T, axis=0)
_, sv, vt = np.linalg.svd(Xc, full_matrices=False)
eig = (sv**2) / (Xc.shape[0] - 1)

out = {
    "n_rows": int(X.shape[0]),
    "total_variance": total,
    "sae_direction_variance": {
        "mean": float(var_v.mean()),
        "min": float(var_v.min()),
        "max": float(var_v.max()),
    },
    "random_axis_variance": {
        "mean": float(var_q.mean()),
        "min": float(var_q.min()),
        "max": float(var_q.max()),
    },
    "ratio_sae_over_random": float(var_v.mean() / var_q.mean()),  # noqa: E501
    "pc1_variance": float(eig[0]),
    "ratio_pc1_over_sae": float(eig[0] / var_v.mean()),
    "sae_share_of_total_mean": float(var_v.mean() / total),
}
print(json.dumps(out, indent=2))
Path("/workspace/results/c_variance_check.json").write_text(json.dumps(out, indent=2) + "\n")
