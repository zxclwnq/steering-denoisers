"""Build every report figure and table from frozen result artifacts.

Reads only immutable artifacts under ``results/``; trains nothing, generates
nothing, and touches no protected path. Every number in the report comes from
here, so a reviewer can regenerate the whole figure set with one command:

    uv run python scripts/build_report_figures.py

Outputs land in ``report/figures`` (PNG + PDF), ``report/tables`` (CSV/Markdown)
and ``report/data`` (the aggregated series behind each figure, as JSON).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"
DAT = ROOT / "report" / "data"
for d in (FIG, TAB, DAT):
    d.mkdir(parents=True, exist_ok=True)

# Ordered, colour-blind-safe, deliberately not a rainbow: one hue per method
# family, greys for controls, red reserved for "the method that fails".
C = {
    "additive": "#1F5FA8",
    "denoiser": "#C1651A",
    "shrink": "#7B8B9B",
    "flow": "#2C6449",
    "flow_alt": "#7FA8C9",
    "clamp": "#14202B",
    "tangent": "#A6432E",
    "projected": "#8E6BA8",
    "clean": "#4C8C6B",
    "grid": "#D2DAE3",
    "faint": "#B4C0CD",
}
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 240,
    "savefig.bbox": "tight",
    "figure.autolayout": True,   # keeps side-by-side panel titles from colliding
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "600",
    "axes.labelsize": 9,
    "axes.edgecolor": "#48596B",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": C["grid"],
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

PHASE_B = RES / "phase_b_wide60m_v1"
PHASE_B_NARROW = (
    RES / "phase_b_narrow16m_fw32m_v1"
    / ("clean_flow_phase_b_dev_narrow16m_fw32m_v1_"
       "c64e524dc339ab37a63f046ab6e96de60d4e6111a3d373bff75d5f9d986ea190")
)
QUANTILES = ("q50", "q75", "q90", "q95", "q99")
QLABEL = {"q50": "p50", "q75": "p75", "q90": "p90", "q95": "p95", "q99": "p99"}


def save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  figure -> report/figures/{name}.png|pdf")


def dump(name: str, payload) -> None:
    (DAT / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")


def write_table(name: str, header: list[str], rows: list[list], note: str = "") -> None:
    csv = [",".join(header)] + [",".join(str(c) for c in r) for r in rows]
    (TAB / f"{name}.csv").write_text("\n".join(csv) + "\n")
    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join("---:" if i else ":---" for i in range(len(header))) + "|"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    if note:
        md += ["", f"_{note}_"]
    (TAB / f"{name}.md").write_text("\n".join(md) + "\n")
    print(f"  table  -> report/tables/{name}.csv|md")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def read_json(path: Path) -> dict:
    """Read one artifact JSON."""

    return json.loads(Path(path).read_text())


def load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


TANGENT_DEV = RES / "tangent_dev_posthoc_v1" / "tangent_dev.jsonl"


def tangent_dev_arm() -> list[dict]:
    """Post-hoc `clamp + tangent flow` arm on the frozen DEV protocol.

    Same prompts, alpha grid, seeds, decoding and metrics as the frozen arms.
    Not part of any preregistered protocol; descriptive only.
    """

    return load_rows(TANGENT_DEV)


def phase_b_arm(name: str, base: Path = PHASE_B) -> list[dict]:
    """One Phase-B arm's per-generation rows."""

    for candidate in (base / "baselines" / f"{name}.rescored.jsonl",
                      base / "flow" / f"{name}.jsonl"):
        if candidate.exists():
            return load_rows(candidate)
    raise FileNotFoundError(name)


def by_alpha(rows: list[dict], metric: str, *, exclude_stress: bool = False):
    """Vector-level means per alpha, then the across-vector mean and CI.

    The bootstrap unit is the steering VECTOR, never the individual generation:
    30 continuations share one vector and are not independent observations.
    """

    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if exclude_stress and r["is_stress"]:
            continue
        cells[r["alpha_hat"]][r["feature"]].append(r["metrics"][metric])
    alphas = sorted(cells)
    per_vec = {a: {f: float(np.mean(v)) for f, v in cells[a].items()} for a in alphas}
    features = sorted({f for a in alphas for f in per_vec[a]})
    mean, lo, hi = [], [], []
    rng = np.random.default_rng(20260813)
    for a in alphas:
        vals = np.array([per_vec[a][f] for f in features if f in per_vec[a]])
        mean.append(float(vals.mean()))
        draws = vals[rng.integers(0, len(vals), size=(4000, len(vals)))].mean(axis=1)
        lo.append(float(np.quantile(draws, 0.025)))
        hi.append(float(np.quantile(draws, 0.975)))
    return alphas, np.array(mean), np.array(lo), np.array(hi), per_vec, features


def clean_nll(rows: list[dict]) -> float:
    """alpha = 0 is the unmodified model, so it defines the clean reference."""

    vals = [r["metrics"]["nll"] for r in rows if r["alpha_hat"] == 0.0]
    return float(np.mean(vals))


# --------------------------------------------------------------------------
# Figure 1 — additive steering Pareto
# --------------------------------------------------------------------------


def figure1() -> dict:
    rows = phase_b_arm("additive")
    base = clean_nll(rows)
    a, q_m, q_lo, q_hi, q_pv, feats = by_alpha(rows, "nll")
    _, c_m, c_lo, c_hi, c_pv, _ = by_alpha(rows, "lexicon_score")
    _, s_m, s_lo, s_hi, _, _ = by_alpha(rows, "sae_act_target")

    quality = -(q_m - base)
    ql, qh = -(q_hi - base), -(q_lo - base)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    ax = axes[0]
    for f in feats:
        fx = [-(q_pv[x][f] - base) for x in a if f in q_pv[x]]
        fy = [c_pv[x][f] for x in a if f in c_pv[x]]
        ax.plot(fx, fy, color=C["faint"], lw=0.7, alpha=0.55, zorder=1)
    ax.errorbar(quality, c_m, xerr=[quality - ql, qh - quality],
                yerr=[c_m - c_lo, c_hi - c_m], color=C["additive"], lw=1.8,
                marker="o", ms=4.5, capsize=2, elinewidth=0.8, zorder=3,
                label="additive (aggregate, 95% CI)")
    for x, y, al in zip(quality, c_m, a, strict=True):
        if al in (0.0, 0.3, 0.5, 1.0, 2.0):
            dx, dy = {0.0: (7, -3), 0.3: (7, -10), 0.5: (5, 7),
                      1.0: (-4, -13), 2.0: (6, 4)}[al]
            ax.annotate(f"α̂={al:g}", (x, y), textcoords="offset points",
                        xytext=(dx, dy), fontsize=7.2, color=C["additive"])
    # Clip to the aggregate's own range: a couple of individual vectors run far
    # off to the left and would otherwise squash the curve that carries the story.
    ax.set_xlim(min(ql.min(), -0.2) * 1.12, max(qh.max(), 0.2) + 0.35)
    ax.set_ylim(-0.004, max(c_hi.max(), 0.05) * 1.18)
    ax.set_xlabel("Качество  =  −ΔNLL   (правее лучше)")
    ax.set_ylabel("Лексиконный concept score   (выше лучше)")
    ax.set_title("A. Компромисс качество / концепт (текстовая метрика)")
    ax.legend(handles=[Line2D([], [], color=C["additive"], marker="o",
                              label="additive (среднее, 95% ДИ)"),
                       Line2D([], [], color=C["faint"], lw=0.8,
                              label="отдельные направления")],
              loc="upper left")

    ax = axes[1]
    ax.errorbar(quality, s_m, xerr=[quality - ql, qh - quality],
                yerr=[s_m - s_lo, s_hi - s_m], color=C["additive"], lw=1.8,
                marker="s", ms=4.5, capsize=2, elinewidth=0.8)
    ax.set_xlim(min(ql.min(), -0.2) * 1.12, max(qh.max(), 0.2) + 0.35)
    ax.set_xlabel("Качество  =  −ΔNLL   (правее лучше)")
    ax.set_ylabel("Активация целевой SAE-фичи")
    ax.set_title("B. То же, механистическая метрика концепта")
    for x, y, al in zip(quality, s_m, a, strict=True):
        if al in (0.4, 1.0):
            ax.annotate(f"α̂={al:g}", (x, y), textcoords="offset points",
                        xytext=(7, 4), fontsize=7.2, color=C["additive"])

    save(fig, "fig01_additive_pareto")

    payload = {"clean_nll": base, "alpha_hat": a, "delta_nll": (q_m - base).tolist(),
               "lexicon": c_m.tolist(), "sae_target": s_m.tolist(),
               "lexicon_ci": [c_lo.tolist(), c_hi.tolist()]}
    dump("fig01_additive_pareto", payload)

    write_table("t01_additive_sweep",
                ["alpha_hat", "ΔNLL", "lexicon", "SAE target", "dist_1", "dist_2",
                 "dist_3", "repetition"],
                [[f"{al:g}", f"{q_m[i]-base:+.4f}", f"{c_m[i]:.4f}", f"{s_m[i]:.4f}",
                  f"{by_alpha(rows,'dist_1')[1][i]:.4f}",
                  f"{by_alpha(rows,'dist_2')[1][i]:.4f}",
                  f"{by_alpha(rows,'dist_3')[1][i]:.4f}",
                  f"{by_alpha(rows,'repetition_rate')[1][i]:.4f}"]
                 for i, al in enumerate(a)],
                note=f"Чистая NLL = {base:.4f}. α̂ = α/E‖h‖, E‖h‖ = 88.76. "
                     "α̂ ∈ {1.5, 2.0} — стресс-точки вне основной сетки.")
    return payload


# --------------------------------------------------------------------------
# Figure 2 — main comparisons
# --------------------------------------------------------------------------


def figure2() -> dict:
    add = phase_b_arm("additive")
    nai = phase_b_arm("naive")
    shr = phase_b_arm("shrinkage_k080")
    flw = phase_b_arm("flow_t010_nfe1")
    base = clean_nll(add)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    ax = axes[0]
    tng = tangent_dev_arm()
    series = [("additive  h+αv", add, C["additive"], "o", "-"),
              ("денойзер из задания  D(h+αv)", nai, C["denoiser"], "^", "-"),
              ("сжатие κ=0.8 (контроль)", shr, C["shrink"], "v", "--"),
              ("модель с обычным шумом 60M (t=0.10, NFE 1)", flw, C["flow"], "s", "-"),
              ("клэмп + модель с ортогональным шумом", tng, C["tangent"], "D", "-")]
    out = {}
    for label, rows, col, mk, ls in series:
        a, q, _, _, _, _ = by_alpha(rows, "nll")
        _, c, clo, chi, _, _ = by_alpha(rows, "lexicon_score")
        x = -(q - base)
        ax.plot(x, c, color=col, lw=1.7, marker=mk, ms=4.2, ls=ls, label=label)
        ax.fill_between(x, clo, chi, color=col, alpha=0.10, lw=0)
        out[label] = {"alpha_hat": a, "delta_nll": (q - base).tolist(),
                      "lexicon": c.tolist()}
    ax.set_xlabel("Качество  =  −ΔNLL   (правее лучше)")
    ax.set_ylabel("Лексиконный concept score")
    ax.set_title("2A. DEV-направления, единая α-сетка")
    ax.legend(loc="upper right", fontsize=6.9)

    # 2B: natural-support family — different x/y definition, kept separate.
    ax = axes[1]
    cn = read_json(RES / "constrained_naturalization_v1" / "constrained_naturalization.json")
    t2 = read_json(RES / "tangent_t2_v1" / "tangent_naturalization.json")
    xs = np.arange(len(QUANTILES))
    clamp = [cn["arms"][f"{q}_clamp_only"]["mean_delta_lm_vs_clean"] for q in QUANTILES]
    proj = [cn["arms"][f"{q}_t0.10_nfe1_projected"]["mean_delta_lm_vs_clean"] for q in QUANTILES]
    tang = [t2["arms"][f"{q}_t0.10_nfe1_tangent_flow"]["mean_delta_lm_vs_clean"] for q in QUANTILES]
    w = 0.26
    ax.bar(xs - w, clamp, w, color=C["clamp"], label="жёсткий клэмп (NFE 0)")
    ax.bar(xs, proj, w, color=C["projected"], label="клэмп + проецируемая условная модель")
    ax.bar(xs + w, tang, w, color=C["tangent"], label="клэмп + модель с ортогональным шумом")
    ax.set_xticks(xs)
    ax.set_xticklabels([QLABEL[q] for q in QUANTILES])
    ax.set_xlabel("Целевая координата (квантиль естественного диапазона)")
    ax.set_ylabel("ΔLM относительно чистой модели   (ниже лучше)")
    ax.set_title("2B. Natural-support: точная координата, NFE 1")
    ax.legend(loc="upper left", fontsize=7.4)
    out["natural_support"] = {"quantiles": list(QUANTILES), "clamp": clamp,
                              "projected": proj, "tangent": tang}

    save(fig, "fig02_method_comparison")
    dump("fig02_method_comparison", out)
    return out


# --------------------------------------------------------------------------
# Figure 3 — matched realised steering strength
# --------------------------------------------------------------------------


def figure3() -> dict:
    add = phase_b_arm("additive")
    nai = phase_b_arm("naive")
    shr = phase_b_arm("shrinkage_k080")
    base = clean_nll(add)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    out = {}
    for label, rows, col, mk in (("additive", add, C["additive"], "o"),
                                 ("денойзер D(h+αv)", nai, C["denoiser"], "^"),
                                 ("сжатие κ=0.8", shr, C["shrink"], "v")):
        # Stress points (α̂ = 1.5, 2.0) sit outside the frozen main grid and are
        # deep in the over-steering collapse, where the curve doubles back and
        # obscures the comparison this figure exists to make.
        a, q, _, _, _, _ = by_alpha(rows, "nll", exclude_stress=True)
        _, c, _, _, _, _ = by_alpha(rows, "lexicon_score", exclude_stress=True)
        ax.plot(a, c, color=col, lw=1.6, marker=mk, ms=4.2, label=label)
        out[label] = {"alpha_hat": a, "lexicon": c.tolist(),
                      "delta_nll": (q - base).tolist()}
    ax.set_xlabel("Номинальная сила стиринга  α̂")
    ax.set_ylabel("Лексиконный concept score")
    ax.set_title("A. Концепт как функция НОМИНАЛЬНОЙ α")
    ax.legend(loc="upper left")

    ax = axes[1]
    for label in out:
        col = {"additive": C["additive"], "денойзер D(h+αv)": C["denoiser"],
               "сжатие κ=0.8": C["shrink"]}[label]
        mk = {"additive": "o", "денойзер D(h+αv)": "^", "сжатие κ=0.8": "v"}[label]
        ax.plot(out[label]["lexicon"], [-d for d in out[label]["delta_nll"]],
                color=col, lw=1.6, marker=mk, ms=4.2, label=label)
    ax.set_xlabel("Достигнутый concept score  (сопоставление по РЕАЛИЗОВАННОМУ уровню)")
    ax.set_ylabel("Качество  =  −ΔNLL")
    ax.set_title("B. При равном концепте преимущество исчезает")
    ax.legend(loc="lower left")

    save(fig, "fig03_matched_strength")
    dump("fig03_matched_strength", out)
    return out


# --------------------------------------------------------------------------
# Figure 4 — the assignment's denoiser really denoises
# --------------------------------------------------------------------------


def figure4() -> dict:
    # The one figure in this report not recomputed from a result artifact. The
    # denoiser weights (SHA 963b4dda...) are absent from the repository, so these
    # values come from the transcribed record. They are parsed back out of that
    # record and cross-checked against the literals below, so report and record
    # cannot silently diverge.
    sigma = [0.25, 0.50, 1.00, 2.00]
    corrupted = [0.0451, 0.1997, 0.8722, 2.7236]
    denoised = [0.0307, 0.1138, 0.4239, 1.1619]

    record = read_json(RES / "gaussian_denoiser_v1" / "reconstruction_record.json")
    parsed = {r["corruption_scale"]: (r["corrupted_delta_lm"], r["denoised_delta_lm"])
              for r in record["reconstruction"]}
    for s_val, c_val, d_val in zip(sigma, corrupted, denoised, strict=True):
        if parsed.get(s_val) != (c_val, d_val):
            raise ValueError(
                f"denoiser figure disagrees with the frozen record at sigma={s_val}: "
                f"record says {parsed.get(s_val)}, figure says {(c_val, d_val)}"
            )

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    x = np.arange(len(sigma))
    ax.bar(x - 0.19, corrupted, 0.38, color=C["shrink"], label="повреждённая  h+ε")
    ax.bar(x + 0.19, denoised, 0.38, color=C["denoiser"], label="восстановленная  D(h+ε)")
    ax.axhline(0, color=C["clean"], lw=1.4, ls="--", label="чистая активация (ΔLM = 0)")
    for i, (cc, dd) in enumerate(zip(corrupted, denoised, strict=True)):
        ax.annotate(f"−{100*(1-dd/cc):.0f}%", (i, max(cc, dd)), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=7.6, color=C["denoiser"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"σ = {s:g}" for s in sigma])
    ax.set_ylabel("ΔLM относительно чистой модели   (ниже лучше)")
    ax.legend()
    save(fig, "fig04_denoiser_ability")
    out = {"sigma": sigma, "corrupted_delta_lm": corrupted, "denoised_delta_lm": denoised,
           "source": "results/gaussian_denoiser_v1/reconstruction_record.json",
           "checkpoint_sha256": "963b4dda162d60f1064b47979843c6ca99b733a1837196abc758432e7770c583"}
    dump("fig04_denoiser_ability", out)
    write_table("t04_denoiser_reconstruction",
                ["σ", "corrupted ΔLM", "denoised ΔLM", "восстановлено"],
                [[f"{s:g}", f"{c:.4f}", f"{d:.4f}", f"{100*(1-d/c):.0f}%"]
                 for s, c, d in zip(sigma, corrupted, denoised, strict=True)],
                note="Перенесено из замороженной записи; веса денойзера отсутствуют локально.")
    return out


# --------------------------------------------------------------------------
# Figure 5 — reconstruction quality does not transfer to steering
# --------------------------------------------------------------------------


def figure5() -> dict:
    rep = {a: read_json(RES / "flow_scaling_2x2_v2" / "reports" / f"{a}.json")
           for a in ("narrow16m_fw32m", "wide60m_fw32m")}
    vloss = {a: rep[a]["validation_flow_loss"] for a in rep}

    narrow = read_json(PHASE_B_NARROW / "analysis.json")
    wide = read_json(PHASE_B / "analysis.json")

    def eq_alpha_nll(an):
        arm = an["arms"]["flow_t010_nfe1"]["equal_alpha_vs_additive"]["nll"]
        return arm["mean"], arm["ci_lower"], arm["ci_upper"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax = axes[0]
    names = ["16M (узкий)", "60M (широкий)"]
    keys = ["narrow16m_fw32m", "wide60m_fw32m"]
    vals = [vloss[k].get("val_flow_mse", vloss[k]) if isinstance(vloss[k], dict) else vloss[k]
            for k in keys]
    vals = [v if isinstance(v, (int, float)) else float(list(v.values())[0]) for v in vals]
    ax.bar(names, vals, 0.5, color=[C["flow_alt"], C["flow"]])
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8.5)
    ax.set_ylabel("Валидационный flow-MSE   (ниже лучше)")
    ax.set_title("A. Модель с обычным шумом 60M восстанавливает заметно лучше")

    ax = axes[1]
    m_n, lo_n, hi_n = eq_alpha_nll(narrow)
    m_w, lo_w, hi_w = eq_alpha_nll(wide)
    ax.bar(names, [m_n, m_w], 0.5, color=[C["flow_alt"], C["flow"]],
           yerr=[[m_n - lo_n, m_w - lo_w], [hi_n - m_n, hi_w - m_w]],
           capsize=4, error_kw={"lw": 1, "ecolor": "#48596B"})
    ax.axhline(0, color=C["clamp"], lw=1.2)
    ax.set_ylabel("ΔNLL против additive при равной α   (ниже лучше)")
    ax.set_title("B. …но выигрыша в стиринге это не даёт")

    save(fig, "fig05_capacity_transfer")
    out = {"val_flow_mse": dict(zip(keys, vals, strict=True)),
           "equal_alpha_nll_vs_additive": {"narrow16m": [m_n, lo_n, hi_n],
                                           "wide60m": [m_w, lo_w, hi_w]}}
    dump("fig05_capacity_transfer", out)
    return out


# --------------------------------------------------------------------------
# Figure 6 — conditional coordinate control
# --------------------------------------------------------------------------


def figure6() -> dict:
    d = read_json(RES / "natural_support_v1" / "natural_support_controllability.json")
    tstarts = [0.50, 0.75, 0.90]
    cols = {0.50: C["flow_alt"], 0.75: C["flow"], 0.90: C["projected"]}

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    out = {}
    for t in tstarts:
        xs, ys = [], []
        for q in QUANTILES:
            a = d["arms"][f"t{t:.2f}_nfe1_{q}_correct"]
            xs.append(a["mean_requested_displacement"])
            ys.append(a["mean_realised_displacement"])
        ax.plot(xs, ys, marker="o", ms=4.5, lw=1.7, color=cols[t],
                label=f"t_start = {t:.2f}")
        out[f"t{t:.2f}"] = {"requested": xs, "realised": ys}
    lim = max(max(v["requested"]) for v in out.values()) * 1.08
    ax.plot([0, lim], [0, lim], color=C["clamp"], lw=1.1, ls=":", label="идеальный контроль")
    ax.set_xlabel("Запрошенное смещение координаты  c_req − c₀")
    ax.set_ylabel("Реализованное смещение  c_real − c₀")
    ax.set_title("A. Запрошенная против реализованной координаты", fontsize=9.5)
    ax.legend(loc="upper left")

    ax = axes[1]
    # Mean across the five target quantiles rather than one arbitrarily chosen
    # cell: the slope varies with the quantile (e.g. 0.906-0.953 at t = 0.90)
    # and picking a single one would be an unmotivated choice.
    slopes, damage, sranges = [], [], []
    for t in tstarts:
        sl = [d["arms"][f"t{t:.2f}_nfe1_{q}_correct"]["calibration"]["slope"]
              for q in QUANTILES]
        dl = [d["arms"][f"t{t:.2f}_nfe1_{q}_correct"]["mean_delta_lm"] for q in QUANTILES]
        slopes.append(float(np.mean(sl)))
        damage.append(float(np.mean(dl)))
        sranges.append([float(min(sl)), float(max(sl))])
    ax.plot(tstarts, slopes, marker="o", ms=5, lw=1.8, color=C["flow"],
            label="наклон контроля координаты")
    ax.fill_between(tstarts, [r[0] for r in sranges], [r[1] for r in sranges],
                    color=C["flow"], alpha=0.14, lw=0)
    ax.set_xlabel("t_start")
    ax.set_ylabel("Наклон контроля (1.0 = идеал)", color=C["flow"])
    ax.tick_params(axis="y", labelcolor=C["flow"])
    ax2 = ax.twinx()
    ax2.plot(tstarts, damage, marker="s", ms=5, lw=1.8, color=C["tangent"], ls="--",
             label="ΔLM (цена)")
    ax2.set_ylabel("ΔLM реконструкции   (ниже лучше)", color=C["tangent"])
    ax2.tick_params(axis="y", labelcolor=C["tangent"])
    ax2.grid(False)
    ax.set_title("B. Точное управление требует высокой ошибки реконструкции",
                 fontsize=9.5)
    ax.legend(handles=[Line2D([], [], color=C["flow"], marker="o", label="наклон контроля"),
                       Line2D([], [], color=C["tangent"], marker="s", ls="--", label="ΔLM")],
              loc="upper left")

    save(fig, "fig06_conditional_control")
    out["slopes"] = dict(zip([f"{t:.2f}" for t in tstarts], slopes, strict=True))
    out["delta_lm"] = dict(zip([f"{t:.2f}" for t in tstarts], damage, strict=True))
    out["slope_range"] = dict(zip([f"{t:.2f}" for t in tstarts], sranges, strict=True))
    dump("fig06_conditional_control", out)
    return out


# --------------------------------------------------------------------------
# Figure 7 — hard clamp is cheap inside natural support
# --------------------------------------------------------------------------


def figure7() -> dict:
    t2 = read_json(RES / "tangent_t2_v1" / "tangent_naturalization.json")
    clamp = [t2["arms"][f"{q}_clamp_only"]["mean_delta_lm_vs_clean"] for q in QUANTILES]

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.bar([QLABEL[q] for q in QUANTILES], clamp, 0.55, color=C["clamp"])
    for i, v in enumerate(clamp):
        ax.annotate(f"+{v:.5f}", (i, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    ax.margins(y=0.14)          # headroom so the p99 label clears the axes edge
    ax.set_xlabel("Целевая координата (квантиль естественного диапазона)")
    ax.set_ylabel("ΔLM относительно чистой модели")
    save(fig, "fig07_hard_clamp_cost")
    out = {"quantiles": list(QUANTILES), "clamp_delta_lm": clamp}
    dump("fig07_hard_clamp_cost", out)
    return out


# --------------------------------------------------------------------------
# Figures 8 / 9 / 10 — the tangent branch
# --------------------------------------------------------------------------


def figure8() -> dict:
    t1 = read_json(RES / "tangent_t1_v1" / "evaluation" / "tangent_reconstruction.json")
    g = t1["t1_gate"]
    cells = [("t0.25", "0.25"), ("t0.50", "0.50"), ("t0.75", "0.75")]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax = axes[0]
    xs = np.arange(len(cells))
    corr = [t1["arms"][f"{c}_corrupted"]["delta_lm_vs_clean"] for c, _ in cells]
    rec = [t1["arms"][f"{c}_nfe1_tangent"]["delta_lm_vs_clean"] for c, _ in cells]
    ax.bar(xs - 0.19, corr, 0.38, color=C["shrink"],
           label="активация после ортогонального повреждения")
    ax.bar(xs + 0.19, rec, 0.38, color=C["flow"], label="восстановленная (NFE 1)")
    ax.axhline(0, color=C["clean"], lw=1.3, ls="--", label="чистая активация")
    for i, (c, _) in enumerate(cells):
        rf = t1["arms"][f"{c}_nfe1_tangent"]["recovered_fraction"]
        ax.annotate(f"{rf:.1%}", (i, corr[i]), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8, color=C["flow"], fontweight="600")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"t_start = {t}" for _, t in cells])
    ax.set_ylabel("ΔLM относительно чистой модели")
    ax.set_title("A. Восстановление своей же задачи (подпись — доля восстановленного)")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ci = t1["arms"][g["primary_cell"]]["paired_delta_nll_vs_corrupted"]
    ax.axvline(0, color=C["clamp"], lw=1.2)
    ax.errorbar([ci["mean"]], [0], xerr=[[ci["mean"] - ci["ci_lower"]],
                                         [ci["ci_upper"] - ci["mean"]]],
                fmt="o", ms=9, color=C["flow"], capsize=5, elinewidth=1.6)
    ax.annotate(f"{ci['mean']:+.4f}\n[{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]",
                (ci["mean"], 0), textcoords="offset points", xytext=(0, 24),
                ha="center", fontsize=9, color=C["flow"])
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlim(min(ci["ci_lower"] * 1.25, -0.05), 0.12)
    ax.set_xlabel("Парный ΔNLL против повреждённого контроля  (отрицательное = восстанавливает)")
    passed = "выполнен" if g["verdict"] == "PASS" else "не выполнен"
    ax.set_title(f"B. Критерий T1 {passed} "
                 f"($t$ = {g['primary_t_start']:.2f}, NFE {g['primary_nfe']})")

    save(fig, "fig08_t1_reconstruction")
    out = {
        "gate": g,
        "ci": ci,
        "cells": {
            c: {
                "corrupted": corr[i],
                "reconstructed": rec[i],
                "recovered_fraction":
                    t1["arms"][f"{c}_nfe1_tangent"]["recovered_fraction"],
            }
            for i, (c, _) in enumerate(cells)
        },
    }
    dump("fig08_t1_reconstruction", out)
    return out


def figure9() -> dict:
    t2 = read_json(RES / "tangent_t2_v1" / "tangent_naturalization.json")
    v = t2["t2_experiment_verdict"]
    cell = t2["arms"][v["primary_cell"]]
    per_q = cell["per_quantile_paired_delta_nll"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2),
                            gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    vals = [per_q[q] for q in QUANTILES]
    ax.bar([QLABEL[q] for q in QUANTILES], vals, 0.55, color=C["tangent"])
    ax.axhline(0, color=C["clamp"], lw=1.4)
    for i, val in enumerate(vals):
        ax.annotate(f"{val:+.5f}", (i, val), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    ax.set_ylabel("NLL(с ортогональной коррекцией) − NLL(клэмп)")
    ax.set_xlabel("Целевая координата")
    ax.set_title("A. Каждый квантиль хуже клэмпа")
    ax.annotate("выше нуля = метод вредит", (0.02, 0.93), xycoords="axes fraction",
                fontsize=8, color=C["tangent"])

    ax = axes[1]
    ci = cell["primary_paired_delta_nll_vs_clamp"]
    ax.axvline(0, color=C["clamp"], lw=1.4)
    ax.errorbar([ci["mean"]], [0], xerr=[[ci["mean"] - ci["ci_lower"]],
                                         [ci["ci_upper"] - ci["mean"]]],
                fmt="o", ms=9, color=C["tangent"], capsize=5, elinewidth=1.6)
    ax.annotate(f"{ci['mean']:+.6f}\n[{ci['ci_lower']:+.6f}, {ci['ci_upper']:+.6f}]",
                (ci["mean"], 0), textcoords="offset points", xytext=(0, 26),
                ha="center", fontsize=9, color=C["tangent"])
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlim(-0.004, 0.014)
    ax.set_xlabel("Объединённый парный ΔNLL  (равный вес квантилей)")
    passed = "пройден" if v["verdict"] == "PASS" else "не пройден"
    ax.set_title(f"B. Основной тест T2 {passed}")

    save(fig, "fig09_t2_naturalization")
    out = {"verdict": v, "per_quantile": per_q, "ci": ci,
           "fraction_directions_negative": cell["fraction_directions_negative"],
           "lovo": [cell["lovo_paired_delta_nll_min"], cell["lovo_paired_delta_nll_max"]]}
    dump("fig09_t2_naturalization", out)
    return out


def figure10() -> dict:
    t2 = read_json(RES / "tangent_t2_v1" / "tangent_naturalization.json")
    cn = read_json(RES / "constrained_naturalization_v1" / "constrained_naturalization.json")

    tan_x, tan_y = [], []
    for t in ("0.10", "0.25", "0.50"):
        xs = [t2["arms"][f"{q}_t{t}_nfe1_tangent_flow"]["orthogonal_correction_norm_mean"]
              for q in QUANTILES]
        ys = [t2["arms"][f"{q}_t{t}_nfe1_tangent_flow"]["primary_paired_delta_nll_vs_clamp"]["mean"]
              for q in QUANTILES]
        tan_x.append(float(np.mean(xs)))
        tan_y.append(float(np.mean(ys)))
    iso_x, iso_y = [], []
    for t in ("0.10", "0.25", "0.50"):
        xs = [cn["arms"][f"{q}_t{t}_nfe1_projected"]["correction"]["orthogonal_norm_mean"]
              for q in QUANTILES]
        ys = [cn["arms"][f"{q}_t{t}_nfe1_projected"]["paired_delta_nll_vs_clamp"]["mean"]
              for q in QUANTILES]
        iso_x.append(float(np.mean(xs)))
        iso_y.append(float(np.mean(ys)))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.axhline(0, color=C["clamp"], lw=1.4)
    ax.plot(tan_x, tan_y, marker="o", ms=7, lw=2.0, color=C["tangent"],
            label="модель с ортогональным шумом 16M")
    ax.plot(iso_x, iso_y, marker="s", ms=6, lw=1.8, color=C["projected"], ls="--",
            label="проецируемая условная модель 60M")
    for x, y, t in zip(tan_x, tan_y, ("0.10", "0.25", "0.50"), strict=True):
        ax.annotate(f"t={t}", (x, y), textcoords="offset points", xytext=(8, -3),
                    fontsize=8, color=C["tangent"])
    ax.set_xlabel("Величина ортогональной коррекции  ‖Δh⊥‖")
    ax.set_ylabel("ΔNLL против жёсткого клэмпа   (выше нуля = вредит)")
    ax.legend(loc="upper left", fontsize=8)
    ax.annotate("‖Δh∥‖ ≈ 5.6e−07 — координата\nне двигается, ослабления нет",
                (0.62, 0.20), xycoords="axes fraction", fontsize=8, color="#48596B")
    # Deliberately not causal: the two models differ by 3.7x in capacity, so the
    # curves lying together is an observation, not an isolation of the objective.
    save(fig, "fig10_mechanism")
    out = {"tangent": {"orth_norm": tan_x, "delta_nll": tan_y},
           "isotropic_projected": {"orth_norm": iso_x, "delta_nll": iso_y}}
    dump("fig10_mechanism", out)
    return out


# --------------------------------------------------------------------------
# Figure 11 — the tangent method on the DEV task, and its zero-steering offset
# --------------------------------------------------------------------------


def figure11() -> dict:
    """Post-hoc: the branch's final method measured on the assignment's own task.

    The alpha = 0 cell is the control that matters. There the clamp target is the
    activation's own coordinate, so the clamp is the identity and the flow merely
    "naturalizes" a clean activation. Any NLL change there is a baseline shift,
    not a steering result, and it has to be subtracted before the paired effect
    at nonzero alpha means anything.
    """

    add = phase_b_arm("additive")
    tng = tangent_dev_arm()
    base = clean_nll(add)

    def keyed(rows):
        return {(r["feature"], r["alpha_hat"], r["prompt_id"], r["generation_seed"]): r
                for r in rows}

    A, T = keyed(add), keyed(tng)
    shared = sorted(set(A) & set(T))
    per_vector = defaultdict(list)
    zero = []
    for k in shared:
        delta = T[k]["metrics"]["nll"] - A[k]["metrics"]["nll"]
        if k[1] == 0.0:
            zero.append(delta)
        elif k[1] <= 1.0:
            per_vector[k[0]].append(delta)
    offset = float(np.mean(zero))
    vectors = sorted(per_vector)
    paired = np.array([float(np.mean(per_vector[v])) for v in vectors])
    rng = np.random.default_rng(20260813)
    draws = paired[rng.integers(0, len(paired), size=(10000, len(paired)))].mean(axis=1)
    lo, hi = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    ax = axes[0]
    a, q_add, _, _, _, _ = by_alpha(add, "nll", exclude_stress=True)
    _, c_add, _, _, _, _ = by_alpha(add, "lexicon_score", exclude_stress=True)
    _, q_tng, _, _, _, _ = by_alpha(tng, "nll", exclude_stress=True)
    _, c_tng, _, _, _, _ = by_alpha(tng, "lexicon_score", exclude_stress=True)
    ax.plot(-(q_add - base), c_add, color=C["additive"], lw=1.8, marker="o", ms=4.4,
            label="additive")
    ax.plot(-(q_tng - base), c_tng, color=C["tangent"], lw=1.8, marker="D", ms=4.2,
            label="клэмп + модель с ортогональным шумом")
    ax.set_xlabel("Качество  =  −ΔNLL   (правее лучше)")
    ax.set_ylabel("Лексиконный concept score")
    ax.set_title("A. Метод ложится на кривую additive")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.axvline(0, color=C["clamp"], lw=1.4)
    ax.errorbar([paired.mean()], [1], xerr=[[paired.mean() - lo], [hi - paired.mean()]],
                fmt="D", ms=8, color=C["tangent"], capsize=5, elinewidth=1.5)
    ax.annotate(f"сырой парный эффект\n{paired.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]",
                (paired.mean(), 1), textcoords="offset points", xytext=(0, 18),
                ha="center", fontsize=8, color=C["tangent"])
    ax.errorbar([offset], [0], fmt="o", ms=8, color=C["shrink"], capsize=5)
    ax.annotate(f"смещение при α̂=0 (стиринга нет)\n{offset:+.4f}",
                (offset, 0), textcoords="offset points", xytext=(0, -30),
                ha="center", fontsize=8, color="#48596B")
    corrected = paired.mean() - offset
    ax.errorbar([corrected], [-1], fmt="s", ms=8, color=C["clamp"], capsize=5)
    ax.annotate(f"эффект за вычетом смещения\n{corrected:+.4f}",
                (corrected, -1), textcoords="offset points", xytext=(0, 16),
                ha="center", fontsize=8, color=C["clamp"])
    ax.set_yticks([])
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("ΔNLL против additive   (правее = хуже)")
    ax.set_title("B. Почему смещение при нулевом стиринге важно")

    save(fig, "fig11_tangent_on_dev")
    out = {"paired_mean": float(paired.mean()), "ci": [lo, hi],
           "alpha_zero_offset": offset,
           "offset_corrected": float(corrected),
           "n_paired_cells": len(shared),
           "directions_negative": int((paired < 0).sum()), "n_directions": len(paired),
           "per_direction": dict(zip([str(v) for v in vectors], paired.tolist(), strict=True))}
    dump("fig11_tangent_on_dev", out)
    return out


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def table_distn() -> dict:
    """Dist-1/2/3 and repetition on a shared alpha grid, per the assignment."""

    base_rows = phase_b_arm("additive")
    clean = {m: float(np.mean([r["metrics"][m] for r in base_rows if r["alpha_hat"] == 0.0]))
             for m in ("dist_1", "dist_2", "dist_3", "repetition_rate", "nll")}
    rows = [["чистая модель (α̂=0)", "—", f"{clean['nll']:.4f}", f"{clean['dist_1']:.4f}",
             f"{clean['dist_2']:.4f}", f"{clean['dist_3']:.4f}",
             f"{clean['repetition_rate']:.4f}"]]
    out = {"clean": clean, "arms": {}}
    arms = [("additive", phase_b_arm("additive")),
            ("денойзер из задания D(h+αv)", phase_b_arm("naive")),
            ("сжатие κ=0.8", phase_b_arm("shrinkage_k080")),
            ("модель с обычным шумом 60M t=0.10 NFE1", phase_b_arm("flow_t010_nfe1")),
            ("клэмп + модель с ортогональным шумом", tangent_dev_arm())]
    for label, r in arms:
        arm = label
        for al in (0.3, 0.6, 1.0):
            sub = [x for x in r if abs(x["alpha_hat"] - al) < 1e-9]
            vals = {m: float(np.mean([s["metrics"][m] for s in sub]))
                    for m in ("dist_1", "dist_2", "dist_3", "repetition_rate", "nll")}
            rows.append([label, f"{al:g}", f"{vals['nll']:.4f}", f"{vals['dist_1']:.4f}",
                         f"{vals['dist_2']:.4f}", f"{vals['dist_3']:.4f}",
                         f"{vals['repetition_rate']:.4f}"])
            out["arms"][f"{arm}_a{al:g}"] = vals
    write_table("t02_distn",
                ["метод", "α̂", "NLL", "Dist-1", "Dist-2", "Dist-3", "repetition"],
                rows,
                note="Одинаковые промпты, seeds, декодирование и бюджет токенов для всех "
                     "строк (8 DEV-направлений × 10 промптов × 3 seed). Основная метрика "
                     "качества — NLL; Dist-n и repetition — дешёвые диагностики уровня "
                     "генерации, а не метрика беглости. Тангенциальный арм — post-hoc, "
                     "вне замороженного протокола.")
    dump("t02_distn", out)
    return out


def table_summary() -> None:
    write_table(
        "t03_methods_summary",
        ["Метод", "Вспом. задача решена?", "Точное сохранение концепта?",
         "Выигрыш на Парето?", "Интерпретация"],
        [["Additive h+αv", "—", "—", "baseline", "задаёт компромисс качество↔концепт"],
         ["Гауссов денойзер (задание)", "да", "нет", "нет устойчивого",
          "выигрыш в основном от ослабления"],
         ["Сжатие κ=0.8 / parallel-only", "—", "контролируемо", "нет",
          "воспроизводит большую часть выигрыша денойзера"],
         ["Модель с обычным шумом 16M", "да", "нет", "нет", "реконструкция не переносится"],
         ["Модель с обычным шумом 60M", "лучше", "нет", "нет",
          "рост качества реконструкции не переносится"],
         ["Условный flow", "да", "внутри диапазона", "нет",
          "конфликт управления и реконструкции"],
         ["Жёсткий клэмп", "—", "точное", "сильный baseline",
          "низкая цена внутри естественного диапазона"],
         ["Проецируемая условная модель", "да", "точное", "нет",
          "ортогональные правки вредят"],
         ["Модель с ортогональным шумом 16M", "да, уверенно", "точное",
          "нет (T2 отрицателен)",
          "геометрическое рассогласование не было причиной"],
         ["Та же модель на DEV", "—", "точное", "нет (+0.060 нат)",
          "тот же вывод на задаче из ТЗ; смещение при α=0 усиливает его до +0.130"]],
        note="Заполнено только измеренными утверждениями.")


# --------------------------------------------------------------------------
# Figure 12 — post-stop A: the variance-preserving path reproduces the result
# --------------------------------------------------------------------------


def figure12() -> dict:
    """A is a control, so it gets one figure: does changing the path change anything?

    The two arms are compared at MATCHED SEVERITY, never at equal t. Panel A is
    each arm's own T1 gate; panel B puts the two T2 decisions on one axis, where
    the whole point is that they land on top of each other.
    """

    lin_t1 = read_json(RES / "tangent_t1_v1" / "evaluation" / "tangent_reconstruction.json")
    vp_t1 = read_json(RES / "vp_tangent_a_t1_v1" / "tangent_reconstruction.json")
    lin_t2 = read_json(RES / "tangent_t2_v1" / "tangent_naturalization.json")
    vp_t2 = read_json(RES / "vp_tangent_a_t2_v1" / "tangent_naturalization.json")

    arms = [
        ("линейный путь", lin_t1["t1_gate"], lin_t2["t2_experiment_verdict"], C["tangent"]),
        ("VP путь", vp_t1["t1_gate"], vp_t2["t2_experiment_verdict"], C["projected"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0),
                            gridspec_kw={"width_ratios": [1, 1.3]})

    ax = axes[0]
    xs = np.arange(len(arms))
    vals = [gate["recovered_fraction"] for _, gate, _, _ in arms]
    ax.bar(xs, vals, 0.45, color=[c for *_, c in arms])
    ax.axhline(arms[0][1]["min_recovered_fraction"], color=C["clamp"], lw=1.3, ls="--",
               label=f"порог критерия {arms[0][1]['min_recovered_fraction']:.2f}")
    for x, val in zip(xs, vals, strict=True):
        ax.annotate(f"{val:.4f}", (x, val), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9, fontweight="600")
    ax.set_xticks(xs)
    ax.set_xticklabels([name for name, *_ in arms])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("доля восстановленного ΔLM")
    ax.set_title("A. Критерий T1 выполнен для обоих путей")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    ax.axvline(0, color=C["clean"], lw=1.4, label="нулевой эффект")
    for index, (_name, _, verdict, colour) in enumerate(arms):
        mean = verdict["paired_delta_nll_mean"]
        low, high = verdict["paired_delta_nll_ci"]
        ax.errorbar([mean], [index], xerr=[[mean - low], [high - mean]],
                    fmt="o", ms=9, color=colour, capsize=5, elinewidth=1.6)
        ax.annotate(f"{mean:+.6f}  [{low:+.6f}, {high:+.6f}]",
                    (mean, index), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=8.5, color=colour)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([
        f"{name}\n" + ("T2: нет улучшения" if v["verdict"] != "PASS" else "T2: улучшение")
        for name, _, v, _ in arms
    ])
    ax.set_ylim(-0.7, len(arms) - 0.3)
    ax.set_xlim(-0.002, 0.013)
    ax.set_xlabel("Объединённый парный ΔNLL против жёсткого клэмпа "
                  "(выше нуля = хуже клэмпа)")
    ax.set_title("B. Решение T2 совпадает при обоих путях")
    ax.legend(loc="lower right", fontsize=8)

    save(fig, "fig12_vp_path_control")
    out = {
        "linear": {
            "t1_recovered_fraction": lin_t1["t1_gate"]["recovered_fraction"],
            "t1_verdict": lin_t1["t1_gate"]["verdict"],
            "t2_mean": lin_t2["t2_experiment_verdict"]["paired_delta_nll_mean"],
            "t2_ci": lin_t2["t2_experiment_verdict"]["paired_delta_nll_ci"],
            "t2_verdict": lin_t2["t2_experiment_verdict"]["verdict"],
            "primary_cell": lin_t2["t2_experiment_verdict"]["primary_cell"],
        },
        "variance_preserving": {
            "t1_recovered_fraction": vp_t1["t1_gate"]["recovered_fraction"],
            "t1_verdict": vp_t1["t1_gate"]["verdict"],
            "t2_mean": vp_t2["t2_experiment_verdict"]["paired_delta_nll_mean"],
            "t2_ci": vp_t2["t2_experiment_verdict"]["paired_delta_nll_ci"],
            "t2_verdict": vp_t2["t2_experiment_verdict"]["verdict"],
            "primary_cell": vp_t2["t2_experiment_verdict"]["primary_cell"],
            "primary_t_start": vp_t2["t2_experiment_verdict"]["primary_t_start"],
        },
        "clamp_arm_reproduction": {
            q: {
                "linear": lin_t2["arms"][f"{q}_clamp_only"]["mean_delta_lm_vs_clean"],
                "variance_preserving": vp_t2["arms"][f"{q}_clamp_only"][
                    "mean_delta_lm_vs_clean"
                ],
            }
            for q in QUANTILES
        },
    }
    dump("fig12_vp_path_control", out)
    return out


# --------------------------------------------------------------------------
# Figure 13 — post-stop B: the steering-corruption denoiser at matched strength
# --------------------------------------------------------------------------

RUNGS = ("p50", "p75", "p90", "p95", "p99")


def figure13() -> dict:
    """The decision is the matched-strength panel; the nominal panel exists to
    show *why* a nominal comparison would have been misleading.

    A denoiser trained on ``z = h + delta v`` can score well against additive
    steering at equal nominal alpha simply by removing the steering, so panel B
    reports how much concept strength each arm actually delivered.
    """

    b = read_json(RES / "steering_denoiser_b_v1" / "steering_denoiser.json")
    spec_lambda = b["verdict"]["primary_lambda"]
    denoise = f"denoise_lambda{spec_lambda:.2f}"
    shrink = f"matched_shrinkage_for_lambda{spec_lambda:.2f}"
    rungs = [r for r in RUNGS if f"{r}_{denoise}" in b["arms"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    ax = axes[0]
    xs = np.arange(len(rungs))
    paired = [b["arms"][f"{r}_{denoise}"]["paired_delta_nll_vs_baseline"] for r in rungs]
    means = [p["mean"] for p in paired]
    ax.bar(xs, means, 0.5, color=C["denoiser"])
    ax.errorbar(xs, means,
                yerr=[[m - p["ci_lower"] for m, p in zip(means, paired, strict=True)],
                      [p["ci_upper"] - m for m, p in zip(means, paired, strict=True)]],
                fmt="none", ecolor=C["clamp"], capsize=4, elinewidth=1.2)
    ax.axhline(0, color=C["shrink"], lw=1.4)
    for x, mean in zip(xs, means, strict=True):
        ax.annotate(f"{mean:+.4f}", (x, mean), textcoords="offset points",
                    xytext=(0, 5 if mean > 0 else -12), ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(rungs)
    ax.set_xlabel("Целевой квантиль (сила стиринга)")
    ax.set_ylabel("NLL(денойзер) − NLL(усадка при той же силе)")
    ax.set_title("A. Решение: при равной реальной силе концепта\n"
                 "(ниже нуля = настоящее исправление)")
    interval = b["verdict"]["interval"]
    ax.annotate(
        f"объединённый {interval['mean']:+.6f}  "
        f"ДИ [{interval['ci_lower']:+.6f}, {interval['ci_upper']:+.6f}]  ·  "
        f"{'улучшение есть' if b['verdict']['verdict'] != 'NEGATIVE' else 'улучшения нет'}",
        (0.5, 0.03), xycoords="axes fraction", ha="center", fontsize=8,
        color=C["clamp"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
              "edgecolor": C["grid"], "linewidth": 0.7},
    )

    ax = axes[1]
    for lam, colour in zip(b["spec"]["corruption"]["lambda_grid"],
                           (C["flow_alt"], C["shrink"], C["additive"], C["denoiser"]),
                           strict=False):
        label = f"denoise_lambda{lam:.2f}"
        retained = [
            b["arms"][f"{r}_{label}"].get("retained_strength_fraction") for r in rungs
        ]
        xs_ok = [x for x, val in enumerate(retained) if val is not None]
        ax.plot(xs_ok, [retained[x] for x in xs_ok], "o-", ms=5, color=colour,
                label=f"λ = {lam:.2f}")
    ax.axhline(1.0, color=C["clean"], lw=1.3, ls="--", label="стиринг сохранён полностью")
    ax.axhline(0.0, color=C["tangent"], lw=1.3, ls=":", label="стиринг снят полностью")
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels(rungs)
    ax.set_ylim(-0.1, 1.15)
    ax.set_xlabel("Целевой квантиль (сила стиринга)")
    ax.set_ylabel(r"$\alpha_{\mathrm{eff}} / \alpha_{\mathrm{nominal}}$")
    ax.set_title("B. Сколько стиринга реально осталось")
    ax.legend(loc="lower left", fontsize=7.5, ncol=2)

    save(fig, "fig13_steering_denoiser_matched")
    verdict = b["verdict"]
    out = {
        "verdict": verdict["verdict"],
        "primary_cell": verdict["primary_cell"],
        "primary_lambda": spec_lambda,
        "pooled_effect": verdict["pooled_effect"],
        "interval": verdict["interval"],
        "per_rung_paired": {r: p for r, p in zip(rungs, paired, strict=True)},
        "retained_strength": {
            f"lambda{lam:.2f}": {
                r: b["arms"][f"{r}_denoise_lambda{lam:.2f}"].get(
                    "retained_strength_fraction"
                )
                for r in rungs
            }
            for lam in b["spec"]["corruption"]["lambda_grid"]
        },
        "nominal_vs_additive": {
            r: b["arms"][f"{r}_{denoise}"]["delta_nll_vs_nominal_additive"]
            for r in rungs
        },
        "strength_match": verdict["per_quantile_strength_match"],
        "corruption": b["spec"]["corruption"],
        "checkpoint_sha256": b["checkpoint_sha256"],
        "checkpoint_selection": b["checkpoint_selection"],
        "shrinkage_arm": {
            r: b["arms"][f"{r}_{shrink}"]["mean_nll"] for r in rungs
        },
    }
    dump("fig13_steering_denoiser_matched", out)
    return out


# --------------------------------------------------------------------------
# Figure 14 — post-stop C: curvature of the natural trajectory
# --------------------------------------------------------------------------


def figure14() -> dict:
    """C after the covariance controls: what survives, and what does not.

    The original panels compared the curvature against ordinary random unit axes.
    Experiment D showed that comparison is confounded by projected variance, so
    the figure now leads with the two controls that are variance-matched by
    construction and puts the concept-versus-random question where it belongs:
    beside the variance-selected principal components that beat it.
    """

    c6 = read_json(RES / "curvature_c6_covariance_v1" / "covariance_controls.json")
    confound = read_json(
        RES / "curvature_c6_variance_confound_v1" / "variance_confound.json"
    )
    surrogate = c6["c6_3_gaussian_surrogate"]
    matched = c6["c6_4_matched_random_directions"]
    fit = c6["c6_2_held_out_conditional_fit"]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1))

    # Panel A -- real vs a Gaussian with the SAME covariance.
    ax = axes[0]
    real = np.asarray(surrogate["real_shortfall"], dtype=float)
    synth = np.asarray(surrogate["surrogate_shortfall"], dtype=float)
    ax.bar([0, 1], [np.nanmean(real), np.nanmean(synth)], 0.55,
           color=[C["flow"], C["shrink"]])
    for x, values in ((0, real), (1, synth)):
        ax.scatter(np.full(values.size, x) + np.random.default_rng(0).normal(0, 0.06, values.size),
                   values, s=9, color=C["clamp"], alpha=0.55, zorder=3)
    diff = surrogate["real_minus_surrogate_shortfall"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["естественные\nактивации",
                        "гауссиана с той же\nковариацией"], fontsize=8.5)
    ax.set_ylabel("недобор до потолка надёжности")
    ax.set_title("A. Кривизна не объясняется\nковариацией второго порядка")
    ax.annotate(f"разница {diff['mean']:+.4f}\n"
                f"ДИ [{diff['ci_lower']:+.4f}, {diff['ci_upper']:+.4f}]\n"
                f"{int(diff['fraction_directions_positive'] * len(real))}/{len(real)} направлений",
                (0.5, 0.72), xycoords="axes fraction", ha="center", fontsize=8,
                color=C["flow"])

    # Panel B -- concept vs matched random vs variance-selected PCs.
    ax = axes[1]
    pcs = [c["shortfall_below_ceiling"] for c in c6["c6_6_principal_components"]]
    bars = [
        ("концептные\n(SAE)", matched["concept_shortfall_mean"], C["flow"]),
        ("ковариационно\nсогласованные", matched["matched_shortfall_mean"], C["shrink"]),
        ("главные\nкомпоненты", float(np.mean(pcs)), C["tangent"]),
    ]
    ax.bar(range(len(bars)), [v for _, v, _ in bars], 0.55,
           color=[c for *_, c in bars])
    for x, (_, value, _) in enumerate(bars):
        ax.annotate(f"{value:.4f}", (x, value), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8.5)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([n for n, *_ in bars], fontsize=8.5)
    ax.set_ylabel("недобор до потолка надёжности")
    ax.set_title("B. Но она не специфична\nдля концептных направлений")
    rho = confound["shortfall_vs_log_projected_variance"]
    ax.annotate(
        "недобор растёт с дисперсией\n"
        f"$\\rho$ = {rho['spearman_rho']:.2f} "
        f"[{rho['ci_lower']:.2f}, {rho['ci_upper']:.2f}]",
        (0.5, 0.06), xycoords="axes fraction", ha="center", fontsize=7.8,
        color=C["clamp"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
              "edgecolor": C["grid"], "linewidth": 0.7})

    # Panel C -- held-out linear vs quadratic conditional model.
    ax = axes[2]
    relative = 100.0 * np.asarray(fit["relative_improvement"], dtype=float)
    ax.hist(relative, bins=12, color=C["flow"], edgecolor="white")
    ax.axvline(0.0, color=C["tangent"], lw=1.5, label="линейная модель не хуже")
    interval = fit["relative_improvement_interval"]
    ax.axvline(100.0 * interval["mean"], color=C["clamp"], lw=1.4, ls="--",
               label=f"среднее {100 * interval['mean']:.3f}%")
    ax.set_xlabel("выигрыш квадратичной модели на отложенной половине, %")
    ax.set_ylabel("направлений")
    ax.set_title("C. Условное среднее нелинейно\n(32/32 направления)")
    ax.legend(loc="upper right", fontsize=7.5)

    save(fig, "fig14_concept_curvature")
    out = {
        "verdict_frozen_rule": c6["verdict"]["outcome"],
        "surrogate": {
            "real_shortfall_mean": surrogate["real_shortfall_mean"],
            "surrogate_shortfall_mean": surrogate["surrogate_shortfall_mean"],
            "difference": surrogate["real_minus_surrogate_shortfall"],
            "covariance_reproduction_error": surrogate["receipt"][
                "relative_covariance_error"
            ],
        },
        "matched_random": {
            "concept": matched["concept_shortfall_mean"],
            "matched": matched["matched_shortfall_mean"],
            "unmatched": matched["unmatched_shortfall_mean"],
            "difference": matched["concept_minus_matched_shortfall"],
            "balance": matched["balance"],
        },
        "principal_components_shortfall_mean": float(np.mean(pcs)),
        "held_out_fit": {
            "mse_linear": float(np.mean(fit["mse_linear"])),
            "mse_quadratic": float(np.mean(fit["mse_quadratic"])),
            "delta_interval": fit["delta_mse_interval"],
            "relative_interval": interval,
            "residual_to_secant_ratio": float(
                np.nanmean(np.asarray(fit["residual_to_secant_ratio_mean"], dtype=float))
            ),
        },
        "cos_dk_v_pooled": c6["c6_1_covariance_predicted_direction"]["cos_dk_v_pooled"],
        "cos_dk_sigma_v_pooled": c6["c6_1_covariance_predicted_direction"][
            "cos_dk_sigma_v_pooled"
        ],
        "shortfall_vs_variance": rho,
        "matching_balance_achieved": confound["matching_balance_achieved"],
        "profile_contrast": c6["c6_5_profile_contrast"],
    }
    dump("fig14_concept_curvature", out)
    return out


# --------------------------------------------------------------------------


def main() -> None:
    print("Building report figures from frozen artifacts...")
    for build in (figure1, figure2, figure3, figure4, figure5,
                  figure6, figure7, figure8, figure9, figure10, figure11,
                  figure12, figure13, figure14,
                  table_distn, table_summary):
        build()
    print(f"\nDone. {len(list(FIG.glob('*.png')))} figures, "
          f"{len(list(TAB.glob('*.md')))} tables, "
          f"{len(list(DAT.glob('*.json')))} data files.")


if __name__ == "__main__":
    main()
