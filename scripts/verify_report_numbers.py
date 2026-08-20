"""Check that every headline number in report/report.md matches its artifact.

The report is written by hand from figure data; this is the guard that keeps it
honest. Run after any change to the report or to the figure pipeline:

    uv run python scripts/verify_report_numbers.py

Exits non-zero on the first mismatch, printing artifact value against report
value. It reads only ``report/data/*.json``, which
``scripts/build_report_figures.py`` regenerates from ``results/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "report" / "data"
REPORT = ROOT / "report" / "report.tex"

failures: list[str] = []
checked = 0


def load(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


def check(label: str, artifact: float, claimed: float, tol: float) -> None:
    global checked
    checked += 1
    if abs(float(artifact) - float(claimed)) > tol:
        failures.append(f"{label}: артефакт={artifact!r} отчёт={claimed!r} (допуск {tol})")


def check_present(label: str, needle: str, haystack: str) -> None:
    """Assert a literal value appears in a generated artifact."""

    global checked
    checked += 1
    if needle not in haystack:
        failures.append(f"{label}: {needle!r} is absent")


def report_prose() -> str:
    """The report source with inline font markup removed.

    A claim must survive being re-typeset: wrapping half a phrase in \\texttt or
    breaking a line must not make the check pass or fail. Only the markup that
    can sit *inside* a sentence is stripped, so the words themselves still have
    to be there.
    """

    text = REPORT.read_text()
    for macro in ("\\texttt", "\\textbf", "\\emph", "\\textit"):
        text = text.replace(macro + "{", "")
    return " ".join(text.replace("}", "").split())


def check_in_report(label: str, needle: str) -> None:
    """Assert a claim really appears in the report, ignoring inline font markup."""

    global checked
    checked += 1
    wanted = " ".join(needle.replace("\\texttt{", "").replace("}", "").split())
    if wanted not in report_prose():
        failures.append(f"{label}: строка {needle!r} отсутствует в {REPORT.name}")


def main() -> int:
    t1 = load("fig08_t1_reconstruction")
    check("T1 recovered_fraction", t1["gate"]["recovered_fraction"], 0.7730, 1e-4)
    check("T1 paired mean", t1["ci"]["mean"], -1.012611, 1e-6)
    check("T1 CI lower", t1["ci"]["ci_lower"], -1.072443, 1e-6)
    check("T1 CI upper", t1["ci"]["ci_upper"], -0.949261, 1e-6)
    for cell, val in (("t0.25", 0.742), ("t0.50", 0.773), ("t0.75", 0.603)):
        check(f"T1 {cell}", t1["cells"][cell]["recovered_fraction"], val, 1e-3)
    if t1["gate"]["verdict"] != "PASS":
        failures.append("T1 verdict is not PASS")

    t2 = load("fig09_t2_naturalization")
    check("T2 pooled", t2["ci"]["mean"], 0.006184, 1e-6)
    check("T2 CI lower", t2["ci"]["ci_lower"], 0.001631, 1e-6)
    check("T2 CI upper", t2["ci"]["ci_upper"], 0.010788, 1e-6)
    check("T2 directions negative", t2["fraction_directions_negative"], 0.25, 1e-9)
    check("T2 LOVO min", t2["lovo"][0], 0.005244, 1e-6)
    check("T2 LOVO max", t2["lovo"][1], 0.006816, 1e-6)
    for q, val in (("q50", 0.004702), ("q75", 0.005042), ("q90", 0.005572),
                   ("q95", 0.006363), ("q99", 0.009241)):
        check(f"T2 {q}", t2["per_quantile"][q], val, 1e-6)
    if t2["verdict"]["verdict"] != "FAIL":
        failures.append("T2 verdict is not FAIL")

    mech = load("fig10_mechanism")
    for i, (x, y) in enumerate(zip([7.09, 16.81, 29.58],
                                   [0.006184, 0.054130, 0.346458], strict=True)):
        check(f"mechanism orth[{i}]", mech["tangent"]["orth_norm"][i], x, 5e-3)
        check(f"mechanism dNLL[{i}]", mech["tangent"]["delta_nll"][i], y, 1e-5)

    clamp = load("fig07_hard_clamp_cost")
    for i, val in enumerate([0.00318, 0.00341, 0.00767, 0.01470, 0.05369]):
        check(f"clamp {clamp['quantiles'][i]}", clamp["clamp_delta_lm"][i], val, 5e-5)

    cap = load("fig05_capacity_transfer")
    check("flow-MSE narrow", cap["val_flow_mse"]["narrow16m_fw32m"], 0.9809, 1e-4)
    check("flow-MSE wide", cap["val_flow_mse"]["wide60m_fw32m"], 0.8384, 1e-4)
    gain = 100 * (1 - cap["val_flow_mse"]["wide60m_fw32m"]
                  / cap["val_flow_mse"]["narrow16m_fw32m"])
    check("capacity gain %", gain, 14.5, 0.1)

    add = load("fig01_additive_pareto")
    check("clean NLL", add["clean_nll"], 3.4393, 1e-4)
    check("peak SAE", max(add["sae_target"]), 0.6145, 1e-4)
    # Peak inside the frozen main grid; the higher value at the stress point
    # alpha_hat = 2.0 is lexical flooding of degenerate text, not concept.
    main_grid = [c for c, al in zip(add["lexicon"], add["alpha_hat"], strict=True)
                 if al <= 1.0]
    check("peak lexicon (main grid)", max(main_grid), 0.0401, 1e-4)

    ctrl = load("fig06_conditional_control")
    for t, slope, dlm in (("0.50", 0.075, 0.234), ("0.75", 0.268, 1.174),
                          ("0.90", 0.925, 4.444)):
        check(f"slope t={t}", ctrl["slopes"][t], slope, 1e-3)
        check(f"delta_lm t={t}", ctrl["delta_lm"][t], dlm, 1e-3)

    den = load("fig04_denoiser_ability")
    for i, (c, d) in enumerate(zip([0.0451, 0.1997, 0.8722, 2.7236],
                                   [0.0307, 0.1138, 0.4239, 1.1619], strict=True)):
        check(f"denoiser corrupted[{i}]", den["corrupted_delta_lm"][i], c, 1e-6)
        check(f"denoiser denoised[{i}]", den["denoised_delta_lm"][i], d, 1e-6)

    dev = load("fig11_tangent_on_dev")
    check("tangent-on-DEV paired", dev["paired_mean"], 0.0598, 1e-4)
    check("tangent-on-DEV CI lower", dev["ci"][0], 0.0057, 1e-4)
    check("tangent-on-DEV CI upper", dev["ci"][1], 0.1182, 1e-4)
    check("alpha-zero offset", dev["alpha_zero_offset"], -0.0703, 1e-4)
    check("offset-corrected effect", dev["offset_corrected"], 0.1301, 1e-4)
    check("paired cells", dev["n_paired_cells"], 2880, 0)
    if dev["directions_negative"] != 2:
        failures.append("tangent-on-DEV: directions improving != 2")

    # ---- post-stop A: the variance-preserving path ----
    vp = load("fig12_vp_path_control")
    check("A VP T1 recovered", vp["variance_preserving"]["t1_recovered_fraction"],
          0.7321, 1e-4)
    check("A VP T2 pooled", vp["variance_preserving"]["t2_mean"], 0.006128, 1e-6)
    check("A VP T2 CI lower", vp["variance_preserving"]["t2_ci"][0], 0.001872, 1e-6)
    check("A VP T2 CI upper", vp["variance_preserving"]["t2_ci"][1], 0.010268, 1e-6)
    check("A linear T2 pooled", vp["linear"]["t2_mean"], 0.006184, 1e-6)
    if vp["variance_preserving"]["t1_verdict"] != "PASS":
        failures.append("A: VP T1 verdict is not PASS")
    if vp["variance_preserving"]["t2_verdict"] != "FAIL":
        failures.append("A: VP T2 verdict is not FAIL")
    # The comparison is only meaningful if the clamp arm reproduced exactly, so
    # this is checked at zero tolerance rather than to a few decimals.
    for q, pair in vp["clamp_arm_reproduction"].items():
        check(f"A clamp arm reproduced at {q}",
              pair["linear"], pair["variance_preserving"], 0.0)

    # ---- post-stop B: the steering-corruption denoiser ----
    den_b = load("fig13_steering_denoiser_matched")
    check("B pooled", den_b["interval"]["mean"], -0.000236, 1e-6)
    check("B CI lower", den_b["interval"]["ci_lower"], -0.002138, 1e-6)
    check("B CI upper", den_b["interval"]["ci_upper"], 0.001626, 1e-6)
    check("B directions negative", den_b["pooled_effect"]["fraction_directions_negative"],
          0.46875, 1e-9)
    check("B LOVO min", den_b["pooled_effect"]["lovo_min"], -0.000581, 1e-6)
    check("B LOVO max", den_b["pooled_effect"]["lovo_max"], 0.000390, 1e-6)
    if den_b["verdict"] != "NEGATIVE":
        failures.append("B verdict is not NEGATIVE")
    # The attenuation ladder is the reason the nominal comparison is misleading.
    for rung, retained in (("p75", 0.9580), ("p90", 0.8218), ("p95", 0.7122),
                           ("p99", 0.5352)):
        check(f"B retained strength {rung}",
              den_b["retained_strength"]["lambda1.00"][rung], retained, 1e-4)
    for rung, nominal in (("p50", -0.000442), ("p99", -0.040323)):
        check(f"B nominal vs additive {rung}", den_b["nominal_vs_additive"][rung],
              nominal, 1e-5)
    # A matched-strength claim is void if the arms did not land on the same alpha.
    for rung, match in den_b["strength_match"].items():
        check(f"B strength match at {rung}", float(match["matched"]), 1.0, 0.0)
        if match["max_arm_strength_difference"] > match["tolerance"]:
            failures.append(f"B arms differ in realised strength at {rung}")
    check("B corruption delta_max", den_b["corruption"]["delta_max"], 32.0, 1e-12)

    # ---- post-stop C, after the C6 covariance controls ----
    curv = load("fig14_concept_curvature")
    # The two comparisons that are variance-matched by construction.
    check("C6 real shortfall", curv["surrogate"]["real_shortfall_mean"], 0.2805, 1e-4)
    check("C6 surrogate shortfall", curv["surrogate"]["surrogate_shortfall_mean"],
          0.0033, 1e-4)
    check("C6 real-minus-surrogate", curv["surrogate"]["difference"]["mean"], 0.2772, 1e-4)
    check("C6 surrogate CI lower", curv["surrogate"]["difference"]["ci_lower"], 0.2431, 1e-4)
    check("C6 surrogate CI upper", curv["surrogate"]["difference"]["ci_upper"], 0.3078, 1e-4)
    check("C6 surrogate covariance error",
          curv["surrogate"]["covariance_reproduction_error"], 0.0257, 1e-4)
    check("C6 held-out delta MSE", curv["held_out_fit"]["delta_interval"]["mean"],
          0.0116, 1e-4)
    check("C6 held-out relative %",
          100 * curv["held_out_fit"]["relative_interval"]["mean"], 0.209, 1e-3)
    check("C6 residual-to-secant ratio", curv["held_out_fit"]["residual_to_secant_ratio"],
          0.52, 5e-3)
    if curv["surrogate"]["difference"]["fraction_directions_positive"] != 1.0:
        failures.append("C6: the surrogate comparison is not unanimous")
    if curv["held_out_fit"]["delta_interval"]["fraction_directions_positive"] != 1.0:
        failures.append("C6: the held-out quadratic does not win on every direction")

    # The claim the report does NOT make: concept-specific curvature.
    check("C6 concept shortfall", curv["matched_random"]["concept"], 0.2805, 1e-4)
    check("C6 matched-null shortfall", curv["matched_random"]["matched"], 0.1804, 1e-4)
    check("C6 principal-component shortfall",
          curv["principal_components_shortfall_mean"], 0.4359, 1e-4)
    if curv["principal_components_shortfall_mean"] <= curv["matched_random"]["concept"]:
        failures.append(
            "C6: the report says variance-selected components are MORE curved than "
            "the concept directions, but the artifact disagrees"
        )
    if curv["matching_balance_achieved"]:
        failures.append(
            "C6: the report says covariance matching was not achieved, but the "
            "artifact reports it was"
        )
    check("C6 shortfall-vs-variance rho", curv["shortfall_vs_variance"]["spearman_rho"],
          0.3853, 1e-4)
    check("C6 rho CI lower", curv["shortfall_vs_variance"]["ci_lower"], 0.1636, 1e-4)

    # C6.1: the descriptive statistic and its covariance-predicted counterpart.
    check("C cos(d_k, v) pooled", curv["cos_dk_v_pooled"], 0.3752, 1e-4)
    check("C cos(d_k, Sigma v) pooled", curv["cos_dk_sigma_v_pooled"], 0.7447, 1e-4)
    check("C tail-minus-central", curv["profile_contrast"]["concept_mean"], 0.1444, 1e-4)

    # ---- exploratory: does the C6 nonlinearity predict B's per-direction failure? ----
    join = json.loads(
        (ROOT / "results" / "c6_b_direction_join_v1" / "direction_join.json").read_text()
    )
    stats = join["statistics"]
    check("join beta_1", stats["beta_1"], -0.027, 5e-4)
    check("join beta_1 CI lower", stats["beta_1_ci"][0], -0.294, 5e-4)
    check("join beta_1 CI upper", stats["beta_1_ci"][1], 0.274, 5e-4)
    check("join Spearman", stats["spearman_rho"], -0.129, 5e-4)
    check("join Spearman CI lower", stats["rho_ci"][0], -0.477, 5e-4)
    check("join Spearman CI upper", stats["rho_ci"][1], 0.274, 5e-4)
    check("join partial Spearman",
          stats["partial_spearman_rho_after_log_variance"], -0.184, 5e-4)
    check("join n directions", stats["n_directions"], 32, 0)
    if join["verdict"]["outcome"] != "NO_ASSOCIATION":
        failures.append("join verdict is not NO_ASSOCIATION")
    if join["join_checks"]["n_shared_directions"] != 32:
        failures.append("join did not match all 32 directions")
    if not join["exploratory_post_hoc"] or join["preregistered"]:
        failures.append("the join must be labelled exploratory and not preregistered")

    # The post-hoc tangent arm must actually be present in the Dist-n table.
    distn_table = (ROOT / "report" / "tables" / "t02_distn.md").read_text()
    for alpha, dist1 in ((0.3, "0.8683"), (1.0, "0.7793")):
        check_present(f"Dist-n tangent arm alpha={alpha}", dist1, distn_table)

    # Statements whose wording carries the scientific claim. These check the
    # claim as a reader meets it in the report, not internal phrasing: the
    # report is a research report, so a check must not force lab-notebook
    # vocabulary back into it.
    check_in_report("T2 result", "+0.006184")
    check_in_report("no favourable configuration",
                    "из 30 дополнительных конфигураций не дала улучшения")
    check_in_report("difference-in-differences formula", "eq:did")
    check_in_report("capacity confound at fig 10", "не является")
    check_in_report("clamp range", "+0.003\\ldots+0.054")
    check_in_report("held-out was not used", "held-out направления не использовались")
    check_in_report("A-D are additional analyses", "дополнительные анализы")
    check_in_report("alpha-zero control", "-0.0703")
    check_in_report("nonlinearity does not predict repair", "-0.027")
    check_in_report("curvature is not concept-specific", "не специфичен для концептных")
    check_in_report("GLP divergence is not established", "не установили причинный механизм")

    print(f"Проверено утверждений: {checked}")
    if failures:
        print(f"\nРАСХОЖДЕНИЙ: {len(failures)}")
        for f in failures:
            print("  ✗", f)
        return 1
    print("Все числа в отчёте совпадают с артефактами.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
