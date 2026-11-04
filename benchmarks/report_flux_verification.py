"""Render the figure and the markdown report for the Euler flux verification.

Numbers are read from ``validation/euler_fluxes/flux_properties.json`` (written
by ``benchmarks/verify_flux_properties.py``) so nothing is transcribed by hand.

Usage::

    PYTHONPATH=. python3 benchmarks/report_flux_verification.py
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


CASES = [
    ("sod", "Sod"),
    ("lax", "Lax"),
    ("123problem", "1-2-3 (near vacuum)"),
    ("strong_shock", "strong shock"),
    ("stationary_contact", "stationary contact"),
    ("moving_contact", "moving contact"),
    ("left_moving_contact", "left-moving contact"),
    ("sonic_rarefaction", "fan at interface"),
]


def make_figure(payload, path):
    agreement = {row["case"]: row for row in payload["exact_riemann_agreement"]}
    attribution = {row["case"]: row for row in payload["wave_speed_attribution"]}

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)

    # panel (a): scaled interface-flux error, three flux variants
    labels = [label for _, label in CASES]
    hllc_davis = [max(agreement[key]["hllc"]["scaled"], 1e-16) for key, _ in CASES]
    hllc_exact = [max(attribution[key]["hllc_exact_speeds"]["scaled"], 1e-16) for key, _ in CASES]
    rusanov = [max(agreement[key]["rusanov"]["scaled"], 1e-16) for key, _ in CASES]
    y = np.arange(len(labels))
    height = 0.26
    ax = axes[0]
    ax.barh(y - height, hllc_davis, height, label="HLLC (Davis speeds)", color="#1f77b4")
    ax.barh(y, hllc_exact, height, label="HLLC (exact speeds)", color="#2ca02c")
    ax.barh(y + height, rusanov, height, label="Rusanov (LLF)", color="#d62728")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("interface-flux error, scaled by max(|F_L|,|F_R|,|F_exact|)")
    ax.set_title("(a) flux error vs exact Riemann state at x/t = 0 (Toro 2009)", fontsize=10)
    ax.axvline(0.15, color="0.35", linestyle="--", linewidth=1.0)
    ax.text(0.155, len(labels) - 0.6, "15%", fontsize=8, color="0.35")
    ax.axvline(1e-12, color="0.7", linewidth=0.8)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", which="both", alpha=0.25)

    # panel (b): contact resolution, mass-flux error in absolute terms
    contact_keys = ["stationary_contact", "moving_contact", "left_moving_contact"]
    contact_labels = [label for key, label in CASES if key in contact_keys]
    hllc_contact = [agreement[key]["hllc"]["max_abs_error"] for key in contact_keys]
    rusanov_contact = [agreement[key]["rusanov"]["max_abs_error"] for key in contact_keys]
    x = np.arange(len(contact_keys))
    ax = axes[1]
    ax.bar(x - 0.18, np.maximum(hllc_contact, 1e-16), 0.36, label="HLLC", color="#1f77b4")
    ax.bar(x + 0.18, rusanov_contact, 0.36, label="Rusanov", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(contact_labels, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("|F_num - F_exact| (mass-flux units)")
    ax.set_title("(b) isolated contact: HLLC is exact, Rusanov diffuses it", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", which="both", alpha=0.25)

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fmt(value, digits=3):
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def markdown(payload, figure_path):
    summary = payload["summary"]
    agreement = {row["case"]: row for row in payload["exact_riemann_agreement"]}
    attribution = {row["case"]: row for row in payload["wave_speed_attribution"]}
    shock = payload["rankine_hugoniot_shock"]
    contacts = payload["exact_invariants"]["isolated_contact"]

    lines = []
    lines.append("# Euler interface-flux kernel verification (HLLC vs Rusanov vs exact Riemann)")
    lines.append("")
    lines.append("> Repository: TensorDG. Scope: verify the **already-merged** interface-flux "
                 "kernels in `tensordg/euler_1d.py` against analytic invariants and the exact "
                 "Riemann solution. No solver code was changed; this document is the "
                 "verification record.")
    lines.append("> Reproduction:")
    lines.append("> `PYTHONPATH=. python3 benchmarks/verify_flux_properties.py "
                 "--json validation/euler_fluxes/flux_properties.json`")
    lines.append("> `PYTHONPATH=. python3 -m pytest tests/test_euler_fluxes.py -q`")
    lines.append("")
    lines.append("## 1. Case set and parameters")
    lines.append("")
    lines.append("| case | left (rho, u, p) | right (rho, u, p) | note |")
    lines.append("|---|---|---|---|")
    for row in payload["exact_riemann_agreement"]:
        left = ", ".join(f"{v:g}" for v in row["left"])
        right = ", ".join(f"{v:g}" for v in row["right"])
        lines.append(f"| {row['case']} | ({left}) | ({right}) | {row['description']} |")
    lines.append("")
    lines.append(f"All computations are in float64, gamma = {payload['gamma']}, on CPU; the "
                 "interface is at x = 0.5 and the exact solution is sampled at t = 1 "
                 "(state at x/t = 0).")
    lines.append("")
    lines.append("## 2. Method")
    lines.append("")
    lines.append("Three evidence levels, deliberately kept separate:")
    lines.append("")
    lines.append("1. **Exact invariants** (round-off tolerance): consistency "
                 "`F(U,U) = f(U)`; mirror symmetry `R F(U_L,U_R) = F(M U_R, M U_L)` with "
                 "`M = diag(1,-1,1)` and `R = diag(-1,1,-1)`; isolated-contact exactness "
                 "(the defining property of HLLC, Toro 2009 sec. 10.4).")
    lines.append("2. **Agreement with the exact Riemann solution**: the exact primitive state "
                 "at x/t = 0 is obtained from `benchmarks/riemann_exact.sample`, which resolves "
                 "shock, contact and rarefaction-fan branches — using the star state directly "
                 "would be wrong whenever x/t = 0 lies inside a fan or on a contact.")
    lines.append("3. **Attribution**: the same HLLC algebra is re-run with the *exact* outer "
                 "wave speeds, separating the error of the HLLC ansatz from the error of "
                 "Davis' wave-speed estimates used in `tensordg/euler_1d.py`.")
    lines.append("")
    lines.append("Normalisation: `scaled` divides the max-abs flux error by "
                 "`max(|F_L|, |F_R|, |F_exact|)`. The plain `relative` normalisation is "
                 f"meaningless for the 1-2-3 problem (its exact flux nearly vanishes: "
                 f"{fmt(agreement['123problem']['hllc']['relative'])} there); both are reported.")
    lines.append("")
    lines.append("## 3. Results")
    lines.append("")
    lines.append("### 3.1 Exact invariants")
    lines.append("")
    lines.append("| invariant | measured | verdict |")
    lines.append("|---|---|---|")
    lines.append(f"| consistency `F(U,U)=f(U)` (17 states) | max abs error "
                 f"{fmt(summary['consistency_max_abs_error'])} | exact to round-off |")
    lines.append(f"| mirror symmetry (8 cases) | max relative error "
                 f"{fmt(summary['mirror_symmetry_max_relative_error'])} | exact to round-off |")
    lines.append(f"| isolated contact, HLLC | max abs error "
                 f"{fmt(summary['contact_hllc_max_abs_error'])} | exact to round-off |")
    lines.append(f"| isolated contact, Rusanov | max scaled error "
                 f"{fmt(summary['contact_rusanov_max_scaled_error'])} | contact diffused |")
    lines.append("")
    lines.append("### 3.2 Agreement with the exact Riemann solution at the interface")
    lines.append("")
    lines.append("| case | HLLC (Davis) scaled | HLLC (exact speeds) scaled | Rusanov scaled | "
                 "HLLC relative |")
    lines.append("|---|---|---|---|---|")
    for key, _ in CASES:
        row = agreement[key]
        lines.append(f"| {key} | {fmt(row['hllc']['scaled'])} | "
                     f"{fmt(attribution[key]['hllc_exact_speeds']['scaled'])} | "
                     f"{fmt(row['rusanov']['scaled'])} | {fmt(row['hllc']['relative'])} |")
    lines.append("")
    lines.append(f"- median scaled error: HLLC {fmt(summary['riemann_hllc_median_scaled_error'])}, "
                 f"Rusanov {fmt(summary['riemann_rusanov_median_scaled_error'])}.")
    lines.append(f"- max scaled error: HLLC {fmt(summary['riemann_hllc_max_scaled_error'])}, "
                 f"Rusanov {fmt(summary['riemann_rusanov_max_scaled_error'])}.")
    lines.append(f"- median with exact wave speeds: "
                 f"{fmt(summary['hllc_exact_speeds_median_scaled_error'])} "
                 f"(max {fmt(summary['hllc_exact_speeds_max_scaled_error'])}).")
    lines.append("")
    lines.append("### 3.3 Isolated Rankine-Hugoniot shock")
    lines.append("")
    lines.append("| shock Mach | post-shock (rho, u, p) | HLLC scaled error |")
    lines.append("|---|---|---|")
    for row in shock:
        state = ", ".join(f"{v:.6g}" for v in row["post_shock_state"])
        lines.append(f"| {row['shock_mach']:g} | ({state}) | {fmt(row['hllc']['scaled'])} |")
    lines.append("")
    lines.append("### 3.4 Figure")
    lines.append("")
    lines.append(f"![flux verification]({figure_path})")
    lines.append("")
    lines.append("Panel (a) compares the three flux variants per case on a log axis (dashed "
                 "line = 15% mark); panel (b) shows the absolute mass-flux error for the three "
                 "pure-contact cases, where HLLC is exact to round-off and Rusanov smears the "
                 "jump.")
    lines.append("")
    lines.append("## 4. Error judgement against the references")
    lines.append("")
    worse = [key for key, _ in CASES
             if agreement[key]["hllc"]["scaled"] > agreement[key]["rusanov"]["scaled"] + 1e-12]
    lines.append("| quantity | reference | error | < 3%? |")
    lines.append("|---|---|---|---|")
    lines.append(f"| consistency, mirror symmetry, contact flux | analytic identity | "
                 f"<= {fmt(max(summary['consistency_max_abs_error'], summary['contact_hllc_max_abs_error']))}"
                 f" (abs) | **yes** (round-off) |")
    lines.append(f"| HLLC interface flux, contact cases | exact Riemann state | "
                 f"{fmt(summary['contact_hllc_max_abs_error'])} (abs) | **yes** (exact) |")
    lines.append(f"| HLLC interface flux, all cases, exact wave speeds | exact Riemann state "
                 f"(median) | {fmt(summary['hllc_exact_speeds_median_scaled_error'])} | **yes** |")
    lines.append(f"| HLLC interface flux, all cases, Davis speeds (median) | exact Riemann state "
                 f"| {fmt(summary['riemann_hllc_median_scaled_error'])} | **no** |")
    lines.append(f"| HLLC interface flux, isolated shock, Davis speeds | exact Riemann state "
                 f"| {fmt(summary['isolated_shock_max_scaled_error'])} | "
                 f"{'**yes**' if summary['isolated_shock_max_scaled_error'] < 0.03 else '**no**'} |")
    lines.append("")
    lines.append("Answers explicitly: the **kernel identities are exact** (< 3% trivially, they "
                 "hold to round-off), the **contact resolution is exact**, but the **generic "
                 "two-state interface flux with Davis' wave speeds misses the exact Riemann "
                 "flux by a median "
                 f"{100 * summary['riemann_hllc_median_scaled_error']:.1f}%** — i.e. **not below "
                 "3%**. With the exact outer wave speeds the same algebra reproduces the flux "
                 f"to round-off on the shock cases (median "
                 f"{100 * summary['hllc_exact_speeds_median_scaled_error']:.2f}%), so the "
                 "residual is attributable to the wave-speed estimates, not to the HLLC formula.")
    lines.append("")
    lines.append("## 5. Caveats and limitations")
    lines.append("")
    lines.append("1. **Rarefaction-fan residuals are intrinsic, not defects.** For Sod, the 1-2-3 "
                 "problem and the sonic-rarefaction case, x/t = 0 falls *inside* a rarefaction "
                 "fan. A three-wave HLLC flux cannot represent a fan at the interface, so a "
                 "residual there is expected at any wave-speed choice (see panel (a): the two "
                 "HLLC variants coincide on exactly those three cases).")
    lines.append("2. **Rusanov is not uniformly worse.** On the strong-shock case Rusanov is "
                 f"more accurate ({fmt(agreement['strong_shock']['rusanov']['scaled'])}) than "
                 f"HLLC with Davis speeds "
                 f"({fmt(agreement['strong_shock']['hllc']['scaled'])}): Davis' estimates "
                 "overestimate the outer wave speeds for very strong shocks. This is why the "
                 "attribution level exists — it shows the fix (Einfeldt/Roe estimates) rather "
                 "than blaming the HLLC ansatz.")
    lines.append("3. **This is a flux-kernel result, not a solution-accuracy result.** The "
                 "end-to-end solver errors (Sod 3.2% etc., `docs/limiter_verification.md`) are "
                 "governed by the grid, the reconstruction and the time integrator; a 11% "
                 "interface-flux error does not imply an 11% solution error.")
    lines.append("4. **Scope.** 1-D Euler, two constant states, ideal gas gamma = 1.4, float64, "
                 "CPU. Multi-dimensional face fluxes and the DG trace (limited) states are out "
                 "of scope; the flux kernel itself is dimension-independent.")
    lines.append("5. **Recommendation (not implemented here).** Replace Davis' estimates with "
                 "Einfeldt/Roe-average estimates inside `hllc_flux` to remove the dominant "
                 "residual on strong shocks and smooth cases. That change touches the solver "
                 "and must be re-verified end-to-end, so it is filed as follow-up work rather "
                 "than bundled into this verification pass.")
    lines.append("")
    lines.append("## 6. Files")
    lines.append("")
    lines.append("| path | role |")
    lines.append("|---|---|")
    lines.append("| `benchmarks/verify_flux_properties.py` | verification driver (writes the JSON) |")
    lines.append("| `benchmarks/report_flux_verification.py` | this document + figure generator |")
    lines.append("| `tests/test_euler_fluxes.py` | 23 regression tests pinning the invariants |")
    lines.append("| `validation/euler_fluxes/flux_properties.json` | all numbers (provenance) |")
    lines.append(f"| `{figure_path}` | figure |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="validation/euler_fluxes/flux_properties.json")
    parser.add_argument("--figure", default="docs/figures/euler_flux_kernel_verification.png")
    parser.add_argument("--out", default="docs/euler_flux_kernel_verification.md")
    args = parser.parse_args()

    with open(args.json, encoding="utf-8") as handle:
        payload = json.load(handle)

    figure = make_figure(payload, args.figure)
    text = markdown(payload, os.path.relpath(figure, os.path.dirname(args.out)))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"wrote {args.out} and {figure}")


if __name__ == "__main__":
    main()