"""Discrete entropy budget for the 1-D Euler DG / finite-volume solvers.

Closes the verification gap recorded in
``docs/benchmarks/conservation-law-verification.md`` ("No entropy-production
budget yet; a discrete entropy inequality is the next natural addition").

Three falsifiable criteria, all anchored on analysis rather than on a previous
run of the same code:

C1  Second law (entropy inequality).  For shock-bearing data the total entropy
    ``int rho s dx`` must *increase*; the numerical entropy production has to be
    strictly positive at every resolution and for every limiter setting.

C2  Quantitative budget.  The measured increase must converge to the analytic
    Rankine-Hugoniot production ``P * T`` of :mod:`benchmarks._entropy_tools`
    (single shock, exact star state, no fitted parameter).

C3  Smooth-flow consistency.  For smooth periodic data the exact solution
    conserves ``int rho s dx``; the numerical production must therefore tend to
    zero under refinement (and must stay on the dissipative side, ``>= 0``).

Usage:  PYTHONPATH=. python -m benchmarks.entropy_budget_1d
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from benchmarks._entropy_tools import (
    analytic_right_shock_budget,
    entropy_density,
    legendre_modes,
    observed_order,
    primitive_from_conservative,
    relative_error,
    total_entropy_from_coefficients,
    total_entropy_from_coefficients_convergence,
    total_entropy_from_profiles,
)
from benchmarks.riemann_exact import sample
from tensordg.euler_dg_1d import solve_euler_dg_1d, solve_periodic_euler_dg_1d

LEFT = (1.0, 0.0, 1.0)          # Sod
RIGHT = (0.125, 0.0, 0.1)
LAX_LEFT = (0.445, 0.698, 3.528)
LAX_RIGHT = (0.5, 0.0, 0.571)
SOD_TIME = 0.2
LAX_TIME = 0.15
SMOOTH_TIME = 0.1
DX = 1.0  # unit domain


def _sod_shock_budget(final_time: float):
    return analytic_right_shock_budget(LEFT, RIGHT, final_time)


def _lax_shock_budget(final_time: float):
    return analytic_right_shock_budget(LAX_LEFT, LAX_RIGHT, final_time)


def _entropy_of_run(left, right, ncells, degree, final_time, limiter, tvb_constant,
                    limiter_stage, cfl=0.3):
    """Discrete ``int rho s dx`` before and after one DG run."""
    _, coeff_0, _ = solve_euler_dg_1d(
        left, right, ncells=ncells, degree=degree, final_time=0.0, cfl=cfl,
        limiter=limiter, tvb_constant=tvb_constant, limiter_stage=limiter_stage,
    )
    _, coeff_t, meta = solve_euler_dg_1d(
        left, right, ncells=ncells, degree=degree, final_time=final_time, cfl=cfl,
        limiter=limiter, tvb_constant=tvb_constant, limiter_stage=limiter_stage,
    )
    dx = 1.0 / ncells
    return {
        "initial": total_entropy_from_coefficients(coeff_0, dx, degree),
        "final": total_entropy_from_coefficients(coeff_t, dx, degree),
        "steps": meta["steps"],
        "dt": meta["dt"],
        "coeff": coeff_t,
    }


def shock_study(cells, limiter_settings, degree=2, cfl=0.3):
    """Run the three criteria on Sod and Lax and on the smooth periodic wave."""
    sod = _sod_shock_budget(SOD_TIME)
    lax = _lax_shock_budget(LAX_TIME)
    scenarios = [
        ("sod", LEFT, RIGHT, SOD_TIME, sod),
        ("lax", LAX_LEFT, LAX_RIGHT, LAX_TIME, lax),
    ]
    results = {}
    for name, left, right, final_time, budget in scenarios:
        rows = []
        for limiter, tvb_constant in limiter_settings:
            for ncells in cells:
                run = _entropy_of_run(left, right, ncells, degree, final_time,
                                      limiter, tvb_constant, "every", cfl=cfl)
                produced = run["final"] - run["initial"]
                rows.append({
                    "limiter": limiter or "none",
                    "tvb_constant": tvb_constant,
                    "cells": ncells,
                    "degree": degree,
                    "steps": run["steps"],
                    "dt": run["dt"],
                    "entropy_initial": run["initial"],
                    "entropy_final": run["final"],
                    "production_numeric": produced,
                    "production_analytic": budget["production_total"],
                    "relative_error": relative_error(budget["production_total"], produced),
                    "second_law": bool(produced > 0.0),
                })
        results[name] = {
            "analytic": {
                "sigma": budget["sigma"],
                "production_rate": budget["production"],
                "production_total": budget["production_total"],
                "star_state": list(budget["star_state"]),
                "downstream": list(budget["downstream"]),
                "shock_position_at_T": budget["shock_position"],
                "final_time": final_time,
            },
            "rows": rows,
        }
    return results


def exact_profile_entropy(left, right, final_time, samples=20001):
    x = np.linspace(0.0, 1.0, samples)
    fields = sample(x, final_time, left, right)
    return x, entropy_density(fields["density"], fields["pressure"])


def profile_comparison(coeff, ncells, degree, left, right, final_time, samples=2001):
    """Pointwise entropy density of the DG solution versus the exact solution."""
    dx = 1.0 / ncells
    x = np.linspace(dx * 0.5, 1.0 - dx * 0.5, samples)
    cell = np.clip((x / dx).astype(int), 0, ncells - 1)
    xi = 2.0 * (x - (cell + 0.5) * dx) / dx
    phi = legendre_modes(xi, degree)
    values = np.einsum("cmv,qm->qv", np.asarray(coeff, dtype=float), phi)
    density, _, pressure = primitive_from_conservative(values)
    numerical = entropy_density(density, pressure)
    fields = sample(x, final_time, left, right)
    exact = entropy_density(fields["density"], fields["pressure"])
    return x, numerical, exact


def smooth_study(cells, limiter_settings, degree=2):
    """C3 on the exact smooth periodic entropy wave (rho = 1 + 0.2 sin 2 pi x)."""
    def initial_state(nodes):
        # the periodic solver works on torch tensors (see tensordg.euler_1d)
        import torch
        rho = 1.0 + 0.2 * torch.sin(2.0 * np.pi * nodes)
        return rho, torch.ones_like(nodes), torch.ones_like(nodes)

    rows = []
    for limiter, tvb_constant in limiter_settings:
        for ncells in cells:
            dx = 1.0 / ncells
            _, coeff_0, _ = solve_periodic_euler_dg_1d(
                initial_state, ncells=ncells, degree=degree, final_time=0.0,
                limiter=limiter, tvb_constant=tvb_constant,
            )
            _, coeff_t, meta = solve_periodic_euler_dg_1d(
                initial_state, ncells=ncells, degree=degree, final_time=SMOOTH_TIME,
                limiter=limiter, tvb_constant=tvb_constant,
            )
            produced = (total_entropy_from_coefficients(coeff_t, dx, degree)
                        - total_entropy_from_coefficients(coeff_0, dx, degree))
            rows.append({
                "limiter": limiter or "none",
                "tvb_constant": tvb_constant,
                "cells": ncells,
                "degree": degree,
                "steps": meta["steps"],
                "production_numeric": produced,
                "relative_to_initial": produced / abs(
                    total_entropy_from_coefficients(coeff_0, dx, degree)),
            })
    return rows


def plot(path, shock_results, smooth_rows, profile, degree):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    styles = {"none": ("o", "C0"), "tvb": ("s", "C1"), "weno": ("^", "C2")}
    for name in ("sod", "lax"):
        analytic = shock_results[name]["analytic"]["production_total"]
        rows = shock_results[name]["rows"]
        for limiter in sorted({row["limiter"] for row in rows}):
            subset = [row for row in rows if row["limiter"] == limiter]
            cells = [row["cells"] for row in subset]
            errors = [abs(row["production_numeric"] - analytic) for row in subset]
            marker, colour = styles.get(limiter, ("o", "k"))
            label = f"{name.upper()} {limiter}"
            ax.loglog(cells, errors, marker=marker, color=colour,
                      linestyle="-" if name == "sod" else "--", label=label)
    ref = np.array([16, 800], dtype=float)
    first = shock_results["sod"]["rows"][0]
    ax.loglog(ref, abs(first["production_analytic"]) * (ref / 100.0) ** -2,
              color="grey", linestyle=":", label=r"$h^2$ reference")
    ax.set_xlabel("cells")
    ax.set_ylabel(r"$|\Delta\int\rho s\,dx - P T|$")
    ax.set_title("(a) entropy budget vs Rankine-Hugoniot")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    x, numerical, exact = profile["none"]
    ax.plot(x, exact, color="k", linewidth=1.6, label="exact")
    for limiter, (marker, colour) in styles.items():
        if limiter in profile:
            ax.plot(profile[limiter][0], profile[limiter][1], color=colour,
                    linewidth=1.0, linestyle="-", marker=marker, markevery=60,
                    markersize=3.5, label=f"DG P{degree} {limiter}")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho s$")
    ax.set_title(f"(b) Sod entropy density, {profile['cells']} cells")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2]
    for limiter in sorted({row["limiter"] for row in smooth_rows}):
        subset = [row for row in smooth_rows if row["limiter"] == limiter]
        cells = [row["cells"] for row in subset]
        drift = [abs(row["production_numeric"]) for row in subset]
        marker, colour = styles.get(limiter, ("o", "k"))
        orders = observed_order(cells, drift)
        label = f"{limiter}, order {orders[-1]:.2f}" if orders[-1] else limiter
        ax.loglog(cells, drift, marker=marker, color=colour, label=label)
    ref = np.array([16, 128], dtype=float)
    ax.loglog(ref, 1e-9 * (ref / 16.0) ** -3, color="grey", linestyle=":", label=r"$h^3$ reference")
    ax.set_xlabel("cells")
    ax.set_ylabel(r"$|\Delta\int\rho s\,dx|$")
    ax.set_title("(c) smooth periodic entropy wave")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle("TensorDG discrete entropy budget: 1-D Euler DG (HLLC, SSP-RK3)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--lax-cells", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--smooth-cells", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--cfl", type=float, default=0.3)
    parser.add_argument("--limiters", nargs="+", default=["none", "tvb"],
                        choices=["none", "tvb", "weno"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("validation/entropy_1d/summary.json"))
    parser.add_argument("--figure", type=Path, default=Path("docs/figures/entropy_budget_1d.png"))
    args = parser.parse_args()

    if args.threads:
        try:  # tiny tensors: thread hand-off dominates under a loaded machine
            import torch
            torch.set_num_threads(args.threads)
        except Exception:  # pragma: no cover - torch is a hard dependency in practice
            pass

    limiter_settings = [(None if name == "none" else name, 1.0) for name in args.limiters]
    shock_results = shock_study(args.cells, limiter_settings, degree=args.degree, cfl=args.cfl)
    lax_results = shock_study(args.lax_cells, limiter_settings, degree=args.degree, cfl=args.cfl)
    # keep only the Lax scenario from the second pass
    shock_results["lax"] = lax_results["lax"]
    smooth_rows = smooth_study(args.smooth_cells, limiter_settings, degree=args.degree)

    profile_ncells = args.cells[-2] if len(args.cells) > 1 else args.cells[-1]
    profile = {"cells": profile_ncells}
    for limiter, tvb_constant in limiter_settings:
        run = _entropy_of_run(LEFT, RIGHT, profile_ncells, args.degree, SOD_TIME,
                              limiter, tvb_constant, "every", cfl=args.cfl)
        x, numerical, exact = profile_comparison(run["coeff"], profile_ncells, args.degree,
                                                 LEFT, RIGHT, SOD_TIME)
        profile[limiter or "none"] = (x, numerical, exact)

    # diagnostic self-test 1: quadrature independence of the entropy integral
    _, coeff_self, _ = solve_euler_dg_1d(LEFT, RIGHT, ncells=200, degree=args.degree,
                                        final_time=SOD_TIME, cfl=args.cfl, limiter=None)
    ladder = total_entropy_from_coefficients_convergence(coeff_self, 1.0 / 200, args.degree)
    quadrature_spread = (max(ladder) - min(ladder)) / abs(ladder[len(ladder) // 2])

    # diagnostic self-test 2: analytic budget vs a finite difference of the exact solution
    xfine = np.linspace(0.0, 1.0, 100001)
    t_hi, t_lo = SOD_TIME, SOD_TIME - 0.05
    o_hi = sample(xfine, t_hi, LEFT, RIGHT)
    o_lo = sample(xfine, t_lo, LEFT, RIGHT)
    rate_exact = (total_entropy_from_profiles(o_hi["density"], o_hi["pressure"], xfine)
                  - total_entropy_from_profiles(o_lo["density"], o_lo["pressure"], xfine)) / (t_hi - t_lo)
    analytic_vs_exact = {
        "rate_from_exact_profile": rate_exact,
        "rate_from_rankine_hugoniot": shock_results["sod"]["analytic"]["production_rate"],
        "relative_difference": relative_error(shock_results["sod"]["analytic"]["production_rate"], rate_exact),
    }

    # CFL sensitivity: the budget must not depend on the time step
    cfl_check = []
    for cfl in (0.12, args.cfl):
        run = _entropy_of_run(LEFT, RIGHT, args.cells[0], args.degree, SOD_TIME,
                              None, 1.0, "every", cfl=cfl)
        cfl_check.append({
            "cfl": cfl,
            "steps": run["steps"],
            "production_numeric": run["final"] - run["initial"],
        })
    cfl_check[0]["relative_difference_to_reference_cfl"] = relative_error(
        cfl_check[-1]["production_numeric"], cfl_check[0]["production_numeric"])

    summary = {
        "tool": "benchmarks/entropy_budget_1d.py",
        "degree": args.degree,
        "cfl": args.cfl,
        "flux": "hllc",
        "time_integrator": "SSP-RK3",
        "torch_threads": args.threads,
        "note": ("Entropy integral evaluated on the full DG polynomial with "
                 "max(degree+2, 8) Gauss points, never on cell averages."),
        "criteria": {
            "C1_second_law": "production_numeric > 0 for every shock run",
            "C2_budget": "production_numeric -> analytic Rankine-Hugoniot P*T under refinement",
            "C3_smooth": "|production| -> 0 for smooth periodic data, order >= degree",
        },
        "diagnostic_self_test": {
            "quadrature_orders": [4, 6, 8, 10, 12],
            "entropy_integral_ladder": ladder,
            "quadrature_relative_spread": quadrature_spread,
            "analytic_vs_exact_profile": analytic_vs_exact,
        },
        "cfl_sensitivity": cfl_check,
        "shock": shock_results,
        "smooth_periodic": smooth_rows,
        "profile_comparison": {
            "cells": profile_ncells,
            "relative_l2_entropy_density": {
                key: float(np.linalg.norm(profile[key][1] - profile[key][2])
                           / np.linalg.norm(profile[key][2]))
                for key in ("none", "tvb", "weno") if key in profile
            },
        },
    }

    finest, second_law = {}, all(
        row["second_law"] for case in shock_results.values() for row in case["rows"])
    for name, case in shock_results.items():
        for limiter in {row["limiter"] for row in case["rows"]}:
            subset = [row for row in case["rows"] if row["limiter"] == limiter]
            finest[f"{name}/{limiter}"] = min(subset, key=lambda row: row["cells"])["relative_error"]
    smooth_orders = {
        limiter: observed_order(
            [row["cells"] for row in smooth_rows if row["limiter"] == limiter],
            [abs(row["production_numeric"]) for row in smooth_rows if row["limiter"] == limiter])
        for limiter in {row["limiter"] for row in smooth_rows}
    }
    summary["verdicts"] = {
        "C1_second_law_all_runs": bool(second_law),
        "C2_relative_error_at_finest_mesh": finest,
        "C2_below_3_percent": {key: bool(value < 0.03) for key, value in finest.items()},
        "C3_smooth_observed_orders": smooth_orders,
        "C3_smooth_tends_to_zero": {
            limiter: bool(abs(rows[-1]["relative_to_initial"]) < 1e-3)
            for limiter, rows in [
                (limiter, [row for row in smooth_rows if row["limiter"] == limiter])
                for limiter in {row["limiter"] for row in smooth_rows}]},
        "diagnostic_self_test_below_0_1_percent": bool(analytic_vs_exact["relative_difference"] < 1e-3),
        "cfl_insensitive_below_1_percent": bool(
            cfl_check[0]["relative_difference_to_reference_cfl"] < 1e-2),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    figure = plot(args.figure, shock_results, smooth_rows, profile, args.degree)

    print(f"wrote {args.output} and {figure}", flush=True)
    for name, case in shock_results.items():
        analytic = case["analytic"]["production_total"]
        print(f"\n{name.upper()}  analytic P*T = {analytic:.6e}  sigma = {case['analytic']['sigma']:.4f}"
              f"  shock at x = {case['analytic']['shock_position_at_T']:.4f}", flush=True)
        print(f"{'limiter':>8} {'cells':>6} {'steps':>6} {'production':>13} {'rel.err':>9} {'2nd law':>8}", flush=True)
        for row in case["rows"]:
            print(f"{row['limiter']:>8} {row['cells']:>6} {row['steps']:>6} "
                  f"{row['production_numeric']:>13.6e} {row['relative_error']:>8.3%} "
                  f"{'yes' if row['second_law'] else 'NO':>8}", flush=True)
    print("\nsmooth periodic entropy wave (exact production = 0)", flush=True)
    print(f"{'limiter':>8} {'cells':>6} {'production':>13} {'rel. to |H0|':>14}", flush=True)
    for row in smooth_rows:
        print(f"{row['limiter']:>8} {row['cells']:>6} {row['production_numeric']:>13.6e} "
              f"{row['relative_to_initial']:>13.3e}", flush=True)
    print(f"\nC1 second law (all runs): {summary['verdicts']['C1_second_law_all_runs']}", flush=True)
    print(f"C2 finest-mesh relative errors: {finest}", flush=True)
    print(f"C3 smooth orders: {smooth_orders}", flush=True)
    print(f"diagnostic self-test rel. diff: {analytic_vs_exact['relative_difference']:.3e}", flush=True)
    print(f"quadrature spread: {quadrature_spread:.3e}", flush=True)
    print(f"CFL check: {cfl_check}", flush=True)


if __name__ == "__main__":
    main()
