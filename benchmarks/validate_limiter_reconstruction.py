"""Literature-grounded validation of the TensorDG TVB / WENO limiters.

Two independent blocks:

Part A -- *analytic* reconstruction consistency (no time integration).  Cell
    averages of a prescribed smooth function are fed straight into the modal
    coefficients, the limiter is applied once, and the modified mode-1
    coefficient is compared with the analytic derivative through the exact
    modal identity ``a_1 = (dx/2) * du/dx`` (normalised Legendre basis on
    ``[-1, 1]``).  Reference: Cockburn & Shu (1989) for the TVB criterion,
    Qiu & Shu (2005) for the WENO reconstruction, LeVeque (2002, ch. 6) for
    the minmod slope.

Part B -- shock-tube errors against the exact Riemann solution (Sod 1978,
    Lax 1954).  Compares ``limiter=None``, ``tvb`` and ``weno`` on the same
    grid: relative L2 errors, total variation, overshoot and mass budget.

Everything printed or plotted is written programmatically to
``artifacts/limiter_validation/<tag>.json``; no number is transcribed by hand.

Usage
-----
    PYTHONPATH=. python benchmarks/validate_limiter_reconstruction.py \
        --tag before          # or: --tag after
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Tiny 1-D problems: PyTorch's default 128-thread pool costs far more than it
# saves here, so the benchmark is pinned to a single thread unless overridden.
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))

from benchmarks.riemann_dg_error import riemann_dg_errors
from benchmarks.riemann_exact import cell_averages as exact_cell_averages
from benchmarks.riemann_exact import sample as exact_sample
from tensordg.euler_dg_1d import solve_euler_dg_1d
from tensordg.limiters import (
    _troubled_cell_indicator,
    apply_tvb_limiter,
    apply_weno_limiter,
)

LEGENDRE_LIMIT = 4.0  # |du/dx| floor used when normalising relative errors


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def cell_averages(fn, ncells: int, domain=(0.0, 1.0), order: int = 10):
    """Cell averages and centres of ``fn`` (Gauss quadrature, exact for polys)."""
    lower, upper = domain
    dx = (upper - lower) / ncells
    centres = lower + (np.arange(ncells) + 0.5) * dx
    nodes, weights = np.polynomial.legendre.leggauss(order)
    points = centres[:, None] + 0.5 * dx * nodes[None, :]
    averages = 0.5 * np.sum(weights[None, :] * fn(points), axis=1)
    return centres, averages, dx


def modal_from_averages(averages: np.ndarray, slope: np.ndarray | None, dx: float):
    """Build a (ncells, 3, 1) modal tensor with mode-0 = average, mode-1 = (dx/2)*slope."""
    ncells = averages.size
    coeff = np.zeros((ncells, 3, 1))
    coeff[:, 0, 0] = averages
    if slope is not None:
        coeff[:, 1, 0] = 0.5 * dx * slope
    return coeff


def _to_tensor(array):
    import torch
    return torch.as_tensor(array, dtype=torch.float64)


def reconstruction_metrics(coeff_np, limited_np, dx, slope_true, troubled, tol=1e-12):
    """Quantify how the limited mode-1 coefficient compares with the analytic slope."""
    a1_limited = limited_np[:, 1, 0]
    a1_expected = 0.5 * dx * slope_true
    interior = troubled.copy()
    interior[0] = interior[-1] = False
    if not interior.any():
        return {"troubled_cells": int(troubled.sum()), "interior_troubled_cells": 0}
    rel = np.abs(a1_limited[interior] - a1_expected[interior]) / np.maximum(
        np.abs(a1_expected[interior]), tol
    )
    return {
        "troubled_cells": int(troubled.sum()),
        "interior_troubled_cells": int(interior.sum()),
        "max_relative_error_percent": float(100.0 * rel.max()),
        "mean_relative_error_percent": float(100.0 * rel.mean()),
    }


def interface_slack(coeff_np, limited_np, dx):
    """Monotonicity-preserving slack of the limited reconstruction (LeVeque 2002).

    For a monotone increasing cell-average sequence the limited polynomial must
    stay inside the neighbouring averages at both interfaces.  The slack is
    returned normalised by the two-cell average jump: >0 means strict
    monotonicity preservation, 0 means the polynomial touches a neighbour
    average (over-steep reconstruction).
    """
    avg = coeff_np[:, 0, 0]
    ncells = avg.size
    pad = np.concatenate([avg[:1], avg, avg[-1:]])
    u_im1, u_i, u_ip1 = pad[:-2], avg, pad[2:]
    a1 = limited_np[:, 1, 0]
    right = u_i + a1
    left = u_i - a1
    scale = np.abs(u_ip1 - u_im1)
    scale = np.where(scale > 0, scale, 1.0)
    slack = np.minimum(u_ip1 - right, left - u_im1) / scale
    return slack, right, left


# ---------------------------------------------------------------------------
# Part A -- analytic reconstruction consistency
# ---------------------------------------------------------------------------

def part_a(ncells_list=(20, 40, 80, 160)):
    results = {"affine_exactness": {}, "quadratic_slope": {}, "smooth_extremum": {},
               "monotone_slack": {}}

    # A1: affine data must not trigger the detector and must pass through unchanged.
    fn_affine = lambda x: 1.0 + 2.0 * x
    for ncells in (20, 80):
        _, avg, dx = cell_averages(fn_affine, ncells)
        coeff_np = modal_from_averages(avg, np.full(ncells, 2.0), dx)
        coeff = _to_tensor(coeff_np)
        for name, fn in (("tvb", apply_tvb_limiter), ("weno", apply_weno_limiter)):
            troubled = _troubled_cell_indicator(coeff, dx, tvb_constant=0.0).numpy()
            limited = fn(coeff, dx, tvb_constant=0.0).numpy()
            results["affine_exactness"][f"{name}_n{ncells}"] = {
                "troubled_cells": int(troubled.sum()),
                "max_abs_deviation": float(np.abs(limited - coeff_np).max()),
            }

    # A2: smooth concave quadratic with the strict TVD threshold (M = 0).
    #     Troubled cells must get a mode-1 coefficient consistent with the
    #     analytic local derivative; the asymptotic order of that consistency
    #     error is also recorded (first order is expected: minmod uses
    #     two-point differences).
    fn_quad = lambda x: 1.0 + 2.0 * x - 3.0 * x ** 2
    d_quad = lambda x: 2.0 - 6.0 * x
    slope_errors = {}
    for ncells in ncells_list:
        centres, avg, dx = cell_averages(fn_quad, ncells)
        coeff_np = modal_from_averages(avg, d_quad(centres), dx)
        coeff = _to_tensor(coeff_np)
        troubled = _troubled_cell_indicator(coeff, dx, tvb_constant=0.0).numpy()
        for name, fn in (("tvb", apply_tvb_limiter), ("weno", apply_weno_limiter)):
            limited = fn(coeff, dx, tvb_constant=0.0).numpy()
            metrics = reconstruction_metrics(
                coeff_np, limited, dx, d_quad(centres), troubled
            )
            results["quadratic_slope"][f"{name}_n{ncells}"] = metrics
            if "max_relative_error_percent" in metrics:
                slope_errors[(name, ncells)] = metrics["max_relative_error_percent"] / 100.0
    orders = {}
    for name in ("tvb", "weno"):
        ns = [n for n in ncells_list if (name, n) in slope_errors]
        if len(ns) >= 2:
            orders[name] = float(
                np.log(slope_errors[(name, ns[0])] / slope_errors[(name, ns[-1])])
                / np.log(ns[-1] / ns[0])
            )
    results["quadratic_slope"]["observed_order"] = orders

    # A3: smooth extremum.  Cockburn-Shu's tolerance M*dx^2 should let a smooth
    #     extremum through once M exceeds ~|u''|; with M = 0 (strict TVD) the
    #     extremum must be clipped.
    fn_wave = lambda x: 1.0 + 0.5 * np.sin(2.0 * np.pi * x)
    for ncells in (40, 160):
        centres, avg, dx = cell_averages(fn_wave, ncells)
        coeff = _to_tensor(modal_from_averages(avg, None, dx))
        for M in (0.0, 1.0, 10.0, 100.0, 1000.0):
            troubled = _troubled_cell_indicator(coeff, dx, tvb_constant=M).numpy()
            results["smooth_extremum"][f"n{ncells}_M{M:g}"] = {
                "troubled_cells": int(troubled.sum()),
                "threshold_M_dx2": float(M * dx * dx),
            }

    # A4: monotonicity-preserving slack on a monotone sigmoid profile.
    fn_sig = lambda x: 1.0 + 2.0 / (1.0 + np.exp(-20.0 * (x - 0.5)))
    d_sig = lambda x: 2.0 * 20.0 * np.exp(-20.0 * (x - 0.5)) / (1.0 + np.exp(-20.0 * (x - 0.5))) ** 2
    for ncells in (40, 160):
        centres, avg, dx = cell_averages(fn_sig, ncells)
        coeff_np = modal_from_averages(avg, d_sig(centres), dx)
        coeff = _to_tensor(coeff_np)
        troubled = _troubled_cell_indicator(coeff, dx, tvb_constant=0.0).numpy()
        for name, fn in (("tvb", apply_tvb_limiter), ("weno", apply_weno_limiter)):
            limited = fn(coeff, dx, tvb_constant=0.0).numpy()
            slack, right, left = interface_slack(coeff_np, limited, dx)
            interior = troubled.copy()
            interior[0] = interior[-1] = False
            if interior.any():
                results["monotone_slack"][f"{name}_n{ncells}"] = {
                    "troubled_cells": int(troubled.sum()),
                    "min_slack_normalised": float(slack[interior].min()),
                    "cells_touching_neighbour_average": int((slack[interior] <= 1e-12).sum()),
                }
    return results


# ---------------------------------------------------------------------------
# Part B -- shock tube errors against the exact Riemann solution
# ---------------------------------------------------------------------------

def _total_variation(values):
    return float(np.abs(np.diff(values)).sum())


def part_b(cases, ncells=200, degree=2, cfl=0.3):
    results = {}
    profiles = {}
    for case_name, left, right, final_time in cases:
        results[case_name] = {}
        profiles[case_name] = {}
        _, exact_fields = exact_cell_averages(ncells, final_time, left, right)
        for tag, limiter, M in (("none", None, 1.0), ("tvb", "tvb", 1.0), ("weno", "weno", 1.0)):
            x, coeff, meta = solve_euler_dg_1d(
                left, right, ncells=ncells, degree=degree, final_time=final_time,
                cfl=cfl, limiter=limiter, tvb_constant=M,
            )
            errors = riemann_dg_errors(x, coeff, final_time, left, right)
            values = coeff.detach().cpu().numpy()
            avg_density = values[:, 0, 0]
            exact_density = exact_fields["density"]
            overshoot = float(np.max(avg_density) - max(left[0], right[0]))
            undershoot = float(min(left[0], right[0]) - np.min(avg_density))
            dx = 1.0 / ncells
            result = {
                "limiter": tag,
                "steps": meta["steps"],
                "dt": meta["dt"],
                "errors": {k: errors[k]["relative_l2_percent"] for k in errors},
                "density_total_variation": _total_variation(avg_density),
                "exact_density_total_variation": _total_variation(exact_density),
                "density_overshoot": overshoot,
                "density_undershoot": undershoot,
                "mass": float(avg_density.sum() * dx),
                "cell_average_l1_percent": float(
                    100.0 * np.abs(avg_density - exact_density).sum() * dx
                    / (np.abs(exact_density).sum() * dx)
                ),
            }
            results[case_name][tag] = result
            points = x.detach().cpu().numpy()
            true_cell = exact_sample(points, final_time, left, right)
            profiles[case_name][tag] = {
                "x": points, "density": avg_density, "exact_density": true_cell["density"],
                "exact_cell_density": exact_density,
            }
    return results, profiles


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def plot_part_a(part_a_results, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax = axes[0]
    keys = [k for k in part_a_results["quadratic_slope"] if k != "observed_order"]
    labels, values = [], []
    for key in sorted(keys):
        labels.append(key)
        values.append(part_a_results["quadratic_slope"][key].get("max_relative_error_percent", np.nan))
    ax.bar(np.arange(len(values)), values, color=["tab:blue" if "tvb" in l else "tab:green" for l in labels])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.axhline(3.0, color="k", ls="--", lw=1, label="3% tolerance")
    ax.set_ylabel("max rel. error of mode-1 [%]")
    ax.set_title("A2 quadratic: limited slope vs analytic du/dx")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    n = 160
    fn_wave = lambda x: 1.0 + 0.5 * np.sin(2.0 * np.pi * x)
    centres, avg, dx = cell_averages(fn_wave, n)
    coeff = _to_tensor(modal_from_averages(avg, None, dx))
    Ms = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    counts = [int(_troubled_cell_indicator(coeff, dx, tvb_constant=float(M)).sum()) for M in Ms]
    ax.semilogx(np.maximum(Ms, 1e-2), counts, "o-")
    ax.set_xlabel("TVB constant M  (threshold M·dx²)")
    ax.set_ylabel("troubled cells (of %d)" % n)
    ax.set_title("A3 smooth extremum: detector switches off as M grows")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_part_b(part_b_results, profiles, path: Path):
    cases = list(part_b_results)
    fig, axes = plt.subplots(len(cases), 2, figsize=(12, 4 * len(cases)), constrained_layout=True)
    if len(cases) == 1:
        axes = np.asarray(axes).reshape(1, 2)
    for row, case in enumerate(cases):
        ax = axes[row, 0]
        prof = profiles[case]["none"]
        ax.plot(prof["x"], prof["exact_cell_density"], "k-", lw=2, label="exact cell average")
        for tag, color in (("none", "tab:red"), ("tvb", "tab:blue"), ("weno", "tab:green")):
            p = profiles[case][tag]
            err = part_b_results[case][tag]["errors"]["density"]
            ax.plot(p["x"], p["density"], color=color, lw=1, alpha=0.85,
                    label=f"{tag} ({err:.3f}%)")
        ax.set_title(f"{case.upper()} density, {len(profiles[case]['none']['x'])} cells P2, exact cell averages")
        ax.set_xlabel("x")
        ax.set_ylabel("rho")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        tags = ["none", "tvb", "weno"]
        width = 0.26
        for idx, var in enumerate(("density", "velocity", "pressure")):
            vals = [part_b_results[case][t]["errors"][var] for t in tags]
            ax.bar(np.arange(len(tags)) + (idx - 1) * width, vals, width, label=var)
        ax.axhline(3.0, color="k", ls="--", lw=1, label="3% tolerance")
        ax.set_xticks(np.arange(len(tags)))
        ax.set_xticklabels(tags)
        ax.set_yscale("log")
        ax.set_ylabel("relative L2 error [%]")
        ax.set_title(f"{case.upper()}: error by limiter")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="run", help="output tag, e.g. before / after")
    parser.add_argument("--cells", type=int, default=100)
    parser.add_argument("--cfl", type=float, default=0.3)
    parser.add_argument("--output-dir", default="artifacts/limiter_validation")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    a = part_a()
    cases = [
        ("sod", (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.2),
        ("lax", (0.445, 0.698, 3.528), (0.5, 0.0, 0.571), 0.13),
    ]
    b, profiles = part_b(cases, ncells=args.cells, cfl=args.cfl)

    plot_part_a(a, out / f"part_a_{args.tag}.png")
    plot_part_b(b, profiles, out / f"part_b_{args.tag}.png")

    payload = {
        "tag": args.tag,
        "config": {"ncells": args.cells, "degree": 2, "cfl": args.cfl},
        "part_a_analytic": a,
        "part_b_shock_tubes": b,
    }
    path = out / f"{args.tag}.json"
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()