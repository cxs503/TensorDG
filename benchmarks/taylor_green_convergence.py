#!/usr/bin/env python
"""Taylor-Green vortex convergence study for the 2-D compressible NS DG solver.

Reference: Taylor, G. I. & Green, A. E. (1937) "Mechanism of the production
of small eddies from large ones", Proc. R. Soc. Lond. A 158:499-521.

The incompressible exact solution (low-Mach limit) is:
    u(x,y,t) = -A cos(kx) sin(ky) exp(-2 nu k^2 t)
    v(x,y,t) =  A sin(kx) cos(ky) exp(-2 nu k^2 t)

This benchmark verifies that the compressible DG Navier-Stokes solver
reproduces this solution at the expected order of accuracy in the
low-Mach, low-viscosity limit.

Usage:
    PYTHONPATH=. python -m benchmarks.taylor_green_convergence --output-dir runs/taylor_green
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

from tensordg.ns_dg_2d import (
    legendre_operators,
    solve_taylor_green_dg_2d,
    taylor_green_exact,
)

# Default parameters
AMPLITUDE = 0.1          # Low Mach number
WAVENUMBER = 2.0 * math.pi
MU = 0.01                # Dynamic viscosity (rho=1 => nu=mu)
FINAL_TIME = 0.05
CFL = 0.3
CFL_VISCOUS = 1.0


def compute_error(ncells, degree, final_time=FINAL_TIME, amplitude=AMPLITUDE, mu=MU):
    """Run TGV and compute relative L2 error vs exact incompressible solution."""
    x, y, coeff, meta = solve_taylor_green_dg_2d(
        ncells=ncells, degree=degree, final_time=final_time,
        amplitude=amplitude, mu=mu, cfl=CFL, cfl_viscous=CFL_VISCOUS,
    )

    # Build quadrature grid
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, max(degree + 2, 3)
    )
    nx, ny = ncells
    dx, dy = 1.0 / nx, 1.0 / ny
    xc = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    yc = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy

    # Quadrature points
    xq = xc[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = yc[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]

    # Evaluate DG solution at quadrature points
    nodal = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    rho = nodal[..., 0]
    u_num = nodal[..., 1] / rho.clamp_min(1e-12)
    v_num = nodal[..., 2] / rho.clamp_min(1e-12)

    # Exact incompressible solution
    nu = mu  # rho = 1
    exact = taylor_green_exact(xq, yq, final_time, amplitude, WAVENUMBER, nu, 1.0)
    u_ex = exact["velocity_x"]
    v_ex = exact["velocity_y"]

    # Relative L2 errors
    err_u = float(torch.sqrt(((u_num - u_ex) ** 2).sum()) / torch.sqrt((u_ex ** 2).sum()))
    err_v = float(torch.sqrt(((v_num - v_ex) ** 2).sum()) / torch.sqrt((v_ex ** 2).sum()))

    # Mass conservation check
    w2 = weights[None, None, :, None] * weights[None, None, None, :]
    mass = float(0.25 * dx * dy * (w2 * rho).sum())

    return {
        "ncells": list(ncells),
        "degree": degree,
        "dofs": nx * ny * (degree + 1) ** 2,
        "relative_l2_u": err_u,
        "relative_l2_v": err_v,
        "relative_l2": max(err_u, err_v),
        "mass": mass,
        "mass_drift": abs(mass - 1.0),
        "steps": meta["steps"],
        "cfl_used": meta.get("cfl", CFL),
    }


def run_convergence_study(output_dir):
    """Run h-refinement study for P1 and P2."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    resolutions = [(4, 4), (8, 8), (16, 16), (32, 32)]
    degrees = [1, 2]

    all_results = []
    for degree in degrees:
        for ncells in resolutions:
            print(f"Running ncells={ncells}, degree={degree}...")
            sys.stdout.flush()
            result = compute_error(ncells, degree)
            all_results.append(result)
            print(f"  err={result['relative_l2']:.6e}, steps={result['steps']}")
            sys.stdout.flush()

    # Compute EOC
    for degree in degrees:
        deg_results = [r for r in all_results if r["degree"] == degree]
        print(f"\n--- P{degree} convergence ---")
        for i in range(1, len(deg_results)):
            coarse = deg_results[i - 1]
            fine = deg_results[i]
            ratio = fine["ncells"][0] / coarse["ncells"][0]
            eoc = math.log(coarse["relative_l2"] / fine["relative_l2"]) / math.log(ratio)
            print(f"  {coarse['ncells']} -> {fine['ncells']}: EOC = {eoc:.2f}")

    # Save results
    results_path = out / "taylor_green_results.json"
    results_path.write_text(json.dumps({
        "schema": "tensordg-taylor-green/1",
        "case": "CASE-DG-TAYLOR-GREEN-VORTEX",
        "reference": {
            "authors": "Taylor & Green",
            "year": 1937,
            "journal": "Proc. R. Soc. Lond. A",
            "volume": 158,
            "pages": "499-521",
            "key_result": "Decaying vortex array with exact incompressible solution",
        },
        "problem": {
            "equation": "Compressible Navier-Stokes (low-Mach limit)",
            "domain": "[0, 1]^2 periodic",
            "initial_condition": "Taylor-Green vortex",
            "amplitude": AMPLITUDE,
            "wavenumber": WAVENUMBER,
            "mu": MU,
            "final_time": FINAL_TIME,
            "exact_solution": "Incompressible TGV (low-Mach approximation)",
        },
        "results": all_results,
        "parameters": {
            "amplitude": AMPLITUDE,
            "wavenumber": WAVENUMBER,
            "mu": MU,
            "final_time": FINAL_TIME,
            "cfl": CFL,
            "cfl_viscous": CFL_VISCOUS,
        },
    }, indent=2))

    print(f"\nResults saved to {results_path}")
    return all_results


def generate_figures(output_dir):
    """Generate convergence plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    results_path = out / "taylor_green_results.json"
    if not results_path.exists():
        print("No results file found; run convergence study first.")
        return

    data = json.loads(results_path.read_text())
    results = data["results"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for degree in [1, 2]:
        deg_results = [r for r in results if r["degree"] == degree]
        dofs = [r["dofs"] for r in deg_results]
        errs = [r["relative_l2"] for r in deg_results]

        ax.loglog(dofs, errs, "o-", label=f"P{degree}", markersize=8)

        # Reference slope
        if len(dofs) >= 2:
            ref_dofs = np.array([dofs[0], dofs[-1]])
            ref_errs = np.array([errs[0], errs[0] * (ref_dofs[0] / ref_dofs[-1]) ** (degree + 1)])
            ax.loglog(ref_dofs, ref_errs, "k--", alpha=0.3,
                      label=f"O(h^{degree + 1}) ref")

    ax.set_xlabel("Number of DOFs", fontsize=12)
    ax.set_ylabel("Relative L2 Error (velocity)", fontsize=12)
    ax.set_title("Taylor-Green Vortex: Convergence Study\n"
                 f"Compressible NS DG, A={AMPLITUDE}, mu={MU}, t={FINAL_TIME}",
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig_path = out / "taylor_green_convergence.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Taylor-Green vortex convergence study")
    parser.add_argument("--output-dir", default="runs/taylor_green",
                        help="Output directory for results and figures")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    results = run_convergence_study(args.output_dir)
    if not args.no_plot:
        generate_figures(args.output_dir)
