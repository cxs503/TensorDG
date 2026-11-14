#!/usr/bin/env python
"""Lightweight p-order convergence study for linear advection (1D & 2D).

Produces machine-readable JSON artifacts and publication-quality figures
for the CASE-DG-ADVECT-ORDER anchor.

This is a lightweight version designed to run under high load.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tensordg.advection_1d import solve_advection, l2_error as l2_error_1d
from tensordg.advection_2d_var import DGAdvection2D, trig_profile
from tensordg.convergence import convergence_table


def _constant_velocity(ax: float, ay: float):
    def velocity(x, y, t):
        del t
        return torch.full_like(x, ax), torch.full_like(y, ay)
    return velocity


def study_1d():
    """1-D periodic linear advection p-order convergence study."""
    results = {}
    for p in [0, 1, 2, 3, 4]:
        if p == 0:
            cells_list = [20, 40, 80, 160]
        elif p <= 2:
            cells_list = [10, 20, 40, 80]
        else:
            cells_list = [8, 16, 32, 64]

        errors = []
        meta_list = []
        for n in cells_list:
            centers, coeff, meta = solve_advection(
                ncells=n, degree=p, final_time=1.0,
                speed=1.0, cfl=0.08, time_integrator="rk4",
            )
            err = l2_error_1d(centers, coeff, 1.0, speed=1.0)
            errors.append(err)
            meta_list.append({"ncells": n, "dt": meta["dt"], "steps": meta["steps"]})
            print(f"  1D P{p} N={n}: err={err:.6e}")

        tbl = convergence_table(cells_list, errors, theoretical_order=p + 1)
        results[f"p{p}"] = {
            "degree": p, "cells": cells_list, "errors_l2": errors,
            "observed_order_lsq": tbl["observed_order_lsq"],
            "pairwise_orders": tbl["pairwise_orders"],
            "theoretical_order": p + 1,
            "order_deviation": tbl["order_deviation"],
            "order_within_tolerance": tbl.get("order_within_tolerance", None),
            "runs": meta_list,
        }
    return results


def study_2d():
    """2-D periodic linear advection p-order convergence study (lightweight)."""
    vel_fn = _constant_velocity(1.0, 0.5)
    results = {}
    for p in [0, 1, 2, 3]:
        if p == 0:
            cells_list = [8, 16, 32]
        elif p == 1:
            cells_list = [8, 16, 32, 64]
        elif p == 2:
            cells_list = [6, 12, 24]
        else:
            cells_list = [6, 12, 24]

        errors = []
        meta_list = []
        for n in cells_list:
            solver = DGAdvection2D(p, n)
            coeff0 = solver.project(trig_profile)
            result = solver.solve(coeff0, vel_fn, 1.0, cfl=0.08)
            coeff = result["coeff"]
            err = solver.l2_error(coeff, trig_profile, 1.0)
            errors.append(err["absolute_l2"])
            meta_list.append({
                "ncells": n, "dt": result["meta"]["dt"],
                "steps": result["meta"]["steps"],
                "mass_drift_rel": result["meta"]["mass_drift_rel"],
            })
            print(f"  2D P{p} N={n}: err={err['absolute_l2']:.6e}")

        tbl = convergence_table(cells_list, errors, theoretical_order=p + 1)
        results[f"p{p}"] = {
            "degree": p, "cells": cells_list, "errors_l2": errors,
            "observed_order_lsq": tbl["observed_order_lsq"],
            "pairwise_orders": tbl["pairwise_orders"],
            "theoretical_order": p + 1,
            "order_deviation": tbl["order_deviation"],
            "order_within_tolerance": tbl.get("order_within_tolerance", None),
            "runs": meta_list,
        }
    return results


def generate_figures_1d(results_1d, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]

    for idx, (key, data) in enumerate(sorted(results_1d.items())):
        p = data["degree"]
        cells = np.array(data["cells"], dtype=float)
        errs = np.array(data["errors_l2"])
        ax.loglog(cells, errs, f"{markers[idx]}-", color=colors[idx],
                  label=f"P{p} (obs={data['observed_order_lsq']:.3f}, theo={p+1})")
        ref_c = cells[0]
        ref_e = errs[0]
        ref_cells = np.array([ref_c, ref_c * 4])
        ref_errs = ref_e * (ref_cells / ref_c) ** (-(p + 1))
        ax.loglog(ref_cells, ref_errs, "--", color=colors[idx], alpha=0.4,
                  label=f"  ref slope {p+1}")

    ax.set_xlabel("Number of cells (N)", fontsize=12)
    ax.set_ylabel("L2 error", fontsize=12)
    ax.set_title("1-D Linear Advection: p-Order Convergence (DG + RK4)\n"
                 r"$u_t + u_x = 0$, periodic, $u(x,0)=\sin(2\pi x)$", fontsize=13)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "advection_1d_porder_convergence.png", dpi=150)
    plt.close(fig)


def generate_figures_2d(results_2d, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^", "D"]

    for idx, (key, data) in enumerate(sorted(results_2d.items())):
        p = data["degree"]
        cells = np.array(data["cells"], dtype=float)
        errs = np.array(data["errors_l2"])
        ax.loglog(cells, errs, f"{markers[idx]}-", color=colors[idx],
                  label=f"P{p} (obs={data['observed_order_lsq']:.3f}, theo={p+1})")
        ref_c = cells[0]
        ref_e = errs[0]
        ref_cells = np.array([ref_c, ref_c * 4])
        ref_errs = ref_e * (ref_cells / ref_c) ** (-(p + 1))
        ax.loglog(ref_cells, ref_errs, "--", color=colors[idx], alpha=0.4,
                  label=f"  ref slope {p+1}")

    ax.set_xlabel("Number of cells per direction (N)", fontsize=12)
    ax.set_ylabel("L2 error", fontsize=12)
    ax.set_title("2-D Linear Advection: p-Order Convergence (DG + RK4)\n"
                 r"$u_t + \mathbf{a}\cdot\nabla u = 0$, $\mathbf{a}=(1,0.5)$, periodic", fontsize=13)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "advection_2d_porder_convergence.png", dpi=150)
    plt.close(fig)


def main():
    outdir = ROOT / "docs" / "benchmarks" / "results" / "linear-advection-porder"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("1-D Linear Advection p-Order Convergence Study")
    print("=" * 60)
    results_1d = study_1d()
    for key, data in sorted(results_1d.items()):
        p = data["degree"]
        print(f"  P{p}: observed={data['observed_order_lsq']:.4f}, "
              f"theoretical={p+1}, deviation={data['order_deviation']:.4f}, "
              f"within_tol={data['order_within_tolerance']}")

    print()
    print("=" * 60)
    print("2-D Linear Advection p-Order Convergence Study")
    print("=" * 60)
    results_2d = study_2d()
    for key, data in sorted(results_2d.items()):
        p = data["degree"]
        print(f"  P{p}: observed={data['observed_order_lsq']:.4f}, "
              f"theoretical={p+1}, deviation={data['order_deviation']:.4f}, "
              f"within_tol={data['order_within_tolerance']}")

    generate_figures_1d(results_1d, outdir)
    generate_figures_2d(results_2d, outdir)

    artifact = {
        "benchmark": "linear_advection_porder",
        "anchor": "CASE-DG-ADVECT-ORDER",
        "repo": "TensorDG",
        "method": "DG modal Legendre + upwind flux + RK4",
        "criterion": "observed_order >= theoretical_order - 0.3 (tolerance=0.3)",
        "study_1d": {
            "problem": "u_t + u_x = 0 on [0,1] periodic, u(x,0)=sin(2πx)",
            "degrees": list(results_1d.keys()),
            "results": results_1d,
        },
        "study_2d": {
            "problem": "u_t + a·∇u = 0 on [0,1]² periodic, a=(1,0.5), u(x,y,0)=1+sin(2πx)sin(2πy)",
            "degrees": list(results_2d.keys()),
            "results": results_2d,
        },
        "figures": [
            {"name": "advection_1d_porder_convergence.png",
             "caption": "1-D linear advection L2 error vs resolution for P0-P4"},
            {"name": "advection_2d_porder_convergence.png",
             "caption": "2-D linear advection L2 error vs resolution for P0-P3"},
        ],
        "verdict": {
            "all_1d_pass": all(d["order_within_tolerance"] for d in results_1d.values()),
            "all_2d_pass": all(d["order_within_tolerance"] for d in results_2d.values()),
            "overall": "PASS" if (
                all(d["order_within_tolerance"] for d in results_1d.values()) and
                all(d["order_within_tolerance"] for d in results_2d.values())
            ) else "FAIL",
        },
    }

    artifact_path = outdir / "porder_convergence.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nArtifact written to {artifact_path}")
    print(f"Overall verdict: {artifact['verdict']['overall']}")
    return artifact


if __name__ == "__main__":
    main()
