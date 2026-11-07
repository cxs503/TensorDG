"""Ritter (1892) dam-break benchmark for the 1-D shallow water solver.

Exact self-similar solution for a dam break over a flat, dry bed:

    Initial:  h = h0 for x < 0,  h = 0 for x > 0,  u = 0
    Bed:      z_b = 0 (flat)
    Gravity:  g = 9.81 m/s²

    Let c0 = sqrt(g * h0).  At time t > 0:

    Region 1 (undisturbed, x ≤ -c0·t):
        h = h0,  u = 0

    Region 2 (rarefaction fan, -c0·t < x < 2·c0·t):
        h = (1/(9g)) · (2·c0 - x/t)²
        u = (2/3) · (c0 + x/t)

    Region 3 (dry, x ≥ 2·c0·t):
        h = 0,  u = 0

The wet front propagates at speed 2·c0.  The solution is continuous in h
and u across the fan, but the derivative is discontinuous at the edges.

Reference: Ritter, A. (1892), "Über die Fortpflanzung von Wasserwellen."
           Z. Ver. deut. Ing. 36, 947–954.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensordg.swe_1d import GRAVITY, reconstruct_fields, solve_swe_1d


def ritter_exact(
    x: np.ndarray, t: float, h0: float, g: float = GRAVITY,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Ritter solution at time t.

    Returns (h_exact, u_exact) at the given positions.
    """
    c0 = math.sqrt(g * h0)
    h = np.zeros_like(x, dtype=np.float64)
    u = np.zeros_like(x, dtype=np.float64)

    # Region 1: undisturbed
    mask1 = x <= -c0 * t
    h[mask1] = h0
    u[mask1] = 0.0

    # Region 2: rarefaction fan
    mask2 = (x > -c0 * t) & (x < 2.0 * c0 * t)
    xi = x[mask2] / t  # self-similar variable
    h[mask2] = (1.0 / (9.0 * g)) * (2.0 * c0 - xi) ** 2
    u[mask2] = (2.0 / 3.0) * (c0 + xi)

    # Region 3: dry (already zero)

    return h, u


def run_ritter(
    ncells: int = 200,
    degree: int = 1,
    h0: float = 1.0,
    final_time: float = 0.5,
    domain: tuple = (-5.0, 5.0),
    cfl: float = 0.15,
    flux_kind: str = "hll",
    limiter: str | None = None,
    rk_order: int = 3,
    dry_tol: float = 1e-8,
) -> dict:
    """Run the Ritter dam break and compare with the exact solution.

    Returns a dict with metrics and the numerical/exact profiles.
    """
    c0 = math.sqrt(GRAVITY * h0)

    def init_h(x):
        return torch.where(x < 0.0, torch.full_like(x, h0), torch.full_like(x, dry_tol))

    def init_u(x):
        return torch.zeros_like(x)

    def zb(x):
        return torch.zeros_like(x)

    t0 = time.perf_counter()
    x, coeff, zb_faces, meta = solve_swe_1d(
        init_h, init_u, zb,
        ncells=ncells, degree=degree,
        final_time=final_time, domain=domain,
        cfl=cfl, flux_kind=flux_kind, limiter=limiter,
        rk_order=rk_order, device="cpu", dtype=torch.float64,
    )
    wall_time = time.perf_counter() - t0

    h_avg, u_avg, eta = reconstruct_fields(coeff, zb_faces)
    x_np = x.numpy()
    h_num = h_avg.numpy()
    u_num = u_avg.numpy()

    h_ex, u_ex = ritter_exact(x_np, final_time, h0)

    # Only compare in the wet region (h > dry_tol)
    wet = h_ex > dry_tol
    n_wet = int(wet.sum())

    # L1 and L2 errors (wet region only)
    if n_wet > 0:
        l1_h = float(np.mean(np.abs(h_num[wet] - h_ex[wet])))
        l2_h = float(np.sqrt(np.mean((h_num[wet] - h_ex[wet]) ** 2)))
        l1_u = float(np.mean(np.abs(u_num[wet] - u_ex[wet])))
        l2_u = float(np.sqrt(np.mean((u_num[wet] - u_ex[wet]) ** 2)))
        rel_l1_h = l1_h / float(np.mean(h_ex[wet]))
        rel_l2_h = l2_h / float(np.sqrt(np.mean(h_ex[wet] ** 2)))
        # For velocity, use max of exact u as reference scale
        u_scale = float(np.max(np.abs(u_ex[wet]))) if np.max(np.abs(u_ex[wet])) > 1e-12 else 1.0
        rel_l1_u = l1_u / u_scale
        rel_l2_u = l2_u / u_scale
    else:
        l1_h = l2_h = l1_u = l2_u = rel_l1_h = rel_l2_h = rel_l1_u = rel_l2_u = float("nan")

    # Wet front position: find where h drops below dry_tol
    wet_front_num = float(x_np[h_num > dry_tol].max()) if (h_num > dry_tol).any() else float("nan")
    wet_front_exact = 2.0 * c0 * final_time

    # Mass conservation: integral of h should equal initial mass
    dx = (domain[1] - domain[0]) / ncells
    mass_num = float(np.sum(h_num) * dx)
    mass_init = h0 * abs(domain[0])  # only left side has water initially
    mass_error = abs(mass_num - mass_init) / mass_init

    # Positivity check
    h_min = float(h_num.min())
    h_max = float(h_num.max())

    return {
        "ncells": ncells,
        "degree": degree,
        "h0": h0,
        "final_time": final_time,
        "domain": list(domain),
        "cfl": cfl,
        "flux_kind": flux_kind,
        "limiter": limiter,
        "rk_order": rk_order,
        "steps": meta["steps"],
        "wall_time": round(wall_time, 3),
        "wet_cells": n_wet,
        "l1_h": l1_h,
        "l2_h": l2_h,
        "l1_u": l1_u,
        "l2_u": l2_u,
        "rel_l1_h": rel_l1_h,
        "rel_l2_h": rel_l2_h,
        "rel_l1_u": rel_l1_u,
        "rel_l2_u": rel_l2_u,
        "wet_front_num": wet_front_num,
        "wet_front_exact": wet_front_exact,
        "wet_front_error_cells": abs(wet_front_num - wet_front_exact) / dx if not np.isnan(wet_front_num) else float("nan"),
        "mass_num": mass_num,
        "mass_init": mass_init,
        "mass_error": mass_error,
        "h_min": h_min,
        "h_max": h_max,
        "x": x_np.tolist(),
        "h_num": h_num.tolist(),
        "u_num": u_num.tolist(),
        "h_ex": h_ex.tolist(),
        "u_ex": u_ex.tolist(),
    }


def convergence_study(
    h0: float = 1.0,
    final_time: float = 0.5,
    degrees: list = (1, 2),
    ncells_list: list = (50, 100, 200, 400),
) -> dict:
    """Run a convergence study for the Ritter dam break.

    Returns a dict with error sequences and observed orders.
    """
    results = {}
    for degree in degrees:
        errors_h = []
        errors_u = []
        for ncells in ncells_list:
            r = run_ritter(ncells=ncells, degree=degree, h0=h0, final_time=final_time)
            errors_h.append(r["rel_l1_h"])
            errors_u.append(r["rel_l1_u"])

        eoc_h = [
            math.log(errors_h[i] / errors_h[i + 1]) / math.log(ncells_list[i + 1] / ncells_list[i])
            for i in range(len(errors_h) - 1)
        ]
        eoc_u = [
            math.log(errors_u[i] / errors_u[i + 1]) / math.log(ncells_list[i + 1] / ncells_list[i])
            for i in range(len(errors_u) - 1)
        ]

        results[f"P{degree}"] = {
            "ncells": ncells_list,
            "rel_l1_h": errors_h,
            "rel_l1_u": errors_u,
            "eoc_h": eoc_h,
            "eoc_u": eoc_u,
        }

    return results


def main():
    """Run the Ritter dam break benchmark and save results."""
    import argparse

    parser = argparse.ArgumentParser(description="Ritter dam break benchmark")
    parser.add_argument("--ncells", type=int, default=200)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--h0", type=float, default=1.0)
    parser.add_argument("--time", type=float, default=0.5)
    parser.add_argument("--convergence", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.convergence:
        results = convergence_study(h0=args.h0, final_time=args.time)
        print("\nRitter dam break convergence study")
        print("=" * 60)
        for deg, data in results.items():
            print(f"\n{deg}:")
            print(f"  ncells:  {data['ncells']}")
            print(f"  rel L1 h: {[f'{e:.4%}' for e in data['rel_l1_h']]}")
            print(f"  EOC h:    {[f'{e:.2f}' for e in data['eoc_h']]}")
            print(f"  rel L1 u: {[f'{e:.4%}' for e in data['rel_l1_u']]}")
            print(f"  EOC u:    {[f'{e:.2f}' for e in data['eoc_u']]}")

        if args.output_dir:
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "ritter_convergence.json", "w") as f:
                json.dump(results, f, indent=2)
    else:
        result = run_ritter(
            ncells=args.ncells, degree=args.degree, h0=args.h0, final_time=args.time,
        )
        print(f"\nRitter dam break (P{result['degree']}, {result['ncells']} cells, t={result['final_time']})")
        print(f"  Steps: {result['steps']}")
        print(f"  Wall time: {result['wall_time']:.2f} s")
        print(f"  Wet cells: {result['wet_cells']}")
        print(f"  rel L1 h: {result['rel_l1_h']:.4%}")
        print(f"  rel L2 h: {result['rel_l2_h']:.4%}")
        print(f"  rel L1 u: {result['rel_l1_u']:.4%}")
        print(f"  rel L2 u: {result['rel_l2_u']:.4%}")
        print(f"  Wet front error: {result['wet_front_error_cells']:.2f} cells")
        print(f"  Mass error: {result['mass_error']:.4%}")
        print(f"  h range: [{result['h_min']:.2e}, {result['h_max']:.2f}]")

        if args.output_dir:
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "ritter_metrics.json", "w") as f:
                json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
