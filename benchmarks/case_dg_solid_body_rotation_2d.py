#!/usr/bin/env python3
"""CASE-DG-SOLID-BODY-ROTATION-2D benchmark.

LeVeque 2002, *Finite Volume Methods for Hyperbolic Problems*, Ch. 19.
DOI: 10.1017/cbo9780511791253

Solid-body rotation of a smooth bump on the periodic unit square.
After one full revolution (T = 1 for omega = 2*pi), the numerical solution
is compared against the initial condition.

Acceptance criteria (from mission total ledger):
  - Grid: 100 x 100
  - CFL: 0.5
  - Metric: L1 error and extremum (peak) decay vs initial value after one rotation
  - Tolerance: < 5%

The rigid-rotation velocity field ``a = omega * (-(y-1/2), x-1/2)`` is
discontinuous across the periodic seam, so the error norm is restricted to
a disk of radius 0.28 about the centre (0.5, 0.5), which contains the bump
trajectory at radius 0.15.  This is the same disk norm used by the existing
``test_advection_2d_var`` gates.

Usage:
    python benchmarks/case_dg_solid_body_rotation_2d.py [--degree D] [--ncells N] [--cfl C]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from tensordg.advection_2d_var import (
    DGAdvection2D,
    gaussian_bump,
    rotated_profile,
    smooth_bump,
    solid_body_rotation,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DISK_RADIUS = 0.28
CENTER = (0.5, 0.5)
OMEGA = 2.0 * math.pi  # one revolution in T=1
TOLERANCE = 0.05  # 5%


def run_case(
    degree: int = 2,
    ncells: int = 100,
    cfl: float = 0.5,
    profile: str = "gaussian",
    device: str = "cpu",
) -> dict:
    """Run the solid-body rotation benchmark and return metrics."""
    t0 = time.time()

    solver = DGAdvection2D(degree=degree, ncells=ncells, device=device)

    if profile == "gaussian":
        init_profile = gaussian_bump
    elif profile == "smooth_bump":
        init_profile = smooth_bump
    else:
        raise ValueError(f"Unknown profile: {profile}")

    coeff0 = solver.project(init_profile)
    peak0_avg, peak0_nodal = solver.peak(coeff0)
    mass0 = solver.mass(coeff0)

    velocity = solid_body_rotation(omega=OMEGA)
    result = solver.solve(coeff0, velocity, final_time=1.0, cfl=cfl, record_mass=True)

    coeff1 = result["coeff"]
    peak1_avg, peak1_nodal = solver.peak(coeff1)
    mass1 = solver.mass(coeff1)

    exact = rotated_profile(init_profile, OMEGA)

    # L1 error (disk-restricted)
    l1 = solver.l1_error(coeff1, exact, t=1.0, radius=DISK_RADIUS, center=CENTER)

    # L2 error (disk-restricted) for completeness
    l2 = solver.l2_error(coeff1, exact, t=1.0, radius=DISK_RADIUS, center=CENTER)

    elapsed = time.time() - t0

    return {
        "degree": degree,
        "ncells": ncells,
        "cfl": cfl,
        "profile": profile,
        "device": device,
        "steps": result["meta"]["steps"],
        "dt": result["meta"]["dt"],
        "dofs": result["meta"]["dofs"],
        "mass_initial": mass0,
        "mass_final": mass1,
        "mass_drift_rel": result["meta"]["mass_drift_rel"],
        "peak_initial_avg": peak0_avg,
        "peak_final_avg": peak1_avg,
        "peak_retention_avg": peak1_avg / peak0_avg if peak0_avg else 0.0,
        "peak_initial_nodal": peak0_nodal,
        "peak_final_nodal": peak1_nodal,
        "peak_retention_nodal": peak1_nodal / peak0_nodal if peak0_nodal else 0.0,
        "l1_absolute": l1["absolute_l1"],
        "l1_reference": l1["reference_l1"],
        "l1_relative": l1["relative_l1"],
        "l2_absolute": l2["absolute_l2"],
        "l2_reference": l2["reference_l2"],
        "l2_relative": l2["relative_l2"],
        "elapsed_s": elapsed,
    }


def convergence_study(degrees: list[int] = (1, 2, 3), ncells_list: list[int] = (25, 50, 100),
                      cfl: float = 0.5, profile: str = "gaussian") -> list[dict]:
    """Run a convergence study across degrees and resolutions."""
    results = []
    for deg in degrees:
        for n in ncells_list:
            print(f"  Running degree={deg}, ncells={n} ...", flush=True)
            r = run_case(degree=deg, ncells=n, cfl=cfl, profile=profile)
            results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser(description="CASE-DG-SOLID-BODY-ROTATION-2D benchmark")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree (default: 2)")
    parser.add_argument("--ncells", type=int, default=100, help="Grid resolution (default: 100)")
    parser.add_argument("--cfl", type=float, default=0.5, help="CFL number (default: 0.5)")
    parser.add_argument("--profile", choices=["gaussian", "smooth_bump"], default="gaussian")
    parser.add_argument("--convergence", action="store_true", help="Run convergence study")
    parser.add_argument("--device", default="cpu", help="Device (default: cpu)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.convergence:
        print("=== Convergence Study ===")
        results = convergence_study(
            degrees=[1, 2, 3],
            ncells_list=[25, 50, 100],
            cfl=args.cfl,
            profile=args.profile,
        )
        # Compute EOC
        for deg in [1, 2, 3]:
            deg_results = [r for r in results if r["degree"] == deg]
            if len(deg_results) >= 2:
                errs = [r["l1_relative"] for r in deg_results]
                ns = [r["ncells"] for r in deg_results]
                eoc = math.log(errs[0] / errs[-1]) / math.log(ns[-1] / ns[0])
                print(f"  P{deg}: L1 rel error {errs[0]:.4e} -> {errs[-1]:.4e}, EOC={eoc:.2f}")
        output = {"convergence": results}
    else:
        print(f"=== CASE-DG-SOLID-BODY-ROTATION-2D ===")
        print(f"  degree={args.degree}, ncells={args.ncells}, cfl={args.cfl}, profile={args.profile}")
        result = run_case(
            degree=args.degree,
            ncells=args.ncells,
            cfl=args.cfl,
            profile=args.profile,
            device=args.device,
        )
        output = {"case": result}

        # Print results
        print(f"  Steps: {result['steps']}, dt: {result['dt']:.6f}")
        print(f"  DOFs: {result['dofs']}")
        print(f"  Mass drift (rel): {result['mass_drift_rel']:.2e}")
        print(f"  Peak retention (avg): {result['peak_retention_avg']:.6f}")
        print(f"  Peak retention (nodal): {result['peak_retention_nodal']:.6f}")
        print(f"  L1 relative error: {result['l1_relative']:.6f}")
        print(f"  L2 relative error: {result['l2_relative']:.6f}")
        print(f"  Elapsed: {result['elapsed_s']:.1f}s")

        # Acceptance check
        l1_pass = result["l1_relative"] < TOLERANCE
        peak_pass = result["peak_retention_nodal"] > (1.0 - TOLERANCE)
        print(f"\n  Acceptance (L1 < {TOLERANCE:.0%}): {'PASS' if l1_pass else 'FAIL'} ({result['l1_relative']:.4f})")
        print(f"  Acceptance (peak retention > {1-TOLERANCE:.0%}): {'PASS' if peak_pass else 'FAIL'} ({result['peak_retention_nodal']:.4f})")

    # Save output
    out_path = args.output or "artifacts/case_dg_solid_body_rotation_2d.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
