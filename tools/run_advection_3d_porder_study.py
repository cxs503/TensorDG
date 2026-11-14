#!/usr/bin/env python
"""3-D p-order convergence study for linear advection (CASE-DG-ADVECT-ORDER).

Extends the 1-D/2-D convergence verification to 3-D tensor-product DG.
For the linear advection equation u_t + a·∇u = 0 with smooth initial data,
the DG method with polynomial degree p achieves design order p+1 in L2 norm.

This is a lightweight study: P0-P2 with 3 mesh levels each, designed to
run under moderate load.

    PYTHONPATH=. python tools/run_advection_3d_porder_study.py
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

from tensordg.advection_3d import solve_advection_3d, error_3d, mass_3d
from tensordg.convergence import convergence_table

VELOCITY = (1.0, 0.5, 0.25)
ORDER_TOLERANCE = 0.35  # slightly relaxed for 3D pre-asymptotic effects


def study_3d():
    """3-D periodic linear advection p-order convergence study."""
    results = {}
    for p in [0, 1, 2]:
        # Mesh levels: keep small for 3D (memory ~ N^3 * (p+1)^3)
        if p == 0:
            cells_list = [4, 6, 8]
        elif p == 1:
            cells_list = [4, 6, 8]
        else:
            cells_list = [3, 4, 6]

        errors_abs = []
        errors_rel = []
        masses = []
        meta_list = []
        for n in cells_list:
            x, y, z, coeff, meta = solve_advection_3d(
                ncells=n, degree=p, final_time=1.0,
                velocity=VELOCITY, cfl=0.06,
            )
            metrics = error_3d(x, y, z, coeff, 1.0, velocity=VELOCITY)
            m = mass_3d(coeff)
            errors_abs.append(metrics["absolute_l2"])
            errors_rel.append(metrics["relative_l2"])
            masses.append(m)
            meta_list.append({
                "ncells": (n, n, n), "dt": meta["dt"],
                "steps": meta["steps"], "mass_drift": abs(m - 1.0),
            })
            print(f"  3D P{p} N={n}: err_rel={metrics['relative_l2']:.6e}, "
                  f"mass_drift={abs(m - 1.0):.2e}")

        # Use relative L2 error for convergence analysis
        tbl = convergence_table(cells_list, errors_rel, theoretical_order=p + 1)
        results[f"p{p}"] = {
            "degree": p,
            "cells": cells_list,
            "errors_abs_l2": errors_abs,
            "errors_rel_l2": errors_rel,
            "masses": masses,
            "observed_order_lsq": tbl["observed_order_lsq"],
            "pairwise_orders": tbl["pairwise_orders"],
            "theoretical_order": p + 1,
            "order_deviation": tbl["order_deviation"],
            "order_within_tolerance": abs(tbl["order_deviation"]) <= ORDER_TOLERANCE,
            "runs": meta_list,
        }
    return results


def main():
    torch.set_num_threads(4)
    print("=" * 60)
    print("3-D Linear Advection p-Order Convergence Study")
    print("=" * 60)

    results = study_3d()

    # Build verdict
    all_pass = all(
        r["order_within_tolerance"] for r in results.values()
    )
    verdict = {
        "overall": "PASS" if all_pass else "PARTIAL",
        "all_within_tolerance": all_pass,
        "details": {
            k: {
                "degree": v["degree"],
                "observed_order": v["observed_order_lsq"],
                "theoretical_order": v["theoretical_order"],
                "within_tolerance": v["order_within_tolerance"],
            }
            for k, v in results.items()
        },
    }

    artifact = {
        "benchmark": "linear_advection_3d_porder",
        "anchor": "CASE-DG-ADVECT-ORDER",
        "repo": "TensorDG",
        "dimension": 3,
        "problem": (
            "u_t + a·∇u = 0 on [0,1]³ periodic, "
            f"a={VELOCITY}, "
            "u(x,y,z,0) = 1 + sin(2πx)sin(2πy)sin(2πz)"
        ),
        "method": "DG modal Legendre + upwind flux + RK4",
        "criterion": f"observed_order within ±{ORDER_TOLERANCE} of theoretical p+1",
        "study_3d": {
            "degrees": list(results.keys()),
            "results": results,
        },
        "verdict": verdict,
    }

    # Save artifact
    out_dir = ROOT / "docs" / "benchmarks" / "results" / "linear-advection-3d-porder"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "porder_convergence_3d.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nArtifact saved to {artifact_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Degree':>6} | {'Observed':>10} | {'Theoretical':>11} | {'Deviation':>10} | {'Pass':>4}")
    print("-" * 60)
    for k, v in results.items():
        p = v["degree"]
        obs = v["observed_order_lsq"]
        theo = v["theoretical_order"]
        dev = v["order_deviation"]
        ok = "✅" if v["order_within_tolerance"] else "❌"
        print(f"  P{p:>3} | {obs:>10.4f} | {theo:>11} | {dev:>+10.4f} | {ok:>4}")
    print(f"\nOverall: {verdict['overall']}")

    return artifact


if __name__ == "__main__":
    main()
