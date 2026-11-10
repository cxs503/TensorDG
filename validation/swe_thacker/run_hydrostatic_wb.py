#!/usr/bin/env python3
"""Hydrostatic well-balanced validation for the Thacker parabolic basin.

This script validates the hydrostatic well-balanced property of the DG SWE solver
on the Thacker (1981) parabolic basin geometry. The test verifies that a lake at
rest in a parabolic basin remains at rest without generating spurious currents.

Thacker (1981) parabolic basin:
    z_b(x) = -H_0 + (H_0 - H_min) * (x/L)^2

where:
    H_0 = depth at center (x=0)
    H_min = depth at edges (x=±L)
    L = half-width of the basin

For hydrostatic equilibrium:
    η = h + z_b = const
    u = 0

This is a stringent test of the well-balanced property because:
1. The bed is curved (quadratic)
2. The water depth varies spatially
3. Any imbalance in the source term will generate spurious oscillations

References
----------
* Thacker (1981), "Some exact solutions of the nonlinear shallow-water equations"
* Xing & Shu (2005), "Efficient and high order accurate... well-balanced schemes"
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from tensordg.swe_1d import reconstruct_fields, solve_swe_1d


def collect_provenance(repo_root: Path | None = None) -> dict:
    """Collect reproducibility metadata for a validation run.

    Records the exact code revision, interpreter/library versions, host
    architecture and command line that produced the JSON artifact, so a
    stale or hand-edited result file is detectable (compare the numbers with
    a re-run, or diff the recorded revision against the tree).

    Parameters
    ----------
    repo_root : pathlib.Path, optional
        Repository root used for the ``git rev-parse`` call.  Defaults to the
        parent of this script's directory.

    Returns
    -------
    dict
        Provenance record (JSON-serialisable).
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:  # pragma: no cover - git absent / not a repo
            return "unknown"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_commit_short": _git("rev-parse", "--short", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "generator": "validation/swe_thacker/run_hydrostatic_wb.py",
    }


def thacker_bed(x: torch.Tensor, H0: float = 2.0, L: float = 1.0) -> torch.Tensor:
    """Thacker parabolic bed elevation.

    z_b(x) = -H0 + (H0/L^2) * x^2

    Parameters
    ----------
    x : torch.Tensor
        Spatial coordinates
    H0 : float
        Depth at the center (x=0)
    L : float
        Half-width of the basin

    Returns
    -------
    torch.Tensor
        Bed elevation z_b(x)
    """
    return -H0 + (H0 / L**2) * x**2


def run_hydrostatic_validation(
    ncells: int = 100,
    degree: int = 1,
    final_time: float = 1.0,
    H0: float = 2.0,
    L: float = 1.0,
    eta0: float = 1.0,  # Still water level above z_b=0
    cfl: float = 0.3,
    device: str = "cpu",
) -> dict:
    """Run hydrostatic well-balanced validation.

    Parameters
    ----------
    ncells : int
        Number of cells
    degree : int
        Polynomial degree (1 or 2)
    final_time : float
        Final simulation time
    H0 : float
        Depth at basin center (z_b(0) = -H0)
    L : float
        Half-width of basin
    eta0 : float
        Still water level (η = h + z_b = eta0). Must be > 0 to avoid drying.
    cfl : float
        CFL number
    device : str
        Device ('cpu' or 'cuda')

    Returns
    -------
    dict
        Validation results including errors and metadata
    """
    print(f"Running hydrostatic validation: N={ncells}, P{degree}, t={final_time}s")

    # Define bed and initial conditions
    def zb(xx):
        return thacker_bed(xx, H0=H0, L=L)

    def initial_h(xx):
        """Initial depth: h = eta0 - z_b(x)"""
        return eta0 - zb(xx)

    def initial_u(xx):
        """Initial velocity: u = 0 (lake at rest)"""
        return torch.zeros_like(xx)

    # Solve SWE with well_balanced=True
    start_time = time.time()
    x, coeff, zb_faces, meta = solve_swe_1d(
        initial_h,
        initial_u,
        zb,
        ncells=ncells,
        degree=degree,
        final_time=final_time,
        domain=(-L, L),
        cfl=cfl,
        flux_kind="hll",
        well_balanced=True,
        device=device,
    )
    wall_time = time.time() - start_time

    # Reconstruct fields.  Use the mode-0 coefficient of the polynomial bed
    # as the cell-centre bed elevation: the default trapezoidal face mean is
    # only the cell average for a piecewise-linear bed, and for the P2
    # (quadratic) representation of the parabolic bed it would leave an
    # O(dx^2) spurious free-surface drift in the reconstructed η (see the
    # reconstruct_fields docstring).  The solver state itself is well-balanced
    # to round-off; this removes the post-processing error.
    h, u, eta = reconstruct_fields(coeff, zb_faces,
                                   zb_center=meta["zb_modes"][:, 0])

    # Compute errors
    eta_error = float((eta - eta0).abs().max())
    eta_error_l2 = float(torch.sqrt(torch.mean((eta - eta0) ** 2)))
    u_max = float(u.abs().max())
    u_l2 = float(torch.sqrt(torch.mean(u**2)))

    # Compute depth error
    h_exact = initial_h(x)
    h_error = float((h - h_exact).abs().max())
    h_error_l2 = float(torch.sqrt(torch.mean((h - h_exact) ** 2)))

    results = {
        "ncells": ncells,
        "degree": degree,
        "final_time": final_time,
        "H0": H0,
        "L": L,
        "eta0": eta0,
        "cfl": cfl,
        "wall_time_s": wall_time,
        "steps": meta["steps"],
        "dt_avg": wall_time / max(meta["steps"], 1),
        "eta_error_max": eta_error,
        "eta_error_l2": eta_error_l2,
        "u_max": u_max,
        "u_l2": u_l2,
        "h_error_max": h_error,
        "h_error_l2": h_error_l2,
        "well_balanced": meta["well_balanced"],
        "bed_representation": meta["bed_representation"],
        "reconstruction": "zb_center=zb_modes[:,0] (cell-average bed)",
        "reached_final_time": meta["reached_final_time"],
        "device": device,
    }

    print(f"  η error (max): {eta_error:.3e} m")
    print(f"  η error (L2):  {eta_error_l2:.3e} m")
    print(f"  u max:         {u_max:.3e} m/s")
    print(f"  h error (max): {h_error:.3e} m")
    print(f"  Steps:         {meta['steps']}")
    print(f"  Wall time:     {wall_time:.2f} s")

    return results


def run_convergence_study(
    degrees: list[int] = [1, 2],
    ncells_list: list[int] = [50, 100, 200],
    final_time: float = 1.0,
    H0: float = 2.0,
    L: float = 1.0,
    eta0: float = 1.0,
) -> dict:
    """Run mesh convergence study for hydrostatic well-balanced property.

    Parameters
    ----------
    degrees : list[int]
        Polynomial degrees to test
    ncells_list : list[int]
        Number of cells for each resolution
    final_time : float
        Final simulation time
    H0 : float
        Depth at basin center
    L : float
        Half-width of basin
    eta0 : float
        Still water level

    Returns
    -------
    dict
        Convergence study results
    """
    print("\n" + "=" * 70)
    print("HYDROSTATIC WELL-BALANCED CONVERGENCE STUDY")
    print("=" * 70)

    convergence_results = {
        "degrees": degrees,
        "ncells_list": ncells_list,
        "final_time": final_time,
        "H0": H0,
        "L": L,
        "eta0": eta0,
        "runs": [],
    }

    for degree in degrees:
        print(f"\n--- Polynomial degree P{degree} ---")
        degree_results = []

        for ncells in ncells_list:
            results = run_hydrostatic_validation(
                ncells=ncells,
                degree=degree,
                final_time=final_time,
                H0=H0,
                L=L,
                eta0=eta0,
            )
            degree_results.append(results)

        convergence_results["runs"].append(
            {"degree": degree, "results": degree_results}
        )

        # Compute convergence rates
        if len(degree_results) >= 2:
            print(f"\n  Convergence rates (P{degree}):")
            for i in range(1, len(degree_results)):
                n1 = degree_results[i - 1]["ncells"]
                n2 = degree_results[i]["ncells"]
                e1 = degree_results[i - 1]["eta_error_max"]
                e2 = degree_results[i]["eta_error_max"]

                if e2 > 0 and e1 > 0:
                    rate = math.log(e1 / e2) / math.log(n2 / n1)
                    print(f"    N={n1}→{n2}: rate = {rate:.2f}")

    return convergence_results


def main():
    """Main entry point for hydrostatic well-balanced validation."""
    print("=" * 70)
    print("TensorDG SWE Hydrostatic Well-Balanced Validation")
    print("Thacker (1981) Parabolic Basin")
    print("=" * 70)

    # Single validation run
    # Use eta0 = 1.0 to ensure h > 0 everywhere (no wetting-drying)
    # At x=0: z_b = -2.0, h = 1.0 - (-2.0) = 3.0
    # At x=±L: z_b = 0, h = 1.0 - 0 = 1.0
    print("\n--- Single Validation Run ---")
    single_results = run_hydrostatic_validation(
        ncells=100,
        degree=1,
        final_time=1.0,
        H0=2.0,
        L=1.0,
        eta0=1.0,  # Water surface at z=1.0, above the highest bed point
    )

    # Convergence study
    convergence_results = run_convergence_study(
        degrees=[1, 2],
        ncells_list=[50, 100, 200],
        final_time=1.0,
        H0=2.0,
        L=1.0,
        eta0=1.0,
    )

    # Save results
    output_dir = Path(__file__).parent
    output_file = output_dir / "hydrostatic_wb_results.json"

    all_results = {
        "provenance": collect_provenance(),
        "single_run": single_results,
        "convergence_study": convergence_results,
        "summary": {
            "test_description": "Hydrostatic well-balanced validation on Thacker parabolic basin",
            "expected_behavior": "η = const, u = 0 preserved to machine precision",
            "pass_criteria": {
                "eta_error_max": "< 1e-8 m",
                "u_max": "< 1e-8 m/s",
            },
            "actual_results": {
                "eta_error_max": single_results["eta_error_max"],
                "u_max": single_results["u_max"],
            },
            "pass": (
                single_results["eta_error_max"] < 1e-8
                and single_results["u_max"] < 1e-8
            ),
        },
    }

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"η error (max): {single_results['eta_error_max']:.3e} m")
    print(f"u max:         {single_results['u_max']:.3e} m/s")
    print(f"Pass criteria: η < 1e-8 m, u < 1e-8 m/s")
    print(f"Result:        {'PASS ✓' if all_results['summary']['pass'] else 'FAIL ✗'}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
