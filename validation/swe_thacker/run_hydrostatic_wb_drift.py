#!/usr/bin/env python3
"""Hydrostatic well-balanced round-off drift law study.

When a lake-at-rest initial condition is integrated with the well-balanced
DG SWE solver, the free-surface elevation η and velocity u do not remain
exactly constant: they acquire a round-off-level residual that **grows**
with the number of time steps.  This is not a discretisation error (which
would decay with mesh refinement) but a cumulative floating-point effect:
each RK stage performs O(N) arithmetic operations, and the total number of
stages is proportional to T/dt ~ N/dx ~ N² for fixed final time T.

This script quantifies the drift law:

    ε_η(N) ≈ C_η · N · ε_machine
    ε_u(N) ≈ C_u · N · ε_machine

where N is the number of cells, ε_machine ≈ 2.2e-16 (float64), and C_η, C_u
are O(1–10) constants that depend on the polynomial degree, bed geometry,
and time integrator.

The drift law has two practical consequences:

1. **Tolerance selection**: acceptance thresholds for well-balanced tests
   must scale with N, not with a fixed absolute value.  A threshold of
   1e-8 m is safe for N ≤ 100 but will be violated for N ≥ 1000.

2. **Long-time integration**: the drift accumulates linearly with the number
   of time steps, so a simulation run for 10× longer will have ≈10× larger
   round-off residuals, even if the physical solution is exactly stationary.

This study runs a mesh refinement sequence on the Thacker parabolic basin
and fits the drift law to the measured η and u errors.

References
----------
* Higham (1996), "Accuracy and Stability of Numerical Algorithms", §2.7.
* Xing & Shu (2005), "Efficient and high order accurate... well-balanced schemes"
* TensorDG docs/swe-hydrostatic-roundoff-drift-law.md
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
    """Collect reproducibility metadata for a validation run."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]

    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
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
        "generator": "validation/swe_thacker/run_hydrostatic_wb_drift.py",
    }


def thacker_bed(x: torch.Tensor, H0: float = 2.0, L: float = 1.0) -> torch.Tensor:
    """Thacker parabolic bed elevation: z_b(x) = -H0 + (H0/L²)·x²."""
    return -H0 + (H0 / L**2) * x**2


def run_drift_study(
    ncells_list: list[int] = None,
    degrees: list[int] = None,
    final_time: float = 1.0,
    H0: float = 2.0,
    L: float = 1.0,
    eta0: float = 1.0,
    cfl: float = 0.3,
    device: str = "cpu",
) -> dict:
    """Run mesh refinement study to quantify the round-off drift law."""
    if ncells_list is None:
        ncells_list = [25, 50, 100, 200]
    if degrees is None:
        degrees = [1, 2]

    print("=" * 70)
    print("HYDROSTATIC WELL-BALANCED ROUND-OFF DRIFT LAW STUDY")
    print("=" * 70)
    print(f"Thacker basin: H0={H0} m, L={L} m, eta0={eta0} m")
    print(f"Final time: {final_time} s, CFL={cfl}")
    print(f"Resolutions: {ncells_list}")
    print(f"Degrees: {degrees}")
    print()

    all_runs = []
    drift_fits = {}

    for degree in degrees:
        print(f"\n--- Polynomial degree P{degree} ---")
        degree_runs = []

        for ncells in ncells_list:
            print(f"  N={ncells}...", end=" ", flush=True)

            def zb(xx):
                return thacker_bed(xx, H0=H0, L=L)

            start_time = time.time()
            x, coeff, zb_faces, meta = solve_swe_1d(
                lambda xx: eta0 - zb(xx),
                lambda xx: torch.zeros_like(xx),
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

            # Reconstruct with zb_center=zb_modes[:,0] to remove post-processing error
            h, u, eta = reconstruct_fields(coeff, zb_faces,
                                           zb_center=meta["zb_modes"][:, 0])

            eta_error_max = float((eta - eta0).abs().max())
            eta_error_l2 = float(torch.sqrt(torch.mean((eta - eta0) ** 2)))
            u_max = float(u.abs().max())
            u_l2 = float(torch.sqrt(torch.mean(u**2)))
            steps = meta["steps"]

            run_data = {
                "ncells": ncells,
                "degree": degree,
                "final_time": final_time,
                "steps": steps,
                "wall_time_s": wall_time,
                "eta_error_max": eta_error_max,
                "eta_error_l2": eta_error_l2,
                "u_max": u_max,
                "u_l2": u_l2,
                "well_balanced": True,
                "bed_representation": meta["bed_representation"],
                "reached_final_time": meta["reached_final_time"],
            }
            degree_runs.append(run_data)
            all_runs.append(run_data)

            print(f"η_max={eta_error_max:.3e}  u_max={u_max:.3e}  steps={steps}")

        # Fit drift law: error = C * N * eps_machine
        eps_machine = np.finfo(np.float64).eps
        ncells_arr = np.array([r["ncells"] for r in degree_runs])
        eta_max_arr = np.array([r["eta_error_max"] for r in degree_runs])
        u_max_arr = np.array([r["u_max"] for r in degree_runs])

        # Fit log(error) = log(C) + log(N) + log(eps)
        log_N = np.log(ncells_arr)
        log_eta_norm = np.log(eta_max_arr / eps_machine)
        log_u_norm = np.log(u_max_arr / eps_machine)

        # Linear fit: y = a + b*x, expect b ≈ 1 for drift law
        eta_fit = np.polyfit(log_N, log_eta_norm, 1)
        u_fit = np.polyfit(log_N, log_u_norm, 1)

        C_eta = np.exp(eta_fit[1])
        C_u = np.exp(u_fit[1])
        slope_eta = eta_fit[0]
        slope_u = u_fit[0]

        # Also compute error per step
        steps_arr = np.array([r["steps"] for r in degree_runs])
        eta_per_step = eta_max_arr / steps_arr
        u_per_step = u_max_arr / steps_arr

        drift_fits[f"P{degree}"] = {
            "C_eta": float(C_eta),
            "C_u": float(C_u),
            "slope_eta": float(slope_eta),
            "slope_u": float(slope_u),
            "eta_per_step_mean": float(np.mean(eta_per_step)),
            "eta_per_step_std": float(np.std(eta_per_step)),
            "u_per_step_mean": float(np.mean(u_per_step)),
            "u_per_step_std": float(np.std(u_per_step)),
            "eps_machine": float(eps_machine),
        }

        print(f"\n  Drift law fit (P{degree}):")
        print(f"    ε_η ≈ {C_eta:.1f} · N · ε_machine  (slope={slope_eta:.2f})")
        print(f"    ε_u ≈ {C_u:.1f} · N · ε_machine  (slope={slope_u:.2f})")
        print(f"    η error per step: {np.mean(eta_per_step):.3e} ± {np.std(eta_per_step):.3e}")
        print(f"    u error per step: {np.mean(u_per_step):.3e} ± {np.std(u_per_step):.3e}")

    return {
        "runs": all_runs,
        "drift_fits": drift_fits,
        "ncells_list": ncells_list,
        "degrees": degrees,
        "final_time": final_time,
        "H0": H0,
        "L": L,
        "eta0": eta0,
        "cfl": cfl,
    }


def generate_drift_table(drift_results: dict) -> str:
    """Generate a markdown table summarizing the drift law study."""
    lines = [
        "# Hydrostatic Well-Balanced Round-Off Drift Law Results",
        "",
        "Thacker parabolic basin, lake-at-rest initial condition, HLL flux, CFL=0.3.",
        "",
        "## Per-Resolution Errors",
        "",
        "| Degree | N   | Steps | η error (max) | η error (L2) | u max | u L2 | Wall time (s) |",
        "|--------|-----|-------|---------------|--------------|-------|------|---------------|",
    ]

    for run in drift_results["runs"]:
        lines.append(
            f"| P{run['degree']} | {run['ncells']:4d} | {run['steps']:6d} "
            f"| {run['eta_error_max']:.3e} | {run['eta_error_l2']:.3e} "
            f"| {run['u_max']:.3e} | {run['u_l2']:.3e} | {run['wall_time_s']:.1f} |"
        )

    lines.append("")
    lines.append("## Drift Law Parameters")
    lines.append("")
    lines.append("| Degree | C_η | C_u | Slope η | Slope u | η/step (mean) | u/step (mean) |")
    lines.append("|--------|-----|-----|---------|---------|---------------|---------------|")

    for deg, fit in drift_results["drift_fits"].items():
        lines.append(
            f"| {deg} | {fit['C_eta']:.1f} | {fit['C_u']:.1f} "
            f"| {fit['slope_eta']:.2f} | {fit['slope_u']:.2f} "
            f"| {fit['eta_per_step_mean']:.3e} | {fit['u_per_step_mean']:.3e} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The drift law states that the round-off error grows linearly with the")
    lines.append("number of cells N (and thus with the number of time steps):")
    lines.append("")
    lines.append("    ε_η(N) ≈ C_η · N · ε_machine")
    lines.append("    ε_u(N) ≈ C_u · N · ε_machine")
    lines.append("")
    lines.append("where ε_machine ≈ 2.2×10⁻¹⁶ (float64).  The fitted slopes should be")
    lines.append("close to 1.0, confirming the linear drift law.  The constants C_η and")
    lines.append("C_u are O(1–10) and depend on the polynomial degree and bed geometry.")
    lines.append("")

    return "\n".join(lines)


def generate_drift_figure(drift_results: dict, output_path: Path):
    """Generate a figure showing the drift law: error vs N on log-log scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {1: "#1f77b4", 2: "#ff7f0e"}
    markers = {1: "o", 2: "s"}

    for degree in drift_results["degrees"]:
        runs = [r for r in drift_results["runs"] if r["degree"] == degree]
        ncells = [r["ncells"] for r in runs]
        eta_max = [r["eta_error_max"] for r in runs]
        u_max = [r["u_max"] for r in runs]

        fit = drift_results["drift_fits"][f"P{degree}"]
        eps = fit["eps_machine"]
        C_eta = fit["C_eta"]
        C_u = fit["C_u"]

        # Theoretical drift law lines
        n_theory = np.array(ncells)
        eta_theory = C_eta * n_theory * eps
        u_theory = C_u * n_theory * eps

        ax = axes[0]
        ax.loglog(ncells, eta_max, markers[degree], color=colors[degree],
                  label=f"P{degree} (measured)", markersize=8)
        ax.loglog(n_theory, eta_theory, "--", color=colors[degree],
                  label=f"P{degree} (C_η·N·ε)")

        ax = axes[1]
        ax.loglog(ncells, u_max, markers[degree], color=colors[degree],
                  label=f"P{degree} (measured)", markersize=8)
        ax.loglog(n_theory, u_theory, "--", color=colors[degree],
                  label=f"P{degree} (C_u·N·ε)")

    # Reference line for ε_machine
    eps = np.finfo(np.float64).eps
    n_ref = np.array(drift_results["ncells_list"])
    for ax in axes:
        ax.loglog(n_ref, eps * n_ref, "k:", label="N·ε_machine", linewidth=1)
        ax.loglog(n_ref, np.full_like(n_ref, eps), "k-.", label="ε_machine", linewidth=1)

    axes[0].set_xlabel("Number of cells N")
    axes[0].set_ylabel("η error (max) [m]")
    axes[0].set_title("Free-surface elevation drift")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].set_xlabel("Number of cells N")
    axes[1].set_ylabel("u error (max) [m/s]")
    axes[1].set_title("Velocity drift")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    fig.suptitle("Hydrostatic Well-Balanced Round-Off Drift Law\n"
                 "Thacker parabolic basin, lake-at-rest, HLL flux, CFL=0.3",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def main():
    """Main entry point for drift law study."""
    # Run drift study with reduced resolutions for CI efficiency
    drift_results = run_drift_study(
        ncells_list=[25, 50, 100, 200],
        degrees=[1, 2],
        final_time=1.0,
        H0=2.0,
        L=1.0,
        eta0=1.0,
        cfl=0.3,
    )

    # Add provenance
    drift_results["provenance"] = collect_provenance()

    # Save JSON results
    output_dir = Path(__file__).parent
    json_path = output_dir / "hydrostatic_wb_drift.json"
    with open(json_path, "w") as f:
        json.dump(drift_results, f, indent=2)
    print(f"\n✓ Results saved to {json_path}")

    # Generate markdown table
    table_md = generate_drift_table(drift_results)
    table_path = output_dir / "hydrostatic_wb_drift_table.md"
    with open(table_path, "w") as f:
        f.write(table_md)
    print(f"✓ Table saved to {table_path}")

    # Generate figure
    fig_dir = Path(__file__).resolve().parents[2] / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "swe_wb_drift_law.png"
    generate_drift_figure(drift_results, fig_path)

    # Print summary
    print("\n" + "=" * 70)
    print("DRIFT LAW STUDY SUMMARY")
    print("=" * 70)
    for deg, fit in drift_results["drift_fits"].items():
        print(f"\n{deg}:")
        print(f"  ε_η ≈ {fit['C_eta']:.1f} · N · ε_machine  (slope={fit['slope_eta']:.2f})")
        print(f"  ε_u ≈ {fit['C_u']:.1f} · N · ε_machine  (slope={fit['slope_u']:.2f})")
        slope_ok_eta = abs(fit["slope_eta"] - 1.0) < 0.3
        slope_ok_u = abs(fit["slope_u"] - 1.0) < 0.3
        print(f"  Slope η ≈ 1.0: {'✓' if slope_ok_eta else '✗'}")
        print(f"  Slope u ≈ 1.0: {'✓' if slope_ok_u else '✗'}")
    print("=" * 70)

    return drift_results


if __name__ == "__main__":
    main()
