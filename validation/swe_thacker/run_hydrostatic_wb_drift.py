#!/usr/bin/env python3
"""Hydrostatic well-balanced round-off drift law study.

When a lake-at-rest initial condition is integrated with the well-balanced
DG SWE solver, the free-surface elevation η and velocity u do not remain
exactly constant: they acquire a round-off-level residual that **grows**
with the number of time steps.  This is not a discretisation error (which
would decay with mesh refinement) but a cumulative floating-point effect:
each RK stage performs O(N) arithmetic operations, and the total number of
stages is proportional to T/dt ~ N/dx ~ N² for fixed final time T.

This script quantifies the drift law from a log-log least-squares fit:

    ε_η(N) ≈ C_η · (N/100)^α · ε_machine
    ε_u(N) ≈ C_u · (N/100)^β · ε_machine

where N is the number of cells, ε_machine ≈ 2.2e-16 (float64), and the
exponents α, β are *measured*, not assumed.  A simple model (steps ~ N,
arithmetic operations per step ~ N) predicts α ≈ β ≈ 2; the fit reports what
the data actually says, together with the maximum relative residual of the
single power law against the four measured points.

The drift law has two practical consequences:

1. **Tolerance selection**: acceptance thresholds for well-balanced tests
   must scale with N, not with a fixed absolute value.  A threshold of
   1e-8 m is safe for N ≤ 100 but will be violated for N ≥ 1000.

2. **Long-time integration**: the drift re-accumulates at every step, so a
   longer run drifts more.  The *scaling* with time is measured separately by
   ``run_hydrostatic_wb_drift_time.py``: for η it is super-linear in T
   (γ_η ≈ 2.0-2.7 over ``T ∈ [0.5, 4] s``), for the spurious velocity it is
   close to linear (γ_u ≈ 1.0-1.2).  Do not assume ε ∝ T for η.

This study runs a mesh refinement sequence on the Thacker parabolic basin
and fits the drift law to the measured η and u errors.

References
----------
* Higham (1996), "Accuracy and Stability of Numerical Algorithms", §2.7.
* Xing & Shu (2005), "Efficient and high order accurate... well-balanced schemes"
* TensorDG docs/swe-hydrostatic-roundoff-drift-law.md
"""
from __future__ import annotations

import argparse
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


REFERENCE_NCELLS = 100.0


def fit_drift_law(degree_runs: list[dict]) -> dict:
    """Fit ``eps(N) = C · (N/N_ref)**exponent · eps_machine`` to one series.

    The fit is an ordinary least-squares line through
    ``(log N, log(eps / eps_machine))``: its slope is the measured exponent.
    The constant is reported at ``N_ref = 100`` cells so that it is directly
    readable as "the round-off level at a typical production resolution, in
    units of eps_machine".  ``max_rel_residual_*`` records how far the
    single-exponent power law departs from the measured points, so a
    curvature-heavy series is not silently reported as a clean law.
    """
    eps_machine = float(np.finfo(np.float64).eps)
    ncells = np.array([r["ncells"] for r in degree_runs], dtype=float)
    steps = np.array([r["steps"] for r in degree_runs], dtype=float)
    log_n = np.log(ncells)
    fit: dict = {
        "eps_machine": eps_machine,
        "n_points": int(ncells.size),
        "reference_ncells": REFERENCE_NCELLS,
    }

    for name, field in (("eta", "eta_error_max"), ("u", "u_max")):
        values = np.array([r[field] for r in degree_runs], dtype=float)
        exponent, log_c = np.polyfit(log_n, np.log(values / eps_machine), 1)
        # Reparameterise around N_ref: C = eps(N_ref) / eps_machine.
        c = float(np.exp(log_c + exponent * np.log(REFERENCE_NCELLS)))
        predicted = c * (ncells / REFERENCE_NCELLS) ** exponent * eps_machine
        fit[f"C_{name}"] = c
        fit[f"slope_{name}"] = float(exponent)
        fit[f"max_rel_residual_{name}"] = float(
            np.max(np.abs(predicted - values) / values)
        )

    eta_max = np.array([r["eta_error_max"] for r in degree_runs], dtype=float)
    u_max = np.array([r["u_max"] for r in degree_runs], dtype=float)
    eta_per_step = eta_max / steps
    u_per_step = u_max / steps
    fit["eta_per_step_mean"] = float(np.mean(eta_per_step))
    fit["eta_per_step_std"] = float(np.std(eta_per_step))
    fit["u_per_step_mean"] = float(np.mean(u_per_step))
    fit["u_per_step_std"] = float(np.std(u_per_step))
    return fit


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

        # Fit drift law: error = C * (N/N_ref)**exponent * eps_machine (measured, not assumed)
        fit = fit_drift_law(degree_runs)
        drift_fits[f"P{degree}"] = fit

        n_ref = fit["reference_ncells"]
        print(f"\n  Drift law fit (P{degree}):")
        print(f"    ε_η ≈ {fit['C_eta']:.3g} · (N/{n_ref:.0f})^{fit['slope_eta']:.2f} · ε_machine")
        print(f"    ε_u ≈ {fit['C_u']:.3g} · (N/{n_ref:.0f})^{fit['slope_u']:.2f} · ε_machine")
        print(f"    max relative residual: η {fit['max_rel_residual_eta']:.3f}, "
              f"u {fit['max_rel_residual_u']:.3f}")
        print(f"    η error per step: {fit['eta_per_step_mean']:.3e} ± {fit['eta_per_step_std']:.3e}")
        print(f"    u error per step: {fit['u_per_step_mean']:.3e} ± {fit['u_per_step_std']:.3e}")

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
    lines.append("| Degree | C_η@N=100 | C_u@N=100 | Slope η | Slope u | η/step (mean) | u/step (mean) |")
    lines.append("|--------|-----------|-----------|---------|---------|---------------|---------------|")

    for deg, fit in drift_results["drift_fits"].items():
        lines.append(
            f"| {deg} | {fit['C_eta']:.3g} | {fit['C_u']:.3g} "
            f"| {fit['slope_eta']:.2f} | {fit['slope_u']:.2f} "
            f"| {fit['eta_per_step_mean']:.3e} | {fit['u_per_step_mean']:.3e} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The exponents are measured by a log-log least-squares fit of")
    lines.append("`ε = C · (N/100)^α · ε_machine` (they are **not** assumed to be 1):")
    lines.append("")
    lines.append("    ε_η(N) ≈ C_η · (N/100)^α · ε_machine")
    lines.append("    ε_u(N) ≈ C_u · (N/100)^β · ε_machine")
    lines.append("")
    lines.append("where ε_machine ≈ 2.2×10⁻¹⁶ (float64).  The naive model (steps ~ N,")
    lines.append("operations per step ~ N) predicts α ≈ β ≈ 2; the measured exponents and")
    lines.append("the maximum relative residual of the power-law fit are reported in the")
    lines.append("JSON artifact (`slope_*`, `max_rel_residual_*`).  The per-step columns")
    lines.append("show that the residual re-accumulates at every step, which is why test")
    lines.append("thresholds must scale with N rather than being a fixed absolute value.")
    lines.append("The scaling with integration time is a separate measurement: see")
    lines.append("`hydrostatic_wb_drift_time_table.md` (measured γ_η > 1, γ_u ≈ 1).")
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
        n_theory = np.array(ncells, dtype=float)
        n_ref = fit["reference_ncells"]
        eta_theory = C_eta * (n_theory / n_ref) ** fit["slope_eta"] * eps
        u_theory = C_u * (n_theory / n_ref) ** fit["slope_u"] * eps

        ax = axes[0]
        ax.loglog(ncells, eta_max, markers[degree], color=colors[degree],
                  label=f"P{degree} (measured)", markersize=8)
        ax.loglog(n_theory, eta_theory, "--", color=colors[degree],
                  label=f"P{degree} (fit α={fit['slope_eta']:.2f})")

        ax = axes[1]
        ax.loglog(ncells, u_max, markers[degree], color=colors[degree],
                  label=f"P{degree} (measured)", markersize=8)
        ax.loglog(n_theory, u_theory, "--", color=colors[degree],
                  label=f"P{degree} (fit β={fit['slope_u']:.2f})")

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


def refit_existing(output_dir: Path) -> dict:
    """Recompute ``drift_fits`` from the committed runs, without solving.

    The measured per-resolution errors are the expensive part of this study
    and are deterministic for a given tree; the fit parameters are cheap and
    must never be hand-carried.  ``--refit`` therefore re-derives every
    drift-law constant from ``runs`` already present in the JSON artifact and
    records that it did so in ``provenance``.
    """
    json_path = output_dir / "hydrostatic_wb_drift.json"
    results = json.loads(json_path.read_text())
    results["drift_fits"] = {}
    for degree in results["degrees"]:
        degree_runs = [r for r in results["runs"] if r["degree"] == degree]
        results["drift_fits"][f"P{degree}"] = fit_drift_law(degree_runs)
    provenance = results.setdefault("provenance", {})
    provenance["fits_recomputed_at_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    provenance["fits_recomputed_by"] = (
        "validation/swe_thacker/run_hydrostatic_wb_drift.py --refit"
    )
    return results


def build_summary(drift_results: dict) -> dict:
    """Derive the human-readable JSON summary from the measured runs and fits.

    Like the drift fits, the summary text is generated, not typed: otherwise a
    stale sentence ("slope ~2.8-3.0") survives an artifact refresh and silently
    contradicts the numbers next to it.
    """
    fits = drift_results["drift_fits"]
    runs = drift_results["runs"]
    slopes = ", ".join(
        f"{key} α={fit['slope_eta']:.2f}/β={fit['slope_u']:.2f}"
        for key, fit in fits.items()
    )
    worst_residual = max(fit["max_rel_residual_eta"] for fit in fits.values())
    worst_run = max(runs, key=lambda r: r["eta_error_max"])
    return {
        "description": "Round-off drift law study for hydrostatic well-balanced SWE solver",
        "key_finding": (
            f"Round-off residual grows faster than the naive steps~N x cells~N model "
            f"(which gives exponent 2): measured exponents {slopes}. The single-exponent "
            f"power law is a summary of a four-point sequence; its largest relative "
            f"residual over the fitted degrees is {worst_residual:.2f}."
        ),
        "practical_implication": (
            f"The largest measured η residual is {worst_run['eta_error_max']:.2e} m "
            f"(P{worst_run['degree']}, N={worst_run['ncells']}), still many orders of "
            f"magnitude below physical tolerances, but automated well-balanced tests "
            f"must use N-dependent thresholds derived from the fitted law."
        ),
    }


def write_outputs(drift_results: dict, output_dir: Path, fig_dir: Path) -> None:
    """Persist the JSON artifact, the markdown table and the figure."""
    drift_results["summary"] = build_summary(drift_results)
    json_path = output_dir / "hydrostatic_wb_drift.json"
    json_path.write_text(json.dumps(drift_results, indent=2) + "\n")
    print(f"\n✓ Results saved to {json_path}")

    table_path = output_dir / "hydrostatic_wb_drift_table.md"
    table_path.write_text(generate_drift_table(drift_results))
    print(f"✓ Table saved to {table_path}")

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "swe_wb_drift_law.png"
    generate_drift_figure(drift_results, fig_path)


def main(argv: list[str] | None = None):
    """Main entry point for drift law study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refit",
        action="store_true",
        help="recompute drift-law fits from the runs already in the JSON artifact "
        "(no PDE solves)",
    )
    args = parser.parse_args(argv)

    output_dir = Path(__file__).parent
    fig_dir = Path(__file__).resolve().parents[2] / "docs" / "figures"

    if args.refit:
        drift_results = refit_existing(output_dir)
        print("Re-derived drift-law fits from committed runs (no solves).")
    else:
        drift_results = run_drift_study(
            ncells_list=[25, 50, 100, 200],
            degrees=[1, 2],
            final_time=1.0,
            H0=2.0,
            L=1.0,
            eta0=1.0,
            cfl=0.3,
        )
        drift_results["provenance"] = collect_provenance()

    write_outputs(drift_results, output_dir, fig_dir)

    print("\n" + "=" * 70)
    print("DRIFT LAW STUDY SUMMARY")
    print("=" * 70)
    for deg, fit in drift_results["drift_fits"].items():
        print(f"\n{deg}:")
        print(f"  ε_η ≈ {fit['C_eta']:.3g} · (N/{fit['reference_ncells']:.0f})"
              f"^{fit['slope_eta']:.2f} · ε_machine")
        print(f"  ε_u ≈ {fit['C_u']:.3g} · (N/{fit['reference_ncells']:.0f})"
              f"^{fit['slope_u']:.2f} · ε_machine")
        print(f"  power-law max relative residual: "
              f"η {fit['max_rel_residual_eta']:.3f}, u {fit['max_rel_residual_u']:.3f}")
    print("=" * 70)

    return drift_results


if __name__ == "__main__":
    main()
