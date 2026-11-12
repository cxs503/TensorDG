"""P-type exponential convergence benchmark for smooth solutions.

Case anchor: CASE-DG-PREFINEMENT-EXPONENTIAL
Reference: Babuska & Suri (1994) "The p- and h-p versions of the finite element
method, basic principles and properties", SIAM Review 36(4), 578-632.
DOI: 10.21236/ada260237

Theorem 3.1 / Section 3.2: for analytic solutions the DG/FEM error decays
exponentially in the polynomial degree p:
    ||u - u_h||_L2 <= C * exp(-sigma * p)
where sigma depends on the analyticity strip of the solution.

Problem: 1D linear advection u_t + u_x = 0 on [0,1] with periodic BC.
Initial condition: sin(2*pi*x) — entire function, analytic everywhere.
Final time: T = 1.0 (one full period, exact solution = initial condition).
Fixed mesh: ncells = 16 (isolates p-refinement from h-refinement).
Polynomial degrees: p = 1, 2, 3, 4, 5, 6.

Tolerance criterion: semi-log error-vs-p slope < -0.5 per unit p
(i.e. decay rate sigma > 0.5).

Run::
    python -m benchmarks.p_refinement_exponential
    python -m benchmarks.p_refinement_exponential --quick   # p=1..3 smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tensordg.advection_1d import l2_error, solve_advection
from tensordg.convergence import p_decay_diagnostics

SCHEMA = "tensordg-p-refinement-exponential/1"
ARTIFACT_RELPATH = Path("docs") / "benchmarks" / "results" / "p-refinement-exponential"

# --- Configuration ---------------------------------------------------------
DEGREES_FULL = (1, 2, 3, 4, 5, 6)
DEGREES_QUICK = (1, 2, 3)
NCELLS = 16
FINAL_TIME = 1.0
SPEED = 1.0
CFL_BASE = 0.08  # will be refined per degree to suppress temporal error

# --- Thresholds (predeclared) ----------------------------------------------
DECAY_RATE_MIN = 0.5       # sigma > 0.5 (slope of log(e) vs p < -0.5)
MONOTONE_REQUIRED = True   # error must decrease at every p level
TEMPORAL_TOLERANCE = 0.01  # temporal error < 1% of spatial error


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:  # pragma: no cover - git not always available
        return None


def temporal_converged_advection(
    ncells: int,
    degree: int,
    final_time: float = FINAL_TIME,
    speed: float = SPEED,
    cfl: float = CFL_BASE,
    tolerance: float = TEMPORAL_TOLERANCE,
    max_refinements: int = 6,
) -> dict:
    """L2 error with RK4 step size refined until temporal error is negligible."""
    history = []
    current_cfl = cfl
    error = None
    converged = False
    for _ in range(max_refinements + 1):
        centers, coefficients, metadata = solve_advection(
            ncells=ncells, degree=degree, final_time=final_time,
            speed=speed, cfl=current_cfl, time_integrator="rk4",
        )
        new_error = l2_error(centers, coefficients, final_time, speed=speed)
        change = (abs(new_error - error) / new_error) if error else float("inf")
        history.append({
            "cfl": current_cfl, "dt": metadata["dt"],
            "steps": metadata["steps"], "relative_l2": new_error,
            "change_vs_previous": change,
        })
        error = new_error
        if change <= tolerance:
            converged = True
            break
        current_cfl *= 0.5
    return {
        "relative_l2": error,
        "cfl": history[-1]["cfl"],
        "steps": history[-1]["steps"],
        "dt": history[-1]["dt"],
        "temporal_change": history[-1]["change_vs_previous"],
        "temporal_converged": converged,
        "refinements": len(history) - 1,
        "history": history,
    }


def p_refinement_study(
    degrees: tuple[int, ...] = DEGREES_FULL,
    ncells: int = NCELLS,
    final_time: float = FINAL_TIME,
) -> dict:
    """Run p-refinement study and collect errors + diagnostics."""
    errors = []
    levels = []
    for degree in degrees:
        run = temporal_converged_advection(ncells, degree, final_time)
        errors.append(run["relative_l2"])
        levels.append({
            "degree": degree,
            "ncells": ncells,
            "h": 1.0 / ncells,
            "dofs": ncells * (degree + 1),
            "relative_l2": run["relative_l2"],
            "cfl": run["cfl"],
            "dt": run["dt"],
            "steps": run["steps"],
            "temporal_change": run["temporal_change"],
            "temporal_converged": run["temporal_converged"],
            "refinements": run["refinements"],
        })

    diag = p_decay_diagnostics(degrees, errors)

    return {
        "equation": "u_t + u_x = 0",
        "domain": "[0, 1] periodic",
        "initial_condition": "sin(2*pi*x)",
        "final_time": final_time,
        "exact_solution": "sin(2*pi*(x - t)) = sin(2*pi*x) at t=1.0",
        "ncells": ncells,
        "degrees": list(degrees),
        "levels": levels,
        "errors": errors,
        "diagnostics": diag,
    }


def check_criterion(diag: dict) -> dict:
    """Predeclared pass/fail checks."""
    lsq_rate = diag["lsq_decay_rate"]
    monotone = diag["monotone_decreasing"]
    return {
        "decay_rate": {
            "measured": lsq_rate,
            "threshold": DECAY_RATE_MIN,
            "pass": lsq_rate > DECAY_RATE_MIN,
        },
        "monotone_decreasing": {
            "measured": monotone,
            "required": MONOTONE_REQUIRED,
            "pass": monotone or not MONOTONE_REQUIRED,
        },
        "overall_pass": lsq_rate > DECAY_RATE_MIN and (monotone or not MONOTONE_REQUIRED),
    }


def generate_figures(study: dict, outdir: Path):
    """Generate semi-log error-vs-p plot and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degrees = study["degrees"]
    errors = study["errors"]
    diag = study["diagnostics"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(degrees, errors, "bo-", linewidth=2, markersize=8, label="DG L2 error")

    # Exponential fit line
    p_fit = np.linspace(min(degrees), max(degrees), 100)
    e_fit = errors[0] * np.exp(-diag["lsq_decay_rate"] * (p_fit - degrees[0]))
    ax.semilogy(p_fit, e_fit, "r--", linewidth=1.5,
                label=f"exp fit: σ = {diag['lsq_decay_rate']:.3f}")

    # Reference slope line (sigma = 0.5)
    e_ref = errors[0] * np.exp(-DECAY_RATE_MIN * (p_fit - degrees[0]))
    ax.semilogy(p_fit, e_ref, "g:", linewidth=1.0, alpha=0.7,
                label=f"threshold: σ = {DECAY_RATE_MIN}")

    ax.set_xlabel("Polynomial degree $p$", fontsize=13)
    ax.set_ylabel("L2 error (log scale)", fontsize=13)
    ax.set_title(
        f"P-type exponential convergence\n"
        f"advection, ncells={study['ncells']}, T={study['final_time']}",
        fontsize=14,
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Annotate error values
    for p, e in zip(degrees, errors):
        ax.annotate(f"{e:.2e}", (p, e), textcoords="offset points",
                    xytext=(8, 5), fontsize=9)

    fig.tight_layout()
    png_path = outdir / "p_refinement_semilog.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # Also generate linear-log plot (log10 error vs p)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    log_errors = [math.log10(e) if e > 0 else -16 for e in errors]
    ax2.plot(degrees, log_errors, "bo-", linewidth=2, markersize=8, label="log10(L2 error)")

    # Linear fit
    p_arr = np.array(degrees, dtype=float)
    e_log = np.array(log_errors)
    slope, intercept = np.polyfit(p_arr, e_log, 1)
    ax2.plot(p_arr, slope * p_arr + intercept, "r--", linewidth=1.5,
             label=f"linear fit: slope = {slope:.3f}")

    ax2.axhline(y=-0.5 * p_arr[0] + intercept, color="g", linestyle=":", alpha=0.5,
                label="threshold slope = -0.5")

    ax2.set_xlabel("Polynomial degree $p$", fontsize=13)
    ax2.set_ylabel("log10(L2 error)", fontsize=13)
    ax2.set_title("Semi-log convergence: log10(error) vs p", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    png2_path = outdir / "p_refinement_log10.png"
    fig2.savefig(png2_path, dpi=150)
    plt.close(fig2)

    # Repo-relative paths: an absolute path bakes in the worktree that happened
    # to run the benchmark and cannot be resolved by anyone else (audit
    # m16499 found the historical artifact pointing at .../work/m16180/...).
    root = Path(__file__).resolve().parent.parent
    return {
        "p_refinement_semilog.png": png_path.relative_to(root).as_posix(),
        "p_refinement_log10.png": png2_path.relative_to(root).as_posix(),
    }


def build_report(study: dict, checks: dict, figures: dict, runtime: float) -> dict:
    """Build the full benchmark report."""
    return {
        "schema": SCHEMA,
        "case": "CASE-DG-PREFINEMENT-EXPONENTIAL",
        "reference": {
            "authors": "Babuska & Suri",
            "year": 1994,
            "journal": "SIAM Review",
            "volume": 36,
            "pages": "578-632",
            "doi": "10.21236/ada260237",
            "key_result": "Theorem 3.1: ||u-u_h||_L2 <= C*exp(-sigma*p) for analytic solutions",
        },
        "problem": {
            "equation": study["equation"],
            "domain": study["domain"],
            "initial_condition": study["initial_condition"],
            "final_time": study["final_time"],
            "exact_solution": study["exact_solution"],
            "ncells": study["ncells"],
        },
        "results": {
            "degrees": study["degrees"],
            "errors": study["errors"],
            "levels": study["levels"],
            "diagnostics": study["diagnostics"],
        },
        "checks": checks,
        "verdict": "PASS" if checks["overall_pass"] else "FAIL",
        "figures": figures,
        "runtime_seconds": runtime,
    }


def main():
    parser = argparse.ArgumentParser(description="P-type exponential convergence benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (p=1..3)")
    args = parser.parse_args()

    degrees = DEGREES_QUICK if args.quick else DEGREES_FULL
    outdir = Path(__file__).resolve().parent.parent / ARTIFACT_RELPATH
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    study = p_refinement_study(degrees=degrees)
    runtime = time.time() - t0

    checks = check_criterion(study["diagnostics"])
    figures = generate_figures(study, outdir)
    report = build_report(study, checks, figures, runtime)

    # Save JSON artifact
    json_path = outdir / "p_refinement_results.json"
    json_path.write_text(json.dumps(report, indent=2))

    # Provenance manifest: hash of the artifact above, plus the environment that
    # produced it, so a verifier can tell a fresh run from a stale file.
    provenance = {
        "case": "CASE-DG-PREFINEMENT-EXPONENTIAL",
        "artifact": json_path.name,
        "artifact_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
        "schema": SCHEMA,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "degrees": list(degrees),
        "ncells": study["ncells"],
        "cfl_base": CFL_BASE,
        "temporal_tolerance": TEMPORAL_TOLERANCE,
        "git_commit": _git_commit(),
        "runtime_seconds": runtime,
    }
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    # Save CSV table
    csv_path = outdir / "p_refinement_table.csv"
    with open(csv_path, "w") as f:
        f.write("p,ncells,dofs,L2_error,decay_rate,monotone\n")
        for i, p in enumerate(degrees):
            mono = "T" if i == 0 else ("T" if study["errors"][i-1] > study["errors"][i] else "F")
            f.write(f"{p},{study['ncells']},{study['levels'][i]['dofs']},"
                    f"{study['errors'][i]:.6e},{study['diagnostics']['lsq_decay_rate']:.4f},{mono}\n")

    # Print summary
    print(f"\n{'='*60}")
    print(f"P-type exponential convergence benchmark")
    print(f"{'='*60}")
    print(f"Equation: {study['equation']}")
    print(f"Domain: {study['domain']}")
    print(f"Initial condition: {study['initial_condition']}")
    print(f"Final time: T = {study['final_time']}")
    print(f"Mesh: ncells = {study['ncells']} (fixed)")
    print(f"Degrees: p = {degrees}")
    print(f"\n{'p':>3}  {'DOFs':>6}  {'L2 Error':>14}  {'Reduction':>12}")
    print(f"{'-'*40}")
    for i, p in enumerate(degrees):
        if i == 0:
            reduction = "-"
        else:
            reduction = f"{study['errors'][i-1]/study['errors'][i]:.1f}x"
        print(f"{p:>3}  {study['levels'][i]['dofs']:>6}  {study['errors'][i]:>14.6e}  {reduction:>12}")
    print(f"\nLeast-squares decay rate (sigma): {study['diagnostics']['lsq_decay_rate']:.4f}")
    print(f"Threshold (sigma > {DECAY_RATE_MIN}): {'PASS' if checks['decay_rate']['pass'] else 'FAIL'}")
    print(f"Monotone decreasing: {'PASS' if checks['monotone_decreasing']['pass'] else 'FAIL'}")
    print(f"Overall verdict: {report['verdict']}")
    print(f"Runtime: {runtime:.1f}s")
    print(f"\nArtifacts saved to: {outdir}")
    print(f"  - {json_path.name}")
    print(f"  - {csv_path.name}")
    for name, path in figures.items():
        print(f"  - {name}")

    return report


if __name__ == "__main__":
    main()
