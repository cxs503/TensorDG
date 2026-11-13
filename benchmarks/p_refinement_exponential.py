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
from tensordg.convergence import exponential_decay_criterion, p_decay_diagnostics

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
DECAY_RATE_MIN = 0.5       # legacy: sigma > 0.5 -- necessary, NOT sufficient
MONOTONE_REQUIRED = True   # error must decrease at every p level
TEMPORAL_TOLERANCE = 0.01  # temporal error < 1% of spatial error

# --- Discriminating criterion (predeclared) --------------------------------
# Audit m16499 (F4) showed that ``sigma > 0.5`` alone is accepted by algebraic
# error laws (p**-1 .. p**-3 with a 3-point fit) and even rejects a genuine
# exponential law sitting exactly on the threshold.  The published verdict is
# therefore gated on the composite criterion below; ``DECAY_RATE_MIN`` is kept
# only as a comparability threshold and is recorded as a sub-check.
SIGMA_MIN_DISCRIMINATING = 1.0     # spectral decay rate must exceed any algebraic rate
ERROR_RATIO_MIN = 1.0e3            # e(p_min)/e(p_max): >3 decades of reduction
DECAY_RATE_TOL = 1.0e-9            # relative slack on "pairwise rates non-decreasing"


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
    """Predeclared pass/fail checks (legacy thresholds + discriminating gate).

    ``decay_rate`` keeps the historical ``sigma > 0.5`` threshold so that older
    audits stay comparable, but the *verdict* is now driven by
    :func:`check_discriminating_criterion`: a bare ``sigma > 0.5`` accepts an
    algebraic error law (audit m16499 F4).
    """
    lsq_rate = diag["lsq_decay_rate"]
    monotone = diag["monotone_decreasing"]
    discriminating = check_discriminating_criterion(diag)
    legacy = {
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
    }
    return {
        **legacy,
        "discriminating": discriminating,
        "overall_pass": bool(
            legacy["decay_rate"]["pass"]
            and legacy["monotone_decreasing"]["pass"]
            and discriminating["pass"]
        ),
    }


def _known_decay_laws(degrees: tuple[int, ...]) -> dict:
    """Sequences whose convergence law is known a priori (falsification suite)."""
    p = np.asarray(degrees, dtype=float)
    p = p - p[0]
    scale = 1.0e-2
    return {
        # no convergence at all
        "constant": np.full_like(p, 1.0),
        # algebraic: what a low-order scheme produces and what the old gate let through
        "algebraic_p^-1": 1.0 / (p + 1.0),
        "algebraic_p^-2": 1.0 / (p + 1.0) ** 2,
        "algebraic_p^-3": 1.0 / (p + 1.0) ** 3,
        "algebraic_p^-4": 1.0 / (p + 1.0) ** 4,
        "algebraic_p^-5": 1.0 / (p + 1.0) ** 5,
        # genuine exponential laws: below / at / above the discriminating threshold
        "exponential_sigma_0.5": np.exp(-0.5 * p),
        "exponential_sigma_1.0": np.exp(-1.0 * p),
        "exponential_sigma_1.5": np.exp(-1.5 * p),
        "exponential_sigma_3.93": np.exp(-3.927 * p),
    }


# Which known laws the composite criterion must accept / reject.  Predeclared so
# that the gate cannot be re-tuned after seeing the data.
FALSIFICATION_EXPECTED = {
    "constant": False,
    "algebraic_p^-1": False,
    "algebraic_p^-2": False,
    "algebraic_p^-3": False,
    "algebraic_p^-4": False,
    "algebraic_p^-5": False,
    "exponential_sigma_0.5": False,   # only 12x reduction over p=1..6: unprovable
    "exponential_sigma_1.0": False,   # 150x < 1e3 decades gate
    "exponential_sigma_1.5": True,    # 1808x
    "exponential_sigma_3.93": True,
}


def check_discriminating_criterion(diag: dict) -> dict:
    """Composite predeclared criterion separating spectral from algebraic decay.

    Thin wrapper around :func:`tensordg.convergence.exponential_decay_criterion`
    (library tier) so that the shipped canonical case
    ``tensordg.cases.prefinement_exponential`` and this benchmark apply exactly
    one implementation of the gate.  The local constants
    (``SIGMA_MIN_DISCRIMINATING`` / ``ERROR_RATIO_MIN`` / ``DECAY_RATE_TOL``)
    are passed through, keeping the predeclared values auditable here.

    All conditions must hold (audit m16499 F4):
      1. ``sigma > SIGMA_MIN_DISCRIMINATING`` (an algebraic law in p cannot),
      2. pairwise decay rates non-decreasing (they decrease for an algebraic law),
      3. ``e(p_min) / e(p_max) > ERROR_RATIO_MIN`` (at least three decades).
    """
    return exponential_decay_criterion(
        diag,
        sigma_min=SIGMA_MIN_DISCRIMINATING,
        error_ratio_min=ERROR_RATIO_MIN,
        rate_tol=DECAY_RATE_TOL,
    )


def falsification_control(degrees: tuple[int, ...] = DEGREES_FULL) -> dict:
    """Run the predeclared criterion on known laws; gate = expected verdicts met."""
    laws = _known_decay_laws(tuple(degrees))
    rows = {}
    for name, seq in laws.items():
        diag = p_decay_diagnostics(tuple(degrees), [float(1.0e-2 * v) for v in seq])
        verdict = check_discriminating_criterion(diag)
        rows[name] = {
            "lsq_decay_rate": float(diag["lsq_decay_rate"]),
            "decay_rates": [float(v) for v in diag["decay_rates"]],
            "error_ratio": float(diag["errors"][0] / diag["errors"][-1]),
            "passes_legacy_sigma_gt_0.5": bool(diag["lsq_decay_rate"] > DECAY_RATE_MIN),
            "passes_discriminating": bool(verdict["pass"]),
            "expected": bool(FALSIFICATION_EXPECTED[name]),
            "as_expected": bool(verdict["pass"] == FALSIFICATION_EXPECTED[name]),
        }
    return {
        "degrees": list(degrees),
        "criterion": {
            "sigma_min": SIGMA_MIN_DISCRIMINATING,
            "error_ratio_min": ERROR_RATIO_MIN,
            "decay_rates_non_decreasing": True,
            "monotone_decreasing": True,
        },
        "laws": rows,
        # the gate: every known law must land on its predeclared side
        "gate_pass": all(r["as_expected"] for r in rows.values()),
        "legacy_threshold_false_positives": [
            name for name, r in rows.items()
            if r["passes_legacy_sigma_gt_0.5"] and not FALSIFICATION_EXPECTED[name]
        ],
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


def build_report(study: dict, checks: dict, figures: dict, runtime: float,
                 falsification: dict | None = None) -> dict:
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
        "criterion": {
            "statement": (
                "spectral decay: sigma > 1 AND pairwise decay rates non-decreasing "
                "AND e(p_min)/e(p_max) > 1e3 AND error monotone decreasing"
            ),
            "legacy_threshold": DECAY_RATE_MIN,
            "discriminating": checks["discriminating"]["conditions"],
        },
        "falsification_control": falsification or {},
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
    falsification = falsification_control(tuple(degrees))
    figures = generate_figures(study, outdir)
    report = build_report(study, checks, figures, runtime, falsification)

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
    disc = checks["discriminating"]["conditions"]
    print(f"sigma > {SIGMA_MIN_DISCRIMINATING}: {'PASS' if disc['sigma_above_threshold']['pass'] else 'FAIL'}"
          f" (measured {disc['sigma_above_threshold']['measured']:.4f})")
    print(f"decay rates non-decreasing: {'PASS' if disc['decay_rates_non_decreasing']['pass'] else 'FAIL'}")
    print(f"error ratio > {ERROR_RATIO_MIN:.0e}: "
          f"{'PASS' if disc['error_ratio_above_threshold']['pass'] else 'FAIL'}"
          f" (measured {disc['error_ratio_above_threshold']['measured']:.3e})")
    print(f"falsification gate (known laws land on their side): "
          f"{'PASS' if falsification['gate_pass'] else 'FAIL'}")
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
