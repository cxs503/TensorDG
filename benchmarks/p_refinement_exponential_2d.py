"""P-type exponential convergence in 2-D (case ``DG-PREFINEMENT-EXPONENTIAL-2D``).

Case anchor: CASE-DG-PREFINEMENT-EXPONENTIAL (2-D extension).
Reference: Babuska & Suri (1994) SIAM Review 36(4):578, DOI 10.21236/ada260237.

Why this benchmark exists (the gap it closes)
---------------------------------------------
The 1-D case (:mod:`benchmarks.p_refinement_exponential`) established
``sigma = 3.93`` on a 16-cell mesh with p = 1..6, and its verifier audit
(m16499) listed "2-D / variable coefficient not covered" as an open limitation.
Meanwhile :func:`tensordg.advection_2d.solve_advection_2d` carried an arbitrary
``degree in (0, 1, 2, 3)`` guard -- the same unprincipled cap that 1-D used to
have -- so no 2-D p-ladder above P3 was even reachable.  This driver (a) lifts
the cap to the documented ``MAX_DEGREE``, (b) measures the 2-D p-ladder, and
(c) applies the *same predeclared composite criterion* that the 1-D case adopted
after audit m16918 (``sigma > 1`` AND pairwise decay rates non-decreasing AND
``e(p_min)/e(p_max) > 1e3`` AND monotone).

Measurement independence
------------------------
Every level is re-measured with a second, numpy-only quadrature path
(``independent_relative_l2``: different Gauss order, different evaluation code)
and the relative dispersion between the two paths is part of the artifact, so a
verifier does not have to trust the solver's own error routine.

Run::

    python -m benchmarks.p_refinement_exponential_2d            # p = 1..6, ~4 min
    python -m benchmarks.p_refinement_exponential_2d --quick    # p = 1..3
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

from benchmarks.p_refinement_exponential import (
    FALSIFICATION_EXPECTED,
    check_discriminating_criterion,
    falsification_control,
)
from tensordg.advection_2d import MAX_DEGREE, solve_advection_2d
from tensordg.cases.prefinement_exponential_2d import (
    CASE_KEY,
    CFL_BASE,
    DEGREES,
    ERROR_RATIO_MIN,
    FINAL_TIME,
    NCELLS,
    REFERENCE,
    SIGMA_MIN_DISCRIMINATING,
    TEMPORAL_TOLERANCE,
    VELOCITY,
    anchor_table,
    p_ladder,
    prefinement_parameters,
)
from tensordg.convergence import roundoff_floor, signal_to_noise

SCHEMA = "tensordg-p-refinement-exponential-2d/1"
ARTIFACT_RELPATH = Path("docs") / "benchmarks" / "results" / "p-refinement-exponential-2d"
DEGREES_QUICK = (1, 2, 3)
LEGACY_SLOPE_THRESHOLD = 0.5


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:  # pragma: no cover - git not always available
        return None


# ---------------------------------------------------------------------------
# Independent measurement path (numpy only, different quadrature order)
# ---------------------------------------------------------------------------
def _legendre_rows(nodes: np.ndarray, degree: int) -> np.ndarray:
    rows = []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        rows.append(np.polynomial.legendre.legval(nodes, polynomial))
    return np.stack(rows, axis=1)          # (nq, degree+1)


def independent_relative_l2(
    x: torch.Tensor,
    y: torch.Tensor,
    coeff: torch.Tensor,
    final_time: float = FINAL_TIME,
    velocity: tuple[float, float] = VELOCITY,
    extra_orders: int = 2,
) -> dict:
    """Relative L2 error recomputed on a higher-order numpy quadrature grid.

    ``error_2d`` (the production routine) uses Gauss order ``2p+4`` and torch.
    Here the order is ``2p+4+extra_orders`` and the whole evaluation is numpy,
    so agreement between the two is not a tautology.
    """
    c = coeff.detach().cpu().numpy() if torch.is_tensor(coeff) else np.asarray(coeff)
    degree = c.shape[-1] - 1
    nx, ny = c.shape[:2]
    nq = 2 * degree + 4 + extra_orders
    nodes, weights = np.polynomial.legendre.leggauss(nq)
    phi = _legendre_rows(nodes, degree)                       # (nq, p+1)
    numerical = np.einsum("xykl,qk,rl->xyqr", c, phi, phi)
    xs = (x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x))
    ys = (y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y))
    hx, hy = 1.0 / nx, 1.0 / ny
    xq = xs[:, None, None, None] + 0.5 * hx * nodes[None, None, :, None]
    yq = ys[None, :, None, None] + 0.5 * hy * nodes[None, None, None, :]
    ax, ay = velocity
    exact = 1.0 + np.sin(2.0 * math.pi * (xq - ax * final_time)) * np.sin(
        2.0 * math.pi * (yq - ay * final_time)
    )
    w2 = weights[None, None, :, None] * weights[None, None, None, :]
    scale = 0.25 * hx * hy
    err2 = scale * float(np.sum(w2 * (numerical - exact) ** 2))
    norm2 = scale * float(np.sum(w2 * exact ** 2))
    return {
        "absolute_l2": math.sqrt(err2),
        "exact_norm": math.sqrt(norm2),
        "relative_l2": math.sqrt(err2 / norm2),
        "quadrature_order": int(nq),
    }


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------
def temporal_converged_level(
    degree: int,
    ncells: int = NCELLS,
    final_time: float = FINAL_TIME,
    velocity: tuple[float, float] = VELOCITY,
    cfl: float = CFL_BASE,
    temporal_tolerance: float = TEMPORAL_TOLERANCE,
    max_refinements: int = 6,
) -> dict:
    """One 2-D p level, with three independent quantities per CFL level.

    Per CFL refinement we record the production error, the independent
    quadrature error and the round-off signal-to-noise ratio, so the artifact
    carries its own credibility checks.
    """
    history: list[dict] = []
    current_cfl = cfl
    production: float | None = None
    converged = False
    independent = None
    norm = None
    metadata = None
    for _ in range(max_refinements + 1):
        x, y, coeff, metadata = solve_advection_2d(
            ncells=ncells, degree=degree, final_time=final_time,
            velocity=velocity, cfl=current_cfl,
        )
        # production error routine
        from tensordg.advection_2d import error_2d

        prod = error_2d(x, y, coeff, final_time, velocity)
        new_error = float(prod["relative_l2"])
        indep = independent_relative_l2(x, y, coeff, final_time, velocity)
        change = (abs(new_error - production) / new_error) if production else float("inf")
        history.append({
            "cfl": current_cfl,
            "dt": metadata["dt"],
            "steps": metadata["steps"],
            "relative_l2": new_error,
            "independent_relative_l2": indep["relative_l2"],
            "path_dispersion": abs(indep["relative_l2"] - new_error) / new_error,
            "change_vs_previous": change,
        })
        production, independent, norm = new_error, indep, float(prod["exact_l2"])
        if change <= temporal_tolerance:
            converged = True
            break
        current_cfl *= 0.5
    assert production is not None and metadata is not None
    nx, ny = metadata["ncells"]
    floor = roundoff_floor(norm)
    return {
        "degree": degree,
        "ncells": [int(nx), int(ny)],
        "dofs": int(nx * ny * (degree + 1) ** 2),
        "relative_l2": production,
        "independent_relative_l2": float(independent["relative_l2"]),
        "path_dispersion": float(history[-1]["path_dispersion"]),
        "quadrature_order_independent": int(independent["quadrature_order"]),
        "exact_norm": norm,
        "roundoff_floor": floor,
        "signal_to_noise": float(signal_to_noise([production], floor)[0]),
        "cfl": history[-1]["cfl"],
        "dt": history[-1]["dt"],
        "steps": history[-1]["steps"],
        "temporal_change": history[-1]["change_vs_previous"],
        "temporal_converged": converged,
        "refinements": len(history) - 1,
        "history": history,
    }


def p_refinement_2d_study(
    degrees: tuple[int, ...] = DEGREES,
    ncells: int = NCELLS,
    final_time: float = FINAL_TIME,
    velocity: tuple[float, float] = VELOCITY,
) -> dict:
    """Run every p level (production + independent + SNR) and the diagnostics."""
    from tensordg.convergence import p_decay_diagnostics

    levels = [
        temporal_converged_level(p, ncells=ncells, final_time=final_time,
                                 velocity=velocity)
        for p in degrees
    ]
    errors = [lv["relative_l2"] for lv in levels]
    diag = p_decay_diagnostics(degrees, errors)
    return {
        "degrees": list(degrees),
        "ncells": ncells,
        "final_time": final_time,
        "velocity": list(velocity),
        "levels": levels,
        "errors": errors,
        "independent_errors": [lv["independent_relative_l2"] for lv in levels],
        "diagnostics": diag,
        "temporal_all_converged": all(lv["temporal_converged"] for lv in levels),
    }


def check_criterion(diag: dict) -> dict:
    """Legacy sigma>0.5 sub-check + the discriminating composite criterion."""
    legacy = {
        "decay_rate": {
            "measured": diag["lsq_decay_rate"],
            "threshold": LEGACY_SLOPE_THRESHOLD,
            "pass": bool(diag["lsq_decay_rate"] > LEGACY_SLOPE_THRESHOLD),
        },
        "monotone_decreasing": {
            "measured": bool(diag["monotone_decreasing"]),
            "required": True,
            "pass": bool(diag["monotone_decreasing"]),
        },
    }
    discriminating = check_discriminating_criterion(diag)
    return {
        **legacy,
        "discriminating": discriminating,
        "overall_pass": bool(
            legacy["decay_rate"]["pass"]
            and legacy["monotone_decreasing"]["pass"]
            and discriminating["pass"]
        ),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _evaluate_field(coeff: torch.Tensor, nx: int, ny: int, degree: int):
    """Cell-centred error field for the contour figure (numpy)."""
    c = coeff.detach().cpu().numpy()
    nq = 6
    nodes, weights = np.polynomial.legendre.leggauss(nq)
    phi = _legendre_rows(nodes, degree)
    numerical = np.einsum("xykl,qk,rl->xyqr", c, phi, phi)
    hx, hy = 1.0 / nx, 1.0 / ny
    xs = (np.arange(nx) + 0.5) * hx
    ys = (np.arange(ny) + 0.5) * hy
    xq = xs[:, None, None, None] + 0.5 * hx * nodes[None, None, :, None]
    yq = ys[None, :, None, None] + 0.5 * hy * nodes[None, None, None, :]
    ax, ay = VELOCITY
    exact = 1.0 + np.sin(2.0 * math.pi * (xq - ax * FINAL_TIME)) * np.sin(
        2.0 * math.pi * (yq - ay * FINAL_TIME)
    )
    err = np.abs(numerical - exact)
    # cell-average the pointwise error for plotting
    return (xs, ys, err.mean(axis=(2, 3)), numerical.mean(axis=(2, 3)),
            exact.mean(axis=(2, 3)))


def generate_figures(study: dict, outdir: Path) -> dict:
    """Semi-log convergence, log10 fit, error field maps and 1-D/2-D comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degrees = study["degrees"]
    errors = study["errors"]
    diag = study["diagnostics"]
    sigma = diag["lsq_decay_rate"]
    figures: dict[str, str] = {}
    root = Path(__file__).resolve().parent.parent

    def _rel(path: Path) -> str:
        # Repo-relative paths so an artifact does not bake in a worktree path
        # (audit m16499 finding); fall back to the raw path outside the repo.
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()

    # 1. semi-log with exponential fit
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogy(degrees, errors, "bo-", linewidth=2, markersize=8,
                label="2-D DG, $8\\times8$ cells")
    p_arr = np.array(degrees, dtype=float)
    p0, e0 = p_arr[0], errors[0]
    fit = [e0 * math.exp(-sigma * (p - p0)) for p in degrees]
    ax.semilogy(p_arr, fit, "r--", linewidth=1.5,
                label=f"fit $\\sigma$ = {sigma:.3f}")
    ax.axhline(errors[0] * math.exp(-0.5 * (p_arr[-1] - p_arr[0])),
               color="g", linestyle=":", alpha=0.6,
               label="Babuska-Suri reference slope $\\sigma$=0.5")
    for p, e in zip(degrees, errors):
        ax.annotate(f"{e:.2e}", (p, e), textcoords="offset points",
                    xytext=(8, 5), fontsize=9)
    ax.set_xlabel("Polynomial degree $p$", fontsize=13)
    ax.set_ylabel("relative $L^2$ error", fontsize=13)
    ax.set_title("2-D p-refinement, exponential convergence\n"
                 f"$u_t+u_x+0.5u_y=0$, $8\\times8$ cells, $T$={study['final_time']}",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    path = outdir / "p_refinement_2d_semilog.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures["p_refinement_2d_semilog.png"] = _rel(path)

    # 2. log10 fit with the discriminating slope
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    log_errors = [math.log10(e) for e in errors]
    ax2.plot(degrees, log_errors, "bo-", linewidth=2, markersize=8,
             label="$\\log_{10}$(L2 error)")
    slope, intercept = np.polyfit(p_arr, np.array(log_errors), 1)
    ax2.plot(p_arr, slope * p_arr + intercept, "r--", linewidth=1.5,
             label=f"linear fit: slope = {slope:.3f}  ($\\sigma$ = {-slope:.3f})")
    ax2.set_xlabel("Polynomial degree $p$", fontsize=13)
    ax2.set_ylabel("$\\log_{10}$(relative L2 error)", fontsize=13)
    ax2.set_title("2-D p-refinement: log-linear decay", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    path2 = outdir / "p_refinement_2d_log10.png"
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    figures["p_refinement_2d_log10.png"] = _rel(path2)

    # 3. solution / error field maps at the lowest and a high degree
    picks = [degrees[0], degrees[-1]]
    fig3, axes = plt.subplots(2, 2, figsize=(10, 8))
    for col, p in enumerate(picks):
        lv = [l for l in study["levels"] if l["degree"] == p][0]
        x, y, coeff, md = solve_advection_2d(
            ncells=study["ncells"], degree=p, final_time=study["final_time"],
            velocity=tuple(study["velocity"]), cfl=lv["cfl"],
        )
        xs, ys, err_field, num_field, exact_field = _evaluate_field(
            coeff, study["ncells"], study["ncells"], p,
        )
        ax_sol = axes[0, col]
        cs = ax_sol.contourf(xs, ys, num_field.T, levels=20, cmap="viridis")
        ax_sol.set_title(f"$u_h$, p={p} (rel. $L^2$={lv['relative_l2']:.2e})",
                         fontsize=11)
        ax_sol.set_xlabel("x"); ax_sol.set_ylabel("y")
        fig3.colorbar(cs, ax=ax_sol)
        ax_err = axes[1, col]
        cs2 = ax_err.contourf(xs, ys, err_field.T, levels=20, cmap="inferno")
        ax_err.set_title(f"|$u_h-u_{{\\rm exact}}$|, p={p} (max={err_field.max():.2e})",
                         fontsize=11)
        ax_err.set_xlabel("x"); ax_err.set_ylabel("y")
        fig3.colorbar(cs2, ax=ax_err)
    fig3.suptitle("2-D p-refinement: solution and pointwise error field", fontsize=13)
    fig3.tight_layout()
    path3 = outdir / "p_refinement_2d_fields.png"
    fig3.savefig(path3, dpi=150)
    plt.close(fig3)
    figures["p_refinement_2d_fields.png"] = _rel(path3)

    # 4. error vs DOFs (cost axis), 1-D reference overlaid from its artifact
    fig4, ax4 = plt.subplots(figsize=(7.5, 5))
    dofs = [lv["dofs"] for lv in study["levels"]]
    ax4.loglog(dofs, errors, "bo-", linewidth=2, markersize=8, label="2-D, 8x8, p=1..P")
    ref_1d = root / "docs" / "benchmarks" / "results" / "p-refinement-exponential" / \
        "p_refinement_results.json"
    if ref_1d.exists():
        try:
            r1d = json.loads(ref_1d.read_text())
            d1 = [lv["dofs"] for lv in r1d["results"]["levels"]]
            e1 = r1d["results"]["errors"]
            ax4.loglog(d1, e1, "s--", color="darkorange", linewidth=1.5,
                       markersize=6,
                       label=f"1-D, 16 cells (published $\\sigma$="
                             f"{r1d['results']['diagnostics']['lsq_decay_rate']:.2f})")
        except Exception:  # pragma: no cover - artifact may be absent
            pass
    ax4.set_xlabel("degrees of freedom", fontsize=13)
    ax4.set_ylabel("relative $L^2$ error", fontsize=13)
    ax4.set_title("Accuracy per degree of freedom: 1-D vs 2-D p-refinement",
                  fontsize=13)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, which="both")
    fig4.tight_layout()
    path4 = outdir / "p_refinement_2d_vs_1d.png"
    fig4.savefig(path4, dpi=150)
    plt.close(fig4)
    figures["p_refinement_2d_vs_1d.png"] = _rel(path4)

    return figures


def build_report(study: dict, checks: dict, falsification: dict, figures: dict,
                 runtime: float, guard_check: dict) -> dict:
    table = anchor_table({
        "levels": study["levels"],
        "diagnostics": study["diagnostics"],
    })
    return {
        "schema": SCHEMA,
        "case": CASE_KEY,
        "ledger_anchor": "CASE-DG-PREFINEMENT-EXPONENTIAL",
        "reference": dict(REFERENCE),
        "problem": {
            "pde": "u_t + ax*u_x + ay*u_y = 0",
            "domain": "[0,1]^2 periodic",
            "initial_condition": "1 + sin(2*pi*x)*sin(2*pi*y)",
            "exact_solution": "1 + sin(2*pi*(x-ax*t))*sin(2*pi*(y-ay*t))",
            "velocity": study["velocity"],
            "final_time": study["final_time"],
            "ncells": study["ncells"],
            "cfl_base": CFL_BASE,
            "temporal_tolerance": TEMPORAL_TOLERANCE,
            "max_degree": MAX_DEGREE,
        },
        "results": {
            "degrees": study["degrees"],
            "errors": study["errors"],
            "independent_errors": study["independent_errors"],
            "levels": study["levels"],
            "diagnostics": study["diagnostics"],
            "table": table,
        },
        "checks": {
            "criterion": checks,
            "temporal_all_converged": study["temporal_all_converged"],
            "max_path_dispersion": max(lv["path_dispersion"] for lv in study["levels"]),
            "min_signal_to_noise": min(lv["signal_to_noise"] for lv in study["levels"]),
            "degree_guard": guard_check,
            "overall_pass": bool(
                checks["overall_pass"]
                and study["temporal_all_converged"]
                and (
                    falsification.get("gate_pass", False)
                    if falsification.get("gate_applicable", True) else True
                )
                and guard_check["pass"]
            ),
        },
        "criterion": {
            "statement": (
                "spectral decay: sigma > 1 AND pairwise decay rates non-decreasing "
                "AND e(p_min)/e(p_max) > 1e3 AND error monotone decreasing"
            ),
            "sigma_min": SIGMA_MIN_DISCRIMINATING,
            "error_ratio_min": ERROR_RATIO_MIN,
            "legacy_slope_threshold": LEGACY_SLOPE_THRESHOLD,
        },
        "falsification_control": falsification,
        "verdict": (
            ("PASS" if checks["overall_pass"] else "FAIL")
            if falsification.get("gate_applicable", True) else "SMOKE"
        ),
        "figures": figures,
        "runtime_seconds": runtime,
    }


def guard_check() -> dict:
    """Prove the P3 cap is gone and the documented bound is enforced."""
    rows = {}
    for degree in (0, 1, 3, 4, 5, MAX_DEGREE):
        try:
            x, y, coeff, md = solve_advection_2d(
                ncells=4, degree=degree, final_time=0.1, velocity=VELOCITY, cfl=0.2,
            )
            rows[str(degree)] = {"accepted": True, "dofs": int(coeff[..., 0, 0].numel())}
            del x, y, coeff, md
        except ValueError as exc:
            rows[str(degree)] = {"accepted": False, "error": str(exc)}
    try:
        solve_advection_2d(ncells=4, degree=MAX_DEGREE + 1, final_time=0.1)
        rows[str(MAX_DEGREE + 1)] = {"accepted": True}
    except ValueError as exc:
        rows[str(MAX_DEGREE + 1)] = {"accepted": False, "error": str(exc)}
    ok = (
        rows["0"]["accepted"] and rows["3"]["accepted"] and rows["4"]["accepted"]
        and rows["5"]["accepted"] and rows[MAX_DEGREE.__str__()]["accepted"]
        and not rows[str(MAX_DEGREE + 1)]["accepted"]
    )
    return {"degrees": rows, "documented_max_degree": MAX_DEGREE, "pass": bool(ok)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="2-D p-type exponential convergence benchmark",
    )
    parser.add_argument("--quick", action="store_true", help="p = 1..3 smoke run")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    degrees = DEGREES_QUICK if args.quick else DEGREES
    root = Path(__file__).resolve().parent.parent
    outdir = Path(args.outdir) if args.outdir else root / ARTIFACT_RELPATH
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    study = p_refinement_2d_study(degrees=degrees)
    checks = check_criterion(study["diagnostics"])
    falsification = falsification_control(tuple(degrees))
    # The predeclared FALSIFICATION_EXPECTED table is calibrated for the full
    # 6-degree ladder; on a shorter smoke ladder (--quick) a genuine exponential
    # law cannot clear the 1e3 ratio gate, so the table is not applicable and the
    # run declares a SMOKE result instead of a verdict.
    falsification["gate_applicable"] = len(degrees) >= len(DEGREES)
    guards = guard_check()
    figures = generate_figures(study, outdir)
    runtime = time.time() - t0

    report = build_report(study, checks, falsification, figures, runtime, guards)
    json_path = outdir / "p_refinement_2d_results.json"
    json_path.write_text(json.dumps(report, indent=2, default=float))

    provenance = {
        "case": CASE_KEY,
        "ledger_anchor": "CASE-DG-PREFINEMENT-EXPONENTIAL",
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
        "parameters": prefinement_parameters(),
    }
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    csv_path = outdir / "p_refinement_2d_table.csv"
    with open(csv_path, "w") as f:
        f.write("p,ncells,dofs,L2_error,independent_L2,path_dispersion,"
                "snr_vs_roundoff,cfl,steps,temporal_converged\n")
        for lv in study["levels"]:
            f.write(
                f"{lv['degree']},{lv['ncells'][0]}x{lv['ncells'][1]},{lv['dofs']},"
                f"{lv['relative_l2']:.6e},{lv['independent_relative_l2']:.6e},"
                f"{lv['path_dispersion']:.2e},{lv['signal_to_noise']:.2e},"
                f"{lv['cfl']},{lv['steps']},{int(lv['temporal_converged'])}\n"
            )

    print("=" * 68)
    print("2-D p-refinement exponential convergence — CASE-DG-PREFINEMENT-EXPONENTIAL-2D")
    print("=" * 68)
    print(f"  mesh {study['ncells']}x{study['ncells']}, T={study['final_time']}, "
          f"velocity={study['velocity']}")
    print(f"  {'p':>2} {'DOFs':>6} {'L2':>12} {'independent':>12} {'SNR':>9} {'cfl':>7}")
    for lv in study["levels"]:
        print(f"  {lv['degree']:>2} {lv['dofs']:>6} {lv['relative_l2']:12.3e} "
              f"{lv['independent_relative_l2']:12.3e} {lv['signal_to_noise']:9.1e} "
              f"{lv['cfl']:7.4f}")
    print(f"  sigma = {checks['decay_rate']['measured']:.4f}, "
          f"pairwise rates = "
          f"{[round(r, 3) for r in study['diagnostics']['decay_rates']]}")
    print(f"  discriminating criterion: "
          f"{'PASS' if checks['discriminating']['pass'] else 'FAIL'}  "
          f"(sigma>{SIGMA_MIN_DISCRIMINATING}, rates non-decreasing, "
          f"ratio>{ERROR_RATIO_MIN:g})")
    print(f"  falsification control: "
          f"{'PASS' if falsification['gate_pass'] else 'FAIL'}"
          f"{'' if falsification['gate_applicable'] else ' (not applicable on smoke ladder)'}")
    print(f"  degree guard (P3 cap removed): "
          f"{'PASS' if guards['pass'] else 'FAIL'}")
    print(f"  VERDICT: {report['verdict']}   runtime {runtime:.1f} s")
    print(f"  artifacts: {outdir}")
    return 0 if report["verdict"] in ("PASS", "SMOKE") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())