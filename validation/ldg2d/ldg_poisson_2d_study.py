#!/usr/bin/env python
"""Validation study for the 2-D LDG Poisson solver (docs/CASE-DG-LDG-2D.md).

What this script establishes (and how it can fail)
--------------------------------------------------
The shipped ``tensordg.poisson_2d_ldg`` module is a modal-Legendre LDG
discretisation of ``-Delta u = f`` on a Cartesian mesh in 2-D.  A previous
prototype was deliberately kept out of the public namespace because its
Dirichlet closure was rank deficient.  This study re-derives the closure with
the two ingredients verified in 1-D (``tensordg.ldg_1d``) -- face-oriented
alternating traces and the discrete Dirichlet flux ``q_hat = q_int + beta/h
(u_int - g_D)`` -- and measures the resulting scheme against the manufactured
analytic solution

    u(x, y) = sin(pi x) sin(pi y),   -Delta u = 2 pi^2 sin(pi x) sin(pi y).

Gates (pre-registered before the run, anchored on Castillo, Cockburn, Perugia &
Schoetzau 2000, SIAM J. Numer. Anal. 38:1676, DOI:10.1137/S0036142900371003):

  G1  h-convergence, k = 1, 2 on N = 4, 8, 16, 32: least-squares log-log slope
      of the relative L2 error >= k + 1 - 0.3, and the finest-grid error < 3%.
  G2  non-homogeneous Dirichlet data (u = sin sin + x): same order gate and
      error < 3% -- a solver that silently dropped the boundary data would fail.
  G3  beta-insensitivity: beta in {1, 2, 4, 8, 16} at N = 16 changes the L2
      error by <= 5% (the flux is a stabilisation, not a tuning knob).
  G4  closure ablation: the legacy mirror closure and the bare closure must
      *fail* on the same meshes (mirror diverges, bare is singular of rank
      deficiency >= 1) while the discrete flux converges -- the gate has teeth.
  G5  polynomial exactness: for u = x^2 (in the k >= 2 space) the relative
      residual of the assembled system is < 1e-12, and for k = 1 (u not
      representable) it is > 1e-6 -- consistency is checked, not assumed.
  G6  conditioning: sigma_min > 0 and the dense-mesh condition number < 1e8.

Every number written into the report comes from this script's JSON artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp

from tensordg.poisson_2d_ldg import (
    ldg_poisson_convergence,
    manufactured_solution,
    solve_poisson_2d_ldg,
    superposition_solution,
)

GATES = {
    "order_margin": 0.3,
    "fine_error_max": 3.0e-2,
    "beta_spread_max": 5.0e-2,
    "polynomial_residual_exact": 1.0e-12,
    "polynomial_residual_coarse": 1.0e-6,
    "condition_max": 1.0e8,
}

CONVERGENCE_GRIDS = (4, 8, 16, 32)
BETA_VALUES = (1.0, 2.0, 4.0, 8.0, 16.0)


def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def _projection_coefficients(func, ncells: int, degree: int, direction: str) -> np.ndarray:
    """Modal coefficients of a separable-argument polynomial on each element."""
    nq = max(2 * degree + 4, 8)
    nodes, weights = np.polynomial.legendre.leggauss(nq)
    modes = degree + 1
    phi = np.zeros((modes, nq))
    for j in range(modes):
        coeffs = np.zeros(j + 1)
        coeffs[-1] = 1.0
        phi[j] = np.polynomial.legendre.legval(nodes, coeffs)
    mass = np.einsum("q,iq,jq->ij", weights, phi, phi)
    h = 1.0 / ncells
    out = np.zeros((ncells, modes))
    for e in range(ncells):
        x = (e + 0.5) * h + 0.5 * h * nodes
        f = np.asarray(func(x), dtype=float)
        out[e] = np.linalg.solve(mass, np.einsum("q,iq,q->i", weights, phi, f))
    return out


def gate_polynomial_exactness(degree: int, ncells: int = 3) -> dict:
    """Relative residual of the assembled system on an exactly representable u."""
    exact = {
        "u": lambda x, y: x * x,
        "grad_x": lambda x, y: 2.0 * x,
        "grad_y": lambda x, y: 0.0 * x,
        "f": lambda x, y: -2.0 + 0.0 * x,
        "dirichlet": lambda x, y: np.asarray(x, dtype=float) ** 2,
    }
    data = solve_poisson_2d_ldg(
        ncells, degree, return_matrix=True,
        rhs=exact["f"], dirichlet=exact["dirichlet"],
    )
    matrix, rhs = data["matrix"], data["rhs"]
    h = 1.0 / ncells
    modes = degree + 1
    cu = _projection_coefficients(lambda x: x * x, ncells, degree, "x")
    cq = _projection_coefficients(lambda x: 2.0 * x, ncells, degree, "x")
    modes_y = np.zeros((ncells, modes))
    modes_y[:, 0] = 1.0  # u does not depend on y: only the constant mode is active
    block = modes * modes
    x = np.zeros((ncells, ncells, 3, modes, modes))
    for ix in range(ncells):
        for iy in range(ncells):
            x[ix, iy, 0] = np.outer(cu[ix], modes_y[iy])
            x[ix, iy, 1] = np.outer(cq[ix], modes_y[iy])
            # qy = 0
    residual = matrix @ x.reshape(-1) - rhs
    scale = np.linalg.norm(rhs)
    return {
        "degree": degree,
        "ncells": ncells,
        "ndofs": int(data["ndofs"]),
        "residual_abs": float(np.linalg.norm(residual)),
        "rhs_norm": float(scale),
        "residual_relative": float(np.linalg.norm(residual) / scale),
    }


def gate_condition(degree: int, ncells: int) -> dict:
    data = solve_poisson_2d_ldg(ncells, degree, return_matrix=True)
    dense = data["matrix"].toarray()
    sv = np.linalg.svd(dense, compute_uv=False)
    return {
        "degree": degree,
        "ncells": ncells,
        "ndofs": int(data["ndofs"]),
        "sigma_max": float(sv[0]),
        "sigma_min": float(sv[-1]),
        "condition": float(sv[0] / sv[-1]),
    }


def gate_rank_deficiency(degree: int, ncells: int) -> dict:
    data = solve_poisson_2d_ldg(
        ncells, degree, boundary_closure="bare", return_matrix=True,
    )
    dense = data["matrix"].toarray()
    sv = np.linalg.svd(dense, compute_uv=False)
    tol = 1e-10 * sv[0]
    return {
        "degree": degree,
        "ncells": ncells,
        "ndofs": int(data["ndofs"]),
        "sigma_max": float(sv[0]),
        "sigma_min": float(sv[-1]),
        "deficiency": int(np.sum(sv < tol)),
    }


def _study_case(solution: dict, degree: int, grids, **kwargs) -> dict:
    errors, grad_errors, rows = [], [], []
    for n in grids:
        result = solve_poisson_2d_ldg(
            n, degree,
            rhs=solution["f"], dirichlet=solution["dirichlet"],
            exact=solution["u"],
            exact_grad=(solution["grad_x"], solution["grad_y"]),
            **kwargs,
        )
        errors.append(result["l2_relative_error"])
        grad_errors.append(result["grad_relative_error"])
        rows.append({
            "degree": degree,
            "ncells": n,
            "ndofs": result["ndofs"],
            "l2_relative_error": result["l2_relative_error"],
            "grad_relative_error": result["grad_relative_error"],
        })
    local = [
        math.log(errors[i] / errors[i + 1]) / math.log(grids[i + 1] / grids[i])
        for i in range(len(grids) - 1)
    ]
    grad_local = [
        math.log(grad_errors[i] / grad_errors[i + 1])
        / math.log(grids[i + 1] / grids[i])
        for i in range(len(grids) - 1)
    ]
    return {
        "rows": rows,
        "local_orders": local,
        "local_orders_grad": grad_local,
        "least_squares_order": float(
            np.polyfit(np.log(grids), np.log(errors), 1)[0] * -1.0
        ),
        "finest_error": float(errors[-1]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="validation/ldg2d")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    base = manufactured_solution()
    nonhom = superposition_solution()

    # ---- G1 / G2: convergence on the two manufactured solutions -------------
    convergence = {}
    for name, solution in (("homogeneous", base), ("non_homogeneous", nonhom)):
        convergence[name] = {}
        for k in (1, 2):
            convergence[name][k] = _study_case(solution, k, CONVERGENCE_GRIDS)

    # ---- G3: beta insensitivity --------------------------------------------
    beta_table = {}
    for k in (1, 2):
        beta_rows = []
        for beta in BETA_VALUES:
            result = solve_poisson_2d_ldg(16, k, dirichlet_flux_beta=beta)
            beta_rows.append({
                "beta": beta,
                "l2_relative_error": result["l2_relative_error"],
                "grad_relative_error": result["grad_relative_error"],
            })
        ref = next(r for r in beta_rows if r["beta"] == 4.0)["l2_relative_error"]
        spread = max(abs(r["l2_relative_error"] - ref) / ref for r in beta_rows)
        beta_table[k] = {"rows": beta_rows, "max_relative_spread": float(spread)}

    # ---- G4: closure ablation ----------------------------------------------
    ablation = {}
    for closure in ("flux", "mirror", "bare"):
        for k in (1, 2):
            errs = []
            failed = None
            for n in (4, 8, 16):
                try:
                    with np.errstate(all="raise"):
                        result = solve_poisson_2d_ldg(n, k, boundary_closure=closure)
                    errs.append(result["l2_relative_error"])
                except Exception as exc:  # singular / non-finite solution
                    failed = f"{type(exc).__name__}: {exc}"
                    break
            ablation[f"{closure}_k{k}"] = {
                "closure": closure,
                "degree": k,
                "errors": errs,
                "failure": failed,
            }
    reverse_rows = {}
    for k in (1, 2):
        try:
            reverse_rows[k] = {
                "errors": [
                    solve_poisson_2d_ldg(n, k, trace_policy="reversed")["l2_relative_error"]
                    for n in (4, 8, 16)
                ],
                "failure": None,
            }
        except Exception as exc:
            reverse_rows[k] = {"errors": [], "failure": f"{type(exc).__name__}: {exc}"}

    # ---- G5: polynomial exactness ------------------------------------------
    polynomial = {k: gate_polynomial_exactness(k) for k in (1, 2)}

    # ---- G6: conditioning and bare-closure rank ----------------------------
    conditions = [gate_condition(k, n) for k in (1, 2) for n in (2, 4)]
    rank_deficiency = [gate_rank_deficiency(k, 4) for k in (1, 2)]

    # ---- verdicts ----------------------------------------------------------
    gate_results = {}
    for name, table in convergence.items():
        for k, record in table.items():
            target = k + 1 - GATES["order_margin"]
            gate_results[f"G1/{name}/k{k}"] = {
                "target_order": target,
                "measured_order": record["least_squares_order"],
                "finest_error": record["finest_error"],
                "pass": bool(record["least_squares_order"] >= target
                             and record["finest_error"] < GATES["fine_error_max"]),
            }
    gate_results["G3"] = {
        "max_relative_spread": max(v["max_relative_spread"] for v in beta_table.values()),
        "pass": bool(max(v["max_relative_spread"] for v in beta_table.values())
                     <= GATES["beta_spread_max"]),
    }
    mirror_errors = [v for key, v in ablation.items() if key.startswith("mirror")]
    gate_results["G4/mirror_diverges"] = {
        "max_error_at_n8": max(v["errors"][1] for v in mirror_errors if len(v["errors"]) > 1),
        "pass": bool(all(len(v["errors"]) > 1 and v["errors"][1] > 1.0 for v in mirror_errors)),
    }
    bare_ok = all(r["deficiency"] >= 1 for r in rank_deficiency)
    gate_results["G4/bare_singular"] = {
        "deficiencies": [r["deficiency"] for r in rank_deficiency],
        "pass": bool(bare_ok),
    }
    gate_results["G5"] = {
        "residuals": {str(k): v["residual_relative"] for k, v in polynomial.items()},
        "pass": bool(
            polynomial[2]["residual_relative"] < GATES["polynomial_residual_exact"]
            and polynomial[1]["residual_relative"] > GATES["polynomial_residual_coarse"]
        ),
    }
    condition_max = max(r["condition"] for r in conditions)
    gate_results["G6"] = {
        "max_condition": condition_max,
        "min_sigma": min(r["sigma_min"] for r in conditions),
        "pass": bool(condition_max < GATES["condition_max"]
                     and min(r["sigma_min"] for r in conditions) > 0.0),
    }
    all_pass = all(v["pass"] for v in gate_results.values() if isinstance(v, dict) and "pass" in v)
    verdict = "PASS" if all_pass else "FAIL"

    report = {
        "case": "CASE-DG-LDG-2D",
        "verdict": verdict,
        "gates": GATES,
        "gate_results": gate_results,
        "convergence": {name: {str(k): v for k, v in table.items()}
                        for name, table in convergence.items()},
        "beta_sensitivity": {str(k): v for k, v in beta_table.items()},
        "ablation": ablation,
        "reversed_traces": {str(k): v for k, v in reverse_rows.items()},
        "polynomial_exactness": {str(k): v for k, v in polynomial.items()},
        "conditioning": conditions,
        "rank_deficiency": rank_deficiency,
        "config": {
            "grids": list(CONVERGENCE_GRIDS),
            "beta_values": list(BETA_VALUES),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "provenance": {
            "script": str(Path(__file__)),
            "script_sha256": _script_sha256(),
            "git_commit": _git_commit(),
            "runtime_seconds": None,
        },
    }

    if not args.no_figures:
        report["figures"] = _figures(convergence, ablation, beta_table, conditions, outdir)

    report["provenance"]["runtime_seconds"] = round(time.time() - t0, 1)
    payload = json.dumps(report, indent=2, sort_keys=True)
    (outdir / "ldg_poisson_2d_study.json").write_text(payload)
    (outdir / "ldg_poisson_2d_study.json.sha256").write_text(
        hashlib.sha256(payload.encode()).hexdigest() + "  ldg_poisson_2d_study.json\n",
    )

    # console summary (the same numbers as the JSON, no separate transcription)
    for key, value in gate_results.items():
        detail = {k: v for k, v in value.items() if k != "pass"}
        print(f"[{'PASS' if value['pass'] else 'FAIL'}] {key}: {detail}")
    print(f"verdict={verdict} runtime={report['provenance']['runtime_seconds']}s -> {outdir}")
    return 0 if verdict == "PASS" else 1


def _figures(convergence, ablation, beta_table, conditions, outdir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for k in (1, 2):
        rows = convergence["homogeneous"][k]["rows"]
        h = np.array([1.0 / r["ncells"] for r in rows])
        err = np.array([r["l2_relative_error"] for r in rows])
        ax[0].loglog(h, err, "o-", label=f"k={k} (slope {convergence['homogeneous'][k]['least_squares_order']:.2f})")
        ax[0].loglog(h, err[0] * (h / h[0]) ** (k + 1), "k:", lw=1,
                     label=f"$h^{{{k + 1}}}$ reference")
    ax[0].set_xlabel("h"); ax[0].set_ylabel("relative $L_2$ error")
    ax[0].set_title("G1: h-convergence, u = sin(pi x) sin(pi y)")
    ax[0].invert_xaxis(); ax[0].axhline(GATES["fine_error_max"], color="r", ls="--", lw=1)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

    for k in (1, 2):
        rows = convergence["non_homogeneous"][k]["rows"]
        h = np.array([1.0 / r["ncells"] for r in rows])
        err = np.array([r["l2_relative_error"] for r in rows])
        ax[1].loglog(h, err, "s-", label=f"k={k} (slope {convergence['non_homogeneous'][k]['least_squares_order']:.2f})")
    ax[1].set_xlabel("h"); ax[1].set_ylabel("relative $L_2$ error")
    ax[1].set_title("G2: non-homogeneous Dirichlet data, u = sin sin + x")
    ax[1].invert_xaxis(); ax[1].axhline(GATES["fine_error_max"], color="r", ls="--", lw=1)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    path = outdir / "fig_ldg2d_convergence.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    figures.append({"name": path.name, "caption": "G1/G2 convergence of the 2-D LDG solver"})

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, record in ablation.items():
        if record["failure"] or not record["errors"]:
            continue
        errs = np.array(record["errors"])
        ax.semilogy([4, 8, 16][: len(errs)], errs, "o-",
                    label=f"{record['closure']}, k={record['degree']}")
    ax.set_xlabel("N (cells per direction)"); ax.set_ylabel("relative $L_2$ error")
    ax.set_title("G4: closure ablation (mirror/bare fail, flux converges)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    path = outdir / "fig_ldg2d_ablation.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    figures.append({"name": path.name, "caption": "G4 closure/orientation ablation"})

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for k in (1, 2):
        rows = beta_table[k]["rows"]
        ax[0].semilogx([r["beta"] for r in rows], [r["l2_relative_error"] for r in rows],
                       "o-", label=f"k={k}")
    ax[0].set_xlabel("$\\beta$"); ax[0].set_ylabel("relative $L_2$ error")
    ax[0].set_title("G3: flux strength insensitivity (N=16)")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    labels = [f"k={r['degree']},N={r['ncells']}" for r in conditions]
    ax[1].bar(labels, [r["condition"] for r in conditions], color="tab:blue")
    ax[1].set_yscale("log"); ax[1].set_ylabel("$\\mathrm{cond}(A)$")
    ax[1].set_title("G6: conditioning of the assembled system")
    ax[1].grid(alpha=0.3, axis="y")
    path = outdir / "fig_ldg2d_beta_condition.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    figures.append({"name": path.name, "caption": "G3 beta insensitivity and G6 conditioning"})
    return figures


if __name__ == "__main__":
    sys.exit(main())