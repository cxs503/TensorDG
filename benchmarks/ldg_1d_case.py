"""Benchmark: mixed-form LDG for the 1-D Poisson problem, with the closure ablations.

Runs four blocks and writes ``validation/ldg_1d/results.json`` (the single source
of every number quoted in ``docs/CASE-DG-LDG-1D-MIXED.md``):

1. ``convergence``      -- Dirichlet MMS ``u = sin(pi x)``, degree 0..4, mesh ladder;
                           L2 errors of ``u`` and ``q = u'``, observed orders, 1-norm
                           condition numbers.
2. ``periodic``         -- same operator, periodic closure (constant mode pinned by a
                           Lagrange multiplier) on ``u = sin(2 pi x)``.
3. ``closure_ablation`` -- the four trace orientations of ``TRACE_POLICIES`` times the
                           two Dirichlet flux closures (``beta = 0`` = bare interior
                           trace, ``beta = 2`` = discrete flux).  Records the rank
                           deficiency, the condition number and the observed order, so
                           that "which convention is wrong, and why" is answered by
                           numbers instead of prose.
4. ``vacuity``          -- polynomial-exactness residuals for every orientation: the
                           trap that makes a consistency test blind to the orientation.

Usage::

    python -m benchmarks.ldg_1d_case --output-dir validation/ldg_1d
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from tensordg.ldg_1d import (
    TRACE_POLICIES,
    assemble_1d,
    convergence_study,
    reference_operators,
    solve_ldg_1d,
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - best effort provenance
        return "unknown"


def _rank_deficiency(matrix, threshold: float = 1e-10) -> int:
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    sv = np.linalg.svd(dense, compute_uv=False)
    return int(np.sum(sv < threshold * sv[0]))


def _order(resolutions, errors):
    out = []
    for i in range(len(errors) - 1):
        if errors[i] > 0 and math.isfinite(errors[i]) and math.isfinite(errors[i + 1]):
            out.append(math.log(errors[i] / errors[i + 1]) / math.log(resolutions[i + 1] / resolutions[i]))
        else:
            out.append(float("nan"))
    return out


def run(output_dir: str, degrees=(0, 1, 2, 3, 4), resolutions=(8, 16, 32, 64)) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Dirichlet manufactured solution ------------------------------
    rhs = lambda x: np.pi ** 2 * np.sin(np.pi * x)
    exact_u = lambda x: np.sin(np.pi * x)
    exact_q = lambda x: np.pi * np.cos(np.pi * x)

    dirichlet = convergence_study(degrees, resolutions, rhs, exact_u, exact_q)

    # --- 2. periodic closure --------------------------------------------
    rhs_p = lambda x: 4.0 * np.pi ** 2 * np.sin(2.0 * np.pi * x)
    exact_up = lambda x: np.sin(2.0 * np.pi * x)
    exact_qp = lambda x: 2.0 * np.pi * np.cos(2.0 * np.pi * x)
    periodic = convergence_study((1, 2, 3), resolutions, rhs_p, exact_up, exact_qp, boundary="periodic")

    # --- 3. closure ablation --------------------------------------------
    ab_degrees = (1, 2)
    ab_resolutions = (8, 16, 32)
    ablation = []
    for policy in TRACE_POLICIES:
        for beta in (0.0, 2.0):
            for degree in ab_degrees:
                errs, conds, deficiency = [], [], []
                for n in ab_resolutions:
                    matrix, force, _ = assemble_1d(n, degree, rhs, beta=beta, trace_policy=policy)
                    deficiency.append(_rank_deficiency(matrix))
                    try:
                        result = solve_ldg_1d(n, degree, rhs, beta=beta, trace_policy=policy)
                        errs.append(result.l2_error_u(exact_u))
                        conds.append(result.condition_number)
                    except FloatingPointError:
                        errs.append(float("nan"))
                        conds.append(float("nan"))
                ablation.append({
                    "trace_policy": policy,
                    "beta": beta,
                    "degree": degree,
                    "resolutions": list(ab_resolutions),
                    "l2_u": errs,
                    "order_u": _order(list(ab_resolutions), errs),
                    "condition_numbers": conds,
                    "rank_deficiency": deficiency,
                })

    # --- 4. vacuity of the polynomial-exactness consistency test --------
    #     u = x (1 - x) is exactly representable for degree >= 2, so every
    #     orientation reproduces it: the residual cannot rank the closures.
    poly_rhs = lambda x: 2.0 + 0.0 * x
    poly_u = lambda x: x * (1.0 - x)
    poly_q = lambda x: 1.0 - 2.0 * x
    vacuity = []
    n_v = 4
    degree_v = 2
    for policy in TRACE_POLICIES:
        matrix, force, ops = assemble_1d(n_v, degree_v, poly_rhs, trace_policy=policy)
        # exact modal coefficients of (u, q) = (x(1-x), 1-2x): representable exactly
        m = degree_v + 1
        h = 1.0 / n_v
        minv = np.linalg.inv(ops.mass)
        x_exact = np.zeros(2 * n_v * m)
        for e in range(n_v):
            xq = (e + 0.5) * h + 0.5 * h * ops.xi
            x_exact[e * 2 * m:e * 2 * m + m] = minv @ (ops.phi @ (ops.weights * poly_q(xq)))
            x_exact[e * 2 * m + m:(e + 1) * 2 * m] = minv @ (ops.phi @ (ops.weights * poly_u(xq)))
        residual = float(np.linalg.norm(matrix @ x_exact - force))
        vacuity.append({
            "trace_policy": policy,
            "degree": degree_v,
            "n_elements": n_v,
            "polynomial_residual": residual,
            "rank_deficiency": _rank_deficiency(matrix),
        })

    # --- 5. k = 0 staggered stencil (structural identity) ---------------
    #     For degree 0 the verified orientation must give, row by row,
    #       h q_e - u_e + u_{e-1} = 0          (q row of element e)
    #       q_e - q_{e+1} = h f_e              (flux row of element e)
    n0 = 4
    matrix, force, ops = assemble_1d(n0, 0, rhs, beta=2.0)
    dense = matrix.toarray()
    h = 1.0 / n0
    stencil_report = []
    for e in range(n0):
        qrow = dense[2 * e]
        frow = dense[2 * e + 1]
        stencil_report.append({
            "element": e,
            "q_row": {"h*q_e": float(qrow[2 * e]), "u_e": float(qrow[2 * e + 1]),
                      "u_e_minus_1": float(qrow[2 * (e - 1) + 1]) if e > 0 else None,
                      "u_boundary": float(force[2 * e])},
            "flux_row_rhs": float(force[2 * e + 1]),
            "flux_row_h_times_f": float(h * rhs(np.array([(e + 0.5) * h]))[0]),
        })

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "benchmarks/ldg_1d_case.py",
        "git_commit": _git_commit(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "case": {
            "name": "dg-ldg-1d-mixed-poisson",
            "problem": "-u'' = f on (0,1)",
            "dirichlet": {"f": "pi^2 sin(pi x)", "u": "sin(pi x)", "q": "pi cos(pi x)"},
            "periodic": {"f": "4 pi^2 sin(2 pi x)", "u": "sin(2 pi x)", "q": "2 pi cos(2 pi x)"},
            "beta": 2.0,
            "beta_sign": 1.0,
        },
        "degrees": list(degrees),
        "resolutions": list(resolutions),
        "convergence": dirichlet,
        "periodic": periodic,
        "closure_ablation": ablation,
        "vacuity": vacuity,
        "k0_stencil": stencil_report,
    }
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="validation/ldg_1d")
    parser.add_argument("--degrees", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 16, 32, 64])
    args = parser.parse_args()
    payload = run(args.output_dir, tuple(args.degrees), tuple(args.resolutions))
    for row in payload["convergence"]:
        print("k=%d order_u=%.3f order_q=%.3f (expected %d)" % (
            row["degree"], row["order_u"], row["order_q"], row["expected_order"]))
    print("wrote", os.path.join(args.output_dir, "results.json"))


if __name__ == "__main__":
    main()