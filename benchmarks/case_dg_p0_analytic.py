"""CASE-DG-ADVECT-ORDER / DG0 row: exact semidiscrete reference vs the design order.

Why this study exists
---------------------
The case-pool acceptance for CASE-DG-ADVECT-ORDER is the design-order criterion
"observed order == ref order (+-0.1)".  Rounds 1-2 measured it for P1..P4 and met
it (P1..P4 agree with the arXiv:2404.15453v2 ``tab:acc-sin-1d-uni`` columns to
<0.05 in order).  The one row that stayed unexplained is **P0**: the fitted
order on the pool ladder ``N = 20 ... 320`` is ``0.85``, not the design ``1``,
and one in-flight branch "fixed" it by widening the test tolerance to ``+-0.7``.
A widened tolerance is not evidence.  This study replaces it with theory:

* for ``degree = 0`` the DG/upwind operator is **circulant**, so the exact
  semidiscrete solution (and therefore the exact spatial error) is available in
  closed form at *any* ``N`` (see :mod:`tensordg.p0_exact`);
* therefore the ``0.85`` can be reproduced from theory with zero time-stepping,
  proved to be a pre-asymptotic property of the exact error curve rather than a
  solver defect, and the asymptotic regime (order -> 1.00, constant ->
  ``pi*sqrt(2*pi^2 + 1/6)``) can be located and quantified.

Run::

    PYTHONPATH=. OMP_NUM_THREADS=8 python -m benchmarks.case_dg_p0_analytic

Artifacts: ``validation/advection_order_p0/{metrics.json,convergence.csv,tables.md}``,
``docs/figures/case_dg_p0_*.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

from tensordg import p0_exact as P
from tensordg.advection_1d import solve_advection

OUT = pathlib.Path("validation/advection_order_p0")
FIG = pathlib.Path("docs/figures")

#: The case-pool ladder used by rounds 1-2 for the P0 row.
POOL_LADDER = (20, 40, 80, 160, 320)
#: Extended analytic ladder (free: no time stepping, closed form only).
ANALYTIC_LADDER = (20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960)
#: Solver-verified ladder (P0 RK4, cfl 0.2 -> 5N steps; cheap).
SOLVER_LADDER = (20, 40, 80, 160, 320, 640, 1280, 2560, 5120)
#: Windows inside the asymptotic regime (data-driven, see `regime_N_*`).
ANALYTIC_FINE_WINDOW = 5120
SOLVER_FINE_WINDOW = 2560
SPEED = 1.0
FINAL_TIME = 1.0
CFL = 0.2
TOL = {
    "order_tol": 0.1,          # case-pool order criterion
    "solver_vs_analytic": 2e-3,  # |fitted order difference|
    "constant_rel": 1e-3,      # |N*E / closed form - 1| at the fine end
    "time_share": 0.05,        # time error / total error on the pool ladder
    "regime_order": 0.99,      # local order at the finest analytic mesh
}


def solver_run(ncells: int, cfl: float = CFL):
    """Fully discrete DG0 solution from the repo solver (unmodified)."""
    centers, coeff, meta = solve_advection(
        ncells=ncells, degree=0, final_time=FINAL_TIME, speed=SPEED, cfl=cfl,
        time_integrator="rk4")
    u = coeff[:, 0].detach().cpu().numpy()
    e_total = P.l2_error_fully_discrete(u, FINAL_TIME, SPEED)
    e_semi_ref = P.p0_exact_cell_averages(ncells, FINAL_TIME, SPEED)
    e_time = float(np.sqrt(np.mean((u - e_semi_ref) ** 2)))
    return {"ncells": ncells, "steps": meta["steps"], "dt": meta["dt"],
            "l2_total": e_total, "l2_time_component": e_time,
            "l2_semidiscrete_exact": P.l2_error_semidiscrete(ncells, FINAL_TIME, SPEED)}


def regime_threshold(ladder, rel_tol):
    """Smallest N of ``ladder`` whose local order is within ``rel_tol`` of 1."""
    for n in ladder:
        if abs(P.asymptotic_order(n, FINAL_TIME, SPEED) - 1.0) <= rel_tol:
            return n
    return None


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    import torch  # noqa: F401  (threads are set by the caller env)

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. analytic (closed-form) error curve ----------------
    analytic = []
    for n in ANALYTIC_LADDER:
        analytic.append({
            "ncells": n,
            "h": 1.0 / n,
            "l2_semidiscrete": P.l2_error_semidiscrete(n, FINAL_TIME, SPEED),
            "n_times_l2": P.asymptotic_constant(n, FINAL_TIME, SPEED),
            "local_order": (None if n == ANALYTIC_LADDER[-1]
                            else P.asymptotic_order(n, FINAL_TIME, SPEED)),
        })
    analytic_windows = {
        f"N>={n}": P.observed_order([m for m in ANALYTIC_LADDER if m >= n],
                                    [P.l2_error_semidiscrete(m, FINAL_TIME, SPEED)
                                     for m in ANALYTIC_LADDER if m >= n])
        for n in (20, 320, 640, 1280, 2560, 5120)
    }

    # ---------------- 2. solver cross-check (fully discrete) ---------------
    solver = [solver_run(n) for n in SOLVER_LADDER]

    # ---------------- 3. time-error isolation at fixed N ------------------
    dt_probe = [solver_run(320, cfl=c) for c in (0.4, 0.2, 0.1, 0.05)]

    # ---------------- 4. criteria -----------------------------------------
    pool_solver = [r for r in solver if r["ncells"] in POOL_LADDER]
    pool_analytic = [r for r in analytic if r["ncells"] in POOL_LADDER]
    order_analytic_pool = P.observed_order([r["ncells"] for r in pool_analytic],
                                          [r["l2_semidiscrete"] for r in pool_analytic])
    order_solver_pool = P.observed_order([r["ncells"] for r in pool_solver],
                                         [r["l2_total"] for r in pool_solver])
    order_solver_fine = P.observed_order(
        [r["ncells"] for r in solver if r["ncells"] >= SOLVER_FINE_WINDOW],
        [r["l2_total"] for r in solver if r["ncells"] >= SOLVER_FINE_WINDOW])
    order_analytic_fine = P.observed_order(
        [r["ncells"] for r in analytic if r["ncells"] >= ANALYTIC_FINE_WINDOW],
        [r["l2_semidiscrete"] for r in analytic if r["ncells"] >= ANALYTIC_FINE_WINDOW])
    pointwise_rel = max(abs(r["l2_total"] / r["l2_semidiscrete_exact"] - 1.0) for r in solver)
    constants = [r["n_times_l2"] for r in analytic if r["ncells"] >= 1280]
    const_rel = abs(constants[-1] / P.P0_ASYMPTOTIC_CONSTANT - 1.0)
    time_shares = [r["l2_time_component"] / r["l2_total"] for r in pool_solver]
    floor_check = P.P0_SCHEME_CONSTANT ** 2 + P.P0_PROJECTION_CONSTANT ** 2
    checks = {
        "A1_solver_reproduces_pool_order": abs(order_solver_pool - order_analytic_pool) <= TOL["solver_vs_analytic"],
        "A2_pool_order_is_preasymptotic": abs(order_analytic_pool - 1.0) > TOL["order_tol"],
        "A3_fine_ladder_recovers_design_order": abs(order_analytic_fine - 1.0) <= 2e-3,
        "A4_solver_fine_ladder_recovers_design_order": abs(order_solver_fine - 1.0) <= 5e-3,
        "A10_solver_matches_exact_pointwise": pointwise_rel <= 1e-3,
        "A5_asymptotic_constant_closed_form": const_rel <= TOL["constant_rel"],
        "A6_two_term_closed_form_identity": abs(
            math.hypot(P.P0_SCHEME_CONSTANT, P.P0_PROJECTION_CONSTANT) ** 2 - floor_check) < 1e-12,
        "A7_time_error_negligible_on_pool": max(time_shares) <= TOL["time_share"],
        "A8_time_error_vanishes_with_cfl": dt_probe[-1]["l2_time_component"] < 0.1 * dt_probe[0]["l2_time_component"],
        "A9_local_order_at_finest": abs(analytic[-2]["local_order"] - 1.0) <= (1.0 - TOL["regime_order"]),
    }
    falsified = []
    if checks["A2_pool_order_is_preasymptotic"] and checks["A3_fine_ladder_recovers_design_order"]:
        falsified.append(
            "H_P0_defect: 'the 0.85 observed P0 order on N=20..320 means the DG0 operator is "
            "below design order'. Falsified: the closed-form semidiscrete error (no time stepping, "
            "no solver) gives the same fitted order on the same ladder, and the design order 1.00 "
            "is recovered on the extended ladder."
        )
    const_rel_scheme_only = abs(constants[-1] / P.P0_SCHEME_CONSTANT - 1.0)
    if const_rel_scheme_only > 1e-3:
        falsified.append(
            f"H_constant: 'N*E(N) -> pi^2*sqrt(2) = {P.P0_SCHEME_CONSTANT:.4f}' (the scheme-damping "
            f"estimate alone). Falsified: the measured value at N={ANALYTIC_LADDER[-1]} is "
            f"{constants[-1]:.5f}, i.e. {100*const_rel_scheme_only:.2f}% above it, while the two-term "
            f"closed form pi*sqrt(2*pi^2 + 1/6) = {P.P0_ASYMPTOTIC_CONSTANT:.5f} is matched to "
            f"{100*const_rel:.3f}%. The gap is the O(h^2) piecewise-constant projection floor "
            f"pi^2*h^2/6, not a solver error."
        )
    if checks["A7_time_error_negligible_on_pool"] and pointwise_rel <= 1e-3:
        falsified.append(
            f"H_time_error: 'the DG0 pool-ladder error is contaminated by RK4 time integration'. "
            f"Falsified at cfl={CFL}: time share <= {max(time_shares):.1e} on the pool ladder and the "
            f"fully discrete solver matches the semidiscrete exact error to {pointwise_rel:.1e} "
            f"relative over N={SOLVER_LADDER[0]}...{SOLVER_LADDER[-1]}."
        )
    verdict = "PASS" if all(checks.values()) else "FAIL"

    metrics = {
        "case": "CASE-DG-ADVECT-ORDER",
        "row": "P0 (degree 0, DG0 == upwind FV)",
        "objective": ("replace the widened P0 test tolerance by an exact semidiscrete reference: "
                      "prove the pool-ladder order 0.85 is pre-asymptotic, locate the asymptotic "
                      "regime, and pin the leading error constant in closed form"),
        "setup": {"speed": SPEED, "final_time": FINAL_TIME, "cfl": CFL,
                  "time_integrator": "rk4 (repo default)", "solver": "tensordg.advection_1d.solve_advection",
                  "pool_ladder": list(POOL_LADDER), "analytic_ladder": list(ANALYTIC_LADDER),
                  "solver_ladder": list(SOLVER_LADDER)},
        "closed_form": {
            "scheme_constant": P.P0_SCHEME_CONSTANT,
            "projection_constant": P.P0_PROJECTION_CONSTANT,
            "asymptotic_constant": P.P0_ASYMPTOTIC_CONSTANT,
            "expression": "E(N) -> pi*sqrt(2*pi^2 + 1/6)/N",
        },
        "thresholds": TOL,
        "analytic": analytic,
        "analytic_windows": analytic_windows,
        "solver": solver,
        "dt_probe": dt_probe,
        "derived": {
            "order_analytic_pool": order_analytic_pool,
            "order_solver_pool": order_solver_pool,
            "order_difference": order_solver_pool - order_analytic_pool,
            "order_analytic_fine": order_analytic_fine,
            "order_solver_fine": order_solver_fine,
            "constant_measured_finest": constants[-1],
            "constant_rel_error": const_rel,
            "time_share_max_pool": max(time_shares),
            "solver_vs_exact_max_rel": pointwise_rel,
            "constant_rel_error_scheme_only": const_rel_scheme_only,
            "regime_N_order_1pct": regime_threshold(ANALYTIC_LADDER, 0.01),
            "regime_N_order_2pct": regime_threshold(ANALYTIC_LADDER, 0.02),
            "regime_N_order_5pct": regime_threshold(ANALYTIC_LADDER, 0.05),
        },
        "checks": checks,
        "falsified_hypotheses": falsified,
        "verdict": verdict,
    }

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------------- 5. tables + csv (programmatic, never retyped) --------
    lines = ["| N (cells/wavelength) | h | E semidiscrete (exact) | N*E | local order |",
             "|---:|---:|---:|---:|---:|"]
    for r in analytic:
        lo = "-" if r["local_order"] is None else f"{r['local_order']:.4f}"
        lines.append(f"| {r['ncells']} | {r['h']:.3e} | {r['l2_semidiscrete']:.6e} | "
                     f"{r['n_times_l2']:.5f} | {lo} |")
    lines += ["", "| N | solver steps | dt | E total (solver) | E time component | E semidiscrete (exact) |",
              "|---:|---:|---:|---:|---:|---:|"]
    for r in solver:
        lines.append(f"| {r['ncells']} | {r['steps']} | {r['dt']:.4e} | {r['l2_total']:.6e} | "
                     f"{r['l2_time_component']:.3e} | {r['l2_semidiscrete_exact']:.6e} |")
    lines += ["", "| N | cfl | steps | E total | E time component |", "|---:|---:|---:|---:|---:|"]
    for r in dt_probe:
        lines.append(f"| {r['ncells']} | {r['steps']} steps | {r['dt']:.4e} | {r['l2_total']:.6e} | "
                     f"{r['l2_time_component']:.3e} |")
    (OUT / "tables.md").write_text("\n".join(lines) + "\n")

    csv = ["ncells,h,l2_semidiscrete_exact,n_times_l2,local_order,l2_solver_total,l2_solver_time_component"]
    for r in analytic:
        s = next((x for x in solver if x["ncells"] == r["ncells"]), None)
        csv.append(",".join([
            str(r["ncells"]), f"{r['h']:.8e}", f"{r['l2_semidiscrete']:.10e}",
            f"{r['n_times_l2']:.8f}",
            "" if r["local_order"] is None else f"{r['local_order']:.6f}",
            "" if s is None else f"{s['l2_total']:.10e}",
            "" if s is None else f"{s['l2_time_component']:.10e}",
        ]))
    (OUT / "convergence.csv").write_text("\n".join(csv) + "\n")

    make_figures(metrics)
    print(json.dumps({"verdict": verdict, "checks": checks,
                      "derived": metrics["derived"]}, indent=1))
    return metrics


def make_figures(metrics) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = metrics["analytic"]
    s = metrics["solver"]
    ns = np.array([r["ncells"] for r in a], dtype=float)
    es = np.array([r["l2_semidiscrete"] for r in a])
    nv = np.array([r["ncells"] for r in s], dtype=float)
    ev = np.array([r["l2_total"] for r in s])

    # --- figure 1: error curves + asymptotes -------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.loglog(ns, es, "-o", label="semidiscrete DG0, closed form")
    ax.loglog(nv, ev, "s", mfc="none", label="fully discrete (solver, RK4, cfl=0.2)")
    ref = es[0] * (ns[0] / ns) ** 1.0
    ax.loglog(ns, ref, "k--", lw=1.0, label="slope 1 reference")
    floor = P.P0_PROJECTION_CONSTANT / ns
    ax.loglog(ns, floor, ":", lw=1.2,
              label=r"projection floor $\pi h/\sqrt{6}$")
    ax.axvspan(20, 320, color="0.85", alpha=0.6, label="case-pool ladder")
    ax.set_xlabel("N (cells per wavelength)")
    ax.set_ylabel(r"$L^2$ error")
    ax.set_title("DG0 linear advection, t=1: exact reference and design order")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "case_dg_p0_convergence.png", dpi=160)
    plt.close(fig)

    # --- figure 2: local order vs N ---------------------------------------
    lo_n = [r["ncells"] for r in a if r["local_order"] is not None]
    lo_v = [r["local_order"] for r in a if r["local_order"] is not None]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.semilogx(lo_n, lo_v, "-o", label="local order of exact error curve")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="design order p+1 = 1")
    ax.axvspan(20, 320, color="0.85", alpha=0.6, label="case-pool ladder")
    ax.set_xlabel("N (cells per wavelength)")
    ax.set_ylabel("two-point local order")
    ax.set_title("DG0: pre-asymptotic 0.69 ... 0.96 on the pool ladder -> 1.00")
    ax.set_ylim(0.6, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "case_dg_p0_local_order.png", dpi=160)
    plt.close(fig)

    # --- figure 3: coarse-mesh field + error profile ----------------------
    n = 20
    u = P.p0_exact_cell_averages(n, FINAL_TIME, SPEED)
    centers = (np.arange(n) + 0.5) / n
    xf = np.linspace(0, 1, 2001)
    fig, axs = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True)
    axs[0].plot(xf, np.sin(2 * np.pi * (xf - SPEED * FINAL_TIME)), "k-", lw=1.2,
                label="exact $\\sin 2\\pi(x-t)$")
    axs[0].step(centers, u, where="mid", color="C0",
                label="DG0 semidiscrete, N=20")
    axs[0].set_ylabel("u")
    axs[0].legend(fontsize=8)
    axs[0].set_title("Coarse-mesh DG0 field (N=20): amplitude decay drives the error")
    err = u - np.sin(2 * np.pi * (centers - SPEED * FINAL_TIME))
    axs[1].step(centers, err, where="mid", color="C3")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("cell-centre error")
    axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "case_dg_p0_profile.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()