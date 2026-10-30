"""Master convergence table runner — generates all tables and figures."""
from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensordg.converge import ConvergenceTable, compute_apparent_order, gci_three_grid
from tensordg.poisson_2d import solve_poisson_2d, poisson_error_2d
from tensordg.advection_1d import solve_advection, l2_error


def make_poisson_solver(degree, backend="dense"):
    def solver(ncells, deg):
        x, y, coeff, meta = solve_poisson_2d(ncells=ncells, degree=deg, backend=backend)
        err = poisson_error_2d(x, y, coeff)
        err["dofs"] = meta["dofs"]
        return err
    return solver


def make_advection_solver(degree, limiter=None):
    def solver(ncells, deg):
        centers, coeff, meta = solve_advection(
            ncells=ncells, degree=deg, final_time=1.0, speed=1.0, cfl=0.08,
        )
        err_l2 = l2_error(centers, coeff, final_time=1.0, speed=1.0)
        # L∞
        h = 1.0 / ncells
        nq = max(2 * degree + 4, 8)
        x_np, w_np = np.polynomial.legendre.leggauss(nq)
        xq = np.asarray(x_np, dtype=np.float64)
        basis = []
        for k in range(degree + 1):
            c = np.zeros(k + 1); c[-1] = 1.0
            basis.append(np.polynomial.legendre.legval(x_np, c))
        phi = np.stack(basis, axis=1)
        centers_np = centers.cpu().numpy()
        coeff_np = coeff.cpu().numpy()
        uh = coeff_np @ phi.T  # [ncells, nq]
        x_phys = centers_np[:, None] + 0.5 * h * xq[None, :]
        exact = np.sin(2.0 * math.pi * (x_phys - 1.0))
        err_linf = float(np.max(np.abs(uh - exact)))
        return {"absolute_l2": err_l2, "absolute_linf": err_linf, "dofs": ncells * (degree + 1)}
    return solver


def run_all():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    results = {}

    # ---- SIPG Poisson ----
    for deg, backend, base in [(1, "dense", 4), (2, "sparse", 4), (3, "sparse", 8)]:
        tbl = ConvergenceTable(
            problem=f"poisson_sipg_p{deg}", degree=deg,
            solver=make_poisson_solver(deg, backend),
            base_ncells=base, n_levels=5, theoretical_order=deg + 1,
        )
        tbl.run()
        print(tbl.print_table())
        tbl.assert_all_orders(tolerance=0.3)
        results[f"poisson_p{deg}"] = tbl.to_dict()
        with open(f"artifacts/converge_poisson_p{deg}.json", "w") as f:
            json.dump(tbl.to_dict(), f, indent=2)

    # ---- Advection (no limiter) ----
    for deg in [1, 2, 3]:
        tbl = ConvergenceTable(
            problem=f"advection_none_p{deg}", degree=deg,
            solver=make_advection_solver(deg, limiter=None),
            base_ncells=10, n_levels=5, theoretical_order=deg + 1,
        )
        tbl.run()
        print(tbl.print_table())
        tbl.assert_all_orders(tolerance=0.3)
        results[f"advection_none_p{deg}"] = tbl.to_dict()
        with open(f"artifacts/converge_advection_none_p{deg}.json", "w") as f:
            json.dump(tbl.to_dict(), f, indent=2)

    # ---- Advection (TVB limiter) ----
    for deg in [1, 2]:
        tbl = ConvergenceTable(
            problem=f"advection_tvb_p{deg}", degree=deg,
            solver=make_advection_solver(deg, limiter="tvb"),
            base_ncells=10, n_levels=5, theoretical_order=deg + 1,
        )
        tbl.run()
        print(tbl.print_table())
        tbl.assert_all_orders(tolerance=0.3)
        results[f"advection_tvb_p{deg}"] = tbl.to_dict()
        with open(f"artifacts/converge_advection_tvb_p{deg}.json", "w") as f:
            json.dump(tbl.to_dict(), f, indent=2)

    # ---- Advection (WENO limiter) ----
    for deg in [1, 2]:
        tbl = ConvergenceTable(
            problem=f"advection_weno_p{deg}", degree=deg,
            solver=make_advection_solver(deg, limiter="weno"),
            base_ncells=10, n_levels=5, theoretical_order=deg + 1,
        )
        tbl.run()
        print(tbl.print_table())
        tbl.assert_all_orders(tolerance=0.3)
        results[f"advection_weno_p{deg}"] = tbl.to_dict()
        with open(f"artifacts/converge_advection_weno_p{deg}.json", "w") as f:
            json.dump(tbl.to_dict(), f, indent=2)

    # ---- GCI analysis for Poisson p2 ----
    tbl_p2 = results["poisson_p2"]
    entries = tbl_p2["entries"]
    f1, f2, f3 = entries[-1]["error_l2"], entries[-2]["error_l2"], entries[-3]["error_l2"]
    gci_result = gci_three_grid(f1, f2, f3, r=2.0)
    results["gci_poisson_p2"] = gci_result
    with open("artifacts/gci_poisson_p2.json", "w") as f:
        json.dump(gci_result, f, indent=2)
    print(f"\nGCI Analysis (Poisson p2):")
    print(f"  Apparent order: {gci_result['apparent_order']:.4f}")
    print(f"  GCI_fine (21):  {gci_result['gci_fine_21']:.6f}%")
    print(f"  GCI_fine (32):  {gci_result['gci_fine_32']:.6f}%")
    print(f"  Extrapolated:   {gci_result['extrapolated_value']:.6e}")
    print(f"  Asymptotic:     {gci_result['in_asymptotic_range']}")

    # ---- Generate convergence plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Poisson convergence
    ax = axes[0]
    for deg in [1, 2, 3]:
        data = results[f"poisson_p{deg}"]
        hs = [e["h"] for e in data["entries"]]
        errs = [e["error_l2"] for e in data["entries"]]
        ax.loglog(hs, errs, "o-", label=f"SIPG p={deg} (obs={data['entries'][-1]['order_l2']:.2f})")
        # Reference slope
        h_ref = [hs[0], hs[-1]]
        e_ref = [errs[0], errs[0] * (hs[-1] / hs[0]) ** (deg + 1)]
        ax.loglog(h_ref, e_ref, "--", alpha=0.4, label=f"O(h^{deg+1})")
    ax.set_xlabel("Mesh size h")
    ax.set_ylabel("L2 Error")
    ax.set_title("SIPG Poisson Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Advection convergence
    ax = axes[1]
    for lim in ["none", "tvb", "weno"]:
        data = results[f"advection_{lim}_p2"]
        hs = [e["h"] for e in data["entries"]]
        errs = [e["error_l2"] for e in data["entries"]]
        ax.loglog(hs, errs, "o-", label=f"{lim.upper()} p=2 (obs={data['entries'][-1]['order_l2']:.2f})")
    h_ref = [hs[0], hs[-1]]
    e_ref = [errs[0], errs[0] * (hs[-1] / hs[0]) ** 3]
    ax.loglog(h_ref, e_ref, "--", alpha=0.4, label="O(h³)")
    ax.set_xlabel("Mesh size h")
    ax.set_ylabel("L2 Error")
    ax.set_title("Advection DG Convergence (p=2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("artifacts/convergence_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved convergence_plots.png")

    # Save master results
    with open("artifacts/convergence_master.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ All convergence tables generated and validated.")
    return results


if __name__ == "__main__":
    run_all()
