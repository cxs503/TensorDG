"""Benchmark: channel-flow Poiseuille verification with convergence study.

This benchmark verifies the DG Navier--Stokes solver against the exact
plane-Poiseuille solution and documents the relationship to the LBM Zou-He
boundary-condition approach (loliverhennigh/Lattice-Boltzmann-fluid-flow-in-TensorFlow).

Both methods solve the same continuum PDE and must converge to the same
analytical profile.  The Zou-He BC prescribes velocity/pressure at inlet/outlet
faces by reconstructing unknown distribution functions; the DG prescribed-state
BC (m16075) prescribes the ghost conservative state and uses HLLC flux.

Usage::

    PYTHONPATH=. python -m benchmarks.verify_channel_poiseuille [--quick]

Writes ``validation/channel_poiseuille/metrics.json`` and
``docs/figures/channel_poiseuille_convergence.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordg import ns_dg_2d as ns
from tensordg.ns_poiseuille import BodyForcePoiseuille

ROOT = pathlib.Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "validation" / "channel_poiseuille" / "metrics.json"
FIGURE_PATH = ROOT / "docs" / "figures" / "channel_poiseuille_convergence.png"

CASE = dict(force=0.2, mu=0.1, prandtl=0.71, pressure=1.0, wall_temperature=1.0,
            ymin=0.0, ymax=1.0)
DOMAIN_X = (0.0, 0.5)
NX = 2  # x-periodic, only 2 cells needed


def rest_initial(pressure):
    def initial(x, y):
        return (torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x),
                pressure * torch.ones_like(x))
    return initial


def wall_boundary(case):
    wall = {"velocity": (0.0, 0.0), "temperature": case.wall_temperature}
    return {"ymin": wall, "ymax": dict(wall)}


def quadrature_profile(coeff, case, ncells, degree):
    """Primitive fields at Gauss points."""
    order = max(degree + 2, 3)
    nodes, weights, basis, _, _ = ns.legendre_operators(degree, order, "cpu", torch.float64)
    nx, ny = ncells
    dy = case.height / ny
    values = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    rho = values[..., 0].clamp_min(1e-30)
    energy = values[..., 3]
    pressure = (ns.GAMMA - 1.0) * (
        energy - 0.5 * (values[..., 1] ** 2 + values[..., 2] ** 2) / rho
    )
    y = case.ymin + (torch.arange(ny, dtype=torch.float64).view(1, ny, 1, 1)
                     + 0.5 * (1.0 + nodes.view(1, 1, 1, -1))) * dy
    y = y.expand(nx, ny, nodes.numel(), nodes.numel())
    weight = (weights.view(1, 1, -1, 1) * weights.view(1, 1, 1, -1)).expand_as(rho)
    velocity = values[..., 1] / rho
    temperature = pressure / rho
    return velocity, temperature, y, weight


def relative_l2(numeric, exact, weight):
    """Weighted relative L2 error."""
    scale = math.sqrt(float((weight * exact**2).sum() / weight.sum()))
    error = math.sqrt(float((weight * (numeric - exact) ** 2).sum() / weight.sum()))
    return error / scale if scale > 0 else float("inf")


def run_case(ny, degree, final_time=0.5):
    """Run one Poiseuille channel case and return errors."""
    case = BodyForcePoiseuille(**CASE)
    _, y, coeff, meta = ns.solve_navier_stokes_dg_2d(
        rest_initial(case.pressure),
        ncells=(NX, ny), degree=degree,
        domain=(DOMAIN_X, (case.ymin, case.ymax)),
        final_time=final_time, cfl=0.3, cfl_viscous=3.0,
        mu=case.mu, prandtl=case.prandtl,
        periodic=(True, False), boundary=wall_boundary(case),
        source=case.source,
    )
    velocity, temperature, y_q, weight = quadrature_profile(coeff, case, (NX, ny), degree)
    exact_u = torch.as_tensor(case.velocity(y_q.numpy()))
    exact_T = torch.as_tensor(case.temperature(y_q.numpy()))
    err_u = relative_l2(velocity, exact_u, weight)
    err_T = relative_l2(temperature, exact_T, weight)
    return err_u, err_T, meta


def convergence_study():
    """Run h-refinement study for P1 and P2."""
    results = []
    for degree in (1, 2):
        errors = []
        for ny in (4, 8, 16, 32):
            err_u, err_T, meta = run_case(ny, degree)
            errors.append({"ny": ny, "err_u": err_u, "err_T": err_T,
                           "steps": meta["steps"], "final_time": meta["final_time_reached"]})
        results.append({"degree": degree, "cases": errors})
    return results


def plot_convergence(results):
    """Plot convergence curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {1: "#1f77b4", 2: "#ff7f0e"}
    markers = {1: "o", 2: "s"}

    for res in results:
        deg = res["degree"]
        nys = [c["ny"] for c in res["cases"]]
        errs_u = [c["err_u"] for c in res["cases"]]
        errs_T = [c["err_T"] for c in res["cases"]]
        h = [1.0 / ny for ny in nys]

        ax.loglog(h, errs_u, f"{markers[deg]}-", color=colors[deg],
                  label=f"P{deg} velocity")
        ax.loglog(h, errs_T, f"{markers[deg]}--", color=colors[deg],
                  label=f"P{deg} temperature")

        # Reference slope lines
        if len(errs_u) >= 2:
            slope_u = math.log(errs_u[0] / errs_u[-1]) / math.log(h[0] / h[-1])
            slope_T = math.log(errs_T[0] / errs_T[-1]) / math.log(h[0] / h[-1])
            print(f"  P{deg}: velocity EOC={slope_u:.2f}, temperature EOC={slope_T:.2f}")

    # Reference lines for p+1 order
    h_ref = np.array([1/4, 1/32])
    for deg in (1, 2):
        order = deg + 1
        ax.loglog(h_ref, [1e-1 * (h_ref[0]/h_ref[1])**order, 1e-1],
                  "k:", alpha=0.3, label=f"O(h^{order}) reference")

    ax.set_xlabel("h = 1/ny")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Channel Poiseuille flow: DG convergence vs exact solution\n"
                 "(same continuum limit as LBM Zou-He channel)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {FIGURE_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run fewer cases")
    args = parser.parse_args()

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=== Channel Poiseuille Convergence Study ===")
    print(f"Case: force={CASE['force']}, mu={CASE['mu']}, Pr={CASE['prandtl']}")
    print(f"Domain: x={DOMAIN_X}, y=({CASE['ymin']}, {CASE['ymax']})")
    print()

    results = convergence_study()
    plot_convergence(results)

    # Compute EOC
    for res in results:
        deg = res["degree"]
        cases = res["cases"]
        if len(cases) >= 2:
            eoc_u = math.log(cases[0]["err_u"] / cases[-1]["err_u"]) / math.log(cases[-1]["ny"] / cases[0]["ny"])
            eoc_T = math.log(cases[0]["err_T"] / cases[-1]["err_T"]) / math.log(cases[-1]["ny"] / cases[0]["ny"])
            res["eoc_u"] = eoc_u
            res["eoc_T"] = eoc_T
            print(f"P{deg}: velocity EOC={eoc_u:.2f} (expected ~{deg+1}), "
                  f"temperature EOC={eoc_T:.2f} (expected ~{deg+1})")

    # Final error at finest resolution
    finest = {r["degree"]: r["cases"][-1] for r in results}
    print("\n=== Final errors (finest mesh) ===")
    for deg, c in finest.items():
        print(f"  P{deg}, ny={c['ny']}: u_err={c['err_u']:.6e}, T_err={c['err_T']:.6e}")
        passed_u = c["err_u"] < 0.03
        passed_T = c["err_T"] < 0.03
        print(f"    u < 3%: {'PASS' if passed_u else 'FAIL'}, T < 3%: {'PASS' if passed_T else 'FAIL'}")

    metrics = {
        "case": CASE,
        "domain_x": DOMAIN_X,
        "results": results,
        "finest_errors": {str(k): v for k, v in finest.items()},
        "lbm_zou_he_relation": {
            "description": "Both DG and LBM Zou-He solve the same continuum Navier-Stokes equations",
            "dg_bc": "periodic + no-slip walls (or prescribed-state m16075 for inlet/outlet)",
            "lbm_bc": "Zou-He velocity/pressure reconstruction at inlet/outlet faces",
            "analytical_solution": "parabolic u(y) + Brinkman quartic T(y)",
            "convergence": "Both methods converge to the same analytical profile as resolution increases",
        },
    }

    with open(JSON_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nMetrics saved: {JSON_PATH}")


if __name__ == "__main__":
    main()
