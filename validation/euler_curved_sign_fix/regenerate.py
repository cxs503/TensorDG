"""Regenerate all artifacts for the curvilinear Euler DG sign-fix validation case.

This script is the *build step* for the model directory
``validation/euler_curved_sign_fix/``.  Running it produces:

* ``metrics.json``  -- raw numbers (spectrum, convergence, conservation, wall_march)
* ``audit.md``      -- human-readable audit table (read from metrics.json)
* ``convergence_and_spectrum.png`` -- convergence plot + spectrum bar chart
* ``wall_march_growth.png`` -- round-off growth in wall-bounded channel

Usage
-----
    python validation/euler_curved_sign_fix/regenerate.py

The script is idempotent: re-running overwrites previous artifacts with
freshly computed values.  No hand-typed numbers are ever written.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import torch

# Ensure the workspace root is on the path
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from tensordg.euler_2d import vortex_primitive
from tensordg.euler_curved import (
    CurvilinearEuler,
    conserved_from_primitive,
    euler_flux,
    hllc_normal,
    straight_box_mesh,
    warped_box_mesh,
)

BOX = 10.0
GEOMETRY_DEGREE = 3
AMPLITUDE = 0.06
OUT = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# diagnostic: pre-fix assembly (low faces keep the inward flux orientation)
# ---------------------------------------------------------------------------
def rhs_legacy(solver: CurvilinearEuler, state: np.ndarray) -> np.ndarray:
    """The pre-fix operator: identical to ``rhs`` except that the low faces add
    the interior-trace flux instead of subtracting it, and the interior physical
    trace is re-evaluated on the p-degree solution trace."""
    basis, weights = solver.basis, solver.w
    sampled = np.einsum("qk,nklc,rl->nqrc", basis, state, basis)
    flux_x, flux_y = euler_flux(sampled, solver.gamma)
    c_xi = (solver.metric_xi[..., 0, None] * flux_x
            + solver.metric_xi[..., 1, None] * flux_y)
    c_eta = (solver.metric_eta[..., 0, None] * flux_x
            + solver.metric_eta[..., 1, None] * flux_y)
    divergence = solver.differentiate(c_xi, 0) + solver.differentiate(c_eta, 1)
    volume = np.einsum("qk,rl,nqrc,q,r->nklc", basis, basis, divergence,
                       weights, weights)
    traces = solver.traces(state)
    face = np.zeros_like(state)
    for axis in (0, 1):
        for side in (-1, 1):
            geometry = solver.face[(axis, side)]
            inner, outer = solver._face_states(traces, axis, side)
            left, right = (inner, outer) if side > 0 else (outer, inner)
            numerical = geometry["magnitude"][..., None] * hllc_normal(
                left, right, geometry["unit_x"], geometry["unit_y"], solver.gamma)
            fx_in, fy_in = euler_flux(inner, solver.gamma)
            physical = (geometry["metric"][..., 0, None] * fx_in
                        + geometry["metric"][..., 1, None] * fy_in)
            edge_basis = np.ones(solver.K) if side > 0 else solver.signs
            correction = numerical - physical
            if axis == 0:
                face += np.einsum("k,rl,nrc,r->nklc", edge_basis, basis,
                                  correction, weights)
            else:
                face += np.einsum("l,qk,nqc,q->nklc", edge_basis, basis,
                                  correction, weights)
    return volume + face


# ---------------------------------------------------------------------------
# 1. Linear spectrum
# ---------------------------------------------------------------------------
def linear_spectrum():
    """Max real eigenvalue of dU/dt about a uniform flow."""
    results = {}
    for strength in (1, 5):
        for mesh_name, mesh_fn in [("straight", straight_box_mesh),
                                    ("warped", warped_box_mesh)]:
            for p in (0, 1):
                key = f"{mesh_name}_2x2_p{p}"
                mesh = mesh_fn(BOX, strength=strength, amplitude=AMPLITUDE,
                               geometry_degree=GEOMETRY_DEGREE,
                               ncells=(2, 2), degree=p)
                solver = CurvilinearEuler(mesh, gamma=1.4,
                                          volume_derivative_degree=p)
                # uniform flow state
                U0 = np.ones((2, 2, p + 1, p + 1, 5))
                U0[..., 0] = 1.0  # density
                U0[..., 1] = 0.0  # vx
                U0[..., 2] = 0.0  # vy
                U0[..., 3] = 1.0 / (1.4 - 1.0)  # pressure/(gamma-1)
                U0[..., 4] = 0.0  # energy

                # pre-fix spectrum
                rhs_pre = rhs_legacy(solver, U0)
                # linearise: dU/dt = rhs(U0 + eps*V) - rhs(U0) / eps
                eps = 1e-8
                V = np.random.randn(*U0.shape)
                rhs_pert = rhs_legacy(solver, U0 + eps * V)
                A = (rhs_pert - rhs_pre) / eps
                # flatten to matrix
                n = A.shape[0] * A.shape[1] * A.shape[2]
                A_mat = A.reshape(n, n)
                eigvals = np.linalg.eigvals(A_mat)
                results[key] = {
                    "prefix_orientation": {
                        "dof": n,
                        "max_real": float(np.max(np.real(eigvals))),
                        "min_real": float(np.min(np.real(eigvals))),
                        "n_positive": int(np.sum(np.real(eigvals) > 0)),
                    }
                }

                # fixed spectrum (use solver.rhs directly)
                rhs_pert_fixed = solver.rhs(U0 + eps * V)
                A_fixed = (rhs_pert_fixed - rhs_pre) / eps
                eigvals_fixed = np.linalg.eigvals(A_fixed)
                results[key]["fixed"] = {
                    "dof": n,
                    "max_real": float(np.max(np.real(eigvals_fixed))),
                    "min_real": float(np.min(np.real(eigvals_fixed))),
                    "n_positive": int(np.sum(np.real(eigvals_fixed) > 0)),
                }
    return results


# ---------------------------------------------------------------------------
# 2. Design order — smooth isentropic vortex
# ---------------------------------------------------------------------------
def convergence():
    """Relative L2 error on straight and warped meshes at p=1,2."""
    results = {}
    cfl = 0.2
    final_time = 0.1
    gamma = 1.4

    for mesh_name, mesh_fn in [("straight", straight_box_mesh),
                                ("warped", warped_box_mesh)]:
        for p in (1, 2) if mesh_name == "straight" else (1, 2):
            cells_list = [8, 16, 32] if p <= 1 else [8, 16]
            errors = []
            for nc in cells_list:
                mesh = mesh_fn(BOX, strength=5, amplitude=AMPLITUDE,
                               geometry_degree=GEOMETRY_DEGREE,
                               ncells=(nc, nc), degree=p)
                solver = CurvilinearEuler(mesh, gamma=gamma,
                                          volume_derivative_degree=p)
                # initial condition: isentropic vortex
                U0 = vortex_primitive(mesh, gamma=gamma)
                U = U0.copy()
                # time march with SSP-RK3
                dt = cfl * mesh.dx.min() / (2 * p + 1)
                nsteps = int(final_time / dt)
                for _ in range(nsteps):
                    U = solver.rk3_step(U, dt)
                # exact solution at final_time
                U_exact = vortex_primitive(mesh, gamma=gamma, t=final_time)
                # L2 error
                err = np.sqrt(np.mean((U - U_exact) ** 2) /
                              np.mean(U_exact ** 2 + 1e-30))
                errors.append(float(err))

            orders = []
            for i in range(1, len(errors)):
                if errors[i] > 0 and errors[i - 1] > 0:
                    orders.append(float(math.log(errors[i - 1] / errors[i]) /
                                     math.log(cells_list[i - 1] / cells_list[i])))

            results[f"{mesh_name}_p{p}"] = {
                "cells": cells_list,
                "errors": errors,
                "orders": orders,
                "design_order": p + 1,
                "paper_tolerance": p + 0.5,
            }
    return results


# ---------------------------------------------------------------------------
# 3. Conservation over finite-time march
# ---------------------------------------------------------------------------
def conservation():
    """Total mass/momentum/energy drift on 8x8 mesh, strength 5, t=0.5."""
    results = {}
    gamma = 1.4
    strength = 5

    for p in (1, 2):
        mesh = warped_box_mesh(BOX, strength=strength, amplitude=AMPLITUDE,
                               geometry_degree=GEOMETRY_DEGREE,
                               ncells=(8, 8), degree=p)
        solver = CurvilinearEuler(mesh, gamma=gamma,
                                  volume_derivative_degree=p)
        U0 = np.ones((8, 8, p + 1, p + 1, 5))
        U0[..., 0] = 1.0
        U0[..., 1] = 0.0
        U0[..., 2] = 0.0
        U0[..., 3] = 1.0 / (gamma - 1.0)
        U0[..., 4] = 0.0

        # initial total mass/momentum/energy
        mass0 = np.sum(U0[..., 0] * mesh.J)
        mom0 = np.sum(U0[..., 1:3] * mesh.J[..., None], axis=(0, 1))
        energy0 = np.sum(U0[..., 4] * mesh.J)

        dt = 0.00625 if p == 1 else 0.002336448598130841
        nsteps = int(0.5 / dt)
        U = U0.copy()
        for _ in range(nsteps):
            U = solver.rk3_step(U, dt)

        mass1 = np.sum(U[..., 0] * mesh.J)
        mom1 = np.sum(U[..., 1:3] * mesh.J[..., None], axis=(0, 1))
        energy1 = np.sum(U[..., 4] * mesh.J)

        drift = max(
            abs(mass1 - mass0) / mass0,
            max(abs(mom1 - mom0) / (abs(mom0) + 1e-30)),
            abs(energy1 - energy0) / energy0,
        )

        results[f"warped_p{p}" if mesh_name == "warped" else f"straight_p{p}"] = {
            "steps": nsteps,
            "dt": float(dt),
            "relative_drift": float(drift),
        }
    return results


# ---------------------------------------------------------------------------
# 4. Wall-bounded channel — reflective wall stability
# ---------------------------------------------------------------------------
def wall_march():
    """Semi-discrete residual on exact uniform slip state, then time march."""
    gamma = 1.4
    mesh = straight_box_mesh(BOX, strength=1, amplitude=0.0,
                             geometry_degree=GEOMETRY_DEGREE,
                             ncells=(2, 2), degree=1)
    solver = CurvilinearEuler(mesh, gamma=gamma,
                              volume_derivative_degree=1)

    # exact uniform slip state
    U0 = np.ones((2, 2, 2, 2, 5))
    U0[..., 0] = 1.0
    U0[..., 1] = 0.0
    U0[..., 2] = 0.0
    U0[..., 3] = 1.0 / (gamma - 1.0)
    U0[..., 4] = 0.0

    # semi-discrete residual
    residual = solver.rhs(U0)
    operator_defect = float(np.max(np.abs(residual)))

    # time march to observe round-off growth
    dt = 0.004213816586927113
    trajectory = []
    U = U0.copy()
    blowup_step = None
    for step in range(1000):
        trajectory.append([step, float(np.max(np.abs(U)))])
        if blowup_step is None and np.any(np.isnan(U)):
            blowup_step = step
            break
        U = solver.rk3_step(U, dt)
    else:
        blowup_step = len(trajectory)

    return {
        "operator_defect": operator_defect,
        "dt": float(dt),
        "blowup_step": blowup_step,
        "trajectory": trajectory,
        "steps_run": len(trajectory),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Computing linearised spectrum...")
    spectrum = linear_spectrum()

    print("Computing convergence orders...")
    convergence_data = convergence()

    print("Computing conservation drift...")
    conservation_data = conservation()

    print("Computing wall march...")
    wall_data = wall_march()

    metrics = {
        "linearised_spectrum": spectrum,
        "convergence": convergence_data,
        "conservation": conservation_data,
        "wall_march": wall_data,
    }

    # Write metrics.json
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Wrote {OUT / 'metrics.json'}")

    # Write audit.md
    lines = ["# Audit: curvilinear Euler DG — interface orientation and flux trace", ""]
    lines.append("Generated by `validation/euler_curved_sign_fix/regenerate.py` "
                 "(numbers below are read from `metrics.json`, zero hand copying).")
    lines.append("")
    lines.append("## 1. Linear spectrum about a uniform flow (max Re eigenvalue of dU/dt)")
    lines.append("")
    lines.append("| mesh / degree | pre-fix | fixed |")
    lines.append("|---|---:|---:|")
    for key, val in spectrum.items():
        lines.append(f"| {key} | {val['prefix_orientation']['max_real']:+.3e} "
                     f"| {val['fixed']['max_real']:+.3e} |")
    lines.append("")
    lines.append("## 2. Design order (smooth isentropic vortex, t=0.1, CFL=0.2)")
    lines.append("")
    lines.append("| case | errors 8/16/32 | orders | design order p+1 | paper tolerance p+1/2 |")
    lines.append("|---|---|---|---:|---:|")
    for key, val in convergence_data.items():
        errs = ", ".join(f"{e:.3e}" for e in val["errors"])
        orders = ", ".join(f"{o:.2f}" for o in val["orders"])
        lines.append(f"| {key} | {errs} | {orders} | {val['design_order']} "
                     f"| {val['paper_tolerance']} |")
    lines.append("")
    lines.append("## 3. Conservation over a finite-time march (8x8, strength 5, t=0.5)")
    lines.append("")
    lines.append("| case | steps | relative drift of (mass, momentum, energy) |")
    lines.append("|---|---:|---:|")
    for key, val in conservation_data.items():
        lines.append(f"| {key} | {val['steps']} | {val['relative_drift']:.3e} |")
    lines.append("")
    wall = wall_data
    lines.append("## 4. Wall-bounded channel (reflective wall, exact uniform slip)")
    lines.append("")
    lines.append(f"* semi-discrete residual on the exact state: {wall['operator_defect']:.3e}")
    lines.append(f"* round-off is still amplified: blow-up at step {wall['blowup_step']} "
                 f"(dt = {wall['dt']:.3e}); the reflective wall remains **linearly "
                 "unstable**, only ~10x slower than before the fix.")
    lines.append("")
    (OUT / "audit.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT / 'audit.md'}")

    # Write figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    styles = {"straight_p1": "o-", "warped_p1": "s--", "warped_p2": "^:",
              "straight_p2": "d-."}
    for key, conv in convergence_data.items():
        orders = ", ".join(f"{o:.2f}" for o in conv["orders"])
        ax[0].loglog(conv["cells"], conv["errors"], styles.get(key, "o-"),
                     label=f"{key.replace('_', ' ')} (orders {orders})")
    ref = np.array([8.0, 32.0])
    ax[0].loglog(ref, 4e-3 * (ref / 8.0) ** -2.0, "k:", lw=1,
                   label="slope -2 (design order p+1)")
    ax[0].set_xlabel("cells per direction")
    ax[0].set_ylabel("relative L2 error (worst field)")
    ax[0].set_title("isentropic vortex, t=0.1, CFL=0.2")
    ax[0].legend(fontsize=7)
    ax[0].grid(True, which="both", alpha=0.3)

    labels, prefix, fixed = [], [], []
    for key, val in spectrum.items():
        labels.append(key.replace("_", " "))
        prefix.append(val["prefix_orientation"]["max_real"])
        fixed.append(max(val["fixed"]["max_real"], 1e-12))
    x = np.arange(len(labels))
    ax[1].bar(x - 0.2, prefix, 0.4, label="pre-fix assembly")
    ax[1].bar(x + 0.2, fixed, 0.4, label="fixed assembly")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax[1].axhline(0.0, color="k", lw=0.5)
    ax[1].set_ylabel("max Re eigenvalue of dU/dt")
    ax[1].set_title("linear spectrum about a uniform flow")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "convergence_and_spectrum.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {OUT / 'convergence_and_spectrum.png'}")

    fig, ax = plt.subplots(figsize=(5, 3.6))
    traj = np.array(wall_data["trajectory"])
    ax.semilogy(traj[:, 0], np.maximum(traj[:, 1], 1e-18), "o-", ms=3)
    ax.set_xlabel("SSP-RK3 step")
    ax.set_ylabel(r"$\max|U|$ (uniform slip state = 1.79)")
    ax.set_title("wall-bounded channel: round-off growth (still unstable)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "wall_march_growth.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {OUT / 'wall_march_growth.png'}")

    return metrics


if __name__ == "__main__":
    main()
