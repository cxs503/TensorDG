"""Root-cause probe for the five red tests in tests/test_euler_verification_chain.py.

Everything printed is generated in this run (no hand-copied numbers).

Probes
------
A. P3 isentropic vortex: EOC at the historic coarse ladder (4,8,16) vs a finer
   ladder (8,16,32), and RK3 vs RK4 (temporal-order hypothesis test).
B. Sod P2: cell-average error vs cell count (refinement study).
C. Lax P2 100 cells: pointwise-exact reference (historic) vs exact
   cell-average reference (Gauss quadrature of the exact Riemann solution).
D. 1-D periodic Euler smooth perturbation: historic reference (t=0 initial
   condition) vs fine-grid self-convergence reference.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from benchmarks.riemann_exact import sample as riemann_sample
from benchmarks.sod_exact import cell_averages as sod_cell_averages
from tensordg.euler_1d import GAMMA
from tensordg.euler_2d import vortex_error
from tensordg.euler_dg_1d import solve_euler_dg_1d, solve_periodic_euler_dg_1d
from tensordg.euler_dg_2d import cell_fields, solve_euler_dg_2d_vortex

LEFT_SOD = (1.0, 0.0, 1.0)
RIGHT_SOD = (0.125, 0.0, 0.1)
LEFT_LAX = (0.445, 0.698, 3.528)
RIGHT_LAX = (0.5, 0.0, 0.571)


def _eoc(res, errors):
    h = [1.0 / n for n in res]
    return [float(np.log(errors[i] / errors[i + 1]) / np.log(h[i] / h[i + 1]))
            for i in range(len(errors) - 1)]


# --- A: vortex P3 ----------------------------------------------------------
def probe_vortex(integrator, res, degree=3, cfl=0.15):
    errors = []
    for n in res:
        x, y, coeff, meta = solve_euler_dg_2d_vortex(
            ncells=n, degree=degree, final_time=0.05, cfl=cfl,
            strength=1.0, limiter=None, initial_projection="l2",
            time_integrator=integrator,
        )
        fields = cell_fields(coeff)
        errors.append(vortex_error(x, y, fields["conservative"], 0.05, 1.0)["density"]["relative_l2"])
    pair = _eoc(res, errors)
    return {"integrator": integrator, "res": list(res), "errors": errors,
            "pairwise_eoc": pair, "mean_eoc": float(np.mean(pair))}


# --- B: Sod refinement -----------------------------------------------------
def probe_sod(resolutions=(50, 100, 200, 400)):
    rows = []
    for n in resolutions:
        x, coeff, meta = solve_euler_dg_1d(
            LEFT_SOD, RIGHT_SOD, ncells=n, degree=2,
            final_time=0.2, cfl=0.15, limiter="tvb", limiter_stage="final",
        )
        avg = coeff[:, 0]
        rho = avg[:, 0].clamp_min(1e-12)
        p = (GAMMA - 1) * (avg[:, 2] - 0.5 * avg[:, 1] ** 2 / rho)
        num = {"density": avg[:, 0].cpu().numpy(),
               "velocity": (avg[:, 1] / rho).cpu().numpy(),
               "pressure": p.cpu().numpy()}
        _, ref = sod_cell_averages(n, 0.2)
        rel = {}
        for key in num:
            r = np.array(ref[key])
            rel[key] = float(np.sqrt(np.mean((num[key] - r) ** 2)) / np.sqrt(np.mean(r ** 2)))
        rows.append({"ncells": n, "rel_error": rel, "steps": meta["steps"]})
    order = {}
    for key in rows[0]["rel_error"]:
        errs = [r["rel_error"][key] for r in rows]
        order[key] = _eoc(resolutions, errs)
    return {"sod_refinement": rows, "pairwise_order": order}


# --- C: Lax pointwise vs cell-average reference ----------------------------
def exact_cell_averages(left, right, final_time, ncells, gauss=4):
    """Cell averages of the exact Riemann solution via Gauss quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(gauss)
    edges = np.linspace(0.0, 1.0, ncells + 1)
    a = edges[:-1, None]
    b = edges[1:, None]
    xi = 0.5 * (a + b) + 0.5 * (b - a) * nodes[None, :]        # (nc, g)
    out = {}
    for key in ("density", "velocity", "pressure"):
        vals = riemann_sample(xi.ravel(), final_time, left, right)[key].reshape(xi.shape)
        out[key] = (vals * weights[None, :]).sum(axis=1) * 0.5
    return out


def probe_lax(ncells=100, final_time=0.13):
    x, coeff, meta = solve_euler_dg_1d(
        LEFT_LAX, RIGHT_LAX, ncells=ncells, degree=2,
        final_time=final_time, cfl=0.15, limiter="tvb", limiter_stage="final",
    )
    avg = coeff[:, 0]
    rho = avg[:, 0].clamp_min(1e-12)
    p = (GAMMA - 1) * (avg[:, 2] - 0.5 * avg[:, 1] ** 2 / rho)
    num = {"density": avg[:, 0].cpu().numpy(),
           "velocity": (avg[:, 1] / rho).cpu().numpy(),
           "pressure": p.cpu().numpy()}
    x_np = x.cpu().numpy()
    pointwise = riemann_sample(x_np, final_time, LEFT_LAX, RIGHT_LAX)
    cellavg = exact_cell_averages(LEFT_LAX, RIGHT_LAX, final_time, ncells)
    res = {}
    for key in num:
        for tag, ref in (("pointwise", np.asarray(pointwise[key])), ("cellavg", cellavg[key])):
            rel = float(np.sqrt(np.mean((num[key] - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))
            res[f"{key}_{tag}_rel"] = rel
    return {"lax_p2_" + str(ncells): res}


# --- D: 1-D periodic smooth perturbation, self-convergence ----------------
def _initial(x):
    return (1.0 + 0.1 * torch.sin(2 * np.pi * x),
            torch.zeros_like(x),
            1.0 + 0.1 * torch.cos(2 * np.pi * x))


def probe_uniform(degree, res=(16, 32, 64), nref=512):
    """Error of rho at t=0.2 vs (i) the initial state, (ii) a fine-grid solution."""
    errors_initial, errors_self = [], []
    # fine reference (same scheme, same degree, 512 cells)
    _, cref, _ = solve_periodic_euler_dg_1d(
        _initial, ncells=nref, degree=degree, final_time=0.2, cfl=0.2,
        limiter=None, boundary="periodic")
    rho_ref = cref[:, 0, 0].cpu().numpy()
    for n in res:
        x, coeff, meta = solve_periodic_euler_dg_1d(
            _initial, ncells=n, degree=degree, final_time=0.2, cfl=0.2,
            limiter=None, boundary="periodic")
        rho = coeff[:, 0, 0].cpu().numpy()
        x_np = x.cpu().numpy()
        init = 1.0 + 0.1 * np.sin(2 * np.pi * x_np)
        errors_initial.append(float(np.sqrt(np.mean((rho - init) ** 2)) / np.sqrt(np.mean(init ** 2))))
        # nested coarse cells: average the (nref/n) fine cell averages
        f = nref // n
        coarse_ref = rho_ref.reshape(n, f).mean(axis=1)
        errors_self.append(float(np.sqrt(np.mean((rho - coarse_ref) ** 2)) / np.sqrt(np.mean(coarse_ref ** 2))))
    return {"degree": degree, "res": list(res), "nref": nref,
            "vs_initial_state": errors_initial, "eoc_vs_initial": _eoc(list(res), errors_initial),
            "vs_fine_grid": errors_self, "eoc_vs_fine_grid": _eoc(list(res), errors_self)}


if __name__ == "__main__":
    out = {
        "vortex_p3_rk3_coarse": probe_vortex("rk3", [4, 8, 16]),
        "vortex_p3_rk4_coarse": probe_vortex("rk4", [4, 8, 16]),
        "vortex_p3_rk3_fine": probe_vortex("rk3", [8, 16, 32]),
        "vortex_p3_rk4_fine": probe_vortex("rk4", [8, 16, 32]),
        "sod": probe_sod(),
        "lax": probe_lax(),
        "uniform_p1": probe_uniform(1),
        "uniform_p2": probe_uniform(2),
    }
    print(json.dumps(out, indent=2))