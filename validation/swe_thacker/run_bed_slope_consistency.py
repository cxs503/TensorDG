"""Bed-slope source term: legacy cell-constant vs exact quadrature.

Produces ``bed_slope_results.json`` plus two figures in this directory:

* ``bed_slope_profiles.png``  -- eta(x) at t = 0.5 s vs the exact Thacker
  solution, legacy vs exact source (the visible phase/tilt error of the legacy
  force).
* ``bed_slope_convergence.png`` -- relative L2 error vs N (legacy flat, exact
  ~2nd order) and lake-at-rest drift vs time on a curved bed (the documented
  trade-off of the exact source).

Run:  PYTHONPATH=. python validation/swe_thacker/run_bed_slope_consistency.py
"""
from __future__ import annotations

import json
import math
import os
import time as _time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordg import swe_exact  # noqa: E402
from tensordg.swe_1d import reconstruct_fields, solve_swe_1d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
G, H0, AMP, L = 9.81, 1.0, 0.5, 1.0


def _thacker(ncells, degree, t_end, exact_source, exact_bc=True):
    def init_h(x):
        return torch.as_tensor(
            swe_exact.thacker_planar_basin_initial(x.numpy(), H0, AMP, L)[0], dtype=x.dtype)

    def zb(x):
        return torch.as_tensor(swe_exact.thacker_parabolic_bed(x.numpy(), H0, L), dtype=x.dtype)

    def zb_slope(x):
        return torch.as_tensor(
            swe_exact.thacker_parabolic_bed_slope(x.numpy(), H0, L), dtype=x.dtype)

    kw = {}
    if exact_source:
        kw["zb_slope"] = zb_slope
    if exact_bc:
        kw["bc_left_flux"] = lambda t: torch.as_tensor(
            swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)[0])
        kw["bc_right_flux"] = lambda t: torch.as_tensor(
            swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)[1])

    x, coeff, zb_faces, meta = solve_swe_1d(
        init_h, lambda z: torch.zeros_like(z), zb, ncells=ncells, degree=degree,
        final_time=t_end, domain=(-L, L), cfl=0.3, flux_kind="hll", limiter=None,
        rk_order=3, dtype=torch.float64, **kw)
    h, u, eta = reconstruct_fields(coeff, zb_faces)
    h_e, u_e, eta_e = swe_exact.thacker_planar_basin(x.numpy(), t_end, H0, AMP, L, G)

    def rel(a, b):
        b = np.asarray(b, dtype=float)
        return float(np.sqrt(np.mean((a.numpy() - b) ** 2)) / np.abs(b).max() * 100.0)

    return {
        "x": x.numpy(), "eta": eta.numpy(), "u": u.numpy(),
        "eta_exact": np.asarray(eta_e), "u_exact": np.asarray(u_e),
        "eta_rel_L2_pct": rel(eta, eta_e), "u_rel_L2_pct": rel(u, u_e),
        "steps": meta["steps"], "time_reached": meta["time_reached"],
    }


def _lake_at_rest(degree, times, ncells=64):
    eta0 = 2.0
    zb = lambda xx: 0.4 * torch.sin(math.pi * xx / 2.0)
    out = []
    for t_end in times:
        x, coeff, zb_faces, meta = solve_swe_1d(
            lambda xx: eta0 - zb(xx), lambda xx: torch.zeros_like(xx), zb,
            ncells=ncells, degree=degree, final_time=t_end,
            domain=(-2.0, 2.0), cfl=0.3)
        h, u, eta = reconstruct_fields(coeff, zb_faces)
        out.append({"t": t_end, "drift_max_m": float((eta - eta0).abs().max()),
                    "max_hu": float(coeff[:, :, 1].abs().max())})
    return out


def main():
    t0 = _time.time()
    runs, errs = [], []
    for degree in (1, 2):
        for ncells in (20, 40):
            for exact_source in (False, True):
                for t_end in (0.1, 0.5):
                    tag = {"degree": degree, "ncells": ncells,
                           "source": "exact" if exact_source else "legacy", "t": t_end}
                    try:
                        r = _thacker(ncells, degree, t_end, exact_source)
                        tag.update({k: r[k] for k in
                                    ("eta_rel_L2_pct", "u_rel_L2_pct", "steps", "time_reached")})
                        runs.append(tag)
                        if abs(t_end - 0.5) < 1e-12:
                            errs.append((degree, ncells, exact_source, r))
                        print(f"{tag}", flush=True)
                    except Exception as exc:  # honest FAIL is recorded, not hidden
                        tag["error"] = f"{type(exc).__name__}: {exc}"
                        runs.append(tag)
                        print(f"{tag}  <-- FAIL", flush=True)

    rest = {f"P{deg}": _lake_at_rest(deg, [0.1, 0.25, 0.5, 1.0]) for deg in (1, 2)}

    payload = {
        "generated_by": "validation/swe_thacker/run_bed_slope_consistency.py",
        "reference": "Thacker (1981) planar sloshing, exact via tensordg.swe_exact",
        "config": {"gravity": G, "h0": H0, "amplitude": AMP, "half_width": L, "cfl": 0.3,
                   "flux": "hll", "rk_order": 3, "boundary": "analytic flux",
                   "dtype": "float64"},
        "thacker_runs": runs,
        "lake_at_rest_curved_bed": rest,
        "runtime_s": round(_time.time() - t0, 1),
    }
    with open(os.path.join(HERE, "bed_slope_results.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    # ---- figure 1: profiles at t = 0.5 s (P1, N = 20) --------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, label in ((axes[0], "eta", "free surface $\\eta$ [m]"),
                           (axes[1], "u", "velocity $u$ [m/s]")):
        for degree, ncells, exact_source, r in errs:
            if degree != 1 or ncells != 20 or abs(r["x"].size - 20) != 0:
                continue
            ax.plot(r["x"], r[key], "o-", ms=3,
                    label=f"{'exact' if exact_source else 'legacy'} source "
                          f"({r[key + '_rel_L2_pct']:.3f} %)")
        sel = [e for e in errs if e[0] == 1 and e[1] == 20][0][3]
        ax.plot(sel["x"], sel[key + "_exact"], "k--", lw=1.4, label="Thacker exact")
        ax.set_xlabel("x [m]")
        ax.set_ylabel(label)
        ax.set_title(f"Thacker t = 0.5 s, P1, N = 20")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Bed-slope source: cell-constant vs exact quadrature")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bed_slope_profiles.png"), dpi=140)

    # ---- figure 2: convergence + rest-balance trade-off -------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for exact_source, marker in ((False, "s"), (True, "o")):
        xs, ys = [], []
        for degree, ncells, es, r in errs:
            if es == exact_source and degree == 1:
                xs.append(ncells)
                ys.append(r["eta_rel_L2_pct"])
        order = np.polyfit(np.log(xs), np.log(ys), 1)[0] if len(xs) > 1 else float("nan")
        ax.loglog(xs, ys, marker + "-", label=f"{'exact' if exact_source else 'legacy'} "
                                              f"source (slope {order:.2f})")
    ax.axhline(3.0, color="r", ls=":", label="3 % acceptance")
    ax.set_xlabel("N (cells)")
    ax.set_ylabel("$\\eta$ relative $L_2$ error [%]")
    ax.set_title("Thacker t = 0.5 s, P1: grid convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for deg in (1, 2):
        tt = [p["t"] for p in rest[f"P{deg}"]]
        dd = [p["drift_max_m"] for p in rest[f"P{deg}"]]
        ax.semilogy(tt, dd, "o-", label=f"legacy source, P{deg}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("lake-at-rest drift $\\max|\\eta-\\eta_0|$ [m]")
    ax.set_title("Curved-bed lake at rest (N = 64): documented gap")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bed_slope_convergence.png"), dpi=140)

    print(json.dumps({"runtime_s": payload["runtime_s"], "n_runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()