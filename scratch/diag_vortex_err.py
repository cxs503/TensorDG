"""Per-checkpoint diagnostic: solver error against the *periodic* and the *free-space*
exact solution, side by side.

Why this script exists
----------------------
The published long-time readings of ``CASE-DG-EULER-VORTEX-ERR`` (round 3) were
measured against ``vortex_primitive`` *without* the periodic min-image fold.  Once
the vortex centre leaves the box the free-space formula is no longer the solution
of the periodic problem the solvers actually solve, so the reported "error"
contains a reference term growing like ``t``.  This script separates the two terms
for the same numerical solution:

    e_free(t)   = ||u_h(t) - u_exact_free(t)|| / ||u_exact_free(t)||
    e_per(t)    = ||u_h(t) - u_exact_per(t) || / ||u_exact_per(t) ||
    defect(t)   = ||u_exact_per(t) - u_exact_free(t)|| / ||u_exact_per(t)||

``defect`` is a pure property of the analytic solution (no solver), so
``e_free - e_per ~ defect`` is a *prediction* the numerical run must satisfy -- a
falsifiable check that the re-referencing, and not a changed scheme, is what moved
the numbers.

    PYTHONPATH=. python3 scratch/diag_vortex_err.py                 # default probe set
    PYTHONPATH=. python3 scratch/diag_vortex_err.py --full          # + t = 10 / p0 128
    PYTHONPATH=. python3 scratch/diag_vortex_err.py --figure docs/fig_vortex_refdefect.png
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from benchmarks import case_euler_vortex_errconserve as ec
from benchmarks.verify_euler_dg_2d import _vortex_error as dg_vortex_error
from tensordg import euler_dg_2d as dg2
from tensordg.euler_2d import solve_isentropic_vortex, vortex_error

FIELDS = ("density", "velocity_x", "velocity_y", "pressure")


def _dg_errors(config: dict) -> tuple[dict, dict]:
    final_time = config["final_time"]
    _x, _y, coeff, meta = dg2.solve_euler_dg_2d_vortex(
        ncells=config["cells"], degree=config["degree"],
        final_time=final_time, cfl=config.get("cfl", 0.2),
        strength=ec.STRENGTH, background_velocity=ec.BACKGROUND, domain=ec.DOMAIN)
    nodes, _weights, basis, _, _ = ec._operators(config["degree"])
    periodic = dg_vortex_error(coeff, basis, nodes, ec.DOMAIN, final_time,
                               ec.STRENGTH, ec.BACKGROUND, period=ec.PERIOD)
    free = dg_vortex_error(coeff, basis, nodes, ec.DOMAIN, final_time,
                           ec.STRENGTH, ec.BACKGROUND, period=None)
    return ({k: float(periodic[k]) for k in FIELDS},
            {k: float(free[k]) for k in FIELDS}, meta)


def _p0_errors(config: dict) -> tuple[dict, dict]:
    final_time = config["final_time"]
    x, y, fields, meta = solve_isentropic_vortex(
        ncells=config["cells"], final_time=final_time, cfl=config.get("cfl", 0.4),
        domain=ec.DOMAIN, strength=ec.STRENGTH, background_velocity=ec.BACKGROUND)
    state = fields["conservative"]
    periodic = vortex_error(x, y, state, final_time, ec.STRENGTH, ec.BACKGROUND,
                            period=ec.PERIOD)
    free = vortex_error(x, y, state, final_time, ec.STRENGTH, ec.BACKGROUND)
    return ({k: float(periodic[k]["relative_l2"]) for k in FIELDS},
            {k: float(free[k]["relative_l2"]) for k in FIELDS}, meta)


def run(config: dict) -> dict:
    errors = _dg_errors if config["task"] == "dg" else _p0_errors
    periodic, free, meta = errors(config)
    return {
        "label": ec.task_key(config),
        "family": ec.family_key({**config, "degree": config.get("degree", 0)}),
        "time": config["final_time"],
        "steps": meta["steps"],
        "max_periodic": max(periodic.values()),
        "max_free": max(free.values()),
        "fields_periodic": periodic,
        "fields_free": free,
    }


def _exponent(times, values) -> float:
    """Least-squares slope of ``log e`` against ``log t`` (``nan`` if undecidable)."""
    pairs = [(t, v) for t, v in zip(times, values) if v > 0.0]
    if len(pairs) < 3:
        return float("nan")
    n = len(pairs)
    lx = [math.log(t) for t, _ in pairs]
    ly = [math.log(v) for _, v in pairs]
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((a - mx) ** 2 for a in lx)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    return sxy / sxx if sxx > 0.0 else float("nan")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true",
                        help="also probe t = 10 and the p0/128 mesh")
    parser.add_argument("--figure", default="")
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)

    times = (1.0, 2.0, 5.0) + ((10.0,) if args.full else ())
    configs = [{"task": "dg", "degree": 2, "cells": 32, "final_time": t, "cfl": 0.2}
               for t in times]
    configs += [{"task": "p0", "cells": 64, "final_time": t, "cfl": 0.4} for t in times]
    if args.full:
        configs += [{"task": "p0", "cells": 128, "final_time": 10.0, "cfl": 0.4}]

    records = [run(c) for c in configs]
    defect = ec.reference_defect_sweep([r["time"] for r in records[:len(times)]])

    print("family        t   steps  max|e| vs periodic %  max|e| vs free-space %  "
          "defect %  e_free-e_per %")
    for r in records:
        d = max(defect[f"{r['time']:g}"].values())
        print(f"{r['family']:<12} {r['time']:5.1f} {r['steps']:6d} "
              f"{100.0 * r['max_periodic']:20.4e} {100.0 * r['max_free']:22.4e} "
              f"{d:9.4f} {100.0 * (r['max_free'] - r['max_periodic']):13.4e}")

    for family in sorted({r["family"] for r in records}):
        rows = sorted((r for r in records if r["family"] == family),
                      key=lambda r: r["time"])
        alpha_p = _exponent([r["time"] for r in rows], [r["max_periodic"] for r in rows])
        alpha_f = _exponent([r["time"] for r in rows], [r["max_free"] for r in rows])
        print(f"\n{family}: alpha(periodic reference) = {alpha_p:.3f}   "
              f"alpha(free-space reference) = {alpha_f:.3f}   "
              f"({len(rows)} checkpoints; gate MAX_GROWTH_EXPONENT={ec.MAX_GROWTH_EXPONENT})")
        print(f"  e(t_last)/e(t=1): periodic {rows[-1]['max_periodic'] / rows[0]['max_periodic']:.3f}"
              f"   free-space {rows[-1]['max_free'] / rows[0]['max_free']:.3f}")

    if args.json:
        payload = {"records": records, "defect_percent": defect}
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.json}")
    if args.figure:
        make_figure(records, args.figure)
        print(f"wrote {args.figure}")
    return 0


def make_figure(records, path: str, points: int = 240) -> None:
    """Two panels: the reference defect in the field, and its growth with time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from tensordg.euler_2d import vortex_primitive

    axis = torch.linspace(ec.DOMAIN[0][0], ec.DOMAIN[0][1], points,
                          dtype=torch.float64)
    gx, gy = torch.meshgrid(axis, axis, indexing="ij")
    per = vortex_primitive(gx, gy, 5.0, ec.STRENGTH, ec.BACKGROUND, period=ec.PERIOD)
    free = vortex_primitive(gx, gy, 5.0, ec.STRENGTH, ec.BACKGROUND)
    delta = np.abs((per[1] - free[1]).numpy())

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    mesh = axes[0].pcolormesh(axis.numpy(), axis.numpy(), delta, cmap="inferno",
                              shading="auto")
    fig.colorbar(mesh, ax=axes[0], label=r"$|u_{\rm periodic}-u_{\rm free}|$")
    axes[0].set_title("reference defect in the field, $t=5$ (max "
                      f"{delta.max():.2f})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")

    times = np.linspace(0.0, 10.0, 41)
    sweep = ec.reference_defect_sweep(tuple(float(t) for t in times))
    axes[1].plot(times, [sweep[f"{t:g}"]["velocity_x"] for t in times], "-",
                 color="C3", label="reference defect (velocity, % of |u|)")
    axes[1].plot(times, [sweep[f"{t:g}"]["density"] for t in times], "--",
                 color="C0", label="reference defect (density, %)")
    for r in records:
        axes[1].plot([r["time"]], [100.0 * r["max_free"]], "x", color="C3",
                     ms=8, label=None)
        axes[1].plot([r["time"]], [100.0 * r["max_periodic"]], "o", mfc="none",
                     color="C0", ms=8, label=None)
    axes[1].plot([], [], "x", color="C3", label="solver error vs free-space reference")
    axes[1].plot([], [], "o", mfc="none", color="C0", label="solver error vs periodic reference")
    axes[1].axhline(100.0 * ec.TOLERANCE, color="k", lw=0.8, ls=":",
                    label=f"case tolerance {100 * ec.TOLERANCE:.0f} %")
    axes[1].set_xlabel("dimensionless time t")
    axes[1].set_ylabel("relative L2 error  [%]")
    axes[1].set_title("error decomposition: defect of the reference vs scheme error")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7, loc="upper left")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())