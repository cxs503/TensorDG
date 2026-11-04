"""Design-order verification of the 2-D high-order compressible Euler DG solver.

``tensordg/euler_dg_2d.py`` is the workhorse of the 2-D compressible stack, but
the only convergence evidence in the repository was a *single* degree (P2) on
*two* meshes (16^2 -> 24^2, observed EOC 2.95, ``validation/euler_dg_2d/
evidence.json``).  One interval cannot separate the design order ``p + 1`` from
super-convergence, from a pre-asymptotic window, or from an order cap caused by
the initial projection.  This benchmark closes that gap: for ``p = 1, 2, 3`` it
sweeps four meshes each on the smooth isentropic vortex (an *exact* solution of
the 2-D Euler equations advected across a periodic box, so the reference is
analytic -- no manufactured source term and no numerical reference), and reports

* the relative L2 error of ``(rho, u, v, p)`` against the analytic solution,
* the adjacent-mesh observed order (EOC) and the least-squares slope,
* the 3 % engineering gate on the finest mesh of each degree.

Run::

    PYTHONPATH=. python -m benchmarks.design_order_euler_dg_2d
    PYTHONPATH=. python -m benchmarks.design_order_euler_dg_2d --degrees 1 --meshes 8 16 32

Numbers go to ``validation/euler_dg_2d/design_order.json`` (written after every
mesh so an interrupted sweep still leaves usable evidence) and the log-log
figure to ``docs/figures/euler_dg_2d_design_order.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import tensordg.euler_dg_2d as dg2
from benchmarks.verify_euler_dg_2d import _vortex_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "euler_dg_2d" / "design_order.json"
FIG = ROOT / "docs" / "figures" / "euler_dg_2d_design_order.png"

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))   # length 10, vortex strength 5, background (1, 1)
FIELDS = ("density", "velocity_x", "velocity_y", "pressure")

# Four rungs per degree at a ~1.33-1.5 refinement ratio: enough for a
# least-squares slope, cheap enough to stay inside a shared 128-core box.
DEFAULT_MESHES = {1: (12, 16, 24, 32), 2: (8, 12, 16, 24), 3: (6, 8, 12, 16)}


def slope(xs, ys):
    """Least-squares slope of ``log(y)`` against ``log(x)``."""
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


def sweep(degree, meshes, final_time=1.0, cfl=0.2, log=None):
    """Run one degree over the mesh ladder; returns the block + the figure data."""
    nodes, weights, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    rows = []
    for ncells in meshes:
        started = time.time()
        _, _, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
            ncells=ncells, degree=degree, final_time=final_time, cfl=cfl
        )
        errors = _vortex_error(coeff, basis, nodes, DOMAIN, final_time)
        row = {
            "degree": degree,
            "cells": ncells,
            "h": 10.0 / ncells,
            "dofs": metadata["dofs"],
            "steps": metadata["steps"],
            "dt": metadata["dt"],
            "errors": {name: float(errors[name]) for name in FIELDS},
            "wall_seconds": round(time.time() - started, 2),
        }
        rows.append(row)
        if log is not None:
            log(f"  P{degree} {ncells:3d}^2: "
                + "  ".join(f"{k} {row['errors'][k]:.4e}" for k in FIELDS)
                + f"  ({row['wall_seconds']:.1f}s)")

    for previous, current in zip(rows, rows[1:]):
        ratio = previous["h"] / current["h"]
        current["eoc"] = {
            name: float(math.log(previous["errors"][name] / current["errors"][name])
                        / math.log(ratio))
            for name in FIELDS
        }
    hs = [row["h"] for row in rows]
    return {
        "degree": degree,
        "design_order": degree + 1,
        "final_time": final_time,
        "cfl": cfl,
        "rows": rows,
        "fit_slope": {name: slope(hs, [row["errors"][name] for row in rows]) for name in FIELDS},
        "finest_errors": {name: rows[-1]["errors"][name] for name in FIELDS},
        "finest_below_three_percent": {
            name: rows[-1]["errors"][name] < 0.03 for name in FIELDS
        },
        "total_seconds": round(sum(row["wall_seconds"] for row in rows), 1),
    }


def write_json(blocks):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"benchmark": "euler_dg_2d_design_order",
                               "domain": DOMAIN, "blocks": blocks}, indent=1))


def make_figure(blocks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colours = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    for block in blocks:
        p = block["degree"]
        hs = [row["h"] for row in block["rows"]]
        for field, marker, ls in (("density", "o", "-"), ("pressure", "s", "--")):
            axes[0].loglog(hs, [row["errors"][field] for row in block["rows"]],
                           marker + ls, color=colours.get(p, "k"),
                           label=f"P{p} {field}")
        ref = block["rows"][0]["errors"]["density"]
        h0 = hs[0]
        axes[0].loglog(hs, [ref * (h / h0) ** (p + 1) for h in hs], ":",
                       color=colours.get(p, "k"), lw=1,
                       label=f"P{p} slope {p + 1}")
        axes[1].semilogx(hs[1:], [r["eoc"]["density"] for r in block["rows"][1:]],
                         "o-", color=colours.get(p, "k"), label=f"P{p} density EOC")
        axes[1].semilogx(hs[1:], [r["eoc"]["pressure"] for r in block["rows"][1:]],
                         "s--", color=colours.get(p, "k"), label=f"P{p} pressure EOC")
        axes[1].axhline(p + 1, color=colours.get(p, "k"), lw=0.8, ls=":")

    axes[0].set_xlabel("h"); axes[0].set_ylabel("relative $L^2$ error")
    axes[0].set_title("Isentropic vortex: error vs h (dotted = design slope)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xlabel("h"); axes[1].set_ylabel("observed order (EOC)")
    axes[1].set_title("Observed order per mesh interval (dotted = p+1)")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    return FIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--meshes", type=int, nargs="+", default=None,
                        help="override the mesh ladder (applies to every degree)")
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    blocks = []
    for degree in args.degrees:
        meshes = args.meshes or DEFAULT_MESHES[degree]
        print(f"[P{degree}] design order {degree + 1}, meshes {meshes}", flush=True)
        block = sweep(degree, meshes, args.final_time, args.cfl, log=lambda m: print(m, flush=True))
        blocks.append(block)
        write_json(blocks)
        print(f"[P{degree}] fit slope "
              + "  ".join(f"{k} {v:.2f}" for k, v in block["fit_slope"].items()), flush=True)

    if not args.no_figure:
        make_figure(blocks)
        print(f"figure -> {FIG.relative_to(ROOT)}")

    print("\nsummary (observed order vs design order p+1)")
    for block in blocks:
        p = block["degree"]
        print(f"P{p}: design {p + 1} | fit "
              + " ".join(f"{k} {v:.2f}" for k, v in block["fit_slope"].items())
              + " | finest max rel-L2 "
              + f"{max(block['finest_errors'].values()):.3e}")
    print(f"json -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()