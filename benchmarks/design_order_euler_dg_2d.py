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

Space/time order coupling
-------------------------
The solver advances in time with SSP-RK3 and ``dt = CFL / rate`` with
``rate ~ 1/h``, i.e. ``dt ∝ CFL · h``.  The spatial error is
``O(h^(p+1))`` but the temporal error is only ``O(dt^3) = O(h^3)``, so for
``p >= 3`` the *observed* order of the space-time discretisation is capped at
3 no matter how fine the mesh becomes.  The original study found P3 at
3.85/3.52/3.48/3.85 (below the design order 4) and left two follow-ups open:
a finer mesh ladder, and a time-step study.  This file now provides both:

* ``--order-matched-dt`` scales the CFL number with the mesh,
  ``CFL(h) = CFL_0 (h/h_0)^gamma`` with ``gamma = (p+1)/3 - 1``, so that the
  temporal term decays as fast as the spatial one; the isolator is
  :func:`cfl_schedule` and ``gamma`` is clipped at 0 for ``p <= 2`` (there the
  temporal order already matches or exceeds the spatial one).
* ``--temporal`` refines ``dt`` at a *fixed* mesh to measure the temporal order
  directly and to show where the space-error floor takes over.

Run::

    PYTHONPATH=. python -m benchmarks.design_order_euler_dg_2d
    PYTHONPATH=. python -m benchmarks.design_order_euler_dg_2d --degrees 1 --meshes 8 16 32
    PYTHONPATH=. python -m benchmarks.design_order_euler_dg_2d --degrees 3 \
        --meshes 8 12 16 24 32 --order-matched-dt --temporal \
        --out validation/euler_dg_2d/order_time_coupling.json

Numbers go to ``validation/euler_dg_2d/design_order.json`` (written after every
mesh so an interrupted sweep still leaves usable evidence) and the log-log
figure to ``docs/figures/euler_dg_2d_design_order.png``; an ``--out`` override
writes the extended study elsewhere and a three-panel figure next to it.
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
FIG_EXTENDED = ROOT / "docs" / "figures" / "euler_dg_2d_order_time_coupling.png"

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))   # length 10, vortex strength 5, background (1, 1)
FIELDS = ("density", "velocity_x", "velocity_y", "pressure")

# Four rungs per degree at a ~1.33-1.5 refinement ratio: enough for a
# least-squares slope, cheap enough to stay inside a shared 128-core box.
DEFAULT_MESHES = {1: (12, 16, 24, 32), 2: (8, 12, 16, 24), 3: (6, 8, 12, 16)}

# Order of the time integrator hard-wired in tensordg/euler_dg_2d.py (SSP-RK3).
TIME_ORDER = 3


def slope(xs, ys):
    """Least-squares slope of ``log(y)`` against ``log(x)``."""
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


def order_matching_exponent(degree, time_order=TIME_ORDER):
    """CFL exponent ``gamma`` that makes the temporal error decay like ``h^(p+1)``.

    The temporal truncation error is ``O(dt^q)`` with ``q = time_order`` and
    ``dt ∝ CFL · h``, so it decays like ``h^(q + gamma·q)``.  Matching the
    spatial term ``h^(p+1)`` needs ``gamma = (p+1)/q - 1``.  For ``p + 1 <= q``
    the time error already decays fast enough and ``gamma`` is clipped at 0
    (which keeps the historical P1/P2 runs bit-identical).
    """
    return max(0.0, (degree + 1) / float(time_order) - 1.0)


def cfl_schedule(degree, meshes, cfl, term="h", time_order=TIME_ORDER):
    """Per-mesh CFL numbers for ``term='h'`` (order matched) or ``term='const'``.

    ``meshes`` are cell counts (the domain length is 10 for the vortex), the
    first entry is the reference rung, so ``cfl_schedule(...)[0] == cfl`` always
    and every finer rung is scaled by ``(h/h_ref)^gamma``.
    """
    if term == "const":
        return [float(cfl) for _ in meshes]
    if term != "h":
        raise ValueError("term must be 'h' (order matched) or 'const'")
    gamma = order_matching_exponent(degree, time_order)
    h = [10.0 / float(n) for n in meshes]
    return [float(cfl) * (hi / h[0]) ** gamma for hi in h]


def sweep(degree, meshes, final_time=1.0, cfl=0.2, log=None, cfl_term="const",
          quadrature=None):
    """Run one degree over the mesh ladder; returns the block + the figure data.

    ``cfl_term='const'`` reproduces the historical fixed-CFL ladder;
    ``cfl_term='h'`` applies :func:`cfl_schedule` (order-matched time step).

    ``quadrature`` overrides the number of volume-quadrature points per
    direction (the solver default is ``max(degree + 2, 3)``).  A nonlinear flux
    is integrated inexactly unless the quadrature is over-integrated, so this
    knob isolates *aliasing* from every other candidate cause of an order
    deficit.
    """
    order = quadrature if quadrature is not None else max(degree + 2, 3)
    nodes, weights, basis, _, _ = dg2.legendre_operators(degree, order)
    cfls = cfl_schedule(degree, meshes, cfl, cfl_term)
    rows = []
    for ncells, cell_cfl in zip(meshes, cfls):
        started = time.time()
        if quadrature is None:
            _, _, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
                ncells=ncells, degree=degree, final_time=final_time, cfl=cell_cfl
            )
        else:
            _, _, coeff, metadata = dg2.solve_euler_dg_2d(
                lambda x, y: dg2.vortex_primitive(x, y, 0.0, 5.0, (1.0, 1.0)),
                ncells=ncells, degree=degree, domain=DOMAIN,
                final_time=final_time, cfl=cell_cfl, quadrature=quadrature,
            )
        errors = _vortex_error(coeff, basis, nodes, DOMAIN, final_time)
        row = {
            "degree": degree,
            "cells": ncells,
            "h": 10.0 / ncells,
            "cfl": cell_cfl,
            "quadrature": order,
            "dofs": metadata["dofs"],
            "steps": metadata["steps"],
            "dt": metadata["dt"],
            # the last step may be clipped by final_time, so use the mean step
            "dt_mean": final_time / metadata["steps"],
            "errors": {name: float(errors[name]) for name in FIELDS},
            "wall_seconds": round(time.time() - started, 2),
        }
        rows.append(row)
        if log is not None:
            log(f"  P{degree} {ncells:3d}^2: cfl {cell_cfl:.4f}  "
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
        "cfl_term": cfl_term,
        "quadrature": order,
        "cfl_exponent": 0.0 if cfl_term == "const" else order_matching_exponent(degree),
        "rows": rows,
        "fit_slope": {name: slope(hs, [row["errors"][name] for row in rows]) for name in FIELDS},
        "finest_errors": {name: rows[-1]["errors"][name] for name in FIELDS},
        "finest_below_three_percent": {
            name: rows[-1]["errors"][name] < 0.03 for name in FIELDS
        },
        "total_seconds": round(sum(row["wall_seconds"] for row in rows), 1),
    }


def temporal_study(degree, ncells, final_time=1.0, cfl=0.2, levels=4, log=None):
    """Refine ``dt`` at a *fixed* mesh to measure the temporal order directly.

    Only the CFL number changes, so the spatial error is a fixed floor: the
    measured slope is the temporal order until the floor dominates, after which
    it collapses towards 0.  Reporting the collapse is the point -- it shows
    that a ``dt``-refinement study alone cannot certify the spatial order.
    """
    nodes, weights, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    rows = []
    for level in range(levels):
        cell_cfl = cfl / 2.0 ** level
        started = time.time()
        _, _, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
            ncells=ncells, degree=degree, final_time=final_time, cfl=cell_cfl
        )
        errors = _vortex_error(coeff, basis, nodes, DOMAIN, final_time)
        row = {
            "degree": degree,
            "cells": ncells,
            "cfl": cell_cfl,
            "steps": metadata["steps"],
            "dt_mean": final_time / metadata["steps"],
            "errors": {name: float(errors[name]) for name in FIELDS},
            "wall_seconds": round(time.time() - started, 2),
        }
        rows.append(row)
        if log is not None:
            log(f"  temporal P{degree} {ncells}^2 cfl {cell_cfl:.4f} "
                f"({metadata['steps']} steps): "
                + "  ".join(f"{k} {row['errors'][k]:.4e}" for k in FIELDS)
                + f"  ({row['wall_seconds']:.1f}s)")

    for previous, current in zip(rows, rows[1:]):
        ratio = previous["dt_mean"] / current["dt_mean"]
        current["temporal_eoc"] = {
            name: float(math.log(previous["errors"][name] / current["errors"][name])
                        / math.log(ratio))
            for name in FIELDS
        }
    dts = [row["dt_mean"] for row in rows]
    return {
        "degree": degree,
        "cells": ncells,
        "design_time_order": TIME_ORDER,
        "rows": rows,
        "fit_slope": {name: slope(dts, [row["errors"][name] for row in rows]) for name in FIELDS},
        "total_seconds": round(sum(row["wall_seconds"] for row in rows), 1),
    }


def write_json(blocks, path=OUT):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"benchmark": "euler_dg_2d_design_order",
                                "domain": DOMAIN, "blocks": blocks}, indent=1))
    return path


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


def make_coupling_figure(fixed_cfl, matched, temporal, path=FIG_EXTENDED):
    """Three panels: fixed vs order-matched ladder, and the dt-refinement study."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    for block, colour, tag in ((fixed_cfl, "tab:red", "fixed CFL"),
                               (matched, "tab:green", "order-matched dt")):
        if block is None:
            continue
        p = block["degree"]
        hs = [row["h"] for row in block["rows"]]
        axes[0].loglog(hs, [row["errors"]["density"] for row in block["rows"]],
                       "o-", color=colour, label=f"P{p} rho, {tag}")
        ref, h0 = block["rows"][0]["errors"]["density"], hs[0]
        for order, ls in ((p + 1, ":"), (TIME_ORDER, "--")):
            axes[0].loglog(hs, [ref * (h / h0) ** order for h in hs], ls,
                           color=colour, lw=1, label=f"slope {order}")
        axes[1].semilogx(hs[1:], [r["eoc"]["density"] for r in block["rows"][1:]],
                         "o-", color=colour, label=f"density EOC ({tag})")
        axes[1].semilogx(hs[1:], [r["eoc"]["pressure"] for r in block["rows"][1:]],
                         "s--", color=colour, label=f"pressure EOC ({tag})")

    if fixed_cfl is not None:
        p = fixed_cfl["degree"]
        axes[1].axhline(p + 1, color="k", lw=0.8, ls=":", label=f"design order {p + 1}")
        axes[1].axhline(TIME_ORDER, color="grey", lw=0.8, ls="-.",
                        label=f"SSP-RK3 order {TIME_ORDER}")

    if temporal is not None:
        dts = [row["dt_mean"] for row in temporal["rows"]]
        axes[2].loglog(dts, [row["errors"]["density"] for row in temporal["rows"]],
                       "o-", color="tab:purple", label="density error")
        ref, d0 = temporal["rows"][0]["errors"]["density"], dts[0]
        for order, ls in ((TIME_ORDER, "--"), (1, ":"), (2, ":")):
            axes[2].loglog(dts, [ref * (d / d0) ** order for d in dts], ls,
                           color="grey", lw=1, label=f"slope {order}")
        axes[2].set_title(f"dt refinement at fixed {temporal['cells']}^2 "
                          f"(P{temporal['degree']})")

    axes[0].set_xlabel("h"); axes[0].set_ylabel("relative $L^2$ error")
    axes[0].set_title("P3 vortex: fixed CFL vs order-matched dt")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("h"); axes[1].set_ylabel("observed order (EOC)")
    axes[1].set_title("Observed order: capped by the time integrator?")
    axes[1].legend(fontsize=7)
    if temporal is not None:
        axes[2].set_xlabel("mean dt"); axes[2].set_ylabel("relative $L^2$ error")
        axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--meshes", type=int, nargs="+", default=None,
                        help="override the mesh ladder (applies to every degree)")
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--order-matched-dt", action="store_true",
                        help="scale CFL with h^(p+1)/q-1 so time and space errors "
                             "decay at the same rate")
    parser.add_argument("--quadrature", type=int, default=None,
                        help="volume quadrature points per direction "
                             "(default max(degree+2,3)); an over-integration "
                             "test for nonlinear-flux aliasing")
    parser.add_argument("--temporal", action="store_true",
                        help="also refine dt at a fixed mesh to measure the "
                             "temporal order")
    parser.add_argument("--temporal-degree", type=int, default=3)
    parser.add_argument("--temporal-cells", type=int, default=16)
    parser.add_argument("--temporal-levels", type=int, default=4)
    parser.add_argument("--out", default=None,
                        help="JSON output path (default validation/euler_dg_2d/"
                             "design_order.json)")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    term = "h" if args.order_matched_dt else "const"
    out_path = Path(args.out) if args.out else OUT
    blocks = []
    for degree in args.degrees:
        meshes = args.meshes or DEFAULT_MESHES[degree]
        print(f"[P{degree}] design order {degree + 1}, meshes {meshes}, "
              f"cfl term {term}", flush=True)
        block = sweep(degree, meshes, args.final_time, args.cfl,
                      log=lambda m: print(m, flush=True), cfl_term=term,
                      quadrature=args.quadrature)
        blocks.append(block)
        write_json(blocks, out_path)
        print(f"[P{degree}] fit slope "
              + "  ".join(f"{k} {v:.2f}" for k, v in block["fit_slope"].items()), flush=True)

    temporal = None
    if args.temporal:
        print(f"[temporal] P{args.temporal_degree} at {args.temporal_cells}^2, "
              f"{args.temporal_levels} dt levels", flush=True)
        temporal = temporal_study(args.temporal_degree, args.temporal_cells,
                                  args.final_time, args.cfl, args.temporal_levels,
                                  log=lambda m: print(m, flush=True))
        figures = {"blocks": blocks, "temporal": temporal}
        out_path.write_text(json.dumps({"benchmark": "euler_dg_2d_design_order",
                                        "domain": DOMAIN, **figures}, indent=1))
        print(f"[temporal] fit slope "
              + "  ".join(f"{k} {v:.2f}" for k, v in temporal["fit_slope"].items()),
              flush=True)

    if not args.no_figure:
        if args.temporal or args.order_matched_dt:
            fig = make_coupling_figure(None, blocks[0] if blocks else None,
                                       temporal)
            print(f"figure -> {fig.relative_to(ROOT)}")
        else:
            make_figure(blocks)
            print(f"figure -> {FIG.relative_to(ROOT)}")

    print("\nsummary (observed order vs design order p+1)")
    for block in blocks:
        p = block["degree"]
        print(f"P{p}: design {p + 1} | fit "
              + " ".join(f"{k} {v:.2f}" for k, v in block["fit_slope"].items())
              + " | finest max rel-L2 "
              + f"{max(block['finest_errors'].values()):.3e}")
    if temporal is not None:
        print("temporal: design "
              + str(TIME_ORDER) + " | fit "
              + " ".join(f"{k} {v:.2f}" for k, v in temporal["fit_slope"].items()))
    print(f"json -> {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path}")


if __name__ == "__main__":
    main()