"""Separate the temporal from the spatial error of the 2-D Euler DG solver.

Background
----------
``docs/reports/tensordg-euler-dg-2d-design-order.md`` (mission m3024) measured the
design order of ``tensordg/euler_dg_2d.py`` on the smooth isentropic vortex for
P1/P2/P3.  P1 and P2 reached ``p + 1``; **P3 stopped at a fitted slope of
3.85/3.52/3.48/3.85 instead of 4** and the report left two items explicitly
open:

1. the P3 ladder topped out at 16^2 (a pre-asymptotic window), and
2. the time integrator is SSP-RK3, i.e. **third** order, while P3 asks for a
   fourth-order spatial error.  With the CFL held fixed ``dt ~ h``, so the
   temporal error ``O(dt^3) = O(h^3)`` decays *slower* than the spatial one and
   contaminates the measured order.

This benchmark closes both items with one sweep:

* extension of the P3 ladder to ``24^2, 32^2`` (and optionally ``40^2``), and
* a **temporal refinement** at every mesh: ``cfl in {0.4, 0.2, 0.1, 0.05}``.
  For a fixed mesh the error is modelled as ``e(dt) = e0 + C dt^q``; fitting it
  gives the temporal order ``q`` and the Richardson/least-squares limit
  ``e0 = e(dt -> 0)``, which is the *spatial-only* error.  The observed order of
  ``e0`` across meshes is the design order the mesh ladder is really delivering.

Everything is compared against the analytic isentropic vortex, so the reference
is exact (no manufactured source, no numerical reference).

Run::

    PYTHONPATH=. python -m benchmarks.temporal_asymptotics_euler_dg_2d \
        --degrees 3 --meshes 16 24 32
    PYTHONPATH=. python -m benchmarks.temporal_asymptotics_euler_dg_2d \
        --degrees 1 2 3 --meshes 8 12 16        # cheap validation of the method

Outputs ``validation/euler_dg_2d/temporal_asymptotics.json`` (rewritten after
every mesh so an interrupted sweep still leaves usable evidence) and the figure
``docs/figures/euler_dg_2d_temporal_asymptotics.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import tensordg.euler_dg_2d as dg2
from benchmarks.verify_euler_dg_2d import _vortex_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "euler_dg_2d" / "temporal_asymptotics.json"
FIG = ROOT / "docs" / "figures" / "euler_dg_2d_temporal_asymptotics.png"

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
FIELDS = ("density", "velocity_x", "velocity_y", "pressure")
DEFAULT_CFLS = (0.2, 0.1, 0.05)
RK_ORDER = 3          # SSP-RK3 time integrator of euler_dg_2d


def _fit_power_law(xs, ys):
    """Least-squares ``y = c x^m`` -> ``m`` (used for orders)."""
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


def _richardson(e_coarse, e_fine, ratio, order):
    """Limit of ``e = e0 + C dt^order`` from two levels (dt_coarse/dt_fine = ratio)."""
    return e_fine - (e_coarse - e_fine) / (ratio ** order - 1.0)


def _fit_three_parameter(dts, es, order):
    """Least-squares ``e = e0 + C dt^q`` with ``q`` fixed -> ``(e0, C)``."""
    a = np.stack([np.ones_like(dts), np.asarray(dts, float) ** order], axis=1)
    (e0, c), *_ = np.linalg.lstsq(a, np.asarray(es, float), rcond=None)
    return float(e0), float(c)


def temporal_block(degree, ncells, cfls, final_time=1.0, log=print):
    """Sweep the CFL at a fixed mesh; returns per-field temporal orders + limits."""
    if len(cfls) < 3:
        raise ValueError("need at least three CFL levels to separate e0 from the dt^q term")
    nodes, _weights, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    rows = []
    for cfl in cfls:
        started = time.time()
        _, _, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
            ncells=ncells, degree=degree, final_time=final_time, cfl=cfl
        )
        errors = _vortex_error(coeff, basis, nodes, DOMAIN, final_time)
        dt_mean = final_time / metadata["steps"]
        row = {
            "cfl": cfl,
            "steps": int(metadata["steps"]),
            "dt": dt_mean,                       # mean step, NOT the last dt
            "errors": {name: float(errors[name]) for name in FIELDS},
            "wall_seconds": round(time.time() - started, 2),
        }
        rows.append(row)
        log(f"  P{degree} {ncells:3d}^2 cfl={cfl:5.3f} steps={row['steps']:5d} "
            + "  ".join(f"{k} {row['errors'][k]:.4e}" for k in FIELDS)
            + f"  ({row['wall_seconds']:.1f}s)")

    out = {"degree": degree, "cells": ncells, "h": 10.0 / ncells, "rows": rows}
    out["temporal_order"] = {}
    out["spatial_limit"] = {}
    for name in FIELDS:
        dts = [row["dt"] for row in rows]
        es = [row["errors"][name] for row in rows]
        adjacent = [
            float(math.log(es[i] / es[i + 1]) / math.log(dts[i] / dts[i + 1]))
            for i in range(len(rows) - 1)
        ]
        # Relative spread of the error over the whole dt ladder.  When it is at
        # round-off level the temporal error is invisible: any Richardson
        # division by (2^q - 1) with q ~ 0 would then explode, so the limit is
        # reported as the raw finest error with the share flagged as zero.
        spread = float(max(es) / min(es) - 1.0)
        noisy = spread < 1e-4
        if noisy:
            q = float(RK_ORDER)                    # theoretical SSP-RK3 order
            rich = es[-1]
            e0_ls = es[-1]
        else:
            q = float(np.clip(np.median(adjacent[-2:]), 0.5, 6.0))
            ratio = dts[-2] / dts[-1]
            rich = float(_richardson(es[-2], es[-1], ratio, q))
            e0_ls, _ = _fit_three_parameter(dts, es, q)
            # a fit can only lower the error, and never below zero
            rich = min(max(rich, 0.0), es[-1])
            e0_ls = min(max(e0_ls, 0.0), es[-1])
        out["temporal_order"][name] = {
            "adjacent": adjacent,
            "fit": _fit_power_law(dts, es),
            "q": q,
            "relative_spread": spread,
            "temporal_negligible": noisy,
            "temporal_share_upper_bound": (0.0 if noisy else float(1.0 - rich / es[-1])),
        }
        out["spatial_limit"][name] = {
            "richardson": float(rich),
            "least_squares": float(e0_ls),
            "raw_finest": es[-1],
            "temporal_share_of_finest": float(1.0 - rich / es[-1]),
        }
    out["total_seconds"] = round(sum(row["wall_seconds"] for row in rows), 1)
    return out


def spatial_orders(blocks):
    """EOC of the dt -> 0 limits and of the raw finest-CFL errors."""
    by_degree = {}
    for block in blocks:
        by_degree.setdefault(block["degree"], []).append(block)
    orders = {}
    for degree, chain in by_degree.items():
        chain.sort(key=lambda b: b["cells"])
        if len(chain) < 2:
            continue
        entry = {"eoc_limit": {}, "eoc_raw": {}, "fit_limit": {}, "fit_raw": {}}
        for name in FIELDS:
            entry["eoc_limit"][name] = [
                float(math.log(chain[i]["spatial_limit"][name]["richardson"]
                               / chain[i + 1]["spatial_limit"][name]["richardson"])
                      / math.log(chain[i]["h"] / chain[i + 1]["h"]))
                for i in range(len(chain) - 1)
            ]
            entry["eoc_raw"][name] = [
                float(math.log(chain[i]["rows"][-1]["errors"][name]
                               / chain[i + 1]["rows"][-1]["errors"][name])
                      / math.log(chain[i]["h"] / chain[i + 1]["h"]))
                for i in range(len(chain) - 1)
            ]
            hs = [b["h"] for b in chain]
            entry["fit_limit"][name] = _fit_power_law(
                hs, [b["spatial_limit"][name]["richardson"] for b in chain])
            entry["fit_raw"][name] = _fit_power_law(
                hs, [b["rows"][-1]["errors"][name] for b in chain])
        entry["design_order"] = degree + 1
        entry["cells"] = [b["cells"] for b in chain]
        orders[degree] = entry
    return orders


def load_blocks():
    """Previously written blocks, so separate invocations accumulate."""
    if not OUT.exists():
        return []
    try:
        return json.loads(OUT.read_text()).get("blocks", [])
    except json.JSONDecodeError:
        return []


def write_json(blocks, orders, cfl_ladder=None):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ladders = sorted({tuple(row["cfl"] for row in b["rows"]) for b in blocks})
    OUT.write_text(json.dumps({
        "benchmark": "euler_dg_2d_temporal_asymptotics",
        "domain": DOMAIN,
        "final_time": 1.0,
        "cfl_ladders": [list(l) for l in ladders],
        "blocks": blocks,
        "spatial_orders": {str(k): v for k, v in orders.items()},
    }, indent=1))


def make_figure(blocks, orders):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    for block in blocks:
        p, c = block["degree"], colours.get(block["degree"], "k")
        dts = [row["dt"] for row in block["rows"]]
        axes[0].loglog(dts, [row["errors"]["density"] for row in block["rows"]],
                       "o-", color=c, label=f"P{p} {block['cells']}$^2$ $\\rho$")
        axes[0].loglog(dts, [row["errors"]["pressure"] for row in block["rows"]],
                       "s--", color=c, label=f"P{p} {block['cells']}$^2$ $p$")
        axes[1].semilogy(dts, [row["errors"]["density"] for row in block["rows"]],
                         "o-", color=c, label=f"P{p} {block['cells']}$^2$")

    axes[0].set_xlabel("mean dt"); axes[0].set_ylabel("relative $L^2$ error")
    axes[0].set_title("Temporal refinement at fixed mesh")
    axes[0].legend(fontsize=6, ncol=2)

    for degree, entry in orders.items():
        c = colours.get(degree, "k")
        hs = [10.0 / n for n in entry["cells"]]
        lim = [b["spatial_limit"]["density"]["richardson"]
               for b in sorted([b for b in blocks if b["degree"] == degree],
                               key=lambda b: b["cells"])]
        raw = [b["rows"][-1]["errors"]["density"]
               for b in sorted([b for b in blocks if b["degree"] == degree],
                               key=lambda b: b["cells"])]
        axes[1].loglog(hs, lim, "o-", color=c, label=f"P{degree} dt->0 limit $\\rho$")
        axes[1].loglog(hs, raw, "x:", color=c, label=f"P{degree} finest-CFL $\\rho$")
        ref = lim[0]
        axes[1].loglog(hs, [ref * (h / hs[0]) ** (degree + 1) for h in hs], ":",
                       color=c, lw=1, label=f"P{degree} slope {degree + 1}")

    axes[1].set_xlabel("h"); axes[1].set_ylabel("relative $L^2$ error")
    axes[1].set_title("Spatial-only error (dt->0) vs h")
    axes[1].legend(fontsize=6, ncol=2)

    for degree, entry in orders.items():
        c = colours.get(degree, "k")
        xs = range(len(entry["eoc_limit"]["density"]))
        axes[2].plot([x - 0.08 for x in xs], entry["eoc_limit"]["density"], "o-",
                     color=c, label=f"P{degree} $\\rho$ (dt->0)")
        axes[2].plot([x + 0.08 for x in xs], entry["eoc_raw"]["density"], "s:",
                     color=c, label=f"P{degree} $\\rho$ (finest CFL)")
        axes[2].axhline(degree + 1, color=c, lw=0.8, ls=":")
    axes[2].set_xlabel("mesh interval"); axes[2].set_ylabel("observed order (EOC)")
    axes[2].set_title("Order before / after removing the RK3 error")
    axes[2].legend(fontsize=6)

    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    return FIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrees", type=int, nargs="+", default=[3])
    parser.add_argument("--meshes", type=int, nargs="+", default=[16, 24, 32])
    parser.add_argument("--cfls", type=float, nargs="+", default=list(DEFAULT_CFLS))
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    blocks = load_blocks()
    for degree in args.degrees:
        for ncells in args.meshes:
            print(f"[P{degree} {ncells}^2] cfl ladder {args.cfls}", flush=True)
            fresh = temporal_block(degree, ncells, args.cfls,
                                   args.final_time, log=lambda m: print(m, flush=True))
            blocks = [b for b in blocks
                      if not (b["degree"] == degree and b["cells"] == ncells)] + [fresh]
            write_json(blocks, spatial_orders(blocks))

    orders = spatial_orders(blocks)
    write_json(blocks, orders)
    if not args.no_figure:
        print(f"figure -> {make_figure(blocks, orders).relative_to(ROOT)}")

    print("\nspatial order summary (density channel)")
    for degree, entry in sorted(orders.items()):
        print(f"P{degree} design {entry['design_order']}: "
              f"dt->0 fit {entry['fit_limit']['density']:.2f} | "
              f"raw fit {entry['fit_raw']['density']:.2f} | "
              f"EOC(dt->0) {['%.2f' % v for v in entry['eoc_limit']['density']]}")
    print(f"json -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()