"""Decompose the P3 vortex velocity-error floor into a mean offset and a shape part.

``validation/euler_dg_2d/p3_ladder_fine.json`` (produced by
``benchmarks.design_order_euler_dg_2d``) shows the density and pressure errors of
the P3 isentropic-vortex run decaying at the design order (EOC ~3.9-4.0) while
the *velocity* error stalls: 1.1621e-04 at 16^2 and 8.6293e-05 at 24^2 (EOC 0.73).

A relative L2 error can stall for two very different reasons:

* a **mean offset** -- the domain-averaged velocity drifts by a constant
  ``delta`` and never converges (relative floor ``delta / ||u||``); this is a
  conservation/consistency defect and would be visible as a non-zero
  domain-integral of the momentum;
* a **shape error** -- the velocity profile itself converges slowly; this is a
  resolution/pre-asymptotic effect and keeps decreasing under refinement.

The probe solves the same case, splits the error

    e_offset = |<u>_num - <u>_exact| / ||u_exact||,
    e_shape  = ||(u - <u>) - (u_exact - <u_exact>)|| / ||u_exact||,

with the same tensor-product Gauss weights the solver uses, and reports both,
plus the same split for density and pressure as controls.  It also dumps the
pressure field and its error at the finest mesh for the report figure.

Run::

    PYTHONPATH=. python -m tools.probe_velocity_order_floor
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import tensordg.euler_dg_2d as dg2
from tensordg.euler_2d import vortex_primitive
from benchmarks.verify_euler_dg_2d import _reconstruct

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
FIELDS = ("density", "velocity_x", "pressure")
LADDER = ROOT / "validation" / "euler_dg_2d" / "p3_ladder_fine.json"
OUT = ROOT / "validation" / "euler_dg_2d" / "velocity_order_floor.json"
FIG = ROOT / "docs" / "figures" / "euler_dg_2d_p3_order_decomposition.png"


def _grid(ncells, degree, final_time, strength=5.0, background=(1.0, 1.0)):
    order = max(degree + 2, 3)
    nodes, weights, basis, _, _ = dg2.legendre_operators(degree, order)
    _, _, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
        ncells=ncells, degree=degree, final_time=final_time,
        strength=strength, background_velocity=background,
    )
    (xmin, xmax), (ymin, ymax) = DOMAIN
    dx, dy = (xmax - xmin) / ncells, (ymax - ymin) / ncells
    xc = xmin + (torch.arange(ncells, dtype=coeff.dtype) + 0.5) * dx
    yc = ymin + (torch.arange(ncells, dtype=coeff.dtype) + 0.5) * dy
    xq = xc[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = yc[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    exact = vortex_primitive(xq, yq, final_time, strength, background)
    state = _reconstruct(coeff, basis)
    rho, u, v, p = dg2._primitive(state)
    # tensor-product Gauss weights normalised so that sum(w) == 1
    w = (weights[:, None] * weights[None, :]) / 4.0
    w = torch.as_tensor(w, dtype=coeff.dtype).expand(ncells, ncells, -1, -1)
    return {
        "numeric": {"density": rho, "velocity_x": u, "pressure": p},
        "exact": {"density": exact[0], "velocity_x": exact[1], "pressure": exact[3]},
        "weights": w,
        "metadata": metadata,
        "xq": xq, "yq": yq,
    }


def _split(nums, exas, w):
    """Relative offset / shape split of the L2 error of one field.

    All norms are volume averages (divided by ``sum(w)``) so that the ``total``
    column reproduces the relative L2 error of
    ``benchmarks.verify_euler_dg_2d._vortex_error`` -- forgetting this factor
    inflates every error by ``sqrt(n_cells)`` and fakes a diverging split.
    """
    wsum = float(w.sum())
    norm = float(torch.sqrt((w * exas ** 2).sum() / wsum))
    mean_num = float((w * nums).sum() / wsum)
    mean_exa = float((w * exas).sum() / wsum)
    offset = abs(mean_num - mean_exa) / norm
    shape = float(torch.sqrt((w * ((nums - mean_num) - (exas - mean_exa)) ** 2).sum() / wsum)) / norm
    total = float(torch.sqrt((w * (nums - exas) ** 2).sum() / wsum)) / norm
    return {"offset": offset, "shape": shape, "total": total,
            "mean_numeric": mean_num, "mean_exact": mean_exa}


def main():
    import os
    final_time, degree = 1.0, 3
    meshes = tuple(int(n) for n in os.environ.get("PROBE_MESHES", "16 24").split())
    rows, contour = [], None
    for ncells in meshes:
        started = time.time()
        data = _grid(ncells, degree, final_time)
        w = data["weights"]
        row = {"degree": degree, "cells": ncells, "h": 10.0 / ncells,
               "steps": data["metadata"]["steps"]}
        for field in FIELDS:
            row[field] = _split(data["numeric"][field], data["exact"][field], w)
        row["wall_seconds"] = round(time.time() - started, 1)
        rows.append(row)
        if ncells == max(meshes):
            contour = {
                "cells": ncells,
                "pressure_numeric": data["numeric"]["pressure"][..., 0, 0].tolist(),
                "pressure_error": (data["numeric"]["pressure"]
                                   - data["exact"]["pressure"])[..., 0, 0].tolist(),
            }
        print(f"P{degree} {ncells}^2: "
              + "  ".join(f"{f} off {row[f]['offset']:.2e} shape {row[f]['shape']:.2e}"
                          for f in FIELDS)
              + f"  ({row['wall_seconds']:.1f}s)", flush=True)

    for previous, current in zip(rows, rows[1:]):
        ratio = previous["h"] / current["h"]
        current["eoc"] = {
            f"{f}_{part}": float(math.log(previous[f][part] / current[f][part]) / math.log(ratio))
            for f in FIELDS for part in ("offset", "shape", "total")
            if current[f][part] > 0 and previous[f][part] > 0
        }

    # self-check: the total split must reproduce the benchmark's own relative L2
    if LADDER.exists():
        ladder = {r["cells"]: r["errors"] for r in
                  json.loads(LADDER.read_text())["blocks"][0]["rows"]}
        for row in rows:
            ref = ladder.get(row["cells"])
            if ref:
                row["ladder_check"] = {f: row[f]["total"] / ref[f] for f in FIELDS}
                print(f"self-check {row['cells']}^2 total/ladder "
                      + "  ".join(f"{f} {row['ladder_check'][f]:.3f}" for f in FIELDS),
                      flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"benchmark": "p3_velocity_order_floor",
                               "domain": DOMAIN, "rows": rows,
                               "contour": contour}, indent=1))
    print(f"json -> {OUT.relative_to(ROOT)}")

    make_figure(rows)
    print(f"figure -> {FIG.relative_to(ROOT)}")


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    colours = {"density": "tab:blue", "velocity_x": "tab:red", "pressure": "tab:green"}

    ladder = json.loads(LADDER.read_text())["blocks"][0] if LADDER.exists() else None
    if ladder is not None:
        hs = [r["h"] for r in ladder["rows"]]
        for field in ("density", "velocity_x", "pressure"):
            axes[0].loglog(hs, [r["errors"][field] for r in ladder["rows"]], "o-",
                           color=colours[field], label=f"P3 {field}")
        ref, h0 = ladder["rows"][0]["errors"]["density"], hs[0]
        axes[0].loglog(hs, [ref * (h / h0) ** 4 for h in hs], "k:", lw=1, label="slope 4")
        axes[0].set_title("P3 vortex ladder, fixed CFL 0.2")

    hs = [r["h"] for r in rows]
    for field in ("density", "velocity_x", "pressure"):
        axes[1].loglog(hs, [r[field]["offset"] for r in rows], "o--",
                       color=colours[field], label=f"{field} offset")
        axes[1].loglog(hs, [r[field]["shape"] for r in rows], "s-",
                       color=colours[field], label=f"{field} shape")
    axes[1].set_title("Error split: mean offset vs shape")

    if LADDER.exists():
        lad = json.loads(LADDER.read_text())["blocks"][0]
        lh = [r["h"] for r in lad["rows"]]
        for field in ("density", "velocity_x", "pressure"):
            axes[2].plot(lh[1:], [r["eoc"][field] for r in lad["rows"][1:]], "o-",
                         color=colours[field], label=f"{field} EOC")
        axes[2].axhline(4, color="k", ls=":", lw=1, label="design order 4")
        axes[2].axhline(3, color="grey", ls="-.", lw=1, label="SSP-RK3 order 3")

    axes[0].set_xlabel("h"); axes[0].set_ylabel("relative $L^2$ error")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("h"); axes[1].set_ylabel("relative contribution")
    axes[1].set_title("P3: offset (dashed) vs shape (solid)")
    axes[1].legend(fontsize=7)
    axes[2].set_xlabel("h"); axes[2].set_ylabel("observed order")
    axes[2].set_title("Per-interval observed order")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()