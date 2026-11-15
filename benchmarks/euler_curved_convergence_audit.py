"""Convergence audit of ``tensordg.euler_curved`` (curvilinear 2-D Euler DG).

Question this script answers: are the two red gates in
``tests/test_euler_curved.py`` (F3 design order, F5 conservation) *gate
artefacts* -- i.e. the smooth-vortex problem being too coarse -- or is the
isoparametric operator itself not converging?

Design of the experiment (each column removes one explanation):

* ``projection``  -- L2 projection error at t=0 on the same mesh ladder.  If this
  is O(h^(p+1)) the mesh, the geometry map, the quadrature and the reference
  solution are all fine, so any error left in the evolved state comes from the
  spatial operator / time march.
* ``evolve``      -- the same march at the solver's own CFL-based dt.
* ``fixed_dt``    -- the same march with a *mesh-independent* dt, so the temporal
  truncation error is identical on every rung (a non-converging error cannot be
  blamed on the time integrator).
* ``cartesian``   -- the repository's verified Cartesian solver
  (``benchmarks.design_order_euler_dg_2d.sweep``, i.e. ``euler_dg_2d``) on the
  *same* vortex, degree, CFL and element counts.  This is the reference for "what
  a correct DG operator does with this problem at this resolution".

Writes ``validation/euler_curved_convergence/audit.json`` and a markdown table
next to it; every number in ``docs/AUDIT-m21473-curvilinear-euler-nonconvergence.md``
comes from that JSON (no hand-copied numbers).

Run::

    OMP_NUM_THREADS=1 PYTHONPATH=. python -m benchmarks.euler_curved_convergence_audit
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensordg.euler_curved import (
    CurvilinearEuler,
    straight_box_mesh,
    warped_box_mesh,
)
from tensordg.euler_2d import vortex_primitive

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "euler_curved_convergence"
BOX = 10.0
AMPLITUDE = 0.06
GEOMETRY_DEGREE = 3
CFL = 0.2
FINAL_TIME = 0.1
MESHES = (8, 16, 32)


def exact_primitive(strength: float, time: float):
    def primitive(x, y):
        rho, u, v, p = vortex_primitive(
            torch.as_tensor(np.ascontiguousarray(x)),
            torch.as_tensor(np.ascontiguousarray(y)),
            time=time, strength=strength, background_velocity=(1.0, 1.0), period=BOX,
        )
        return (rho.numpy(), u.numpy(), v.numpy(), p.numpy())

    return primitive


def mesh_of(family: str, cells: int):
    if family == "warped":
        return warped_box_mesh(cells, cells, GEOMETRY_DEGREE, length=BOX,
                               amplitude=AMPLITUDE)
    return straight_box_mesh(cells, cells, 1, length=BOX)


def projection_row(family: str, cells: int, strength: float) -> dict:
    solver = CurvilinearEuler(mesh_of(family, cells), degree=1, cfl=CFL)
    state = solver.project(exact_primitive(strength, 0.0))
    err = solver.relative_errors(state, exact_primitive(strength, 0.0))["worst"]
    return {"cells": cells, "worst": float(err)}


def evolve_row(family: str, cells: int, strength: float, dt=None) -> dict:
    solver = CurvilinearEuler(mesh_of(family, cells), degree=1, cfl=CFL)
    state = solver.project(exact_primitive(strength, 0.0))
    unperturbed = solver.totals(state)
    try:
        state, used_dt, steps = solver.evolve(state, FINAL_TIME, dt=dt)
    except RuntimeError as exc:
        return {"cells": cells, "blown_up": True, "message": str(exc)[:120]}
    err = solver.relative_errors(state, exact_primitive(strength, FINAL_TIME))["worst"]
    drift = np.abs(solver.totals(state) - unperturbed) / np.maximum(
        np.abs(unperturbed), 1e-30)
    return {
        "cells": cells,
        "blown_up": False,
        "worst": float(err),
        "dt": float(used_dt),
        "steps": int(steps),
        "mass_drift_relative": float(drift[0]),
    }


def orders(rows, key="worst"):
    out = []
    for left, right in zip(rows, rows[1:]):
        if left.get("blown_up") or right.get("blown_up"):
            out.append(None)
            continue
        out.append(float(math.log(left[key] / right[key])
                         / math.log(right["cells"] / left["cells"])))
    return out


def cartesian_rows(strength: float) -> list:
    from benchmarks.design_order_euler_dg_2d import sweep

    rows = []
    for cells in MESHES:
        report = sweep(1, [cells], final_time=FINAL_TIME, cfl=CFL)
        row = report["rows"][0]
        rows.append({"cells": cells,
                     "worst": float(max(row["errors"].values())),
                     "errors": {k: float(v) for k, v in row["errors"].items()},
                     "steps": row["steps"], "dt": row["dt"]})
    return rows


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"final_time": FINAL_TIME, "cfl": CFL, "degree": 1,
              "meshes": list(MESHES), "amplitude": AMPLITUDE, "blocks": {}}
    t_start = time.time()
    for strength in (1.0, 5.0):
        tag = f"strength_{strength:g}"
        block = {}
        for family in ("straight", "warped"):
            proj = [projection_row(family, c, strength) for c in MESHES]
            ev = [evolve_row(family, c, strength) for c in MESHES]
            fixed = [evolve_row(family, c, strength, dt=1e-3) for c in (8, 16)]
            block[family] = {
                "projection": proj, "projection_orders": orders(proj),
                "evolve": ev, "evolve_orders": orders(ev),
                "fixed_dt_1e-3": fixed,
            }
            print(f"[{tag}] {family}")
            print("   projection errs", [f"{r['worst']:.3e}" for r in proj],
                  "orders", block[family]["projection_orders"])
            print("   evolved errs",
                  ["BLOWUP" if r.get("blown_up") else f"{r['worst']:.3e}" for r in ev],
                  "orders", block[family]["evolve_orders"])
            print("   fixed-dt errs",
                  [f"{r['worst']:.3e}" for r in fixed])
        block["cartesian_reference"] = cartesian_rows(strength)
        block["cartesian_orders"] = orders(block["cartesian_reference"])
        print(f"[{tag}] cartesian(euler_dg_2d) errs",
              [f"{r['worst']:.3e}" for r in block["cartesian_reference"]],
              "orders", block["cartesian_orders"])
        report["blocks"][tag] = block
    report["wall_seconds"] = time.time() - t_start
    (OUT / "audit.json").write_text(json.dumps(report, indent=2))
    (OUT / "audit.md").write_text(markdown_table(report))
    print(f"wrote {OUT/'audit.json'} ({report['wall_seconds']:.1f}s)")
    return report


def markdown_table(report: dict) -> str:
    lines = [
        "# euler_curved 收敛审计（程序化生成，勿手改）",
        "",
        f"final_time = {report['final_time']}, cfl = {report['cfl']}, "
        f"degree = {report['degree']}, meshes = {report['meshes']}, "
        f"warp amplitude = {report['amplitude']}",
        "",
        "| strength | family | 量 | " + " | ".join(str(m) for m in report["meshes"])
        + " | 相邻阶 |",
        "|---|---|---|---|---|---|",
    ]
    for tag, block in report["blocks"].items():
        strength = tag.split("_")[1]
        for family in ("straight", "warped"):
            data = block[family]
            lines.append(
                f"| {strength} | {family} | 投影 t=0 | "
                + " | ".join(f"{r['worst']:.3e}" for r in data["projection"])
                + " | " + ", ".join(f"{o:.2f}" if o else "—"
                                    for o in data["projection_orders"]) + " |")
            lines.append(
                f"| {strength} | {family} | 演化 t=0.1 | "
                + " | ".join("BLOWUP" if r.get("blown_up") else f"{r['worst']:.3e}"
                             for r in data["evolve"])
                + " | " + ", ".join(f"{o:.2f}" if o else "—"
                                    for o in data["evolve_orders"]) + " |")
            lines.append(
                f"| {strength} | {family} | 固定 dt=1e-3 | "
                + " | ".join(f"{r['worst']:.3e}" if not r.get("blown_up") else "BLOWUP"
                             for r in data["fixed_dt_1e-3"])
                + " | — |")
        lines.append(
            f"| {strength} | cartesian(euler_dg_2d) | 演化 t=0.1 | "
            + " | ".join(f"{r['worst']:.3e}" for r in block["cartesian_reference"])
            + " | " + ", ".join(f"{o:.2f}" if o else "—"
                                for o in block["cartesian_orders"]) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()