#!/usr/bin/env python
"""2-D error-norm audit for CASE-DG-ADVECT-ORDER (m20097 cross-audit of m20026).

The m20026 deliverable computes its 2-D error as

    solver.l2_error(coeff, trig_profile, 1.0)

with ``trig_profile(x, y, t)`` of ``tensordg.advection_2d_var``.  That profile
*ignores* its time argument (``del t``), i.e. it is the **initial** field
``1 + sin(2 pi x) sin(2 pi y)``.  With ``a = (1, 0.5)`` and ``t = 1`` the exact
transported field is ``1 - sin(2 pi x) sin(2 pi y)``, so the "error" the tool
measures is ``|| u_h(T) - u(0) ||`` -- a quantity of O(1) that does not vanish
under refinement.

This script measures both variants on the same ladders the tool declares:

  * ``err_vs_initial``  -- exactly the tool's call (the harness defect);
  * ``err_vs_exact``    -- against the true transported analytic field.

Output: docs/board_audit_m20097/repro/repro_2d_metric.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tensordg.advection_2d_var import DGAdvection2D  # noqa: E402
from tensordg.convergence import convergence_table  # noqa: E402

OUT = ROOT / "docs" / "board_audit_m20097" / "repro"
LADDERS = {0: [8, 16, 32], 1: [8, 16, 32, 64], 2: [6, 12, 24], 3: [6, 12, 24]}
AX, AY = 1.0, 0.5


def trig(x, y, t=0.0):
    """Initial field u(x,y,0)."""
    return 1.0 + torch.sin(2.0 * math.pi * x) * torch.sin(2.0 * math.pi * y)


def exact_advected(x, y, t=0.0):
    """Exact solution of u_t + a.grad(u) = 0 for the constant velocity above."""
    return trig(x - AX * t, y - AY * t)


def vel(x, y, t):
    return torch.full_like(x, AX), torch.full_like(y, AY)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = {}
    for p, cells in LADDERS.items():
        e_init, e_exact, runs = [], [], []
        for n in cells:
            s = DGAdvection2D(p, n)
            c0 = s.project(trig)
            res = s.solve(c0, vel, 1.0, cfl=0.08)
            a = s.l2_error(res["coeff"], trig, 1.0)
            b = s.l2_error(res["coeff"], exact_advected, 1.0)
            e_init.append(a["absolute_l2"])
            e_exact.append(b["absolute_l2"])
            runs.append({"ncells": n, "steps": res["meta"]["steps"],
                         "mass_drift_rel": res["meta"]["mass_drift_rel"]})
            print(f"P{p} N={n:3d}: vs_initial={a['absolute_l2']:.6g} "
                  f"vs_exact={b['absolute_l2']:.6g} (rel={b['relative_l2']:.4g})")
        t_init = convergence_table(cells, e_init, theoretical_order=p + 1)
        t_exact = convergence_table(cells, e_exact, theoretical_order=p + 1)
        blocks[f"p{p}"] = {
            "cells": cells,
            "err_vs_initial": e_init,
            "order_vs_initial": t_init["observed_order_lsq"],
            "err_vs_exact": e_exact,
            "order_vs_exact": t_exact["observed_order_lsq"],
            "theoretical_order": p + 1,
            "runs": runs,
        }
    (OUT / "repro_2d_metric.json").write_text(json.dumps(
        {"anchor": "CASE-DG-ADVECT-ORDER", "velocity": [AX, AY], "final_time": 1.0,
         "note": "err_vs_initial reproduces the m20026 tool call; err_vs_exact is the "
                 "well-posed norm against the transported analytic field",
         "results": blocks}, indent=2, default=float))
    print("\nsummary (observed LSQ order):")
    for k, v in blocks.items():
        print(f"  {k}: vs_initial={v['order_vs_initial']:.3f} (err {v['err_vs_initial'][0]:.4g}"
              f"->{v['err_vs_initial'][-1]:.4g})  vs_exact={v['order_vs_exact']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())