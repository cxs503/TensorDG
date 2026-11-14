"""Diagnostic 1: filter activity, h-convergence, and error localisation.

Read-only w.r.t. the library (only monkey-patches the module global
``_positivity_filter`` in this process to count repair activations).
"""
import json
import math

import torch

import tensordg.euler_dg_2d as dg2
from benchmarks.case_euler_vortex_errconserve import (
    BACKGROUND, DOMAIN, FIELDS, STRENGTH, _operators, run_dg,
)
from benchmarks.verify_euler_dg_2d import _reconstruct
from tensordg.euler_2d import vortex_primitive

torch.set_num_threads(8)

FILTER = {"calls": 0, "steps_with_repair": 0, "cells_repaired": 0}
_orig_filter = dg2._positivity_filter


def counting_filter(coeff, basis, signs, iterations=10):
    ok = dg2._admissible_cells(coeff, basis, signs)
    FILTER["calls"] += 1
    if not bool(ok.all()):
        FILTER["steps_with_repair"] += 1
        FILTER["cells_repaired"] += int((~ok).sum())
    return _orig_filter(coeff, basis, signs, iterations)


dg2._positivity_filter = counting_filter


def cfg(cells, degree, t):
    return {"task": "dg", "degree": degree, "cells": cells, "final_time": t, "cfl": 0.2}


def pmax(r):
    return 100.0 * max(r["errors"][n] for n in FIELDS)


def main():
    out = {}

    # ---- A: positivity-filter activity on smooth data at the worst checkpoint
    FILTER.update(calls=0, steps_with_repair=0, cells_repaired=0)
    r = run_dg(cfg(32, 2, 10.0))
    out["A_filter_t10_n32"] = {"filter": dict(FILTER), "max_percent": pmax(r),
                               "steps": r["steps"]}
    print(json.dumps(out["A_filter_t10_n32"]), flush=True)
    dg2._positivity_filter = _orig_filter  # back to the untouched code path

    # ---- B: h-convergence of the DG-P2 family at t=1 and at t=10
    ladder = {}
    for t in (1.0, 10.0):
        rows = []
        for n in (8, 16, 32):
            rr = run_dg(cfg(n, 2, t))
            rows.append({"cells": n, "max_percent": pmax(rr),
                         "errors": rr["errors"], "steps": rr["steps"],
                         "wall_s": round(rr["wall_seconds"], 2)})
            print("t=%g n=%d %.6g" % (t, n, rows[-1]["max_percent"]), flush=True)
        ladder["t%g" % t] = rows
    out["B_h_convergence"] = ladder
    out["B_observed_order"] = {
        key: [round(math.log(a["max_percent"] / b["max_percent"]) / math.log(2.0), 3)
              for a, b in zip(rows, rows[1:])]
        for key, rows in ladder.items()
    }

    # ---- C: where does the error live at t=10, n=32?
    _x, _y, coeff, meta = dg2.solve_euler_dg_2d_vortex(
        ncells=32, degree=2, final_time=10.0, cfl=0.2, domain=DOMAIN,
    )
    nodes, weights, basis, _, _ = _operators(2)
    nx = 32
    state = _reconstruct(coeff, basis)                 # (nx, ny, nq, nq, 4)
    (xmin, xmax), _ = DOMAIN
    dx = (xmax - xmin) / nx
    xs = xmin + (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    X = xs[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    Y = xs[None, :, None, None] + 0.5 * dx * nodes[None, None, None, :]
    nq = len(nodes)
    rho_n = state[..., 0]
    rho_e, _ux, _uy, _p = vortex_primitive(X, Y, 10.0, STRENGTH, BACKGROUND)
    rho_e = rho_e.expand(nx, nx, nq, nq)
    err = (rho_n - rho_e).abs()
    R = torch.sqrt(X.expand(nx, nx, nq, nq) ** 2 + Y.expand(nx, nx, nq, nq) ** 2)
    core = R < 2.0
    corner = (X.expand(nx, nx, nq, nq).abs() > 4.0) & (Y.expand(nx, nx, nq, nq).abs() > 4.0)
    tot = float((err ** 2).sum())
    out["C_error_localisation_t10_n32"] = {
        "l2_fraction_core_R<2": float((err[core] ** 2).sum()) / tot,
        "l2_fraction_corner_absxy>4": float((err[corner] ** 2).sum()) / tot,
        "area_fraction_core": float(core.sum()) / core.numel(),
        "area_fraction_corner": float(corner.sum()) / corner.numel(),
        "max_abs_err_core": float(err[core].max()),
        "max_abs_err_corner": float(err[corner].max()),
        "rho_min_numeric": float(rho_n.min()),
        "rho_min_exact": float(rho_e.min()),
    }
    print("RESULT_JSON " + json.dumps(out, default=float))


if __name__ == "__main__":
    main()