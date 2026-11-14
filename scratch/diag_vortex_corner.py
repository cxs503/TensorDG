"""Diagnostic 2: locate the t=2 -> t=5 error jump and test the corner hypothesis.

The vortex centre travels along (u_inf, v_inf)*t.  With u_inf = v_inf = 1 and a
[-5,5]^2 box, the core sits on the periodic corner at t = 5 (and is back at the
origin at t = 10).  If the jump is a corner/seam event, then:
  * the error curve must jump between t = 2 and t = 5 and stay high afterwards
    (permanent damage), i.e. t = 6.5 / 7.0 must NOT return to the t = 2 level;
  * a *streamwise-only* background (1,0) crosses only one seam pair and must
    stay much cleaner at the same t.
If instead the error is a monotone in-time truncation accumulation, the curve
sits on a smooth power law and the (1,0) case matches (1,1).
"""
import json
import math

import torch

from benchmarks.case_euler_vortex_errconserve import FIELDS, run_dg

torch.set_num_threads(8)


def pmax(r):
    return 100.0 * max(r["errors"][n] for n in FIELDS)


def cfg(cells, t, bg=None):
    c = {"task": "dg", "degree": 2, "cells": cells, "final_time": t, "cfl": 0.2}
    if bg is not None:
        c["background_velocity"] = bg
    return c


def main():
    out = {}

    # ---- D: map the growth curve at n = 32, diagonal background (1,1)
    curve = []
    for t in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 10.0):
        r = run_dg(cfg(32, t))
        curve.append({
            "t": t,
            "max_percent": pmax(r),
            "steps": r["steps"],
            "rho_error_percent": 100.0 * r["errors"]["density"],
            "wall_s": round(r["wall_seconds"], 2),
        })
        print(json.dumps(curve[-1]), flush=True)
    out["D_curve_bg11_n32"] = curve

    # local growth exponents between consecutive interior checkpoints
    exps = []
    for a, b in zip(curve, curve[1:]):
        if a["max_percent"] > 0 and b["max_percent"] > 0:
            exps.append({
                "window": [a["t"], b["t"]],
                "alpha_local": math.log(b["max_percent"] / a["max_percent"])
                / math.log(b["t"] / a["t"]),
            })
    out["D_local_exponents"] = exps

    # ---- E: streamwise-only background (1,0): seam crossing but no corner
    bg10 = []
    for t in (5.0, 10.0):
        r = run_dg(cfg(32, t, bg=(1.0, 0.0)))
        bg10.append({"t": t, "max_percent": pmax(r), "steps": r["steps"],
                     "wall_s": round(r["wall_seconds"], 2)})
        print(json.dumps(bg10[-1]), flush=True)
    out["E_bg10_n32"] = bg10

    # ---- F: does the t=5 error depend on the corner being *occupied*?
    # background (1,0) puts the core on the middle of the right seam at t = 5;
    # background (1,1) puts it on the corner at t = 5.  Compare n = 16 so the
    # corner is resolved by the same relative cell count.
    out["F_corner_vs_seam_n16"] = {}
    for label, bg in (("diag_1_1_corner", (1.0, 1.0)), ("stream_1_0_seam", (1.0, 0.0))):
        r = run_dg(cfg(16, 5.0, bg=bg))
        out["F_corner_vs_seam_n16"][label] = {
            "max_percent": pmax(r), "steps": r["steps"],
        }
        print(label, out["F_corner_vs_seam_n16"][label], flush=True)

    print("RESULT_JSON " + json.dumps(out, default=float))


if __name__ == "__main__":
    main()