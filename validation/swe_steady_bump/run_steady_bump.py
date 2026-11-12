"""Validation driver: is a *moving* steady state a discrete fixed point?

Runs the classical subcritical bump case (Goutal & Maurel 1997 / Garcia-Navarro
& Vazquez-Cendon 2000) through the production solver and reports

* the semi-discrete steady-state defect ``||dU/dt||/||U||`` versus resolution
  for three discretisation paths (well-balanced nodal; L2 projection with the
  analytic bed slope; L2 projection with the cell-difference slope), and
* the end-to-end drift of the time-marched solution started *at* the exact
  steady state, compared with the drift ``defect * t`` predicted a priori by the
  defect.

Everything is written to ``validation/swe_steady_bump/`` (numbers in
``results.json``, figures as PNG).  Run:

    PYTHONPATH=. python validation/swe_steady_bump/run_steady_bump.py
"""
from __future__ import annotations

import json
import math
import os

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordg import swe_1d
from tensordg.swe_steady_bump import (
    DISCHARGE,
    DEPTH_OUT,
    DOMAIN,
    bump_bed,
    bump_bed_slope,
    bump_boundary_flux,
    bump_steady_state,
    critical_depth,
    steady_state_defect,
)

OUT = os.path.dirname(os.path.abspath(__file__))
GRAVITY = 9.81


def _h_fn(x):
    return bump_steady_state(x)["h"]


def _u_fn(x):
    return bump_steady_state(x)["u"]


def _rel_l2(a, b):
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


def _order(coarse, fine, ratio=2.0):
    return math.log(coarse / fine) / math.log(ratio)


def defect_table(idxs, degrees):
    rows = []
    for path in ("well_balanced", "legacy_exact_slope", "legacy_face_slope"):
        for p in degrees:
            series = []
            for n in idxs:
                kwargs = dict(h_fn=_h_fn, u_fn=_u_fn, bed_fn=bump_bed,
                              ncells=n, degree=p)
                if path == "well_balanced":
                    d = steady_state_defect(**kwargs, well_balanced=True)
                elif path == "legacy_exact_slope":
                    d = steady_state_defect(**kwargs, well_balanced=False,
                                            zb_slope=bump_bed_slope)
                else:
                    d = steady_state_defect(**kwargs, well_balanced=False)
                series.append(d)
            orders = [
                _order(series[i]["defect_hu"], series[i + 1]["defect_hu"])
                for i in range(len(series) - 1)
            ]
            rows.append({"path": path, "degree": p, "runs": series, "orders": orders})
            print("%-19s p=%d  defect_hu = %s  orders = %s"
                  % (path, p,
                     " ".join("%.3e" % r["defect_hu"] for r in series),
                     " ".join("%.2f" % o for o in orders)))
    return rows


def drift_run(ncells, degree, final_time, sample_times):
    left = bump_boundary_flux("left")
    right = bump_boundary_flux("right")
    samples = []
    t_prev = 0.0
    for t in sample_times:
        x, coeff, zb_faces, meta = swe_1d.solve_swe_1d(
            _h_fn, _u_fn, bump_bed, ncells=ncells, degree=degree,
            final_time=t, domain=DOMAIN, cfl=0.4, rk_order=3,
            well_balanced=True, bc_left_flux=left, bc_right_flux=right)
        fields = swe_1d.reconstruct_fields(coeff, zb_faces)
        h, u = fields[0], fields[1]
        h_avg = h.mean(dim=-1) if h.dim() > 1 else h
        u_avg = u.mean(dim=-1) if u.dim() > 1 else u
        exact = bump_steady_state(x)
        samples.append({
            "time": t,
            "steps": meta["steps"],
            "dt_last": meta["dt"],
            "rel_l2_h": _rel_l2(h_avg, exact["h"]),
            "rel_l2_hu": _rel_l2(h_avg * u_avg, exact["h"] * exact["u"]),
            "hu_relative_spread": float((h_avg * u_avg).std() / (h_avg * u_avg).mean()),
            "time_reached": meta["time_reached"],
        })
        print("  drift N=%d p=%d t=%.1f steps=%d  rel_l2(h)=%.5f  spread(hu)=%.3e"
              % (ncells, degree, t, meta["steps"], samples[-1]["rel_l2_h"],
                 samples[-1]["hu_relative_spread"]))
        t_prev = t
    return {"ncells": ncells, "degree": degree, "samples": samples}


def make_figures(result):
    ex = bump_steady_state(torch.linspace(*DOMAIN, 401))
    xg = torch.linspace(*DOMAIN, 401)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(xg, ex["eta"], "k-", lw=2, label="exact free surface")
    ax.plot(xg, bump_bed(xg), "k--", lw=1, label="bed $z_b$")
    ax.plot(xg, ex["h"], "C0:", lw=1.5, label="exact depth $h$")
    for run in result["drift"]:
        final = run["samples"][-1]
        n = run["ncells"]
        p = run["degree"]
        x, coeff, zb_faces, _ = swe_1d.solve_swe_1d(
            _h_fn, _u_fn, bump_bed, ncells=n, degree=p,
            final_time=final["time"], domain=DOMAIN, cfl=0.4, rk_order=3,
            well_balanced=True, bc_left_flux=bump_boundary_flux("left"),
            bc_right_flux=bump_boundary_flux("right"))
        fields = swe_1d.reconstruct_fields(coeff, zb_faces)
        h = fields[0]
        h_avg = h.mean(dim=-1) if h.dim() > 1 else h
        ax.plot(x, h_avg + bump_bed(x), "C1-", lw=1.2,
                label=f"DG P{p} N={n}, t={final['time']:.0f}s")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("elevation [m]")
    ax.set_title("steady free surface over the bump")
    ax.legend(fontsize=7)

    ax = axes[1]
    for row in result["defect_rows"]:
        dx = [r["dx"] for r in row["runs"]]
        d = [r["defect_hu"] for r in row["runs"]]
        ax.loglog(dx, d, "o-", label=f"{row['path']}, P{row['degree']}")
    dxs = [r["dx"] for r in result["defect_rows"][0]["runs"]]
    ref = result["defect_rows"][0]["runs"][-1]["defect_hu"]
    for slope, style in ((1.0, "k:"), (2.0, "k--")):
        ax.loglog(dxs, [ref * (dxs[-1] / dx) ** slope for dx in dxs], style,
                  lw=1, label=f"slope {slope:g}")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"$\|dU/dt\|/\|U\|$")
    ax.set_title("semi-discrete steady-state defect")
    ax.legend(fontsize=7)

    ax = axes[2]
    for run in result["drift"]:
        ts = [s["time"] for s in run["samples"]]
        err = [s["rel_l2_h"] for s in run["samples"]]
        ax.loglog(ts, err, "o-", label=f"observed P{run['degree']} N={run['ncells']}")
        d0 = run["predicted_defect"]
        ax.loglog(ts, [d0 * t for t in ts], "k:", lw=1,
                  label=f"defect*t upper bound (P{run['degree']} N={run['ncells']})")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"relative $L^2$ error of $h$")
    ax.set_title("drift from the exact steady state")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_steady_bump.png"), dpi=130)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    ex = bump_steady_state(torch.tensor([0.0, 10.0, 25.0]))
    print("exact solution: H=%.6f h(crest)=%.6f h_c=%.6f subcritical margin=%.6f"
          % (ex["head"], float(ex["h"][1]), critical_depth(), ex["critical_margin"]))

    result = {
        "case": "subcritical steady flow over a parabolic bump",
        "reference": "Goutal & Maurel (1997); Garcia-Navarro & Vazquez-Cendon (2000)",
        "parameters": {"discharge": DISCHARGE, "depth_out": DEPTH_OUT,
                       "bump_height": 0.2, "bump_half_width": 2.0,
                       "domain": list(DOMAIN), "gravity": GRAVITY},
        "exact": {"head": ex["head"],
                  "h_crest": float(ex["h"][1]),
                  "critical_depth": critical_depth(),
                  "critical_margin": ex["critical_margin"],
                  "froude_max": ex["froude_max"]},
    }

    print("\n== semi-discrete steady-state defect ==")
    result["defect_rows"] = defect_table((25, 50, 100, 200, 400), (1, 2))

    print("\n== time-marched drift started from the exact steady state ==")
    result["drift"] = []
    for ncells, degree, tmax in ((50, 1, 4.0), (100, 1, 4.0)):
        run = drift_run(ncells, degree, tmax, (1.0, 2.0, 4.0))
        d = steady_state_defect(h_fn=_h_fn, u_fn=_u_fn, bed_fn=bump_bed,
                                ncells=ncells, degree=degree, well_balanced=True)
        run["predicted_defect"] = d["defect_hu"]
        run["defect_h"] = d["defect_h"]
        result["drift"].append(run)

    make_figures(result)
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print("\nwrote %s/results.json and fig_steady_bump.png" % OUT)


if __name__ == "__main__":
    main()