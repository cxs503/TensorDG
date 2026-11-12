"""Validation driver: can the production DG scheme hold a *transcritical* steady
state (subcritical -> critical at the crest -> supercritical -> stationary jump)?

Case anchor: CASE-DG-TRANSCRITICAL-BUMP-SHOCK (new, see
``docs/CASE-DG-TRANSCRITICAL-BUMP-SHOCK.md``)

Reference case: Goutal & Maurel (1997) test 3 / Delestre et al. (2013) SWASHES
"Bump with shock": ``z_b = 0.2 (1 - ((x-10)/2)^2)`` on ``[8,12]``,
``q = 0.18 m^2/s``, ``h(25) = 0.33 m``.  The exact solution is constructed here
from Bernoulli + Rankine-Hugoniot (see ``tensordg.swe_steady_bump``) and is
verified independently of the branch inversion in
``tests/test_swe_steady_bump_shock.py``.

Two measurements:

``defect``
    Semi-discrete residual ``||dU/dt||/||U||`` of the *production* operator
    evaluated on the interpolant of the exact state, versus mesh size -- the
    same measurement as the subcritical study
    (``validation/swe_steady_bump/``), which is re-run here for contrast.

``march``
    Time march started *at* the exact state with the exact boundary fluxes.
    Reports how long the run survives, whether it aborts (``dt`` underflow when
    a nodal depth reaches the dry floor), and the depth error at that moment.

Run::

    PYTHONPATH=. python validation/swe_steady_bump_transcritical/run_transcritical_bump.py
    PYTHONPATH=. python validation/swe_steady_bump_transcritical/run_transcritical_bump.py --quick
\end_data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import warnings

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordg import swe_1d
from tensordg.swe_steady_bump import (
    BUMP_HEIGHT,
    BUMP_SHOCK_DEPTH_OUT,
    BUMP_SHOCK_DISCHARGE,
    DISCHARGE,
    DEPTH_OUT,
    DOMAIN,
    GRAVITY,
    bump_bed,
    bump_steady_state,
    bump_transcritical_state,
    steady_state_defect,
)

OUT = os.path.dirname(os.path.abspath(__file__))
Q = BUMP_SHOCK_DISCHARGE
H_OUT = BUMP_SHOCK_DEPTH_OUT
#: predeclared tolerances / knobs (fixed before any number is looked at)
DEFECT_N = (50, 100, 200, 400, 800)
DEFECT_P = (1, 2)
MARCH_N = (100, 200, 400)
MARCH_P = (1, 2)
MARCH_LIMITERS = ((None, None), ("tvb", 1e-6))
MARCH_TMAX = 400.0            # physical time to march if the run survives
CFL = 0.4
SHOCK_BAND_CELLS = 2.0        # post-processing band, set from the analytic x_s only
DEPTH_TOL = 0.05              # 5% depth tolerance away from the jump


# ---------------------------------------------------------------------------
# exact state accessors
# ---------------------------------------------------------------------------
def trans_h(x):
    return bump_transcritical_state(x)["h"]


def trans_u(x):
    return bump_transcritical_state(x)["u"]


def sub_h(x):
    return bump_steady_state(x)["h"]


def sub_u(x):
    return bump_steady_state(x)["u"]


def boundary_flux(state, index):
    h = float(state["h"][index])
    u = float(state["u"][index])
    flux = torch.tensor([h * u, h * u * u + 0.5 * GRAVITY * h * h], dtype=torch.float64)

    def boundary(time: float) -> torch.Tensor:
        return flux

    return boundary


# ---------------------------------------------------------------------------
# A. semi-discrete steady-state defect
# ---------------------------------------------------------------------------
def defect_ladder(ncells, degrees, which):
    h_fn, u_fn = (trans_h, trans_u) if which == "transcritical" else (sub_h, sub_u)
    rows = []
    for p in degrees:
        runs = []
        for n in ncells:
            t0 = time.time()
            d = steady_state_defect(h_fn, u_fn, bump_bed, ncells=n, degree=p,
                                    well_balanced=True)
            d["wall_s"] = time.time() - t0
            runs.append(d)
        orders = [math.log(runs[i]["defect_hu"] / runs[i + 1]["defect_hu"]) / math.log(2.0)
                  for i in range(len(runs) - 1)]
        ratios = [runs[i + 1]["defect_hu"] / runs[i]["defect_hu"] for i in range(len(runs) - 1)]
        rows.append({"regime": which, "degree": p, "runs": runs,
                     "orders": orders, "ratios": ratios})
        print("[defect] %-13s p=%d  defect_hu = %s  order = %s"
              % (which, p, " ".join("%.3e" % r["defect_hu"] for r in runs),
                 " ".join("%.2f" % o for o in orders)))
    return rows


# ---------------------------------------------------------------------------
# B. time march from the exact state
# ---------------------------------------------------------------------------
def march_run(ncells, degree, limiter, tvb_constant, tmax=MARCH_TMAX, cfl=CFL):
    state = bump_transcritical_state(torch.tensor([DOMAIN[0], DOMAIN[1]]))
    t0 = time.time()
    aborted = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x, coeff, zb_faces, meta = swe_1d.solve_swe_1d(
            trans_h, trans_u, bump_bed, ncells=ncells, degree=degree,
            final_time=tmax, domain=DOMAIN, cfl=cfl, rk_order=3,
            well_balanced=True,
            bc_left_flux=boundary_flux(state, 0), bc_right_flux=boundary_flux(state, 1),
            limiter=limiter, tvb_constant=(tvb_constant if tvb_constant else 1.0))
        aborted = any("stopped at" in str(w.message) for w in caught)
    wall = time.time() - t0
    h, u, _ = swe_1d.reconstruct_fields(coeff, zb_faces)
    h = h.mean(dim=-1) if h.dim() > 1 else h
    exact = bump_transcritical_state(x)
    err = (h - exact["h"]).abs()
    x_s = exact["x_shock"]
    dx = float(x[1] - x[0])
    away = (x - x_s).abs() > SHOCK_BAND_CELLS * dx
    row = {
        "ncells": ncells,
        "degree": degree,
        "limiter": limiter,
        "tvb_constant": tvb_constant,
        "requested_time": tmax,
        "time_reached": float(meta["time_reached"]),
        "aborted_dt_underflow": aborted,
        "steps": int(meta["steps"]),
        "wall_s": wall,
        "l1_rel_depth": float(err.sum() / exact["h"].abs().sum()),
        "linf_depth": float(err.max()),
        "linf_depth_x": float(x[err.argmax()]),
        "l1_rel_depth_away_from_jump": float(err[away].sum() / exact["h"][away].abs().sum()),
        "linf_depth_away_from_jump": float(err[away].max()) if bool(away.any()) else None,
        "depth_min": float(h.min()),
        "depth_max": float(h.max()),
        "discharge_spread": float((h * u).std() / (h * u).mean()),
        "within_5pct_away_from_jump": bool(float(err[away].max()) < DEPTH_TOL) if bool(away.any()) else None,
    }
    print("[march ] N=%4d p=%d lim=%-4s t_reached=%7.2f abort=%s steps=%5d "
          "L1=%.5f Linf=%.5f(minus jump %.5f) depth_min=%.4f"
          % (ncells, degree, limiter, row["time_reached"], aborted, row["steps"],
             row["l1_rel_depth"], row["linf_depth"], row["linf_depth_away_from_jump"],
             row["depth_min"]))
    return row


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(result):
    xg = torch.linspace(*DOMAIN, 801)
    exact = bump_transcritical_state(xg)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    ax = axes[0]
    ax.plot(xg, exact["eta"], "k-", lw=2, label="exact free surface $h+z_b$")
    ax.plot(xg, bump_bed(xg), "k--", lw=1, label="bed $z_b$")
    ax.plot(xg, exact["h"], "C0:", lw=1.5, label="exact depth $h$")
    ax.axvline(exact["x_shock"], color="C3", lw=1, ls="-.",
               label="stationary jump $x_s=%.3f$ m" % exact["x_shock"])
    ax.axhline(exact["critical_depth"], color="0.6", lw=0.8, ls=":",
               label="critical depth $h_c$")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("elevation / depth [m]")
    ax.set_title("transcritical bump: exact solution (Goutal & Maurel test 3)")
    ax.legend(fontsize=7)

    ax = axes[1]
    for row in result["defect"]:
        dx = [r["dx"] for r in row["runs"]]
        d = [r["defect_hu"] for r in row["runs"]]
        style = "o-" if row["regime"] == "transcritical" else "s--"
        ax.loglog(dx, d, style, label="%s, P%d" % (row["regime"], row["degree"]))
    sub = [r for r in result["defect"] if r["regime"] == "subcritical"][0]
    dxs = [r["dx"] for r in sub["runs"]]
    ax.loglog(dxs, [sub["runs"][0]["defect_hu"] * (dxs[0] / dx) for dx in dxs], "k:",
              lw=1, label="slope 1")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"$\|dU/dt\|/\|U\|$")
    ax.set_title("steady-state defect: smooth vs jumping steady state")
    ax.legend(fontsize=7)

    ax = axes[2]
    for row in result["march"]:
        x = torch.linspace(*DOMAIN, row["ncells"]) + 0.5 * (DOMAIN[1] - DOMAIN[0]) / row["ncells"]
        lab = "N=%d P%d lim=%s" % (row["ncells"], row["degree"], row["limiter"])
        ax.bar(lab, row["time_reached"], color="C0" if not row["aborted_dt_underflow"] else "C3")
    ax.set_ylabel("time reached before abort [s]")
    ax.set_title("time march from the exact transcritical state")
    ax.tick_params(axis="x", labelrotation=70, labelsize=6)

    fig.tight_layout()
    path = os.path.join(OUT, "fig_transcritical_bump.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=OUT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                     # pragma: no cover
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="defect ladder N=50..400, march N=100 only")
    ap.add_argument("--stage", choices=("all", "defect", "march"), default="all")
    ap.add_argument("--out", default=os.path.join(OUT, "results.json"))
    args = ap.parse_args(argv)
    torch.set_num_threads(1)

    defect_n = (50, 100, 200, 400) if args.quick else DEFECT_N
    march_n = (100,) if args.quick else MARCH_N
    march_p = (1,) if args.quick else MARCH_P

    st = bump_transcritical_state(torch.tensor([DOMAIN[0], 10.0, DOMAIN[1]]))
    result = {
        "case": "transcritical steady flow over a parabolic bump with a stationary jump",
        "reference": ("Goutal & Maurel (1997) 2nd Workshop on Dam-Break Wave Simulation, "
                      "test 3; Delestre et al. (2013) SWASHES 'Bump with shock'"),
        "parameters": {"discharge": Q, "depth_out": H_OUT, "bump_height": BUMP_HEIGHT,
                       "domain": list(DOMAIN), "gravity": GRAVITY,
                       "cfl": CFL, "rk_order": 3,
                       "shock_band_cells": SHOCK_BAND_CELLS, "depth_tol": DEPTH_TOL},
        "exact": {"x_shock": st["x_shock"], "head_left": st["head_left"],
                  "head_right": st["head_right"], "head_loss": st["head_loss"],
                  "critical_depth": st["critical_depth"],
                  "shock_depths": list(st["shock_depths"]),
                  "froude_left": st["froude_left"],
                  "froude_right": st["froude_right"],
                  "h_inlet": float(st["h"][0]), "h_outlet": float(st["h"][2]),
                  "depth_min_supercritical": float(st["shock_depths"][0])},
        "defect": [],
        "march": [],
        "git_commit": _git_commit(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print("exact: x_s=%.6f  h_L=%.5f h_R=%.5f  h_c=%.5f  loss=%.5f  Fr(L/R)=%.3f/%.3f"
          % (st["x_shock"], st["shock_depths"][0], st["shock_depths"][1],
             st["critical_depth"], st["head_loss"], st["froude_left"], st["froude_right"]))

    if args.stage in ("all", "defect"):
        result["defect"] += defect_ladder(defect_n, DEFECT_P, "transcritical")
        result["defect"] += defect_ladder((20, 40, 80, 160, 320), DEFECT_P, "subcritical")
    if args.stage in ("all", "march"):
        for n in march_n:
            for p in march_p:
                for limiter, tvb in MARCH_LIMITERS:
                    result["march"].append(march_run(n, p, limiter, tvb))

    if result["defect"] or result["march"]:
        result["figure"] = os.path.basename(make_figures(result))
        result["figure_sha256"] = _sha256(os.path.join(OUT, result["figure"]))
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print("wrote %s" % args.out)
    return result


if __name__ == "__main__":
    main()