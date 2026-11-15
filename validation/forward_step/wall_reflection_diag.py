"""Diagnose G3 of ``benchmarks.forward_facing_step``: shock reflection off an
*embedded* solid wall versus the same reflection off a domain-boundary wall.

Motivation
----------
``docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md`` §4(a) records G3 as an honest
FAIL with the physical attribution *undetermined*: either the embedded-wall flux
is wrong, or the near-wall resolution / limiter is at fault.  This script makes
that attribution a measurement instead of a guess by running **two routes with
the same physics, the same mesh spacing and the same limiter**:

``embedded``
    the wall is an internal solid block (``solid=`` mask, the capability added in
    ``docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md``);

``outer``
    the domain stops at the wall and the wall is the ``right`` boundary
    (``bcs={"right": "wall"}``), the treatment that
    ``benchmarks/euler_dg_2d_wall_bc.py`` already verifies.

If the two routes agree to round-off on the shared fluid cells, the embedded-wall
flux is not the suspect and any remaining error against the exact two-shock
solution is a discretisation property (resolution / limiter) shared by both.
If they disagree, the difference *is* the defect, localised to the wall.

Both routes use ``benchmarks.euler_dg_2d_wall_bc.WallReflection`` as the exact
reference (incident shock from Rankine-Hugoniot, reflected shock from the jump
conditions), so nothing here introduces a new analytic model.

Usage
-----
    python validation/forward_step/wall_reflection_diag.py \
        --ncells 60 --degree 1 --limiter tvb --extra-time 0.06 \
        --out validation/forward_step/diag_n60_p1.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.euler_dg_2d_wall_bc import WallReflection  # noqa: E402
from tensordg.euler_dg_2d import (  # noqa: E402
    _pressure as pressure_from_conservative,
    solve_euler_dg_2d,
)

XMAX = 1.2           # length of the embedded-wall box (gate geometry)
YMAX = 0.1
X_WALL = 1.0         # analytic wall position, snapped onto a cell face
X_SHOCK = 0.4
MACH = 2.0


def _snap(value: float, dx: float) -> float:
    return round(value / dx) * dx


def _exact(wr: WallReflection, x: torch.Tensor, t: float):
    rho, u, p = wr.state(x.numpy(), t)
    return torch.as_tensor(rho), torch.as_tensor(u), torch.as_tensor(p)


def _initial(wr: WallReflection):
    pre_rho, pre_u, pre_p = wr.pre
    post_rho, post_u, post_p = wr.post

    def initial(x, y):
        ahead = x < wr.x_zero
        return (
            torch.where(ahead, torch.full_like(x, pre_rho), torch.full_like(x, post_rho)),
            torch.where(ahead, torch.full_like(x, pre_u), torch.full_like(x, post_u)),
            torch.zeros_like(x),
            torch.where(ahead, torch.full_like(x, pre_p), torch.full_like(x, post_p)),
        )

    return initial


def _summary(name, route, x, coeff, meta, wr, t_end, dx, extra):
    avg = coeff[..., 0, 0, :]
    rho = avg[..., 0]
    p = pressure_from_conservative(avg)
    out = {
        "route": route,
        "ncells": list(name),
        "dx": dx,
        "steps": meta["steps"],
        "final_time": t_end,
        "wall_time_exact": wr.wall_time,
        "reflected_front_exact": wr.front(t_end),
        "reflected_rho_exact": float(wr.reflected[0]),
        "reflected_p_exact": float(wr.reflected[2]),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "p_min": float(p.min()),
        "p_max": float(p.max()),
        "mass_relative_residual": abs(meta["conservation_residual_total"]["mass"]["relative"]),
        "energy_relative_residual": abs(
            meta["conservation_residual_total"]["energy"]["relative"]),
        "profile": {
            "x": [float(v) for v in x],
            "rho": [float(v) for v in rho[:, 0]],
            "p": [float(v) for v in p[:, 0]],
        },
    }
    out.update(extra)
    return out


def run_embedded(ncells, degree, limiter, tvb, extra_time, x_wall=X_WALL):
    """Wall as an interior solid block inside the box ``[0, XMAX]``."""
    nx, ny = ncells
    dx = XMAX / nx
    x_wall_face = _snap(x_wall, dx)
    wr = WallReflection(x_shock=_snap(X_SHOCK, dx), x_wall=x_wall_face, incident_mach=MACH)
    xc = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    # A cell is solid when its *left face* is at or beyond the wall face, so the
    # last fluid cell ends exactly on ``x_wall_face``.
    solid = torch.zeros(nx, ny, dtype=torch.bool)
    solid[xc >= x_wall_face + 0.5 * dx - 1e-12] = True
    fluid = ~solid[:, 0]
    t_end = wr.wall_time + extra_time
    started = time.time()
    x, y, coeff, meta = solve_euler_dg_2d(
        _initial(wr), ncells=ncells, degree=degree, final_time=t_end, cfl=0.12,
        domain=((0.0, XMAX), (0.0, YMAX)), periodic=(False, False), solid=solid,
        limiter=limiter, tvb_constant=tvb, limiter_stage="final",
        track_boundary_flux=True,
        bcs={"left": "outflow", "right": "wall", "bottom": "wall", "top": "wall"},
    )
    elapsed = time.time() - started
    wall_face_actual = float(xc[int(torch.nonzero(solid[:, 0]).flatten()[0])] - 0.5 * dx)
    return _summary(ncells, "embedded", x[fluid], coeff[fluid], meta, wr, t_end, dx,
                    {"wall_face_actual": wall_face_actual,
                     "x_shock_face": wr.x_zero, "seconds": elapsed,
                     "solid_cells": int(solid[:, 0].sum())})


def run_outer(ncells, degree, limiter, tvb, extra_time, x_wall=X_WALL):
    """Wall as the domain's right boundary; same dx as the embedded route."""
    nx, ny = ncells
    dx = XMAX / nx
    x_wall_face = _snap(x_wall, dx)
    n_fluid = int(round(x_wall_face / dx))  # fluid cells that fit left of the wall
    wr = WallReflection(x_shock=_snap(X_SHOCK, dx), x_wall=x_wall_face, incident_mach=MACH)
    ncells = (n_fluid, ny)
    t_end = wr.wall_time + extra_time
    started = time.time()
    x, y, coeff, meta = solve_euler_dg_2d(
        _initial(wr), ncells=ncells, degree=degree, final_time=t_end, cfl=0.12,
        domain=((0.0, x_wall_face), (0.0, YMAX)), periodic=(False, False),
        limiter=limiter, tvb_constant=tvb, limiter_stage="final",
        track_boundary_flux=True,
        bcs={"left": "outflow", "right": "wall", "bottom": "wall", "top": "wall"},
    )
    elapsed = time.time() - started
    return _summary(ncells, "outer", x, coeff, meta, wr, t_end, dx,
                    {"wall_face_actual": x_wall_face, "x_shock_face": wr.x_zero,
                     "seconds": elapsed, "solid_cells": 0})


def plateau_stats(run, wr, front_buffer=3.0, wall_buffer=1.0):
    """Exact-reference error on cell averages buffered away from both waves."""
    x = torch.as_tensor(run["profile"]["x"])
    rho = torch.as_tensor(run["profile"]["rho"])
    p = torch.as_tensor(run["profile"]["p"])
    dx = run["dx"]
    t_end = run["final_time"]
    front = wr.front(t_end)
    mask = (x > front + front_buffer * dx) & (x < run["wall_face_actual"] - wall_buffer * dx)
    rho5 = run["reflected_rho_exact"]
    p5 = run["reflected_p_exact"]
    stats = {"plateau_cells": int(mask.sum()),
             "plateau_width_cells": (run["wall_face_actual"] - front) / dx}
    if mask.any():
        stats["plateau_rho_error_percent"] = 100.0 * (float(rho[mask].mean()) / rho5 - 1.0)
        stats["plateau_p_error_percent"] = 100.0 * (float(p[mask].mean()) / p5 - 1.0)
        stats["worst_cell_error_percent"] = float(
            torch.maximum((rho[mask] / rho5 - 1.0).abs(), (p[mask] / p5 - 1.0).abs()).max() * 100.0)
    # whole-domain norms against the exact solution (no sampling band)
    xall = x
    exact_rho, _, exact_p = _exact(wr, xall, t_end)
    for name, got, want in (("density", rho, exact_rho), ("pressure", p, exact_p)):
        diff = (got - want).abs()
        scale = float(want.abs().mean())
        stats[f"l1_{name}"] = float(diff.mean()) / scale
        stats[f"linf_{name}"] = float(diff.max()) / scale
    return stats


def compare(embedded, outer):
    """Max |difference| between the two routes on the fluid cells they share."""
    x_e = torch.as_tensor(embedded["profile"]["x"])
    x_o = torch.as_tensor(outer["profile"]["x"])
    # share the cells of the shorter (outer) route, cell-centred comparison
    n = min(x_e.numel(), x_o.numel())
    rho_e = torch.as_tensor(embedded["profile"]["rho"][:n])
    rho_o = torch.as_tensor(outer["profile"]["rho"][:n])
    p_e = torch.as_tensor(embedded["profile"]["p"][:n])
    p_o = torch.as_tensor(outer["profile"]["p"][:n])
    x_off = float((x_e[:n] - x_o[:n]).abs().max())
    return {
        "cells_compared": n,
        "max_cell_centre_offset": x_off,
        "max_abs_drho": float((rho_e - rho_o).abs().max()),
        "max_abs_dp": float((p_e - p_o).abs().max()),
        "max_rel_drho": float(((rho_e - rho_o).abs() / rho_o.abs()).max()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncells", type=int, default=60)
    ap.add_argument("--ny", type=int, default=1)
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--limiter", default="tvb")
    ap.add_argument("--tvb-constant", type=float, default=1.0)
    ap.add_argument("--extra-time", type=float, default=0.06)
    ap.add_argument("--route", choices=("embedded", "outer", "both"), default="both")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ncells = (args.ncells, args.ny)
    kw = dict(degree=args.degree, limiter=(None if args.limiter in ("", "none", "None")
                                          else args.limiter),
              tvb=args.tvb_constant, extra_time=args.extra_time)
    payload = {"settings": {"ncells": list(ncells), **{k: v for k, v in kw.items()}}}
    if args.route in ("embedded", "both"):
        t = time.time()
        payload["embedded"] = run_embedded(ncells, **kw)
        payload["embedded"]["seconds"] = time.time() - t
    if args.route in ("outer", "both"):
        t = time.time()
        payload["outer"] = run_outer(ncells, **kw)
        payload["outer"]["seconds"] = time.time() - t
    wr = WallReflection(x_shock=_snap(X_SHOCK, XMAX / args.ncells),
                        x_wall=_snap(X_WALL, XMAX / args.ncells), incident_mach=MACH)
    for route in ("embedded", "outer"):
        if route in payload:
            payload[route].update(plateau_stats(payload[route], wr))
    if "embedded" in payload and "outer" in payload:
        payload["routes_agree"] = compare(payload["embedded"], payload["outer"])
    text = json.dumps(payload, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "profile"}
                      for k, v in payload.items() if isinstance(v, dict)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())