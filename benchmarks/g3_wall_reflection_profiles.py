"""Density/pressure profiles of the embedded-wall reflection (figures for the report).

``benchmarks.g3_internal_wall_ladder`` answers *whether* the reflected plateau is
right; it only stores scalar metrics.  This driver stores the one-dimensional
fields at the final time so the report can plot the numerical solution against
the closed-form two-shock solution instead of asserting agreement in prose.

Geometry, data, CFL and the analytic reference are taken from the same helpers as
the gate (``wall_solid_mask`` / ``_snap_to_face`` / ``WallReflection``), so the
figure cannot drift away from the numbers it illustrates.

Usage::

    python -m benchmarks.g3_wall_reflection_profiles --index 0 --shards 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from benchmarks.euler_dg_2d_wall_bc import WallReflection
from benchmarks.forward_facing_step import (
    _pressure,
    _snap_to_face,
    wall_solid_mask,
)
from tensordg.euler_dg_2d import solve_euler_dg_2d

OUT = Path("validation/g3_internal_wall")
CONFIGS = [
    ("profile_solid_p1_nx240", dict(ncells=(240, 1), degree=1, extra_time=0.08)),
    ("profile_solid_p2_nx240", dict(ncells=(240, 1), degree=2, extra_time=0.08)),
    ("profile_solid_p1_nx120", dict(ncells=(120, 1), degree=1, extra_time=0.12)),
    ("profile_solid_p1_nx60", dict(ncells=(60, 1), degree=1, extra_time=0.20)),
]


def run_profile(ncells=(240, 1), degree=1, extra_time=0.08, mach=2.0, cfl=0.12,
                x_wall=1.0, x_shock=0.4, xmax=1.2, limiter=None, tvb_constant=1.0):
    nx, ny = ncells
    ymax = 0.1
    dx = xmax / nx
    x_wall_face, _ = _snap_to_face(x_wall, dx)
    x_shock_face, _ = _snap_to_face(x_shock, dx)
    wr = WallReflection(x_shock=x_shock_face, x_wall=x_wall_face, incident_mach=mach)
    xc = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    solid = wall_solid_mask(nx, ny, x_wall_face, dx)
    solid[xc >= x_wall_face + 0.5 * dx - 1e-12] = True
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

    t_end = wr.wall_time + extra_time
    started = time.time()
    x, _, coeff, meta = solve_euler_dg_2d(
        initial, ncells=ncells, degree=degree, final_time=t_end, cfl=cfl,
        domain=((0.0, xmax), (0.0, ymax)), periodic=(False, False), solid=solid,
        limiter=limiter, tvb_constant=tvb_constant, limiter_stage="final",
        track_boundary_flux=True,
        bcs={"left": "outflow", "right": "wall", "bottom": "wall", "top": "wall"},
    )
    avg = coeff[..., 0, 0, :]
    rho = avg[..., 0]
    pressure = _pressure(avg)
    fluid = ~solid[:, 0]
    exact_rho, _, exact_p = wr.state(x, t_end)
    return {
        "ncells": list(ncells), "degree": degree, "extra_time": extra_time,
        "limiter": limiter, "tvb_constant": tvb_constant,
        "dx": dx, "t_end": t_end, "wall_time": wr.wall_time,
        "steps": meta["steps"], "elapsed_seconds": time.time() - started,
        "x_wall": x_wall_face, "x_shock": x_shock_face,
        "reflected_front": wr.front(t_end),
        "reflected_state": {"rho": wr.reflected[0], "p": wr.reflected[2]},
        "post_state": {"rho": wr.post[0], "p": wr.post[2]},
        "x": [float(v) for v in xc[fluid]],
        "rho": [float(v) for v in rho[fluid, 0]],
        "p": [float(v) for v in pressure[fluid, 0]],
        "rho_exact": [float(v) for v in exact_rho[fluid].reshape(-1)],
        "p_exact": [float(v) for v in exact_p[fluid].reshape(-1)],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--shards", type=int, default=None)
    ap.add_argument("--out", default=str(OUT / "profiles.json"))
    args = ap.parse_args(argv)
    configs = CONFIGS
    if args.index is not None and args.shards:
        configs = [c for i, c in enumerate(configs) if i % args.shards == args.index]
    rows = []
    for name, kwargs in configs:
        row = run_profile(**kwargs)
        row["name"] = name
        rows.append(row)
        print(f"[{name}] steps={row['steps']} t={row['elapsed_seconds']:.1f}s "
              f"plateau rho={row['reflected_state']['rho']:.4f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.index is None else f".shard{args.index}of{args.shards}"
    path = out.with_name(out.stem + suffix + out.suffix)
    path.write_text(json.dumps({"rows": rows}, indent=1))
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":  # pragma: no cover
    main()