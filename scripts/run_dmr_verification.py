"""Verification runner for the double Mach reflection (anchor REPRO-DG-CKS1998).

Runs the Woodward--Colella case with the standard boundary set, then measures the
undisturbed part of the incident front and compares it with the exact
Rankine--Hugoniot plane-front law ``x_front(y, t) = 1/6 + y/sqrt3 + W/sin60 * t``.

Every number in ``metrics.json`` is produced here; nothing is hand-copied.

Example
-------
    python scripts/run_dmr_verification.py --nx 160 --ny 40 --degree 2 \
        --time 0.2 --out validation/dmr/artifacts/p2_160x40
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordg.euler_dg_2d import cell_fields
from benchmarks.double_mach_reflection import (
    POST_SHOCK, PRE_SHOCK, SHOCK_SPEED, DENSITY_LEVEL, analytic_front_x,
    measure_primary_front, _compute_shock_angle, rh_residuals,
    solve_double_mach,
)

GAMMA = 1.4


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=160)
    ap.add_argument("--ny", type=int, default=40)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--time", type=float, default=0.2)
    ap.add_argument("--cfl", type=float, default=0.15)
    ap.add_argument("--limiter", default="tvb")
    ap.add_argument("--tvb-constant", type=float, default=100.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    limiter = None if args.limiter == "none" else args.limiter
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    x, y, coeff, meta = solve_double_mach(
        ncells=(args.nx, args.ny), degree=args.degree, final_time=args.time,
        cfl=args.cfl, limiter=limiter, tvb_constant=args.tvb_constant,
        limiter_stage="final",
    )
    wall = time.time() - t0
    fields = cell_fields(coeff)
    rho = fields["density"].cpu().numpy()
    ux = fields["velocity_x"].cpu().numpy()
    uy = fields["velocity_y"].cpu().numpy()
    p = fields["pressure"].cpu().numpy()
    xc = np.asarray(x).ravel()
    yc = np.asarray(y).ravel()

    front = measure_primary_front(rho, x, y)
    exact = {k: float(analytic_front_x(k, args.time)) for k in front}
    err = {k: front[k] - exact[k] for k in front}
    angle, pts = _compute_shock_angle(x, y, rho)

    # measured primitives a fixed distance behind the undisturbed front
    # sample well behind the numerical shock profile: 5 cells or 0.05, whichever
    # is larger (a 2-cell offset sits *inside* the smeared front and reads low)
    offset = max(5.0 * 4.0 / args.nx, 0.05)
    j = int(np.argmin(np.abs(yc - 0.85)))
    i = int(np.argmin(np.abs(xc - (float(analytic_front_x(0.85, args.time)) - offset))))
    sample = {"x": float(xc[i]), "y": float(yc[j]),
              "rho": float(rho[i, j]), "u": float(ux[i, j]),
              "v": float(uy[i, j]), "p": float(p[i, j])}
    state = {"rho": POST_SHOCK[0], "u": POST_SHOCK[1], "v": POST_SHOCK[2],
             "p": POST_SHOCK[3]}
    speed_s = float(np.hypot(sample["u"], sample["v"]))
    speed_e = float(np.hypot(state["u"], state["v"]))

    metrics = {
        "case": "double_mach_reflection",
        "anchor": "REPRO-DG-CKS1998",
        "grid": [args.nx, args.ny],
        "degree": args.degree,
        "final_time": args.time,
        "cfl": args.cfl,
        "limiter": args.limiter,
        "tvb_constant": args.tvb_constant,
        "steps": meta.get("steps"),
        "wall_seconds": wall,
        "dx": 4.0 / args.nx,
        "front_x": front,
        "front_x_exact": exact,
        "front_error": err,
        "front_error_cells": {k: v / (4.0 / args.nx) for k, v in err.items()},
        "front_angle_deg": angle,
        "front_angle_exact_deg": 60.0,
        "density_min": float(rho.min()),
        "density_max": float(rho.max()),
        "rh_residuals": {k: float(v) for k, v in
                         rh_residuals(PRE_SHOCK, POST_SHOCK).items()},
        "post_shock_sample_offset": offset,
        "post_shock_exact": state,
        "post_shock_measured": sample,
        "post_shock_rel_error": {
            "rho": (sample["rho"] - state["rho"]) / state["rho"],
            "p": (sample["p"] - state["p"]) / state["p"],
            "speed": (speed_s - speed_e) / speed_e,
        },
        "exact_front_speed_x": float(SHOCK_SPEED / np.sin(np.deg2rad(60.0))),
        "density_level": DENSITY_LEVEL,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # figure 1: density field (classic DMR picture)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    im = axes[0].pcolormesh(xc, yc, rho.T, cmap="viridis", shading="auto")
    axes[0].set_title(f"DG P{args.degree} {args.nx}x{args.ny}, t={args.time}")
    axes[0].set_aspect("equal")
    fig.colorbar(im, ax=axes[0], label="density")
    axes[1].contour(xc, yc, rho.T, levels=np.linspace(1.5, 22.0, 40),
                    colors="black", linewidths=0.35)
    axes[1].set_title("density contours (Woodward--Colella levels)")
    axes[1].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / "dmr_density.png", dpi=150)
    plt.close(fig)

    # figure 2: measured front vs exact plane-front law
    fig2, ax = plt.subplots(figsize=(6, 4.5))
    yy = np.linspace(0.0, 1.0, 100)
    ax.plot(analytic_front_x(yy, args.time), yy, "k-", lw=1.2,
            label="exact RH plane front")
    if pts is not None:
        ax.plot(pts[:, 0], pts[:, 1], "o", ms=4, label="measured (y=0.5..0.9)")
    worst = max(abs(v) for v in err.values()) / (4.0 / args.nx)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"incident front, t={args.time}: max |err| = {worst:.2f} cells")
    ax.set_aspect("equal")
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(out / "dmr_front_vs_exact.png", dpi=150)
    plt.close(fig2)

    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()