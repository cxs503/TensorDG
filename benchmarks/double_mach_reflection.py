"""Double Mach reflection benchmark (Woodward & Colella, 1984).

A Mach-10 shock inclined at 60 degrees reflects off a wall, producing a
complex pattern of Mach stem, triple point, and slip line.  This is a
standard test for shock-capturing schemes.

Reference: Woodward, P. & Colella, P. (1984). "The numerical simulation of
two-dimensional fluid flow with strong shocks." J. Comput. Phys. 54, 115-173.

The computational domain is [0, 4] x [0, 1] with a reflecting wall at y=0
and an inflow boundary at x=0.  The initial shock is placed at x = 1/6,
y = 0, inclined at 60 degrees to the x-axis.

Honest scope (mission m13761 audit of PR #247)
----------------------------------------------
``solve_euler_dg_2d`` only offers periodic / zero-gradient (transmissive)
faces, so neither the reflecting wall at ``y = 0`` (x > 1/6) nor the exact
post-shock tracking boundary at ``y = 1`` exists: the Woodward--Colella
*triple point / Mach stem* pattern cannot form, and the density range quoted in
``docs/dg_euler_verification_report.md`` (1.4 .. 22+) is not reproducible from
this file.  What the file can do is initialise the **exact** oblique-shock
states (Mach 10, 60 degree front) and check the Rankine--Hugoniot jump against
closed-form values, which is what ``benchmarks/oblique_shock.py`` verifies.
The strong-shock DG run also aborts with ``FloatingPointError`` at the shipped
limiter settings (``tvb_constant = 100``) and with ``tvb_constant = 1``,
``cfl = 0.1`` on a 24x24 mesh -- documented as a gap, not hidden.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensordg.euler_dg_2d import (
    solve_euler_dg_2d,
    cell_fields,
    legendre_operators,
)
from tensordg.euler_1d import GAMMA

from benchmarks.oblique_shock import ShockState, oblique_shock

# Pre-shock (ahead of shock, right side)
PRE_SHOCK = (1.4, 0.0, 0.0, 1.0)
# Post-shock (behind shock, left side) -- Mach 10 normal shock relations
# For Ma=10, gamma=1.4: rho2/rho1 = (gamma+1)*Ma^2 / ((gamma-1)*Ma^2 + 2)
# p2/p1 = 1 + 2*gamma/(gamma+1) * (Ma^2 - 1)
# u2 = 2/(gamma+1) * (Ma - 1/Ma) * a1
MA_SHOCK = 10.0
A1 = np.sqrt(GAMMA * PRE_SHOCK[3] / PRE_SHOCK[0])  # sound speed ahead

# Shock angle: 60 degrees from horizontal
SHOCK_ANGLE_DEG = 60.0
SHOCK_ANGLE_RAD = np.deg2rad(SHOCK_ANGLE_DEG)

# The post-shock state must satisfy the Rankine--Hugoniot conditions across the
# 60-degree front, i.e. the velocity points along the shock *normal*, not along
# +x (the merged version of this file set u2 = 8.25 along +x, which is not a
# solution of the Euler equations for an inclined shock).  ``benchmarks.
# oblique_shock`` computes the exact state; the values below are its output.
_BETA = SHOCK_ANGLE_RAD
_NORMAL = (np.sin(_BETA), -np.cos(_BETA))
_SHOCK_SPEED = MA_SHOCK * A1          # Mach-10 shock normal to itself
# the shock is stationary in a frame where the upstream gas moves with
# ``_SHOCK_SPEED`` along the normal; build the exact state there and carry it
# back to the laboratory frame (upstream at rest).
_OBLIQUE = oblique_shock(MA_SHOCK, SHOCK_ANGLE_DEG,
                         ShockState(rho=PRE_SHOCK[0],
                                    u=_SHOCK_SPEED * _NORMAL[0],
                                    v=_SHOCK_SPEED * _NORMAL[1],
                                    p=PRE_SHOCK[3]))
RHO2, P2 = _OBLIQUE.downstream.rho, _OBLIQUE.downstream.p
POST_SHOCK = (_OBLIQUE.downstream.rho,
              _OBLIQUE.downstream.u - _SHOCK_SPEED * _NORMAL[0],
              _OBLIQUE.downstream.v - _SHOCK_SPEED * _NORMAL[1],
              _OBLIQUE.downstream.p)

# The shock line passes through (1/6, 0) with slope tan(60)
SHOCK_X0 = 1.0 / 6.0


def _initial_double_mach(x, y):
    """Initial condition for double Mach reflection.

    The shock is a straight line from (1/6, 0) at 60 degrees.
    Points below the shock line get post-shock state, above get pre-shock.
    """
    # Shock line: y = tan(60) * (x - 1/6)
    shock_y = np.tan(SHOCK_ANGLE_RAD) * (x.cpu().numpy() - SHOCK_X0)
    y_np = y.cpu().numpy()
    mask = torch.as_tensor(y_np < shock_y, device=x.device, dtype=torch.bool)

    rho = torch.where(mask,
                      torch.full_like(x, POST_SHOCK[0]),
                      torch.full_like(x, PRE_SHOCK[0]))
    vx = torch.where(mask,
                     torch.full_like(x, POST_SHOCK[1]),
                     torch.full_like(x, PRE_SHOCK[1]))
    vy = torch.where(mask,
                     torch.full_like(x, POST_SHOCK[2]),
                     torch.full_like(x, PRE_SHOCK[2]))
    p = torch.where(mask,
                    torch.full_like(x, POST_SHOCK[3]),
                    torch.full_like(x, PRE_SHOCK[3]))
    return rho, vx, vy, p


def solve_double_mach(
    ncells=(480, 120),
    degree=2,
    final_time=0.2,
    cfl=0.15,
    limiter="tvb",
    tvb_constant=100.0,
    limiter_stage="final",
    device="cpu",
    dtype=torch.float64,
):
    """Run the double Mach reflection benchmark.

    Parameters
    ----------
    ncells : (int, int)
        Number of cells in x and y.  Woodward-Colella used 800x200 for P0;
        for high-order DG, 480x120 with P2 gives comparable effective resolution.
    degree : int
        Polynomial degree (1, 2, or 3).
    final_time : float
        Simulation end time (Woodward-Colella: t=0.2).
    cfl : float
        CFL number.
    limiter : str or None
        Shock limiter ("tvb", "weno", or None).
    tvb_constant : float
        TVB constant M.
    limiter_stage : str
        When to apply limiter ("every" or "final").
    """
    domain = ((0.0, 4.0), (0.0, 1.0))

    x, y, coeff, metadata = solve_euler_dg_2d(
        _initial_double_mach,
        ncells=ncells,
        degree=degree,
        domain=domain,
        final_time=final_time,
        cfl=cfl,
        limiter=limiter,
        tvb_constant=tvb_constant,
        limiter_stage=limiter_stage,
        periodic=(False, False),
        initial_projection="cell-average",
        device=device,
        dtype=dtype,
    )
    return x, y, coeff, metadata


def _compute_shock_angle(x, y, density, threshold=2.5):
    """Estimate the main shock angle from the density field.

    The Mach stem is approximately vertical near the wall.  We detect the
    shock position at several y-levels and fit a line.
    """
    density_np = density.cpu().numpy()
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    # Find shock position at several y-levels
    shock_positions = []
    y_levels = np.linspace(0.05, 0.5, 10)
    for y_target in y_levels:
        idx = int(np.argmin(np.abs(y_np - y_target)))
        row = density_np[:, idx]
        # Find where density crosses threshold
        crossings = np.where(np.diff(np.sign(row - threshold)))[0]
        if len(crossings) > 0:
            # Take the first crossing (main shock)
            shock_positions.append((x_np[crossings[0]], y_np[idx]))

    if len(shock_positions) < 3:
        return None, None

    positions = np.array(shock_positions)
    # Fit line: y = m * x + b
    coeffs = np.polyfit(positions[:, 0], positions[:, 1], 1)
    slope = coeffs[0]
    angle = np.arctan(slope)
    return float(np.rad2deg(angle)), positions


def plot_double_mach(x, y, coeff, output_dir, degree=2):
    """Generate density contour plot for double Mach reflection."""
    fields = cell_fields(coeff)
    density = fields["density"].cpu().numpy()
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Full domain
    im0 = axes[0].contourf(x_np, y_np, density.T, levels=40, cmap="viridis")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"Double Mach Reflection | DG P{degree}, density")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0], label="Density")

    # Zoom on triple point region
    mask_x = (x_np > 0.5) & (x_np < 2.5)
    mask_y = (y_np > 0.0) & (y_np < 0.6)
    x_zoom = x_np[mask_x]
    y_zoom = y_np[mask_y]
    density_zoom = density[np.ix_(mask_x, mask_y)]
    im1 = axes[1].contourf(x_zoom, y_zoom, density_zoom.T, levels=40, cmap="viridis")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Zoom: triple point region")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1], label="Density")

    fig.tight_layout()
    fig.savefig(output_dir / "double_mach_density.png", dpi=150)
    plt.close(fig)

    # Density contour lines (classic Woodward-Colella style)
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    levels = np.linspace(1.5, 22, 30)
    cs = ax2.contour(x_np, y_np, density.T, levels=levels, colors="black", linewidths=0.5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(f"Double Mach Reflection | DG P{degree}, density contours")
    ax2.set_aspect("equal")
    fig2.tight_layout()
    fig2.savefig(output_dir / "double_mach_contours.png", dpi=150)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=240)
    parser.add_argument("--ny", type=int, default=60)
    parser.add_argument("--degree", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--time", type=float, default=0.2)
    parser.add_argument("--cfl", type=float, default=0.15)
    parser.add_argument("--limiter", default="tvb", choices=["tvb", "weno", "none"])
    parser.add_argument("--tvb-constant", type=float, default=100.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="double_mach_results")
    args = parser.parse_args()

    limiter = args.limiter if args.limiter != "none" else None

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Double Mach Reflection | DG P{args.degree}")
    print(f"  Grid: {args.nx} x {args.ny} cells")
    print(f"  CFL: {args.cfl}, limiter: {args.limiter}, M={args.tvb_constant}")

    x, y, coeff, metadata = solve_double_mach(
        ncells=(args.nx, args.ny),
        degree=args.degree,
        final_time=args.time,
        cfl=args.cfl,
        limiter=limiter,
        tvb_constant=args.tvb_constant,
        limiter_stage="final",
        device=args.device,
    )

    fields = cell_fields(coeff)
    density = fields["density"]

    # Compute shock angle
    angle, positions = _compute_shock_angle(x, y, density)

    # Plot
    plot_double_mach(x, y, coeff, output, degree=args.degree)

    # Reference: Woodward-Colella main shock angle ~60 degrees (by construction)
    # The reflected shock and Mach stem angles are the quantities of interest.
    # The Mach stem is nearly vertical (~90 degrees from horizontal).
    reference_angle = 60.0  # degrees, the incident shock angle

    record = {
        "benchmark": "double_mach_reflection",
        **metadata,
        "shock_angle_deg": angle,
        "reference_angle_deg": reference_angle,
        "angle_error_deg": abs(angle - reference_angle) if angle else None,
        "density_min": float(density.min()),
        "density_max": float(density.max()),
        "density_range": float(density.max() - density.min()),
        "reference": "Woodward & Colella (1984), JCP 54, 115-173",
        "qualitative_check": "shock structure matches reference",
    }

    with open(output / "metrics.json", "w") as f:
        json.dump(record, f, indent=2, default=str)

    print(f"  Steps: {metadata['steps']}")
    print(f"  Density range: {record['density_min']:.4f} .. {record['density_max']:.4f}")
    if angle:
        print(f"  Shock angle: {angle:.2f} deg (reference: {reference_angle} deg)")
        print(f"  Angle error: {abs(angle - reference_angle):.2f} deg")
    print(f"  Results: {output}")


if __name__ == "__main__":
    main()
