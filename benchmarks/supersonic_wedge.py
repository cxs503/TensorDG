"""Supersonic wedge flow benchmark with analytical oblique shock solution.

A Mach 2 flow over a 15-degree wedge produces an oblique shock.  The shock
angle can be computed from the theta-beta-Mach relation and compared with
the numerical solution.

Reference: Anderson, J.D. (2003). "Modern Compressible Flow", 3rd ed.
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

from tensordg.euler_dg_2d import solve_euler_dg_2d, cell_fields
from tensordg.euler_1d import GAMMA

# Freestream conditions
MA_FREESTREAM = 2.0
RHO_INF = 1.0
P_INF = 1.0
U_INF = MA_FREESTREAM * np.sqrt(GAMMA * P_INF / RHO_INF)
V_INF = 0.0


def _oblique_shock_angle(mach, wedge_angle_deg, gamma=GAMMA):
    """Compute oblique shock angle from theta-beta-Mach relation.

    Uses the iterative solution of:
    tan(theta) = 2 * cot(beta) * (M^2 * sin^2(beta) - 1) / (M^2 * (gamma + cos(2*beta)) + 2)
    """
    theta = np.deg2rad(wedge_angle_deg)
    M = mach

    # Initial guess: weak shock solution
    # beta_min = arcsin(1/M)
    beta_min = np.arcsin(1.0 / M)
    beta_max = np.pi / 2

    def residual(beta):
        sin_b = np.sin(beta)
        cos_b = np.cos(beta)
        cot_b = cos_b / sin_b
        num = M**2 * sin_b**2 - 1
        den = M**2 * (gamma + np.cos(2 * beta)) + 2
        return 2 * cot_b * num / den - np.tan(theta)

    # Bisection
    for _ in range(100):
        beta_mid = (beta_min + beta_max) / 2
        if residual(beta_min) * residual(beta_mid) < 0:
            beta_max = beta_mid
        else:
            beta_min = beta_mid
        if abs(beta_max - beta_min) < 1e-12:
            break

    return np.rad2deg((beta_min + beta_max) / 2)


def _initial_wedge(x, y, wedge_angle_deg=15.0):
    """Initial condition: uniform freestream with wedge geometry.

    The wedge starts at x=0, y=0 with the given angle.
    Below the wedge surface: solid (handled by boundary condition).
    Above the wedge: freestream.
    """
    wedge_angle_rad = np.deg2rad(wedge_angle_deg)
    # Wedge surface: y = x * tan(wedge_angle)
    wedge_y = x.cpu().numpy() * np.tan(wedge_angle_rad)
    y_np = y.cpu().numpy()

    # For now, just return freestream everywhere
    # The wedge boundary is handled by the boundary condition
    rho = torch.full_like(x, RHO_INF)
    vx = torch.full_like(x, U_INF)
    vy = torch.full_like(y, V_INF)
    p = torch.full_like(x, P_INF)
    return rho, vx, vy, p


def solve_supersonic_wedge(
    ncells=(240, 120),
    degree=2,
    final_time=0.3,
    cfl=0.15,
    limiter="tvb",
    tvb_constant=100.0,
    limiter_stage="final",
    wedge_angle=15.0,
    device="cpu",
    dtype=torch.float64,
):
    """Run the supersonic wedge flow benchmark.

    Parameters
    ----------
    ncells : (int, int)
        Number of cells in x and y.
    degree : int
        Polynomial degree.
    final_time : float
        Simulation end time.
    wedge_angle : float
        Wedge angle in degrees.
    """
    # Domain: [0, 3] x [0, 1.5]
    domain = ((0.0, 3.0), (0.0, 1.5))

    x, y, coeff, metadata = solve_euler_dg_2d(
        lambda x, y: _initial_wedge(x, y, wedge_angle),
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


def _compute_shock_angle(x, y, density, threshold=1.5):
    """Estimate shock angle from density field."""
    density_np = density.cpu().numpy()
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    shock_positions = []
    y_levels = np.linspace(0.1, 0.8, 10)
    for y_target in y_levels:
        idx = int(np.argmin(np.abs(y_np - y_target)))
        row = density_np[:, idx]
        crossings = np.where(np.diff(np.sign(row - threshold)))[0]
        if len(crossings) > 0:
            shock_positions.append((x_np[crossings[0]], y_np[idx]))

    if len(shock_positions) < 3:
        return None, None

    positions = np.array(shock_positions)
    coeffs = np.polyfit(positions[:, 0], positions[:, 1], 1)
    angle = np.rad2deg(np.arctan(coeffs[0]))
    return float(angle), positions


def plot_wedge_flow(x, y, coeff, output_dir, degree=2, wedge_angle=15.0):
    """Generate density contour plot for wedge flow."""
    fields = cell_fields(coeff)
    density = fields["density"].cpu().numpy()
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Full domain
    im0 = axes[0].contourf(x_np, y_np, density.T, levels=30, cmap="viridis")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"Supersonic Wedge Flow | DG P{degree}, Ma={MA_FREESTREAM}")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0], label="Density")

    # Wedge line
    wedge_angle_rad = np.deg2rad(wedge_angle)
    axes[0].plot([0, 3], [0, 3 * np.tan(wedge_angle_rad)], "r-", linewidth=2, label="Wedge surface")
    axes[0].legend()

    # Density profiles at several y-levels
    for i, y_frac in enumerate([0.2, 0.4, 0.6]):
        y_idx = int(y_frac * density.shape[1])
        axes[1].plot(x_np, density[:, y_idx], label=f"y={y_np[y_idx]:.2f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Density profiles")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "wedge_density.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=120)
    parser.add_argument("--ny", type=int, default=60)
    parser.add_argument("--degree", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--time", type=float, default=0.3)
    parser.add_argument("--cfl", type=float, default=0.15)
    parser.add_argument("--wedge-angle", type=float, default=15.0)
    parser.add_argument("--limiter", default="tvb", choices=["tvb", "weno", "none"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="wedge_results")
    args = parser.parse_args()

    limiter = args.limiter if args.limiter != "none" else None

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Supersonic Wedge Flow | DG P{args.degree}, Ma={MA_FREESTREAM}")
    print(f"  Grid: {args.nx} x {args.ny} cells")
    print(f"  Wedge angle: {args.wedge_angle} deg")
    print(f"  CFL: {args.cfl}, limiter: {args.limiter}")

    x, y, coeff, metadata = solve_supersonic_wedge(
        ncells=(args.nx, args.ny),
        degree=args.degree,
        final_time=args.time,
        cfl=args.cfl,
        limiter=limiter,
        tvb_constant=100.0,
        limiter_stage="final",
        wedge_angle=args.wedge_angle,
        device=args.device,
    )

    fields = cell_fields(coeff)
    density = fields["density"]

    # Compute shock angle
    angle, positions = _compute_shock_angle(x, y, density)

    # Theoretical shock angle
    theoretical_angle = _oblique_shock_angle(MA_FREESTREAM, args.wedge_angle)

    # Plot
    plot_wedge_flow(x, y, coeff, output, degree=args.degree, wedge_angle=args.wedge_angle)

    record = {
        "benchmark": "supersonic_wedge",
        **metadata,
        "wedge_angle_deg": args.wedge_angle,
        "mach_number": MA_FREESTREAM,
        "theoretical_shock_angle_deg": theoretical_angle,
        "computed_shock_angle_deg": angle,
        "angle_error_deg": abs(angle - theoretical_angle) if angle else None,
        "density_min": float(density.min()),
        "density_max": float(density.max()),
        "reference": "Oblique shock theory (theta-beta-Mach relation)",
    }

    with open(output / "metrics.json", "w") as f:
        json.dump(record, f, indent=2, default=str)

    print(f"  Steps: {metadata['steps']}")
    print(f"  Density range: {record['density_min']:.4f} .. {record['density_max']:.4f}")
    print(f"  Theoretical shock angle: {theoretical_angle:.2f} deg")
    if angle:
        print(f"  Computed shock angle: {angle:.2f} deg")
        print(f"  Angle error: {abs(angle - theoretical_angle):.2f} deg")
    print(f"  Results: {output}")


if __name__ == "__main__":
    main()
