"""Field visualization for CASE-DG-EULER-VORTEX-ERR.

Generates:
- fig_density_dg_t1.png: DG-P2 density field at t=1
- fig_density_dg_t5.png: DG-P2 density field at t=5
- fig_density_p0_t1.png: P0/128 density field at t=1
- fig_density_p0_t5.png: P0/128 density field at t=5
- fig_density_error_dg_t5.png: DG-P2 density error field at t=5
- fig_density_error_p0_t5.png: P0/128 density error field at t=5
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensordg.euler_2d import (
    solve_isentropic_vortex,
    vortex_primitive,
    _cell_averages,
)
from tensordg.euler_dg_2d import solve_euler_dg_2d_vortex
from benchmarks.verify_euler_dg_2d import _reconstruct

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
STRENGTH = 5.0
BACKGROUND = (1.0, 1.0)
GAMMA = 1.4

torch.set_num_threads(4)
OUTPUT_DIR = Path("case_euler_vortex_errconserve")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_exact_density_on_grid(nx=200, ny=200, final_time=1.0):
    """Evaluate exact density on a uniform grid."""
    x = np.linspace(DOMAIN[0][0], DOMAIN[0][1], nx)
    y = np.linspace(DOMAIN[1][0], DOMAIN[1][1], ny)
    xx, yy = np.meshgrid(x, y)
    # Periodic shift
    period = (10.0, 10.0)
    xc = BACKGROUND[0] * final_time
    yc = BACKGROUND[1] * final_time
    # Shift coordinates back
    xx_shift = ((xx - xc - DOMAIN[0][0]) % period[0]) + DOMAIN[0][0]
    yy_shift = ((yy - yc - DOMAIN[1][0]) % period[1]) + DOMAIN[1][1]
    
    r2 = (xx_shift - 0)**2 + (yy_shift - 0)**2  # vortex centered at origin initially
    beta = STRENGTH
    rho = (1 - (beta**2 * GAMMA * np.exp(1 - r2)) / (8 * np.pi**2))**(1 / (GAMMA - 1))
    return x, y, rho


def plot_dg_field(degree=2, ncells=32, final_time=1.0, cfl=0.2, suffix=""):
    """Plot DG density field."""
    x_nodes, y_nodes, solution, meta = solve_euler_dg_2d_vortex(
        degree=degree, ncells=ncells, final_time=final_time, cfl=cfl, strength=STRENGTH
    )
    # solution: [nx, ny, qx, qy, n_fields]
    # Reconstruct to get density at quadrature points
    # For visualization, average over quadrature points
    density = solution[:, :, :, :, 0].mean(dim=(2, 3)).cpu().numpy()  # [nx, ny]
    
    # Get exact solution for comparison
    xe, ye, rho_exact = get_exact_density_on_grid(nx=200, ny=200, final_time=final_time)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Computed field
    im0 = axes[0].pcolormesh(
        np.linspace(DOMAIN[0][0], DOMAIN[0][1], ncells + 1),
        np.linspace(DOMAIN[1][0], DOMAIN[1][1], ncells + 1),
        density.T,
        shading='auto', cmap='viridis'
    )
    axes[0].set_title(f"DG P{degree} {ncells}x{ncells}, t={final_time}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], label="density")
    
    # Exact field
    im1 = axes[1].pcolormesh(xe, ye, rho_exact.T, shading='auto', cmap='viridis')
    axes[1].set_title(f"Exact solution, t={final_time}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], label="density")
    
    fig.suptitle(f"Isentropic vortex: density field (DG P{degree}, {ncells}x{ncells})")
    fig.tight_layout()
    fname = OUTPUT_DIR / f"fig_density_dg_t{int(final_time)}{suffix}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")
    return fname


def plot_p0_field(ncells=128, final_time=1.0, cfl=0.4, suffix=""):
    """Plot P0 density field."""
    x, y, fields, meta = solve_isentropic_vortex(
        ncells=ncells, final_time=final_time, cfl=cfl, strength=STRENGTH
    )
    # fields is a dict with 'conservative' tensor [4, ncells, ncells]
    conservative = fields["conservative"] if isinstance(fields, dict) else fields
    density = conservative[0].cpu().numpy()  # [ncells, ncells]
    
    # Get exact solution
    xe, ye, rho_exact = get_exact_density_on_grid(nx=200, ny=200, final_time=final_time)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Computed field
    im0 = axes[0].pcolormesh(
        np.linspace(DOMAIN[0][0], DOMAIN[0][1], ncells + 1),
        np.linspace(DOMAIN[1][0], DOMAIN[1][1], ncells + 1),
        density.T,
        shading='auto', cmap='viridis'
    )
    axes[0].set_title(f"P0 {ncells}x{ncells}, t={final_time}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], label="density")
    
    # Exact field
    im1 = axes[1].pcolormesh(xe, ye, rho_exact.T, shading='auto', cmap='viridis')
    axes[1].set_title(f"Exact solution, t={final_time}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], label="density")
    
    fig.suptitle(f"Isentropic vortex: density field (P0 FV, {ncells}x{ncells})")
    fig.tight_layout()
    fname = OUTPUT_DIR / f"fig_density_p0_t{int(final_time)}{suffix}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")
    return fname


def plot_error_field_dg(degree=2, ncells=32, final_time=5.0, cfl=0.2):
    """Plot DG density error field."""
    x_nodes, y_nodes, solution, meta = solve_euler_dg_2d_vortex(
        degree=degree, ncells=ncells, final_time=final_time, cfl=cfl, strength=STRENGTH
    )
    density = solution[:, :, :, :, 0].mean(dim=(2, 3)).cpu().numpy()
    
    # Exact on same grid
    xe, ye, rho_exact = get_exact_density_on_grid(nx=ncells, ny=ncells, final_time=final_time)
    error = np.abs(density - rho_exact)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(
        np.linspace(DOMAIN[0][0], DOMAIN[0][1], ncells + 1),
        np.linspace(DOMAIN[1][0], DOMAIN[1][1], ncells + 1),
        error.T,
        shading='auto', cmap='Reds'
    )
    ax.set_title(f"DG P{degree} |rho - rho_exact|, t={final_time}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="|error|")
    fig.tight_layout()
    fname = OUTPUT_DIR / f"fig_density_error_dg_t{int(final_time)}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")
    return fname


def plot_error_field_p0(ncells=128, final_time=5.0, cfl=0.4):
    """Plot P0 density error field."""
    x, y, fields, meta = solve_isentropic_vortex(
        ncells=ncells, final_time=final_time, cfl=cfl, strength=STRENGTH
    )
    conservative = fields["conservative"] if isinstance(fields, dict) else fields
    density = conservative[0].cpu().numpy()
    
    xe, ye, rho_exact = get_exact_density_on_grid(nx=ncells, ny=ncells, final_time=final_time)
    error = np.abs(density - rho_exact)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(
        np.linspace(DOMAIN[0][0], DOMAIN[0][1], ncells + 1),
        np.linspace(DOMAIN[1][0], DOMAIN[1][1], ncells + 1),
        error.T,
        shading='auto', cmap='Reds'
    )
    ax.set_title(f"P0 |rho - rho_exact|, t={final_time}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="|error|")
    fig.tight_layout()
    fname = OUTPUT_DIR / f"fig_density_error_p0_t{int(final_time)}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")
    return fname


def main():
    print("Generating DG field plots...")
    plot_dg_field(degree=2, ncells=32, final_time=1.0)
    plot_dg_field(degree=2, ncells=32, final_time=5.0)
    
    print("Generating P0 field plots...")
    plot_p0_field(ncells=128, final_time=1.0)
    plot_p0_field(ncells=128, final_time=5.0)
    
    print("Generating error field plots...")
    plot_error_field_dg(degree=2, ncells=32, final_time=5.0)
    plot_error_field_p0(ncells=128, final_time=5.0)
    
    print("All field plots generated.")


if __name__ == "__main__":
    main()
