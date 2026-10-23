"""Reconstruction and plotting helpers shared by two-dimensional benchmarks."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _legendre_values(points: np.ndarray, degree: int) -> np.ndarray:
    values = []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        values.append(np.polynomial.legendre.legval(points, polynomial))
    return np.stack(values, axis=-1)


def reconstruct(
    coeff: torch.Tensor, samples_per_cell: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate tensor-product modes on a uniform plotting mesh."""
    if samples_per_cell < 1:
        raise ValueError("samples_per_cell must be positive")
    values = coeff.detach().cpu().numpy()
    nx, ny, modes, _ = values.shape
    reference = 2.0 * (np.arange(samples_per_cell) + 0.5) / samples_per_cell - 1.0
    phi = _legendre_values(reference, modes - 1)
    cell_values = np.einsum("xykl,ak,bl->xyab", values, phi, phi)
    field = cell_values.transpose(1, 3, 0, 2).reshape(
        ny * samples_per_cell, nx * samples_per_cell,
    )
    x = (np.arange(nx * samples_per_cell) + 0.5) / (nx * samples_per_cell)
    y = (np.arange(ny * samples_per_cell) + 0.5) / (ny * samples_per_cell)
    xx, yy = np.meshgrid(x, y)
    return xx, yy, field


def sample_centerline(
    coeff: torch.Tensor, fixed: float = 0.5, points: int = 401,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample u(x,fixed), including the exact fixed coordinate in the return."""
    values = coeff.detach().cpu().numpy()
    nx, ny, modes, _ = values.shape
    x = np.linspace(0.0, 1.0, points)
    y = np.full_like(x, fixed)
    ix = np.minimum((x * nx).astype(int), nx - 1)
    iy = np.minimum((y * ny).astype(int), ny - 1)
    xi = 2.0 * (x * nx - ix) - 1.0
    eta = 2.0 * (y * ny - iy) - 1.0
    phi_x = _legendre_values(xi, modes - 1)
    phi_y = _legendre_values(eta, modes - 1)
    numerical = np.einsum("nkl,nk,nl->n", values[ix, iy], phi_x, phi_y)
    return x, numerical, fixed


def save_contours(
    path: str | Path,
    xx: np.ndarray,
    yy: np.ndarray,
    numerical: np.ndarray,
    exact: np.ndarray,
    title: str,
):
    """Write numerical, exact, and absolute-error contour panels."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    vmin = min(float(numerical.min()), float(exact.min()))
    vmax = max(float(numerical.max()), float(exact.max()))
    levels = np.linspace(vmin, vmax, 41)
    for ax, field, panel in zip(
        axes[:2], (numerical, exact), ("DG numerical", "Exact / reference"),
    ):
        image = ax.contourf(xx, yy, field, levels=levels, cmap="viridis")
        ax.set_title(panel)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, shrink=0.86)
    error = np.abs(numerical - exact)
    image = axes[2].contourf(xx, yy, error, levels=40, cmap="magma")
    axes[2].set_title(f"Absolute error (max={error.max():.3e})")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    fig.colorbar(image, ax=axes[2], shrink=0.86)
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_centerline(
    path: str | Path,
    x: np.ndarray,
    numerical: np.ndarray,
    exact: np.ndarray,
    fixed: float,
    title: str,
    ylabel: str = "u",
):
    """Write a centerline comparison and its absolute-error curve."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    axes[0].plot(x, exact, "k-", linewidth=2.0, label="Exact / reference")
    axes[0].plot(x, numerical, "r--", linewidth=1.5, label="DG numerical")
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title(f"{title}: y = {fixed:g}")
    axes[1].semilogy(x, np.maximum(np.abs(numerical - exact), 1e-16), color="tab:blue")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Absolute error")
    axes[1].grid(True, alpha=0.3, which="both")
    fig.savefig(path, dpi=160)
    plt.close(fig)
