"""Small reconstruction helpers for one-dimensional modal DG benchmarks."""
from __future__ import annotations

import numpy as np
import torch


def legendre_values(points: np.ndarray, degree: int) -> np.ndarray:
    values = []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        values.append(np.polynomial.legendre.legval(points, polynomial))
    return np.stack(values, axis=-1)


def reconstruct(coeff: torch.Tensor, samples_per_cell: int = 5):
    values = coeff.detach().cpu().numpy()
    cells, modes = values.shape
    reference = 2.0 * (np.arange(samples_per_cell) + 0.5) / samples_per_cell - 1.0
    phi = legendre_values(reference, modes - 1)
    numerical = (values @ phi.T).reshape(-1)
    x = (np.arange(cells * samples_per_cell) + 0.5) / (cells * samples_per_cell)
    return x, numerical


def sample(coeff: torch.Tensor, points: int = 401):
    values = coeff.detach().cpu().numpy()
    cells, modes = values.shape
    x = np.linspace(0.0, 1.0, points)
    indices = np.minimum((x * cells).astype(int), cells - 1)
    reference = 2.0 * (x * cells - indices) - 1.0
    phi = legendre_values(reference, modes - 1)
    numerical = np.einsum("nk,nk->n", values[indices], phi)
    return x, numerical


def save_profile(path, x, numerical, exact, title, ylabel="u"):
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    axes[0].plot(x, exact, "k-", linewidth=2.0, label="Exact / reference")
    axes[0].plot(x, numerical, "r--", linewidth=1.4, label="DG numerical")
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].semilogy(x, np.maximum(np.abs(numerical - exact), 1e-16))
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Absolute error")
    axes[1].grid(True, alpha=0.3, which="both")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_convergence(path, cells, errors, degree):
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = np.asarray(cells)
    errors = np.asarray(errors)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.loglog(cells, errors, "o-", label=f"DG P{degree}")
    reference = errors[-1] * (cells / cells[-1]) ** (-(degree + 1))
    ax.loglog(cells, reference, "k--", label=f"Reference slope {degree + 1}")
    ax.set_xlabel("Number of cells")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Burgers mesh convergence")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
