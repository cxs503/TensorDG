"""Conservation + positivity + self-convergence audit for the CKS1998 2-D Riemann case.

Anchor: ``REPRO-DG-CKS1998`` (Cockburn & Shu, JCP 141 (1998) 199-224, Part V; the
four-quadrant case is the Lax & Liu (1998) configuration 3).

Why this script exists
----------------------
``docs/CASE-DG-CKS1998-2D-RIEMANN.md`` recorded two honest FAILs against the
pre-declared gates of ``benchmarks/riemann_2d_cks1998.py``:

* **C4 conservation**: total mass changed by 25.16 % (> 1 % gate), attributed in
  the old report to "the TVB limiter's modal scaling not being conservative";
* **C2 self-convergence**: ``not_run`` -- the harness compared *total masses* of
  two meshes, which is not a self-convergence metric at all;
* **C1 positivity**: ``rho_quad_min = -21.44``, i.e. the positivity filter was
  measured *before* ``_positivity_filter`` grew its Zhang-Shu scaling form.

This audit replaces the ill-posed gates with well-posed ones and measures them:

1. **Flux-balance identity** (valid for the transmissive boundaries the case
   actually uses).  For a weak-form DG update, the sum over cells of the cell
   averages changes only through the non-periodic boundary faces, because every
   interior interface contributes ``+F_i`` to one cell and ``-F_i`` to its
   neighbour.  ``solve_euler_dg_2d(..., track_boundary_flux=True)`` accumulates
   exactly that integral with the SSP-RK3 weights, so
   ``Q(t) - Q(0) - integral(boundary flux) == 0`` to round-off *iff* the scheme
   (including limiting and the positivity filter) is conservative.
2. **Periodic control**: the same initial data on a periodic domain conserves
   mass, momentum and energy to round-off (no boundary at all).
3. **Physical vs numerical drift**: if the non-periodic mass change is the same
   physical number on every mesh, it is boundary flux, not a limiting artefact.
4. **Self-convergence**: block-averaged cell means of the fine mesh compared with
   the coarse mesh, i.e. a real L1 metric.
5. **Positivity**: density and pressure at the volume quadrature nodes, measured
   independently of the solver's own admissibility test.

Usage
-----
    OMP_NUM_THREADS=1 PYTHONPATH=. python validation/riemann2d_conservation/audit.py \
        --meshes 16 32 64 --final-time 0.25 --out validation/riemann2d_conservation/artifacts
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensordg.euler_1d import GAMMA
from tensordg.euler_dg_2d import (
    LAX_LIU_CONFIG_3,
    legendre_operators,
    solve_euler_dg_2d,
)

VARIANTS = ("mass", "x_momentum", "y_momentum", "energy")


def make_initial(quadrants=None):
    """Four-quadrant initial condition factory (Lax & Liu config 3 by default)."""
    quadrants = LAX_LIU_CONFIG_3 if quadrants is None else quadrants

    def initial(x, y):
        shape = torch.broadcast_shapes(x.shape, y.shape)
        lower = torch.broadcast_to(y < 0.5, shape)
        left = torch.broadcast_to(x < 0.5, shape)
        rho = torch.zeros(shape, dtype=x.dtype)
        velocity_x = torch.zeros(shape, dtype=x.dtype)
        velocity_y = torch.zeros(shape, dtype=x.dtype)
        pressure = torch.zeros(shape, dtype=x.dtype)
        halves = (
            ("lower-left", lower & left),
            ("lower-right", lower & ~left),
            ("upper-left", ~lower & left),
            ("upper-right", ~lower & ~left),
        )
        for name, mask in halves:
            state = quadrants[name]
            rho = torch.where(mask, torch.full(shape, state[0], dtype=x.dtype), rho)
            velocity_x = torch.where(mask, torch.full(shape, state[1], dtype=x.dtype), velocity_x)
            velocity_y = torch.where(mask, torch.full(shape, state[2], dtype=x.dtype), velocity_y)
            pressure = torch.where(mask, torch.full(shape, state[3], dtype=x.dtype), pressure)
        return rho, velocity_x, velocity_y, pressure

    return initial


def quadrature_state(coeff, degree):
    """Conservative state at the volume quadrature nodes, shape (nx, ny, q, q, 4)."""
    nodes, _weights, basis, _derivative, _signs = legendre_operators(
        degree, max(degree + 2, 3), device=coeff.device, dtype=coeff.dtype
    )
    return torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)


def positivity_report(coeff, degree):
    """rho/p at volume nodes and at cell averages (independent of the solver)."""
    state = quadrature_state(coeff, degree)
    rho = state[..., 0]
    energy = state[..., 3]
    momentum_sq = state[..., 1] ** 2 + state[..., 2] ** 2
    pressure = (GAMMA - 1.0) * (energy - 0.5 * momentum_sq / rho.clamp_min(1e-30))
    mean = coeff[..., 0, 0, :]
    mean_pressure = (GAMMA - 1.0) * (
        mean[..., 3] - 0.5 * (mean[..., 1] ** 2 + mean[..., 2] ** 2) / mean[..., 0].clamp_min(1e-30)
    )
    return {
        "rho_min": float(rho.min()),
        "p_min": float(pressure.min()),
        "rho_min_mean": float(mean[..., 0].min()),
        "p_min_mean": float(mean_pressure.min()),
        "rho_positive": bool(torch.all(rho > 0.0)),
        "p_positive": bool(torch.all(pressure > 0.0)),
        "rp_finite": bool(torch.isfinite(state).all()),
    }


def block_average(means, factor):
    """Average a (n, n) array of cell means down by an integer ``factor``."""
    n = means.shape[0]
    if n % factor:
        raise ValueError(f"cannot coarsen {n} by {factor}")
    m = n // factor
    return means.reshape(m, factor, m, factor).mean(dim=(1, 3))


def self_convergence(coarse_means, fine_means):
    """Relative L1 distance between a coarse mesh and the block-averaged fine mesh."""
    factor = fine_means.shape[0] // coarse_means.shape[0]
    folded = block_average(fine_means, factor)
    numerator = float(torch.abs(folded - coarse_means).sum())
    denominator = float(torch.abs(folded).sum())
    return numerator / max(denominator, 1e-30)


def run_case(ncells, degree, final_time, cfl, periodic, limiter="tvb", tvb_constant=1.0):
    started = time.time()
    x, y, coeff, meta = solve_euler_dg_2d(
        make_initial(),
        ncells=ncells,
        degree=degree,
        domain=((0.0, 1.0), (0.0, 1.0)),
        final_time=final_time,
        cfl=cfl,
        limiter=limiter,
        tvb_constant=tvb_constant,
        periodic=(periodic, periodic),
        initial_projection="l2",
        track_boundary_flux=True,
    )
    wall = time.time() - started
    means = coeff[..., 0, 0, 0].clone()
    record = {
        "ncells": ncells,
        "degree": degree,
        "periodic": bool(periodic),
        "final_time": final_time,
        "cfl": cfl,
        "steps": meta["steps"],
        "dt_last": meta.get("dt"),
        "wall_seconds": round(wall, 2),
        "positivity": positivity_report(coeff, degree),
        "conservation": {},
    }
    for name in VARIANTS:
        row = meta["conservation_residual"][name]
        record["conservation"][name] = {
            "change": row["change"],
            "boundary_flux": row["boundary_flux"],
            "residual": row["residual"],
            "relative_residual": row["relative"],
        }
    record["cell_means_rho"] = means
    record["x"] = x
    record["y"] = y
    record["coeff"] = coeff
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--final-time", type=float, default=0.25)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--periodic-meshes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--out", type=Path, default=Path("validation/riemann2d_conservation/artifacts"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = {}
    for ncells in args.meshes:
        record = run_case(ncells, args.degree, args.final_time, args.cfl, periodic=False)
        runs[ncells] = record
        worst = max(
            record["conservation"][name]["relative_residual"] for name in VARIANTS
        )
        print(
            f"[open ] n={ncells:3d} steps={record['steps']:4d} "
            f"mass_change={record['conservation']['mass']['change']:+.6f} "
            f"max_flux_balance_residual={worst:.3e} "
            f"rho_min={record['positivity']['rho_min']:.4e} "
            f"p_min={record['positivity']['p_min']:.4e} "
            f"{record['wall_seconds']:.1f}s",
            flush=True,
        )

    periodic_runs = {}
    for ncells in args.periodic_meshes:
        record = run_case(ncells, args.degree, args.final_time, args.cfl, periodic=True)
        periodic_runs[ncells] = record
        drift = max(
            abs(record["conservation"][name]["change"])
            / max(abs(record["conservation"][name]["boundary_flux"]), 1.0)
            for name in VARIANTS
        )
        print(
            f"[per  ] n={ncells:3d} steps={record['steps']:4d} "
            f"max_abs_change={max(abs(record['conservation'][n]['change']) for n in VARIANTS):.3e} "
            f"{record['wall_seconds']:.1f}s",
            flush=True,
        )

    meshes = sorted(runs)
    convergence = {}
    for coarse, fine in zip(meshes, meshes[1:]):
        convergence[f"{coarse}_vs_{fine}"] = self_convergence(
            runs[coarse]["cell_means_rho"], runs[fine]["cell_means_rho"]
        )

    metrics = {
        "case": "CKS1998 / Lax-Liu 1998 configuration 3 (four-quadrant Riemann)",
        "anchor": "REPRO-DG-CKS1998",
        "degree": args.degree,
        "final_time": args.final_time,
        "cfl": args.cfl,
        "limiter": "tvb(M=1.0)",
        "transmissive": {
            str(n): {
                k: v
                for k, v in runs[n].items()
                if k not in ("cell_means_rho", "x", "y", "coeff")
            }
            for n in meshes
        },
        "periodic": {
            str(n): {
                k: v
                for k, v in periodic_runs[n].items()
                if k not in ("cell_means_rho", "x", "y", "coeff")
            }
            for n in sorted(periodic_runs)
        },
        "self_convergence_l1_rho": convergence,
        "summary": {
            "max_flux_balance_residual_transmissive": max(
                runs[n]["conservation"][v]["relative_residual"]
                for n in meshes
                for v in VARIANTS
            ),
            "max_abs_conservation_change_periodic": max(
                abs(periodic_runs[n]["conservation"][v]["change"])
                for n in periodic_runs
                for v in VARIANTS
            ),
            "mass_change_transmissive": {
                str(n): runs[n]["conservation"]["mass"]["change"] for n in meshes
            },
            "rho_min_all_meshes": min(runs[n]["positivity"]["rho_min"] for n in meshes),
            "p_min_all_meshes": min(runs[n]["positivity"]["p_min"] for n in meshes),
            "self_convergence_l1_rho": convergence,
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if not args.no_figures:
        make_figures(runs, periodic_runs, meshes, metrics, out)

    print(json.dumps(metrics["summary"], indent=2))
    return metrics


def make_figures(runs, periodic_runs, meshes, metrics, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # --- density fields -----------------------------------------------------
    fig, axes = plt.subplots(1, len(meshes), figsize=(4.2 * len(meshes), 4.0))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, meshes):
        means = runs[n]["cell_means_rho"].cpu().numpy()
        x = runs[n]["x"].cpu().numpy()
        y = runs[n]["y"].cpu().numpy()
        mesh = ax.pcolormesh(x, y, means.T, shading="nearest", cmap="viridis")
        ax.set_aspect("equal")
        ax.set_title(f"P{runs[n]['degree']} {n}x{n}, t={runs[n]['final_time']}")
        ax.set_xlabel("x")
        fig.colorbar(mesh, ax=ax, label="density")
    axes[0].set_ylabel("y")
    fig.suptitle("CKS1998 2-D Riemann (Lax-Liu config 3): cell-mean density")
    fig.tight_layout()
    fig.savefig(out / "density_fields.png", dpi=130)
    plt.close(fig)

    # --- flux-balance residual ---------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    width = 0.8 / len(VARIANTS)
    for i, name in enumerate(VARIANTS):
        values = [max(runs[n]["conservation"][name]["relative_residual"], 1e-18) for n in meshes]
        ax.bar(np.arange(len(meshes)) + i * width, values, width=width, label=name)
    ax.set_yscale("log")
    ax.axhline(1e-12, color="k", ls="--", lw=1, label="gate 1e-12")
    ax.set_xticks(np.arange(len(meshes)) + 0.4 - width / 2)
    ax.set_xticklabels([f"{n}x{n}" for n in meshes])
    ax.set_ylabel("|change - boundary flux| / scale")
    ax.set_title("Discrete flux-balance residual (transmissive boundaries)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "flux_balance_residual.png", dpi=130)
    plt.close(fig)

    # --- self-convergence ---------------------------------------------------
    convergence = metrics["self_convergence_l1_rho"]
    pairs = list(convergence)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    mids = [0.5 * (float(a.split("_vs_")[0]) + float(a.split("_vs_")[1])) for a in pairs]
    ax.plot(mids, [convergence[a] for a in pairs], "o-")
    for mid, a in zip(mids, pairs):
        ax.annotate(f"{a}: {convergence[a]:.4f}", (mid, convergence[a]),
                    textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mesh (cells per direction)")
    ax.set_ylabel(r"relative $L^1$ density difference")
    ax.set_title("Mesh self-convergence of the 2-D Riemann case")
    fig.tight_layout()
    fig.savefig(out / "self_convergence.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()