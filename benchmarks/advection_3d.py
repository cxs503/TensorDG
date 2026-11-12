"""Verification ladder for the 3-D tensor-product DG advection solver.

Solves the separable periodic advection problem of
:mod:`tensordg.advection_3d` on a sequence of cubic meshes, measures the
quadrature ``L2`` error against the closed-form translated solution, the mass
drift and the observed order of accuracy, and writes machine-readable gates:

* ``metrics.json`` -- configuration, provenance, per-level measurements, gates
  with their thresholds and statuses, and the figure list;
* ``convergence.csv`` -- the same per-level numbers in flat form;
* ``summary.md`` -- tables rendered from the measurements at generation time
  (no hand-typed numbers);
* ``advection_3d_convergence.png`` -- log-log error vs ``h`` with the design
  slopes, plus the mid-plane error map of the finest P2 run.

The exit status is 0 only when every gate passes. A failing measurement is
stored as ``FAIL`` and is never rewritten.

Gates
-----
``design_order_P{p}``
    Observed order of the finest pair must be within 15 % of the theoretical
    ``p + 1``.
``mass_conservation_P{p}``
    Relative drift of the global integral (analytically ``1``) below ``1e-12``
    -- i.e. round-off, for every velocity sign.
``documented_case_below_3pct``
    The documented 8x8x8 run of P1 and P2 must be below the ecosystem's 3 %
    error gate.

Usage::

    python -m benchmarks.advection_3d                      # full ladder (~1 min)
    python -m benchmarks.advection_3d --quick              # coarse ladder
    python -m benchmarks.advection_3d --output-dir /tmp/x
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensordg.advection_3d import (  # noqa: E402
    error_3d, mass_3d, solve_advection_3d,
)

DESIGN_THRESHOLD_PERCENT = 3.0
MASS_DRIFT_THRESHOLD = 1e-12
ORDER_TOLERANCE = 0.15


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance best effort
        return "unknown"


def _ladders(quick: bool) -> dict[int, list[int]]:
    if quick:
        return {1: [4, 8], 2: [4, 8], 3: [4, 6]}
    return {1: [4, 8, 16], 2: [4, 8, 12], 3: [4, 6, 8]}


def _midplane_error(coeff, degree: int, final_time: float, velocity):
    """Numerical, exact and |error| on the z = 0.5 quadrature plane."""
    nx, ny, nz = coeff.shape[:3]
    hx, hy, hz = 1.0 / nx, 1.0 / ny, 1.0 / nz
    nq = max(2 * degree + 4, 6)
    nodes_np, _ = np.polynomial.legendre.leggauss(nq)
    nodes = torch.as_tensor(nodes_np, dtype=coeff.dtype)
    basis = []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        basis.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, polynomial),
            dtype=coeff.dtype,
        ))
    phi = torch.stack(basis, dim=1)
    z_centres = torch.as_tensor([(i + 0.5) * hz for i in range(nz)], dtype=coeff.dtype)
    zc = int(torch.argmin(torch.abs(z_centres - 0.5)).item())
    s_index = int(np.argmin(np.abs(z_centres[zc].item() + 0.5 * hz * nodes_np - 0.5)))
    plane = torch.einsum("xyklm,qk,rl->xyqrm", coeff[:, :, zc], phi, phi)
    numerical = torch.einsum("xyqrm,m->xyqr", plane, phi[s_index])
    xq = torch.as_tensor([(i + 0.5) * hx for i in range(nx)], dtype=coeff.dtype)
    yq = torch.as_tensor([(i + 0.5) * hy for i in range(ny)], dtype=coeff.dtype)
    xg = xq[:, None] + 0.5 * hx * nodes[None, :]
    yg = yq[:, None] + 0.5 * hy * nodes[None, :]
    zg = float(z_centres[zc].item() + 0.5 * hz * nodes_np[s_index])
    ax, ay, az = velocity
    exact = 1.0 + (
        torch.sin(2.0 * math.pi * (xg - ax * final_time))[:, None, :, None]
        * torch.sin(2.0 * math.pi * (yg - ay * final_time))[None, :, None, :]
        * math.sin(2.0 * math.pi * (zg - az * final_time))
    )
    return (
        xg.reshape(-1).numpy(), yg.reshape(-1).numpy(),
        numerical.numpy(), exact.numpy(),
        float(torch.max(torch.abs(numerical - exact)).item()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", type=float, default=0.25)
    parser.add_argument("--velocity", type=float, nargs=3, default=[1.0, 0.5, 0.25])
    parser.add_argument("--cfl", type=float, default=0.08)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output-dir", default="docs/validation/advection_3d",
    )
    args = parser.parse_args()

    # The tensors here are small enough that BLAS thread-pool dispatch dominates
    # the runtime (measured 10x slower on 128 threads than on one); pin the
    # thread count so the ladder cost is reproducible.
    torch.set_num_threads(int(os.environ.get("TENSORDG_THREADS", "1")))

    velocity = tuple(args.velocity)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    levels: list[dict] = []
    gates: list[dict] = []
    stores: dict[int, tuple] = {}
    for degree, cells in sorted(_ladders(args.quick).items()):
        previous_error = None
        previous_h = None
        for n in cells:
            x, y, z, coeff, meta = solve_advection_3d(
                ncells=n, degree=degree, final_time=args.time,
                velocity=velocity, cfl=args.cfl,
            )
            metrics = error_3d(x, y, z, coeff, args.time, velocity=velocity)
            drift = abs(mass_3d(coeff) - 1.0)
            h = 1.0 / n
            eoc = (
                None if previous_error is None
                else math.log(previous_error / metrics["relative_l2"])
                / math.log(previous_h / h)
            )
            levels.append({
                "degree": degree, "cells": n, "h": h,
                "relative_l2": metrics["relative_l2"],
                "absolute_l2": metrics["absolute_l2"],
                "mass_drift": drift, "steps": meta["steps"], "dt": meta["dt"],
                "eoc": eoc,
            })
            stores[degree] = (x, y, z, coeff, metrics)
            previous_error = metrics["relative_l2"]
            previous_h = h

        degree_levels = [row for row in levels if row["degree"] == degree]
        observed = degree_levels[-1]["eoc"]
        target = degree + 1
        gates.append({
            "name": f"design_order_P{degree}",
            "measured": observed,
            "threshold": f"{target} +/- {ORDER_TOLERANCE:.0%}",
            "status": "PASS" if observed is not None and abs(observed - target) <= ORDER_TOLERANCE * target else "FAIL",
        })
        worst_drift = max(row["mass_drift"] for row in degree_levels)
        gates.append({
            "name": f"mass_conservation_P{degree}",
            "measured": worst_drift,
            "threshold": MASS_DRIFT_THRESHOLD,
            "status": "PASS" if worst_drift < MASS_DRIFT_THRESHOLD else "FAIL",
        })

    documented = {}
    for degree in (1, 2):
        match = [row for row in levels if row["degree"] == degree and row["cells"] == 8]
        worst = 100.0 * match[0]["relative_l2"] if match else float("nan")
        documented[f"P{degree}_8x8x8_relative_percent"] = worst
        gates.append({
            "name": f"documented_case_below_3pct_P{degree}",
            "measured": worst,
            "threshold": DESIGN_THRESHOLD_PERCENT,
            "status": "PASS" if match and worst < DESIGN_THRESHOLD_PERCENT else "FAIL",
        })

    verdict = "PASS" if all(gate["status"] == "PASS" for gate in gates) else "FAIL"

    # --- figures ---------------------------------------------------------
    figure_path = output / "advection_3d_convergence.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for degree in sorted(_ladders(args.quick)):
        rows = [row for row in levels if row["degree"] == degree]
        hs = [row["h"] for row in rows]
        errors = [row["relative_l2"] for row in rows]
        axes[0].loglog(hs, errors, "o-", label=f"P{degree} (DG)")
        h_ref = np.array([hs[0], hs[-1]])
        axes[0].loglog(
            h_ref,
            errors[0] * (h_ref / hs[0]) ** (degree + 1),
            "--", color="0.6",
            label=f"slope {degree + 1}" if degree == 1 else None,
        )
    axes[0].set_xlabel("h = 1/N")
    axes[0].set_ylabel("relative L2 error")
    axes[0].set_title("3-D periodic advection: design order")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    x, y, z, coeff, _ = stores[max(stores)]
    degree = coeff.shape[-1] - 1
    xg, yg, numerical, exact, peak = _midplane_error(
        coeff, degree, args.time, velocity,
    )
    nq = int(np.sqrt(numerical.size / (coeff.shape[0] * coeff.shape[1])))
    nx, ny = coeff.shape[:2]
    error_map = np.abs(numerical - exact).reshape(nx * nq, ny * nq)
    grid_x, grid_y = np.meshgrid(xg, yg, indexing="ij")
    mesh = axes[1].pcolormesh(
        grid_x, grid_y, error_map, shading="auto", cmap="viridis",
    )
    axes[1].set_title(
        f"|u_h - u_exact| on z=0.5, P{degree} {nx}x{ny}x{coeff.shape[2]} "
        f"(peak {peak:.2e})"
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    fig.colorbar(mesh, ax=axes[1], shrink=0.85)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)

    payload = {
        "benchmark": "advection_3d",
        "verdict": verdict,
        "provenance": {
            "commit": _git_commit(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "configuration": {
            "final_time": args.time, "velocity": list(velocity),
            "cfl": args.cfl, "ladders": _ladders(args.quick),
            "degree_scaling": "modal Legendre, cell average = mode [0,0,0]",
        },
        "thresholds": {
            "order_tolerance": ORDER_TOLERANCE,
            "mass_drift": MASS_DRIFT_THRESHOLD,
            "relative_l2_percent": DESIGN_THRESHOLD_PERCENT,
        },
        "levels": levels,
        "documented": documented,
        "gates": gates,
        "figures": [figure_path.name],
    }
    with open(output / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with open(output / "convergence.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "degree", "cells", "h", "relative_l2", "absolute_l2",
            "mass_drift", "steps", "dt", "eoc",
        ])
        for row in levels:
            writer.writerow([
                row["degree"], row["cells"], f"{row['h']:.6e}",
                f"{row['relative_l2']:.6e}", f"{row['absolute_l2']:.6e}",
                f"{row['mass_drift']:.3e}", row["steps"], f"{row['dt']:.6e}",
                "" if row["eoc"] is None else f"{row['eoc']:.4f}",
            ])

    lines = [
        "# 3-D periodic advection -- verification ladder",
        "",
        f"Verdict: **{verdict}**  |  commit `{payload['provenance']['commit']}`",
        f"| config: `t={args.time}`, `a={list(velocity)}`, `cfl={args.cfl}`,",
        f"torch {payload['provenance']['torch']}, numpy {payload['provenance']['numpy']}",
        "",
        "| degree | cells | h | relative L2 | absolute L2 | EOC | mass drift | steps |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in levels:
        eoc = "--" if row["eoc"] is None else f"{row['eoc']:.3f}"
        lines.append(
            f"| {row['degree']} | {row['cells']} | {row['h']:.4f} | "
            f"{row['relative_l2']:.4e} | {row['absolute_l2']:.4e} | {eoc} | "
            f"{row['mass_drift']:.2e} | {row['steps']} |"
        )
    lines += ["", "| gate | measured | threshold | status |", "|---|---:|---|---|"]
    for gate in gates:
        measured = gate["measured"]
        shown = f"{measured:.4e}" if isinstance(measured, float) else str(measured)
        lines.append(
            f"| `{gate['name']}` | {shown} | {gate['threshold']} | {gate['status']} |"
        )
    lines += ["", f"Figure: `{figure_path.name}` (left: convergence; right: mid-plane error).", ""]
    with open(output / "summary.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(json.dumps({"verdict": verdict, "gates": gates}, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())