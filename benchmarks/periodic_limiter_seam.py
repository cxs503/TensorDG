"""Periodic-domain limiter stencil: measure the seam defect and its fix.

The TVB/WENO limiters in :mod:`tensordg.limiters` build their minmod decision
from the cell averages of the immediate neighbours.  Historically that padding
was done by *clamping* (``avg[0]`` / ``avg[-1]`` are duplicated) and the two end
cells were never tested at all.  That is right for an open/outflow boundary and
wrong for a periodic one: a discontinuity sitting on the seam ``x = 0 / 1`` is
never limited, and the minmod stencil of the first/last cells sees a clamped
neighbour instead of the wrapped one.
``docs/validation/limiter-tvb-weno-validation.md`` §8 records this as a known
limitation ("a periodic advection test would be affected"); this script turns it
into a machine-checkable quantity.

Benchmark (exact solution, zero-cost reference):

    Euler 1-D, periodic [0,1], u = 1, p = 1,
    rho = 2 on (0, 1/2), rho = 1 on (1/2, 1)

A pure *contact* discontinuity (density only) transported unchanged at speed 1,
with one jump at ``x = 1/2`` (interior) and one on the seam ``x = 0 / 1``.

Primary criterion -- **mesh shift symmetry**
--------------------------------------------
The initial data satisfies ``rho0(x + 1/2) = 3 - rho0(x)`` and the exact
solution commutes with that half-period translation, so on a periodic mesh of
an even number of cells a correct scheme must produce

    rho_i(t) + rho_{i+N/2}(t) = 3   and   c1_i(t) + c1_{i+N/2}(t) = 0

for every cell ``i`` (``c1`` = modal slope coefficient).  This is an *exact*
discrete identity: it needs no reference solution and no tolerance argument.
Any global limiter treatment that is not translation invariant breaks it, and
only the seam cells can break it.

Usage
-----
    PYTHONPATH=. python benchmarks/periodic_limiter_seam.py --tag before
    PYTHONPATH=. python benchmarks/periodic_limiter_seam.py --tag after --plot

Artifacts go to ``artifacts/periodic_limiter_seam/<tag>.json`` (+ ``.png``).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from tensordg.euler_dg_1d import solve_periodic_euler_dg_1d

HIGH, LOW = 2.0, 1.0
VELOCITY, PRESSURE = 1.0, 1.0
RHO_JUMP = 0.5  # interior jump at x = 1/2


def _rho0(x: np.ndarray) -> np.ndarray:
    """Initial density: HIGH on (0, 1/2), LOW on (1/2, 1); periodic."""
    return np.where(np.mod(x, 1.0) < RHO_JUMP, HIGH, LOW)


def _initial_state(x):
    """``solve_periodic_euler_dg_1d`` initial data (tensor in, tuple out)."""
    rho = _rho0(x.detach().cpu().numpy())
    return (
        torch.as_tensor(rho, dtype=x.dtype, device=x.device).reshape(x.shape),
        torch.full_like(x, VELOCITY),
        torch.full_like(x, PRESSURE),
    )


def _quadrature(degree: int):
    nodes_np, weights_np = np.polynomial.legendre.leggauss(max(degree + 2, 4))
    phi = np.stack(
        [np.polynomial.legendre.legval(nodes_np, np.pad([1.0], (k, 0)))
         for k in range(degree + 1)], axis=1)
    return nodes_np, weights_np, phi


def _metrics(x, coeff, final_time, degree, seam_cells=4):
    """Errors against the exact translated profile + the symmetry identities."""
    cells = coeff.shape[0]
    half = cells // 2
    dx = float((x[-1] - x[0]) / (cells - 1))
    values = coeff.detach().cpu().numpy()
    centres = x.detach().cpu().numpy()
    nodes, weights, phi = _quadrature(degree)

    # High-order point values (an overshoot hidden by the cell averages shows up
    # here) and the exact profile translated by t (advection speed = 1).
    point = np.einsum("cmv,qm->cqv", values, phi)[..., 0]
    points = centres[:, None] + 0.5 * dx * nodes[None, :]
    exact_point = _rho0(points - final_time)
    err_point = np.abs(point - exact_point)
    exact_avg = 0.5 * np.sum(weights[None, :] * exact_point, axis=1)
    avg = values[:, 0, 0]

    # Shift-symmetry identities (exact, no reference needed).
    sym_avg = np.abs(avg[:half] + avg[half:] - (HIGH + LOW))
    sym_mode1 = np.abs(values[:half, 1, 0] + values[half:, 1, 0])
    worst = int(np.argmax(sym_avg))

    # Window split: the seam cells vs an interior patch holding the other jump.
    seam = np.r_[np.arange(seam_cells), cells - np.arange(1, seam_cells + 1)]
    interior = np.arange(seam_cells + 6, cells - seam_cells - 6)
    dq = 0.5 * dx * weights  # [nquad]

    def window_max(idx):
        return float(err_point[idx].max())

    def window_l1(idx):
        return float(np.sum(dq[None, :] * err_point[idx]) / (len(idx) * dx))

    def window_excursion(idx):
        vals = point[idx]
        return float(max(np.max(vals) - HIGH, LOW - np.min(vals)))

    return {
        "cells": int(cells),
        "degree": int(degree),
        "final_time": float(final_time),
        "l1_point": window_l1(np.arange(cells)),
        "linf_point": float(err_point.max()),
        "l1_avg": float(np.sum(dx * np.abs(avg - exact_avg))),
        "symmetry_defect_avg": float(sym_avg.max()),
        "symmetry_defect_mode1": float(sym_mode1.max()),
        "symmetry_worst_cells": [worst, int((worst + half) % cells)],
        "seam_linf": window_max(seam),
        "seam_l1": window_l1(seam),
        "interior_linf": window_max(interior),
        "interior_l1": window_l1(interior),
        "seam_excursion": window_excursion(seam),
        "interior_excursion": window_excursion(interior),
        "overshoot": float(np.max(point) - HIGH),
        "undershoot": float(LOW - np.min(point)),
        "mass_drift": float(abs(dx * avg.sum() - dx * _rho0(centres).sum())),
        "seam_mode_amplitude": [float(np.abs(values[0, 1:, 0]).max()),
                                float(np.abs(values[half, 1:, 0]).max())],
        "profile": {
            "x": centres.tolist(),
            "rho_average": avg.tolist(),
            "rho_exact_average": exact_avg.tolist(),
            "rho_point_seam": point[0].tolist(),
            "rho_point_seam_exact": exact_point[0].tolist(),
            "rho_point_interior": point[half].tolist(),
            "rho_point_interior_exact": exact_point[half].tolist(),
            "x_point": (centres[0] + 0.5 * dx * nodes).tolist(),
        },
    }


def run_case(cells=48, degree=2, final_time=0.02, limiter="tvb",
             tvb_constant=1.0, limiter_stage="every", dtype=torch.float64):
    x, coeff, meta = solve_periodic_euler_dg_1d(
        _initial_state, ncells=cells, degree=degree, final_time=final_time,
        limiter=limiter, tvb_constant=tvb_constant, limiter_stage=limiter_stage,
        boundary="periodic", dtype=dtype,
    )
    out = _metrics(x, coeff, final_time, degree)
    out.update({
        "limiter": limiter, "tvb_constant": tvb_constant,
        "limiter_stage": limiter_stage, "steps": meta.get("steps"),
    })
    return out


CASES = {
    "unlimited": dict(limiter=None),
    "tvb_M1_every": dict(limiter="tvb", tvb_constant=1.0),
    "weno_M1_every": dict(limiter="weno", tvb_constant=1.0),
}
FIELDS = ("l1_point", "seam_linf", "interior_linf", "seam_excursion",
          "interior_excursion", "symmetry_defect_avg", "symmetry_defect_mode1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="unset")
    parser.add_argument("--cells", type=int, default=48)
    parser.add_argument("--times", type=float, nargs="+", default=[0.02, 0.25])
    parser.add_argument("--outdir", default="artifacts/periodic_limiter_seam")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    results = {}
    for final_time in args.times:
        results[str(final_time)] = {}
        for name, kwargs in CASES.items():
            m = run_case(cells=args.cells, final_time=final_time, **kwargs)
            results[str(final_time)][name] = m
            print(f"t={final_time:<5} {name:14s} " + " ".join(
                f"{f}={m[f]:.4e}" for f in FIELDS) +
                f" seam_modes={['%.3e' % v for v in m['seam_mode_amplitude']]}")

    os.makedirs(args.outdir, exist_ok=True)
    payload = {
        "tag": args.tag,
        "setup": {
            "problem": "1-D Euler contact discontinuity on a periodic mesh",
            "domain": [0.0, 1.0], "cells": args.cells, "degree": 2,
            "final_times": args.times,
            "rho": f"{HIGH} on (0,1/2), {LOW} on (1/2,1)",
            "velocity": VELOCITY, "pressure": PRESSURE, "boundary": "periodic",
            "criterion": "rho_i + rho_{i+N/2} = 3, c1_i + c1_{i+N/2} = 0",
        },
        "cases": results,
    }
    path = os.path.join(args.outdir, f"{args.tag}.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print("wrote", path)

    if args.plot:
        plot(payload, os.path.join(args.outdir, f"{args.tag}.png"))


def plot(payload, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = payload["setup"]["final_times"]
    early, late = payload["cases"][str(times[0])], payload["cases"][str(times[-1])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    ax = axes[0]
    for name in CASES:
        prof = early[name]["profile"]
        ax.plot(prof["x_point"], prof["rho_point_seam"], "o-", ms=2.5, label=name)
    prof = early["unlimited"]["profile"]
    ax.plot(prof["x_point"], prof["rho_point_seam_exact"], "k--", label="exact")
    ax.set_xlim(prof["x_point"][0] - 0.01, prof["x_point"][-1] + 0.01)
    ax.set_xlabel("x (seam cell 0, quadrature nodes)")
    ax.set_ylabel("density")
    ax.set_title(f"seam cell: unlimited vs limited (t={times[0]})")
    ax.legend(fontsize=7)

    ax = axes[1]
    prof = early["unlimited"]["profile"]
    ax.plot(prof["x_point"], prof["rho_point_interior"], "o-", ms=2.5, label="cell N/2")
    shift = 0.5
    ax.plot(np.asarray(prof["x_point"]) + shift, prof["rho_point_seam"], "s--", ms=2.5,
            label="cell 0 shifted by 1/2")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("interior cell vs half-period shifted seam cell")
    ax.legend(fontsize=7)

    ax = axes[2]
    names = list(CASES)
    idx = np.arange(len(names))
    width = 0.2
    for k, t in enumerate(times):
        vals = [payload["cases"][str(t)][n]["symmetry_defect_avg"] for n in names]
        ax.bar(idx + (k - 0.5) * width * 2, vals, width * 2, label=f"t={t}")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=15, fontsize=7)
    ax.set_ylabel(r"$\max_i|\rho_i+\rho_{i+N/2}-3|$")
    ax.set_title("shift-symmetry defect (exact identity)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print("wrote", path)


if __name__ == "__main__":
    main()