"""Verification study for the 2-D TVB limiter (mission m2546).

Reproduces the numbers and the figure of ``docs/limiter_2d_verification.md``:

A. detector parity    2-D ``_troubled_1d`` vs the 1-D criterion on extruded data
B. detector behaviour  troubled-cell fraction on a *smooth* vortex vs M
                       (new scale-consistent criterion vs the historic copy)
C. smooth accuracy    isentropic vortex, P2, M in {None, 1, 10}: error and EOC
D. shock capturing    extruded Sod vs the exact Riemann solution, M in {None, 0}

Usage:
    python -m benchmarks.verify_euler_dg_2d_limiter [--quick] [--outdir DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from benchmarks.sod_exact import cell_averages as sod_cell_averages
from benchmarks.verify_euler_dg_2d import _vortex_error
from tensordg import euler_dg_2d as dg2
from tensordg.limiters import _troubled_cell_indicator, apply_tvb_limiter

DTYPE = torch.float64
DEVICE = torch.device("cpu")
DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
THREE_PERCENT = 3.0


def _historic_mask(averages, axis, spacing, tvb_constant):
    """The pre-m2546 2-D detector (kept only to quantify the change).

    It compared ``max|d_left|, max|d_right|`` -- an O(dx) measure -- against
    ``M * spacing**2`` and required strict non-monotonicity.
    """
    if axis == 0:
        plus = torch.roll(averages, dims=0, shifts=-1)
        minus = torch.roll(averages, dims=0, shifts=1)
    else:
        plus = torch.roll(averages, dims=1, shifts=-1)
        minus = torch.roll(averages, dims=1, shifts=1)
    d_left = averages - minus
    d_right = plus - averages
    variation = torch.maximum(d_left.abs(), d_right.abs()).amax(dim=-1)
    minmod = 0.5 * (torch.sign(d_left) + torch.sign(d_right)) * torch.minimum(
        d_left.abs(), d_right.abs()
    )
    not_monotone = (minmod - d_left).abs().amax(dim=-1) >= 1e-14
    return not_monotone & (variation > tvb_constant * spacing * spacing)


def detector_parity():
    """Extruded smooth data: the 2-D detector must equal the 1-D criterion."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tests.test_euler_dg_2d_limiter import _extrude, _smooth_1d_coefficients

    rows = []
    for ncells, degree in ((12, 2), (24, 2), (24, 3)):
        coeff_1d = _smooth_1d_coefficients(ncells, degree)
        coeff_2d = _extrude(coeff_1d)
        spacing = 1.0 / ncells
        for tvb_constant in (0.0, 0.1, 1.0, 10.0):
            mask_1d = _troubled_cell_indicator(coeff_1d, spacing, tvb_constant)
            mask_2d, _ = dg2._troubled_1d(
                coeff_2d[..., 0, 0, :], 0, spacing, tvb_constant, periodic=False
            )
            limited_1d = apply_tvb_limiter(coeff_1d, spacing, tvb_constant)
            limited_2d = dg2.apply_tvb_limiter_2d(
                coeff_2d, spacing, spacing, tvb_constant, periodic=(False, True)
            )
            rows.append({
                "ncells": ncells,
                "degree": degree,
                "M": tvb_constant,
                "mask_equal": bool(torch.equal(mask_1d, mask_2d[:, 0])),
                "mode1_max_abs_diff": float(
                    (limited_2d[:, 0, 1, 0, :] - limited_1d[:, 1, :]).abs().max()
                ),
                "average_preserved": bool(
                    torch.equal(limited_2d[..., 0, 0, :], coeff_2d[..., 0, 0, :])
                ),
                "cells_flagged_1d": int(mask_1d.sum()),
            })
    return rows


def detector_behaviour(meshes=(16, 24), constants=(0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)):
    """Troubled-cell fraction on a smooth vortex: historic vs scale-consistent."""
    rows = []
    for ncells in meshes:
        _, _, coeff, _ = dg2.solve_euler_dg_2d_vortex(
            ncells=ncells, degree=2, final_time=1e-9
        )
        averages = coeff[..., 0, 0, :]
        spacing = 10.0 / ncells
        for M in constants:
            entry = {"ncells": ncells, "M": M, "cells": ncells * ncells}
            for name, function in (
                ("historic", _historic_mask),
                ("scale_consistent", None),
            ):
                total = 0
                for axis in (0, 1):
                    if function is None:
                        mask, _ = dg2._troubled_1d(averages, axis, spacing, M)
                    else:
                        mask = function(averages, axis, spacing, M)
                    total += int(mask.sum())
                entry[f"{name}_flagged"] = total
                entry[f"{name}_fraction"] = total / (2 * ncells * ncells)
            rows.append(entry)
    return rows


def vortex_accuracy(meshes=(16, 24), degree=2, constants=(None, 1.0, 10.0),
                    final_time=1.0, cfl=0.2):
    rows = []
    for M in constants:
        previous = None
        for ncells in meshes:
            started = time.time()
            _, _, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
                ncells=ncells,
                degree=degree,
                final_time=final_time,
                cfl=cfl,
                limiter=None if M is None else "tvb",
                tvb_constant=1.0 if M is None else M,
            )
            _, nodes, basis, _, _ = dg2.legendre_operators(degree, 2 * degree + 1)
            errors = _vortex_error(coeff, basis, nodes, DOMAIN, final_time)
            entry = {
                "M": M if M is not None else "none",
                "ncells": ncells,
                "degree": degree,
                "steps": metadata["steps"],
                "wall_seconds": time.time() - started,
                "errors_percent": {k: 100.0 * v for k, v in errors.items()},
            }
            if previous is not None:
                entry["eoc"] = {
                    name: math.log(previous["errors_percent"][name] / entry["errors_percent"][name])
                    / math.log(ncells / previous["ncells"])
                    for name in ("density", "pressure")
                }
            rows.append(entry)
            previous = entry
    return rows


def shock_capturing(ncells=200, degree=2, final_time=0.2, cfl=0.2,
                    constants=(None, 0.0)):
    x_reference, reference = sod_cell_averages(ncells, final_time)
    rows = []
    for M in constants:
        started = time.time()
        x, _, coeff, metadata = dg2.solve_euler_dg_2d_sod(
            ncells=ncells,
            degree=degree,
            final_time=final_time,
            cfl=cfl,
            limiter=None if M is None else "tvb",
            tvb_constant=0.0 if M is None else M,
        )
        fields = dg2.cell_fields(coeff)
        density = fields["density"][:, 0].detach().cpu().numpy()
        velocity = fields["velocity_x"][:, 0].detach().cpu().numpy()
        pressure_np = fields["pressure"][:, 0].detach().cpu().numpy()
        errors = {}
        for name, numerical in (("density", density), ("velocity", velocity),
                                ("pressure", pressure_np)):
            absolute = float(np.sqrt(np.mean((numerical - reference[name]) ** 2)))
            norm = float(np.sqrt(np.mean(reference[name] ** 2)))
            errors[name] = {"relative_l2_percent": 100.0 * absolute / norm}
        troubled, _ = dg2._troubled_1d(coeff[..., 0, 0, :], 0, 1.0 / ncells, 0.0,
                                       periodic=False)
        rows.append({
            "M": M if M is not None else "none",
            "ncells": ncells,
            "degree": degree,
            "steps": metadata["steps"],
            "wall_seconds": time.time() - started,
            "errors": errors,
            "max_error_percent": max(v["relative_l2_percent"] for v in errors.values()),
            "flagged_cells_end": int(troubled.sum()),
            "min_density": float(np.min(density)),
            "max_density": float(np.max(density)),
            "profile": {
                "x": np.asarray(x).ravel().tolist(),
                "density": density.tolist(),
                "pressure": pressure_np.tolist(),
            },
        })
    return {
        "runs": rows,
        "reference": {
            "x": np.asarray(x_reference).tolist(),
            "density": np.asarray(reference["density"]).tolist(),
            "pressure": np.asarray(reference["pressure"]).tolist(),
        },
    }


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    colors = {None: "black", "none": "black", 1.0: "tab:orange", 10.0: "tab:green"}

    # 1) detector behaviour
    axis = axes[0]
    for ncells, marker in ((16, "o"), (24, "s")):
        rows = [r for r in report["detector"] if r["ncells"] == ncells]
        for key, style in (("historic", "--"), ("scale_consistent", "-")):
            axis.loglog(
                [r["M"] for r in rows],
                [max(r[f"{key}_fraction"], 1e-4) for r in rows],
                marker + style,
                label=f"{key} {ncells}$^2$",
            )
    axis.set_xlabel("TVB constant M")
    axis.set_ylabel("flagged cell fraction (smooth vortex)")
    axis.set_title("(a) detector: historic vs scale-consistent")
    axis.legend(fontsize=7)
    axis.grid(True, which="both", alpha=0.3)

    # 2) smooth accuracy
    axis = axes[1]
    for M in (None, 1.0, 10.0):
        rows = [r for r in report["vortex"] if r["M"] == (M if M is not None else "none")]
        axis.loglog(
            [1.0 / r["ncells"] for r in rows],
            [r["errors_percent"]["density"] for r in rows],
            "o-" if M is None else "o--",
            color=colors[M],
            label=f"M={M if M is not None else 'none'}",
        )
    axis.set_xlabel("1 / ncells")
    axis.set_ylabel("density relative $L^2$ error [%]")
    axis.set_title("(b) smooth isentropic vortex, P2")
    axis.legend(fontsize=7)
    axis.grid(True, which="both", alpha=0.3)

    # 3) shock profile
    axis = axes[2]
    reference = report["sod"]["reference"]
    axis.plot(reference["x"], reference["density"], "k-", lw=2.5, label="exact")
    for run, style in zip(report["sod"]["runs"], ("-.", "-")):
        profile = run["profile"]
        axis.plot(profile["x"], profile["density"], style, lw=1.0,
                  label=f"M={run['M']} ({run['max_error_percent']:.2f}%)")
    axis.set_xlabel("x")
    axis.set_ylabel("density")
    axis.set_title("(c) extruded Sod, P2, t=0.2")
    axis.legend(fontsize=7)
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller meshes")
    parser.add_argument("--outdir", default="validation/limiter_2d")
    arguments = parser.parse_args(argv)
    os.makedirs(arguments.outdir, exist_ok=True)

    report = {
        "parity": detector_parity(),
        "detector": detector_behaviour(
            meshes=(16, 24) if arguments.quick else (16, 24, 32)
        ),
        "vortex": vortex_accuracy(
            meshes=(12, 16) if arguments.quick else (16, 24),
            constants=(None, 1.0, 10.0),
        ),
        "sod": shock_capturing(ncells=100 if arguments.quick else 200),
        "cwd": os.getcwd(),
        "quick": arguments.quick,
    }
    figure_directory = os.path.join("docs", "figures")
    os.makedirs(figure_directory, exist_ok=True)
    report["figure"] = plot(
        report, os.path.join(figure_directory, "limiter_2d_verification.png")
    )
    path = os.path.join(arguments.outdir, "limiter_2d_results.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    print("[A] detector parity (2-D vs 1-D criterion, extruded data)")
    for row in report["parity"]:
        print(f"  n={row['ncells']:3d} P{row['degree']} M={row['M']:<5} "
              f"mask_equal={row['mask_equal']} mode1_diff={row['mode1_max_abs_diff']:.2e} "
              f"avg_preserved={row['average_preserved']} flagged={row['cells_flagged_1d']}")
    print("[B] troubled-cell fraction on a smooth vortex")
    for row in report["detector"]:
        print(f"  n={row['ncells']:3d} M={row['M']:<5} historic={row['historic_fraction']:.3f} "
              f"scale_consistent={row['scale_consistent_fraction']:.3f}")
    print("[C] smooth isentropic vortex")
    for row in report["vortex"]:
        eoc = row.get("eoc", {})
        print(f"  M={row['M']!s:<5} n={row['ncells']:3d} "
              f"rho={row['errors_percent']['density']:.4f}% "
              f"p={row['errors_percent']['pressure']:.4f}% "
              f"EOC(rho)={eoc.get('density', float('nan')):.3f} ({row['wall_seconds']:.1f}s)")
    print("[D] extruded Sod vs exact Riemann")
    for row in report["sod"]["runs"]:
        print(f"  M={row['M']!s:<5} max={row['max_error_percent']:.4f}% "
              f"rho_min={row['min_density']:.4f} flagged={row['flagged_cells_end']} "
              f"({row['wall_seconds']:.1f}s)")
    print(f"results: {path}\nfigure: {report['figure']}")


if __name__ == "__main__":
    main()