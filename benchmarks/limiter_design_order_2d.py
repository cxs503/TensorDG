"""Does the 2-D limiter keep the design order on *smooth* flow?

The repository measures the 2-D TVB/WENO limiter only on shocked data
(``docs/limiter_2d_verification.md``: squeezed Sod, Lax-Liu four-quadrant) and
the 1-D limiter's accuracy debt only in one dimension
(``docs/limiter_policy_verification.md``: smooth entropy wave).  The 2-D claim
"use the limiter on a smooth problem only with ``M`` tuned up, or not at all"
was therefore never quantified in 2-D: nobody knew whether a given
``tvb_constant`` silently caps the observed order of the workhorse solver
``solve_euler_dg_2d``.

This benchmark closes that gap with an *analytic* reference.  The isentropic
vortex of ``solve_euler_dg_2d_vortex`` is an exact solution of the 2-D Euler
equations advected across a periodic box, so the error of ``(rho, u, v, p)`` is
measured against the closed-form solution -- no manufactured source term and no
numerical reference solution.

For every (degree, mesh, limiter configuration) it reports

* the relative L2 error of each primitive field against the exact solution,
* the adjacent-mesh observed order (EOC),
* the *troubled-cell fraction* actually flagged by the detector, averaged over
  every limiter call of the run (``mean``) and its peak (``peak``),
* the relative mass drift (the limiter must never touch the cell average).

The theory being checked (Cockburn & Shu, Math. Comp. 52 (1989) 411-435): the
TVB test compares the deviation of the one-sided difference from the minmod
slope, an ``O(h**2 * u'')`` quantity for smooth data, against the tolerance
``M h**2``.  Smooth data are therefore left untouched -- and the design order
``p + 1`` recovered -- as soon as ``M > max|u''|`` (in the units of the
conserved variables, with ``h`` the cell size), while any smaller ``M`` flags a
mesh-independent set of cells and caps the observed order near 2.  The script
also measures ``max|u''|`` of the initial data directly (fine-grid central
differences) so the predicted threshold sits next to the measured one instead
of being a hand-wave.

Run::

    PYTHONPATH=. python -m benchmarks.limiter_design_order_2d            # full sweep
    PYTHONPATH=. python -m benchmarks.limiter_design_order_2d --degrees 1 --meshes 8 16 --configs none tvb1

Numbers go to ``validation/limiter_design_order_2d/sweep.json`` (rewritten after
every run, so an interrupted sweep still leaves usable evidence).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import tensordg.euler_dg_2d as dg2
from benchmarks.verify_euler_dg_2d import _vortex_error

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
STRENGTH = 5.0
BACKGROUND = (1.0, 1.0)

#: name -> (limiter keyword, tvb_constant M)
CONFIGS = {
    "none": (None, 0.0),
    "tvb_M1": ("tvb", 1.0),
    "tvb_M10": ("tvb", 10.0),
    "tvb_M100": ("tvb", 100.0),
    "weno_M1": ("weno", 1.0),
    "weno_M100": ("weno", 100.0),
}


def flagged_fraction(coeff, dx, dy, tvb_constant):
    """Fraction of (cell, axis) pairs the TVB/WENO detector would flag.

    Uses the production detector :func:`tensordg.euler_dg_2d._troubled_1d` on
    both axes, so the number reported here is the mask the limiter acts on and
    not a re-implementation.
    """
    averages = coeff[..., 0, 0, :]
    flagged = 0
    total = 0
    for axis, spacing in ((0, dx), (1, dy)):
        troubled, _ = dg2._troubled_1d(averages, axis, spacing, tvb_constant, True)
        flagged += int(troubled.sum().item())
        total += int(troubled.numel())
    return flagged / total if total else 0.0


class _Activity:
    """Record the troubled-cell fraction of every limiter call in one run."""

    def __init__(self):
        self.samples: list[float] = []

    def mean(self):
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    def peak(self):
        return max(self.samples) if self.samples else 0.0


def run_one(degree, ncells, config, final_time, cfl, dtype=torch.float64):
    """Solve the vortex once and return its error / order / limiter-activity row."""
    limiter, tvb_constant = CONFIGS[config]
    nodes, _, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    activity = _Activity()

    original_tvb = dg2.apply_tvb_limiter_2d
    original_weno = dg2.apply_weno_limiter_2d

    def counting(fn):
        def wrapper(coeff, dx, dy, tvb_constant=1.0, periodic=(True, True)):
            activity.samples.append(flagged_fraction(coeff, dx, dy, tvb_constant))
            return fn(coeff, dx, dy, tvb_constant, periodic)

        return wrapper

    dg2.apply_tvb_limiter_2d = counting(original_tvb)
    dg2.apply_weno_limiter_2d = counting(original_weno)
    started = time.time()
    try:
        _, _, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
            ncells=ncells,
            degree=degree,
            final_time=final_time,
            cfl=cfl,
            strength=STRENGTH,
            background_velocity=BACKGROUND,
            limiter=limiter,
            tvb_constant=tvb_constant,
            domain=DOMAIN,
            dtype=dtype,
        )
    finally:
        dg2.apply_tvb_limiter_2d = original_tvb
        dg2.apply_weno_limiter_2d = original_weno
    wall = time.time() - started

    errors = _vortex_error(
        coeff, basis, nodes, DOMAIN, final_time, strength=STRENGTH, background=BACKGROUND
    )
    # Mass is the sum of the (never limited) cell averages; normalise by it.
    (xmin, xmax), (ymin, ymax) = DOMAIN
    dx, dy = (xmax - xmin) / ncells, (ymax - ymin) / ncells
    mass = float(coeff[..., 0, 0, 0].sum().item()) * dx * dy
    return {
        "config": config,
        "degree": degree,
        "cells": ncells,
        "h": (xmax - xmin) / ncells,
        "dofs": metadata["dofs"],
        "steps": metadata["steps"],
        "dt": metadata["dt"],
        "wall_seconds": wall,
        "errors": errors,
        "flagged_mean": activity.mean(),
        "flagged_peak": activity.peak(),
        "limiter_calls": len(activity.samples),
        "mass": mass,
        "mode2_energy": float(coeff[..., 2:, :, :].abs().sum().item()) if degree >= 2 else 0.0,
    }


def second_derivative_scale(samples=2001):
    """``max|u''|`` of the initial vortex data (fine-grid central differences).

    The TVB tolerance is ``M h**2`` against a deviation that equals
    ``|u''| h**2`` on smooth data, so the *predicted* limiter-silent threshold
    is ``M > max|u''|``.  The second derivatives of the primitive fields are
    taken with a uniform spacing of ``10 / (samples - 1)`` along the x axis
    through the vortex centre (the field is symmetric in y, and the y direction
    adds nothing new).
    """
    axis = torch.linspace(DOMAIN[0][0], DOMAIN[0][1], samples, dtype=torch.float64)
    y = torch.zeros_like(axis)
    rho, u, v, p = dg2.vortex_primitive(axis, y, 0.0, STRENGTH, BACKGROUND)
    h = float(axis[1] - axis[0])
    out = {}
    for name, field in (("density", rho), ("velocity_x", u), ("velocity_y", v), ("pressure", p)):
        second = (field[2:] - 2.0 * field[1:-1] + field[:-2]) / (h * h)
        out[name] = float(second.abs().max().item())
    # Conservative variables are what the detector actually sees.
    momentum_x = rho * u
    momentum_y = rho * v
    total = p / 0.4 + 0.5 * rho * (u * u + v * v)
    for name, field in (("momentum_x", momentum_x), ("momentum_y", momentum_y), ("energy", total)):
        second = (field[2:] - 2.0 * field[1:-1] + field[:-2]) / (h * h)
        out[name] = float(second.abs().max().item())
    out["spacing"] = h
    out["max_over_conserved"] = max(out["density"], out["momentum_x"], out["momentum_y"], out["energy"])
    return out


def eoc(previous: float, current: float, ratio: float):
    if previous <= 0.0 or current <= 0.0:
        return None
    return math.log(previous / current) / math.log(ratio)


def summarise(rows, expected_order):
    """Attach adjacent-mesh EOC (per field) and the design-order verdict."""
    by_key = {}
    for row in rows:
        by_key.setdefault((row["config"], row["degree"]), []).append(row)
    for (config, degree), group in by_key.items():
        group.sort(key=lambda item: item["cells"])
        for index, row in enumerate(group):
            if index == 0:
                row["eoc"] = {name: None for name in row["errors"]}
                continue
            previous = group[index - 1]
            ratio = row["cells"] / previous["cells"]
            row["eoc"] = {
                name: eoc(previous["errors"][name], row["errors"][name], ratio)
                for name in row["errors"]
            }
        finest = group[-1]
        density_eoc = finest["eoc"].get("density")
        finest["design_order"] = degree + 1
        finest["meets_design_order"] = (
            density_eoc is not None and density_eoc >= (degree + 1) - 0.3
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--meshes", type=int, nargs="+", default=None,
                        help="cell counts per side; defaults per degree below")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--final-time", type=float, default=0.5)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "validation" / "limiter_design_order_2d" / "sweep.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    default_meshes = {1: [16, 24, 32, 48], 2: [12, 16, 24, 32]}
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case": "isentropic vortex, analytic exact solution",
        "domain": DOMAIN,
        "strength": STRENGTH,
        "background_velocity": BACKGROUND,
        "final_time": args.final_time,
        "cfl": args.cfl,
        "dtype": "float64",
        "second_derivative_scale": second_derivative_scale(),
        "runs": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = payload["runs"]
    for degree in args.degrees:
        meshes = args.meshes or default_meshes[degree]
        for config in args.configs:
            for ncells in meshes:
                row = run_one(degree, ncells, config, args.final_time, args.cfl)
                rows.append(row)
                summarise(rows, degree + 1)
                payload["runs"] = rows
                args.out.write_text(json.dumps(payload, indent=2))
                if not args.quiet:
                    eoc_text = "".join(
                        f" {name}:{value:.2f}" if value is not None else f" {name}:  - "
                        for name, value in row["eoc"].items()
                    )
                    print(
                        f"P{degree} {config:9s} {ncells:3d}^2  "
                        f"rho {row['errors']['density']:.3e}  "
                        f"flag mean {row['flagged_mean']:.3f} peak {row['flagged_peak']:.3f}  "
                        f"EOC{eoc_text}  ({row['wall_seconds']:.1f}s)",
                        flush=True,
                    )
    return payload


if __name__ == "__main__":
    main()