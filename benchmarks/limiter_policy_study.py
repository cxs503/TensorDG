"""Limiter *application policy* verification harness (TensorDG).

Motivation
----------
``docs/limiter_verification.md`` closed the scaling bug of the TVB/WENO
limiters but left two open verification items:

* **P0** the limiting *policy*: applying the limiter after every SSP-RK3
  sub-step degrades accuracy inside smooth regions (Sod velocity error
  3.20% -> 4.79%), so the default path was reported as harmful;
* **P1** the 3% verdict had no resolution meaning (a single 200-cell mesh).

This harness turns both into reproducible measurements with machine verdicts,
using only quantities that are defined before the runs:

1. **Smooth periodic entropy wave (exact solution available).**
   ``rho = 1 + 0.2 sin(2 pi x)``, ``u = 1``, ``p = 1`` on a periodic domain is
   an exact solution of the 1D Euler equations: with constant velocity and
   pressure the momentum and energy equations reduce to the scalar advection
   equation, so ``rho(x, t) = rho0(x - u t)``.  Observed order of accuracy is
   therefore measured against an *analytic* reference (no numerical
   reference), on a mesh sequence 20/40/80/160 cells, for each policy.
   The design order for P2 modal DG is ``p + 1 = 3``.
2. **Sod shock tube on a mesh sequence 100/200/400 cells** against the exact
   Riemann solution, so "is it below 3%?" is answered per resolution.
3. **Discrete conservation** of mass / momentum / energy (periodic case): the
   drift must stay at round-off level, a limiter must not leak conserved
   quantities.

Usage
-----
    PYTHONPATH=. python -m benchmarks.limiter_policy_study
    PYTHONPATH=. python -m benchmarks.limiter_policy_study --quick

Results: ``validation/limiter_policy/policy_study.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.riemann_dg_error import riemann_dg_errors
from tensordg.euler_dg_1d import solve_euler_dg_1d, solve_periodic_euler_dg_1d

OUT_DIR = Path("validation/limiter_policy")

SOD_LEFT = (1.0, 0.0, 1.0)
SOD_RIGHT = (0.125, 0.0, 0.1)
SOD_FINAL_TIME = 0.2

# Smooth entropy wave: exact solution is the translated initial profile.
WAVE_AMPLITUDE = 0.2
WAVE_VELOCITY = 1.0
WAVE_PRESSURE = 1.0
WAVE_FINAL_TIME = 0.1
WAVE_CFL = 0.12

# (label, limiter kind, limiter_stage)
# The WENO limiter is ~3x more expensive than TVB (pure-Python per-cell
# stencils), so it is measured on the shock-tube path only.
# (label, limiter kind, limiter_stage, TVB constant)
WAVE_POLICIES = [
    ("none", None, "every", 1.0),
    ("tvb_M1", "tvb", "every", 1.0),
    ("tvb_M10", "tvb", "every", 10.0),
    ("tvb_M10_final", "tvb", "final", 10.0),
]
SOD_POLICIES = [
    ("none", None, "every", 1.0),
    ("tvb_M1", "tvb", "every", 1.0),
    ("tvb_M10", "tvb", "every", 10.0),
    ("weno_M10", "weno", "every", 10.0),
]

SOD_CELLS = (100, 200)
WAVE_CELLS = (20, 40, 80)
SOD_CELLS_QUICK = (100,)
WAVE_CELLS_QUICK = (20, 40)


def _wave_initial_state(x):
    rho = 1.0 + WAVE_AMPLITUDE * torch.sin(2.0 * np.pi * x)
    shape = x.shape
    return (
        rho,
        torch.full(shape, WAVE_VELOCITY, dtype=x.dtype, device=x.device),
        torch.full(shape, WAVE_PRESSURE, dtype=x.dtype, device=x.device),
    )


def _wave_exact(x, time):
    return 1.0 + WAVE_AMPLITUDE * np.sin(
        2.0 * np.pi * (x - WAVE_VELOCITY * time)
    )


def _wave_error(x, coeff, time, quadrature_order=8):
    """Relative L2 of the density field against the exact translated wave."""
    values = coeff.detach().cpu().numpy()
    centres = x.detach().cpu().numpy()
    cells, modes, _ = values.shape
    dx = (centres[-1] - centres[0]) / (cells - 1)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    phi = np.stack(
        [
            np.polynomial.legendre.legval(nodes, np.pad([1.0], (k, 0)))
            for k in range(modes)
        ],
        axis=1,
    )
    numerical = np.einsum("cmv,qm->cqv", values, phi)[..., 0]
    points = centres[:, None] + 0.5 * dx * nodes[None, :]
    exact = _wave_exact(points, time)
    error2 = 0.5 * dx * np.sum(weights[None, :] * (numerical - exact) ** 2)
    norm2 = 0.5 * dx * np.sum(weights[None, :] * exact ** 2)
    return float(100.0 * np.sqrt(error2 / norm2))


def _orders(rows):
    """Observed order between successive meshes (log-log slope)."""
    orders = []
    for previous, current in zip(rows[:-1], rows[1:]):
        ratio = previous["relative_l2_percent"] / current["relative_l2_percent"]
        order = np.log(ratio) / np.log(current["cells"] / previous["cells"])
        orders.append(float(order))
    return orders


def run_wave(degree, cells_list):
    rows = {}
    for label, kind, stage, tvb in WAVE_POLICIES:
        entries = []
        for cells in cells_list:
            start = time.perf_counter()
            x, coeff, meta = solve_periodic_euler_dg_1d(
                _wave_initial_state, ncells=cells, degree=degree,
                final_time=WAVE_FINAL_TIME, cfl=WAVE_CFL,
                limiter=kind, limiter_stage=stage, tvb_constant=tvb,
            )
            elapsed = time.perf_counter() - start
            error = _wave_error(x, coeff, WAVE_FINAL_TIME)
            entries.append({
                "cells": cells,
                "relative_l2_percent": error,
                "steps": meta["steps"],
                "seconds": round(elapsed, 3),
                "conservation_drift": meta["conservation_drift"],
            })
            print(f"[wave] {label:10s} cells={cells:4d} "
                  f"relL2={error:.6e}% steps={meta['steps']} {elapsed:.2f}s", flush=True)
        orders = _orders(entries)
        finest_order = orders[-1] if orders else float("nan")
        rows[label] = {
            "limiter": kind, "limiter_stage": stage, "tvb_constant": tvb,
            "degree": degree, "rows": entries, "observed_orders": orders,
            "finest_order": finest_order,
        }
    return rows


def run_sod(degree, cells_list):
    rows = {}
    for label, kind, stage, tvb in SOD_POLICIES:
        entries = []
        for cells in cells_list:
            start = time.perf_counter()
            x, coeff, meta = solve_euler_dg_1d(
                SOD_LEFT, SOD_RIGHT, ncells=cells, degree=degree,
                final_time=SOD_FINAL_TIME, limiter=kind, limiter_stage=stage,
                tvb_constant=tvb,
            )
            elapsed = time.perf_counter() - start
            errors = riemann_dg_errors(x, coeff, SOD_FINAL_TIME, SOD_LEFT, SOD_RIGHT)
            percents = {key: errors[key]["relative_l2_percent"]
                        for key in ("density", "velocity", "pressure")}
            entries.append({
                "cells": cells,
                "steps": meta["steps"],
                "seconds": round(elapsed, 3),
                **{f"{key}_percent": value for key, value in percents.items()},
                "max_percent": max(percents.values()),
            })
            print(f"[sod ] {label:10s} cells={cells:4d} "
                  f"rho={percents['density']:.4f}% u={percents['velocity']:.4f}% "
                  f"p={percents['pressure']:.4f}% {elapsed:.2f}s", flush=True)
        rows[label] = {
            "limiter": kind, "limiter_stage": stage, "tvb_constant": tvb,
            "degree": degree, "rows": entries,
        }
    return rows


def verdicts(wave, sod, degree):
    design_order = degree + 1
    out = {}
    baseline = wave["none"]["finest_order"]
    for label, data in wave.items():
        order = data["finest_order"]
        worst_drift = max(
            max(entry["conservation_drift"].values()) for entry in data["rows"]
        )
        out[f"wave::{label}"] = {
            "finest_order": order,
            "design_order": design_order,
            "order_loss_vs_design": float(design_order - order),
            "order_ratio_vs_unlimited": float(order / baseline) if baseline else None,
            "max_conservation_drift": worst_drift,
            "conservation_ok": worst_drift < 1e-10,
            "order_ok": bool(order >= 0.9 * design_order),
        }
    for label, data in sod.items():
        for entry in data["rows"]:
            out[f"sod::{label}::{entry['cells']}"] = {
                "max_percent": entry["max_percent"],
                "below_three_percent": entry["max_percent"] < 3.0,
            }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--quick", action="store_true",
                        help="smaller mesh sequences for a fast smoke run")
    parser.add_argument("--tag", type=str, default="policy_study")
    parser.add_argument("--wave-only", action="store_true",
                        help="skip the Sod mesh sequence (fast indicator check)")
    args = parser.parse_args()

    wave_cells = WAVE_CELLS_QUICK if args.quick else WAVE_CELLS
    sod_cells = SOD_CELLS_QUICK if args.quick else SOD_CELLS

    start = time.perf_counter()
    wave = run_wave(args.degree, wave_cells)
    sod = {} if args.wave_only else run_sod(args.degree, sod_cells)
    total = time.perf_counter() - start

    payload = {
        "tag": args.tag,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "loadavg": os.getloadavg(),
        "degree": args.degree,
        "smooth_wave": {
            "description": "periodic entropy wave rho=1+0.2 sin(2 pi x), u=1, p=1 "
                           "(exact solution: translated profile)",
            "final_time": WAVE_FINAL_TIME, "cfl": WAVE_CFL,
            "configurations": wave,
        },
        "sod": {
            "left": SOD_LEFT, "right": SOD_RIGHT, "final_time": SOD_FINAL_TIME,
            "configurations": sod,
        },
        "verdicts": verdicts(wave, sod, args.degree),
        "total_seconds": round(total, 2),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.tag}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwrote {path} ({total:.1f}s)")

    print("\n=== verdicts ===")
    for key, value in sorted(payload["verdicts"].items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()