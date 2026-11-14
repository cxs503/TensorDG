"""CASE-EULER-VORTEX: 2-D isentropic Euler vortex against its exact solution.

Case-pool entry ``EULER-VORTEX`` fixes the reference (the *exact* isentropic
vortex of Shu 1998 §4.4 / C.-W. Shu, ICASE Report 97-65) and the acceptance
tolerance (every primitive field relative-L2 < 5 %).  This module is the
machine-checkable conformance harness for that entry: it runs one
configuration per invocation, checkpoints the record to ``<output>/results``,
and aggregates the ladder into tables, figures and a pass/fail gate.

Design notes
------------
* Two families are exercised, because they use two *independent* flux /
  quadrature paths in the package:
  - ``p0``  : :func:`tensordg.euler_2d.solve_isentropic_vortex`
              (P0 / Rusanov / SSP-RK3, cell-average metric)
  - ``dg``  : :func:`tensordg.euler_dg_2d.solve_euler_dg_2d_vortex`
              (modal DG / HLLC / SSP-RK3, volume-quadrature metric)
  Both metrics are relative L2 against the exact solution at the final time,
  so the two families can be compared on one axis.
* One task per process keeps the ladder embarrassingly parallel and makes a
  killed process cost only its own task, never the whole study.
* The exact solution itself is verified independently (``--task residual``):
  the conservative state is differentiated with autograd and the Euler
  residual ``U_t + F_x + G_y`` must vanish to round-off.  A "reference" that
  is not actually an exact solution would invalidate every error number
  below, so this check is part of the evidence bundle.

Usage
-----
Single configuration::

    python benchmarks/case_euler_vortex.py --task p0 --cells 64 --final-time 1.0
    python benchmarks/case_euler_vortex.py --task dg --degree 2 --cells 32

Ladder + gate + figures + report tables::

    python benchmarks/case_euler_vortex.py --aggregate --output-dir case_euler_vortex
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensordg import euler_dg_2d as dg2
from tensordg.euler_1d import GAMMA
from tensordg.euler_2d import solve_isentropic_vortex, vortex_error, vortex_primitive
from benchmarks.verify_euler_dg_2d import _vortex_error as dg_vortex_error

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
STRENGTH = 5.0
BACKGROUND = (1.0, 1.0)
FIELDS = ("density", "velocity_x", "velocity_y", "pressure")
CONSERVATION_NAMES = ("mass", "x_momentum", "y_momentum", "energy")

#: Case-pool tolerance: every field must be < 5 % relative L2.
TOLERANCE = 0.05

#: Reference citation for both the exact solution and the tolerance.
REFERENCE = (
    "C.-W. Shu, 'Essentially non-oscillatory and weighted essentially "
    "non-oscillatory schemes for hyperbolic conservation laws', ICASE "
    "Report 97-65 / NASA CR-97-206253 (1998), §4.4.2 smooth isentropic "
    "vortex: domain [-5,5]^2, beta=5, (u,v)_inf=(1,1), gamma=1.4, periodic, "
    "exact solution; tolerance <5% relative L2 per primitive field."
)

#: Configurations the acceptance gate is evaluated on (the "gate set").
GATE_SET = (
    {"task": "dg", "degree": 2, "cells": 32, "final_time": 1.0, "cfl": 0.2},
    {"task": "p0", "cells": 128, "final_time": 1.0, "cfl": 0.4},
)

#: Resolution ladders and sensitivity sweeps of the study.
LADDER = (
    # low-order baseline: first-order in h, needs a fine mesh for 5 %
    *({"task": "p0", "cells": n, "final_time": 1.0, "cfl": 0.4} for n in (32, 64, 96, 128)),
    # high-order: P1 catches up at coarse meshes, P2/P3 are orders below the gate
    *({"task": "dg", "degree": 1, "cells": n, "final_time": 1.0, "cfl": 0.2} for n in (16, 32, 64)),
    *({"task": "dg", "degree": 2, "cells": n, "final_time": 1.0, "cfl": 0.2} for n in (16, 32, 48)),
    *({"task": "dg", "degree": 3, "cells": n, "final_time": 1.0, "cfl": 0.2} for n in (12, 16, 24)),
)

#: Parameter sensitivity: CFL and final time at the gate configuration.
SENSITIVITY = (
    *({"task": "dg", "degree": 2, "cells": 32, "final_time": 1.0, "cfl": c} for c in (0.1, 0.2, 0.4)),
    *({"task": "dg", "degree": 2, "cells": 32, "final_time": t, "cfl": 0.2} for t in (0.5, 2.0)),
)


def task_key(config: dict) -> str:
    """Stable, filesystem-safe identifier for one configuration."""
    if config["task"] == "dg":
        return (f"dg_d{config['degree']}_n{config['cells']}"
                f"_t{config['final_time']:g}_cfl{config['cfl']:g}")
    if config["task"] == "p0":
        return (f"p0_n{config['cells']}"
                f"_t{config['final_time']:g}_cfl{config['cfl']:g}")
    if config["task"] == "residual":
        return "exact_solution_residual"
    raise ValueError(f"unknown task {config['task']!r}")


# ---------------------------------------------------------------------------
# exact-solution self-verification
# ---------------------------------------------------------------------------

def _conservative_and_flux(x, y, time, background, strength=STRENGTH):
    rho, u, v, p = vortex_primitive(x, y, time, strength, background)
    energy = p / (GAMMA - 1.0) + 0.5 * rho * (u ** 2 + v ** 2)
    state = (rho, rho * u, rho * v, energy)
    flux_x = (rho * u, rho * u ** 2 + p, rho * u * v, u * (energy + p))
    flux_y = (rho * v, rho * u * v, rho * v ** 2 + p, v * (energy + p))
    return state, flux_x, flux_y, (rho, u, v, p)


def _divergence(flux_x, flux_y, x, y):
    """Pointwise ``d(F)/dx + d(G)/dy`` per conservative component (autograd)."""
    out = []
    for component in range(4):
        d_dx = torch.autograd.grad(flux_x[component].sum(), x, retain_graph=True)[0]
        d_dy = torch.autograd.grad(flux_y[component].sum(), y, retain_graph=True)[0]
        out.append((d_dx + d_dy, d_dx.abs() + d_dy.abs()))
    return out


def _relative(worst_abs: float, scale: float) -> float:
    return worst_abs / scale if scale > 0.0 else float("inf")


def steady_residual(points: int = 201, span: float = 4.0, dtype=torch.float64):
    """Steady frame: the vortex must satisfy ``div F = 0`` (no background flow).

    This is the mathematically decisive check on the *reference*: if the
    analytic profile is not an exact steady solution, every error number in
    this study would be measured against the wrong answer.
    """
    x = torch.linspace(-span, span, points, dtype=dtype).requires_grad_(True)
    y = torch.linspace(-span, span, points, dtype=dtype).requires_grad_(True)
    _, flux_x, flux_y, _ = _conservative_and_flux(x, y, 0.0, (0.0, 0.0))
    worst = 0.0
    per_component = {}
    for name, (net, gross) in zip(("mass", "momentum_x", "momentum_y", "energy"),
                                 _divergence(flux_x, flux_y, x, y)):
        value = _relative(float(net.abs().max().item()), float(gross.max().item()))
        per_component[name] = value
        worst = max(worst, value)
    return worst, per_component


def spacetime_residual(points: int = 201, span: float = 4.0,
                       final_time: float = 1.0, step: float = 1e-5,
                       dtype=torch.float64):
    """Full residual ``U_t + F_x + G_y`` for the advected vortex.

    The time derivative is a central difference of the *analytic* field (the
    spatial derivatives stay exact via autograd).  Autograd through the scalar
    ``time`` argument is not usable in this package build: a 0-dim leaf mixed
    with float64 tensors loses its gradient on the loongarch torch build, so
    the ``d/dt`` term is differenced instead -- truncation ``O(step^2)`` is
    ~1e-10 relative here, far below the 5 % acceptance scale.
    """
    x = torch.linspace(-span, span, points, dtype=dtype).requires_grad_(True)
    y = torch.linspace(-span, span, points, dtype=dtype).requires_grad_(True)
    plus = _conservative_and_flux(x, y, final_time + step, BACKGROUND)[0]
    minus = _conservative_and_flux(x, y, final_time - step, BACKGROUND)[0]
    _, flux_x, flux_y, _ = _conservative_and_flux(x, y, final_time, BACKGROUND)
    worst = 0.0
    per_component = {}
    for name, index, (net, gross) in zip(("mass", "momentum_x", "momentum_y", "energy"),
                                         range(4), _divergence(flux_x, flux_y, x, y)):
        d_dt = (plus[index] - minus[index]) / (2.0 * step)
        value = _relative(float((d_dt + net).abs().max().item()),
                          float(gross.max().item()))
        per_component[name] = value
        worst = max(worst, value)
    return worst, per_component


def solution_properties(points: int = 201, span: float = 4.0, dtype=torch.float64,
                        strength: float = STRENGTH):
    """Independent characterisations the exact vortex must satisfy exactly."""
    x = torch.linspace(-span, span, points, dtype=dtype).requires_grad_(True)
    y = torch.linspace(-span, span, points, dtype=dtype).requires_grad_(True)
    rho, u, v, p = vortex_primitive(x, y, 0.0, strength, (0.0, 0.0))
    # 1. isentropic law p = rho^gamma
    isentropic = float((p - rho ** GAMMA).abs().max().item()) / float(p.abs().max().item())
    # 2. rotational part is divergence free (incompressible swirl)
    dudx = torch.autograd.grad(u.sum(), x, retain_graph=True)[0]
    dvdy = torch.autograd.grad(v.sum(), y, retain_graph=True)[0]
    divergence = float((dudx + dvdy).abs().max().item()) / float(dudx.abs().max().item())
    # 3. radial momentum balance dp/dr = rho * v_theta^2 / r on a radial ray
    theta = math.pi / 5.0
    radius = torch.linspace(0.2, 4.0, 400, dtype=dtype, requires_grad=True)
    ray_x = radius * math.cos(theta)
    ray_y = radius * math.sin(theta)
    rho_r, u_r, v_r, p_r = vortex_primitive(ray_x, ray_y, 0.0, strength, (0.0, 0.0))
    dp_dr = torch.autograd.grad(p_r.sum(), radius, retain_graph=True)[0]
    v_theta = v_r * math.cos(theta) - u_r * math.sin(theta)
    balance = (dp_dr - rho_r * v_theta ** 2 / radius).abs()
    radial = float(balance.max().item()) / float(dp_dr.abs().max().item())
    return {"isentropic_law": isentropic, "divergence_free": divergence,
            "radial_balance": radial}


def run_residual():
    steady, steady_parts = steady_residual()
    spacetime, spacetime_parts = spacetime_residual()
    properties = solution_properties()
    worst = max([steady, spacetime, *properties.values()])
    return {
        "task": "residual",
        "steady_frame_residual": steady,
        "steady_frame_per_component": steady_parts,
        "spacetime_residual": spacetime,
        "spacetime_per_component": spacetime_parts,
        "solution_properties": properties,
        "max_relative_residual": worst,
        "max_relative_l2": worst,
        "gate_pass": steady < 1e-12 and spacetime < 1e-8 and max(properties.values()) < 1e-8,
        "formula": ("steady: max|div F| / max(|F_x|+|F_y|); space-time: "
                    "max|U_t+F_x+G_y| / max(|F_x|+|F_y|) with central-difference U_t"),
    }


# ---------------------------------------------------------------------------
# one configuration per invocation
# ---------------------------------------------------------------------------

def run_p0(config: dict):
    started = time.time()
    # Conservation tracking via observer hook
    _p0_totals_initial = None
    _p0_totals_final = None
    def _p0_observer(t, steps, u):
        nonlocal _p0_totals_initial, _p0_totals_final
        # u shape: (nx, ny, 4) - sum over spatial dims, keep 4 components
        cell_area = (10.0 / config["cells"]) ** 2
        totals = (u.sum(dim=(0, 1)) * cell_area).clone()
        if steps == 0:
            _p0_totals_initial = totals
        _p0_totals_final = totals
    x, y, fields, metadata = solve_isentropic_vortex(
        ncells=config["cells"], final_time=config["final_time"],
        cfl=config["cfl"], strength=config.get("strength", STRENGTH),
        observer=_p0_observer,
    )
    errors = vortex_error(x, y, fields["conservative"], config["final_time"],
                          config.get("strength", STRENGTH), BACKGROUND)
    wall = time.time() - started
    record = {
        "task": "p0",
        "cells": config["cells"],
        "dx": 10.0 / config["cells"],
        "degree": 0,
        "steps": metadata["steps"],
        "dt": metadata["dt"],
        "errors": {name: errors[name]["relative_l2"] for name in FIELDS},
        "metric": "cell-average relative L2 (4x4 exact cell quadrature)",
        "fields_np": (fields["density"].cpu().numpy(), x.cpu().numpy(), y.cpu().numpy()),
    }
    # Conservation drift
    if _p0_totals_initial is not None and _p0_totals_final is not None:
        cons_drift = {}
        for i, name in enumerate(CONSERVATION_NAMES):
            init_val = float(_p0_totals_initial[i].item())
            final_val = float(_p0_totals_final[i].item())
            denom = max(abs(init_val), 1e-30)
            cons_drift[name] = abs(final_val - init_val) / denom
        record["conservation_drift"] = cons_drift
        record["max_conservation_drift"] = max(cons_drift.values())
    else:
        record["conservation_drift"] = {name: float("nan") for name in CONSERVATION_NAMES}
        record["max_conservation_drift"] = float("nan")
    record["wall_seconds"] = wall
    return record


def run_dg(config: dict):
    started = time.time()
    degree = config["degree"]
    nodes, weights, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    _, _, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
        ncells=config["cells"], degree=degree, final_time=config["final_time"],
        cfl=config["cfl"], strength=config.get("strength", STRENGTH),
        domain=DOMAIN,
    )
    errors = dg_vortex_error(coeff, basis, nodes, DOMAIN, config["final_time"],
                             config.get("strength", STRENGTH), BACKGROUND)
    averages = dg2.cell_fields(coeff)
    wall = time.time() - started
    record = {
        "task": "dg",
        "degree": degree,
        "cells": config["cells"],
        "dx": 10.0 / config["cells"],
        "steps": metadata["steps"],
        "dt": metadata["dt"],
        "dofs": metadata["dofs"],
        "errors": {name: float(errors[name]) for name in FIELDS},
        "metric": "volume-quadrature relative L2 (L2-projected modal state)",
        "fields_np": (averages["density"].cpu().numpy(),
                      np.linspace(-5.0 + 5.0 / config["cells"], 5.0 - 5.0 / config["cells"],
                                  config["cells"]),
                      np.linspace(-5.0 + 5.0 / config["cells"], 5.0 - 5.0 / config["cells"],
                                  config["cells"])),
    }
    record["wall_seconds"] = wall
    # Conservation drift from solver metadata
    cons_drift = metadata.get("conservation_drift", {})
    record["conservation_drift"] = {name: cons_drift.get(name, float("nan")) for name in CONSERVATION_NAMES}
    record["max_conservation_drift"] = max(
        (v for v in record["conservation_drift"].values() if not isinstance(v, float) or not __import__("math").isnan(v)),
        default=0.0,
    )
    return record


def run_task(config: dict) -> dict:
    if config["task"] == "residual":
        return run_residual()
    runner = run_p0 if config["task"] == "p0" else run_dg
    record = runner(config)
    for name in ("final_time", "cfl", "strength"):
        record[name] = config.get(name, STRENGTH if name == "strength" else None)
    record["max_relative_l2"] = max(record["errors"].values())
    record["max_percent"] = 100.0 * record["max_relative_l2"]
    record["gate_pass"] = record["max_relative_l2"] < TOLERANCE
    record["reference"] = REFERENCE
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("p0", "dg", "residual"), default="dg")
    parser.add_argument("--cells", type=int, default=32)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--strength", type=float, default=STRENGTH)
    parser.add_argument("--output-dir", default="case_euler_vortex")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--save-fields", action="store_true",
                        help="store the density field of this run as .npz for plotting")
    parser.add_argument("--aggregate", action="store_true",
                        help="build tables/figures/gate from <output>/results and exit")
    args = parser.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    output = Path(args.output_dir)
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        from benchmarks._case_euler_vortex_report import aggregate
        return aggregate(output)

    config = {"task": args.task, "cells": args.cells, "degree": args.degree,
              "final_time": args.final_time, "cfl": args.cfl, "strength": args.strength}
    key = task_key(config)
    record = run_task(config)
    record["config"] = config
    record["task_key"] = key
    fields_np = record.pop("fields_np", None)
    if fields_np is not None and args.save_fields:
        np.savez(results / f"{key}_fields.npz", density=fields_np[0],
                 x=fields_np[1], y=fields_np[2])
    with open(results / f"{key}.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    if record.get("errors"):
        print(key, " ".join(f"{k}={100.0 * v:.5f}%" for k, v in record["errors"].items()),
              f"max={record['max_percent']:.4f}%", f"gate={'PASS' if record['gate_pass'] else 'FAIL'}",
              f"steps={record['steps']}", f"wall={record['wall_seconds']:.1f}s")
    else:
        print(key, f"steady={record['steady_frame_residual']:.3e}",
              f"spacetime={record['spacetime_residual']:.3e}",
              " ".join(f"{k}={v:.2e}" for k, v in record['solution_properties'].items()),
              f"gate={'PASS' if record['gate_pass'] else 'FAIL'}")
    return 0 if record["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())