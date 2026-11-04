"""Verification harness for the P0 two-dimensional Euler path (``euler_2d``).

Five checks, each with a machine-decidable criterion and an external reference
(no reference value is produced by the solver under test):

A  **extruded Sod tube vs the exact Riemann solution.**  The 1-D Sod problem is
   extruded to a 2-D Cartesian mesh (``y``-invariant, outflow in ``x``) and the
   cell-averaged fields are compared with the exact Riemann solution of
   ``benchmarks/sod_exact.py`` (Sod 1978; star state pinned in the test suite).
   3 % relative L2 gate, plus the observed order under refinement.
B  **``y``-invariance is exact.**  With ``y``-independent data the ``y``-flux of
   identical neighbouring states reduces to the physical flux, so the residual
   is column-independent up to round-off; any leak (e.g. a boundary closure
   mixing columns) shows up immediately.
C  **discrete conservation / flux telescoping.**  For a single right-hand side
   evaluation, ``sum(rhs) * dx * dy`` must equal the net boundary flux exactly,
   which pins the outflow face flux to the *same* flux function as the interior.
D  **shock position vs the exact Riemann shock speed.**  The shock speed of the
   exact solution is a literature quantity (Sod 1978); the numerical shock is
   located at the steepest density gradient and must fall inside a two-cell
   window of the exact position.
E  **rotational consistency.**  The same extruded Sod initial condition run in
   the transposed orientation (``x``-invariant, outflow in ``y``) must reproduce
   the ``x``-oriented density/velocity/pressure profiles, which exercises the
   ``y``-flux path and the ``y`` boundary closure.

Everything printed is written to ``validation/euler_2d_p0/evidence.json`` (and
``profiles.npz``) so that report numbers can be injected programmatically.

Run with::

    PYTHONPATH=. python -m benchmarks.verify_euler_2d_p0
    PYTHONPATH=. python -m benchmarks.verify_euler_2d_p0 --cells 800 1600 --quick
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import tensordg.euler_2d as e2d
from benchmarks.sod_exact import cell_averages, sample

OUTPUT = Path("validation/euler_2d_p0")
LEFT = (1.0, 0.0, 1.0)
RIGHT = (0.125, 0.0, 0.1)
DIAPHRAGM = 0.5
GATE = 0.03


def sod_initial(x, y):
    """1-D Sod initial condition extruded along ``y``."""
    rho = torch.where(x < DIAPHRAGM, torch.as_tensor(LEFT[0], dtype=x.dtype),
                      torch.as_tensor(RIGHT[0], dtype=x.dtype))
    pressure = torch.where(x < DIAPHRAGM, torch.as_tensor(LEFT[2], dtype=x.dtype),
                           torch.as_tensor(RIGHT[2], dtype=x.dtype))
    zero = torch.zeros_like(rho)
    return rho, zero, zero, pressure


def sod_initial_transposed(x, y):
    """Same data with the diaphragm normal to ``y``."""
    rho, u, v, pressure = sod_initial(y, x)
    return rho, v, u, pressure


def relative_l2(numerical: np.ndarray, reference: np.ndarray) -> float:
    absolute = float(np.sqrt(np.mean((numerical - reference) ** 2)))
    norm = float(np.sqrt(np.mean(reference ** 2)))
    return absolute / norm


def nyquist_order(coarse: float, fine: float, ratio: float) -> float:
    return math.log(coarse / fine) / math.log(ratio)


def run_case(nx: int, ny: int, final_time: float, cfl: float, periodic, transposed=False):
    initial = sod_initial_transposed if transposed else sod_initial
    start = time.perf_counter()
    x, y, fields, metadata = e2d.solve_euler_2d(
        initial,
        ncells=(nx, ny),
        final_time=final_time,
        cfl=cfl,
        domain=((0.0, 1.0), (0.0, 1.0)),
        periodic=periodic,
    )
    wall = time.perf_counter() - start
    return x, y, fields, metadata, wall


def check_a_extruded_sod(cells, final_time, cfl):
    """Extruded Sod vs the exact Riemann solution, plus the observed order."""
    reference_cells = max(cells)
    centers, reference = cell_averages(reference_cells, final_time)
    rows = []
    profiles = {}
    for nx in cells:
        x, y, fields, metadata, wall = run_case(nx, 4, final_time, cfl, (False, True))
        density = fields["density"][:, 0].cpu().numpy()
        velocity = fields["velocity_x"][:, 0].cpu().numpy()
        pressure = fields["pressure"][:, 0].cpu().numpy()
        _, exact = cell_averages(nx, final_time)
        errors = {
            "density": relative_l2(density, exact["density"]),
            "velocity_x": relative_l2(velocity, exact["velocity"]),
            "pressure": relative_l2(pressure, exact["pressure"]),
        }
        rows.append({
            "cells": nx,
            "rows": 4,
            "steps": metadata["steps"],
            "dt": metadata["dt"],
            "wall_seconds": wall,
            "relative_l2": errors,
            "relative_l2_percent": {k: 100.0 * v for k, v in errors.items()},
            "max_relative_l2_percent": 100.0 * max(errors.values()),
        })
        profiles[nx] = {
            "x": x.cpu().numpy(), "density": density,
            "velocity_x": velocity, "pressure": pressure,
            "exact_density": exact["density"], "exact_velocity": exact["velocity"],
            "exact_pressure": exact["pressure"],
        }
    for coarse, fine in zip(rows[:-1], rows[1:]):
        ratio = fine["cells"] / coarse["cells"]
        coarse["pair_with_next"] = None
        fine["order_vs_previous"] = {
            key: nyquist_order(coarse["relative_l2"][key], fine["relative_l2"][key], ratio)
            for key in coarse["relative_l2"]
        }
    ratios = np.array([row["cells"] for row in rows], dtype=float)
    orders = {
        key: float(-np.polyfit(np.log(ratios), np.log([row["relative_l2"][key] for row in rows]), 1)[0])
        for key in rows[0]["relative_l2"]
    }
    finest = rows[-1]
    return {
        "reference": {
            "source": "benchmarks/sod_exact.cell_averages (exact Riemann, Sod 1978)",
            "final_time": final_time, "quadrature_order": 8,
            "exact_density_norm": float(np.sqrt(np.mean(reference["density"] ** 2))),
            "exact_velocity_norm": float(np.sqrt(np.mean(reference["velocity"] ** 2))),
            "exact_pressure_norm": float(np.sqrt(np.mean(reference["pressure"] ** 2))),
            "centers": [float(v) for v in centers],
        },
        "rows": rows,
        "observed_order_least_squares": orders,
        "finest_max_relative_l2_percent": finest["max_relative_l2_percent"],
        "gate_percent": 100.0 * GATE,
        "passed": finest["max_relative_l2_percent"] < 100.0 * GATE,
        "profiles": profiles,
    }


def check_b_y_invariance(cells=200, final_time=0.2, cfl=0.4):
    _, _, fields, metadata, _ = run_case(cells, 4, final_time, cfl, (False, True))
    state = fields["conservative"]

    def spread(tensor):
        columns = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        deviation = (columns - columns[:, :1, :]).abs().max().item()
        scale = columns.abs().max().item()
        return deviation, deviation / max(scale, 1e-300)

    density_dev, density_rel = spread(fields["density"])
    momentum_y_dev, momentum_y_rel = spread(state[..., 2])
    state_dev, state_rel = spread(state)
    return {
        "cells": [cells, 4], "steps": metadata["steps"],
        "density_max_abs_deviation": density_dev,
        "density_max_rel_deviation": density_rel,
        "momentum_y_max_abs_deviation": momentum_y_dev,
        "momentum_y_max_rel_deviation": momentum_y_rel,
        "conservative_max_abs_deviation": state_dev,
        "conservative_max_rel_deviation": state_rel,
        "criterion": "column spread <= 1e-12 (round-off of a column-constant update)",
        "passed": bool(momentum_y_dev == 0.0 and state_rel < 1e-12),
    }


def check_c_flux_telescoping(cells=64, cfl=0.4):
    """sum(rhs)*dx*dy must equal the net boundary flux of the outflow faces."""
    nx, ny = cells, 5
    u = e2d._cell_averages(sod_initial, *e2d_coords(nx, ny), 2, torch.device("cpu"), torch.float64)
    dx, dy = 1.0 / nx, 1.0 / ny
    rhs = e2d._rhs(u, dx, dy, (False, False))
    net_rate = float(rhs[..., 0].sum().item()) * dx * dy
    faces = e2d._face_fluxes(u, "x", False)
    boundary_left = float(faces[0, :, 0].sum().item()) * dy
    boundary_right = float(faces[-1, :, 0].sum().item()) * dy
    expected = boundary_left - boundary_right
    faces_y = e2d._face_fluxes(u, "y", False)
    boundary_y = float(faces_y[:, 0, 0].sum().item()) * dx - float(faces_y[:, -1, 0].sum().item()) * dx
    return {
        "cells": [nx, ny], "dx": dx, "dy": dy,
        "discrete_mass_rate": net_rate,
        "boundary_flux_rate": expected + boundary_y,
        "absolute_difference": abs(net_rate - (expected + boundary_y)),
        "round_off_scale": max(abs(net_rate), abs(expected + boundary_y), 1e-300),
        "criterion": "|discrete rate - boundary rate| <= 1e-12 * scale",
        "passed": bool(abs(net_rate - (expected + boundary_y))
                       <= 1e-12 * max(abs(net_rate), 1.0)),
    }


def e2d_coords(nx, ny):
    dx, dy = 1.0 / nx, 1.0 / ny
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    return x, y, dx, dy


def _crossing(points, values, level, from_right=True):
    """First crossing of ``level`` scanning from the right (or the left)."""
    count = len(points)
    indices = range(count - 2, -1, -1) if from_right else range(count - 1)
    for left in indices:
        right = left + 1
        if (values[left] - level) * (values[right] - level) <= 0.0:
            weight = (level - values[left]) / (values[right] - values[left])
            return float(points[left] + weight * (points[right] - points[left]))
    return float("nan")


def check_d_discontinuity_positions(cells=3200, final_time=0.2, cfl=0.4):
    """Locate the numerical shock and contact and compare with the exact ones.

    Sod's density profile has two jumps: the contact discontinuity (left) and
    the shock (right).  Both are located as midpoint crossings, which is
    insensitive to the smearing width, and the exact positions come from the
    exact Riemann solution of ``benchmarks/sod_exact.py`` (Sod 1978); the shock
    speed is the literature quantity ``S = 1.7522`` in these units.
    """
    x, _, fields, metadata, _ = run_case(cells, 4, final_time, cfl, (False, True))
    density = fields["density"][:, 0].cpu().numpy()
    centers = x.cpu().numpy()
    grid = np.linspace(0.0, 1.0, 200001)
    exact_density = sample(grid, final_time)["density"]

    pre_shock, post_shock = RIGHT[0], 0.2655737  # exact Sod plateau behind the shock
    left_of_contact, right_of_contact = 0.4263194, 0.2655737
    shock_level = 0.5 * (pre_shock + post_shock)
    contact_level = 0.5 * (left_of_contact + right_of_contact)

    exact_shock = _crossing(grid, exact_density, shock_level, from_right=True)
    exact_contact = _crossing(grid, exact_density, contact_level, from_right=False)
    numerical_shock = _crossing(centers, density, shock_level, from_right=True)
    numerical_contact = _crossing(centers, density, contact_level, from_right=False)

    dx = 1.0 / cells
    rows = {}
    for name, numerical, exact in (
        ("shock", numerical_shock, exact_shock),
        ("contact", numerical_contact, exact_contact),
    ):
        offset = abs(numerical - exact)
        rows[name] = {
            "numerical_position": numerical,
            "exact_position": exact,
            "absolute_offset": offset,
            "cells_offset": offset / dx,
            "relative_error_percent": 100.0 * offset / max(abs(exact), 1e-300),
        }
    shock_speed_numerical = (numerical_shock - DIAPHRAGM) / final_time
    return {
        "cells": cells, "steps": metadata["steps"], "dx": dx,
        "positions": rows,
        "numerical_shock_speed": shock_speed_numerical,
        "exact_shock_speed": (exact_shock - DIAPHRAGM) / final_time,
        "criterion": ("shock within 1 dx (the jump the exact speed fixes) and "
                      "contact within 5 dx (the most diffused wave of a P0 scheme)"),
        "passed": bool(rows["shock"]["cells_offset"] <= 1.0
                       and rows["contact"]["cells_offset"] <= 5.0),
    }


def check_e_rotational_consistency(cells=400, final_time=0.2, cfl=0.4):
    x, _, fields_x, _, _ = run_case(cells, 4, final_time, cfl, (False, True))
    y, _, fields_y, _, _ = run_case(4, cells, final_time, cfl, (True, False), transposed=True)
    checks = {}
    for key, orient_key in (("velocity_x", "velocity_y"), ("density", "density"), ("pressure", "pressure")):
        profile_x = fields_x[key].mean(dim=1).cpu().numpy()
        source = fields_y[key] if key != "velocity_x" else fields_y["velocity_y"]
        profile_y = source.mean(dim=0).cpu().numpy()
        difference = float(np.max(np.abs(profile_x - profile_y)))
        scale = float(np.max(np.abs(profile_x))) + 1e-300
        checks[key] = {
            "max_absolute_difference": difference,
            "max_relative_difference": difference / scale,
        }
    worst = max(value["max_relative_difference"] for value in checks.values())
    return {
        "cells": cells, "checks": checks,
        "worst_relative_difference": worst,
        "criterion": "transposed run reproduces the profile to <= 1e-10 relative",
        "passed": bool(worst <= 1e-10),
    }


def legacy_periodic_regression(cells=32, final_time=0.2, cfl=0.4):
    """Guard the refactor: the periodic path is bit-for-bit the rolled stencil."""
    _, _, fields, metadata, _ = run_case(cells, cells, final_time, cfl, (True, True))
    x, y, dx, dy = e2d_coords(cells, cells)

    def rolled_rhs(u):
        right_x = torch.roll(u, shifts=-1, dims=0)
        right_y = torch.roll(u, shifts=-1, dims=1)
        flux_right_x = e2d._interface_flux(u, right_x, "x")
        flux_right_y = e2d._interface_flux(u, right_y, "y")
        flux_left_x = torch.roll(flux_right_x, shifts=1, dims=0)
        flux_left_y = torch.roll(flux_right_y, shifts=1, dims=1)
        return -(flux_right_x - flux_left_x) / dx - (flux_right_y - flux_left_y) / dy

    u = e2d._cell_averages(sod_initial, x, y, dx, dy, 2, torch.device("cpu"), torch.float64)
    padded = e2d._rhs(u, dx, dy, (True, True))
    rolled = rolled_rhs(u)
    difference = float((padded - rolled).abs().max())
    return {
        "cells": [cells, cells], "steps": metadata["steps"],
        "max_absolute_rhs_difference": difference,
        "criterion": "identical to machine precision (<= 1e-14 absolute)",
        "passed": bool(difference <= 1e-14),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[400, 800, 1600, 3200],
                        help="x-cell counts for the extruded Sod refinement study")
    parser.add_argument("--time", type=float, default=0.2)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--quick", action="store_true", help="skip the 3200-cell gate run")
    args = parser.parse_args(argv)
    cells = sorted(set(args.cells))
    if args.quick:
        cells = [value for value in cells if value <= 800]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "final_time": args.time,
        "cfl": args.cfl,
        "gate_percent": 100.0 * GATE,
        "checks": {},
    }
    report["checks"]["A_extruded_sod"] = check_a_extruded_sod(cells, args.time, args.cfl)
    print("A extruded Sod:", {
        row["cells"]: row["max_relative_l2_percent"] for row in
        report["checks"]["A_extruded_sod"]["rows"]
    })
    report["checks"]["B_y_invariance"] = check_b_y_invariance()
    print("B y-invariance:", report["checks"]["B_y_invariance"]["conservative_max_rel_deviation"])
    report["checks"]["C_flux_telescoping"] = check_c_flux_telescoping()
    print("C flux telescoping:", report["checks"]["C_flux_telescoping"]["absolute_difference"])
    report["checks"]["D_discontinuity_positions"] = check_d_discontinuity_positions(
        cells[-1], args.time, args.cfl
    )
    print("D discontinuity positions:", {
        name: row["cells_offset"] for name, row in
        report["checks"]["D_discontinuity_positions"]["positions"].items()
    })
    report["checks"]["E_rotational_consistency"] = check_e_rotational_consistency()
    print("E rotational consistency:", report["checks"]["E_rotational_consistency"]["worst_relative_difference"])
    report["checks"]["F_periodic_roll_equivalence"] = legacy_periodic_regression()
    print("F periodic roll equivalence:",
          report["checks"]["F_periodic_roll_equivalence"]["max_absolute_rhs_difference"])
    report["all_checks_passed"] = all(
        entry["passed"] for entry in report["checks"].values()
    )

    profiles = report["checks"]["A_extruded_sod"].pop("profiles")
    np.savez_compressed(
        OUTPUT / "profiles.npz",
        **{f"{key}_{name}": value for key, entry in profiles.items()
           for name, value in entry.items()},
    )
    reference = report["checks"]["A_extruded_sod"].pop("reference")
    (OUTPUT / "evidence.json").write_text(json.dumps(report, indent=2))
    (OUTPUT / "reference.json").write_text(json.dumps(reference, indent=2))
    print("all checks passed:", report["all_checks_passed"])
    print("evidence written to", OUTPUT / "evidence.json")


if __name__ == "__main__":
    main()