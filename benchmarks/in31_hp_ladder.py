"""IN31 ladder: linear advection, diffusion, Sod shock tube and hp-refinement.

Zero-to-one verification ladder for campaign node ``IN31-G01``, kept separate
from ``benchmarks/verification_ladder.py`` (mission m5280, artifact
``docs/benchmarks/results/verification_ladder/``).  That ladder stops at
``p = 3``, gates the Sod shock position at ``0.02 dx`` - below the resolution of
its own sub-cell estimator, which is why its C3 is an inherited FAIL - and never
measures the shock *strength*.  This node closes those gaps and gates strictly on
the mission criteria:

C1  1D advection ``u_t + u_x = 0`` (periodic, smooth sine): observed L2 order in
    h equals ``p + 1`` within 10 % for ``p = 1..5``, with the RK4 temporal error
    refined below 2 % of the spatial error.
C2  1D diffusion ``u_t = nu u_xx`` (SIPG, zero Dirichlet): the semidiscrete
    system is integrated *exactly* in time (matrix exponential), so the measured
    order is purely spatial: ``p + 1`` within 10 % for ``p = 1..5``.
C3  Sod shock tube at 160 cells, ``p = 1,2,3`` unlimited and TVB limited:
    captured shock position within 2 cells of the exact Riemann solution and
    captured shock strength (density and pressure jump) within 5 %.  The
    sub-cell locator is calibrated on synthetic fronts so the gate cannot be
    finer than the measurement.
C4  hp consistency: (a) fixed p, h-refinement order ``p+1``; (b) fixed h,
    p-refinement: the log-slope of the error in p matches the sharp
    L2-projection slope within 15 % for ``p = 2..5`` and the error drops by at
    least three orders of magnitude from ``p = 1`` to ``p = 5``.
C5  the ladder is pytest-gated and serialised to ``gates_verified.json``; every
    check carries its measured value, target, threshold and comparator, so the
    verdict is re-derivable from the record.

Run::

    python -m benchmarks.in31_hp_ladder            # full ladder + figures
    python -m benchmarks.in31_hp_ladder --quick    # smoke configuration
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from benchmarks.riemann_exact import GAMMA
from benchmarks.sod_exact import cell_averages
from benchmarks.verification_ladder import (
    SOD_DIAPHRAGM,
    SOD_FINAL_TIME,
    SOD_LEFT,
    SOD_RIGHT,
    gibbs_amplitude,
    locate_shock,
    locate_shock_centroid,
    observed_order,
    pairwise_orders,
    sod_reference,
)
from tensordg.advection_1d import l2_error, solve_advection
from tensordg.euler_dg_1d import solve_euler_dg_1d
from tensordg.heat_1d import heat_error, solve_heat_1d, solve_heat_1d_exact_time

SCHEMA = "tensordg-in31-hp-ladder/1"
ARTIFACT_RELPATH = Path("docs") / "benchmarks" / "results" / "in31-hp-ladder"
TEST_RELPATH = Path("tests") / "test_in31_hp_ladder.py"

# --- Configuration ---------------------------------------------------------
ADV_DEGREES = (1, 2, 3, 4, 5)
ADV_RESOLUTIONS = (8, 16, 32)
ADV_FINAL_TIME = 0.5
ADV_CFL = 0.3

DIFF_DEGREES = (1, 2, 3, 4, 5)
DIFF_RESOLUTIONS = (4, 8, 16)
DIFF_FINAL_TIME = 0.1

SOD_NCELLS = 160
SOD_DEGREES = (1, 2, 3)
SOD_LIMITERS = (None, "tvb")
SOD_CFL = 0.2

# --- Thresholds (mission criteria, not fitted) -----------------------------
ORDER_REL_TOLERANCE = 0.10            # C1/C2: order within 10 % of p+1
HP_RATE_REL_TOLERANCE = 0.15          # C4: hp rate within 15 % of theory
SHOCK_POSITION_TOLERANCE_CELLS = 2.0  # C3: < 2 cells
SHOCK_STRENGTH_TOLERANCE = 0.05       # C3: < 5 % of the exact jump
TEMPORAL_AUDIT_TOLERANCE = 0.02       # C1: temporal error < 2 % of spatial
P_DECAY_WINDOW = (2, 5)               # C4: asymptotic p window (see report)
P_DECAY_MIN_RATIO = 1.0e3             # C4: e(p=1)/e(p=5) >= 1e3


# ---------------------------------------------------------------------------
# Analytic references (no solver code below this line until the studies)
# ---------------------------------------------------------------------------

def projection_error(ncells: int, degree: int,
                     wavenumber: float = 2.0 * math.pi) -> float:
    """Sharp L2-projection error of ``sin(k x)`` onto piecewise ``P_p``.

    Independent of any DG solver: the modal coefficients are the exact Fourier
    moments of the sine in the Legendre basis.  With the default wavenumber it
    reproduces :func:`benchmarks.verification_ladder.sine_projection_error`
    (asserted in the tests), which makes this routine a cross-validation of the
    two references rather than a second source of truth.  ``wavenumber=pi`` is
    the reference for the heat problem, whose exact solution is
    ``sin(pi x) exp(-pi^2 t)``.
    """
    if ncells < 1 or degree < 0:
        raise ValueError("ncells >= 1 and degree >= 0 are required")
    nq = max(4 * (degree + 1), 40)
    nodes, weights = np.polynomial.legendre.leggauss(nq)
    h = 1.0 / ncells
    x = (np.arange(ncells)[:, None] + 0.5 + 0.5 * nodes[None, :]) * h
    exact = np.sin(wavenumber * x)
    basis = np.stack(
        [np.polynomial.legendre.legval(nodes, np.eye(degree + 1)[k])
         for k in range(degree + 1)],
        axis=1,
    )
    scaling = (2.0 * np.arange(degree + 1) + 1.0) / 2.0
    coefficients = np.einsum("q,cq,qk->ck", weights, exact, basis) * scaling
    approximate = np.einsum("ck,qk->cq", coefficients, basis)
    error_squared = 0.5 * h * np.sum(weights[None, :] * (approximate - exact) ** 2)
    return float(math.sqrt(error_squared))


def log_slope(degrees: Sequence[int], errors: Sequence[float]) -> float:
    """Decay rate of the error in the polynomial degree, ``-d log e / dp``."""
    p = np.asarray(degrees, dtype=float)
    e = np.asarray(errors, dtype=float)
    if np.any(e <= 0.0):
        raise ValueError("errors must be positive")
    return float(-np.polyfit(p, np.log(e), 1)[0])


# ---------------------------------------------------------------------------
# C1: linear advection
# ---------------------------------------------------------------------------

def temporal_converged_advection(
    ncells: int,
    degree: int,
    final_time: float = ADV_FINAL_TIME,
    cfl: float = ADV_CFL,
    tolerance: float = TEMPORAL_AUDIT_TOLERANCE,
    max_refinements: int = 4,
) -> dict:
    """L2 error of the advection run with the RK4 step size refined until stable.

    The ladder measures the *spatial* error, so the temporal error has to be
    shown negligible rather than assumed: the CFL number is halved until the
    error changes by less than ``tolerance`` (relative) between two successive
    refinements.
    """
    history: list[dict] = []
    current_cfl = cfl
    error = None
    converged = False
    for _ in range(max_refinements + 1):
        centers, coefficients, metadata = solve_advection(
            ncells=ncells, degree=degree, final_time=final_time, cfl=current_cfl,
        )
        new_error = l2_error(centers, coefficients, final_time)
        change = (abs(new_error - error) / new_error) if error else float("inf")
        history.append({"cfl": current_cfl, "dt": metadata["dt"],
                        "steps": metadata["steps"], "relative_l2": new_error,
                        "change_vs_previous": change})
        error = new_error
        if change <= tolerance:
            converged = True
            break
        current_cfl *= 0.5
    return {
        "relative_l2": error,
        "cfl": history[-1]["cfl"],
        "steps": history[-1]["steps"],
        "dt": history[-1]["dt"],
        "temporal_change": history[-1]["change_vs_previous"],
        "temporal_converged": converged,
        "refinements": len(history) - 1,
        "history": history,
    }


def advection_study(
    degrees: Sequence[int] = ADV_DEGREES,
    resolutions: Sequence[int] = ADV_RESOLUTIONS,
    final_time: float = ADV_FINAL_TIME,
    cfl: float = ADV_CFL,
) -> dict:
    block: dict = {"study": "advection_1d", "equation": "u_t + u_x = 0",
                   "initial_data": "sin(2*pi*x)", "final_time": final_time,
                   "degrees": {}}
    for degree in degrees:
        errors, levels = [], []
        for ncells in resolutions:
            run = temporal_converged_advection(ncells, degree, final_time, cfl)
            errors.append(run["relative_l2"])
            levels.append({
                "ncells": ncells,
                "h": 1.0 / ncells,
                "relative_l2": run["relative_l2"],
                "projection_floor": projection_error(ncells, degree),
                "cfl": run["cfl"],
                "dt": run["dt"],
                "steps": run["steps"],
                "temporal_change": run["temporal_change"],
                "temporal_converged": run["temporal_converged"],
                "refinements": run["refinements"],
                "history": run["history"],
            })
        block["degrees"][str(degree)] = {
            "levels": levels,
            "pairwise_orders": pairwise_orders(resolutions, errors),
            "observed_order": observed_order(resolutions, errors),
            "theoretical_order": degree + 1,
        }
    return block


# ---------------------------------------------------------------------------
# C2: diffusion
# ---------------------------------------------------------------------------

def diffusion_study(
    degrees: Sequence[int] = DIFF_DEGREES,
    resolutions: Sequence[int] = DIFF_RESOLUTIONS,
    final_time: float = DIFF_FINAL_TIME,
    cn_degrees: Sequence[int] | None = None,
) -> dict:
    block: dict = {"study": "diffusion_1d", "equation": "u_t = u_xx (nu=1)",
                   "initial_data": "sin(pi*x), zero Dirichlet data",
                   "space": "SIPG, face penalty 20*(p+1)^2/h",
                   "time": "matrix exponential exp(-t M^-1 K), exact",
                   "final_time": final_time, "degrees": {}}
    for degree in degrees:
        errors, levels = [], []
        for ncells in resolutions:
            centers, coefficients, metadata = solve_heat_1d_exact_time(
                ncells=ncells, degree=degree, final_time=final_time,
            )
            metrics = heat_error(centers, coefficients, final_time)
            errors.append(metrics["relative_l2"])
            levels.append({
                "ncells": ncells,
                "h": 1.0 / ncells,
                "relative_l2": metrics["relative_l2"],
                "absolute_l2": metrics["absolute_l2"],
                "projection_floor": projection_error(ncells, degree, math.pi),
                "dofs": metadata["dofs"],
                "stiffness_eigenvalue_min": metadata["stiffness_eigenvalue_min"],
                "stiffness_eigenvalue_max": metadata["stiffness_eigenvalue_max"],
            })
        block["degrees"][str(degree)] = {
            "levels": levels,
            "pairwise_orders": pairwise_orders(resolutions, errors),
            "observed_order": observed_order(resolutions, errors),
            "theoretical_order": degree + 1,
        }
    # Crank--Nicolson cross-check (same operator, second-order time stepping).
    # It shows why exact time integration is needed for p >= 4: the default step
    # (dt = 0.02 h) leaves a temporal floor above the spatial error.
    block["crank_nicolson_masking"] = {}
    for degree in (degrees if cn_degrees is None else cn_degrees):
        centers, coefficients, default_metadata = solve_heat_1d(
            ncells=8, degree=degree, final_time=final_time,
        )
        default = heat_error(centers, coefficients, final_time)["relative_l2"]
        centers, coefficients, _ = solve_heat_1d(
            ncells=8, degree=degree, final_time=final_time, time_steps=20000,
        )
        refined = heat_error(centers, coefficients, final_time)["relative_l2"]
        centers, coefficients, _ = solve_heat_1d_exact_time(
            ncells=8, degree=degree, final_time=final_time,
        )
        exact = heat_error(centers, coefficients, final_time)["relative_l2"]
        block["crank_nicolson_masking"][str(degree)] = {
            "default_steps": default_metadata["steps"],
            "cn_default_relative_l2": default,
            "cn_20000_steps_relative_l2": refined,
            "exact_time_relative_l2": exact,
            "default_over_exact": default / exact,
        }
    return block


# ---------------------------------------------------------------------------
# C3: Sod shock tube
# ---------------------------------------------------------------------------

def _plateau_mean(values: np.ndarray, centers: np.ndarray, low: float,
                  high: float) -> tuple[float, int]:
    mask = (centers > low) & (centers < high)
    count = int(mask.sum())
    if count == 0:
        return float("nan"), 0
    return float(np.mean(values[mask])), count


def contact_diagnostics(density: np.ndarray, exact_density: np.ndarray,
                        centers: np.ndarray, reference: dict) -> dict:
    """Contact-discontinuity position error and smearing width.

    The exact star densities bracket the contact (they are read from the exact
    solution on the cells adjacent to ``x_contact``); the contact position is the
    mid-level crossing inside ``x_contact +- 6 dx`` and the smearing width counts
    the interfaces in that window across which the density still changes by more
    than 5 % of the exact contact jump.
    """
    dx = float(centers[1] - centers[0])
    contact = SOD_DIAPHRAGM + reference["u_star"] * reference["final_time"]
    index = int(np.argmin(np.abs(centers - contact)))
    if index < 1 or index + 1 >= centers.size:  # pragma: no cover - degenerate grid
        return {"contact_position_error_cells": float("nan"),
                "contact_width_cells": 0, "window": []}
    # Star states bracketing the contact, taken from the exact solution: the cell
    # to the left is the plateau behind the contact, the cell to the right is the
    # star-right state in front of it.
    left = float(exact_density[index - 1])
    right = float(exact_density[index + 1])
    jump = left - right
    level = 0.5 * (left + right)
    window = (contact - 6.0 * dx, contact + 6.0 * dx)
    inside = (centers >= window[0]) & (centers <= window[1])
    if jump <= 0.0 or not np.any(inside):  # pragma: no cover - degenerate window
        return {"contact_position_error_cells": float("nan"),
                "contact_width_cells": 0, "window": list(window)}
    jumps = np.where(inside[:-1], np.abs(np.diff(density)), -1.0)
    steepest = int(np.argmax(jumps))
    a, b = density[steepest], density[steepest + 1]
    fraction = 0.5 if a == b else float(np.clip((level - a) / (b - a), 0.0, 1.0))
    position = centers[steepest] + fraction * dx
    # Smearing width: interfaces inside the contact window across which the
    # density still changes by more than 5 % of the exact contact jump.  The
    # window keeps the rarefaction and the shock out of the count.
    width = int(np.count_nonzero(jumps > 0.05 * jump))
    return {
        "contact_position": float(position),
        "contact_position_exact": float(contact),
        "contact_position_error_cells": abs(position - contact) / dx,
        "contact_width_cells": width,
        "contact_jump": float(jump),
        "window": [float(window[0]), float(window[1])],
    }


def sod_run(ncells: int, degree: int, limiter: str | None,
            final_time: float = SOD_FINAL_TIME, cfl: float = SOD_CFL) -> dict:
    """One Sod run with position, strength and resolution diagnostics."""
    started = time.time()
    centers, coefficients, metadata = solve_euler_dg_1d(
        SOD_LEFT, SOD_RIGHT, ncells=ncells, degree=degree, final_time=final_time,
        diaphragm=SOD_DIAPHRAGM, cfl=cfl, limiter=limiter, limiter_stage="final",
    )
    density = coefficients[:, 0, 0].detach().cpu().numpy()
    momentum = coefficients[:, 0, 1].detach().cpu().numpy()
    energy = coefficients[:, 0, 2].detach().cpu().numpy()
    pressure = (energy - 0.5 * momentum ** 2 / density) * (GAMMA - 1.0)
    centers_np = centers.detach().cpu().numpy()
    dx = float(centers_np[1] - centers_np[0])

    reference = sod_reference(final_time)
    shock = locate_shock(density, reference["rho_star_right"], SOD_RIGHT[0])
    centroid = float(locate_shock_centroid(
        coefficients[:, :, 0].detach().cpu().numpy(),
    ))
    _, exact_fields = cell_averages(ncells, final_time)
    exact_density = np.asarray(exact_fields["density"], dtype=float)

    contact_position = SOD_DIAPHRAGM + reference["u_star"] * final_time
    shock_position = reference["shock_position"]
    margin = 3.0 * dx
    rho_up, n_up = _plateau_mean(density, centers_np, contact_position + margin,
                                 shock_position - margin)
    rho_down, n_down = _plateau_mean(density, centers_np, shock_position + margin,
                                     1.0)
    p_up, _ = _plateau_mean(pressure, centers_np, contact_position + margin,
                            shock_position - margin)
    p_down, _ = _plateau_mean(pressure, centers_np, shock_position + margin, 1.0)
    exact_drho = reference["rho_star_right"] - SOD_RIGHT[0]
    exact_dp = reference["p_star"] - SOD_RIGHT[2]
    measured_drho = rho_up - rho_down
    measured_dp = p_up - p_down

    relative_l2 = float(
        math.sqrt(np.mean((density - exact_density) ** 2))
        / math.sqrt(np.mean(exact_density ** 2))
    )
    return {
        "ncells": ncells,
        "degree": degree,
        "limiter": limiter if limiter else "unlimited",
        "final_time": final_time,
        "cfl": cfl,
        "dx": dx,
        "steps": metadata["steps"],
        "shock_position": shock["position"],
        "shock_position_interface": shock["interface"],
        "shock_position_centroid": centroid,
        "shock_position_exact": reference["shock_position"],
        "shock_position_error_cells": abs(shock["position"] - shock_position) / dx,
        "shock_position_interface_error_cells":
            abs(shock["interface"] - shock_position) / dx,
        "shock_position_centroid_error_cells": abs(centroid - shock_position) / dx,
        "shock_level_fraction": shock["fraction"],
        "density_jump_measured": measured_drho,
        "density_jump_exact": exact_drho,
        "density_jump_error": abs(measured_drho - exact_drho) / exact_drho,
        "pressure_jump_measured": measured_dp,
        "pressure_jump_exact": exact_dp,
        "pressure_jump_error": abs(measured_dp - exact_dp) / exact_dp,
        "plateau_cells_upstream": n_up,
        "plateau_cells_downstream": n_down,
        "gibbs_amplitude": gibbs_amplitude(density)["amplitude"],
        "density_relative_l2": relative_l2,
        "contact": contact_diagnostics(density, exact_density, centers_np, reference),
        "density_profile": density.tolist(),
        "pressure_profile": pressure.tolist(),
        "centers_profile": centers_np.tolist(),
        "runtime_seconds": time.time() - started,
    }


def sod_study(ncells: int = SOD_NCELLS, degrees: Sequence[int] = SOD_DEGREES,
              limiters: Sequence[str | None] = SOD_LIMITERS,
              final_time: float = SOD_FINAL_TIME, cfl: float = SOD_CFL) -> dict:
    cases = {}
    for degree in degrees:
        for limiter in limiters:
            label = "p%d_%s" % (degree, limiter if limiter else "unlimited")
            cases[label] = sod_run(ncells, degree, limiter, final_time, cfl)
    return {
        "study": "sod_1d",
        "problem": "Sod shock tube: rho_L=1 p_L=1, rho_R=0.125 p_R=0.1, gamma=1.4",
        "ncells": ncells,
        "cfl": cfl,
        "final_time": final_time,
        "limiter_stage": "final",
        "reference": sod_reference(final_time),
        "cases": cases,
    }


def estimator_floor(ncells: int = SOD_NCELLS, widths=(0.8, 1.5, 2.5),
                    offsets: int = 21, amplitude: float = 1.0,
                    right_level: float = 0.125) -> dict:
    """Bias of the sub-cell shock locator on synthetic fronts of known position.

    A front of known sub-cell position is generated as ``tanh``-shaped cell
    averages spanning ``width`` cells and :func:`locate_shock` is asked for its
    position for a sweep of sub-cell offsets.  The returned RMS/max errors are
    the smallest tolerance the estimator can honour; a gate tighter than that
    measures the estimator, not the solver.
    """
    results = {}
    dx = 1.0 / ncells
    centers = (np.arange(ncells) + 0.5) * dx
    for width in widths:
        errors = []
        for offset in np.linspace(0.05, 0.95, offsets):
            position = 0.55 + offset * dx
            values = right_level + (amplitude - right_level) * 0.5 * (
                1.0 - np.tanh((centers - position) / (0.5 * width * dx))
            )
            located = locate_shock(
                values, amplitude, right_level,
                window=(position - 6.0 * dx, position + 6.0 * dx),
            )
            errors.append(abs(located["position"] - position) / dx)
        results[str(width)] = {
            "width_cells": float(width),
            "rms_error_cells": float(np.sqrt(np.mean(np.square(errors)))),
            "max_error_cells": float(max(errors)),
            "samples": int(offsets),
        }
    return {"ncells": ncells, "fronts": results,
            "estimator_floor_cells": max(
                item["max_error_cells"] for item in results.values()
            )}


# ---------------------------------------------------------------------------
# C4: hp refinement
# ---------------------------------------------------------------------------

def hp_study(advection: dict, diffusion: dict) -> dict:
    """p-refinement at fixed h for both equations, theory from sharp projection."""
    block: dict = {"study": "hp_refinement", "p_window": list(P_DECAY_WINDOW),
                   "advection_1d": {}, "diffusion_1d": {}}
    for name, data, wavenumber in (
        ("advection_1d", advection, 2.0 * math.pi),
        ("diffusion_1d", diffusion, math.pi),
    ):
        first = data["degrees"][min(data["degrees"], key=int)]
        resolutions = [level["ncells"] for level in first["levels"]]
        for level, ncells in enumerate(resolutions):
            degrees, errors, floors = [], [], []
            for key, entry in sorted(data["degrees"].items(),
                                     key=lambda kv: int(kv[0])):
                degrees.append(int(key))
                errors.append(entry["levels"][level]["relative_l2"])
                floors.append(projection_error(ncells, int(key), wavenumber))
            window = [d for d in degrees if P_DECAY_WINDOW[0] <= d <= P_DECAY_WINDOW[1]]
            measured_window = [errors[degrees.index(d)] for d in window]
            floor_window = [floors[degrees.index(d)] for d in window]
            block[name][str(ncells)] = {
                "degrees": degrees,
                "errors": errors,
                "projection_floors": floors,
                "error_over_floor": [e / f for e, f in zip(errors, floors)],
                "measured_slope_all_p": log_slope(degrees, errors),
                "theory_slope_all_p": log_slope(degrees, floors),
                "measured_slope_window": log_slope(window, measured_window),
                "theory_slope_window": log_slope(window, floor_window),
                "error_ratio_p1_to_p5": errors[0] / errors[-1],
            }
    return block


# ---------------------------------------------------------------------------
# Gate machinery
# ---------------------------------------------------------------------------

def check(name: str, metric: str, measured: float, comparator: str,
          threshold: float, target: float | None = None, **extra) -> dict:
    """One machine-checkable gate record (its status is re-derivable from it)."""
    measured = float(measured)
    if comparator == "rel_dev_le":
        if target is None:
            raise ValueError("comparator 'rel_dev_le' needs a target")
        passed = abs(measured - target) / abs(target) <= threshold
    elif comparator == "le":
        passed = measured <= threshold
    elif comparator == "ge":
        passed = measured >= threshold
    else:  # pragma: no cover - programming error
        raise ValueError("unknown comparator %r" % (comparator,))
    record = {
        "name": name,
        "metric": metric,
        "measured": measured,
        "comparator": comparator,
        "threshold": float(threshold),
        "status": "PASS" if passed else "FAIL",
    }
    if target is not None:
        record["target"] = float(target)
        record["relative_deviation"] = abs(measured - target) / abs(target)
    record.update(extra)
    return record


def gate(gate_id: str, name: str, criterion: str, checks: list[dict],
         measured: dict | None = None) -> dict:
    failed = [item["name"] for item in checks if item["status"] != "PASS"]
    return {
        "id": gate_id,
        "name": name,
        "criterion": criterion,
        "status": "FAIL" if failed else "PASS",
        "checks": checks,
        "failed_checks": failed,
        "measured": measured or {},
    }


def evaluate_c1(advection: dict) -> dict:
    checks = []
    for key, entry in sorted(advection["degrees"].items(),
                             key=lambda kv: int(kv[0])):
        degree = int(key)
        checks.append(check(
            "advection_order_p%d" % degree,
            "observed L2 order in h",
            entry["observed_order"], "rel_dev_le", ORDER_REL_TOLERANCE,
            target=degree + 1,
            pairwise_orders=entry["pairwise_orders"],
            resolutions=list(ADV_RESOLUTIONS),
        ))
        worst = max(level["temporal_change"] for level in entry["levels"])
        checks.append(check(
            "advection_temporal_subdominance_p%d" % degree,
            "max relative error change when the RK4 step is halved",
            worst, "le", TEMPORAL_AUDIT_TOLERANCE,
            max_refinements=max(level["refinements"] for level in entry["levels"]),
        ))
    measured = {}
    for key, entry in sorted(advection["degrees"].items(), key=lambda kv: int(kv[0])):
        measured[key] = {
            "observed_order": entry["observed_order"],
            "theoretical_order": entry["theoretical_order"],
            "pairwise_orders": entry["pairwise_orders"],
            "errors": [level["relative_l2"] for level in entry["levels"]],
            "projection_floors": [level["projection_floor"] for level in entry["levels"]],
            "cfls": [level["cfl"] for level in entry["levels"]],
            "steps": [level["steps"] for level in entry["levels"]],
            "temporal_changes": [level["temporal_change"] for level in entry["levels"]],
        }
    return gate(
        "C1",
        "1D advection: observed order p+1 within 10 %, temporal error below 2 %",
        "L2 error of u_t+u_x=0 (sin(2*pi*x), periodic, T=0.5) on 8/16/32 cells; "
        "least-squares order within 10 % of p+1 for p=1..5; halving the RK4 step "
        "changes the error by less than 2 % at every level",
        checks, measured,
    )


def evaluate_c2(diffusion: dict) -> dict:
    checks = []
    for key, entry in sorted(diffusion["degrees"].items(),
                             key=lambda kv: int(kv[0])):
        degree = int(key)
        checks.append(check(
            "diffusion_order_p%d" % degree,
            "observed L2 order in h (time-exact)",
            entry["observed_order"], "rel_dev_le", ORDER_REL_TOLERANCE,
            target=degree + 1,
            pairwise_orders=entry["pairwise_orders"],
            resolutions=list(DIFF_RESOLUTIONS),
        ))
        finest = entry["levels"][-1]
        checks.append(check(
            "diffusion_fundamental_eigenvalue_p%d" % degree,
            "smallest SIPG stiffness eigenvalue / pi^2 on the finest grid minus 1",
            abs(finest["stiffness_eigenvalue_min"] / math.pi ** 2 - 1.0),
            "le", 0.02,
            grid=finest["ncells"],
            eigenvalue=finest["stiffness_eigenvalue_min"],
        ))
    measured = {}
    for key, entry in sorted(diffusion["degrees"].items(), key=lambda kv: int(kv[0])):
        measured[key] = {
            "observed_order": entry["observed_order"],
            "theoretical_order": entry["theoretical_order"],
            "pairwise_orders": entry["pairwise_orders"],
            "errors": [level["relative_l2"] for level in entry["levels"]],
            "projection_floors": [level["projection_floor"] for level in entry["levels"]],
            "eigenvalue_min": entry["levels"][-1]["stiffness_eigenvalue_min"],
            "eigenvalue_max": entry["levels"][-1]["stiffness_eigenvalue_max"],
        }
    return gate(
        "C2",
        "1D diffusion: observed order p+1 within 10 % with exact time integration",
        "L2 error of u_t=u_xx (sin(pi x), Dirichlet, T=0.1) on 4/8/16 cells with "
        "the semidiscrete system integrated by expm; least-squares order within "
        "10 % of p+1 for p=1..5; smallest stiffness eigenvalue within 2 % of pi^2 "
        "on the finest grid",
        checks, measured,
    )


def evaluate_c3(sod: dict, floor: dict | None = None) -> dict:
    checks = []
    for label, case in sorted(sod["cases"].items()):
        checks.append(check(
            "sod_shock_position_%s" % label,
            "|x_num - x_exact| / dx (sub-cell mid-level crossing)",
            case["shock_position_error_cells"], "le",
            SHOCK_POSITION_TOLERANCE_CELLS,
            interface_error_cells=case["shock_position_interface_error_cells"],
            centroid_error_cells=case["shock_position_centroid_error_cells"],
            level_fraction=case["shock_level_fraction"],
            limiter=case["limiter"],
        ))
        checks.append(check(
            "sod_density_jump_%s" % label,
            "relative error of the captured density jump",
            case["density_jump_error"], "le", SHOCK_STRENGTH_TOLERANCE,
            measured_jump=case["density_jump_measured"],
            exact_jump=case["density_jump_exact"],
        ))
        checks.append(check(
            "sod_pressure_jump_%s" % label,
            "relative error of the captured pressure jump",
            case["pressure_jump_error"], "le", SHOCK_STRENGTH_TOLERANCE,
            measured_jump=case["pressure_jump_measured"],
            exact_jump=case["pressure_jump_exact"],
        ))
    if floor is not None:
        checks.append(check(
            "sod_gate_exceeds_estimator_floor",
            "sub-cell locator floor (cells) / gate tolerance (cells)",
            floor["estimator_floor_cells"] / SHOCK_POSITION_TOLERANCE_CELLS,
            "le", 1.0,
            estimator_floor_cells=floor["estimator_floor_cells"],
            tolerance_cells=SHOCK_POSITION_TOLERANCE_CELLS,
        ))
    measured = {}
    for label, case in sorted(sod["cases"].items()):
        measured[label] = {
            "shock_position_error_cells": case["shock_position_error_cells"],
            "shock_position_interface_error_cells":
                case["shock_position_interface_error_cells"],
            "shock_position_centroid_error_cells":
                case["shock_position_centroid_error_cells"],
            "density_jump_error": case["density_jump_error"],
            "pressure_jump_error": case["pressure_jump_error"],
            "density_relative_l2": case["density_relative_l2"],
            "gibbs_amplitude": case["gibbs_amplitude"],
            "contact_position_error_cells":
                case["contact"]["contact_position_error_cells"],
            "contact_width_cells": case["contact"]["contact_width_cells"],
            "steps": case["steps"],
        }
    return gate(
        "C3",
        "Sod: shock position within 2 cells, shock strength within 5 %",
        "DG/HLLC/SSP-RK3 at 160 cells, p=1..3, unlimited and TVB limited: "
        "captured shock position (sub-cell mid-level crossing) within 2 cells of "
        "the exact Riemann solution; captured density and pressure jumps "
        "(plateau means on both sides, 3 cells clear of the fronts) within 5 %; "
        "gate tolerance verified to exceed the synthetic-front estimator floor",
        checks, measured,
    )


def evaluate_c4(advection: dict, diffusion: dict, hp: dict) -> dict:
    checks = []
    for name, data, resolutions in (
        ("advection_1d", advection, ADV_RESOLUTIONS),
        ("diffusion_1d", diffusion, DIFF_RESOLUTIONS),
    ):
        for key, entry in sorted(data["degrees"].items(),
                                 key=lambda kv: int(kv[0])):
            checks.append(check(
                "h_refinement_%s_p%s" % (name, key),
                "observed L2 order in h",
                entry["observed_order"], "rel_dev_le", ORDER_REL_TOLERANCE,
                target=int(key) + 1,
                resolutions=list(resolutions),
            ))
    for name in ("advection_1d", "diffusion_1d"):
        for ncells, entry in sorted(hp[name].items(), key=lambda kv: int(kv[0])):
            checks.append(check(
                "p_refinement_rate_%s_n%s" % (name, ncells),
                "log-slope of the error in p over p=2..5",
                entry["measured_slope_window"], "rel_dev_le",
                HP_RATE_REL_TOLERANCE, target=entry["theory_slope_window"],
                measured_slope_all_p=entry["measured_slope_all_p"],
                theory_slope_all_p=entry["theory_slope_all_p"],
                errors=entry["errors"],
                error_over_floor=entry["error_over_floor"],
            ))
            checks.append(check(
                "p_refinement_gain_%s_n%s" % (name, ncells),
                "e(p=1)/e(p=5)",
                entry["error_ratio_p1_to_p5"], "ge", P_DECAY_MIN_RATIO,
                errors=entry["errors"],
            ))
    measured = {"p_window": list(P_DECAY_WINDOW)}
    for name in ("advection_1d", "diffusion_1d"):
        measured[name] = {
            key: {
                "measured_slope_window": value["measured_slope_window"],
                "theory_slope_window": value["theory_slope_window"],
                "measured_slope_all_p": value["measured_slope_all_p"],
                "theory_slope_all_p": value["theory_slope_all_p"],
                "errors": value["errors"],
                "projection_floors": value["projection_floors"],
                "error_over_floor": value["error_over_floor"],
                "error_ratio_p1_to_p5": value["error_ratio_p1_to_p5"],
            }
            for key, value in hp[name].items()
        }
    return gate(
        "C4",
        "hp consistency: h-rate p+1 and p-decay rate within 15 % of theory",
        "(a) fixed p, h-refinement: observed order within 10 % of p+1 for p=1..5; "
        "(b) fixed h, p-refinement: the error log-slope in p over p=2..5 is within "
        "15 % of the sharp L2-projection slope, and e(p=1)/e(p=5) >= 1e3",
        checks, measured,
    )


def _rederive(record: dict) -> str:
    measured = record["measured"]
    threshold = record["threshold"]
    comparator = record["comparator"]
    if comparator == "rel_dev_le":
        ok = abs(measured - record["target"]) / abs(record["target"]) <= threshold
    elif comparator == "le":
        ok = measured <= threshold
    elif comparator == "ge":
        ok = measured >= threshold
    else:  # pragma: no cover - programming error
        raise ValueError("unknown comparator %r" % (comparator,))
    return "PASS" if ok else "FAIL"


FIELD_CHECK_KEYS = ("name", "metric", "measured", "comparator", "threshold", "status")


def evaluate_c5(gates: dict[str, dict], artifact_path: Path,
                payload_sha256: str, repo_root: Path,
                test_relpath: Path = TEST_RELPATH) -> dict:
    checks = []
    missing = [
        "%s/%s" % (gate_id, record.get("name")) for gate_id, body in gates.items()
        for record in body["checks"]
        for key in FIELD_CHECK_KEYS if key not in record
    ]
    checks.append(check(
        "gate_records_self_describing", "number of incomplete check records",
        float(len(missing)), "le", 0.0, missing=missing[:10],
    ))
    mismatched = [
        "%s/%s" % (gate_id, record["name"]) for gate_id, body in gates.items()
        for record in body["checks"] if _rederive(record) != record["status"]
    ]
    checks.append(check(
        "verdicts_rederivable", "number of hand-edited verdicts",
        float(len(mismatched)), "le", 0.0, mismatched=mismatched[:10],
    ))
    failing = [gate_id for gate_id, body in gates.items() if body["status"] != "PASS"]
    checks.append(check(
        "all_other_gates_pass", "number of failing gates C1..C4",
        float(len(failing)), "le", 0.0, failing_gates=failing,
    ))
    artifact_ok = artifact_path.exists()
    checks.append(check(
        "artifact_written", "1 if gates_verified.json exists", float(artifact_ok),
        "ge", 1.0, path=_relative_or_absolute(artifact_path, repo_root),
        payload_sha256=payload_sha256,
    ))
    if artifact_ok:
        on_disk = json.loads(artifact_path.read_text())
        checks.append(check(
            "artifact_roundtrip", "1 if schema and gate ids match the payload",
            float(on_disk.get("schema") == SCHEMA
                  and sorted(on_disk.get("gates", {})) == sorted(gates)),
            "ge", 1.0,
        ))
        recorded = on_disk.get("payload_sha256")
        checks.append(check(
            "artifact_hash_matches", "1 if the recorded payload hash matches",
            float(recorded == payload_sha256), "ge", 1.0,
            recorded=recorded, recomputed=payload_sha256,
        ))
    test_path = repo_root / test_relpath
    collected = 0
    collection_ok = False
    if test_path.exists():
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--collect-only", str(test_relpath)],
            cwd=str(repo_root), capture_output=True, text=True,
        )
        collection_ok = completed.returncode == 0
        collected = sum(
            1 for line in completed.stdout.splitlines()
            if line.strip().startswith("tests/") and "::" in line
        )
    checks.append(check(
        "pytest_suite_collects", "1 if the node's pytest file collects cleanly",
        float(collection_ok and collected > 0), "ge", 1.0,
        path=str(test_relpath), collected=collected,
    ))
    return gate(
        "C5",
        "Ladder serialised and pytest-gated",
        "gates_verified.json carries one self-describing record per check "
        "(measured value, target, threshold, comparator); every verdict is "
        "re-derived from its own record; the payload hash is recorded and "
        "re-checked on disk; tests/test_in31_hp_ladder.py collects cleanly",
        checks,
        measured={"artifact": str(artifact_path), "payload_sha256": payload_sha256},
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(advection: dict, diffusion: dict, hp: dict, sod: dict,
                 floor: dict, outdir: Path, repo_root: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, dict] = {}

    def save(fig, name: str, caption: str) -> None:
        path = outdir / name
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        relative = _relative_or_absolute(path, repo_root)
        figures[name] = {"path": relative, "caption": caption}

    # 1) h-convergence for both equations
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for axis, (title, data, resolutions) in zip(
        axes,
        (("advection  $u_t+u_x=0$", advection, ADV_RESOLUTIONS),
         ("diffusion  $u_t=u_{xx}$", diffusion, DIFF_RESOLUTIONS)),
    ):
        for key, entry in sorted(data["degrees"].items(),
                                 key=lambda kv: int(kv[0])):
            degree = int(key)
            hs = np.array([level["h"] for level in entry["levels"]])
            errors = np.array([level["relative_l2"] for level in entry["levels"]])
            axis.loglog(hs, errors, "o-",
                        label="p=%d (obs %.2f)" % (degree, entry["observed_order"]))
            axis.loglog(hs, errors[0] * (hs / hs[0]) ** (degree + 1), "--",
                        color=axis.lines[-1].get_color(), lw=0.8, alpha=0.6)
        axis.set_xlabel("h = 1/N")
        axis.set_ylabel("relative $L_2$ error")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=7, ncol=2)
    fig.suptitle("h-refinement, p = 1..5 (dashed: slope p+1)")
    save(fig, "h_convergence.png",
         "Left: advection L2 error vs h with the RK4 temporal error refined away. "
         "Right: diffusion with exact (matrix-exponential) time integration. "
         "Dashed lines are the theoretical slopes p+1.")

    # 2) p-refinement at fixed h
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for axis, (title, name) in zip(
        axes, (("advection", "advection_1d"), ("diffusion", "diffusion_1d")),
    ):
        for ncells, entry in sorted(hp[name].items(), key=lambda kv: int(kv[0])):
            axis.semilogy(entry["degrees"], entry["errors"], "o-", label="N=%s" % ncells)
            axis.semilogy(entry["degrees"], entry["projection_floors"], "kx--",
                          lw=0.8, alpha=0.6)
        axis.set_xlabel("polynomial degree p")
        axis.set_ylabel("relative $L_2$ error")
        axis.set_title("%s: p-refinement at fixed h" % title)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("hp p-refinement (black x: sharp L2-projection floor)")
    save(fig, "hp_prefinement.png",
         "Error vs polynomial degree at fixed h for both equations. Black crosses "
         "are the sharp L2-projection floor, the predicted decay law.")

    # 3) Sod profiles vs the exact Riemann solution
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    case = sod["cases"].get("p3_tvb") or next(iter(sod["cases"].values()))
    exact = cell_averages(sod["ncells"], sod["final_time"])[1]
    centers = np.array(case["centers_profile"])
    axes[0].plot(centers, case["density_profile"], "-",
                 label="DG p=%d + %s" % (case["degree"], case["limiter"]))
    axes[0].plot(centers, exact["density"], "--", label="exact Riemann")
    axes[1].plot(centers, case["pressure_profile"], "-",
                 label="DG p=%d + %s" % (case["degree"], case["limiter"]))
    axes[1].plot(centers, exact["pressure"], "--", label="exact Riemann")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$\rho$")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("p")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    axes[0].set_title("Sod, t=0.2: density")
    axes[1].set_title("Sod, t=0.2: pressure")
    save(fig, "sod_profiles.png",
         "Sod shock tube at t=0.2 on 160 cells, cell averages against the exact "
         "Riemann solution (density and pressure).")

    # 4) sub-cell locator calibration
    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    labels = ["%.1f" % item["width_cells"] for item in floor["fronts"].values()]
    rms = [item["rms_error_cells"] for item in floor["fronts"].values()]
    worst = [item["max_error_cells"] for item in floor["fronts"].values()]
    index = np.arange(len(labels))
    axis.bar(index - 0.18, rms, 0.36, label="RMS")
    axis.bar(index + 0.18, worst, 0.36, label="max")
    axis.axhline(SHOCK_POSITION_TOLERANCE_CELLS, color="r", ls="--",
                 label="mission gate (2 cells)")
    axis.axhline(0.02, color="k", ls=":",
                 label="inherited 0.02 cell gate (m5280)")
    axis.set_xticks(index)
    axis.set_xticklabels(labels)
    axis.set_xlabel("synthetic front width [cells]")
    axis.set_ylabel("locator error [cells]")
    axis.set_yscale("log")
    axis.legend(fontsize=7)
    axis.set_title("Sub-cell shock locator calibration")
    save(fig, "estimator_calibration.png",
         "Error of the sub-cell shock locator on synthetic fronts of known "
         "position. The 2-cell mission gate sits far above the estimator floor, "
         "whereas the inherited 0.02-cell gate sits below it.")
    return figures


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def provenance(repo_root: Path) -> dict:
    import torch

    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  cwd=str(repo_root)).stdout.strip()
        except Exception:  # pragma: no cover - git missing
            return ""

    return {
        "git_commit": git("rev-parse", "HEAD") or "unknown",
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "worktree_dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float64",
    }


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def markdown_tables(result: dict) -> str:
    """Programmatic tables (no hand-copied numbers) for the report."""
    lines = ["# IN31 ladder - measured numbers", ""]
    adv = result["studies"]["advection_1d"]
    lines += ["## C1 advection h-refinement", "",
              "| p | N | h | rel. L2 | projection floor | CFL used | steps | temporal change |",
              "|---|---|---|---|---|---|---|---|"]
    for key, entry in sorted(adv["degrees"].items(), key=lambda kv: int(kv[0])):
        for level in entry["levels"]:
            lines.append("| %s | %d | %.5f | %.3e | %.3e | %.4f | %d | %.2e |" % (
                key, level["ncells"], level["h"], level["relative_l2"],
                level["projection_floor"], level["cfl"], level["steps"],
                level["temporal_change"]))
        lines.append("| %s | order | | **%.4f (theory %d)** | | | | |" % (
            key, entry["observed_order"], entry["theoretical_order"]))
    diff = result["studies"]["diffusion_1d"]
    lines += ["", "## C2 diffusion h-refinement (time-exact)", "",
              "| p | N | h | rel. L2 | projection floor | lambda_min | lambda_max |",
              "|---|---|---|---|---|---|---|"]
    for key, entry in sorted(diff["degrees"].items(), key=lambda kv: int(kv[0])):
        for level in entry["levels"]:
            lines.append("| %s | %d | %.5f | %.3e | %.3e | %.4f | %.1f |" % (
                key, level["ncells"], level["h"], level["relative_l2"],
                level["projection_floor"], level["stiffness_eigenvalue_min"],
                level["stiffness_eigenvalue_max"]))
        lines.append("| %s | order | | **%.4f (theory %d)** | | | |" % (
            key, entry["observed_order"], entry["theoretical_order"]))
    lines += ["", "## C2b Crank-Nicolson masking (N=8)", "",
              "| p | default steps | CN default | CN 20000 steps | expm | default/expm |",
              "|---|---|---|---|---|---|"]
    for key, entry in sorted(diff["crank_nicolson_masking"].items(),
                             key=lambda kv: int(kv[0])):
        lines.append("| %s | %d | %.3e | %.3e | %.3e | %.1f |" % (
            key, entry["default_steps"], entry["cn_default_relative_l2"],
            entry["cn_20000_steps_relative_l2"], entry["exact_time_relative_l2"],
            entry["default_over_exact"]))
    sod = result["studies"]["sod_1d"]
    lines += ["", "## C3 Sod diagnostics (%d cells, t=%.2f)" % (
        sod["ncells"], sod["final_time"]), "",
        "| case | pos err [cells] | raw interface [cells] | centroid [cells] | "
        "rho jump err | p jump err | contact err [cells] | contact width | "
        "density rel. L2 | Gibbs |", "|---|---|---|---|---|---|---|---|---|---|"]
    for label, case in sorted(sod["cases"].items()):
        lines.append("| %s | %.3f | %.3f | %.3f | %.2f %% | %.2f %% | %.2f | %d | %.3e | %.2e |" % (
            label, case["shock_position_error_cells"],
            case["shock_position_interface_error_cells"],
            case["shock_position_centroid_error_cells"],
            100 * case["density_jump_error"], 100 * case["pressure_jump_error"],
            case["contact"]["contact_position_error_cells"],
            case["contact"]["contact_width_cells"], case["density_relative_l2"],
            case["gibbs_amplitude"]))
    hp = result["studies"]["hp_refinement"]
    lines += ["", "## C4 p-refinement at fixed h", "",
              "| equation | N | errors p=1..5 | slope (2..5) | theory slope | ratio p1/p5 |",
              "|---|---|---|---|---|---|"]
    for name in ("advection_1d", "diffusion_1d"):
        for ncells, entry in sorted(hp[name].items(), key=lambda kv: int(kv[0])):
            lines.append("| %s | %s | %s | %.3f | %.3f | %.1f |" % (
                name, ncells, " / ".join("%.2e" % e for e in entry["errors"]),
                entry["measured_slope_window"], entry["theory_slope_window"],
                entry["error_ratio_p1_to_p5"]))
    floor = result["studies"]["estimator_floor"]
    lines += ["", "## Sub-cell locator calibration", "",
              "| front width [cells] | RMS [cells] | max [cells] |",
              "|---|---|---|"]
    for key, entry in sorted(floor["fronts"].items(), key=lambda kv: float(kv[0])):
        lines.append("| %s | %.3f | %.3f |" % (
            key, entry["rms_error_cells"], entry["max_error_cells"]))
    lines += ["", "## Verdicts", ""]
    for gate_id, body in result["gates"].items():
        lines.append("- **%s** %s - %s" % (gate_id, body["status"], body["name"]))
        if body["failed_checks"]:
            lines.append("  - failed: %s" % ", ".join(body["failed_checks"]))
    return "\n".join(lines) + "\n"


def run_ladder(repo_root: Path | str | None = None,
               output_dir: Path | str | None = None,
               quick: bool = False,
               figures: bool = True,
               test_relpath: Path = TEST_RELPATH) -> dict:
    """Run the whole ladder, write the artifact and return the payload."""
    started = time.time()
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    outdir = Path(output_dir) if output_dir else root / ARTIFACT_RELPATH
    outdir.mkdir(parents=True, exist_ok=True)

    if quick:
        # Same degree range (the C4 window is p=2..5), coarser grids, cheaper Sod.
        adv_degrees, adv_res = ADV_DEGREES, (8, 16)
        diff_degrees, diff_res = DIFF_DEGREES, (4, 8)
        sod_ncells, sod_degrees = 80, (1, 2)
        floor_offsets = 9
    else:
        adv_degrees, adv_res = ADV_DEGREES, ADV_RESOLUTIONS
        diff_degrees, diff_res = DIFF_DEGREES, DIFF_RESOLUTIONS
        sod_ncells, sod_degrees = SOD_NCELLS, SOD_DEGREES
        floor_offsets = 21

    timings = {}
    tick = time.time()
    advection = advection_study(adv_degrees, adv_res)
    timings["advection_1d"] = time.time() - tick
    tick = time.time()
    diffusion = diffusion_study(
        diff_degrees, diff_res,
        cn_degrees=(1, 2) if quick else None,
    )
    timings["diffusion_1d"] = time.time() - tick
    tick = time.time()
    sod = sod_study(sod_ncells, sod_degrees)
    timings["sod_1d"] = time.time() - tick
    tick = time.time()
    hp = hp_study(advection, diffusion)
    floor = estimator_floor(ncells=sod_ncells, offsets=floor_offsets)
    timings["hp_and_estimator"] = time.time() - tick

    gates: dict[str, dict] = {}
    gates["C1"] = evaluate_c1(advection)
    gates["C2"] = evaluate_c2(diffusion)
    gates["C3"] = evaluate_c3(sod, floor)
    gates["C4"] = evaluate_c4(advection, diffusion, hp)

    config = {
        "quick": quick,
        "advection": {"degrees": list(adv_degrees), "resolutions": list(adv_res),
                      "final_time": ADV_FINAL_TIME, "cfl": ADV_CFL},
        "diffusion": {"degrees": list(diff_degrees), "resolutions": list(diff_res),
                      "final_time": DIFF_FINAL_TIME, "penalty": 20.0},
        "sod": {"ncells": sod_ncells, "degrees": list(sod_degrees),
                "limiters": list(SOD_LIMITERS), "cfl": SOD_CFL,
                "final_time": SOD_FINAL_TIME},
        "thresholds": {
            "order_relative_tolerance": ORDER_REL_TOLERANCE,
            "hp_rate_relative_tolerance": HP_RATE_REL_TOLERANCE,
            "shock_position_tolerance_cells": SHOCK_POSITION_TOLERANCE_CELLS,
            "shock_strength_tolerance": SHOCK_STRENGTH_TOLERANCE,
            "temporal_audit_tolerance": TEMPORAL_AUDIT_TOLERANCE,
            "p_decay_window": list(P_DECAY_WINDOW),
            "p_decay_min_ratio": P_DECAY_MIN_RATIO,
        },
    }
    studies = {"advection_1d": advection, "diffusion_1d": diffusion,
               "sod_1d": sod, "hp_refinement": hp, "estimator_floor": floor}

    measurement = {"schema": SCHEMA, "config": config, "studies": studies,
                   "gates": gates}
    payload_sha256 = canonical_sha256(measurement)

    figures_map = {}
    if figures:
        tick = time.time()
        figures_map = make_figures(advection, diffusion, hp, sod, floor, outdir, root)
        timings["figures"] = time.time() - tick

    artifact_path = outdir / "gates_verified.json"
    payload = dict(measurement)
    payload["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload["provenance"] = provenance(root)
    payload["gates"] = gates
    payload["verdict"] = "FAIL" if any(
        body["status"] != "PASS" for body in gates.values()) else "PASS"
    payload["figures"] = figures_map
    payload["timings_seconds"] = timings
    payload["payload_sha256"] = payload_sha256
    artifact_path.write_text(json.dumps(payload, indent=1))

    # C5 sees the artifact on disk; then the final payload is written again.
    gates["C5"] = evaluate_c5(gates, artifact_path, payload_sha256, root, test_relpath)
    payload["gates"] = gates
    payload["verdict"] = "FAIL" if any(
        body["status"] != "PASS" for body in gates.values()) else "PASS"
    payload["runtime_seconds"] = time.time() - started
    artifact_path.write_text(json.dumps(payload, indent=1))
    (outdir / "tables.md").write_text(markdown_tables(payload))

    rows = ["degree,ncells,h,relative_l2,projection_floor,steps"]
    for key, entry in sorted(advection["degrees"].items(), key=lambda kv: int(kv[0])):
        for level in entry["levels"]:
            rows.append("%s,%d,%.6f,%.6e,%.6e,%d" % (
                key, level["ncells"], level["h"], level["relative_l2"],
                level["projection_floor"], level["steps"]))
    (outdir / "advection_convergence.csv").write_text("\n".join(rows) + "\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="reduced configuration (smoke test)")
    parser.add_argument("--output-dir", default=None,
                        help="artifact directory (default: docs/benchmarks/results/in31-hp-ladder)")
    parser.add_argument("--no-figures", action="store_true", help="skip the PNGs")
    args = parser.parse_args(argv)
    payload = run_ladder(output_dir=args.output_dir, quick=args.quick,
                         figures=not args.no_figures)
    print("verdict: %s" % payload["verdict"])
    for gate_id, body in payload["gates"].items():
        print("  %s %s" % (gate_id, body["status"]))
        for failed in body["failed_checks"]:
            print("    FAILED: %s" % failed)
    print("runtime: %.1f s" % payload["runtime_seconds"])
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
