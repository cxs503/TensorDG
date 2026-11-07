"""Zero-to-one verification ladder for the TensorDG kernels (gates C1--C6).

The ladder walks the solver from *smooth analytic* problems to a *discontinuous*
one and finally to the *order-of-accuracy* contract of the method:

=========  ==========================================================  ==================
gate       physics                                                     reference
=========  ==========================================================  ==================
C1         1D/2D linear advection ``u_t + a.grad u = 0``, sine wave     analytic translation
C1b        2D tensor-product advection on the periodic unit square      analytic translation
C2         1D heat conduction ``u_t = u_xx`` (SIPG, Crank--Nicolson)    analytic decaying mode
C3         Sod shock tube (HLLC + SSP-RK3, DG p = 1,2,3)                exact Riemann solution
C4         hp order: h-rate == p+1, p-refinement == projection floor    theory + analytic
C5         negative control: the unlimited scheme must oscillate        physical bounds
C6         every criterion is pytest-gated and serialised               artifact contract
=========  ==========================================================  ==================

Design rules (why this is verification and not a plausibility check):

* No reference value is produced by the code under test.  Analytic solutions,
  the exact Riemann solver (:mod:`benchmarks.riemann_exact`), and the *sharp*
  ``L2``-projection error of ``sin(2*pi*x)`` -- computed from Legendre
  quadrature only -- supply every reference.
* Every gate stores its measured numbers **and** its threshold in
  ``gates_verified.json``; ``tests/test_verification_ladder.py`` recomputes the
  cheap gates and audits the artifact, so a hand-edited JSON cannot pass.
* Thresholds are the campaign criteria (order within 15 % of ``p+1``, shock
  position within 2 % of a cell).  A failing measurement is reported as
  ``FAIL``; it is never rewritten to pass.

Usage::

    python -m benchmarks.verification_ladder --quick      # reduced cost
    python -m benchmarks.verification_ladder             # full ladder
    python -m benchmarks.verification_ladder --output docs/benchmarks/results/verification_ladder

Exit status is 0 only when every gate passes.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import tensordg
from tensordg import (
    error_2d,
    heat_error,
    l2_error,
    solve_advection,
    solve_advection_2d,
    solve_euler_dg_1d,
    solve_heat_1d,
)
from tensordg.convergence import observed_order, pairwise_orders

from benchmarks.riemann_exact import GAMMA, star_state
from benchmarks.sod_exact import cell_averages

# ---------------------------------------------------------------------------
# Gate thresholds (campaign criteria; never lowered to make a run pass)
# ---------------------------------------------------------------------------

ORDER_RELATIVE_TOLERANCE = 0.15          # C1/C2/C4: |p_obs-(p+1)|/(p+1) <= 15 %
SHOCK_POSITION_TOLERANCE_CELLS = 0.02    # C3: |x_num-x_exact| <= 2 % of dx
GIBBS_MONOTONICITY_TOLERANCE = 1.0e-3    # C3 slack, fraction of the density jump
GIBBS_NEGATIVE_CONTROL_FLOOR = 1.0e-3    # C5: unlimited overshoot must exceed this

# Campaign states are (1,0,1) | (0.125,0,0.1) on [0,1] at t = 0.2 (Sod 1978).
SOD_LEFT = (1.0, 0.0, 1.0)
SOD_RIGHT = (0.125, 0.0, 0.1)
SOD_FINAL_TIME = 0.2
SOD_DIAPHRAGM = 0.5

JUMP = SOD_LEFT[0] - SOD_RIGHT[0]        # density jump used to normalise Gibbs


# ---------------------------------------------------------------------------
# Analytic references (independent of the DG code under test)
# ---------------------------------------------------------------------------

def sine_projection_error(ncells: int, degree: int) -> float:
    """Sharp ``L2``-projection error of ``sin(2*pi*x)`` onto piecewise ``P_p``.

    This is the asymptotic floor of a ``P_p`` DG solution of the advection
    equation with exact time integration: the discrete error cannot fall below
    the interpolation error of the same polynomial space.  It is evaluated with
    high-order Gauss--Legendre quadrature from Legendre polynomials only, so it
    contains no DG solver code.
    """
    if ncells < 1 or degree < 0:
        raise ValueError("ncells >= 1 and degree >= 0 are required")
    nq = max(4 * (degree + 1), 40)
    nodes, weights = np.polynomial.legendre.leggauss(nq)
    h = 1.0 / ncells
    x = (np.arange(ncells)[:, None] + 0.5 + 0.5 * nodes[None, :]) * h
    exact = np.sin(2.0 * math.pi * x)
    basis = np.stack(
        [
            np.polynomial.legendre.legval(nodes, np.eye(degree + 1)[k])
            for k in range(degree + 1)
        ],
        axis=1,
    )
    scaling = (2.0 * np.arange(degree + 1) + 1.0) / 2.0
    coefficients = np.einsum("q,cq,qk->ck", weights, exact, basis) * scaling
    approximate = np.einsum("ck,qk->cq", coefficients, basis)
    error_squared = 0.5 * h * np.sum(weights[None, :] * (approximate - exact) ** 2)
    return float(math.sqrt(error_squared))


def sod_reference(final_time: float = SOD_FINAL_TIME) -> dict:
    """Exact Sod star state and shock trajectory from the Rankine--Hugoniot data.

    The star pressure and velocity come from the iterative exact Riemann solver;
    the shock speed then follows the single-shock jump condition
    ``s = u_R + a_R sqrt((g+1)/(2g) p*/p_R + (g-1)/(2g))`` (Toro, *Riemann
    Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., Sec. 4.3).
    """
    p_star, u_star = star_state(SOD_LEFT, SOD_RIGHT, GAMMA)
    rho_r, u_r, p_r = SOD_RIGHT
    a_r = math.sqrt(GAMMA * p_r / rho_r)
    ratio = p_star / p_r
    if p_star <= p_r:
        raise ValueError("the right wave of Sod's problem must be a shock")
    shock_speed = u_r + a_r * math.sqrt(
        (GAMMA + 1.0) / (2.0 * GAMMA) * ratio + (GAMMA - 1.0) / (2.0 * GAMMA)
    )
    rho_star_right = rho_r * (
        (ratio + (GAMMA - 1.0) / (GAMMA + 1.0))
        / ((GAMMA - 1.0) / (GAMMA + 1.0) * ratio + 1.0)
    )
    return {
        "p_star": float(p_star),
        "u_star": float(u_star),
        "rho_star_right": float(rho_star_right),
        "shock_speed": float(shock_speed),
        "shock_position": float(SOD_DIAPHRAGM + shock_speed * final_time),
        "final_time": float(final_time),
        "gamma": float(GAMMA),
    }


# ---------------------------------------------------------------------------
# Shock-tube diagnostics
# ---------------------------------------------------------------------------

def locate_shock(
    density: np.ndarray,
    left_level: float,
    right_level: float,
    window: tuple[float, float] = (0.75, 1.0),
) -> dict:
    """Locate the captured shock by interpolating the mid-level crossing.

    The interface with the largest cell-average density jump *inside* the shock
    window is taken to be the shock interface; the ``(left_level+right_level)/2``
    crossing is then interpolated linearly between the two adjacent cell
    centres.  This is the estimator used for the Burgers shock tube in
    :func:`tensordg.burgers_shock_1d.shock_position`, applied to Sod's density
    with ``left_level = rho_star_right``, ``right_level = rho_R``.

    The window excludes the contact discontinuity, which carries a *larger*
    density jump than the shock (0.161 vs 0.141 for Sod) and would otherwise win
    the ``argmax``.

    Returns both the interpolated position and the raw interface position, so
    the report can show how much sub-cell interpolation is responsible for.
    """
    values = np.asarray(density, dtype=float)
    if values.ndim != 1 or values.size < 8:
        raise ValueError("density must be a 1D array with at least 8 cells")
    ncells = values.size
    dx = 1.0 / ncells
    centers = (np.arange(ncells) + 0.5) * dx
    inside = (centers[1:] >= window[0]) & (centers[1:] <= window[1])
    if not np.any(inside):
        raise ValueError("the shock window contains no interior interface")
    jumps = np.abs(np.diff(values))
    jumps = np.where(inside, jumps, -1.0)
    index = int(np.argmax(jumps))
    middle = 0.5 * (left_level + right_level)
    a, b = values[index], values[index + 1]
    fraction = 0.5 if a == b else float(np.clip((middle - a) / (b - a), 0.0, 1.0))
    position = centers[index] + fraction * dx
    return {
        "position": float(position),
        "interface": float(centers[index] + 0.5 * dx),
        "fraction": fraction,
        "jump": float(jumps[index]),
        "window": [float(window[0]), float(window[1])],
    }


def locate_shock_centroid(
    coefficients: np.ndarray,
    window: tuple[float, float] = (0.75, 1.0),
    ncells_window: int = 3,
) -> float:
    """Wave-front centroid of ``|d rho/dx|`` around the steepest interface.

    Same construction as :func:`tensordg.burgers_shock_1d.shock_position_moment`:
    the derivative is evaluated from the modal DG representation on Gauss nodes
    inside a window of ``ncells_window`` cells around the steepest cell-average
    jump, and the position is the first moment of ``|d rho/dx|``.  It is kept as
    a diagnostic because it converges at first order and is insensitive to the
    mid-level crossing falling near a cell face.
    """
    coeff = np.asarray(coefficients, dtype=float)
    if coeff.ndim != 2:
        raise ValueError("coefficients must be shaped (ncells, nmodes)")
    ncells, nmodes = coeff.shape
    degree = nmodes - 1
    dx = 1.0 / ncells
    centers = (np.arange(ncells) + 0.5) * dx
    inside = (centers[1:] >= window[0]) & (centers[1:] <= window[1])
    jumps = np.where(inside, np.abs(np.diff(coeff[:, 0])), -1.0)
    steepest = int(np.argmax(jumps))
    lo = max(steepest - ncells_window, 0)
    hi = min(steepest + ncells_window + 2, ncells)
    nq = max(2 * degree + 4, 6)
    nodes, _ = np.polynomial.legendre.leggauss(nq)
    derivatives = np.stack(
        [
            np.polynomial.legendre.legval(
                nodes, np.polynomial.legendre.legder(np.eye(nmodes)[k]),
            ) if k > 0 else np.zeros(nq)
            for k in range(nmodes)
        ],
        axis=1,
    )
    derivative = np.einsum("cm,qm->cq", coeff[lo:hi], derivatives)
    xq = (np.arange(lo, hi)[:, None] + 0.5 * (nodes[None, :] + 1.0)) * dx
    weight = np.abs(derivative)
    total = weight.sum()
    if total == 0.0:  # pragma: no cover - degenerate data
        return float(centers[steepest])
    return float((xq * weight).sum() / total)


def gibbs_amplitude(density: np.ndarray) -> dict:
    """Non-physical overshoot/undershoot of a density profile, normalised.

    Sod's exact solution reaches neither ``rho > 1`` nor ``rho < 0.125``, so any
    excursion beyond these bounds is a Gibbs oscillation.  Both excursions are
    divided by the initial density jump, making the amplitude independent of
    the shock strength.
    """
    values = np.asarray(density, dtype=float)
    upper = max(0.0, float(values.max()) - SOD_LEFT[0]) / JUMP
    lower = max(0.0, SOD_RIGHT[0] - float(values.min())) / JUMP
    return {
        "overshoot": upper,
        "undershoot": lower,
        "amplitude": max(upper, lower),
    }


def total_variation_ratio(density: np.ndarray,
                         final_time: float = SOD_FINAL_TIME) -> float:
    """``TV(u_h) / TV(u_exact)`` of the cell averages (1 = non-oscillatory)."""
    values = np.asarray(density, dtype=float)
    _, exact_fields = cell_averages(values.size, final_time)
    exact = np.asarray(exact_fields["density"], dtype=float)
    return float(np.sum(np.abs(np.diff(values))) / np.sum(np.abs(np.diff(exact))))


# ---------------------------------------------------------------------------
# Ladder blocks
# ---------------------------------------------------------------------------

def advection_ladder(
    degrees: Sequence[int] = (1, 2, 3),
    resolutions: Sequence[int] = (8, 16, 32),
    final_time: float = 0.5,
    cfl: float = 0.3,
) -> dict:
    """C1: 1D periodic advection ``u_t + u_x = 0`` against the translated sine."""
    block: dict = {"study": "advection_1d", "final_time": final_time, "cfl": cfl,
                   "degrees": {}}
    for degree in degrees:
        errors, tables = [], []
        for ncells in resolutions:
            centers, coefficients, metadata = solve_advection(
                ncells=ncells, degree=degree, final_time=final_time, cfl=cfl,
            )
            error = l2_error(centers, coefficients, final_time)
            errors.append(error)
            tables.append({
                "ncells": ncells,
                "h": 1.0 / ncells,
                "relative_l2": error,
                "projection_floor": sine_projection_error(ncells, degree),
                "steps": metadata["steps"],
                "dt": metadata["dt"],
            })
        block["degrees"][degree] = {
            "levels": tables,
            "pairwise_orders": pairwise_orders(resolutions, errors),
            "observed_order": observed_order(resolutions, errors),
            "theoretical_order": degree + 1,
        }
    return block


def advection_2d_ladder(
    degrees: Sequence[int] = (1, 2),
    resolutions: Sequence[int] = (8, 16, 32),
    final_time: float = 0.2,
    cfl: float = 0.3,
    velocity: tuple[float, float] = (1.0, 0.5),
) -> dict:
    """C1b: tensor-product DG advection on the periodic unit square."""
    block: dict = {"study": "advection_2d", "final_time": final_time, "cfl": cfl,
                   "velocity": list(velocity), "degrees": {}}
    for degree in degrees:
        errors, tables = [], []
        for ncells in resolutions:
            x, y, coefficients, metadata = solve_advection_2d(
                ncells=ncells, degree=degree, final_time=final_time,
                velocity=velocity, cfl=cfl,
            )
            error = error_2d(x, y, coefficients, final_time,
                             velocity=velocity)["relative_l2"]
            errors.append(error)
            tables.append({"ncells": ncells, "h": 1.0 / ncells,
                           "relative_l2": error, "steps": metadata["steps"]})
        block["degrees"][degree] = {
            "levels": tables,
            "pairwise_orders": pairwise_orders(resolutions, errors),
            "observed_order": observed_order(resolutions, errors),
            "theoretical_order": degree + 1,
        }
    return block


def heat_ladder(
    degrees: Sequence[int] = (1, 2, 3),
    resolutions: Sequence[int] = (4, 8, 16),
    final_time: float = 0.1,
    time_steps: int = 2000,
) -> dict:
    """C2: SIPG/Crank--Nicolson heat equation against the decaying sine mode.

    ``time_steps`` is deliberately large: the temporal error of Crank--Nicolson
    would otherwise mask the ``h**(p+1)`` spatial rate at the finest level (see
    the limitations section of the report).
    """
    block: dict = {"study": "heat_1d", "final_time": final_time,
                   "time_steps": time_steps, "degrees": {}}
    for degree in degrees:
        errors, tables = [], []
        for ncells in resolutions:
            centers, coefficients, metadata = solve_heat_1d(
                ncells=ncells, degree=degree, final_time=final_time,
                time_steps=time_steps,
            )
            metrics = heat_error(centers, coefficients, final_time)
            errors.append(metrics["relative_l2"])
            tables.append({
                "ncells": ncells,
                "h": 1.0 / ncells,
                "relative_l2": metrics["relative_l2"],
                "absolute_l2": metrics["absolute_l2"],
                "dt": final_time / time_steps,
            })
        block["degrees"][degree] = {
            "levels": tables,
            "pairwise_orders": pairwise_orders(resolutions, errors),
            "observed_order": observed_order(resolutions, errors),
            "theoretical_order": degree + 1,
        }
    return block


def sod_case(
    ncells: int,
    degree: int,
    limiter: str | None,
    final_time: float = SOD_FINAL_TIME,
    cfl: float = 0.2,
    limiter_stage: str = "final",
    keep_profile: bool = True,
) -> dict:
    """One Sod run plus its shock-tube diagnostics."""
    started = time.time()
    centers, coefficients, metadata = solve_euler_dg_1d(
        SOD_LEFT, SOD_RIGHT, ncells=ncells, degree=degree, final_time=final_time,
        diaphragm=SOD_DIAPHRAGM, cfl=cfl, limiter=limiter,
        limiter_stage=limiter_stage,
    )
    runtime = time.time() - started
    density = coefficients[:, 0, 0].detach().cpu().numpy()
    reference = sod_reference(final_time)
    shock = locate_shock(density, reference["rho_star_right"], SOD_RIGHT[0])
    centroid = locate_shock_centroid(
        coefficients[:, :, 0].detach().cpu().numpy(),
    )
    amplitude = gibbs_amplitude(density)
    _, exact_fields = cell_averages(ncells, final_time)
    exact = np.asarray(exact_fields["density"], dtype=float)
    relative_l2 = float(
        math.sqrt(np.mean((density - exact) ** 2)) / math.sqrt(np.mean(exact ** 2))
    )
    dx = 1.0 / ncells
    return {
        "ncells": ncells,
        "degree": degree,
        "limiter": limiter if limiter else "none",
        "limiter_stage": limiter_stage,
        "final_time": final_time,
        "cfl": cfl,
        "dx": dx,
        "steps": metadata["steps"],
        "shock_position": shock["position"],
        "shock_position_interface": shock["interface"],
        "shock_position_centroid": centroid,
        "shock_position_exact": reference["shock_position"],
        "shock_position_error": abs(shock["position"] - reference["shock_position"]),
        "shock_position_error_cells": (
            abs(shock["position"] - reference["shock_position"]) / dx
        ),
        "shock_position_centroid_cells": (
            abs(centroid - reference["shock_position"]) / dx
        ),
        "shock_level_fraction": shock["fraction"],
        "gibbs": amplitude,
        "gibbs_amplitude": amplitude["amplitude"],
        "total_variation_ratio": total_variation_ratio(density, final_time),
        "density_relative_l2": relative_l2,
        "density_profile": density.tolist() if keep_profile else None,
        "runtime_seconds": runtime,
    }


def sod_ladder(
    ncells: int = 200,
    degrees: Sequence[int] = (1, 2, 3),
    limiters: Sequence[str | None] = (None, "tvb"),
    final_time: float = SOD_FINAL_TIME,
    cfl: float = 0.2,
    limiter_stage: str = "final",
) -> dict:
    """C3 + C5 data: Sod at DG p = 1,2,3, unlimited and TVB-limited."""
    cases: dict[str, dict] = {}
    for degree in degrees:
        for limiter in limiters:
            label = f"p{degree}_{limiter if limiter else 'unlimited'}"
            cases[label] = sod_case(
                ncells=ncells, degree=degree, limiter=limiter,
                final_time=final_time, cfl=cfl, limiter_stage=limiter_stage,
            )
    return {
        "study": "sod_1d",
        "ncells": ncells,
        "final_time": final_time,
        "cfl": cfl,
        "limiter_stage": limiter_stage,
        "reference": sod_reference(final_time),
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

def _order_check(name: str, measured: float, theoretical: float,
                 tolerance: float = ORDER_RELATIVE_TOLERANCE) -> dict:
    relative = abs(measured - theoretical) / theoretical
    return {
        "name": name,
        "measured": measured,
        "theoretical": theoretical,
        "relative_error": relative,
        "tolerance": tolerance,
        "status": "PASS" if relative <= tolerance else "FAIL",
    }


def _gate(gate_id: str, name: str, criterion: str, checks: list[dict],
          measured: dict | None = None) -> dict:
    failed = [check for check in checks if check["status"] != "PASS"]
    return {
        "id": gate_id,
        "name": name,
        "criterion": criterion,
        "status": "FAIL" if failed else "PASS",
        "checks": checks,
        "failed_checks": [check["name"] for check in failed],
        "measured": measured or {},
    }


def evaluate_c1(advection: dict) -> dict:
    checks = [
        _order_check(f"advection_1d_p{degree}", data["observed_order"], degree + 1)
        for degree, data in sorted(advection["degrees"].items())
    ]
    return _gate(
        "C1",
        "linear advection L2 error converges at h**(p+1), p = 1,2,3",
        "least-squares observed order on (8,16,32) cells within 15 % of p+1",
        checks,
        measured={str(degree): {"observed_order": data["observed_order"],
                                "pairwise_orders": data["pairwise_orders"]}
                  for degree, data in sorted(advection["degrees"].items())},
    )


def evaluate_c1b(advection_2d: dict) -> dict:
    checks = [
        _order_check(f"advection_2d_p{degree}", data["observed_order"], degree + 1)
        for degree, data in sorted(advection_2d["degrees"].items())
    ]
    return _gate(
        "C1b",
        "2D tensor-product advection L2 error converges at h**(p+1)",
        "least-squares observed order on (8,16,32)^2 cells within 15 % of p+1",
        checks,
        measured={str(degree): {"observed_order": data["observed_order"],
                                "pairwise_orders": data["pairwise_orders"]}
                  for degree, data in sorted(advection_2d["degrees"].items())},
    )


def evaluate_c2(heat: dict) -> dict:
    checks = [
        _order_check(f"heat_1d_p{degree}", data["observed_order"], degree + 1)
        for degree, data in sorted(heat["degrees"].items())
    ]
    return _gate(
        "C2",
        "heat equation L2 error converges at h**(p+1), p = 1,2,3",
        "least-squares observed order on (4,8,16) cells within 15 % of p+1",
        checks,
        measured={str(degree): {"observed_order": data["observed_order"],
                                "pairwise_orders": data["pairwise_orders"]}
                  for degree, data in sorted(heat["degrees"].items())},
    )


def evaluate_c3(sod: dict) -> dict:
    checks = []
    degrees = sorted({case["degree"] for case in sod["cases"].values()})
    for degree in degrees:
        for limiter in ("tvb", "unlimited"):
            case = sod["cases"].get(f"p{degree}_{limiter}")
            if case is None:
                continue
            error = case["shock_position_error_cells"]
            checks.append({
                "name": f"sod_shock_position_p{degree}_{limiter}",
                "measured_cells": error,
                "tolerance_cells": SHOCK_POSITION_TOLERANCE_CELLS,
                "centroid_cells": case["shock_position_centroid_cells"],
                "limiter": case["limiter"],
                "status": "PASS" if error <= SHOCK_POSITION_TOLERANCE_CELLS else "FAIL",
            })
    amplitudes = [
        (degree, sod["cases"][f"p{degree}_unlimited"]["gibbs_amplitude"])
        for degree in degrees if f"p{degree}_unlimited" in sod["cases"]
    ]
    for (degree_lo, amplitude_lo), (degree_hi, amplitude_hi) in zip(
        amplitudes, amplitudes[1:]
    ):
        decrease = amplitude_lo - amplitude_hi
        checks.append({
            "name": f"gibbs_monotone_p{degree_lo}_to_p{degree_hi}",
            "lower_degree_amplitude": amplitude_lo,
            "higher_degree_amplitude": amplitude_hi,
            "decrease": decrease,
            "tolerance": GIBBS_MONOTONICITY_TOLERANCE,
            "status": "PASS" if decrease >= -GIBBS_MONOTONICITY_TOLERANCE else "FAIL",
        })
    return _gate(
        "C3",
        "Sod shock position to 2 % of a cell; Gibbs amplitude decreases with p",
        "|x_num - x_exact| <= 0.02 dx against the exact Riemann solution (shock "
        "located by sub-cell profile matching); unlimited overshoot amplitude "
        "non-increasing in p",
        checks,
        measured={
            label: {
                "shock_position_error_cells": case["shock_position_error_cells"],
                "shock_position_centroid_cells":
                    case["shock_position_centroid_cells"],
                "shock_level_fraction": case["shock_level_fraction"],
                "gibbs_amplitude": case["gibbs_amplitude"],
                "gibbs_overshoot": case["gibbs"]["overshoot"],
                "gibbs_undershoot": case["gibbs"]["undershoot"],
                "density_relative_l2": case["density_relative_l2"],
                "total_variation_ratio": case["total_variation_ratio"],
                "steps": case["steps"],
            }
            for label, case in sorted(sod["cases"].items())
        },
    )


def evaluate_c4(advection: dict, heat: dict, advection_2d: dict | None = None) -> dict:
    """hp gate: h-rate equals p+1 *and* p-refinement tracks the projection floor."""
    checks = []
    for degree, data in sorted(advection["degrees"].items()):
        checks.append(_order_check(
            f"h_refinement_advection_1d_p{degree}", data["observed_order"], degree + 1,
        ))
    for degree, data in sorted(heat["degrees"].items()):
        checks.append(_order_check(
            f"h_refinement_heat_1d_p{degree}", data["observed_order"], degree + 1,
        ))
    if advection_2d is not None:
        for degree, data in sorted(advection_2d["degrees"].items()):
            checks.append(_order_check(
                f"h_refinement_advection_2d_p{degree}", data["observed_order"],
                degree + 1,
            ))
    # p-refinement at fixed h: the measured error ratio e(p)/e(p+1) must follow
    # the ratio of the analytic L2-projection floor, which is what the hp error
    # estimate e ~ (h/2)**(p+1) predicts on a fixed mesh (asymptotic regime).
    degree_list = sorted(advection["degrees"])
    ratios: dict[str, dict] = {}
    for index, level in enumerate(advection["degrees"][degree_list[0]]["levels"]):
        ncells = level["ncells"]
        measured_errors = [
            advection["degrees"][degree]["levels"][index]["relative_l2"]
            for degree in degree_list
        ]
        floor_errors = [sine_projection_error(ncells, degree) for degree in degree_list]
        entry = {"ncells": ncells, "degrees": degree_list,
                 "measured_errors": measured_errors,
                 "projection_floor": floor_errors, "ratios": []}
        for position in range(len(degree_list) - 1):
            measured_ratio = measured_errors[position] / measured_errors[position + 1]
            floor_ratio = floor_errors[position] / floor_errors[position + 1]
            relative = abs(measured_ratio - floor_ratio) / floor_ratio
            entry["ratios"].append({
                "degree_pair": [degree_list[position], degree_list[position + 1]],
                "measured_ratio": measured_ratio,
                "projection_ratio": floor_ratio,
                "relative_error": relative,
                "status": "PASS" if relative <= ORDER_RELATIVE_TOLERANCE else "FAIL",
            })
        ratios[str(ncells)] = entry
        if ncells < 8:
            # Deliberately excluded: on 4 cells the asymptotic expansion of the
            # projection floor is not yet valid, so the ratio has no theory to
            # be compared against.  The measured value is still reported.
            continue
        for item in entry["ratios"]:
            checks.append({
                "name": f"p_refinement_n{ncells}_p{item['degree_pair'][0]}"
                        f"_to_p{item['degree_pair'][1]}",
                "measured": item["measured_ratio"],
                "theoretical": item["projection_ratio"],
                "relative_error": item["relative_error"],
                "tolerance": ORDER_RELATIVE_TOLERANCE,
                "status": item["status"],
            })
    return _gate(
        "C4",
        "hp convergence: h-rate == p+1 and p-refinement == analytic projection floor",
        "observed h-order within 15 % of p+1 for every p; e(p)/e(p+1) within 15 % "
        "of the sharp L2-projection ratio at fixed h (ncells >= 8)",
        checks,
        measured={"p_refinement": ratios},
    )


def evaluate_c5(sod: dict) -> dict:
    """Negative control: the unlimited scheme must oscillate at the shock."""
    checks = []
    degrees = sorted({case["degree"] for case in sod["cases"].values()})
    for degree in degrees:
        unlimited = sod["cases"].get(f"p{degree}_unlimited")
        limited = sod["cases"].get(f"p{degree}_tvb")
        if unlimited is None or limited is None:
            continue
        checks.append({
            "name": f"unlimited_oscillates_p{degree}",
            "measured": unlimited["gibbs_amplitude"],
            "floor": GIBBS_NEGATIVE_CONTROL_FLOOR,
            "status": "PASS"
            if unlimited["gibbs_amplitude"] >= GIBBS_NEGATIVE_CONTROL_FLOOR else "FAIL",
        })
        reduction = unlimited["gibbs_amplitude"] - limited["gibbs_amplitude"]
        checks.append({
            "name": f"limiter_reduces_oscillation_p{degree}",
            "unlimited_amplitude": unlimited["gibbs_amplitude"],
            "limited_amplitude": limited["gibbs_amplitude"],
            "reduction": reduction,
            "minimum_reduction": 0.0,
            "status": "PASS" if reduction > 0.0 else "FAIL",
        })
        checks.append({
            "name": f"unlimited_variation_p{degree}",
            "unlimited_tv_ratio": unlimited["total_variation_ratio"],
            "limited_tv_ratio": limited["total_variation_ratio"],
            "status": "PASS" if unlimited["total_variation_ratio"] > 1.0 else "FAIL",
        })
    return _gate(
        "C5",
        "negative control: no limiter => non-physical oscillation at the shock",
        "unlimited Gibbs amplitude >= 0.1 % of the density jump, strictly larger "
        "than the TVB-limited amplitude, and total variation above the exact one",
        checks,
        measured={
            label: {"gibbs_amplitude": case["gibbs_amplitude"],
                    "total_variation_ratio": case["total_variation_ratio"]}
            for label, case in sorted(sod["cases"].items())
        },
    )


def evaluate_c6(gates: dict[str, dict], test_path: Path,
                artifact: Path | None = None) -> dict:
    """Artifact contract: every gate serialised, pytest suite present and collectable."""
    checks = []
    for gate_id in ("C1", "C1b", "C2", "C3", "C4", "C5"):
        gate = gates.get(gate_id)
        checks.append({
            "name": f"serialised_{gate_id}",
            "status": "PASS" if gate is not None and gate.get("checks") else "FAIL",
            "check_count": len(gate["checks"]) if gate else 0,
        })
    exists = test_path.is_file()
    collected = 0
    if exists:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", str(test_path)],
            capture_output=True, text=True, timeout=600,
        )
        collected = sum(1 for line in completed.stdout.splitlines() if "::" in line)
    checks.append({
        "name": "pytest_suite",
        "path": str(test_path),
        "exists": exists,
        "collected_tests": collected,
        "status": "PASS" if exists and collected >= 6 else "FAIL",
    })
    if artifact is not None:
        checks.append({
            "name": "artifact_written",
            "path": str(artifact),
            "exists": artifact.is_file(),
            "status": "PASS" if artifact.is_file() else "FAIL",
        })
    return _gate(
        "C6",
        "every criterion is pytest-gated and serialised with its thresholds",
        "C1,C1b,C2,C3,C4,C5 serialised with measured values and thresholds; "
        "tests/test_verification_ladder.py exists and collects >= 6 tests; "
        "gates_verified.json present on disk",
        checks,
    )


def make_figures(advection: dict, heat: dict, advection_2d: dict, sod: dict,
                 p_refinement: dict, output: Path) -> list[Path]:
    """Comparison figures: h-convergence, Sod profiles, Gibbs amplitude, p-refinement."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []

    def reference_slope(axis, x0, y0, order, label):
        xs = np.array([x0, x0 * 4.0])
        axis.plot(xs, y0 * (xs / x0) ** (-order), "k--", linewidth=1.0, alpha=0.6,
                  label=label)

    def h_convergence(block: dict, title: str, name: str) -> None:
        figure, axis = plt.subplots(figsize=(6.4, 4.6))
        for degree, data in sorted(block["degrees"].items()):
            h = np.array([level["h"] for level in data["levels"]])
            error = np.array([level["relative_l2"] for level in data["levels"]])
            axis.loglog(h, error, "o-",
                        label=f"p={degree}, order={data['observed_order']:.2f}")
            reference_slope(axis, h[-1], error[-1], degree + 1, f"slope {degree + 1}")
        axis.set_xlabel("h = 1/N")
        axis.set_ylabel("relative $L^2$ error")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = output / name
        figure.savefig(path, dpi=140)
        plt.close(figure)
        figures.append(path)

    h_convergence(advection, "1D linear advection", "c1_advection_1d_convergence.png")
    h_convergence(advection_2d, "2D tensor-product advection",
                  "c1_advection_2d_convergence.png")
    h_convergence(heat, "1D heat equation (SIPG, Crank--Nicolson)",
                  "c2_heat_1d_convergence.png")

    reference = sod["reference"]
    ncells = sod["ncells"]
    centers = (np.arange(ncells) + 0.5) / ncells
    _, exact_fields = cell_averages(ncells, sod["final_time"])
    exact = np.asarray(exact_fields["density"], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].plot(centers, exact, "k-", linewidth=1.6, label="exact Riemann")
    for degree in sorted({case["degree"] for case in sod["cases"].values()}):
        case = sod["cases"].get(f"p{degree}_tvb")
        if case is None or case["density_profile"] is None:
            continue
        axes[0].plot(centers, case["density_profile"], linewidth=1.1,
                     label=f"TVB p={degree}")
    axes[0].axvline(reference["shock_position"], color="0.5", linestyle=":",
                    label="exact shock position")
    axes[0].set_xlim(0.6, 0.95)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$\rho$")
    axes[0].set_title("Sod t=0.2, TVB-limited DG")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(centers, exact, "k-", linewidth=1.4, label="exact Riemann")
    for degree in sorted({case["degree"] for case in sod["cases"].values()}):
        case = sod["cases"].get(f"p{degree}_unlimited")
        if case is None or case["density_profile"] is None:
            continue
        axes[1].plot(centers, case["density_profile"], linewidth=1.0,
                     label=f"unlimited p={degree}")
    axes[1].set_xlim(0.6, 0.95)
    axes[1].set_ylim(reference["rho_star_right"] - 0.08,
                     reference["rho_star_right"] + 0.02)
    axes[1].set_xlabel("x")
    axes[1].set_title("negative control: no limiter (shock zoom)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    path = output / "c3_sod_density_profiles.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    figures.append(path)

    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    for limiter, style in (("unlimited", "o-"), ("tvb", "s-")):
        degrees = [degree for degree in sorted({case["degree"]
                                               for case in sod["cases"].values()})
                   if f"p{degree}_{limiter}" in sod["cases"]]
        amplitudes = [sod["cases"][f"p{degree}_{limiter}"]["gibbs_amplitude"]
                      for degree in degrees]
        axis.semilogy(degrees, amplitudes, style, label=limiter)
    axis.set_xlabel("polynomial degree p")
    axis.set_ylabel("Gibbs amplitude / density jump")
    axis.set_title("Sod: non-physical oscillation amplitude")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path = output / "c5_gibbs_amplitude_vs_p.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    figures.append(path)

    figure, axis = plt.subplots(figsize=(6.2, 4.4))
    for ncells, entry in sorted(p_refinement.items(), key=lambda item: int(item[0])):
        axis.semilogy(entry["degrees"], entry["measured_errors"], "o-",
                      label=f"DG N={ncells}")
        axis.semilogy(entry["degrees"], entry["projection_floor"], "k--",
                      linewidth=0.9, alpha=0.6)
    axis.set_xlabel("polynomial degree p")
    axis.set_ylabel("relative $L^2$ error")
    axis.set_title("p-refinement vs analytic $L^2$-projection floor")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = output / "c4_p_refinement.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    figures.append(path)
    return figures


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------

def provenance() -> dict:
    """Environment fingerprint stored with every gate result."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=60).stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=60).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True,
                                    timeout=60).stdout.strip())
    except Exception:  # pragma: no cover - git is optional at runtime
        commit, branch, dirty = "unknown", "unknown", False
    return {
        "git_commit": commit,
        "git_branch": branch,
        "worktree_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float64",
    }


def convergence_table_rows(block: dict) -> list[dict]:
    """Flatten an h-refinement block into CSV rows."""
    rows = []
    for degree, data in sorted(block["degrees"].items()):
        for level in data["levels"]:
            rows.append({
                "degree": degree,
                "ncells": level["ncells"],
                "h": level["h"],
                "relative_l2": level["relative_l2"],
                "pairwise_order": None,
                "observed_order": data["observed_order"],
                "theoretical_order": data["theoretical_order"],
            })
        # attach pairwise orders to the fine levels they belong to
        for offset, order in enumerate(data["pairwise_orders"]):
            rows[-(len(data["levels"]) - offset - 1)]["pairwise_order"] = order
    return rows


def write_csv(path: Path, rows: list[dict], columns: Sequence[str]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def markdown_report(result: dict) -> str:
    gates = result["gates"]
    lines = [
        "# TensorDG verification ladder: advection -> diffusion -> shock -> hp order",
        "",
        "Generated by `python -m benchmarks.verification_ladder`; every number in",
        "this file is rendered from `gates_verified.json` (no hand-typed values).",
        "",
        f"- git commit: `{result['provenance']['git_commit']}` "
        f"(branch `{result['provenance']['git_branch']}`)",
        f"- torch {result['provenance']['torch']}, numpy {result['provenance']['numpy']}, "
        f"python {result['provenance']['python']}",
        f"- wall time: {result['runtime_seconds']:.1f} s",
        f"- overall verdict: **{result['verdict']}**",
        "",
        "## Gate summary",
        "",
        "| gate | criterion | status | failing checks |",
        "|---|---|---|---|",
    ]
    for gate_id in ("C1", "C1b", "C2", "C3", "C4", "C5", "C6"):
        gate = gates.get(gate_id)
        if gate is None:
            continue
        failing = ", ".join(gate["failed_checks"]) if gate["failed_checks"] else "-"
        lines.append(f"| {gate_id} | {gate['name']} | {gate['status']} | {failing} |")

    advection = result["studies"]["advection_1d"]
    lines += ["", "### C1: 1D linear advection", "",
              "| p | errors (N=8,16,32) | pairwise orders | least-squares order | p+1 | rel. error |",
              "|---|---|---|---|---|---|"]
    for degree, data in sorted(advection["degrees"].items()):
        errors = ", ".join(f"{level['relative_l2']:.3e}" for level in data["levels"])
        orders = ", ".join(f"{value:.2f}" for value in data["pairwise_orders"])
        relative = abs(data["observed_order"] - (degree + 1)) / (degree + 1)
        lines.append(f"| {degree} | {errors} | {orders} | {data['observed_order']:.3f} "
                     f"| {degree + 1} | {relative:.2%} |")

    advection_2d = result["studies"].get("advection_2d")
    if advection_2d:
        lines += ["", "### C1b: 2D tensor-product advection", "",
                  "| p | errors (N=8,16,32)^2 | pairwise orders | least-squares order | p+1 | rel. error |",
                  "|---|---|---|---|---|---|"]
        for degree, data in sorted(advection_2d["degrees"].items()):
            errors = ", ".join(f"{level['relative_l2']:.3e}" for level in data["levels"])
            orders = ", ".join(f"{value:.2f}" for value in data["pairwise_orders"])
            relative = abs(data["observed_order"] - (degree + 1)) / (degree + 1)
            lines.append(f"| {degree} | {errors} | {orders} | {data['observed_order']:.3f} "
                         f"| {degree + 1} | {relative:.2%} |")

    heat = result["studies"]["heat_1d"]
    lines += ["", "### C2: 1D heat equation (SIPG + Crank--Nicolson)", "",
              "| p | errors (N=4,8,16) | pairwise orders | least-squares order | p+1 | rel. error |",
              "|---|---|---|---|---|---|"]
    for degree, data in sorted(heat["degrees"].items()):
        errors = ", ".join(f"{level['relative_l2']:.3e}" for level in data["levels"])
        orders = ", ".join(f"{value:.2f}" for value in data["pairwise_orders"])
        relative = abs(data["observed_order"] - (degree + 1)) / (degree + 1)
        lines.append(f"| {degree} | {errors} | {orders} | {data['observed_order']:.3f} "
                     f"| {degree + 1} | {relative:.2%} |")

    sod = result["studies"]["sod_1d"]
    reference = sod["reference"]
    lines += ["", "### C3: Sod shock tube at t = 0.2", "",
              f"Exact reference: p* = {reference['p_star']:.6f}, u* = {reference['u_star']:.6f}, "
              f"shock speed = {reference['shock_speed']:.6f}, "
              f"x_shock = {reference['shock_position']:.6f}.",
              "",
              "| case | shock position | error [cells] | centroid error [cells] "
              "| Gibbs amplitude | TV/TV_exact | density rel. L2 |",
              "|---|---|---|---|---|---|---|"]
    for label, case in sorted(sod["cases"].items()):
        lines.append(
            f"| {label} | {case['shock_position']:.6f} | "
            f"{case['shock_position_error_cells']:.3f} | "
            f"{case['shock_position_centroid_cells']:.3f} | "
            f"{case['gibbs_amplitude']:.3e} | {case['total_variation_ratio']:.4f} | "
            f"{case['density_relative_l2']:.3e} |"
        )

    p_refinement = gates["C4"]["measured"]["p_refinement"]
    lines += ["", "### C4: p-refinement against the analytic projection floor", "",
              "| N | p | e(p) | projection floor | e(p)/e(p+1) | floor(p)/floor(p+1) | rel. error |",
              "|---|---|---|---|---|---|---|"]
    for ncells, entry in sorted(p_refinement.items(), key=lambda item: int(item[0])):
        for position, degree in enumerate(entry["degrees"]):
            ratio = entry["ratios"][position] if position < len(entry["ratios"]) else None
            lines.append(
                f"| {ncells} | {degree} | {entry['measured_errors'][position]:.3e} | "
                f"{entry['projection_floor'][position]:.3e} | "
                f"{ratio['measured_ratio']:.2f} | {ratio['projection_ratio']:.2f} | "
                f"{ratio['relative_error']:.2%} |" if ratio else
                f"| {ncells} | {degree} | {entry['measured_errors'][position]:.3e} | "
                f"{entry['projection_floor'][position]:.3e} | - | - | - |"
            )

    lines += ["", "### Figures", ""]
    for figure in result["figures"]:
        lines.append(f"- `{figure['name']}`: {figure['caption']}")
    return "\n".join(lines) + "\n"


def run_ladder(
    output: Path,
    quick: bool = False,
    with_2d: bool = True,
    sod_ncells: int | None = None,
    sod_degrees: Sequence[int] = (1, 2, 3),
    sod_cfl: float = 0.2,
    keep_profiles: bool = True,
) -> dict:
    """Run every block, evaluate every gate, write JSON/CSV/figures/Markdown."""
    started = time.time()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ncells_sod = sod_ncells if sod_ncells is not None else (100 if quick else 200)
    cfl_sod = 0.3 if quick else sod_cfl
    heat_time_steps = 1200 if quick else 2000

    timings: dict[str, float] = {}

    stamp = time.time()
    advection = advection_ladder()
    timings["advection_1d"] = time.time() - stamp

    stamp = time.time()
    advection_2d = advection_2d_ladder(degrees=(1, 2) if quick else (1, 2, 3))
    timings["advection_2d"] = time.time() - stamp

    stamp = time.time()
    heat = heat_ladder(time_steps=heat_time_steps)
    timings["heat_1d"] = time.time() - stamp

    stamp = time.time()
    sod = sod_ladder(ncells=ncells_sod, degrees=sod_degrees, cfl=cfl_sod,
                     limiter_stage="final")
    timings["sod_1d"] = time.time() - stamp
    if not keep_profiles:
        for case in sod["cases"].values():
            case["density_profile"] = None

    gates: dict[str, dict] = {}
    gates["C1"] = evaluate_c1(advection)
    if with_2d:
        gates["C1b"] = evaluate_c1b(advection_2d)
    gates["C2"] = evaluate_c2(heat)
    gates["C3"] = evaluate_c3(sod)
    gates["C4"] = evaluate_c4(advection, heat, advection_2d if with_2d else None)
    gates["C5"] = evaluate_c5(sod)

    figures = make_figures(advection, heat, advection_2d, sod,
                           gates["C4"]["measured"]["p_refinement"], output)

    test_path = Path(__file__).resolve().parent.parent / "tests" / \
        "test_verification_ladder.py"
    artifact = output / "gates_verified.json"
    # The artifact is written before C6 is evaluated (the gate checks that the
    # file exists on disk), then rewritten with the C6 record included.
    gates["C6"] = _gate("C6", "pending", "pending", [])
    verdict = "PASS" if all(gate["status"] == "PASS" for gate in gates.values()) else "FAIL"
    result = {
        "schema": "tensordg-verification-ladder/1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": verdict,
        "provenance": provenance(),
        "config": {
            "quick": quick,
            "sod_ncells": ncells_sod,
            "sod_cfl": cfl_sod,
            "sod_degrees": list(sod_degrees),
            "heat_time_steps": heat_time_steps,
            "thresholds": {
                "order_relative_tolerance": ORDER_RELATIVE_TOLERANCE,
                "shock_position_tolerance_cells": SHOCK_POSITION_TOLERANCE_CELLS,
                "gibbs_monotonicity_tolerance": GIBBS_MONOTONICITY_TOLERANCE,
                "gibbs_negative_control_floor": GIBBS_NEGATIVE_CONTROL_FLOOR,
            },
        },
        "studies": {
            "advection_1d": advection,
            "advection_2d": advection_2d,
            "heat_1d": heat,
            "sod_1d": sod,
        },
        "gates": gates,
        "figures": [
            {"name": path.name,
             "caption": {
                 "c1_advection_1d_convergence.png":
                     "1D advection: relative L2 error vs h with p+1 reference slopes",
                 "c1_advection_2d_convergence.png":
                     "2D tensor-product advection: error vs h with p+1 reference slopes",
                 "c2_heat_1d_convergence.png":
                     "Heat equation: relative L2 error vs h with p+1 reference slopes",
                 "c3_sod_density_profiles.png":
                     "Sod density at t=0.2: TVB-limited (left) and unlimited (right) "
                     "against the exact Riemann solution",
                 "c5_gibbs_amplitude_vs_p.png":
                     "Gibbs amplitude (fraction of the density jump) vs polynomial degree",
                 "c4_p_refinement.png":
                     "p-refinement at fixed h against the analytic L2-projection floor",
             }.get(path.name, path.name),
             "path": str(path)}
            for path in figures
        ],
        "timings_seconds": timings,
        "runtime_seconds": time.time() - started,
    }
    artifact.write_text(json.dumps(result, indent=1))
    gates["C6"] = evaluate_c6(gates, test_path, artifact)
    result["verdict"] = (
        "PASS" if all(gate["status"] == "PASS" for gate in gates.values()) else "FAIL"
    )
    artifact.write_text(json.dumps(result, indent=1))
    write_csv(output / "advection_1d_convergence.csv",
              convergence_table_rows(advection),
              ("degree", "ncells", "h", "relative_l2", "pairwise_order",
               "observed_order", "theoretical_order"))
    write_csv(output / "advection_2d_convergence.csv",
              convergence_table_rows(advection_2d),
              ("degree", "ncells", "h", "relative_l2", "pairwise_order",
               "observed_order", "theoretical_order"))
    write_csv(output / "heat_1d_convergence.csv",
              convergence_table_rows(heat),
              ("degree", "ncells", "h", "relative_l2", "pairwise_order",
               "observed_order", "theoretical_order"))
    sod_rows = []
    for label, case in sorted(sod["cases"].items()):
        sod_rows.append({
            "case": label,
            "degree": case["degree"],
            "limiter": case["limiter"],
            "ncells": case["ncells"],
            "shock_position": case["shock_position"],
            "shock_position_exact": case["shock_position_exact"],
            "shock_position_error_cells": case["shock_position_error_cells"],
            "shock_position_centroid_cells":
                case["shock_position_centroid_cells"],
            "gibbs_amplitude": case["gibbs_amplitude"],
            "total_variation_ratio": case["total_variation_ratio"],
            "density_relative_l2": case["density_relative_l2"],
            "steps": case["steps"],
        })
    write_csv(output / "sod_diagnostics.csv", sod_rows,
              ("case", "degree", "limiter", "ncells", "shock_position",
               "shock_position_exact", "shock_position_error_cells",
               "shock_position_centroid_cells", "gibbs_amplitude",
               "total_variation_ratio", "density_relative_l2", "steps"))
    (output / "summary.md").write_text(markdown_report(result))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/benchmarks/results/verification_ladder",
                        help="directory for gates_verified.json, CSV, PNG and summary.md")
    parser.add_argument("--quick", action="store_true",
                        help="reduced cost (coarser Sod mesh, fewer steps)")
    parser.add_argument("--no-2d", action="store_true", help="skip the 2D advection block")
    parser.add_argument("--sod-ncells", type=int, default=None)
    parser.add_argument("--sod-degrees", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sod-cfl", type=float, default=0.2)
    args = parser.parse_args(argv)

    result = run_ladder(
        output=Path(args.output),
        quick=args.quick,
        with_2d=not args.no_2d,
        sod_ncells=args.sod_ncells,
        sod_degrees=tuple(args.sod_degrees),
        sod_cfl=args.sod_cfl,
    )
    for gate_id in ("C1", "C1b", "C2", "C3", "C4", "C5", "C6"):
        gate = result["gates"].get(gate_id)
        if gate is None:
            continue
        print(f"{gate_id:>4} {gate['status']:<4} {gate['name']}")
        for check in gate["checks"]:
            if check["status"] != "PASS":
                print(f"       FAIL {check['name']}: {check}")
    print(f"verdict: {result['verdict']}  ({result['runtime_seconds']:.1f} s)")
    print(f"artifact: {Path(args.output) / 'gates_verified.json'}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
