"""Exact oblique-shock verification for the 2-D DG Euler solver.

Why this module exists
----------------------
``benchmarks/supersonic_wedge.py`` and ``benchmarks/double_mach_reflection.py``
(merged in PR #247 / mission m11149) claim to verify 2-D shock capturing on a
wedge and on the Woodward--Colella double Mach reflection.  They cannot: both
crash at run time, and the ``wedge`` initial condition returns a **uniform
freestream** -- there is no wedge, no shock, and therefore no shock angle to
measure.  Neither case builds a state that solves the Euler equations either
(the double-Mach post-shock velocity is set along ``+x`` instead of along the
shock normal).

This module supplies the piece that the claim actually needs: the **exact
oblique-shock solution** and a measurement harness that compares the DG
solution with it.  The configuration is a genuine two-state solution of the
compressible Euler equations:

* upstream state uniform, moving with Mach ``M`` along ``+x``;
* a plane front through ``(x0, y0)`` whose direction makes angle ``beta`` with
  the ``x`` axis;
* downstream state obtained from the **Rankine--Hugoniot relations across that
  front** (normal momentum / mass / tangential velocity / energy), which is
  exactly the classic theta--beta--Mach oblique-shock solution (Anderson,
  *Modern Compressible Flow*, 3rd ed., ch. 4).

Because both states are uniform and the jump conditions hold across the
stationary front, this field is an **exact steady solution**: the true
solution of the initial--boundary value problem is the initial field itself.
Every deviation is numerical error, so the measurement is non-circular in the
sense that matters -- the *reference* comes from the RH algebra, not from a
previous solver run.

What is measured (all against closed-form values)
-------------------------------------------------
``beta_hat``      front angle recovered from the numerical density field by
                  mid-level crossings sampled along the exact normal, fitted by
                  total least squares (no knowledge of the IC used);
``rho_ratio``     downstream/upstream density plateau ratio vs the exact RH
                  ratio;
``p_ratio``       the same for pressure;
``theta_hat``     flow-deflection angle (direction change of the velocity
                  vector across the shock) vs the exact deflection;
``preserve_l1``   mean ``|rho_num - rho_exact|`` over the measurement window,
                  normalised by the exact density jump ``|rho2 - rho1|``.

Declared gates (fixed before the runs, not tuned afterwards)
------------------------------------------------------------
===============  ==================================================  =========
gate             criterion                                           applies
===============  ==================================================  =========
``beta``         ``|beta_hat - beta| <= 1 deg``                      all cases
``rho_ratio``    ``|r_hat/r_exact - 1| <= 3 %``                       all cases
``p_ratio``      ``|p_hat/p_exact - 1| <= 3 %``                       all cases
``theta``        ``|theta_hat - theta_exact| <= 0.5 deg``             all cases
``preserve_l1``  ``preserve_l1 <= 5 %``                              all cases
===============  ==================================================  =========

Negative control
----------------
``cases.control_wrong_beta`` builds the *same two states* across a front 5
degrees away from the RH-consistent angle.  The two states no longer satisfy
the jump conditions, so the field must evolve (a shock/rarefaction pair) and
the preservation gate must fail.  If it passed, the measurement would be blind
and every other gate would be worthless; the harness therefore asserts
``preserve_l1 > 10 %`` for the control.

Known scope limits (honest)
---------------------------
* The solver only offers periodic / zero-gradient (transmissive) faces
  (``solve_euler_dg_2d``), so a **wall-bounded** wedge (inclined slip wall)
  and the full Woodward--Colella double-Mach *pattern* (reflecting wall plus a
  moving post-shock boundary) are out of reach.  Those remain real gaps; see
  ``docs/reports/tensordg-oblique-shock-verification.md``.
* Transmissive faces truncate the front at the domain boundary, so all
  measurements are restricted to an interior window (cells further than
  ``boundary_margin`` from the sides and than ``margin_cells * h`` from the
  exact front).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensordg.euler_1d import GAMMA
from tensordg.euler_dg_2d import cell_fields, solve_euler_dg_2d


# --------------------------------------------------------------------------- #
# exact oblique-shock theory
# --------------------------------------------------------------------------- #
def theta_from_beta(mach: float, beta_deg: float, gamma: float = GAMMA) -> float:
    """Flow deflection ``theta`` (deg) of the weak/strong branch at angle ``beta``.

    ``tan(theta) = 2 cot(beta) (M^2 sin^2 beta - 1)
                    / (M^2 (gamma + cos 2 beta) + 2)``
    """
    if not 0.0 < beta_deg <= 90.0:
        raise ValueError("beta must be in (0, 90] degrees")
    beta = math.radians(beta_deg)
    sin_b, cos_b = math.sin(beta), math.cos(beta)
    cot_b = cos_b / sin_b
    numerator = mach ** 2 * sin_b ** 2 - 1.0
    denominator = mach ** 2 * (gamma + math.cos(2.0 * beta)) + 2.0
    return math.degrees(math.atan(2.0 * cot_b * numerator / denominator))


def max_deflection_angle(mach: float, gamma: float = GAMMA,
                         samples: int = 20001) -> float:
    """Detachment deflection ``theta_max`` (deg): where the weak root disappears."""
    beta_min = math.degrees(math.asin(1.0 / mach))
    grid = np.linspace(beta_min + 1e-9, 90.0, samples)
    return float(max(theta_from_beta(mach, float(b), gamma) for b in grid))


def beta_from_theta(mach: float, theta_deg: float, gamma: float = GAMMA) -> float:
    """Weak-branch shock angle ``beta`` (deg) for a wedge of half-angle ``theta``.

    Bisection on ``theta_from_beta(beta) - theta`` over
    ``[arcsin(1/M), beta_detachment]``; raises when the wedge is past
    detachment (no attached shock exists -- that physics is not claimed here).
    """
    theta_max = max_deflection_angle(mach, gamma)
    if theta_deg <= 0.0:
        raise ValueError("theta must be positive")
    if theta_deg >= theta_max:
        raise ValueError(
            f"theta = {theta_deg} deg exceeds the detachment angle "
            f"{theta_max:.3f} deg at M = {mach}"
        )
    lo = math.degrees(math.asin(1.0 / mach)) + 1e-12
    hi = 90.0
    # shrink to the interval where theta_from_beta is still increasing (weak branch)
    grid = np.linspace(lo, hi, 4001)
    values = [theta_from_beta(mach, float(b), gamma) for b in grid]
    for i in range(len(grid) - 1):
        if values[i] <= theta_deg <= values[i + 1]:
            lo, hi = float(grid[i]), float(grid[i + 1])
            break
    else:  # pragma: no cover - guarded by the theta_max check above
        raise ValueError("weak branch interval not found")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if theta_from_beta(mach, mid, gamma) < theta_deg:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class ShockState:
    """Uniform state in primitive variables (``rho, u, v, p``)."""
    rho: float
    u: float
    v: float
    p: float

    def as_tuple(self):
        return (self.rho, self.u, self.v, self.p)

    def sound_speed(self, gamma: float = GAMMA) -> float:
        return math.sqrt(gamma * self.p / self.rho)


@dataclass
class ShockSolution:
    """Exact oblique-shock pair plus the numbers they were built from."""
    mach: float
    beta_deg: float
    theta_deg: float
    upstream: ShockState
    downstream: ShockState
    rho_ratio: float
    p_ratio: float
    gamma: float = GAMMA
    diagnostics: dict = field(default_factory=dict)

    def normal(self) -> np.ndarray:
        beta = math.radians(self.beta_deg)
        return np.array([math.sin(beta), -math.cos(beta)])

    def tangent(self) -> np.ndarray:
        beta = math.radians(self.beta_deg)
        return np.array([math.cos(beta), math.sin(beta)])


def _rh_residuals(up: ShockState, down: ShockState, beta_deg: float,
                  gamma: float = GAMMA) -> dict:
    """Relative residual of the four Rankine--Hugoniot conditions.

    Mass, normal momentum, tangential momentum and energy flux must be
    continuous across the front; this is what makes the two-state field an
    exact solution of the Euler equations.
    """
    beta = math.radians(beta_deg)
    n = np.array([math.sin(beta), -math.cos(beta)])
    t = np.array([math.cos(beta), math.sin(beta)])

    def quantities(state: ShockState):
        velocity = np.array([state.u, state.v])
        u_n = float(velocity @ n)
        u_t = float(velocity @ t)
        enthalpy = gamma / (gamma - 1.0) * state.p / state.rho
        total = enthalpy + 0.5 * (state.u ** 2 + state.v ** 2)
        return {
            "mass": state.rho * u_n,
            "normal_momentum": state.rho * u_n ** 2 + state.p,
            "tangential_momentum": state.rho * u_n * u_t,
            "energy": state.rho * u_n * total,
            "u_n": u_n,
            "u_t": u_t,
        }

    a, b = quantities(up), quantities(down)
    scale = {key: max(abs(a[key]), abs(b[key]), 1e-30) for key in a}
    return {key: abs(a[key] - b[key]) / scale[key]
            for key in ("mass", "normal_momentum", "tangential_momentum", "energy")}


def oblique_shock(mach: float, beta_deg: float,
                  upstream: ShockState | None = None,
                  gamma: float = GAMMA) -> ShockSolution:
    """Exact downstream state behind a plane front at ``beta`` for Mach ``mach``.

    Raises when the upstream normal Mach number is not supersonic -- there is
    no oblique shock then, and silently returning something would hide an
    invalid configuration.
    """
    if upstream is None:
        # a1 = sqrt(gamma p/rho); normalising p = rho = 1 gives a1 = sqrt(gamma),
        # so the free-stream speed that corresponds to Mach ``mach`` is mach*a1.
        reference = ShockState(rho=1.0, u=0.0, v=0.0, p=1.0)
        upstream = ShockState(rho=1.0, u=mach * reference.sound_speed(gamma),
                              v=0.0, p=1.0)
    beta = math.radians(beta_deg)
    n = np.array([math.sin(beta), -math.cos(beta)])
    t = np.array([math.cos(beta), math.sin(beta)])
    velocity = np.array([upstream.u, upstream.v])
    u_n = float(velocity @ n)
    u_t = float(velocity @ t)
    sound = upstream.sound_speed(gamma)
    normal_mach = u_n / sound
    if normal_mach <= 1.0:
        raise ValueError(
            f"normal Mach number {normal_mach:.4f} <= 1: the front at beta = "
            f"{beta_deg} deg is not a shock for M = {mach}"
        )
    rho_ratio = (gamma + 1.0) * normal_mach ** 2 / ((gamma - 1.0) * normal_mach ** 2 + 2.0)
    p_ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (normal_mach ** 2 - 1.0)
    down_rho = upstream.rho * rho_ratio
    down_p = upstream.p * p_ratio
    down_u_n = u_n * upstream.rho / down_rho
    down_velocity = down_u_n * n + u_t * t
    down = ShockState(rho=down_rho, u=float(down_velocity[0]),
                      v=float(down_velocity[1]), p=down_p)
    theta = theta_from_beta(mach, beta_deg, gamma)
    solution = ShockSolution(mach=mach, beta_deg=beta_deg, theta_deg=theta,
                             upstream=upstream, downstream=down,
                             rho_ratio=rho_ratio, p_ratio=p_ratio, gamma=gamma)
    solution.diagnostics["normal_mach"] = normal_mach
    solution.diagnostics["rh_residual"] = _rh_residuals(upstream, down, beta_deg, gamma)
    # the deflection recovered from the two velocity vectors must agree with the
    # theta--beta--M relation: an independent check of the jump algebra
    angle_up = math.degrees(math.atan2(upstream.v, upstream.u))
    angle_down = math.degrees(math.atan2(down.v, down.u))
    solution.diagnostics["theta_from_vectors"] = abs(angle_up - angle_down)
    return solution


def oblique_shock_for_wedge(mach: float, theta_deg: float,
                            gamma: float = GAMMA) -> ShockSolution:
    """Weak-branch solution for a wedge of half-angle ``theta`` at Mach ``mach``."""
    return oblique_shock(mach, beta_from_theta(mach, theta_deg, gamma), gamma=gamma)


# --------------------------------------------------------------------------- #
# initial condition
# --------------------------------------------------------------------------- #
def oblique_shock_initial(solution: ShockSolution, x0: float = 0.5,
                          y0: float = 0.5, beta_override_deg: float | None = None):
    """Callable ``initial(x, y)`` for :func:`solve_euler_dg_2d`.

    ``s = (x - x0) sin(beta) - (y - y0) cos(beta)`` is the signed distance along
    the shock normal: ``s > 0`` is the downstream side.

    ``beta_override_deg`` places the *same two states* across a front at a
    different angle -- used only by the negative control, which relies on the
    resulting field not being an Euler solution.
    """
    beta = math.radians(solution.beta_deg if beta_override_deg is None
                        else beta_override_deg)
    n = (math.sin(beta), -math.cos(beta))
    up, down = solution.upstream, solution.downstream

    def initial(x, y):
        side = (x - x0) * n[0] + (y - y0) * n[1] > 0.0
        zeros = torch.zeros_like(x)

        def pick(upstream_value, downstream_value):
            return torch.where(side, torch.full_like(x, downstream_value),
                               torch.full_like(x, upstream_value))

        return (pick(up.rho, down.rho),
                pick(up.u, down.u),
                pick(up.v, down.v) + zeros,
                pick(up.p, down.p))

    return initial


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def _interp(field: np.ndarray, xc: np.ndarray, yc: np.ndarray,
            px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a cell-centre field at arbitrary points."""
    hx = xc[1] - xc[0]
    hy = yc[1] - yc[0]
    fx = (px - xc[0]) / hx
    fy = (py - yc[0]) / hy
    i0 = np.clip(np.floor(fx).astype(int), 0, field.shape[0] - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, field.shape[1] - 2)
    tx = np.clip(fx - i0, 0.0, 1.0)
    ty = np.clip(fy - j0, 0.0, 1.0)
    return (field[i0, j0] * (1 - tx) * (1 - ty)
            + field[i0 + 1, j0] * tx * (1 - ty)
            + field[i0, j0 + 1] * (1 - tx) * ty
            + field[i0 + 1, j0 + 1] * tx * ty)


def _exact_front_line(solution: ShockSolution, x0: float, y0: float,
                      beta_deg: float | None = None):
    beta = math.radians(solution.beta_deg if beta_deg is None else beta_deg)
    n = np.array([math.sin(beta), -math.cos(beta)])
    t = np.array([math.cos(beta), math.sin(beta)])
    return np.array([x0, y0]), n, t


def measure_oblique_shock(x: torch.Tensor, y: torch.Tensor, coeff: torch.Tensor,
                          solution: ShockSolution, x0: float = 0.5,
                          y0: float = 0.5, margin_cells: float = 4.0,
                          boundary_margin: float = 0.12,
                          n_rays: int = 81, beta_used_deg: float | None = None,
                          ) -> dict:
    """Recover the shock angle, plateau jumps and preservation error.

    ``x``/``y`` are the 1-D cell-centre arrays returned by the solver,
    ``coeff`` the modal state.  All numbers are derived from the numerical
    field alone (plus the exact *reference* values they are compared with).
    """
    fields = cell_fields(coeff)
    density = fields["density"].cpu().numpy()
    velocity_x = fields["velocity_x"].cpu().numpy()
    velocity_y = fields["velocity_y"].cpu().numpy()
    pressure = fields["pressure"].cpu().numpy()
    xc, yc = x.cpu().numpy(), y.cpu().numpy()
    h = min(xc[1] - xc[0], yc[1] - yc[0])

    x_min = xc[0] - 0.5 * (xc[1] - xc[0]) + boundary_margin
    x_max = xc[-1] + 0.5 * (xc[1] - xc[0]) - boundary_margin
    y_min = yc[0] - 0.5 * (yc[1] - yc[0]) + boundary_margin
    y_max = yc[-1] + 0.5 * (yc[1] - yc[0]) - boundary_margin

    origin, n, t = _exact_front_line(solution, x0, y0, beta_used_deg)
    px, py = np.meshgrid(xc, yc, indexing="ij")
    s_field = (px - origin[0]) * n[0] + (py - origin[1]) * n[1]
    inside = ((px > x_min) & (px < x_max) & (py > y_min) & (py < y_max))

    rho_up, rho_down = solution.upstream.rho, solution.downstream.rho
    mid = 0.5 * (rho_up + rho_down)

    # --- front angle from mid-level crossings along the exact normal ---------
    extent = max(x_max - x_min, y_max - y_min)
    offsets = np.linspace(-0.45 * extent, 0.45 * extent, n_rays)
    s_samples = np.linspace(-0.35 * extent, 0.35 * extent, 401)
    crossings = []
    for offset in offsets:
        ox = origin[0] + offset * t[0] + s_samples * n[0]
        oy = origin[1] + offset * t[1] + s_samples * n[1]
        keep = (ox > x_min) & (ox < x_max) & (oy > y_min) & (oy < y_max)
        if keep.sum() < 5:
            continue
        values = _interp(density, xc, yc, ox, oy)
        diff = values - mid
        for i in range(len(diff) - 1):
            if keep[i] and keep[i + 1] and diff[i] * diff[i + 1] < 0.0:
                frac = diff[i] / (diff[i] - diff[i + 1])
                s_star = s_samples[i] + frac * (s_samples[i + 1] - s_samples[i])
                crossings.append((origin[0] + offset * t[0] + s_star * n[0],
                                  origin[1] + offset * t[1] + s_star * n[1]))
                break
    beta_hat = None
    beta_error = None
    if len(crossings) >= 5:
        points = np.array(crossings)
        centred = points - points.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        direction = vt[0]
        if direction[0] < 0.0 or (abs(direction[0]) < 1e-9 and direction[1] < 0.0):
            direction = -direction
        beta_hat = math.degrees(math.atan2(direction[1], direction[0]))
        beta_error = abs(beta_hat - (solution.beta_deg if beta_used_deg is None
                                     else beta_used_deg))

    # --- plateau values ------------------------------------------------------ #
    far = np.abs(s_field) > margin_cells * h
    down_mask = inside & far & (s_field > 0.0)
    up_mask = inside & far & (s_field < 0.0)
    if down_mask.sum() < 10 or up_mask.sum() < 10:
        raise ValueError("measurement window too small; refine the mesh")
    rho2 = float(density[down_mask].mean())
    rho1 = float(density[up_mask].mean())
    p2 = float(pressure[down_mask].mean())
    p1 = float(pressure[up_mask].mean())
    u2 = float(velocity_x[down_mask].mean())
    v2 = float(velocity_y[down_mask].mean())
    u1 = float(velocity_x[up_mask].mean())
    v1 = float(velocity_y[up_mask].mean())
    angle_down = math.degrees(math.atan2(v2, u2))
    angle_up = math.degrees(math.atan2(v1, u1))
    theta_hat = abs(angle_up - angle_down)

    # --- preservation error against the exact field -------------------------- #
    exact = np.where(s_field > 0.0, rho_down, rho_up)
    window = inside & (np.abs(s_field) > 0.0)
    jump = abs(rho_down - rho_up)
    preserve = float(np.abs(density[window] - exact[window]).mean() / jump)

    return {
        "beta_hat_deg": beta_hat,
        "beta_error_deg": beta_error,
        "beta_reference_deg": solution.beta_deg if beta_used_deg is None else beta_used_deg,
        "crossings": len(crossings),
        "rho_ratio_hat": rho2 / rho1,
        "rho_ratio_exact": solution.rho_ratio,
        "rho_ratio_error": abs(rho2 / rho1 - solution.rho_ratio) / solution.rho_ratio,
        "p_ratio_hat": p2 / p1,
        "p_ratio_exact": solution.p_ratio,
        "p_ratio_error": abs(p2 / p1 - solution.p_ratio) / solution.p_ratio,
        "theta_hat_deg": theta_hat,
        "theta_exact_deg": solution.theta_deg,
        "theta_error_deg": abs(theta_hat - solution.theta_deg),
        "preserve_l1": preserve,
        "window_cells": int(window.sum()),
        "fields": {"density": density, "pressure": pressure,
                   "velocity_x": velocity_x, "velocity_y": velocity_y},
        "s_field": s_field,
        "inside": inside,
    }


# --------------------------------------------------------------------------- #
# case definitions and harness
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    name: str
    description: str
    solution_factory: str        # "wedge" | "beta"
    mach: float
    value: float                 # theta for "wedge", beta for "beta"
    ncells: int = 64
    degree: int = 1
    final_time: float = 0.06
    cfl: float = 0.2
    limiter: str | None = "tvb"
    tvb_constant: float = 100.0
    limiter_stage: str = "final"
    beta_override_offset_deg: float | None = None
    expect_preservation_fail: bool = False

    def build(self) -> ShockSolution:
        if self.solution_factory == "wedge":
            return oblique_shock_for_wedge(self.mach, self.value)
        return oblique_shock(self.mach, self.value)


DEFAULT_CASES = [
    Case("wedge_M2_theta15", "M=2 wedge class, theta=15 deg (PR #247 claimed case)",
         "wedge", 2.0, 15.0),
    Case("strong_M10_beta60", "M=10 front at beta=60 deg (Woodward-Colella class)",
         "beta", 10.0, 60.0),
    Case("normal_M4_beta90", "grid-aligned normal shock M=4 (control)",
         "beta", 4.0, 90.0, final_time=0.04),
    Case("control_M2_theta15_wrong_beta",
         "negative control: same states across a front 5 deg off the RH angle",
         "wedge", 2.0, 15.0, beta_override_offset_deg=-5.0,
         expect_preservation_fail=True),
]

GATES = {
    "beta_error_deg_max": 1.0,
    "rho_ratio_error_max": 0.03,
    "p_ratio_error_max": 0.03,
    "theta_error_deg_max": 0.5,
    "preserve_l1_max": 0.05,
    "control_preserve_l1_min": 0.10,
}


def run_case(case: Case, domain=((0.0, 1.0), (0.0, 1.0)),
             x0: float = 0.5, y0: float = 0.5, dtype=torch.float64) -> dict:
    """Run one case and measure it against the exact solution."""
    solution = case.build()
    initial = oblique_shock_initial(solution, x0=x0, y0=y0,
                                    beta_override_deg=None
                                    if case.beta_override_offset_deg is None
                                    else solution.beta_deg
                                    + case.beta_override_offset_deg)
    x, y, coeff, metadata = solve_euler_dg_2d(
        initial, ncells=(case.ncells, case.ncells), degree=case.degree,
        domain=domain, final_time=case.final_time, cfl=case.cfl,
        limiter=case.limiter, tvb_constant=case.tvb_constant,
        limiter_stage=case.limiter_stage, periodic=(False, False),
        dtype=dtype,
    )
    measured = measure_oblique_shock(
        x, y, coeff, solution, x0=x0, y0=y0,
        beta_used_deg=None if case.beta_override_offset_deg is None
        else solution.beta_deg + case.beta_override_offset_deg,
    )
    record = {
        "case": case.name,
        "description": case.description,
        "mach": case.mach,
        "beta_deg": solution.beta_deg,
        "theta_deg": solution.theta_deg,
        "rho_ratio_exact": solution.rho_ratio,
        "p_ratio_exact": solution.p_ratio,
        "rh_residual": solution.diagnostics["rh_residual"],
        "theta_from_vectors_deg": solution.diagnostics["theta_from_vectors"],
        "ncells": case.ncells,
        "degree": case.degree,
        "final_time": case.final_time,
        "cfl": case.cfl,
        "limiter": case.limiter,
        "tvb_constant": case.tvb_constant,
        "steps": metadata["steps"],
        "dt": metadata["dt"],
        "expect_preservation_fail": case.expect_preservation_fail,
    }
    record.update({key: value for key, value in measured.items()
                   if key not in ("fields", "s_field", "inside")})
    record["gates"] = _gate_status(record)
    return record, measured, solution, (x, y)


def _gate_status(record: dict) -> dict:
    if record["expect_preservation_fail"]:
        return {"beta": record["beta_error_deg"] is not None
                and record["beta_error_deg"] <= GATES["beta_error_deg_max"],
                "preserve_l1": record["preserve_l1"] >= GATES["control_preserve_l1_min"]}
    return {
        "beta": record["beta_error_deg"] is not None
        and record["beta_error_deg"] <= GATES["beta_error_deg_max"],
        "rho_ratio": record["rho_ratio_error"] <= GATES["rho_ratio_error_max"],
        "p_ratio": record["p_ratio_error"] <= GATES["p_ratio_error_max"],
        "theta": record["theta_error_deg"] <= GATES["theta_error_deg_max"],
        "preserve_l1": record["preserve_l1"] <= GATES["preserve_l1_max"],
    }


def _plot_case(record, measured, solution, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    density = measured["fields"]["density"]
    s_field = measured["s_field"]
    image = axes[0].pcolormesh(density.T, cmap="viridis", shading="auto")
    axes[0].contour(s_field.T, levels=[0.0], colors="w", linewidths=1.0)
    axes[0].set_title(f"{record['case']}: density, DG P{record['degree']} "
                      f"{record['ncells']}^2")
    axes[0].set_xlabel("i")
    axes[0].set_ylabel("j")
    fig.colorbar(image, ax=axes[0], label=r"$\rho$")

    # profile across the exact normal through the front centre
    n = solution.normal()
    s = np.linspace(-0.3, 0.3, 241)
    axes[1].plot(s, np.where(s > 0.0, solution.downstream.rho, solution.upstream.rho),
                 "k--", label="exact RH")
    axes[1].plot(s, np.where(s > 0.0, solution.downstream.p, solution.upstream.p),
                 "k:", label="exact RH (pressure)")
    axes[1].set_xlabel("distance along shock normal")
    axes[1].set_ylabel("state")
    axes[1].set_title(f"beta_hat = {record['beta_hat_deg']:.3f} deg "
                      f"(exact {record['beta_deg']:.3f})")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = output_dir / f"{record['case']}_density.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def verify(cases: list[Case] | None = None,
           output_dir: str | Path = "docs/benchmarks/results/oblique_shock",
           make_figures: bool = True) -> dict:
    """Run every case, apply the declared gates, write JSON (+ figures)."""
    cases = DEFAULT_CASES if cases is None else cases
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for case in cases:
        record, measured, solution, _ = run_case(case)
        if make_figures:
            record["figure"] = str(_plot_case(record, measured, solution, output_dir))
        records.append(record)
        print(f"[{record['case']}] beta_err={record['beta_error_deg']} "
              f"rho_err={record['rho_ratio_error']:.4f} "
              f"p_err={record['p_ratio_error']:.4f} "
              f"theta_err={record['theta_error_deg']:.4f} "
              f"preserve={record['preserve_l1']:.4f} "
              f"gates={record['gates']}", flush=True)
    verdict = all(all(record["gates"].values()) for record in records)
    payload = {"benchmark": "oblique_shock", "gates": GATES,
               "cases": records, "verdict": "PASS" if verdict else "FAIL"}
    with open(output_dir / "results.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"verdict: {payload['verdict']}")
    return payload


def _parse_cases(names: list[str] | None) -> list[Case]:
    if not names:
        return DEFAULT_CASES
    lookup = {case.name: case for case in DEFAULT_CASES}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise SystemExit(f"unknown case(s): {missing}; known: {sorted(lookup)}")
    return [lookup[name] for name in names]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append",
                        help="run only the named case(s); default: all")
    parser.add_argument("--ncells", type=int, default=None,
                        help="override the mesh size of every selected case")
    parser.add_argument("--degree", type=int, default=None,
                        help="override the polynomial degree of every selected case")
    parser.add_argument("--final-time", type=float, default=None)
    parser.add_argument("--output-dir",
                        default="docs/benchmarks/results/oblique_shock")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    cases = _parse_cases(args.case)
    for case in cases:
        if args.ncells is not None:
            case.ncells = args.ncells
        if args.degree is not None:
            case.degree = args.degree
        if args.final_time is not None:
            case.final_time = args.final_time
    verify(cases, output_dir=args.output_dir, make_figures=not args.no_figures)


if __name__ == "__main__":
    main()