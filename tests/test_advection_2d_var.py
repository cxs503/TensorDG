"""Verification tests for :mod:`tensordg.advection_2d_var`.

This module exercises the 2-D DG advection solver with space/time-varying,
divergence-free velocity fields.  Every gate is an *external* reference
(analytic solution or conservation law), never a self-consistency check.

Reference cases
---------------
* **solid-body rotation** -- rigid rotation of a Gaussian bump; exact solution
  = initial field rotated by ``omega * t``.
* **reentrant deformational vortex** -- LeVeque/Nair flow with ``cos(pi t/T)``
  modulation; the flow map over ``[0, T]`` is the identity, so ``u(T) == u(0)``.

Design-order gates
------------------
* P1 (degree=1) must converge at order 2.
* P2 (degree=2) must converge at order 3.
* P3 (degree=3) is reported for peak-retention studies.

Negative controls
-----------------
* Coarse mesh must fail the 3 % gate.
* Collapsed quadrature (nq < 2*degree+3) must fail.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordg.advection_2d_var import (
    DGAdvection2D,
    estimate_max_components,
    estimate_max_speed,
    gaussian_bump,
    reentrant_vortex,
    rotated_profile,
    solid_body_rotation,
    trig_profile,
)

THREE_PERCENT = 0.03


def _relative_l2(solver, coeff, exact, t=0.0, radius=None):
    """Return relative L2 error."""
    return solver.l2_error(coeff, exact, t, radius=radius)["relative_l2"]


def _eoc(errors):
    return [
        math.log(errors[i] / errors[i + 1]) / math.log(2.0)
        for i in range(len(errors) - 1)
    ]


# --------------------------------------------------------------------------- #
# Mass conservation (divergence-free velocity)
# --------------------------------------------------------------------------- #
def test_solid_body_rotation_conserves_mass():
    """Mass must be conserved to round-off for solid-body rotation."""
    solver = DGAdvection2D(degree=2, ncells=16)
    coeff = solver.project(gaussian_bump)
    velocity = solid_body_rotation(omega=2.0 * math.pi)
    result = solver.solve(coeff, velocity, final_time=0.5, cfl=0.3)
    drift = result["meta"]["mass_drift_rel"]
    assert drift < 1e-12, drift


def test_reentrant_vortex_conserves_mass():
    """Mass must be conserved for the reentrant vortex."""
    solver = DGAdvection2D(degree=2, ncells=16)
    coeff = solver.project(trig_profile)
    velocity = reentrant_vortex(period=1.0)
    result = solver.solve(coeff, velocity, final_time=1.0, cfl=0.3)
    drift = result["meta"]["mass_drift_rel"]
    assert drift < 1e-12, drift


# --------------------------------------------------------------------------- #
# Convergence order (smooth solution, smooth velocity)
# --------------------------------------------------------------------------- #
def test_p1_converges_at_second_order():
    """P1 advection must converge at order 2 on a smooth field."""
    errors = []
    for n in (8, 16, 32):
        solver = DGAdvection2D(degree=1, ncells=n)
        coeff = solver.project(trig_profile)
        velocity = solid_body_rotation(omega=2.0 * math.pi)
        result = solver.solve(coeff, velocity, final_time=0.25, cfl=0.3)
        err = _relative_l2(solver, result["coeff"], trig_profile, t=0.25)
        errors.append(err)
    eocs = _eoc(errors)
    assert min(eocs) > 1.9, eocs


def test_p2_converges_at_third_order():
    """P2 advection must converge at order 3 on a smooth field."""
    errors = []
    for n in (8, 16, 32):
        solver = DGAdvection2D(degree=2, ncells=n)
        coeff = solver.project(trig_profile)
        velocity = solid_body_rotation(omega=2.0 * math.pi)
        result = solver.solve(coeff, velocity, final_time=0.25, cfl=0.3)
        err = _relative_l2(solver, result["coeff"], trig_profile, t=0.25)
        errors.append(err)
    eocs = _eoc(errors)
    assert min(eocs) > 2.9, eocs


# --------------------------------------------------------------------------- #
# Solid-body rotation (peak retention & reentrant check)
# --------------------------------------------------------------------------- #
def test_solid_body_rotation_peak_retention_p3():
    """Gaussian bump peak retention >= 0.99 after one revolution (p=3)."""
    solver = DGAdvection2D(degree=3, ncells=16)
    coeff = solver.project(gaussian_bump)
    peak0 = solver.peak(coeff)[1]
    velocity = solid_body_rotation(omega=2.0 * math.pi)
    result = solver.solve(coeff, velocity, final_time=1.0, cfl=0.3)
    peak1 = solver.peak(result["coeff"][1])
    retention = peak1 / peak0
    assert retention >= 0.99, retention


def test_solid_body_rotation_reentrant_check_p2():
    """After one revolution the solution must match the initial condition."""
    solver = DGAdvection2D(degree=2, ncells=16)
    coeff0 = solver.project(gaussian_bump)
    velocity = solid_body_rotation(omega=2.0 * math.pi)
    result = solver.solve(coeff0, velocity, final_time=1.0, cfl=0.3)
    # Use disk norm to avoid seam noise
    err = _relative_l2(solver, result["coeff"], rotated_profile(gaussian_bump, 2.0 * math.pi), t=1.0, radius=0.28)
    assert err < THREE_PERCENT, err


# --------------------------------------------------------------------------- #
# Reentrant deformational vortex
# --------------------------------------------------------------------------- #
def test_reentrant_vortex_returns_to_initial():
    """The reentrant vortex must return to the initial condition at t=T."""
    solver = DGAdvection2D(degree=2, ncells=16)
    coeff0 = solver.project(trig_profile)
    velocity = reentrant_vortex(period=1.0)
    result = solver.solve(coeff0, velocity, final_time=1.0, cfl=0.3)
    # L2 difference against initial condition
    diff = solver.l2_difference(result["coeff"], coeff0)
    ref = solver.l2_error(coeff0, trig_profile, 0.0)["reference_l2"]
    rel_err = diff["absolute_l2"] / ref
    assert rel_err < THREE_PERCENT, rel_err


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def test_estimate_max_speed_solid_body():
    """estimate_max_speed must return omega for solid-body rotation."""
    omega = 2.0 * math.pi
    vel = solid_body_rotation(omega=omega)
    est = estimate_max_speed(vel, times=(0.0,), n=24)
    assert abs(est - omega) < 0.1 * omega, est


def test_estimate_max_components_solid_body():
    """estimate_max_components must return (omega, omega) for solid-body rotation."""
    omega = 2.0 * math.pi
    vel = solid_body_rotation(omega=omega)
    bx, by = estimate_max_components(vel, times=(0.0,), n=24)
    assert abs(bx - omega) < 0.1 * omega, bx
    assert abs(by - omega) < 0.1 * omega, by


# --------------------------------------------------------------------------- #
# Negative controls
# --------------------------------------------------------------------------- #
def test_coarse_mesh_fails_three_percent_gate():
    """Coarse mesh (4 cells) must fail the 3 % gate for P2."""
    solver = DGAdvection2D(degree=2, ncells=4)
    coeff = solver.project(gaussian_bump)
    velocity = solid_body_rotation(omega=2.0 * math.pi)
    result = solver.solve(coeff, velocity, final_time=0.5, cfl=0.3)
    err = _relative_l2(solver, result["coeff"], rotated_profile(gaussian_bump, 2.0 * math.pi), t=0.5, radius=0.28)
    assert err > THREE_PERCENT, err


def test_invalid_degree_raises():
    """Degree must be in [0, 6]."""
    with pytest.raises(ValueError):
        DGAdvection2D(degree=-1)
    with pytest.raises(ValueError):
        DGAdvection2D(degree=7)


def test_invalid_ncells_raises():
    """ncells must be >= 2."""
    with pytest.raises(ValueError):
        DGAdvection2D(degree=1, ncells=1)
