"""Verification tests for :mod:`tensordg.advection_2d_var`.

This module exercises the 2-D DG advection solver with space/time-varying,
divergence-free velocity fields.  Every gate is an *external* reference
(analytic solution or conservation law), never a self-consistency check.

Reference-case validity (audited 2026-11-09)
-------------------------------------------
Measured evidence and derivation: ``docs/reports/tensordg-advection-2d-var-reference-audit.md``.

* **reentrant deformational vortex** -- LeVeque/Nair flow with ``cos(pi t/T)``
  modulation.  The velocity is smooth *on the torus*, and the flow map over
  ``[0, T]`` is the identity, so ``u(T) == u(0)`` exactly.  This is the only
  reference case here that is valid as a **global** norm and therefore the only
  one used for the design-order gates.

* **solid-body rotation** -- rigid rotation of a Gaussian bump.  The velocity
  ``omega * (-(y-1/2), x-1/2)`` **jumps by ``omega`` across the periodic seam**
  (at ``y = 0 == 1`` and at ``x = 0 == 1``), i.e. it is not a smooth periodic
  field.  Consequently:

  - the "exact solution" (the initial field rotated back by ``omega t``) does
    **not** satisfy the transport equation across the seam.  For a
    domain-filling profile this poisons a global L2 norm, which then has an
    O(1) error that *grows* under refinement (measured P2, ``trig_profile``:
    1.19e-1 at n = 8, 1.41e-1 at n = 16).  For a compactly supported bump the
    seam contribution is small and the global norm happens to converge
    (0.56 % -> 0.070 % for P2 at n = 16 -> 32), but it is not a *robust*
    quantity: it only works while the field stays away from the seam;
  - the characteristics of a rigid rotation are circles about the centre, so the
    seam layer stays at its own radius and a disk ``r <= 0.28`` about
    ``(1/2, 1/2)`` (which contains the bump trajectory at radius 0.15) gives a
    well-posed norm.  All rotation gates below use that disk.

Design-order gates
------------------
* P1 (degree=1) observed order >= 1.9 (design 2).
* P2 (degree=2) observed order >= 2.9 (design 3).
Observed order = least-squares slope of ``log(err)`` vs ``log(n)`` over the
mesh ladder, the same estimator as the benchmark registry
(``tensordg.ice_edge_bench`` / ``docs/benchmarks/dg_advection_bench.md``).

Negative controls
-----------------
* Coarse mesh must fail the 3 % gate.
* Collapsed quadrature (nq < 2*degree+3) must fail.
* The rigid-rotation **global** norm must NOT satisfy a 3 % gate at a fine mesh
  (guard against re-introducing the invalid reference).
"""
from __future__ import annotations

import math

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
DISK_RADIUS = 0.28


def _relative_l2(solver, coeff, exact, t=0.0, radius=None):
    """Return relative L2 error."""
    return solver.l2_error(coeff, exact, t, radius=radius)["relative_l2"]


def _eoc(errors):
    return [
        math.log(errors[i] / errors[i + 1]) / math.log(2.0)
        for i in range(len(errors) - 1)
    ]


def _observed_order(meshes, errors):
    """Least-squares slope of ``log(err)`` vs ``log(n)`` (benchmark-registry estimator)."""
    xs = [math.log(float(n)) for n in meshes]
    ys = [math.log(float(e)) for e in errors]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return -num / den


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
# Convergence order (smooth periodic velocity, analytic identity flow map)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("degree, required", [(1, 1.9), (2, 2.9)])
def test_reentrant_vortex_design_order(degree, required):
    """Reentrant vortex must converge at the design order ``degree + 1``.

    The exact solution at ``t = T`` is the initial field (identity flow map), so
    the gate is a genuine external reference and the velocity is smooth on the
    torus -- unlike the rigid-rotation case, whose seam makes global norms
    meaningless (see module docstring).
    """
    meshes = (12, 16, 24)
    errors = []
    for n in meshes:
        solver = DGAdvection2D(degree=degree, ncells=n)
        coeff = solver.project(trig_profile)
        velocity = reentrant_vortex(period=1.0)
        result = solver.solve(coeff, velocity, final_time=1.0, cfl=0.3)
        errors.append(_relative_l2(solver, result["coeff"], trig_profile, t=1.0))
    order = _observed_order(meshes, errors)
    assert order > required, (order, errors, _eoc(errors))


# --------------------------------------------------------------------------- #
# Solid-body rotation (disk norm: the seam layer is excluded, see docstring)
# --------------------------------------------------------------------------- #
def test_solid_body_rotation_disk_accuracy_p2():
    """P2 rotating front must match the rotated profile to < 3 % on the disk."""
    solver = DGAdvection2D(degree=2, ncells=16)
    coeff = solver.project(gaussian_bump)
    velocity = solid_body_rotation(omega=2.0 * math.pi)
    result = solver.solve(coeff, velocity, final_time=1.0, cfl=0.3)
    exact = rotated_profile(gaussian_bump, 2.0 * math.pi)
    err = _relative_l2(solver, result["coeff"], exact, t=1.0, radius=DISK_RADIUS)
    assert err < THREE_PERCENT, err


def test_solid_body_rotation_reentrant_check_p2():
    """After one revolution the solution must match the initial condition."""
    solver = DGAdvection2D(degree=2, ncells=16)
    coeff0 = solver.project(gaussian_bump)
    velocity = solid_body_rotation(omega=2.0 * math.pi)
    result = solver.solve(coeff0, velocity, final_time=1.0, cfl=0.3)
    err = _relative_l2(solver, result["coeff"], rotated_profile(gaussian_bump, 2.0 * math.pi),
                       t=1.0, radius=DISK_RADIUS)
    assert err < THREE_PERCENT, err


def test_solid_body_rotation_peak_retention_p3():
    """Gaussian bump peak retention >= 0.99 after one revolution (p=3)."""
    solver = DGAdvection2D(degree=3, ncells=16)
    coeff = solver.project(gaussian_bump)
    peak0 = solver.peak(coeff)[1]
    velocity = solid_body_rotation(omega=2.0 * math.pi)
    result = solver.solve(coeff, velocity, final_time=1.0, cfl=0.3)
    # ``result["coeff"]`` is already the modal field; indexing it would drop a mode axis.
    peak1 = solver.peak(result["coeff"])[1]
    retention = peak1 / peak0
    assert retention >= 0.99, retention


# --------------------------------------------------------------------------- #
# Reentrant deformational vortex
# --------------------------------------------------------------------------- #
def test_reentrant_vortex_returns_to_initial():
    """The reentrant vortex must return to the initial condition at t=T."""
    solver = DGAdvection2D(degree=2, ncells=16)
    coeff0 = solver.project(trig_profile)
    velocity = reentrant_vortex(period=1.0)
    result = solver.solve(coeff0, velocity, final_time=1.0, cfl=0.3)
    diff = solver.l2_difference(result["coeff"], coeff0)
    ref = solver.l2_error(coeff0, trig_profile, 0.0)["reference_l2"]
    rel_err = diff["absolute_l2"] / ref
    assert rel_err < THREE_PERCENT, rel_err


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def _centre_offsets(n):
    """Node offsets from the rotation centre used by the speed estimators."""
    return [(i + 0.5) / n - 0.5 for i in range(n)]


def test_estimate_max_speed_solid_body():
    """estimate_max_speed must return the sampled max |a| = omega * r_max * sqrt(2).

    The rigid-rotation speed at the box centre is 0, so the maximum is attained at
    the corner-ward sampled nodes, NOT at ``omega`` (the historic expectation
    ``est == omega`` in this file was wrong by a factor ``sqrt(2)*r_max``).
    """
    n = 24
    omega = 2.0 * math.pi
    vel = solid_body_rotation(omega=omega)
    est = estimate_max_speed(vel, times=(0.0,), n=n)
    r = max(abs(o) for o in _centre_offsets(n))
    expected = omega * math.sqrt(2.0) * r
    assert est == pytest.approx(expected, rel=1e-12), (est, expected)


def test_estimate_max_components_solid_body():
    """estimate_max_components must return ``omega * r_max`` per component."""
    n = 24
    omega = 2.0 * math.pi
    vel = solid_body_rotation(omega=omega)
    bx, by = estimate_max_components(vel, times=(0.0,), n=n)
    expected = omega * max(abs(o) for o in _centre_offsets(n))
    assert bx == pytest.approx(expected, rel=1e-12), (bx, expected)
    assert by == pytest.approx(expected, rel=1e-12), (by, expected)


# --------------------------------------------------------------------------- #
# Negative controls
# --------------------------------------------------------------------------- #
def test_coarse_mesh_fails_three_percent_gate():
    """Coarse mesh (4 cells) must fail the 3 % gate for P2 (disk norm)."""
    solver = DGAdvection2D(degree=2, ncells=4)
    coeff = solver.project(gaussian_bump)
    velocity = solid_body_rotation(omega=2.0 * math.pi)
    result = solver.solve(coeff, velocity, final_time=0.5, cfl=0.3)
    err = _relative_l2(solver, result["coeff"], rotated_profile(gaussian_bump, 2.0 * math.pi),
                       t=0.5, radius=DISK_RADIUS)
    assert err > THREE_PERCENT, err


def test_rigid_rotation_global_norm_is_not_a_reference():
    """Guard: the rigid-rotation *global* norm does not converge and is not a gate.

    The rotation velocity jumps across the periodic seam, so the rotated profile
    is not a solution of the PDE there.  For a domain-filling profile the error is
    therefore O(1) and grows under refinement (measured P2, ``trig_profile``:
    1.19e-1 / 1.41e-1 at n = 8 / 16).  A compactly supported bump happens to keep
    a convergent global norm (5.6e-3 -> 7.0e-4 for P2 at n = 16 -> 32) because it
    never touches the seam, but that is a property of the profile, not of the
    norm; the disk norm ``r <= 0.28`` is the robust choice used by the other
    rotation gates.
    """
    velocity = solid_body_rotation(omega=2.0 * math.pi)
    exact = rotated_profile(trig_profile, 2.0 * math.pi)
    errors = []
    for n in (8, 16):
        solver = DGAdvection2D(degree=2, ncells=n)
        coeff = solver.project(trig_profile)
        result = solver.solve(coeff, velocity, final_time=0.25, cfl=0.3)
        errors.append(_relative_l2(solver, result["coeff"], exact, t=0.25))
    assert errors[-1] > THREE_PERCENT, errors
    assert errors[-1] > 0.9 * errors[0], errors  # no convergence under refinement


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