"""Acceptance tests: curvilinear 2-D Euler DG (``tensordg.euler_curved``).

The gates mirror ``benchmarks/euler_curved_verification.py`` (which produces the
report numbers); these tests are the fast regression subset.

Invariants asserted (not merely "the code runs"):

F1  free-stream preservation on a *warped* periodic mesh: the semi-discrete
    operator annihilates a constant state at round-off, and a short ssp-rk3
    march does not move it.
F1n the same gate must FAIL for the aliasing volume derivative -- a vacuity
    check on the gate itself.
F2  the discrete metric identity ``d/dxi (J xi_x) + d/deta (J eta_x) = 0`` at
    round-off.
F3  smooth isentropic vortex on the warped mesh keeps the paper's accuracy
    tolerance ``order >= p + 1/2``.
F4  equal element counts: warped and straight meshes give comparable errors
    (curvature must not cost accuracy).
F5  conservative DG + consistent metrics: total mass/momentum/energy drift at
    round-off over a finite-time march.
F6  the general-normal HLLC reduces to the repository's verified axis-aligned
    HLLC (``euler_dg_2d._hllc``) to machine precision, is consistent with the
    physical flux for equal states, and is antisymmetric under state swap with
    reversed normal.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordg import euler_dg_2d
from tensordg.euler_curved import (
    CurvilinearEuler,
    conserved_from_primitive,
    euler_flux,
    hllc_normal,
    straight_box_mesh,
    warped_box_mesh,
)
from tensordg.euler_2d import vortex_primitive

BOX = 10.0
STRENGTH = 5.0
BACKGROUND = (1.0, 1.0)
AMPLITUDE = 0.06
GEOMETRY_DEGREE = 3


def exact_primitive(time):
    def primitive(x, y):
        rho, velocity_x, velocity_y, pressure = vortex_primitive(
            torch.as_tensor(np.ascontiguousarray(x)),
            torch.as_tensor(np.ascontiguousarray(y)),
            time=time, strength=STRENGTH, background_velocity=BACKGROUND, period=BOX,
        )
        return (rho.numpy(), velocity_x.numpy(), velocity_y.numpy(), pressure.numpy())

    return primitive


def warped(cells, degree):
    return warped_box_mesh(cells, cells, GEOMETRY_DEGREE, length=BOX, amplitude=AMPLITUDE)


# ---------------------------------------------------------------------------
# F6 HLLC: equivalence with the reference solver
# ---------------------------------------------------------------------------
def test_hllc_reduces_to_the_verified_axis_aligned_solver():
    rng = np.random.default_rng(3)
    left = rng.standard_normal((6, 4))
    right = rng.standard_normal((6, 4))
    left[:, 0] = 1.0 + np.abs(left[:, 0])
    right[:, 0] = 1.0 + np.abs(right[:, 0])
    left[:, 3] = 2.0 + np.abs(left[:, 3])
    right[:, 3] = 2.0 + np.abs(right[:, 3])
    for axis in (0, 1):
        normal = np.zeros(2)
        normal[axis] = 1.0
        mine = hllc_normal(
            left, right, np.full(6, normal[0]), np.full(6, normal[1])
        )
        reference = euler_dg_2d._hllc(
            torch.as_tensor(left), torch.as_tensor(right), axis
        ).numpy()
        assert np.max(np.abs(mine - reference)) < 1e-13


def test_hllc_consistency_and_rotational_antisymmetry():
    state = conserved_from_primitive(
        np.array([1.0, 0.4]), np.array([0.3, -0.7]), np.array([-0.2, 0.5]),
        np.array([1.0, 0.35]),
    )
    angle = np.array([0.0, 0.9272952180016122])
    normal_x, normal_y = np.cos(angle), np.sin(angle)
    same = hllc_normal(state, state, normal_x, normal_y)
    flux_x, flux_y = euler_flux(state)
    exact = flux_x * normal_x[:, None] + flux_y * normal_y[:, None]
    assert np.max(np.abs(same - exact)) < 1e-12
    # F(u_L, u_R; n) = -F(u_R, u_L; -n)
    forward = hllc_normal(state, state[::-1], normal_x, normal_y)
    backward = hllc_normal(state[::-1], state, -normal_x, -normal_y)
    assert np.max(np.abs(forward + backward)) < 1e-12


# ---------------------------------------------------------------------------
# F1 / F1n / F2 free-stream preservation and the metric identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_free_stream_is_preserved_to_roundoff(degree):
    solver = CurvilinearEuler(warped(4, degree), degree=degree)
    report = solver.free_stream_defect(steps=20)
    assert report["operator"] < 1e-11
    assert report["metric"] < 1e-11
    assert report["drift"] < 1e-10


def test_free_stream_gate_detects_aliased_volume_derivative():
    consistent = CurvilinearEuler(warped(4, 1), degree=1)
    aliased = CurvilinearEuler(warped(4, 1), degree=1, volume_derivative_degree=1)
    good = consistent.free_stream_defect(steps=0)["relative"]
    bad = aliased.free_stream_defect(steps=0)["relative"]
    assert good < 1e-13
    assert bad > 1e3 * max(good, 1e-16)
    assert bad > 1e-3


def test_project_reproduces_a_constant_state_exactly():
    solver = CurvilinearEuler(warped(4, 1), degree=2)
    state = solver.project(lambda x, y: (1.0, 0.3, -0.2, 1.0))
    constant = np.zeros_like(state)
    constant[:, 0, 0, :] = state[:, 0, 0, :]
    assert np.max(np.abs(state - constant)) < 1e-14


# ---------------------------------------------------------------------------
# F3 / F4 accuracy on curved meshes
# ---------------------------------------------------------------------------
def test_design_order_on_the_warped_mesh():
    errors = []
    for cells in (8, 16):
        solver = CurvilinearEuler(warped(cells, 1), degree=1, cfl=0.2)
        state = solver.project(exact_primitive(0.0))
        state, _, _ = solver.evolve(state, 0.1)
        errors.append(solver.relative_errors(state, exact_primitive(0.1))["worst"])
    order = math.log(errors[0] / errors[1]) / math.log(2.0)
    assert order >= 1.5, (order, errors)


def test_warped_and_straight_mesh_accuracy_are_comparable():
    cells = 16
    results = {}
    for family, mesh in (
        ("warped", warped(cells, 1)),
        ("straight", straight_box_mesh(cells, cells, 1, length=BOX)),
    ):
        solver = CurvilinearEuler(mesh, degree=1, cfl=0.2)
        state = solver.project(exact_primitive(0.0))
        state, _, _ = solver.evolve(state, 0.1)
        results[family] = solver.relative_errors(state, exact_primitive(0.1))["worst"]
    ratio = results["warped"] / results["straight"]
    assert 0.3 <= ratio <= 3.0, (ratio, results)


# ---------------------------------------------------------------------------
# F5 conservation
# ---------------------------------------------------------------------------
def test_total_conserved_quantities_hold_on_the_warped_mesh():
    solver = CurvilinearEuler(warped(8, 1), degree=1, cfl=0.2)
    state = solver.project(exact_primitive(0.0))
    before = solver.totals(state)
    state, _, steps = solver.evolve(state, 0.5)
    after = solver.totals(state)
    assert steps > 10
    drift = np.max(np.abs(after - before)) / np.max(np.abs(before))
    assert drift < 1e-10, drift


# ---------------------------------------------------------------------------
# Input discipline
# ---------------------------------------------------------------------------
def test_mesh_builders_reject_bad_input():
    with pytest.raises(ValueError):
        warped_box_mesh(4, 4, 1, amplitude=0.06)   # bilinear cannot curve
    with pytest.raises(ValueError):
        CurvilinearEuler(warped(2, 1), degree=0 - 1)  # negative degree
    with pytest.raises(ValueError):
        CurvilinearEuler(warped(2, 1), degree=1, nquad=1)  # too few quadrature points