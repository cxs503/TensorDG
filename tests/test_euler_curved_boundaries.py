"""Boundary conditions of the curvilinear Euler DG solver.

The module adds wall / far-field boundary kinds and a body-fitted cylinder
O-grid.  What is asserted here is exactly what holds, and nothing more:

B1  the boundary-kind resolver is total and refuses inconsistent requests
    (periodic kind on a non-periodic axis, unknown kinds, missing far-field);
B2  ``cylinder_annulus_mesh`` resolves the analytic annulus area, is periodic in
    theta, flags only the cylinder as a wall, and its radial spacing is
    clustered (``growth > 1``) and strictly increasing;
B3  free-stream preservation **with walls**: a uniform slip flow in a
    wall-bounded channel is annihilated by the semi-discrete operator to
    round-off (this is what makes the wall flux sign- and orientation-correct);
B4  the wall numerical flux is impermeable *and* adiabatic: mass and energy
    components vanish to round-off for arbitrary admissible states, the momentum
    component is ``p n``;
B5  a wall-closed annulus conserves total mass and total energy to round-off
    over a short march;
B6  HONEST NEGATIVE RESULT: the reflective wall is linearly unstable -- the
    explicit march of an exact steady state amplifies round-off until it blows
    up.  If a later round fixes the boundary stability this test fails, which is
    the intended trigger to remove the warning in the module docstring.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordg.curved_geometry import CurvedMesh, IsoparametricMap
from tensordg.euler_curved import (
    BOUNDARY_KINDS,
    CurvilinearEuler,
    conserved_from_primitive,
    cylinder_annulus_mesh,
    resolve_boundaries,
    wall_normal_flux,
)


def _channel(nx: int, ny: int, lx: float = 4.0, ly: float = 1.0,
             periodic=(False, False)) -> CurvedMesh:
    maps = [[IsoparametricMap.affine(lx * i / nx, lx * (i + 1) / nx,
                                     ly * j / ny, ly * (j + 1) / ny, 1)
             for j in range(ny)] for i in range(nx)]
    return CurvedMesh(maps, periodic=periodic, walls=(False, False, True, True))


def _uniform_slip(x, y):
    return (np.ones_like(x), 0.3 * np.ones_like(x), np.zeros_like(x),
            np.ones_like(x))


def _closed_annulus(nr=8, ntheta=16, ratio=2.0):
    mesh = cylinder_annulus_mesh(nr, ntheta, 1.0, 1.0 + ratio, degree=2)
    mesh.walls = (True, True, False, False)
    mesh.neighbours = mesh._build_neighbours()
    return mesh


# -- B1 --------------------------------------------------------------------
def test_boundary_kinds_resolve_from_mesh():
    mesh = cylinder_annulus_mesh(4, 8, 1.0, 2.0, degree=2)
    kinds = resolve_boundaries(mesh, None)
    assert kinds[(0, -1)] == "wall" and kinds[(0, 1)] == "outflow"
    assert kinds[(1, -1)] == kinds[(1, 1)] == "periodic"
    explicit = resolve_boundaries(mesh, ("wall", "farfield", "periodic", "periodic"))
    assert explicit[(0, 1)] == "farfield"


def test_boundary_kinds_reject_inconsistent_requests():
    mesh = cylinder_annulus_mesh(4, 8, 1.0, 2.0, degree=2)
    with pytest.raises(ValueError):
        resolve_boundaries(mesh, ("wall", "wall", "periodic", "periodic")[:3])
    with pytest.raises(ValueError):
        resolve_boundaries(mesh, ("wall", "wall", "periodic", "wall"))
    with pytest.raises(ValueError):
        resolve_boundaries(mesh, ("wall", "nowhere", "periodic", "periodic"))
    assert "farfield" in BOUNDARY_KINDS


def test_farfield_kind_requires_a_state():
    mesh = cylinder_annulus_mesh(4, 8, 1.0, 2.0, degree=2)
    with pytest.raises(ValueError):
        CurvilinearEuler(mesh, 1, boundaries=("wall", "farfield", "periodic", "periodic"))
    solver = CurvilinearEuler(mesh, 1, boundaries=("wall", "farfield", "periodic", "periodic"),
                              farfield=(1.0, 0.44, 0.0, 1.0))
    np.testing.assert_allclose(solver.farfield, conserved_from_primitive(1.0, 0.44, 0.0, 1.0))


# -- B2 --------------------------------------------------------------------
@pytest.mark.parametrize("nr,nt,gdeg", [(4, 8, 1), (8, 16, 2), (12, 24, 2)])
def test_annulus_mesh_geometry(nr, nt, gdeg):
    r_in, r_out = 1.0, 3.0
    mesh = cylinder_annulus_mesh(nr, nt, r_in, r_out, degree=gdeg)
    exact = math.pi * (r_out ** 2 - r_in ** 2)
    # g = 1 is the straight-edge control: circle chords are O(h^2) poor on
    # purpose; superparametric (g = 2) is the production geometry
    tol = 2e-1 if gdeg == 1 else 1e-3
    assert abs(float(np.sum(mesh.areas())) - exact) / exact < tol
    assert mesh.periodic == (False, True)
    assert mesh.walls == (True, False, False, False)
    # the mesh is usable by the solver and exposes both circular wall families
    solver = CurvilinearEuler(mesh, 1)
    assert (0, -1) in solver.face and (0, 1) in solver.face
    assert (1, -1) in solver.face and (1, 1) in solver.face


def test_annulus_mesh_radial_clustering():
    """Geometric radial spacing: per-ring cell area must grow with radius."""
    def ring_areas(growth):
        mesh = cylinder_annulus_mesh(6, 12, 1.0, 5.0, degree=1, growth=growth)
        areas = mesh.areas()
        return np.array([areas[i * mesh.ny:(i + 1) * mesh.ny].mean()
                         for i in range(mesh.nx)])
    uniform = ring_areas(1.0)
    clustered = ring_areas(1.8)
    assert np.all(np.diff(clustered) > 0.0)
    # growth > 1 must make the innermost ring thinner and the outermost thicker
    # than the uniform mesh with the same element counts
    assert clustered[0] < uniform[0]
    assert clustered[-1] > uniform[-1]
    assert abs(clustered.sum() - uniform.sum()) / uniform.sum() < 1e-2
    with pytest.raises(ValueError):
        cylinder_annulus_mesh(6, 12, 1.0, 5.0, degree=1, growth=0.9)


# -- B3 --------------------------------------------------------------------
def test_free_stream_preservation_with_walls():
    mesh = _channel(8, 6)
    solver = CurvilinearEuler(mesh, 1)
    assert solver.wall_sides == ((1, -1), (1, 1))
    state = solver.project(_uniform_slip)
    assert np.abs(solver.rhs(state)).max() < 1e-12


def test_free_stream_preservation_is_not_vacuous():
    """A wall flux with the wrong sign must break B3 -- gate-vacuity control."""
    mesh = _channel(4, 3)
    solver = CurvilinearEuler(mesh, 1)
    state = solver.project(_uniform_slip)
    broken = CurvilinearEuler(mesh, 1)
    original = broken._boundary_normal_flux

    def flipped(kind, inner, side, unit_x, unit_y):
        return -original(kind, inner, side, unit_x, unit_y)

    broken._boundary_normal_flux = flipped
    assert np.abs(broken.rhs(state)).max() > 1e-2


# -- B4 --------------------------------------------------------------------
def test_wall_flux_is_impermeable_and_adiabatic():
    rng = np.random.default_rng(20261115)
    normal_x = rng.normal(size=40)
    normal_y = rng.normal(size=40)
    norm = np.hypot(normal_x, normal_y)
    normal_x, normal_y = normal_x / norm, normal_y / norm
    rho = 1.0 + 0.3 * rng.random(40)
    tangential = rng.normal(size=40)      # admissible wall state: u_n = 0
    pressure = 0.7 + 0.6 * rng.random(40)
    state = conserved_from_primitive(
        rho, -tangential * normal_y, tangential * normal_x, pressure)
    flux = wall_normal_flux(state, normal_x, normal_y)
    assert np.abs(flux[:, 0]).max() < 1e-12          # no mass flux
    assert np.abs(flux[:, 3]).max() < 1e-12          # no energy flux
    np.testing.assert_allclose(flux[:, 1], pressure * normal_x, atol=1e-12)
    np.testing.assert_allclose(flux[:, 2], pressure * normal_y, atol=1e-12)


def test_wall_force_closes_on_a_closed_annulus():
    mesh = _closed_annulus()
    solver = CurvilinearEuler(mesh, 1)
    state = solver.project(lambda x, y: (np.ones_like(x), np.zeros_like(x),
                                         np.zeros_like(x), np.ones_like(x)))
    force = solver.wall_force(state)
    # uniform pressure: the closed-surface pressure integral must cancel
    assert abs(force["Fx"]) < 1e-10 and abs(force["Fy"]) < 1e-10


# -- B5 --------------------------------------------------------------------
def test_closed_annulus_conserves_mass_and_energy():
    mesh = _closed_annulus()
    solver = CurvilinearEuler(mesh, 1)
    state = solver.project(lambda x, y: (
        1.0 + 0.05 * np.exp(-((x - 1.6) ** 2 + y ** 2) / 0.2),
        np.zeros_like(x), np.zeros_like(x), np.ones_like(x)))
    before = solver.totals(state)
    dt = solver.stable_dt(state)
    for _ in range(20):
        state = solver.ssp_rk3_step(state, dt)
    after = solver.totals(state)
    assert abs(after[0] - before[0]) / before[0] < 1e-10
    assert abs(after[3] - before[3]) / before[3] < 1e-10


# -- B6 --------------------------------------------------------------------
def test_wall_boundary_march_is_still_unstable_honest():
    """HONEST NEGATIVE RESULT -- updated after the interface-orientation fix.

    The state below is an *exact* steady solution (uniform slip flow, wall flux
    equal to the physical flux) and the semi-discrete residual is round-off, yet
    the explicit march still amplifies round-off exponentially: the reflective
    wall is *still* linearly unstable as implemented here.  What the interface
    fix changed is only the rate: it blows up at step ~700 (dt = 4.2e-3) where
    the pre-fix operator needed a few hundred steps and grew ~6x per 10 steps.
    This test pins the remaining defect so it cannot be claimed as a working
    capability; deleting it is only allowed together with a stable two-point /
    BR1 wall flux (see the module docstring of ``euler_curved``).
    """
    mesh = _channel(8, 6)
    solver = CurvilinearEuler(mesh, 1)
    state = solver.project(_uniform_slip)
    dt = solver.stable_dt(state)
    start = float(np.max(np.abs(solver.rhs(state))))
    assert start < 1e-10, start          # the exact state *is* steady
    drift = []
    state0 = state.copy()
    blew_up = False
    for step in range(2000):
        state = solver.ssp_rk3_step(state, dt)
        if step % 200 == 0:
            drift.append(float(np.max(np.abs(state - state0))))
        if not np.all(np.isfinite(state)) or np.abs(state).max() > 1e3:
            blew_up = True
            break
    assert blew_up, ("the wall-boundary march stayed bounded over 2000 steps -- "
                     f"if the boundary is now stable, remove the instability "
                     f"warning from the module docstring (drift={drift})")
    assert len(drift) >= 2 and drift[-1] > 1e3 * max(drift[0], 1e-16), drift
