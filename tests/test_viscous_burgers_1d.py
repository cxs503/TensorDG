"""Gates for CASE-DG-VISCOUS-BURGERS-COLEHOPF (periodic viscous Burgers, Cole--Hopf).

Three layers, deliberately separable:

1. **Reference layer** -- the Cole--Hopf formula must be coded correctly, which is
   checked against a second, independent implementation (FFT + exact-in-time heat
   decay) and against the exact invariants of the datum (zero mean, antisymmetry
   about ``x = 1/2``).
2. **Operator layer** -- the ring SIPG stiffness must be the *production*
   stencil (:func:`tensordg.heat_1d.assemble_sipg_1d`) closed periodically: its
   interior rows are asserted identical, it must annihilate constants exactly
   and be symmetric positive semi-definite.
3. **Solution layer** -- mass conservation, energy decay against the reference,
   observed order in ``h`` at fixed degree, and the documented <3 % run.

The three solutions used by several tests are cached: the diffusive timestep
restriction (``dt ~ h^2``) makes a P2/N16 march ~12 s, so recomputing it per test
would waste the suite's budget without adding evidence.
"""
from __future__ import annotations

import functools
import math

import numpy as np
import pytest
import torch

from tensordg.burgers_1d import _operators as flux_operators
from tensordg.burgers_1d import _rhs as burgers_rhs
from tensordg.cases import viscous_burgers_1d as vb
from tensordg.heat_1d import assemble_sipg_1d

NU = vb.NU
T = vb.FINAL_TIME


# ---------------------------------------------------------------------------
# cached marches
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _solve(ncells: int, degree: int, nu: float = NU, final_time: float = T):
    return vb.solve_viscous_burgers_1d(ncells, degree, final_time, nu)


def _reference_energy(nu: float = NU, final_time: float = T, points: int = 4096) -> float:
    x = (np.arange(points) + 0.5) / points
    u = vb.exact_solution(x, final_time, nu)
    return float(0.5 * np.mean(u ** 2))


def _grid_error(ncells: int, degree: int) -> float:
    return vb.l2_error(_solve(ncells, degree))


# ---------------------------------------------------------------------------
# layer 1: the reference
# ---------------------------------------------------------------------------
def test_reference_implementations_agree():
    """Cole--Hopf quadrature and the independent Fourier solver agree to round-off."""
    x = np.linspace(0.013, 0.987, 23)
    direct = vb.exact_solution(x, T, NU)
    for modes in (64, 128, 256):
        fourier = vb.exact_solution_fourier(x, T, NU, modes=modes)
        assert np.max(np.abs(direct - fourier)) < 1e-13, modes


def test_reference_is_quadrature_resolution_independent():
    """The midpoint rule is spectrally accurate here, so the value must be frozen."""
    x = np.linspace(0.013, 0.987, 23)
    base = vb.exact_solution(x, T, NU, samples=1024)
    for samples in (64, 128, 256, 512):
        assert np.max(np.abs(vb.exact_solution(x, T, NU, samples=samples) - base)) < 1e-13, samples


def test_reference_honours_the_exact_invariants():
    """Zero mean and antisymmetry about x = 1/2 are exact for this datum."""
    x = (np.arange(2048) + 0.5) / 2048
    u = vb.exact_solution(x, T, NU)
    assert abs(u.mean()) < 1e-12
    shifts = np.array([0.1, 0.25, 0.4])
    mirror = (vb.exact_solution(0.5 + shifts, T, NU) + vb.exact_solution(0.5 - shifts, T, NU))
    assert np.max(np.abs(mirror)) < 1e-12  # u = 0 on the symmetry axis
    assert abs(float(vb.exact_solution(np.array([0.5]), T, NU)[0])) < 1e-12


def test_reference_reduces_to_the_datum_at_zero_time():
    x = np.linspace(0.0, 1.0, 17)
    assert np.max(np.abs(vb.exact_solution(x, 0.0, NU) - vb.initial_profile(x))) == 0.0


# ---------------------------------------------------------------------------
# layer 2: the ring SIPG operator
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ncells,degree", [(8, 1), (8, 2), (6, 3)])
def test_periodic_sipg_interior_rows_are_the_production_stencil(ncells, degree):
    """Every cell that is not cut by the seam must reproduce assemble_sipg_1d."""
    ring, _, _ = vb.periodic_sipg(ncells, degree, vb.PENALTY)
    production = assemble_sipg_1d(ncells=ncells, degree=degree, penalty=vb.PENALTY)["diffusion"]
    modes = degree + 1
    for cell in range(1, ncells - 1):
        rows = slice(cell * modes, (cell + 1) * modes)
        assert torch.max(torch.abs(ring[rows] - production[rows])) < 1e-12, (ncells, degree, cell)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_periodic_sipg_is_symmetric_positive_semidefinite(degree):
    ring, mass_diag, _ = vb.periodic_sipg(6, degree, vb.PENALTY)
    assert torch.max(torch.abs(ring - ring.T)) < 1e-10
    scaled = torch.diag(1.0 / torch.sqrt(mass_diag)) @ ring @ torch.diag(1.0 / torch.sqrt(mass_diag))
    eigenvalues = torch.linalg.eigvalsh(scaled)
    assert float(eigenvalues.min()) > -1e-8      # null space = constants only
    assert float(eigenvalues.max()) > 0.0


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_periodic_sipg_annihilates_constants(degree):
    """A constant field has no jump and no gradient: the operator must see zero."""
    ring, mass_diag, _ = vb.periodic_sipg(5, degree, vb.PENALTY)
    constant = vb._project(5, degree, lambda z: np.full_like(z, 2.5))
    residual = (ring / mass_diag[:, None]) @ constant.reshape(-1)
    assert float(residual.abs().max()) < 1e-9


def test_zero_viscosity_removes_the_diffusion_contribution_exactly():
    """nu = 0 must reproduce the production inviscid operator bit for bit."""
    ncells, degree = 8, 2
    rhs, _, _, h = vb.rhs_operator(ncells, degree, 0.0)
    _, weights, phi, dphi, phi_left, phi_right = flux_operators(degree, torch.device("cpu"), torch.float64)
    u = vb._project(ncells, degree, vb.initial_profile)
    expected = burgers_rhs(u, h, weights, phi, dphi, phi_left, phi_right)
    assert torch.equal(rhs(u), expected)


# ---------------------------------------------------------------------------
# layer 3: the marched solution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ncells,degree", [(8, 2), (16, 2)])
def test_mass_is_conserved(ncells, degree):
    assert abs(vb.total_mass(_solve(ncells, degree))) < 1e-12


def test_energy_decays_and_tracks_the_reference():
    """Burgers diffuses: E must fall, and the *discrete* E must match the exact E."""
    result = _solve(16, 2)
    initial = 0.5 * float(np.mean(vb.initial_profile((np.arange(4096) + 0.5) / 4096) ** 2))
    final = vb.total_energy(result)
    reference = _reference_energy()
    assert final < initial
    assert abs(final - reference) / reference < 0.01, (final, reference)


def test_documented_case_is_below_three_percent():
    """The documented run (P2, 16 cells) must beat the 3 % acceptance tolerance."""
    error = _grid_error(vb.CELLS, vb.DEGREE)
    assert error < vb.TOLERANCE, error


def test_p1_converges_at_second_order():
    coarse, fine = _grid_error(8, 1), _grid_error(16, 1)
    order = math.log2(coarse / fine)
    assert order > 1.8, (coarse, fine, order)


def test_p2_converges_faster_than_p1_on_the_same_meshes():
    coarse, fine = _grid_error(8, 2), _grid_error(16, 2)
    order = math.log2(coarse / fine)
    assert order > 1.9, (coarse, fine, order)
    assert fine < _grid_error(16, 1)


def test_timestep_respects_the_measured_diffusive_spectrum():
    """The RK4 bound must be met with the measured lambda_max, not a guess."""
    ncells, degree = 8, 2
    stiffness, mass_diag, _ = vb.periodic_sipg(ncells, degree, vb.PENALTY)
    dt, lambda_max = vb.stable_timestep(
        ncells, degree, NU, penalty=vb.PENALTY, stiffness=stiffness, mass_diag=mass_diag)
    assert lambda_max > 0.0
    assert NU * lambda_max * dt <= vb.RK4_NEGATIVE_REAL_BOUND
    result = _solve(ncells, degree)
    assert result["steps"] < 20000 and result["dt"] > 0.0


def test_case_is_registered():
    from tensordg.cases import VISCOUS_BURGERS_CASE_KEY, viscous_burgers_parameters

    assert VISCOUS_BURGERS_CASE_KEY == vb.CASE_KEY
    parameters = viscous_burgers_parameters()
    assert parameters["nu"] == NU and parameters["final_time"] == T
    assert parameters["tolerance"] == vb.TOLERANCE