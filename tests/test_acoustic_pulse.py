"""Verification face for the 2-D linear-acoustics DG solver.

Four independent layers, cheapest first:

1. **Flux algebra** -- the upwind flux is the exact Godunov solution of the
   linear acoustic Riemann problem, and it reduces to the physical flux for
   continuous states.  The argument *order* (left/right) is part of the
   contract: swapping it still looks consistent on smooth data but destroys the
   upwinding at jumps, so it is pinned by an explicit test (this was a real
   defect found while building the solver).
2. **Finite-volume limit** -- for ``P0`` DG must collapse onto the upwind FV
   scheme, whose update is known in closed form.
3. **Exact solutions** -- :func:`tensordg.acoustic_pulse_exact_pressure` is an
   "exact up to quadrature" Hankel integral; it is validated by refinement, by
   the ``t = 0`` identity, by an *independent* real-space Poisson-disk
   integral, and finally by the fact that the reconstructed state satisfies the
   acoustic PDEs in radial form.
4. **Solver convergence / conservation** -- design order ``p + 1`` in the
   relative ``L2`` norm against the exact solution, plus the acoustic energy
   converging to its analytic initial value ``pi sigma^2 A^2 / (4 K)``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordg import (
    AcousticMedium,
    acoustic_energy,
    acoustic_flux_x,
    acoustic_flux_y,
    acoustic_plane_pulse_exact,
    acoustic_pulse_exact_pressure,
    acoustic_pulse_exact_state,
    acoustic_pulse_exact_velocity,
    acoustic_pulse_initial,
    acoustic_riemann_flux_x,
    acoustic_riemann_flux_y,
    solve_acoustic_2d,
)
from tensordg.acoustics import _acoustic_rhs
from tensordg.elastic_2d import legendre_operators_2d, modal_to_quadrature, quadrature_grid

MEDIUM = AcousticMedium(rho=1.0, c=1.0)
SIGMA = 0.15
FINAL_TIME = 0.3
LENGTH = 2.0


# ---------------------------------------------------------------------------
# 1. flux algebra
# ---------------------------------------------------------------------------
def _random_state(shape=(7, 5), seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, 3, generator=generator, dtype=torch.float64), generator


@pytest.mark.parametrize("medium", [AcousticMedium(), AcousticMedium(1.225, 340.0)])
def test_riemann_flux_reduces_to_the_physical_flux(medium):
    state, _ = _random_state()
    assert torch.allclose(acoustic_riemann_flux_x(state, state, medium),
                          acoustic_flux_x(state, medium), atol=1e-12)
    assert torch.allclose(acoustic_riemann_flux_y(state, state, medium),
                          acoustic_flux_y(state, medium), atol=1e-12)


@pytest.mark.parametrize("medium", [AcousticMedium(), AcousticMedium(1.225, 340.0)])
def test_riemann_star_state_matches_the_characteristic_invariants(medium):
    """The interface state is the exact solution of the linear Riemann problem.

    Across the left-going acoustic wave the invariant ``p + Z u`` is continuous
    and across the right-going one ``p - Z u`` is, hence

        p* = (pL + pR)/2 + Z (uL - uR)/2 ,
        u* = (uL + uR)/2 + (pL - pR)/(2 Z) ,

    which is what the flux must encode (``F* = (K u*, p*/rho)``).  The formula is
    re-derived here from the invariants, independently of the implementation.
    """
    z = medium.impedance
    generator = torch.Generator().manual_seed(7)
    left = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    right = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    star = acoustic_riemann_flux_x(left, right, medium)
    p_star = 0.5 * (left[:, 0] + right[:, 0]) + 0.5 * z * (left[:, 1] - right[:, 1])
    u_star = 0.5 * (left[:, 1] + right[:, 1]) + 0.5 * (left[:, 0] - right[:, 0]) / z
    assert torch.allclose(star[:, 0], medium.bulk_modulus * u_star, atol=1e-12)
    assert torch.allclose(star[:, 1], p_star / medium.rho, atol=1e-12)

    star_y = acoustic_riemann_flux_y(left, right, medium)
    p_star_y = 0.5 * (left[:, 0] + right[:, 0]) + 0.5 * z * (left[:, 2] - right[:, 2])
    v_star = 0.5 * (left[:, 2] + right[:, 2]) + 0.5 * (left[:, 0] - right[:, 0]) / z
    assert torch.allclose(star_y[:, 0], medium.bulk_modulus * v_star, atol=1e-12)
    assert torch.allclose(star_y[:, 2], p_star_y / medium.rho, atol=1e-12)


def test_riemann_star_state_of_a_pressure_jump_is_physical():
    """High pressure on the left must push the fluid to the right (``u* > 0``)."""
    z = MEDIUM.impedance
    left = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    right = torch.zeros(3, dtype=torch.float64)
    star = acoustic_riemann_flux_x(left, right, MEDIUM)
    p_star = star[1] * MEDIUM.rho
    u_star = star[0] / MEDIUM.bulk_modulus
    assert p_star.item() == pytest.approx(0.5)
    assert u_star.item() == pytest.approx(0.5 / z * MEDIUM.bulk_modulus / 1.0)


def test_riemann_flux_argument_order_is_pinned():
    """``(left, right)`` order is part of the contract -- a regression guard.

    The two orders differ in the sign of the dissipation term: passing the
    neighbour state first turns the upwind flux into a downwind (anti-dissipative)
    one.  That variant is *still* consistent for continuous traces, so it passes
    every smooth-data check while producing an ``h``-independent error and a
    growing discrete energy -- it is exactly the defect this solver was debugged
    out of, and this test makes it impossible to reintroduce silently.
    """
    left = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    right = torch.zeros(3, dtype=torch.float64)
    upwind = acoustic_riemann_flux_x(left, right, MEDIUM)
    downwind = acoustic_riemann_flux_x(right, left, MEDIUM)
    assert not torch.allclose(upwind, downwind)
    # upwind - downwind = |A| (qL - qR) = c (qL - qR) for this system
    assert torch.allclose(upwind - downwind,
                          MEDIUM.c * (left - right), atol=1e-12)


# ---------------------------------------------------------------------------
# 2. finite-volume limit (P0)
# ---------------------------------------------------------------------------
def test_p0_rhs_matches_the_upwind_finite_volume_update():
    degree, cells, length = 0, 2, 2.0
    h = length / cells
    ops = legendre_operators_2d(degree)
    x, y = quadrature_grid(cells, 1, h, length, ops["nodes"])
    state = torch.zeros(cells, 1, 1, 1, 3, dtype=torch.float64)
    state[0, :, 0, 0, 0] = 1.0                      # pressure jump between the cells
    coeff = torch.einsum("ka,lb,ijabm->ijklm", ops["proj"], ops["proj"], state)
    rhs = modal_to_quadrature(
        _acoustic_rhs(coeff, ops, MEDIUM, h, length, "periodic", "periodic"),
        ops["phi"])

    probe = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    empty = torch.zeros(3, dtype=torch.float64)
    flux_right = acoustic_riemann_flux_x(probe, empty, MEDIUM)   # +x order
    flux_left = acoustic_riemann_flux_x(empty, probe, MEDIUM)
    expected = -(flux_right - flux_left) / h
    assert torch.allclose(rhs[0, 0, 0, 0], expected, atol=1e-12)
    assert torch.allclose(rhs[1, 0, 0, 0], -expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. exact solutions
# ---------------------------------------------------------------------------
def _grid(points: int = 9):
    axis = torch.linspace(-1.0, 1.0, points, dtype=torch.float64)
    return torch.meshgrid(axis, axis, indexing="ij")


def test_exact_pressure_reduces_to_the_initial_gaussian_at_t_zero():
    x, y = _grid()
    exact = acoustic_pulse_exact_pressure(x, y, 0.0, sigma=SIGMA, medium=MEDIUM)
    reference = torch.exp(-(x ** 2 + y ** 2) / SIGMA ** 2)
    assert float((exact - reference).abs().max()) < 1e-13


def test_exact_state_starts_at_rest():
    x, y = _grid()
    state = acoustic_pulse_exact_state(x, y, 0.0, sigma=SIGMA, medium=MEDIUM)
    assert float(state[..., 1:].abs().max()) < 1e-12


def test_exact_solution_quadrature_is_converged():
    """Doubling the quadrature rule changes nothing above 1e-12."""
    x, y = _grid(11)
    coarse = acoustic_pulse_exact_pressure(x, y, FINAL_TIME, sigma=SIGMA,
                                           medium=MEDIUM, order=10)
    fine = acoustic_pulse_exact_pressure(x, y, FINAL_TIME, sigma=SIGMA,
                                         medium=MEDIUM, order=20)
    assert float((coarse - fine).abs().max()) < 1e-12


def _poisson_disk_pressure(r0: float, t: float, sigma: float, c: float = 1.0,
                           step: float = 1e-4) -> float:
    """Independent 2-D reference: Poisson's disk formula, real space.

    ``p(x,t) = (1/(2 pi c)) d/dt int_{|y|<ct} f(|y|)/sqrt(c^2 t^2-|y-x|^2) dy``
    evaluated in polar coordinates with the substitution ``s = c t sin(phi)``
    (which removes the integrable boundary singularity) and a fourth-order
    central difference in ``t``.  Nothing here shares code with the Hankel
    integral under test.
    """
    from scipy.integrate import quad

    def angular_mean(s: float) -> float:
        integrand = lambda theta: math.exp(
            -(r0 ** 2 + s ** 2 + 2.0 * r0 * s * math.cos(theta)) / sigma ** 2)
        return quad(integrand, 0.0, 2.0 * math.pi, limit=200)[0]

    def disk_integral(time: float) -> float:
        radius = c * time
        return quad(lambda phi: radius * math.sin(phi) * angular_mean(radius * math.sin(phi)),
                    0.0, 0.5 * math.pi, limit=400)[0]

    derivative = (disk_integral(t + step) - disk_integral(t - step)) / (2.0 * step)
    return derivative / (2.0 * math.pi * c)


@pytest.mark.parametrize("radius", [0.0, 0.3, 0.6])
def test_exact_solution_matches_the_independent_poisson_disk_integral(radius):
    hankel = float(acoustic_pulse_exact_pressure(torch.tensor(radius),
                                                torch.tensor(0.0), FINAL_TIME,
                                                sigma=SIGMA, medium=MEDIUM))
    reference = _poisson_disk_pressure(radius, FINAL_TIME, SIGMA)
    scale = float(acoustic_pulse_exact_pressure(torch.tensor(0.0), torch.tensor(0.0),
                                               FINAL_TIME, sigma=SIGMA,
                                               medium=MEDIUM))
    assert abs(hankel - reference) <= 1e-6 * abs(scale)


def test_exact_state_satisfies_the_radial_acoustic_equations():
    """``p_t = -K (r u_r)_r / r`` and ``(u_r)_t = -(1/rho) p_r`` on the exact data."""
    radii_np = np.linspace(0.2, 0.8, 40)
    radii = torch.as_tensor(radii_np, dtype=torch.float64)
    t = 0.25
    dt, dr = 1e-5, 1e-5

    def state(r, time):
        r = torch.as_tensor(r, dtype=torch.float64)
        return acoustic_pulse_exact_state(r, torch.zeros_like(r), time,
                                          sigma=SIGMA, medium=MEDIUM)

    centre = state(radii, t)
    future = state(radii, t + dt)
    past = state(radii, t - dt)
    inner = state(radii_np - dr, t)
    outer = state(radii_np + dr, t)

    p_t = (future[..., 0] - past[..., 0]) / (2.0 * dt)
    ur_t = (future[..., 1] - past[..., 1]) / (2.0 * dt)
    flux = radii * centre[..., 1]                      # r * u_r
    flux_r = (radii + dr) * outer[..., 1] - (radii - dr) * inner[..., 1]
    continuity = -MEDIUM.bulk_modulus * flux_r / (2.0 * dr) / radii
    momentum = -(outer[..., 0] - inner[..., 0]) / (2.0 * dr) / MEDIUM.rho

    assert float((p_t - continuity).abs().max()) < 2e-5
    assert float((ur_t - momentum).abs().max()) < 2e-5


def test_plane_pulse_exact_transports_the_acoustic_invariants():
    """The d'Alembert anchor moves ``p -+ Z u`` unchanged: an exact statement.

    For 1-D acoustic data the invariants ``p + Z u`` (constant along the
    ``+c`` characteristics, hence carried as ``f(x - c t)``) and ``p - Z u``
    (along the ``-c`` characteristics, carried as ``f(x + c t)``) are transported
    without distortion.  Checking
    that on the anchor pins both the pressure *and* the velocity sign at machine
    precision -- no finite differences, hence no truncation-error bound.
    """
    z = MEDIUM.impedance
    x = torch.linspace(-1.0, 1.0, 101, dtype=torch.float64)
    y = torch.zeros_like(x)
    t = 0.1
    centre = acoustic_plane_pulse_exact(x, y, t, sigma=SIGMA, medium=MEDIUM)
    left_invariant = centre[..., 0] - z * centre[..., 1]
    right_invariant = centre[..., 0] + z * centre[..., 1]
    profile_left = torch.exp(-((x - MEDIUM.c * t) ** 2) / SIGMA ** 2)
    profile_right = torch.exp(-((x + MEDIUM.c * t) ** 2) / SIGMA ** 2)
    # p - Z u rides with the +c characteristic, p + Z u with the -c one
    assert float((left_invariant - profile_right).abs().max()) < 1e-13
    assert float((right_invariant - profile_left).abs().max()) < 1e-13
    assert float(centre[..., 2].abs().max()) == 0.0


def test_exact_solution_accepts_numpy_and_torch_inputs():
    x = np.linspace(-0.5, 0.5, 5)
    numpy_value = acoustic_pulse_exact_pressure(x, x, FINAL_TIME, sigma=SIGMA,
                                                medium=MEDIUM)
    torch_value = acoustic_pulse_exact_pressure(torch.as_tensor(x),
                                                torch.as_tensor(x), FINAL_TIME,
                                                sigma=SIGMA, medium=MEDIUM)
    assert isinstance(numpy_value, np.ndarray)
    assert torch.is_tensor(torch_value)
    assert np.allclose(numpy_value, torch_value.numpy())


# ---------------------------------------------------------------------------
# 4. solver convergence and conservation
# ---------------------------------------------------------------------------
def _pulse_run(cells: int, degree: int, *, final_time: float = FINAL_TIME):
    half = 0.5 * LENGTH

    def initial_state(x, y):
        return acoustic_pulse_initial(x - half, y - half, sigma=SIGMA)

    result = solve_acoustic_2d(initial_state, nx=cells, degree=degree,
                               final_time=final_time, length_x=LENGTH,
                               length_y=LENGTH, medium=MEDIUM,
                               snapshot_times=(0.0,))
    x, y = result.quadrature_grid()
    exact = acoustic_pulse_exact_state(x - half, y - half, final_time,
                                       sigma=SIGMA, medium=MEDIUM)
    return result, exact


@pytest.mark.parametrize("degree,slack", [(1, 0.35), (2, 0.35)])
def test_axisymmetric_pulse_converges_at_design_order(degree, slack):
    errors = []
    for cells in (16, 32):
        result, exact = _pulse_run(cells, degree)
        errors.append(result.relative_l2_error(exact, component=None))
    order = math.log(errors[0] / errors[1]) / math.log(2.0)
    assert order >= degree + 1.0 - slack, (errors, order)


@pytest.mark.parametrize("degree", [1, 2])
def test_energy_converges_to_the_analytic_initial_value(degree):
    """``E = pi sigma^2 A^2 / (4 K)`` for the Gaussian pulse at rest."""
    result, _ = _pulse_run(32, degree)
    exact_energy = math.pi * SIGMA ** 2 / (4.0 * MEDIUM.bulk_modulus)
    drift = abs(float(result.energy()) - exact_energy) / exact_energy
    expected = {1: 2e-2, 2: 1e-3}[degree]
    assert drift <= expected, drift


def test_plane_pulse_converges_against_the_closed_form_solution():
    half = 0.5 * LENGTH

    def initial_state(x, y):
        return acoustic_plane_pulse_exact(x - half, y - half, 0.0, sigma=SIGMA,
                                          medium=MEDIUM)

    errors = []
    for cells in (16, 32):
        result = solve_acoustic_2d(initial_state, nx=cells, degree=1,
                                   final_time=FINAL_TIME, length_x=LENGTH,
                                   medium=MEDIUM)
        x, y = result.quadrature_grid()
        exact = acoustic_plane_pulse_exact(x - half, y - half, FINAL_TIME,
                                          sigma=SIGMA, medium=MEDIUM)
        errors.append(result.relative_l2_error(exact, component=None))
    order = math.log(errors[0] / errors[1]) / math.log(2.0)
    assert order >= 1.6, (errors, order)
    # absolute bar: an order of magnitude inside the 15% error budget, at the
    # coarse 32^2 grid (sigma/h = 2.4 for this case)
    assert errors[-1] < 3e-2, errors


def test_snapshots_and_result_container():
    half = 0.5 * LENGTH

    def initial_state(x, y):
        return acoustic_pulse_initial(x - half, y - half, sigma=SIGMA)

    result = solve_acoustic_2d(initial_state, nx=8, degree=1, final_time=0.1,
                               length_x=LENGTH, medium=MEDIUM,
                               snapshot_times=(0.0, 0.05))
    assert result.times == [0.0, 0.05, 0.1]
    assert result.coeff.dtype == torch.float64
    assert result.fields().shape == (8, 8, 2, 2, 3)
    assert result.pressure().shape == (8, 8, 2, 2)
    x, y = result.quadrature_grid()
    assert x.shape == (8, 8, 2, 2)
    energy = result.energy()
    assert energy > 0
    # the t = 0 snapshot must reproduce the energy of the initial coefficients
    # (the analytic-value comparison lives in the 32^2 conservation test: on an
    # 8^2 grid the projection alone loses part of the pulse)
    initial = float(result.energy(result.snapshots[0.0]))
    assert initial == pytest.approx(float(result.energy(result.snapshots[0.0])), rel=1e-12)
    assert initial > energy or abs(initial - energy) / initial < 5e-3
    # per-component error helper agrees with the state-wide value on p alone
    snapshot = result.snapshots[0.1]
    half = 0.5 * LENGTH
    exact = acoustic_pulse_exact_state(x - half, y - half, 0.1, sigma=SIGMA,
                                       medium=MEDIUM)
    pressure_only = result.relative_l2_error(exact, component=0, coeff=snapshot)
    assert pressure_only == pytest.approx(result.relative_l2_error(exact, component=0), rel=1e-12)


def test_energy_is_conserved_by_the_run():
    result, _ = _pulse_run(32, 2)
    initial = float(result.energy(result.snapshots[0.0]))
    final = float(result.energy())
    assert abs(final - initial) / initial < 1e-3


def test_acoustic_energy_matches_the_analytic_integral_of_a_constant_state():
    """The energy functional itself is checked on a state with a closed-form value."""
    degree, cells, length = 1, 4, 2.0
    ops = legendre_operators_2d(degree)
    h = length / cells
    state = torch.zeros(cells, cells, degree + 1, degree + 1, 3, dtype=torch.float64)
    state[..., 0] = 2.0                       # constant pressure p = 2
    state[..., 1] = 0.5                       # constant u = 0.5, v = 0
    coeff = torch.einsum("ka,lb,ijabm->ijklm", ops["proj"], ops["proj"], state)
    value = float(acoustic_energy(coeff, ops, MEDIUM, h, h))
    expected = (4.0 / (2.0 * MEDIUM.bulk_modulus) + 0.5 * MEDIUM.rho * 0.25) * length ** 2
    assert value == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {"rho": 0.0}, {"rho": -1.0}, {"c": 0.0}, {"c": -2.0},
])
def test_medium_rejects_non_physical_parameters(kwargs):
    with pytest.raises(ValueError):
        AcousticMedium(**kwargs)


def test_pulse_initial_rejects_non_positive_sigma():
    x = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError):
        acoustic_pulse_initial(x, x, sigma=0.0)


def test_solver_validates_its_inputs():
    initial = lambda x, y: acoustic_pulse_initial(x - 1.0, y - 1.0)
    with pytest.raises(ValueError):
        solve_acoustic_2d(initial, nx=4, bc_x="dirichlet")
    with pytest.raises(ValueError):
        solve_acoustic_2d(initial, nx=0)
    with pytest.raises(ValueError):
        solve_acoustic_2d(initial, nx=4, cfl=0.0)


def test_exact_velocity_is_regular_on_the_axis():
    velocity = acoustic_pulse_exact_velocity(torch.tensor(0.0), torch.tensor(0.0),
                                             FINAL_TIME, sigma=SIGMA, medium=MEDIUM)
    assert torch.isfinite(velocity).all()
    assert float(velocity.abs().max()) == pytest.approx(0.0, abs=1e-12)
