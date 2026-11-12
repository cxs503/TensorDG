"""Verification of the 1-D DG elastodynamics solver against exact bar solutions.

Case pool anchor: CASE-DG-DALEMBERT-ELASTIC-1D
(metric: stress profile vs the d'Alembert two-wave analytic solution, <2%).

Every test compares a *computed* field with a closed-form solution of the
velocity--stress system, not with a stored regression baseline.
"""
import math

import torch

from tensordg.elastic_1d import (
    dalembert_pulse,
    elastic_energy,
    elastic_wave_speed,
    initial_pulse_stress,
    initial_pulse_velocity,
    initial_standing_stress,
    peak_stress,
    relative_error,
    solve_elastic_1d,
    standing_wave,
    stress_profile,
    upwind_flux,
)

RHO = 1.0
E = 1.0
C = elastic_wave_speed(RHO, E)


def _boundary_traces(coeff):
    """(v, sigma) at the left and right domain ends from the modal traces."""
    nmodes = coeff.shape[1]
    left = sum(((-1.0) ** k) * coeff[0, k] for k in range(nmodes))
    right = sum(coeff[-1, k] for k in range(nmodes))
    return left, right


def _pulse(ncells, final_time, degree=2, cfl=0.3):
    centers, coeff, meta = solve_elastic_1d(
        ncells=ncells, degree=degree, final_time=final_time, cfl=cfl,
        bc="periodic")
    x, v, s = stress_profile(centers, coeff, nsample=2 * degree + 3)
    return x, v, s, meta


def test_upwind_flux_recovers_physical_flux_for_a_uniform_state():
    v = torch.tensor([0.4, -0.2])
    s = torch.tensor([1.5, 0.7])
    flux = upwind_flux(v, s, v, s, RHO, E)
    assert torch.allclose(flux[:, 0], -s / RHO)
    assert torch.allclose(flux[:, 1], -E * v)


def test_initial_projection_is_exact_at_zero_final_time():
    sine = lambda x: torch.sin(2.0 * math.pi * x)          # noqa: E731
    centers, coeff, meta = solve_elastic_1d(
        ncells=32, degree=3, final_time=0.0,
        initial_velocity=sine, initial_stress=initial_pulse_stress)
    assert meta["steps"] == 0
    x, v, s = stress_profile(centers, coeff, nsample=7)
    # a P3 projection of sin(2 pi x) is accurate to the projection error scale
    assert relative_error(x, v, sine(x)) < 1e-5
    assert torch.allclose(s, torch.zeros_like(s), atol=1e-12)


def test_periodic_pulse_splits_into_two_waves_with_half_amplitude():
    t = 0.2
    x, v, s, _meta = _pulse(ncells=100, final_time=t)
    # left-going wave at x = 0.5 - c t, right-going at x = 0.5 + c t
    x_left, s_left = peak_stress(x, s, 0.0, 0.5)
    x_right, s_right = peak_stress(x, s, 0.5, 1.0)
    assert abs(x_left - (0.5 - C * t)) < 2.0 / 100
    assert abs(x_right - (0.5 + C * t)) < 2.0 / 100
    # each carries half of the released momentum flux: sigma = +- rho c / 2
    half = 0.5 * RHO * C
    assert abs(abs(s_left) / half - 1.0) < 0.01
    assert abs(abs(s_right) / half - 1.0) < 0.01
    assert s_left * s_right < 0.0            # tension vs compression
    # the velocity field is the sum of the two half-amplitude pulses
    assert relative_error(x, v, 0.5 * (initial_pulse_velocity(x - C * t)
                                       + initial_pulse_velocity(x + C * t))) < 0.01


def test_stress_profile_matches_dalembert_within_two_percent():
    """Acceptance criterion of the case-pool anchor (<2 % on sigma)."""
    t = 0.2
    x, _v, s, _meta = _pulse(ncells=100, final_time=t)
    _v_ex, s_ex = dalembert_pulse(x, t, RHO, E)
    error = relative_error(x, s, s_ex)
    assert error < 0.02, error
    # and the residual should be far from the 2 % bound at this resolution
    assert error < 1e-3, error


def test_degree_two_pulse_converges_at_third_order():
    errors = []
    for n in (50, 100, 200):
        x, _v, s, _meta = _pulse(ncells=n, final_time=0.1)
        _v_ex, s_ex = dalembert_pulse(x, 0.1, RHO, E)
        errors.append(relative_error(x, s, s_ex))
    rate_a = math.log(errors[0] / errors[1], 2.0)
    rate_b = math.log(errors[1] / errors[2], 2.0)
    assert rate_a > 2.7 and rate_b > 2.7, (errors, rate_a, rate_b)


def test_degree_one_pulse_converges_at_second_order():
    errors = []
    for n in (50, 100, 200):
        x, _v, s, _meta = _pulse(ncells=n, final_time=0.1, degree=1)
        _v_ex, s_ex = dalembert_pulse(x, 0.1, RHO, E)
        errors.append(relative_error(x, s, s_ex))
    rate = math.log(errors[0] / errors[2], 2.0) / 2.0
    assert rate > 1.8, (errors, rate)


def test_standing_wave_matches_exact_solution_and_conserves_energy():
    length, amplitude, t = 1.0, 1.0, 0.5
    centers, coeff, meta = solve_elastic_1d(
        ncells=40, degree=2, final_time=t, cfl=0.3, bc="fixed-free",
        length=length)
    x, v, s = stress_profile(centers, coeff, nsample=7)
    v_ex, s_ex, _u = standing_wave(x, t, length, RHO, E, amplitude)
    assert relative_error(x, s, s_ex) < 0.02
    assert relative_error(x, v, v_ex) < 0.02
    # characteristic boundary conditions: v(0) = 0, sigma(L) = 0
    left, right = _boundary_traces(coeff)
    assert abs(float(left[0])) < 1e-6
    assert abs(float(right[1])) < 1e-6
    # energy of the discrete solution equals the analytic eigenmode energy
    k = math.pi / (2.0 * length)
    energy_exact = 0.25 * amplitude ** 2 * k ** 2 * E
    assert abs(elastic_energy(centers, coeff, RHO, E) - energy_exact) / energy_exact < 1e-6


def test_standing_wave_half_period_returns_to_minus_initial_state():
    """At t = 2 pi / (2 omega) the eigenmode is exactly inverted."""
    length, t_half = 1.0, 2.0
    centers, coeff, meta = solve_elastic_1d(
        ncells=20, degree=2, final_time=t_half, cfl=0.3, bc="fixed-free",
        length=length)
    # the time axis must actually be integrated to the requested final time
    assert abs(meta["steps"] * meta["dt"] - t_half) < 1e-12
    x, _v, s = stress_profile(centers, coeff, nsample=7)
    v_ex, s_ex, _u = standing_wave(x, t_half, length, RHO, E)
    # sigma(t = 2 pi / omega) = -sigma(0) for the clamped-free eigenmode
    sigma0_minus = -initial_standing_stress(x, length, E)
    assert relative_error(x, sigma0_minus, s_ex) < 1e-12    # closed-form check
    assert relative_error(x, s, s_ex) < 1e-2


def test_free_surface_and_clamped_end_reflect_correctly():
    """v = 0 at the clamped end; sigma = 0 at the free end (sign of reflection)."""
    length, t = 1.0, 0.5
    centers, coeff, _meta = solve_elastic_1d(
        ncells=60, degree=3, final_time=t, cfl=0.2, bc="fixed-free",
        length=length)
    x, v, s = stress_profile(centers, coeff, nsample=3)
    # boundary traces read straight from the modal coefficients
    left, right = _boundary_traces(coeff)
    assert abs(float(left[0])) < 1e-6, float(left[0])
    assert abs(float(right[1])) < 1e-6, float(right[1])
    v_ex, s_ex, _u = standing_wave(x, t, length, RHO, E)
    assert relative_error(x, s, s_ex) < 0.01


def test_rejects_invalid_configuration():
    for kwargs in ({"degree": -1}, {"cfl": 0.0}, {"bc": "clamped"},
                   {"time_integrator": "euler"}):
        try:
            solve_elastic_1d(ncells=8, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_pulse_initial_conditions_are_stress_free():
    x = torch.linspace(0.0, 1.0, 7)
    assert torch.allclose(initial_pulse_stress(x), torch.zeros_like(x))
    assert torch.allclose(initial_pulse_velocity(x), torch.exp(-((x - 0.5) / 0.05) ** 2))

def test_energy_gate_detects_wrong_characteristic_pairing():
    """Fault injection: the accuracy/energy gate must have teeth.

    Swapping the incoming/outgoing Riemann invariant at the two ends (a
    plausible slip) turns the reflecting boundaries into amplifying ones; the
    discrete energy then explodes instead of staying at the eigenmode value.
    See ``benchmarks/elastic_1d_bc_fault_injection.py``.
    """
    import tensordg.elastic_1d as elastic

    original = elastic._boundary_flux

    def swapped(v_int, s_int, side, rho, young, bc):
        other = "right" if side == "left" else "left"
        return original(v_int, s_int, other, rho, young, bc)

    elastic._boundary_flux = swapped
    try:
        centers, coeff, _meta = solve_elastic_1d(
            ncells=30, degree=2, final_time=0.3, cfl=0.3, bc="fixed-free")
    finally:
        elastic._boundary_flux = original
    k = math.pi / 2.0
    energy_exact = 0.25 * k ** 2
    assert elastic_energy(centers, coeff) > 100.0 * energy_exact
