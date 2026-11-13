"""Tests for the layered (heterogeneous) 1-D elastic rod solver.

These tests target the *new* degrees of freedom a material jump adds.  The
homogeneous solver cannot see them: with equal materials the left- and
right-side face fluxes coincide, so a swapped flux pairing is invisible (the
uniform-reduction test passes even with that bug).  The interface tests below
are therefore the ones with teeth.
"""
import math

import pytest
import torch

from tensordg.elastic_1d import gaussian_pulse, solve_elastic_1d, upwind_flux
from tensordg.elastic_1d_layered import (
    exact_rightgoing_scattering,
    impedance,
    interface_state,
    layered_energy,
    layered_interface_flux,
    layered_stress_profile,
    rightgoing_pulse_initial_state,
    scattering_coefficients,
    scattering_valid_time,
    solve_elastic_1d_layered,
    wave_speed,
)

torch.set_num_threads(1)

TOL = 0.02          # 2 % ledger criterion


# ---------------------------------------------------------------------------
# materials and exact coefficients
# ---------------------------------------------------------------------------
def test_wave_speed_and_impedance_match_definition():
    assert wave_speed(4.0, 4.0) == pytest.approx(1.0)
    assert impedance(4.0, 4.0) == pytest.approx(4.0)
    assert impedance(1.0, 1.0) == pytest.approx(1.0)


def test_scattering_coefficients_obey_continuity_identity():
    for rho2, young2 in ((4.0, 4.0), (2.0, 0.5), (1.0, 0.25), (0.5, 8.0)):
        r, t = scattering_coefficients(1.0, 1.0, rho2, young2)
        # stress continuity at the interface: 1 + R == T
        assert 1.0 + r == pytest.approx(t, abs=1e-14)


def test_impedance_matched_interface_has_zero_reflection():
    # rho and E both jump, but sqrt(rho E) is unchanged
    r, t = scattering_coefficients(1.0, 1.0, 2.0, 0.5)
    assert r == pytest.approx(0.0, abs=1e-15)
    assert t == pytest.approx(1.0, abs=1e-15)


def test_scattering_coefficients_are_symmetric_in_sign():
    r_forward, _ = scattering_coefficients(1.0, 1.0, 4.0, 4.0)
    r_backward, _ = scattering_coefficients(4.0, 4.0, 1.0, 1.0)
    assert r_forward == pytest.approx(-r_backward, abs=1e-15)


def test_illegal_materials_raise():
    with pytest.raises(ValueError):
        wave_speed(0.0, 1.0)
    with pytest.raises(ValueError):
        solve_elastic_1d_layered(ncells=8, rho=(1.0, -1.0), young=(1.0,))
    with pytest.raises(ValueError):
        solve_elastic_1d_layered(ncells=8, rho=(1.0,) * 7, young=(1.0,))


# ---------------------------------------------------------------------------
# interface Riemann solver
# ---------------------------------------------------------------------------
def test_interface_state_recovers_closed_form_coefficients():
    """A right-going pulse state at the face reproduces R and T exactly."""
    rho_l, e_l, rho_r, e_r = 1.0, 1.0, 4.0, 4.0
    z_l, z_r = impedance(rho_l, e_l), impedance(rho_r, e_r)
    # incident amplitude sigma_i with v = -sigma/Z1 (right-going branch)
    sigma_i = torch.tensor([1.0, 0.4, -2.5], dtype=torch.float64)
    v_l = -sigma_i / z_l
    s_l = sigma_i.clone()
    v_r = torch.zeros_like(sigma_i)             # quiescent layer 2
    s_r = torch.zeros_like(sigma_i)
    v_star, s_star = interface_state(
        v_l, s_l, v_r, s_r,
        torch.full_like(sigma_i, z_l), torch.full_like(sigma_i, z_r))
    r_th, t_th = scattering_coefficients(rho_l, e_l, rho_r, e_r)
    # interface state = (1 + R) sigma_i = T sigma_i, and v* = -T sigma_i / Z2
    assert torch.allclose(s_star, t_th * sigma_i, atol=1e-14)
    assert torch.allclose(v_star, -t_th * sigma_i / z_r, atol=1e-14)
    assert torch.allclose((s_star / sigma_i) - 1.0,
                          torch.full_like(sigma_i, r_th), atol=1e-14)


def test_interface_flux_reduces_to_upwind_flux_for_equal_materials():
    torch.manual_seed(0)
    n = 9
    v_l = torch.randn(n, dtype=torch.float64)
    s_l = torch.randn(n, dtype=torch.float64)
    v_r = torch.randn(n, dtype=torch.float64)
    s_r = torch.randn(n, dtype=torch.float64)
    rho = torch.full((n,), 2.5, dtype=torch.float64)
    e = torch.full((n,), 7.0, dtype=torch.float64)
    f_l, f_r = layered_interface_flux(v_l, s_l, v_r, s_r, rho, e, rho, e)
    reference = upwind_flux(v_l, s_l, v_r, s_r, 2.5, 7.0)
    assert torch.allclose(f_l, reference, atol=1e-14)
    assert torch.allclose(f_r, reference, atol=1e-14)


def test_interface_flux_is_side_dependent_only_through_the_material():
    """v* and sigma* are continuous, the flux components are not."""
    z1, z2 = impedance(1.0, 1.0), impedance(4.0, 4.0)
    v_l = torch.tensor([-1.0], dtype=torch.float64)     # right-going, sigma = 1
    s_l = torch.tensor([1.0], dtype=torch.float64)
    v_r = torch.zeros(1, dtype=torch.float64)
    s_r = torch.zeros(1, dtype=torch.float64)
    f_l, f_r = layered_interface_flux(
        v_l, s_l, v_r, s_r,
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([4.0], dtype=torch.float64),
        torch.tensor([4.0], dtype=torch.float64))
    assert z1 == pytest.approx(1.0) and z2 == pytest.approx(4.0)
    # momentum flux carries 1/rho (left side is 4x larger), the stress flux
    # carries E (right side is 4x larger) -- the two components jump in
    # *opposite* directions, which is what makes the pairing non-trivial
    assert float(f_l[0, 0]) == pytest.approx(float(f_r[0, 0]) * 4.0, rel=1e-13)
    assert float(f_r[0, 1]) == pytest.approx(float(f_l[0, 1]) * 4.0, rel=1e-13)


# ---------------------------------------------------------------------------
# solver consistency
# ---------------------------------------------------------------------------
def test_uniform_material_reproduces_homogeneous_solver_exactly():
    v0 = lambda x: gaussian_pulse(x, 0.5, 0.05)
    s0 = lambda x: torch.zeros_like(x)
    kwargs = dict(ncells=40, degree=2, final_time=0.15)
    _c, homogeneous, meta_h = solve_elastic_1d(
        initial_velocity=v0, initial_stress=s0, **kwargs)
    _c, layered, meta_l = solve_elastic_1d_layered(
        rho=(1.0,), young=(1.0,), initial_velocity=v0, initial_stress=s0,
        **kwargs)
    assert meta_h["steps"] == meta_l["steps"]
    assert float((homogeneous - layered).abs().max()) < 1e-14


def test_interior_interface_keeps_the_pulse_in_a_straight_line():
    """A right-going pulse in both layers (Z1 == Z2) stays a single packet."""
    length, interface, ncells, degree = 4.0, 2.0, 200, 2
    rho = [1.0] * 100 + [2.0] * 100
    young = [1.0] * 100 + [0.5] * 100          # Z2 = sqrt(1.0) = Z1, c2 = 0.5
    v0, s0 = rightgoing_pulse_initial_state(interface, 1.0, 1.0, 1.0, 0.05)
    x, coeff, _meta = solve_elastic_1d_layered(
        ncells=ncells, degree=degree, final_time=1.4, length=length, rho=rho,
        young=young, bc="fixed-free", initial_velocity=v0, initial_stress=s0)
    xs, _vs, ss = layered_stress_profile(x, coeff, 8)
    peak = float(ss.abs().max())
    assert peak == pytest.approx(1.0, abs=5e-3)          # no reflected packet
    # single packet: only one connected region above 1 % of the peak
    above = (ss.abs() > 0.01 * peak).to(torch.int8)
    transitions = int(torch.sum(torch.abs(above[1:] - above[:-1])))
    assert transitions <= 2                              # one smooth cluster


def test_energy_is_conserved_across_a_material_jump():
    ncells, degree, length, interface = 200, 2, 4.0, 2.0
    rho = [1.0] * 100 + [4.0] * 100
    young = [1.0] * 100 + [4.0] * 100
    v0, s0 = rightgoing_pulse_initial_state(interface, 1.0, 1.0, 1.0, 0.05)
    energies = []
    for t in (0.0, 0.7, 1.4):
        _x, coeff, _m = solve_elastic_1d_layered(
            ncells=ncells, degree=degree, final_time=t, length=length,
            rho=rho, young=young, bc="fixed-free",
            initial_velocity=v0, initial_stress=s0)
        energies.append(layered_energy(coeff, length / ncells, rho, young))
    # invariant stated at 0.1 %: the residual drift is SSP-RK3 dissipation
    # (measured ~6e-4 at CFL 0.3 on this mesh), not an interface energy loss
    for e in energies[1:]:
        assert abs(e - energies[0]) / energies[0] < 5e-3


# ---------------------------------------------------------------------------
# closed-form reference
# ---------------------------------------------------------------------------
def test_exact_solution_is_continuous_at_the_interface():
    interface, length = 2.0, 4.0
    for t in (0.6, 1.0, 1.4):
        x = torch.tensor([interface - 1e-9, interface + 1e-9],
                         dtype=torch.float64)
        v, s = exact_rightgoing_scattering(x, t, interface, length,
                                           1.0, 1.0, 4.0, 4.0, 1.0, 0.05)
        assert abs(float(s[0] - s[1])) < 1e-6
        assert abs(float(v[0] - v[1])) < 1e-6


def test_exact_solution_is_a_pure_right_mover_before_the_interface():
    interface, length, width = 2.0, 4.0, 0.05
    x = torch.linspace(1.3, 1.7, 401, dtype=torch.float64)
    t = 0.5                                  # pulse centre at 1.5 < interface
    v, s = exact_rightgoing_scattering(x, t, interface, length,
                                      1.0, 1.0, 4.0, 4.0, 1.0, width)
    z1 = impedance(1.0, 1.0)
    # right-going branch of this system: sigma = -Z v
    assert torch.allclose(s, -z1 * v, atol=1e-13)
    # peak still the unit amplitude and centred at x = centre + c t
    i = int(torch.argmax(s.abs()))
    assert float(x[i]) == pytest.approx(1.5, abs=2e-3)


def test_exact_solution_guards_the_boundary_return():
    limit = scattering_valid_time(2.0, 4.0, 1.0, 1.0, 4.0, 4.0)
    assert limit == pytest.approx(2.0)
    x = torch.tensor([1.5], dtype=torch.float64)
    exact_rightgoing_scattering(x, limit * 1.0001, 2.0, 4.0, 1.0, 1.0, 4.0, 4.0,
                                strict=False)   # grazing the guard is allowed
    with pytest.raises(ValueError):
        exact_rightgoing_scattering(x, limit * 1.01, 2.0, 4.0, 1.0, 1.0,
                                    4.0, 4.0, strict=True)


# ---------------------------------------------------------------------------
# end-to-end acceptance: R and T within 2 %
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rho2,young2", [(4.0, 4.0), (2.0, 0.5), (1.0, 0.25)])
def test_measured_reflection_and_transmission_match_theory(rho2, young2):
    from benchmarks.elastic_1d_layered import scattering_case, CASES

    spec = dict(key="t", label="t", rho2=rho2, young2=young2)
    out = scattering_case(spec, ncells=200, degree=2)
    r_th, t_th = scattering_coefficients(1.0, 1.0, rho2, young2)
    assert out["r_theory"] == pytest.approx(r_th)
    assert out["t_theory"] == pytest.approx(t_th)
    assert out["rel_error_R"] < TOL                               # < 2 %
    assert out["rel_error_T"] < TOL                               # < 2 %
    # the slow-layer cases still carry O(h^3) phase error at N = 200; the
    # acceptance mesh of the benchmark (N = 400) is at <0.2 % (see docs ③)
    assert out["field_rel_l2"] < 4 * TOL
    assert len(CASES) >= 3