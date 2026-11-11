"""Verification tests for the 2-D radial (circular-basin) Thacker solution.

The closed form is certified *independently* of its own algebra:

* the radial shallow-water PDE residual under 4th-order finite differences,
* agreement with a DOP853 integration of the closed ODE reduction,
* the exact volume invariant ``pi h0 L^2 / 2`` that holds for every ``t``,
* the closed-form shoreline root and the zero initial velocity,
* and consumability by ``swe_2d`` (the analytic initial state projects to a
  finite, non-negative-depth DG state), which is what makes it usable as the
  reference for a 2-D well-balanced / wet-dry run.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordg import swe_exact as se

H0 = 1.0
L = 1.0
A = 0.2
G = se.GRAVITY
OMEGA = se.thacker_radial_params(H0, L, G)
PERIOD = se.thacker_radial_period(H0, L, G)


def _fields(r, t):
    return se.thacker_radial_basin(r, t, h0=H0, amplitude=A, half_width=L,
                                   gravity=G, clip=False)


def _residual(r, t, eps):
    from benchmarks.verify_thacker_radial_2d import radial_residual
    return radial_residual(r, t, eps)


def test_omega_and_period_follow_the_linear_radial_theory():
    assert OMEGA == pytest.approx(math.sqrt(8.0 * G * H0) / L, rel=1e-14)
    assert PERIOD == pytest.approx(2.0 * math.pi / OMEGA, rel=1e-14)
    # The 1-D planar basin uses sqrt(2 g h0)/L; the radial factor 8 must differ.
    assert OMEGA / se._thacker_params(H0, L, G) == pytest.approx(2.0, rel=1e-14)


def test_closed_form_satisfies_the_radial_shallow_water_equations():
    """Continuity and radial momentum residuals are at truncation-error level."""
    for r, t in ((0.25, 0.05), (0.55, 0.12), (0.70, 0.30), (0.40, 0.45)):
        cont, mom = _residual(r, t, 2.5e-3)
        assert cont < 1e-7, (r, t, cont)
        assert mom < 1e-7, (r, t, mom)


def test_residual_is_fourth_order_in_the_step():
    coarse = _residual(0.5, 0.15, 2.0e-2)
    fine = _residual(0.5, 0.15, 1.0e-2)
    for a, b in zip(coarse, fine):
        assert math.log2(a / b) > 3.0


def test_ode_reduction_reproduces_the_closed_form():
    """DOP853 on the reduced ODE system (an independent derivation) matches."""
    from benchmarks.verify_thacker_radial_2d import ode_reference
    times = np.linspace(0.0, PERIOD, 17)
    closed = np.array([_fields(np.array(0.0), t)[0] for t in times])
    ode = ode_reference(times)
    assert np.max(np.abs(ode - closed)) / np.max(np.abs(closed)) < 1e-10


def test_volume_is_the_exact_constant_pi_h0_L2_over_2():
    exact = math.pi * H0 * L ** 2 / 2.0
    for frac in (0.0, 0.17, 0.29, 0.5, 0.73, 0.91):
        t = frac * PERIOD
        r_s = se.thacker_radial_shoreline(t, H0, A, L, G)
        rr = np.linspace(0.0, r_s, 4001)
        h = se.thacker_radial_basin(rr, t, h0=H0, amplitude=A, half_width=L)[0]
        integrand = 2.0 * np.pi * rr * h
        vol = np.trapezoid(integrand, rr) if hasattr(np, "trapezoid") else np.trapz(integrand, rr)
        assert vol / exact == pytest.approx(1.0, abs=1e-6)


def test_shoreline_is_the_root_of_the_depth_and_velocity_starts_at_rest():
    for frac in (0.0, 0.2, 0.45, 0.8):
        t = frac * PERIOD
        r_s = se.thacker_radial_shoreline(t, H0, A, L, G)
        h, _, _ = _fields(np.array(r_s), t)
        assert abs(h) < 1e-12
        assert r_s > 0.8 * L
    u0 = se.thacker_radial_velocity(np.array(0.3), np.array(-0.2), 0.0, H0, A, L)
    assert np.max(np.abs(u0)) < 1e-15
    # u_r is linear in r and purely radial.
    t = 0.1 * PERIOD
    u1, v1 = se.thacker_radial_velocity(np.array(0.4), np.array(0.0), t, H0, A, L)
    u2, v2 = se.thacker_radial_velocity(np.array(0.0), np.array(0.4), t, H0, A, L)
    assert u1 == pytest.approx(v2, rel=1e-13)
    assert v1 == pytest.approx(u2, rel=1e-13)


def test_swe_2d_accepts_the_analytic_initial_state():
    """The reference is consumable by the DG solver (final_time = 0 => no steps)."""
    import torch

    from tensordg.swe_2d import solve_swe_2d

    def ic_h(x, y):
        rr = torch.sqrt(x * x + y * y)
        h, _, _ = se.thacker_radial_basin(rr.numpy(), 0.0, h0=H0, amplitude=A, half_width=L)
        return torch.as_tensor(h)

    def zero(x, y):
        uu, _ = se.thacker_radial_velocity(x.numpy(), y.numpy(), 0.0, H0, A, L)
        return torch.as_tensor(uu)

    bed = lambda x, y: torch.as_tensor(  # noqa: E731
        se.thacker_radial_bed(np.sqrt((x * x + y * y).numpy()), H0, L))
    out = solve_swe_2d(ic_h, zero, zero, bed, nx=8, ny=8, degree=1,
                       final_time=0.0, domain=((-1.45, 1.45), (-1.45, 1.45)),
                       dtype=torch.float64, well_balanced=True)
    coeff = out[2]
    assert torch.isfinite(coeff).all()
    assert float(coeff[:, :, 0, 0, 0].min()) >= 0.0
