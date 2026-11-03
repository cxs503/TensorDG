"""Analytic verification of the Euler interface-flux kernels.

The end-to-end Riemann benchmarks (``tests/test_euler_sod_1d.py``,
``tests/test_lax_1d.py``) validate the *solver*; these tests validate the
*flux kernel* itself against properties that hold exactly in arithmetic:

* consistency ``F(U, U) = f(U)``,
* mirror symmetry ``R F(U_L, U_R) = F(M U_R, M U_L)``,
* isolated-contact exactness (the property separating HLLC from Rusanov),
* agreement with the exact Riemann state sampled at the interface
  (Toro 2009, ch. 4) within a stated tolerance, including an attribution
  test showing that the residual is caused by Davis' wave-speed estimates.

Evidence and the reproduction command: ``docs/euler_flux_kernel_verification.md``
and ``benchmarks/verify_flux_properties.py``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.riemann_exact import sample as sample_exact
from benchmarks.verify_flux_properties import (
    REFLECTION,
    STANDARD_CASES,
    exact_wave_speeds,
    hllc_flux_with_speeds,
    mirror,
    pair,
    physical_flux,
    relative_error_scale,
    rusanov_flux,
    state,
)
from tensordg.euler_1d import hllc_flux

DTYPE = torch.float64

CONSISTENCY_STATES = [
    (1.0, 0.0, 1.0),
    (1.0, 3.0, 0.1),
    (0.125, 0.0, 0.1),
    (5.99924, 19.5975, 460.894),
    (1.0, -1.5, 2.5),
    (0.001, 0.0, 1e-6),
]


def exact_interface_flux(left, right):
    sampled = sample_exact(np.array([0.5]), 1.0, left, right)
    conserved = state(float(sampled["density"][0]), float(sampled["velocity"][0]),
                      float(sampled["pressure"][0]))
    return physical_flux(conserved)


@pytest.mark.parametrize("case", CONSISTENCY_STATES)
def test_hllc_is_consistent(case):
    """F(U, U) must equal the physical flux of U to round-off."""
    u = state(*case)
    exact = physical_flux(u)
    numerical = hllc_flux(u, u)
    assert torch.allclose(numerical, exact, rtol=1e-12, atol=1e-12 * max(1.0, float(exact.abs().max())))


@pytest.mark.parametrize("case", STANDARD_CASES)
def test_hllc_is_mirror_symmetric(case):
    """x -> -x reflection: R F(U_L, U_R) = F(M U_R, M U_L) exactly.

    ``M = diag(1, -1, 1)`` mirrors the velocity, ``R = diag(-1, 1, -1)`` is the
    corresponding flux reflection.
    """
    _, left, right, _ = case
    ul, ur = pair(left, right)
    lhs = REFLECTION * hllc_flux(ul, ur)
    rhs = hllc_flux(mirror(ur), mirror(ul))
    assert relative_error_scale(rhs, lhs) < 1e-12


@pytest.mark.parametrize("case", [c for c in STANDARD_CASES if abs(c[1][1] - c[2][1]) < 1e-15
                                  and abs(c[1][2] - c[2][2]) < 1e-15])
def test_hllc_resolves_an_isolated_contact_exactly(case):
    """Pure contact: the flux must equal the upwind physical flux to round-off."""
    _, left, right, _ = case
    ul, ur = pair(left, right)
    upwind = ul if left[1] > 0.0 else ur
    exact = physical_flux(upwind)
    numerical = hllc_flux(ul, ur)
    assert float((numerical - exact).abs().max()) < 1e-14


def test_rusanov_diffuses_an_isolated_contact():
    """Control: an HLL/Rusanov flux must smear the same contact (> 50% error)."""
    left, right = (1.0, 0.5, 1.0), (0.125, 0.5, 1.0)
    ul, ur = pair(left, right)
    exact = physical_flux(ul)
    error = relative_error_scale(rusanov_flux(ul, ur), exact)
    assert error > 0.5, error


def test_hllc_agrees_with_the_exact_riemann_state_at_the_interface():
    """Davis wave speeds: every standard case within 30%, median within 15%."""
    scaled = []
    for _, left, right, _ in STANDARD_CASES:
        exact = exact_interface_flux(left, right)
        ul, ur = pair(left, right)
        scale = max(float(physical_flux(ul).abs().max()),
                    float(physical_flux(ur).abs().max()),
                    float(exact.abs().max()))
        scaled.append(float((hllc_flux(ul, ur) - exact).abs().max()) / scale)
    assert max(scaled) < 0.30, scaled
    assert float(np.median(scaled)) < 0.15, scaled


def test_exact_wave_speeds_remove_the_shock_residual():
    """Attribution: the residual on isolated shocks comes from Davis' estimates.

    With the exact outer wave speeds the same HLLC algebra reproduces the exact
    interface flux of a strong isolated shock to round-off.
    """
    cases = {name: (left, right) for name, left, right, _ in STANDARD_CASES}
    left, right = cases["strong_shock"]
    ul, ur = pair(left, right)
    exact = exact_interface_flux(left, right)
    scale = float(exact.abs().max())
    speed_l, speed_r = exact_wave_speeds(left, right)
    exact_speeds = hllc_flux_with_speeds(ul, ur, speed_l, speed_r)
    davis = hllc_flux(ul, ur)
    assert float((exact_speeds - exact).abs().max()) / scale < 1e-12
    assert float((davis - exact).abs().max()) / scale > 0.1


def test_exact_wave_speeds_agree_with_rankine_hugoniot():
    """The wave-speed helper must return the shock speed and satisfy the RH jump.

    At Mach 5 the whole fan moves downstream (u_l - a_l > 0), so the right-going
    wave speed is the shock speed and the Rankine-Hugoniot relation
    ``f(U_post) - f(U_pre) = S (U_post - U_pre)`` must hold to round-off.
    """
    gamma = 1.4
    mach = 5.0
    ratio = (2.0 * gamma * mach**2 - (gamma - 1.0)) / (gamma + 1.0)
    rho_l = (gamma + 1.0) * mach**2 / ((gamma - 1.0) * mach**2 + 2.0)
    # mach is the shock Mach number, i.e. shock speed / a_r with a_r = 1 here.
    shock_speed = mach * np.sqrt(gamma)
    u_l = shock_speed * (1.0 - 1.0 / rho_l)
    speed_l, speed_r = exact_wave_speeds((rho_l, u_l, ratio), (1.0, 0.0, 1.0))
    assert abs(speed_r - shock_speed) < 1e-9
    assert speed_l < shock_speed
    ul, ur = pair((rho_l, u_l, ratio), (1.0, 0.0, 1.0))
    jump = ul - ur
    flux_jump = physical_flux(ul) - physical_flux(ur)
    assert torch.allclose(flux_jump, speed_r * jump, rtol=1e-10, atol=1e-10)


def test_hllc_is_finite_for_a_near_vacuum_star_region():
    """The 1-2-3 problem (p* ~ 2e-3) must not produce NaN or Inf."""
    left, right = (1.0, -2.0, 0.4), (1.0, 2.0, 0.4)
    ul, ur = pair(left, right)
    flux = hllc_flux(ul, ur)
    assert torch.isfinite(flux).all()
    # symmetric data -> zero mass flux at the interface
    assert abs(float(flux[0, 0])) < 1e-12
    assert abs(float(flux[0, 2])) < 1e-12


def test_rankine_hugoniot_shock_flux_is_reproduced_within_three_percent():
    """Isolated shock: the HLLC flux matches the post-shock flux within 3%."""
    gamma = 1.4
    mach = 2.0
    ratio = (2.0 * gamma * mach**2 - (gamma - 1.0)) / (gamma + 1.0)
    rho_l = (gamma + 1.0) * mach**2 / ((gamma - 1.0) * mach**2 + 2.0)
    u_l = mach * (1.0 - 1.0 / rho_l)
    ul, ur = pair((rho_l, u_l, ratio), (1.0, 0.0, 1.0))
    exact = physical_flux(ul)
    error = float((hllc_flux(ul, ur) - exact).abs().max()) / float(exact.abs().max())
    assert error < 0.03, error