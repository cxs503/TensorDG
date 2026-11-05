"""Verification tests for Stokes' first problem (unsteady viscous wall shear).

These tests pin the evidence of ``docs/reports/tensordg-ns-stokes-first.md``:
the compressible BR1/SIPG solver reproduces the classic impulsively started
plate solution

    u(y, t) = U erfc(y / (2 sqrt(nu t))),   tau_w = mu U / sqrt(pi nu t)

so the *unsteady wall shear stress* -- the quantity that
``docs/reports/tensordg-ns-dg-2d.md`` §5.6 had to mark as unverified -- now has
a closed-form reference.  They are cheap: the projection tests never advance
time, and the single marched test runs the coarsest useful grid.
"""
import math

import pytest
import torch

import benchmarks.verify_ns_stokes_first as stokes

DTYPE = torch.float64


def _projected(degree, ny):
    op = stokes.grid_operators(degree, (1, ny))
    return op, stokes.project(stokes.exact_initial(), op)


def test_analytic_reference_values():
    """Guard the reference itself: tau_w scales like 1/sqrt(t) and is positive."""
    assert stokes.wall_shear_exact() > 0.0
    short = stokes.wall_shear_exact(0.25 * stokes.FINAL_TIME)
    long = stokes.wall_shear_exact(stokes.FINAL_TIME)
    assert short / long == pytest.approx(2.0, rel=1e-12)
    assert stokes.layer_length() == pytest.approx(
        2.0 * math.sqrt(stokes.NU * stokes.FINAL_TIME), rel=1e-15)


def test_projected_wall_shear_sign_and_magnitude():
    """The plate drags the fluid downstream: tau > 0 and within 5% at n_y = 16."""
    op, coeff = _projected(1, 16)
    tau = stokes.wall_shear(coeff=coeff, op=op)
    exact = stokes.wall_shear_exact()
    assert tau > 0.0, "sign slip in the one-sided wall derivative"
    assert abs(tau - exact) / exact < 0.05


def test_wall_shear_converges_second_order():
    """The shipped wall closure is second order for curved data after the m2972 fix."""
    errors = {}
    for ny in (8, 32):
        op, coeff = _projected(2, ny)
        errors[ny] = abs(stokes.wall_shear(coeff=coeff, op=op)
                         - stokes.wall_shear_exact()) / stokes.wall_shear_exact()
    order = math.log2(errors[8] / errors[32]) / 2.0
    assert 1.6 < order < 2.4, "measured order %.2f (errors %s)" % (order, errors)
    assert errors[32] < 0.03


def test_profile_error_converges_at_design_order():
    """Volume diffusion operator: O(h^{p+1}) for the profile itself."""
    for degree, low, high in ((1, 1.8, 2.4), (2, 2.8, 3.4)):
        errors = []
        for ny in (8, 32):
            op, coeff = _projected(degree, ny)
            errors.append(stokes.profile_error(coeff, op))
        order = math.log2(errors[0] / errors[1]) / 2.0
        assert low < order < high, "P%d order %.2f errors %s" % (degree, order, errors)


def test_transient_momentum_budget_matches_stokes_integral():
    """The plate is the only x-momentum source, so the budget is closure-free.

    ``(1/Lx) int rho u dA = 2 mu U sqrt(t / (pi nu))`` after the impulsive start.
    """
    op = stokes.grid_operators(1, (1, 8))
    _, _, coeff, meta = stokes.ns.solve_navier_stokes_dg_2d(
        stokes.rest_initial,
        ncells=(1, 8),
        degree=1,
        domain=((0.0, stokes.WIDTH), (0.0, stokes.HEIGHT)),
        final_time=stokes.FINAL_TIME,
        cfl=0.3,
        cfl_viscous=2.0,
        mu=stokes.MU,
        prandtl=stokes.PRANDTL,
        periodic=(True, False),
        boundary=stokes.make_boundary(),
        device=stokes.DEVICE,
        dtype=DTYPE,
    )
    measured, expected = stokes.momentum_budget(coeff, op)
    assert measured > 0.0
    assert abs(measured - expected) / expected < 0.03
    assert meta["final_time_reached"] == pytest.approx(stokes.FINAL_TIME, rel=1e-12)
    # the marched wall shear must also sit on the analytic value (4% at n_y = 8,
    # i.e. the coarse-grid error is already the one the projection study predicts)
    tau = stokes.wall_shear(coeff=coeff, op=op)
    assert abs(tau - stokes.wall_shear_exact()) / stokes.wall_shear_exact() < 0.06