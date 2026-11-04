"""Stability envelope of the shipped ``ns_dg_2d`` Dirichlet-wall operator (m3595).

History, so the assertions cannot be mistaken for a re-derivation:

* ``docs/reports/tensordg-ns-wall-spectrum.md`` (m2731/m2920) measured the
  *pre-m2782* wall operator at one point of the option space and found
  ``max Re(lambda) = +348`` growing **linearly with the SIPG penalty**
  (``sigma = 16 -> 2960.9``), i.e. a wall term that no penalty magnitude can
  repair.  That was an honest FAIL of the wall-bounded transient.
* ``test_ns_dg_2d_wall.py::test_wall_operator_has_no_growing_eigenmode``
  retires the limitation after the ``outward_sign`` fix, but at a **single
  penalty** (``sigma = 8``) and a single closure.
* The m2972 symmetric Nitsche adjoint (``wall_adjoint=True``) was justified by
  transients; its *spectrum* was never measured.

The tests here generalise the retirement to the whole shipped envelope
(penalty, wall-gradient closure, adjoint on/off, degree) and keep the pre-fix
sign-blind wall flux as a **negative control**, so a broken eigensolver or an
accidentally vacuous gate turns red.  Numbers behind every threshold:
``validation/ns_dg_2d_wall/stability_envelope.json`` (regenerate with
``PYTHONPATH=. python3 tools/spectrum_ns_wall_envelope.py``) and
``docs/reports/tensordg-ns-wall-stability-envelope.md``.
"""
import numpy as np
import pytest

import tensordg.ns_dg_2d as ns
from tools.diag_ns_wall import build_state
from tools.diag_ns_wall_spectrum import wall_flux_variant
from tools.spectrum_ns_wall import _spectrum

REFERENCE_WALL_FLUX = ns._boundary_flux
# Byte-for-byte the wall flux shipped before the m2782 ``outward_sign`` fix.
PREFIX_WALL_FLUX = wall_flux_variant((1.0, 1.0, 1.0))

# Double-precision floor of the eigensolve: the spectrum magnitude is 3.5e2
# (sigma = 2) to 2.3e4 (sigma = 128), so 1e-9 is still ~1e-13 *relative*; the
# pre-fix operator measured > 50 at the smallest penalty.
FLOOR = 1e-9


def _spectrum_at(state, sigma, gradient=None, adjoint=False, pre_fix=False):
    gradient = ns.WALL_GRADIENT if gradient is None else gradient
    ns._boundary_flux = PREFIX_WALL_FLUX if pre_fix else REFERENCE_WALL_FLUX
    try:
        return _spectrum(state["coeff"], state, state["boundary"], (True, False),
                         sigma, gradient, "default", wall_adjoint=adjoint)
    finally:
        ns._boundary_flux = REFERENCE_WALL_FLUX


def _residual(state, sigma, adjoint, gradient=None):
    gradient = ns.WALL_GRADIENT if gradient is None else gradient
    ns._boundary_flux = REFERENCE_WALL_FLUX
    rhs = ns._rhs_ns(state["coeff"], state["basis"], state["derivative"],
                     state["weights"], state["signs"], state["dx"], state["dy"],
                     (True, False), state["boundary"], 0.1,
                     0.1 * ns.CP / 0.71, sigma, gradient, wall_adjoint=adjoint)
    return float(rhs.abs().max())


@pytest.fixture(scope="module")
def p1_state():
    return build_state((2, 8), 1)


@pytest.fixture(scope="module")
def p2_state():
    return build_state((2, 8), 2)


@pytest.fixture(scope="module")
def penalty_sweep(p1_state):
    """Shipped operator at the shipped and a 16x larger penalty, adjoint off."""
    return {sigma: _spectrum_at(p1_state, sigma) for sigma in (8.0, 128.0)}


@pytest.fixture(scope="module")
def adjoint_off_on(p1_state):
    """The m2972 adjoint term at the shipped penalty (sigma = 8, P1)."""
    return {False: _spectrum_at(p1_state, 8.0),
            True: _spectrum_at(p1_state, 8.0, adjoint=True)}


@pytest.fixture(scope="module")
def closures(p1_state):
    """The opt-in wall-gradient closures at the shipped penalty."""
    return {gradient: _spectrum_at(p1_state, 8.0, gradient=gradient)
            for gradient in ("fd2", "taylor")}


@pytest.fixture(scope="module")
def prefix_control(p1_state):
    return {sigma: _spectrum_at(p1_state, sigma, pre_fix=True) for sigma in (8.0, 128.0)}


@pytest.fixture(scope="module")
def p2_adjoint(p2_state):
    sigma = 2.0 * (2.0 + 1.0) ** 2
    return {False: _spectrum_at(p2_state, sigma),
            True: _spectrum_at(p2_state, sigma, adjoint=True)}


def test_shipped_operator_has_no_growing_eigenmode_at_any_penalty(penalty_sweep):
    """The wall operator is growth-free at the shipped penalty *and* at a 16x
    larger one (measured max Re = -2.8e-13 / -9.9e-13, i.e. roundoff)."""
    for sigma, result in penalty_sweep.items():
        assert abs(result["max_real"]) <= FLOOR, (sigma, result["max_real"])


def test_growing_mode_no_longer_scales_with_the_penalty(penalty_sweep, prefix_control):
    """The pre-fix fingerprint was ``max Re proportional to sigma``.  On the
    shipped operator the growth stays at the roundoff floor while the sign-blind
    control grows by 16.8x over the same 16x penalty increase -- so the
    proportionality is gone, not merely shifted."""
    shipped_8, shipped_128 = penalty_sweep[8.0]["max_real"], penalty_sweep[128.0]["max_real"]
    prefix_8, prefix_128 = prefix_control[8.0]["max_real"], prefix_control[128.0]["max_real"]
    assert prefix_8 > 50.0 and prefix_128 > 1000.0             # gate is not vacuous
    assert 14.0 < prefix_128 / prefix_8 < 20.0, prefix_128 / prefix_8
    assert abs(shipped_8) <= FLOOR and abs(shipped_128) <= FLOOR


def test_adjoint_term_adds_wall_dissipation_without_adding_growth(adjoint_off_on):
    """m2972's symmetric (derivative-projected) wall term is the piece the m2731
    spectrum said was missing.  Measured at sigma = 8, P1: growth stays at the
    floor while the most negative real part deepens by 12 % (1315.8 -> 1472.8),
    i.e. the adjoint is dissipative exactly in the direction that was missing."""
    off, on = adjoint_off_on[False], adjoint_off_on[True]
    assert abs(off["max_real"]) <= FLOOR and abs(on["max_real"]) <= FLOOR
    ratio = on["min_real"] / off["min_real"]          # both negative
    assert 1.05 < ratio < 1.30, ratio


def test_adjoint_dissipation_grows_with_the_polynomial_degree(p2_adjoint):
    """At P2 (sigma = 18) the adjoint nearly doubles the wall dissipation
    (5568.9 -> 10986, ratio 1.97) at the same roundoff-level growth: the
    derivative projection matters more the higher the wall cell's modal content."""
    off, on = p2_adjoint[False], p2_adjoint[True]
    assert abs(off["max_real"]) <= FLOOR and abs(on["max_real"]) <= FLOOR
    ratio = on["min_real"] / off["min_real"]
    assert 1.50 < ratio < 2.50, ratio


def test_every_opt_in_wall_gradient_closure_is_growth_free(closures):
    """``fd2`` (m2831 one-sided parabola) and ``taylor`` (m2920 two-point
    extrapolation) are both growth-free at the shipped penalty: the stability
    verdict does not hinge on the closure choice (measured |max Re| <= 1.9e-13)."""
    for gradient, result in closures.items():
        assert abs(result["max_real"]) <= FLOOR, (gradient, result["max_real"])


def test_analytic_state_residual_is_a_wall_closure_quantity(p1_state):
    """Scope of the verdict above, measured rather than assumed.

    The analytic steady Couette state is *not* a discrete steady state of the
    scheme: ``max|rhs|`` is 4.5e-2 at 8 cells and 5.1e-2 at 32 cells (order
    -0.04, i.e. O(1); the O(h) wall-flux error divided by the cell volume
    cancels).  The spectra above are therefore Jacobians at a state with a
    wall-closure residual, which is what makes the *linear stability* statement
    meaningful for the marching code -- but the closure residual is not a
    discretisation error and the m2972 adjoint, being proportional to the same
    Dirichlet deficit, multiplies it (0.0452 -> 0.3949 at P1, 8.7x; 0.0102 ->
    0.0303 at P2, 3.0x).  That is why the adjoint stays opt-in.  When the wall
    closure itself is repaired, this test must be rewritten together with
    ``docs/reports/tensordg-ns-wall-stability-envelope.md``."""
    coarse = _residual(p1_state, 8.0, adjoint=False)
    fine_state = build_state((2, 32), 1)
    fine = _residual(fine_state, 8.0, adjoint=False)
    assert 0.9 < fine / coarse < 1.4, (coarse, fine)
    adjoint = _residual(p1_state, 8.0, adjoint=True)
    assert adjoint / coarse > 3.0, (coarse, adjoint)