"""Scope audit tests for the ``ns_dg_2d`` Dirichlet-wall path (m2786).

``tests/test_ns_dg_2d_wall.py`` (m2589) recorded three open defects and
measured all of them at **degree 1** only.  This file pins down what is a
genuine defect and what is a degree-1 measurement:

* the wall heat flux converges at the **design order** of the element
  (1 / 2 / 3 for P1 / P2 / P3), so "first order" was a P1 statement, not a
  property of the wall operator;
* the ``O(1)`` residual of the exact analytic steady state at P1 is the wall
  flux error divided by ``h`` (the residual is a divergence), and it does
  decay at P2/P3 -- it is not an independent inconsistency;
* the steady state is a stable fixed point of the wall path at P2/P3, while
  the *impulsive start* fails at every degree.  That last item is the only
  remaining open defect; it is kept as a ``strict`` xfail so fixing it forces
  the limitation to be retired instead of silently kept.

Numbers behind these assertions: ``benchmarks/verify_ns_wall_scope.py`` ->
``validation/ns_dg_2d_wall/scope_audit.json`` and
``docs/reports/tensordg-ns-wall-scope-audit.md``.

m3139 (ratchet red-main repair): four of these assertions went red after PR #36
(m2920) replaced the default wall normal derivative by an ``O(h^2)`` two-point
extrapolation.  The bisection table
(``tools/bisect_ns_wall_gradient.py`` ->
``validation/ns_dg_2d_wall/bisect_wall_gradient.json``) shows the *assertions
were right*: they reproduce the pre-#36 numbers bit-for-bit on the pre-#36
solver, and the stencil caps the wall heat flux at second order at every
degree.  The default is therefore the one-sided DG derivative again and the
stencil is opt-in; the two guards at the end of this file keep it that way.
"""
import math

import pytest
import torch

import tensordg.ns_dg_2d as ns
from tools.diag_ns_wall import (
    ANALYTIC_E_FLUX_BOTTOM,
    KAPPA,
    MU,
    build_state,
    march,
    wall_fluxes,
)

DTYPE = torch.float64
DEVICE = torch.device("cpu")
DESIGN_ORDER = {1: 1.0, 2: 2.0, 3: 3.0}


def _flux_errors(degree, refinements=(8, 16, 32)):
    errors = []
    for ny in refinements:
        flux = wall_fluxes((2, ny), degree)
        errors.append(abs(flux["energy_bottom"] - ANALYTIC_E_FLUX_BOTTOM))
    return errors


def _residual(degree, ny):
    d = build_state((2, ny), degree)
    penalty = 2.0 * (degree + 1.0) ** 2
    rhs = ns._rhs_ns(d["coeff"], d["basis"], d["derivative"], d["weights"],
                     d["signs"], d["dx"], d["dy"], (True, False), d["boundary"],
                     MU, KAPPA, penalty)
    return float(rhs.abs().max()), float(rhs[..., 3].abs().max())


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_wall_heat_flux_converges_at_design_order(degree):
    """Observed order of the wall heat flux must match the element design order.

    P1 -> 1, P2 -> 2, P3 -> 3 (P3 is already at round-off by 32 cells, so the
    tolerance there is loose and the floor test below covers it).
    """
    errors = _flux_errors(degree)
    order = math.log2(errors[0] / errors[-1]) / 2.0  # 8 -> 32 cells
    target = DESIGN_ORDER[degree]
    assert order > target - 0.35, (degree, order, errors)


def test_wall_heat_flux_is_round_off_accurate_at_p3():
    """At P3 the Brinkman temperature parabola is resolved, so the wall heat
    flux must be exact to round-off (the m2589 green test, at higher degree)."""
    flux = wall_fluxes((2, 32), 3)
    assert abs(flux["energy_bottom"] - ANALYTIC_E_FLUX_BOTTOM) < 2e-7


def test_exact_state_residual_decays_with_degree():
    """The exact analytic steady state leaves a residual that shrinks with h at
    P2/P3.  At P1 it is flat because the flux error is ``O(h)`` and the
    residual is a divergence (``O(h)/h``); that is not an inconsistency."""
    p1, _ = _residual(1, 8)
    p2_coarse, _ = _residual(2, 8)
    p2_fine, _ = _residual(2, 32)
    p3_coarse, _ = _residual(3, 8)
    p3_fine, _ = _residual(3, 32)
    assert p2_fine < 0.6 * p2_coarse, (p2_coarse, p2_fine)
    assert p3_fine < 0.25 * p3_coarse, (p3_coarse, p3_fine)
    assert p3_fine < 1e-3 * p1


def test_residual_equals_flux_error_over_h_at_p1():
    """Identity check at P1: ``residual * h / flux_error = 3`` exactly.

    The residual is a divergence, so a wall flux error ``eps`` enters it as
    ``eps / h``; the factor 3 is the P1 modal weight ``2k+1 = 3``.  The flat
    P1 residual of m2589 is therefore *this* identity, not a second defect.
    (At P2 the residual is dominated by the interior contribution, so the
    identity only holds at P1 -- that is why the flat-residual statement must
    not be carried over to higher degrees.)
    """
    for ny in (16, 32):
        flux_error = _flux_errors(1, (ny,))[0]
        _, residual_energy = _residual(1, ny)
        ratio = residual_energy * (1.0 / ny) / flux_error
        assert 2.9 < ratio < 3.1, (ny, ratio, residual_energy, flux_error)


def test_steady_state_march_survives_at_p2():
    """Starting from the exact steady state, the wall path is stable at P2
    (m2589 measured P1 only, where it blows up after ~70 steps)."""
    history, failure = march(ncells=(2, 8), degree=2, max_steps=40)
    assert failure is None, failure


@pytest.mark.xfail(strict=True, reason=(
    "open defect (m2786): the impulsively started wall-bounded transient goes "
    "non-physical at every degree (P1/P2/P3), so this is not a P1 "
    "representability issue but an instability of the wall closure"
))
def test_impulsively_started_couette_reaches_final_time_at_p2():
    """Impulsively started Couette at P2 must reach t = 0.05.  It currently
    raises the positivity guard well before that."""
    ns.solve_couette_dg_2d(ncells=(2, 8), degree=2, mu=MU, final_time=0.05)

# ---------------------------------------------------------------------------
# m3139 guards: the design-order path must stay the default
# ---------------------------------------------------------------------------
def test_default_wall_gradient_is_the_design_order_path():
    """Fail-closed: a user of the public solver must get the design-order wall
    operator, not the O(h^2)-capped extrapolation of PR #36."""
    import inspect

    assert inspect.signature(ns._rhs_ns).parameters["wall_gradient"].default == "face"
    assert (inspect.signature(ns.solve_navier_stokes_dg_2d)
            .parameters["wall_gradient"].default == "face")


def test_taylor_wall_gradient_is_not_the_design_order_path_at_p3():
    """Negative control for the guard above: the opt-in stencil really is the
    wrong default.  At P3/32 cells it is >10x the one-sided derivative error,
    and its observed order stays at 2 instead of 3."""
    face = _flux_errors(3)
    taylor = [abs(wall_fluxes((2, ny), 3, wall_gradient="taylor")["energy_bottom"]
                  - ANALYTIC_E_FLUX_BOTTOM) for ny in (8, 16, 32)]
    assert taylor[-1] > 10.0 * face[-1], (taylor, face)
    order = math.log2(taylor[0] / taylor[-1]) / 2.0
    assert order < 2.65, order
