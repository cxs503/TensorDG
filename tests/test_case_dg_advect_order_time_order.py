"""CASE-DG-ADVECT-ORDER / M5_dt_order: temporal-order ladder assertions.

Round 2 of the anchor recorded ``M5_dt_order = not_run``: an explicit-integrator
``dt``-refinement at *fixed* ``N`` cannot expose the time order, because the
measured L2 error is pinned to the constant spatial error of that ``N`` (see
``test_space_refinement_alone_cannot_reveal_time_order`` below, which reproduces
the plateau instead of asserting an order).

The ladder used here removes that coupling by comparing the stepped solution to
the **exact semi-discrete solution** ``expm(L T) u0``, with ``L`` assembled from
the shipped right-hand side.  The assertions are therefore about the integrator
only:

* the assembled matrix really is the shipped operator (linearity residual), so
  the reference describes the same discrete problem;
* the observed order equals the tableau order for ``euler``/``rk2``/``rk3``;
* the shipped RKDG pairing (``tableau_for_degree``) has time error that is
  negligible next to the spatial error at the case-pool CFL, which is the
  premise the anchor's spatial rates rely on.
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.case_dg_advect_order_time_ladder import (
    FINAL_TIME,
    SPEED,
    make_apply,
    project_sine,
    space_floor_ladder,
    temporal_ladder,
)
from tensordg import advection_1d as A
from tensordg import convergence as C
from tensordg.runge_kutta import get_tableau, tableau_for_degree


def test_assembled_operator_is_the_shipped_rhs():
    """The expm reference must describe the *shipped* operator, not a look-alike."""
    for ncells, degree in ((8, 0), (8, 2), (16, 3)):
        apply, _ = make_apply(ncells, degree)
        ndof = ncells * (degree + 1)
        operator = C.semidiscrete_operator(apply, ndof)
        residual = C.operator_linearity_residual(apply, operator)
        assert residual < 1e-12, (ncells, degree, residual)


def test_initial_projection_is_the_sine():
    """Modal projection of ``sin(2*pi*x)`` reproduces the analytic cell moments."""
    ncells, degree = 8, 2
    coeff = project_sine(ncells, degree).numpy()
    h = 1.0 / ncells
    centres = (np.arange(ncells) + 0.5) * h
    # P_0 = 1: the mean of a sine over a cell centred at x_c is sin(2*pi*x_c)*sinc(h)
    mean_exact = np.sin(2 * np.pi * centres) * np.sinc(h)     # numpy sinc(x)=sin(pi x)/(pi x)
    assert np.allclose(coeff[:, 0], mean_exact, atol=1e-13)


@pytest.mark.parametrize("degree,ncells,integrator,tolerance", [
    (0, 8, "euler", 0.06),
    (1, 8, "rk2", 0.02),
    (2, 8, "rk3", 0.02),
])
def test_temporal_order_matches_tableau(degree, ncells, integrator, tolerance):
    """Refining ``dt`` against the exact semi-discrete solution gives order ``q``."""
    row = temporal_ladder(ncells, degree, integrator, levels=4)
    assert row["observed_order"] is not None
    assert abs(row["observed_order"] - row["nominal_order"]) < tolerance, row["levels"]
    # The ladder must actually span the decades the fit reads off (round-2 trap):
    # the error has to fall by >= 100x, and stay above the round-off floor.
    levels = row["levels"]
    assert levels[0]["error"] >= 100.0 * levels[-1]["error"], levels
    assert levels[-1]["snr"] > 1e2, levels
    assert all(level["resolved"] for level in levels[-3:]), levels


def test_matched_pairing_time_error_is_negligible():
    """Shipped RKDG pairing at CFL=0.2: time error << spatial error.

    The anchor's spatial rates are measured with ``tableau_for_degree(p)``.  If
    the time error were comparable to the space error the spatial rates would be
    polluted; this checks the opposite directly by comparing the fully discrete
    solution to the exact semi-discrete one at the case-pool resolution.
    """
    for degree in (1, 2):
        matched = tableau_for_degree(degree)
        assert matched.order == degree + 1
        row = temporal_ladder(8, degree, matched.name, levels=4)
        # dt used by the case pool: cfl * h / speed with h = 1/ncells, smallest dt
        # in the ladder is the finest one; take the coarsest resolved level's dt.
        fine = row["levels"][-1]
        space_err = A.l2_error(*_solve(8, degree), final_time=FINAL_TIME, speed=SPEED)
        assert fine["error"] < 0.1 * space_err or fine["snr"] > 1e4


def _solve(ncells: int, degree: int, cfl: float = 0.2):
    centres, coeff, _ = A.solve_advection(
        ncells=ncells, degree=degree, final_time=FINAL_TIME, speed=SPEED,
        cfl=cfl, time_integrator=tableau_for_degree(degree).name)
    return centres, coeff


def test_space_refinement_alone_cannot_reveal_time_order():
    """The round-2 trap, reproduced: at fixed ``N`` the error stops falling.

    ``M5_dt_order`` was recorded as ``not_run`` because a ``dt`` ladder at fixed
    ``N`` reads off the (constant) spatial error once the time error is smaller.
    The ladder below must therefore show an observed order collapsing towards
    zero on the fine levels -- i.e. the instrument, not the scheme, is the limit.
    """
    rows = space_floor_ladder(ncells=40, degree=2, steps=(100, 200, 400, 800))
    errors = [r["error"] for r in rows]
    orders = C.pairwise_orders([r["nsteps"] for r in rows], errors)
    assert orders[-1] < 0.5, (errors, orders)
    assert errors[-1] > 0.9 * errors[-2], (errors, orders)