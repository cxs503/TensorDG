"""Verification gates for :mod:`tensordg.ldg_dirichlet_1d` (m16505).

These tests are the acceptance criteria fixed *before* the implementation, in the
order of the falsifiable claims in ``docs/LDG-DIRICHLET-CLOSURE-1D.md``:

C1 consistency   -- an exactly representable solution is reproduced to machine
                    precision (k >= 2), so the closure adds no modelling error;
C2 optimal order -- ``L2`` error of both ``u`` and ``q`` decays as ``h**(k+1)``;
C3 regularity    -- the mixed system is non-singular, including at ``k = 0``
                    where the historical ``pure`` closure had no chance at all;
C4 robustness    -- the stabilisation parameter ``sigma`` does not change the
                    solution beyond a few percent (a consistent penalty term).

The negative control (``sigma = 0`` recovers a rank deficiency of exactly one)
is what makes these gates meaningful: it proves the fix is load-bearing and is
not a numerically benign no-op, and it pins the historical defect recorded by
``docs/ldg_poisson_status.md`` / ``tests/test_ldg_closure_rank.py``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordg.ldg_dirichlet_1d import (
    assemble_ldg_dirichlet_1d,
    ldg_l2_error_1d,
    solve_ldg_dirichlet_1d,
)

SINE = lambda x: np.sin(np.pi * x)                 # noqa: E731 - exact solution
SINE_RHS = lambda x: np.pi ** 2 * np.sin(np.pi * x)  # noqa: E731
SINE_DERIV = lambda x: np.pi * np.cos(np.pi * x)       # noqa: E731 - exact q = u'
GRID = (8, 16, 32)


def _order(errs):
    return [math.log(errs[i] / errs[i + 1]) / math.log(2.0) for i in range(len(errs) - 1)]


# --------------------------------------------------------------------------- #
# C1: consistency -- exact polynomial reproduction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("degree", [2, 3])
def test_quadratic_solution_is_reproduced_to_machine_precision(degree):
    """``u = x^2`` lies in the discrete space, so the discrete solution must be exact.

    This is the strongest consistency statement available: it fails immediately
    if the derivative matrix, a trace orientation, or a penalty sign is wrong.
    """
    sol = solve_ldg_dirichlet_1d(
        8, degree, lambda x: -2.0 * np.ones_like(x), g_left=0.0, g_right=1.0)
    assert ldg_l2_error_1d(sol, lambda x: x * x, "u") < 1e-10
    assert ldg_l2_error_1d(sol, lambda x: 2.0 * x, "q") < 1e-10


def test_dirichlet_data_is_honoured_for_nonzero_values():
    """Both boundary values must enter, not just the homogeneous case."""
    sol = solve_ldg_dirichlet_1d(
        8, 2, lambda x: np.zeros_like(x), g_left=3.0, g_right=-1.0)
    # the exact solution of -u'' = 0 with u(0)=3, u(1)=-1 is u = 3 - 4x
    assert ldg_l2_error_1d(sol, lambda x: 3.0 - 4.0 * x, "u") < 1e-10
    assert ldg_l2_error_1d(sol, lambda x: -4.0 * np.ones_like(x), "q") < 1e-10


# --------------------------------------------------------------------------- #
# C2: optimal order k + 1 in L2 for u and q
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_u_converges_at_order_degree_plus_one(degree):
    errs = [ldg_l2_error_1d(solve_ldg_dirichlet_1d(n, degree, SINE_RHS, 0.0, 0.0), SINE)
            for n in GRID]
    orders = _order(errs)
    assert min(orders[-2:]) >= (degree + 1) - 0.12, (degree, errs, orders)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_q_converges_at_order_degree_plus_one(degree):
    errs = [ldg_l2_error_1d(solve_ldg_dirichlet_1d(n, degree, SINE_RHS, 0.0, 0.0), SINE_DERIV, "q")
            for n in GRID]
    orders = _order(errs)
    assert min(orders[-2:]) >= (degree + 1) - 0.12, (degree, errs, orders)


# --------------------------------------------------------------------------- #
# C3: regularity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("n_cells", [8, 16])
def test_system_is_nonsingular(degree, n_cells):
    A, _b, meta = assemble_ldg_dirichlet_1d(
        n_cells, degree, lambda x: np.ones_like(x), 0.0, 0.0)
    rank = int(np.linalg.matrix_rank(A))
    assert rank == meta["ndofs"]
    assert np.isfinite(np.linalg.cond(A))
    sol = solve_ldg_dirichlet_1d(n_cells, degree, SINE_RHS, 0.0, 0.0)
    assert np.all(np.isfinite(sol["u"])) and np.all(np.isfinite(sol["q"]))


def test_degree_zero_is_solvable_at_all():
    """The ``pure`` closure was *inconsistent* for P0 (``0 = h f``); the fix is not.

    For P0 the historical mass row of the end element read ``0 = h f`` with
    ``f != 0``: no solution exists.  Here the same problem must be solvable, and
    the residual must be O(h) rather than machine-zero (P0 is only first order).
    """
    errs = [ldg_l2_error_1d(solve_ldg_dirichlet_1d(n, 0, SINE_RHS, 0.0, 0.0), SINE)
            for n in GRID]
    assert all(np.isfinite(e) and e > 0 for e in errs)
    assert errs[-1] < errs[0]                       # converges, at least


# --------------------------------------------------------------------------- #
# negative control: the fix is load-bearing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_without_stabilisation_the_deficiency_is_exactly_one(degree):
    """``sigma = 0`` reproduces the historical defect: rank deficiency of one."""
    A, _b, meta = assemble_ldg_dirichlet_1d(
        8, degree, lambda x: np.ones_like(x), 0.0, 0.0, sigma=0.0)
    assert meta["ndofs"] - int(np.linalg.matrix_rank(A)) == 1


# --------------------------------------------------------------------------- #
# C4: the stabilisation is a consistent penalty, not a tuning knob
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("degree", [1, 2])
def test_solution_is_insensitive_to_the_penalty_parameter(degree):
    errors = [ldg_l2_error_1d(solve_ldg_dirichlet_1d(16, degree, SINE_RHS, 0.0, 0.0, sigma=s), SINE)
              for s in (1.0, 10.0, 100.0)]
    spread = (max(errors) - min(errors)) / min(errors)
    assert spread < 0.10, errors


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0, -3])
def test_invalid_cell_count_is_rejected(bad):
    with pytest.raises(ValueError):
        assemble_ldg_dirichlet_1d(bad, 1, lambda x: np.ones_like(x), 0.0, 0.0)


def test_negative_degree_is_rejected():
    with pytest.raises(ValueError):
        assemble_ldg_dirichlet_1d(4, -1, lambda x: np.ones_like(x), 0.0, 0.0)