"""Verification-debt closure for the matrix-free (PCG) SIPG backends.

The matrix-free backend (``tensordg.sipg_matrixfree``) was originally verified
only *against the dense backend of the same discretisation*.  Cross-backend
agreement proves implementation equivalence, but it cannot prove that the
discretisation plus the iterative solve reproduces the analytic solution, nor
that the requested linear-solver tolerance is sub-dominant to the
discretisation error.  These tests close that gap:

* matrix-free P1/P2 Poisson is compared with the manufactured solution
  ``u = sin(pi x) sin(pi y)`` and the experimental order of convergence (EOC)
  is measured on the DG solution itself;
* tolerance independence: loosening/tightening ``rtol`` by seven orders of
  magnitude must not change the error field;
* an exhausted CG iteration budget must be reported honestly
  (``cg_converged == False``) instead of silently returning an unconverged
  iterate as if it were a solution;
* the 1-D heat matrix-free solve is compared with the analytic solution
  ``sin(pi x) exp(-pi^2 t)`` and with the dense backend;
* the operators handed to CG are verified symmetric, which is what makes CG
  applicable at all (structural evidence, not accuracy evidence).
"""
from __future__ import annotations

import math

import torch

from tensordg import sipg_matrixfree as mf
from tensordg.heat_1d import (
    _operators as heat_operators,
    heat_error,
    solve_heat_1d,
)
from tensordg.poisson_2d import (
    _operators as poisson_operators,
    poisson_error_2d,
    solve_poisson_2d,
)


def _order(coarse: float, fine: float, ratio: float = 2.0) -> float:
    return math.log(coarse / fine, ratio)


def _dense_columns(matvec, ndofs: int) -> torch.Tensor:
    """Materialise an operator from its matvec (verification-only, tiny ndofs)."""
    matrix = torch.zeros(ndofs, ndofs, dtype=torch.float64)
    for column in range(ndofs):
        unit = torch.zeros(ndofs, dtype=torch.float64)
        unit[column] = 1.0
        matrix[:, column] = matvec(unit)
    return matrix


# ---------------------------------------------------------------------------
# 1. Accuracy against the manufactured Poisson solution (not just vs dense)
# ---------------------------------------------------------------------------
def test_matrixfree_poisson_p1_matches_manufactured_solution_at_second_order():
    errors = {}
    for ncells in (8, 16):
        x, y, coeff, meta = solve_poisson_2d(
            ncells=ncells, degree=1, backend="matrix-free",
        )
        assert meta["cg_converged"] is True, meta
        assert meta["cg_relative_residual"] <= meta["cg_rtol"] * 1.05, meta
        errors[ncells] = poisson_error_2d(x, y, coeff)["relative_l2"]

    assert errors[16] < 1.0e-2, errors
    order = _order(errors[8], errors[16])
    assert abs(order - 2.0) <= 0.1, (order, errors)


def test_matrixfree_poisson_p2_matches_manufactured_solution_at_third_order():
    errors = {}
    for ncells in (4, 8):
        x, y, coeff, meta = solve_poisson_2d(
            ncells=ncells, degree=2, backend="matrix-free",
        )
        assert meta["cg_converged"] is True, meta
        errors[ncells] = poisson_error_2d(x, y, coeff)["relative_l2"]

    assert errors[8] < 1.0e-3, errors
    order = _order(errors[4], errors[8])
    assert abs(order - 3.0) <= 0.25, (order, errors)


# ---------------------------------------------------------------------------
# 2. The linear-solver tolerance must not pollute the discrete solution
# ---------------------------------------------------------------------------
def test_matrixfree_solution_is_independent_of_cg_tolerance():
    _, _, coeff_loose, meta_loose = solve_poisson_2d(
        ncells=8, degree=1, backend="matrix-free", rtol=1e-6,
    )
    _, _, coeff_tight, meta_tight = solve_poisson_2d(
        ncells=8, degree=1, backend="matrix-free", rtol=1e-13,
    )
    assert meta_loose["cg_converged"] is True
    assert meta_tight["cg_converged"] is True
    assert meta_tight["cg_iterations"] >= meta_loose["cg_iterations"]

    scale = coeff_tight.abs().max()
    assert float((coeff_loose - coeff_tight).abs().max()) / float(scale) < 1e-4

    x, y, _, _ = solve_poisson_2d(ncells=8, degree=1, backend="matrix-free")
    err_loose = poisson_error_2d(x, y, coeff_loose)["relative_l2"]
    err_tight = poisson_error_2d(x, y, coeff_tight)["relative_l2"]
    assert abs(err_loose - err_tight) / err_tight < 1e-3, (err_loose, err_tight)


def test_exhausted_cg_budget_is_reported_instead_of_hidden():
    x, y, coeff, meta = solve_poisson_2d(
        ncells=8, degree=1, backend="matrix-free", rtol=1e-14, maxiter=2,
    )
    assert meta["cg_converged"] is False, meta
    assert meta["cg_iterations"] == 2, meta
    assert meta["cg_relative_residual"] > 1e-6, meta
    # ... and the unconverged iterate must not be silently mistaken for a
    # converged one: its discretisation error is far above the verified level.
    unconverged_error = poisson_error_2d(x, y, coeff)["relative_l2"]
    x_ref, y_ref, coeff_ref, meta_ref = solve_poisson_2d(
        ncells=8, degree=1, backend="matrix-free",
    )
    assert meta_ref["cg_converged"] is True
    converged_error = poisson_error_2d(x_ref, y_ref, coeff_ref)["relative_l2"]
    assert unconverged_error > converged_error


# ---------------------------------------------------------------------------
# 3. Heat equation: analytic solution, cross-backend, iteration budget
# ---------------------------------------------------------------------------
def test_heat_matrixfree_matches_analytic_solution_and_dense_backend():
    final_time = 0.1
    steps = 20
    centers, coeff, meta = mf.solve_heat_1d_matrixfree(
        ncells=16, degree=2, final_time=final_time, time_steps=steps,
        rtol=1e-10,
    )
    assert meta["steps"] == steps
    assert meta["cg_all_converged"] is True, meta

    error = heat_error(centers, coeff, final_time)["relative_l2"]
    assert error < 3.0e-2, error

    centers_dense, coeff_dense, _ = solve_heat_1d(
        ncells=16, degree=2, final_time=final_time, time_steps=steps,
    )
    assert float((centers - centers_dense).abs().max()) < 1e-12
    scale = coeff_dense.abs().max()
    assert float((coeff - coeff_dense).abs().max()) / float(scale) < 1e-9

    iterations_per_step = meta["total_cg_iterations"] / steps
    assert iterations_per_step <= 200.0, meta


# ---------------------------------------------------------------------------
# 4. Structural evidence: the CG system operators are symmetric positive definite
# ---------------------------------------------------------------------------
def test_cg_system_operators_are_symmetric():
    ncells, degree, penalty = 8, 2, 20.0
    operators = heat_operators(degree, torch.device("cpu"), torch.float64)
    ndofs = ncells * (degree + 1)
    mass = _dense_columns(
        lambda v: mf.matvec_heat_mass_1d(v, ncells, degree, operators), ndofs,
    )
    diffusion = _dense_columns(
        lambda v: mf.matvec_heat_diffusion_1d(
            v, ncells, degree, penalty, operators,
        ),
        ndofs,
    )
    assert float((mass - mass.T).abs().max()) < 1e-12
    assert float((diffusion - diffusion.T).abs().max()) < 1e-12

    h = 1.0 / ncells
    dt = 0.02 * h
    system = mass + 0.5 * dt * diffusion
    assert float((system - system.T).abs().max()) < 1e-12
    assert float(torch.linalg.eigvalsh(system).min()) > 0.0


def test_poisson_matrixfree_operator_is_symmetric():
    ncells, degree, penalty = 4, 1, 20.0
    operators = poisson_operators(degree, torch.device("cpu"), torch.float64)
    modes = degree + 1
    ndofs = ncells * ncells * modes * modes
    matrix = _dense_columns(
        lambda v: mf.matvec_poisson_2d(
            v, ncells, ncells, degree, penalty, operators,
        ),
        ndofs,
    )
    assert float((matrix - matrix.T).abs().max()) < 1e-10