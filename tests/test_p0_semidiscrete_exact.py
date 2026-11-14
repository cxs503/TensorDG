"""Tests for the closed-form DG0 (upwind FV) semidiscrete reference.

CASE-DG-ADVECT-ORDER / P0 row.  The point of these tests is that the *reference*
used to judge the DG0 convergence rate is verified independently three ways:

1. the analytic circulant symbol equals the operator the solver actually applies
   (assembled from ``tensordg.advection_1d._rhs``);
2. the closed-form semidiscrete solution equals ``expm(M*T) u0`` with that
   assembled matrix (an independent linear-algebra path, no time stepping);
3. the closed-form L2 error equals a brute-force piecewise quadrature of the
   semidiscrete field against the exact sine.

Then the convergence-order statements that the case pool cares about are
asserted against those references, including the honest pre-asymptotic value
``0.853`` on the pool ladder ``N = 20 ... 320``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordg import advection_1d as A
from tensordg import p0_exact as P


def assemble_operator(ncells: int, speed: float = 1.0) -> np.ndarray:
    """DG0 operator as the solver applies it: columns = _rhs(e_j)."""
    xq, wq, phi, volume, phi_l, phi_r = A._operators(0, torch.device("cpu"), torch.float64)
    h = 1.0 / ncells
    M = np.zeros((ncells, ncells))
    for j in range(ncells):
        e = np.zeros((ncells, 1))
        e[j, 0] = 1.0
        M[:, j] = A._rhs(torch.as_tensor(e), h, speed, volume, phi_l, phi_r).numpy()[:, 0]
    return M


def test_symbol_matches_solver_operator():
    """The analytic circulant symbol is the operator's actual matrix."""
    n = 8
    h = 1.0 / n
    M = assemble_operator(n)
    shift = np.zeros((n, n))
    for j in range(n):
        shift[j, (j - 1) % n] = 1.0          # u_{j-1}
    expected = -(1.0 / h) * (np.eye(n) - shift)
    assert np.allclose(M, expected, atol=1e-12)


def test_eigenvalue_multiset_matches():
    """Spectrum of the assembled operator equals :func:`p0_eigenvalues`."""
    for n in (8, 16, 40):
        lam_num = np.linalg.eigvals(assemble_operator(n))
        lam_ana = P.p0_eigenvalues(n)
        assert np.allclose(np.sort(lam_num.real), np.sort(lam_ana.real), atol=1e-10)
        assert np.allclose(np.sort(lam_num.imag), np.sort(lam_ana.imag), atol=1e-10)


def test_initial_averages_match_exact_projection():
    """Analytic cell averages equal the P0 L2 projection of sin(2*pi*x)."""
    n = 32
    h = 1.0 / n
    centers, coeff, _ = A.solve_advection(ncells=n, degree=0, final_time=0.0)
    got = coeff[:, 0].detach().cpu().numpy()
    assert np.allclose(got, P.p0_initial_cell_averages(n), atol=1e-14)
    assert abs(float(np.sum(got)) * h) < 1e-14


@pytest.mark.parametrize("ncells", [16, 40, 160])
def test_semidiscrete_solution_matches_expm(ncells):
    """Closed-form solution == expm(M*T) u0 on the solver's own matrix."""
    from scipy.linalg import expm

    T = 1.0
    M = assemble_operator(ncells)
    u_expm = expm(M * T) @ P.p0_initial_cell_averages(ncells)
    assert np.allclose(u_expm, P.p0_exact_cell_averages(ncells, T), atol=1e-12)


@pytest.mark.parametrize("ncells", [20, 80])
def test_l2_error_matches_piecewise_quadrature(ncells):
    """Closed-form L2 error == brute-force Gauss quadrature of the DG0 field."""
    T = 1.0
    u = P.p0_exact_cell_averages(ncells, T)
    xq, wq = np.polynomial.legendre.leggauss(12)
    h = 1.0 / ncells
    x = (np.arange(ncells)[:, None] + 0.5 * (1.0 + xq[None, :])) * h
    truth = np.sin(2.0 * math.pi * (x - T))
    err2 = 0.5 * h * np.sum(wq[None, :] * (u[:, None] - truth) ** 2)
    assert abs(math.sqrt(err2) - P.l2_error_semidiscrete(ncells, T)) < 1e-12


def test_pool_ladder_order_is_preasymptotic_0p853():
    """The pool ladder N=20..320 gives 0.853 -- from theory, not from the solver."""
    ladder = (20, 40, 80, 160, 320)
    order = P.observed_order(ladder, [P.l2_error_semidiscrete(n) for n in ladder])
    assert abs(order - 0.85347) < 5e-4, f"pool-ladder order {order}"


def test_asymptotic_regime_recovers_design_order():
    """Local order -> 1 and N*E(N) -> the two-term closed-form constant."""
    assert P.asymptotic_order(40960) > 0.9995
    assert abs(P.asymptotic_constant(40960) / P.P0_ASYMPTOTIC_CONSTANT - 1.0) < 1e-3
    assert abs(P.asymptotic_constant(20480) / P.P0_ASYMPTOTIC_CONSTANT - 1.0) < 2e-3


def test_closed_form_constant_identity():
    """pi^2*sqrt(2) and pi/sqrt(6) combine to pi*sqrt(2*pi^2 + 1/6)."""
    assert abs(P.P0_SCHEME_CONSTANT - math.pi ** 2 * math.sqrt(2.0)) < 1e-12
    assert abs(P.P0_PROJECTION_CONSTANT - math.pi / math.sqrt(6.0)) < 1e-12
    assert abs(math.hypot(P.P0_SCHEME_CONSTANT, P.P0_PROJECTION_CONSTANT)
               - P.P0_ASYMPTOTIC_CONSTANT) < 1e-12
    assert abs(P.P0_ASYMPTOTIC_CONSTANT - 14.01653) < 1e-4


def test_solver_reproduces_analytic_pool_order():
    """Fully discrete DG0 solver (RK4, cfl=0.2) matches the analytic error curve."""
    ladder = (20, 40, 80, 160, 320)
    got = []
    for n in ladder:
        centers, coeff, meta = A.solve_advection(ncells=n, degree=0, final_time=1.0, cfl=0.2)
        u = coeff[:, 0].detach().cpu().numpy()
        got.append(P.l2_error_fully_discrete(u, 1.0))
        assert abs(got[-1] / P.l2_error_semidiscrete(n) - 1.0) < 1e-4, meta
    order = P.observed_order(ladder, got)
    exact = P.observed_order(ladder, [P.l2_error_semidiscrete(n) for n in ladder])
    assert abs(order - exact) < 2e-3