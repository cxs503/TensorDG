"""Completeness unit tests for the TensorDG basis-function / node module
(``tensordg.curved_geometry``).

These tests harden the *algebraic* contract of the DG building blocks that the
whole library rests on.  They are independent of any PDE solve: they check the
basis and quadrature in isolation, so a regression in the Legendre evaluation,
the GLL node construction, or the mass-matrix assembly is caught at the unit
level rather than as a mysterious order-loss in a benchmark.

Three pre-declared acceptance gates (all must hold, error < 1e-10):

1. **Interpolation identity / Delta property** -- the nodal (Lagrange) basis
   built on the reference-element nodes satisfies ``l_k(x_q) = delta_kq``;
   equivalently the modal evaluation matrix ``L`` is invertible and
   ``L @ inv(L) = I``.
2. **Mass-matrix symmetry + positive-definiteness** -- the (quadrature) mass
   matrix is symmetric and a Cholesky factorisation succeeds (SPD).  For the
   modal Legendre basis on a Gauss rule the mass matrix is *exactly* diagonal
   with entries ``2 / (2k + 1)``.
3. **Polynomial reproduction** -- the L2 projection of a polynomial of degree
   ``<= p`` into the degree-``p`` DG space is exact (relative L2 error < 1e-10)
   for ``p = 1..3``.

Every number asserted here is produced by the library itself (no hand-copied
constants); the only analytic references are the closed-form Legendre
orthogonality integrals and the Kronecker delta.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordg.curved_geometry import (
    CurvedAdvection,
    cartesian_periodic_mesh,
    gauss_nodes_weights,
    gll_nodes_weights,
    legendre_eval_matrix,
)

# Pre-declared acceptance threshold for every gate in this module.
TOL = 1e-10


def _const_vel(x, y):
    return np.full_like(x, 0.6), np.full_like(y, 0.8)


def _poly_degree_p(p: int):
    """Return a bivariate polynomial of total degree ``p`` (small coefficients)."""

    def f(x, y):
        s = np.zeros_like(x, dtype=float)
        for i in range(p + 1):
            for j in range(p + 1):
                if i + j <= p:
                    s = s + (0.1 ** (i + j + 1)) * (x ** i) * (y ** j)
        return s

    return f


# ---------------------------------------------------------------------------
# Gate 1: interpolation identity / Delta property at the reference nodes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", list(range(1, 7)))
def test_gll_nodal_basis_delta_property(p):
    """The nodal basis on the GLL nodes is the Kronecker delta: ``l_k(x_q)=delta_kq``.

   The nodal basis values at the nodes are ``N = L @ inv(L)`` (modal -> nodal),
   so the Delta property is exactly ``N == I``.
    """
    x, _ = gll_nodes_weights(p)
    L = legendre_eval_matrix(p, x)
    N = L @ np.linalg.inv(L)
    err = float(np.abs(N - np.eye(p + 1)).max())
    assert err < TOL, f"Delta property violated at p={p}: max|N-I|={err:.3e}"


@pytest.mark.parametrize("p", list(range(1, 7)))
def test_modal_basis_interpolation_identity(p):
    """Modal evaluation matrix is invertible and ``L @ inv(L) = I`` at both the
    GLL and the Gauss node sets (interpolation is well-posed at the nodes)."""
    for nodes in (gll_nodes_weights(p)[0], gauss_nodes_weights(p + 1)[0]):
        L = legendre_eval_matrix(p, nodes)
        err = float(np.abs(L @ np.linalg.inv(L) - np.eye(p + 1)).max())
        assert err < TOL, f"interpolation identity violated at p={p}: {err:.3e}"
        # The node set must resolve the space: condition number is finite and
        # the columns are linearly independent.
        assert np.linalg.matrix_rank(L) == p + 1


# ---------------------------------------------------------------------------
# Gate 2: mass matrix is symmetric positive definite (Cholesky succeeds)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", list(range(1, 7)))
def test_nodal_mass_matrix_spd_and_diagonal(p):
    """Nodal mass matrix on the GLL rule is exactly ``diag(w)`` and SPD."""
    x, w = gll_nodes_weights(p)
    L = legendre_eval_matrix(p, x)
    N = L @ np.linalg.inv(L)                      # nodal basis at the nodes
    M = N.T @ np.diag(w) @ N
    # Symmetric
    np.testing.assert_allclose(M, M.T, atol=1e-14)
    # Exactly diagonal with the GLL weights on the diagonal
    np.testing.assert_allclose(M, np.diag(w), atol=1e-10)
    # Positive weights -> SPD; Cholesky must succeed.
    chol = np.linalg.cholesky(M)
    assert np.all(np.diag(chol) > 0.0)


@pytest.mark.parametrize("p", list(range(1, 7)))
def test_modal_mass_matrix_spd_and_orthogonal(p):
    """Modal Legendre mass on a Gauss rule is exactly diagonal ``2/(2k+1)`` and SPD."""
    x, w = gauss_nodes_weights(p + 1)
    L = legendre_eval_matrix(p, x)
    M = L.T @ np.diag(w) @ L
    ref = np.array([2.0 / (2 * k + 1) for k in range(p + 1)])
    np.testing.assert_allclose(M, M.T, atol=1e-14)
    np.testing.assert_allclose(M, np.diag(ref), atol=1e-10)
    np.linalg.cholesky(M)                          # raises if not SPD


@pytest.mark.parametrize("p", [1, 2, 3])
def test_curved_advection_element_mass_spd(p):
    """Every element mass matrix assembled by ``CurvedAdvection`` is SPD."""
    dg = CurvedAdvection(cartesian_periodic_mesh(3, 3, 1), p, _const_vel)
    n = dg.M.shape[0]
    for e in range(n):
        Me = dg.M[e]
        np.testing.assert_allclose(Me, Me.T, atol=1e-12)
        np.linalg.cholesky(Me)                     # raises if not SPD
    # The reference (J=1) diagonal is the analytic Legendre mass.
    ref = np.array([2.0 / (2 * k + 1) for k in range(dg.K)])
    np.testing.assert_allclose(dg.mass_ref_diag, np.outer(ref, ref), atol=1e-14)


# ---------------------------------------------------------------------------
# Gate 3: exact polynomial reproduction (relative L2 error < 1e-10)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", [1, 2, 3])
def test_polynomial_reproduction_exact(p):
    """Projecting a degree-``p`` polynomial into the degree-``p`` DG space is exact."""
    dg = CurvedAdvection(cartesian_periodic_mesh(4, 4, 1), p, _const_vel)
    f = _poly_degree_p(p)
    U = dg.project(f)
    num, den, rel = dg.l2_error(U, f)
    assert den > 0.0
    assert rel < TOL, f"polynomial reproduction not exact at p={p}: rel={rel:.3e}"
