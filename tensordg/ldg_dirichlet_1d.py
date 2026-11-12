"""Validated 1-D local discontinuous Galerkin (LDG) solver for ``-u'' = f``.

Why this module exists
----------------------
``docs/ldg_poisson_status.md`` (m16089) and ``docs/ldg_closure_reassessment_m16388.md``
established that the alternating-flux LDG closure with Dirichlet data at *both*
ends is structurally rank deficient, and that the ``c/h`` penalty used to restore
full rank only reached order 2 instead of ``k + 1``.  ``docs/LDG-DIRICHLET-CLOSURE-1D.md``
(m16505) closes that open problem: the missing ingredient is a symmetric
stabilisation of *both* Dirichlet-face fluxes, which is **consistent** because
each added term is proportional to the boundary residual ``u_int - g`` and hence
vanishes for the exact solution.

The method
----------
Mixed form on ``q = u'``::

    int q r dx + int u r' dx - [u_hat r]      = 0
    int q v' dx - [q_hat v]                   = int f v dx

with modal Legendre bases, the derivative acting on the *test* function
(``D_test[i, j] = int L_i' L_j``; using ``D_coef`` breaks consistency), and the
numerical traces

* interior faces: alternating, ``u_hat = u^-`` and ``q_hat = q^+``;
* Dirichlet faces: ``u_hat = g - sigma/h (u_int - g)`` and
  ``q_hat = q_int + sigma/h (u_int - g)``.

The two penalty signs are *opposite*; that is what makes the pair a consistent
numerical flux for the mixed form rather than two unrelated penalties.  The
scheme is parameter robust: ``sigma`` anywhere in ``[0.5, 100]`` gives the same
error curve to three digits (see ``docs/LDG-DIRICHLET-CLOSURE-1D.md`` §3.4).

Verified properties (``tests/test_ldg_dirichlet_1d.py``)
--------------------------------------------------------
* consistency   -- ``u = x^2, q = 2x`` is reproduced to machine precision for
  ``k >= 2`` (residual ``~1e-13``);
* optimal order -- ``L2`` error of both ``u`` and ``q`` decays as ``h**(k+1)``
  for ``k = 1, 2, 3``;
* regularity    -- the assembled matrix is non-singular for every ``(k, N)``
  tested, including ``sigma = 0`` where it is singular by exactly one mode (the
  historical defect, pinned by ``tests/test_ldg_closure_rank.py``).

Scope
-----
1-D, uniform mesh, Dirichlet data at both ends, Cartesian.  The 2-D prototype
``tensordg/poisson_2d_ldg.py`` remains unvalidated and unexported; this module
does not change its status.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "assemble_ldg_dirichlet_1d",
    "solve_ldg_dirichlet_1d",
    "ldg_l2_error_1d",
]

DEFAULT_SIGMA = 10.0


def _basis(degree: int):
    """Modal Legendre basis on [-1, 1]: mass, test-derivative and face values."""
    nq = max(2 * degree + 4, 8)
    xi, w = np.polynomial.legendre.leggauss(nq)
    m = degree + 1
    phi = np.zeros((m, nq))
    dphi = np.zeros((m, nq))
    for j in range(m):
        c = np.zeros(j + 1)
        c[-1] = 1.0
        phi[j] = np.polynomial.legendre.legval(xi, c)
        dc = np.polynomial.legendre.legder(c)
        dphi[j] = np.polynomial.legendre.legval(xi, dc) if len(dc) else 0.0
    mass = np.einsum("q,iq,jq->ij", w, phi, phi)
    dtest = np.einsum("q,iq,jq->ij", w, dphi, phi)
    face = {"l": np.array([(-1.0) ** j for j in range(m)]), "r": np.ones(m)}
    return xi, w, phi, mass, dtest, face


def assemble_ldg_dirichlet_1d(n_cells: int, degree: int, rhs, g_left: float,
                              g_right: float, sigma: float = DEFAULT_SIGMA):
    """Assemble the mixed LDG system for ``-u'' = f`` on ``(0, 1)``.

    Parameters
    ----------
    n_cells, degree : int
        Uniform cell count and polynomial degree (``ndofs = 2 * n_cells * (degree + 1)``).
    rhs : callable
        Source term ``f(x)``.
    g_left, g_right : float
        Dirichlet data at ``x = 0`` and ``x = 1``.
    sigma : float
        Dirichlet-face stabilisation.  ``sigma = 0`` reproduces the historical
        rank-deficient ``pure`` closure and is accepted only so that the defect
        can be *demonstrated*; validated use is ``sigma > 0``.

    Returns
    -------
    (A, b, meta) : the square system, the right-hand side and a metadata dict
        with the block layout (``iq``/``iu`` index helpers, ``h``, ``ndofs``).
    """
    if n_cells < 1:
        raise ValueError("n_cells must be >= 1")
    if degree < 0:
        raise ValueError("degree must be >= 0")
    xi, w, phi, mass, dtest, face = _basis(degree)
    m = degree + 1
    h = 1.0 / n_cells
    ndofs = 2 * n_cells * m
    A = np.zeros((ndofs, ndofs))
    b = np.zeros(ndofs)
    iq = lambda e: e * 2 * m          # noqa: E731 - block offsets, kept literal
    iu = lambda e: e * 2 * m + m      # noqa: E731
    for e in range(n_cells):
        xc = (e + 0.5) * h
        A[iq(e):iq(e) + m, iq(e):iq(e) + m] += 0.5 * h * mass
        A[iq(e):iq(e) + m, iu(e):iu(e) + m] += dtest
        A[iu(e):iu(e) + m, iq(e):iq(e) + m] += dtest
        b[iu(e):iu(e) + m] += 0.5 * h * (w * rhs(xc + 0.5 * h * xi)) @ phi.T
        for side in ("l", "r"):
            own = face[side]
            far = face["r" if side == "l" else "l"]
            sgn = -1.0 if side == "l" else 1.0        # -[.]_d = -right + left
            neighbour = e - 1 if side == "l" else e + 1
            if 0 <= neighbour < n_cells:
                # interior face: alternating traces, each expanded on the owning
                # element's *facing* endpoint basis
                u_owner = neighbour if side == "l" else e
                q_owner = e if side == "l" else neighbour
                u_basis = far if u_owner == neighbour else own
                q_basis = far if q_owner == neighbour else own
                A[iq(e):iq(e) + m, iu(u_owner):iu(u_owner) + m] += -sgn * np.outer(own, u_basis)
                A[iu(e):iu(e) + m, iq(q_owner):iq(q_owner) + m] += -sgn * np.outer(own, q_basis)
            else:
                data = g_left if side == "l" else g_right
                # u_hat = g - sigma/h (u_int - g): consistent (vanishes for exact u)
                b[iq(e):iq(e) + m] += sgn * own * data
                A[iq(e):iq(e) + m, iu(e):iu(e) + m] += -sgn * sigma / h * np.outer(own, own)
                b[iq(e):iq(e) + m] += -sgn * sigma / h * own * data
                # q_hat = q_int + sigma/h (u_int - g): the matching half of the pair
                A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sgn * np.outer(own, own)
                A[iu(e):iu(e) + m, iu(e):iu(e) + m] += sgn * sigma / h * np.outer(own, own)
                b[iu(e):iu(e) + m] += sgn * sigma / h * own * data
    meta = {"n_cells": n_cells, "degree": degree, "h": h, "ndofs": ndofs, "sigma": sigma}
    return A, b, meta


def solve_ldg_dirichlet_1d(n_cells: int, degree: int, rhs, g_left: float,
                           g_right: float, sigma: float = DEFAULT_SIGMA) -> dict:
    """Solve ``-u'' = f`` with Dirichlet data at both ends.

    Returns a dict with ``q`` and ``u`` (modal coefficients, cell-major),
    ``h``, ``n_cells``, ``degree``, ``sigma``, ``ndofs`` and ``condition_number``.
    """
    A, b, meta = assemble_ldg_dirichlet_1d(n_cells, degree, rhs, g_left, g_right, sigma)
    sol = np.linalg.solve(A, b)
    m = degree + 1
    out = dict(meta)
    out["q"] = np.array([sol[e * 2 * m: e * 2 * m + m] for e in range(n_cells)])
    out["u"] = np.array([sol[e * 2 * m + m: e * 2 * m + 2 * m] for e in range(n_cells)])
    out["condition_number"] = float(np.linalg.cond(A))
    return out


def ldg_l2_error_1d(solution: dict, exact, field: str = "u") -> float:
    """``L2`` error of the computed ``u`` or ``q`` field against ``exact(x)``."""
    xi, w, phi, _mass, _dtest, _face = _basis(solution["degree"])
    h = solution["h"]
    coeffs = solution["u"] if field == "u" else solution["q"]
    err2 = 0.0
    for e in range(solution["n_cells"]):
        xc = (e + 0.5) * h
        pts = xc + 0.5 * h * xi
        numeric = coeffs[e] @ phi
        err2 += 0.5 * h * float(np.sum(w * (numeric - exact(pts)) ** 2))
    return math.sqrt(err2)