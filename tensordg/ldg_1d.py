"""Local discontinuous Galerkin (LDG) solver for the 1-D Poisson problem.

Model problem::

    -u''(x) = f(x),  x in (0, 1),     u(0) = g0,  u(1) = g1

or the periodic variant (f must have zero mean; the constant mode is pinned by a
Lagrange multiplier so that the system stays non-singular).

Mixed (first-order) form: introduce the auxiliary unknown ``q = u'``.  With
``(r, v)`` test functions and ``K`` a mesh element::

    int_K q r + int_K u r'  - [u_hat r]_{dK} = 0
    int_K q v' - [q_hat v]_{dK}             = int_K f v

Numerical traces (face-oriented alternating choice, Cockburn & Shu 1998;
Castillo, Cockburn, Perugia & Schoetzau 2000).  On every face let the element on
the left of the face be ``-`` and the one on the right be ``+``::

    u_hat = u^-        q_hat = q^+

This orientation -- *not* the per-element "own/neighbour" reading -- is what
reproduces the classical staggered scheme for ``k = 0``
(``h q_e = u_e - u_{e-1}``, ``q_e - q_{e+1} = h f_e``) and is the one that is
both well conditioned and of order ``k + 1``; see
``docs/CASE-DG-LDG-1D-MIXED.md`` for the four-orientation ablation and the
conditioning audit.

Dirichlet faces need one extra ingredient.  ``u_hat = g_D`` as usual, but the
flux must be the *discrete* flux

    q_hat = q_int + beta/h (u_int - g_D)          (beta > 0)

and not the bare interior trace ``q_int``.  With ``beta = 0`` the last equation
of a ``k = 0`` discretisation reads ``0 = h f`` (derivation in the case
document), i.e. the matrix is singular -- this is the rank deficiency reported
against an earlier probe, and it is a *boundary closure* defect, not an interior
one.  ``beta`` does not affect the convergence order (``beta in [2, 10]`` gives
the same orders to three digits); it restores invertibility and improves the
error constant.

The scheme is consistent for any trace in the alternating family (the traces
coincide for a continuous exact solution), so a polynomial-exactness residual
*cannot* discriminate between orientations -- that trap is documented in the case
document and gated in ``tests/test_ldg_1d.py``.

References
----------
* B. Cockburn, C.-W. Shu (1998), SIAM J. Numer. Anal. 35:2440.
* P. Castillo, B. Cockburn, I. Perugia, D. Schoetzau (2000),
  SIAM J. Numer. Anal. 38:1676, DOI:10.1137/S0036142900371003.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from tensordg.convergence import observed_order

__all__ = [
    "ReferenceOperators",
    "reference_operators",
    "LDG1DResult",
    "assemble_1d",
    "solve_ldg_1d",
    "convergence_study",
    "TRACE_POLICIES",
]

#: Trace orientations.  ``"ldg"`` is the verified scheme; the other three
#: reproduce (and are kept only for) the ablation evidence.
TRACE_POLICIES = ("ldg", "per_element", "far_both", "self_endpoint")


@dataclass(frozen=True)
class ReferenceOperators:
    """Modal Legendre operators on the reference element ``[-1, 1]``."""

    xi: np.ndarray            # Gauss nodes
    weights: np.ndarray       # Gauss weights
    phi: np.ndarray           # basis[j, q] = L_j(xi_q)
    dphi: np.ndarray          # basis derivative
    mass: np.ndarray          # mass[i, j] = int L_i L_j
    derivative_test: np.ndarray   # D[i, j] = int L_i' L_j   (derivative on the test function)
    derivative_coef: np.ndarray   # C[i, j] = int L_i L_j'   (derivative on the coefficient)
    left: np.ndarray          # L_j(-1) = (-1)^j
    right: np.ndarray         # L_j(+1) = 1

    @property
    def degree(self) -> int:
        return self.phi.shape[0] - 1


@lru_cache(maxsize=None)
def reference_operators(degree: int) -> ReferenceOperators:
    """Legendre operators on ``[-1, 1]`` with exact (Gauss) quadrature."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    nq = max(2 * degree + 4, 10)
    xi, weights = np.polynomial.legendre.leggauss(nq)
    modes = degree + 1
    phi = np.zeros((modes, nq))
    dphi = np.zeros((modes, nq))
    for j in range(modes):
        coeffs = np.zeros(j + 1)
        coeffs[-1] = 1.0
        phi[j] = np.polynomial.legendre.legval(xi, coeffs)
        dcoeffs = np.polynomial.legendre.legder(coeffs)
        dphi[j] = np.polynomial.legendre.legval(xi, dcoeffs) if dcoeffs.size else 0.0
    mass = np.einsum("q,iq,jq->ij", weights, phi, phi)
    derivative_test = np.einsum("q,iq,jq->ij", weights, dphi, phi)
    derivative_coef = np.einsum("q,iq,jq->ij", weights, phi, dphi)
    return ReferenceOperators(
        xi=xi,
        weights=weights,
        phi=phi,
        dphi=dphi,
        mass=mass,
        derivative_test=derivative_test,
        derivative_coef=derivative_coef,
        left=np.array([(-1.0) ** j for j in range(modes)]),
        right=np.ones(modes),
    )


@dataclass
class LDG1DResult:
    """Discrete solution plus the diagnostics needed by the verification gates."""

    coefficients: np.ndarray
    n_elements: int
    degree: int
    length: float
    beta: float
    trace_policy: str
    boundary: str
    matrix: sp.spmatrix = field(repr=False, default=None)
    ops: ReferenceOperators = field(repr=False, default=None)

    # -- reconstruction ----------------------------------------------------
    def _element_coefficients(self, e: int, offset: int) -> np.ndarray:
        m = self.degree + 1
        start = e * 2 * m + offset
        return self.coefficients[start:start + m]

    def _evaluate(self, field_name: str, x: np.ndarray, points_per_element: int = 0) -> np.ndarray:
        ops = self.ops
        m = self.degree + 1
        h = self.length / self.n_elements
        offset = 0 if field_name == "q" else m
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.empty_like(x)
        idx = np.clip((x / h).astype(int), 0, self.n_elements - 1)
        for e in np.unique(idx):
            sel = idx == e
            xc = (e + 0.5) * h
            xi = 2.0 * (x[sel] - xc) / h
            basis = np.polynomial.legendre.legvander(xi, self.degree)
            out[sel] = basis @ self._element_coefficients(e, offset)
        return out

    def u(self, x) -> np.ndarray:
        """Reconstruct ``u_h`` at the physical points ``x``."""
        return self._evaluate("u", x)

    def q(self, x) -> np.ndarray:
        """Reconstruct ``q_h ~ u'`` at the physical points ``x``."""
        return self._evaluate("q", x)

    # -- error norms -------------------------------------------------------
    def _l2_error(self, field_name: str, exact: Callable[[np.ndarray], np.ndarray],
                  nq: int | None = None) -> float:
        ops = self.ops
        m = self.degree + 1
        order = nq or (2 * self.degree + 8)
        xi, w = np.polynomial.legendre.leggauss(max(order, 8))
        basis = np.polynomial.legendre.legvander(xi, self.degree)
        h = self.length / self.n_elements
        offset = 0 if field_name == "q" else m
        err2 = 0.0
        for e in range(self.n_elements):
            xc = (e + 0.5) * h
            x = xc + 0.5 * h * xi
            uh = basis @ self._element_coefficients(e, offset)
            err2 += 0.5 * h * np.sum(w * (uh - exact(x)) ** 2)
        return math.sqrt(err2)

    def l2_error_u(self, exact: Callable[[np.ndarray], np.ndarray]) -> float:
        return self._l2_error("u", exact)

    def l2_error_q(self, exact: Callable[[np.ndarray], np.ndarray]) -> float:
        return self._l2_error("q", exact)

    @property
    def condition_number(self) -> float:
        """1-norm condition number (LU based, cheap enough for the mesh ladder)."""
        dense = self.matrix.toarray() if sp.issparse(self.matrix) else np.asarray(self.matrix)
        return float(np.linalg.cond(dense, 1))


def _trace_endpoint(ops: ReferenceOperators, on_left_of_face: bool) -> np.ndarray:
    """Endpoint values of the trace basis of an element on one side of a face."""
    return ops.right if on_left_of_face else ops.left


def assemble_1d(
    n_elements: int,
    degree: int,
    rhs: Callable[[np.ndarray], np.ndarray],
    *,
    length: float = 1.0,
    boundary: str = "dirichlet",
    g0: float = 0.0,
    g1: float = 0.0,
    beta: float = 2.0,
    beta_sign: float = 1.0,
    trace_policy: str = "ldg",
) -> tuple[sp.spmatrix, np.ndarray, ReferenceOperators]:
    """Assemble the mixed-form LDG system for ``-u'' = f``.

    ``trace_policy`` selects the trace orientation.  ``"ldg"`` is the verified
    face-oriented alternating choice; ``"per_element"`` takes ``u_hat`` from the
    element itself and ``q_hat`` from its neighbour on *both* faces;
    ``"far_both"`` takes both traces from the neighbouring element;
    ``"self_endpoint"`` keeps the ``"ldg"`` sides but expands the neighbour trace
    in the *own* endpoint basis instead of the neighbour's facing one.  The last
    three exist to document why those conventions fail (see the case document).
    """
    if n_elements < 1:
        raise ValueError("n_elements must be >= 1")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if trace_policy not in TRACE_POLICIES:
        raise ValueError(f"unknown trace_policy {trace_policy!r}")
    if boundary not in ("dirichlet", "periodic"):
        raise ValueError(f"unknown boundary {boundary!r}")

    ops = reference_operators(degree)
    m = degree + 1
    h = length / n_elements
    nd = 2 * n_elements * m
    periodic = boundary == "periodic"
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    force = np.zeros(nd)

    def add(r0: int, c0: int, block: np.ndarray) -> None:
        mm, nn = block.shape
        for i in range(mm):
            for j in range(nn):
                if block[i, j] != 0.0:
                    rows.append(r0 + i)
                    cols.append(c0 + j)
                    vals.append(float(block[i, j]))

    idx_q = lambda e: e * 2 * m
    idx_u = lambda e: e * 2 * m + m

    for e in range(n_elements):
        xc = (e + 0.5) * h
        xq = xc + 0.5 * h * ops.xi
        add(idx_q(e), idx_q(e), 0.5 * h * ops.mass)
        add(idx_q(e), idx_u(e), ops.derivative_test)
        add(idx_u(e), idx_q(e), ops.derivative_test)
        force[idx_u(e):idx_u(e) + m] += 0.5 * h * (ops.weights * rhs(xq)) @ ops.phi.T

        for side in ("l", "r"):
            own_end = ops.left if side == "l" else ops.right
            normal = -1.0 if side == "l" else 1.0
            far = e - 1 if side == "l" else e + 1
            if periodic:
                far %= n_elements
                has_far = True
            else:
                has_far = 0 <= far < n_elements

            # ---- which element supplies each trace -----------------------
            if trace_policy == "ldg":
                # face-oriented: u_hat from the element on the left of the face,
                # q_hat from the element on the right of the face.
                u_own = side == "r"
                q_own = side == "l"
            elif trace_policy == "per_element":
                u_own = True
                q_own = False
            elif trace_policy == "far_both":
                u_own = False
                q_own = False
            else:  # self_endpoint
                u_own = side == "r"
                q_own = side == "l"

            def endpoint(from_own: bool, side: str = side) -> np.ndarray:
                if trace_policy == "self_endpoint":
                    return own_end
                if from_own:
                    return own_end
                return ops.right if side == "l" else ops.left

            if has_far:
                u_tgt = idx_u(e) if u_own else idx_u(far)
                q_tgt = idx_q(e) if q_own else idx_q(far)
                add(idx_q(e), u_tgt, -normal * np.outer(own_end, endpoint(u_own)))
                add(idx_u(e), q_tgt, -normal * np.outer(own_end, endpoint(q_own)))
            else:
                gv = g0 if side == "l" else g1
                force[idx_q(e):idx_q(e) + m] += normal * own_end * gv
                add(idx_u(e), idx_q(e), -normal * np.outer(own_end, own_end))
                if beta:
                    add(idx_u(e), idx_u(e), -normal * beta_sign * beta / h * np.outer(own_end, own_end))
                    force[idx_u(e):idx_u(e) + m] += -normal * beta_sign * beta / h * own_end * gv

    matrix = sp.coo_matrix((vals, (rows, cols)), shape=(nd, nd)).tocsr()

    if periodic:
        # f must have zero mean; pin int u_h = 0 with one Lagrange multiplier.
        const = np.zeros(nd)
        for e in range(n_elements):
            const[idx_u(e)] = h  # int_K L_0 dx (L_0 = 1 on the reference element)
        bordered = sp.bmat(
            [[matrix, sp.csr_matrix(const.reshape(-1, 1))],
             [sp.csr_matrix(const.reshape(1, -1)), None]],
            format="csr",
        )
        force = np.concatenate([force, [0.0]])
        return bordered, force, ops
    return matrix, force, ops


def solve_ldg_1d(
    n_elements: int,
    degree: int,
    rhs: Callable[[np.ndarray], np.ndarray],
    *,
    length: float = 1.0,
    boundary: str = "dirichlet",
    g0: float = 0.0,
    g1: float = 0.0,
    beta: float = 2.0,
    beta_sign: float = 1.0,
    trace_policy: str = "ldg",
) -> LDG1DResult:
    """Assemble and solve the mixed-form LDG discretisation of ``-u'' = f``."""
    matrix, force, ops = assemble_1d(
        n_elements, degree, rhs, length=length, boundary=boundary,
        g0=g0, g1=g1, beta=beta, beta_sign=beta_sign, trace_policy=trace_policy,
    )
    coefficients = spla.spsolve(matrix.tocsc(), force)
    if not np.all(np.isfinite(coefficients)):
        raise FloatingPointError("LDG solve produced a non-finite solution")
    if boundary == "periodic":
        coefficients = coefficients[:-1]
    return LDG1DResult(
        coefficients=coefficients,
        n_elements=n_elements,
        degree=degree,
        length=length,
        beta=beta,
        trace_policy=trace_policy,
        boundary=boundary,
        matrix=matrix,
        ops=ops,
    )


def convergence_study(
    degrees: Sequence[int],
    resolutions: Sequence[int],
    rhs: Callable[[np.ndarray], np.ndarray],
    exact: Callable[[np.ndarray], np.ndarray],
    exact_q: Callable[[np.ndarray], np.ndarray],
    **kwargs,
) -> list[dict]:
    """L2 errors and observed orders for ``u`` and ``q`` over a mesh ladder."""
    resolutions = list(resolutions)
    rows: list[dict] = []
    for degree in degrees:
        err_u, err_q, cond = [], [], []
        for n in resolutions:
            result = solve_ldg_1d(n, degree, rhs, **kwargs)
            err_u.append(result.l2_error_u(exact))
            err_q.append(result.l2_error_q(exact_q))
            cond.append(result.condition_number)
        rows.append({
            "degree": degree,
            "resolutions": resolutions,
            "l2_u": err_u,
            "l2_q": err_q,
            "order_u": observed_order(resolutions, err_u),
            "order_q": observed_order(resolutions, err_q),
            "condition_numbers": cond,
            "expected_order": degree + 1,
        })
    return rows