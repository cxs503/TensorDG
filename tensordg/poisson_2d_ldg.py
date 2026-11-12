"""Local discontinuous Galerkin (LDG) solver for the 2-D Poisson problem.

STATUS: VALIDATED (2-D extension of the verified 1-D LDG closure).
    The Dirichlet closure that carried the structural rank deficiency reported
    in ``docs/ldg_poisson_status.md`` is fixed here with the two ingredients
    derived and verified in one dimension (``tensordg.ldg_1d``,
    ``docs/CASE-DG-LDG-1D-MIXED.md``):

    1. *face-oriented* alternating traces -- on every face the element on the
       lower-index ("-") side supplies ``u_hat`` and the element on the
       higher-index ("+") side supplies ``q_hat`` (``trace_policy="ldg"``);
    2. a *discrete Dirichlet flux* ``q_hat = q_int + beta/h (u_int - g_D)`` on
       boundary faces instead of the bare interior trace ``q_int``.

    With those two the 2-D scheme is algebraically non-singular and converges at
    order ``k + 1`` in the L2 norm for smooth solutions; the h-convergence
    table, the closure ablation, the conditioning audit and the
    polynomial-exactness residual are recorded in ``docs/CASE-DG-LDG-2D.md``
    (artifacts under ``validation/ldg2d/``).  The legacy mirror closure
    ``u_hat = 2 g_D - u_int`` and the bare closure are kept only as ablation
    controls (``boundary_closure="mirror"`` / ``"bare"``) and are *not*
    convergent.

Original module description follows.

The model problem is the Dirichlet problem

    -Delta(u) = f      in Omega = (0,1)^2,
           u  = g_D   on dOmega,

discretised on a uniform Cartesian mesh with a modal (tensor-product Legendre)
DG space of degree ``k``.  Unlike the symmetric interior-penalty (SIPG) scheme
in :mod:`tensordg.poisson_2d`, LDG has no penalty parameter at all: stability
comes from *alternating* numerical traces, so the method is stable as written
and there is nothing to tune.

Formulation (Cockburn & Shu 1998; Castillo, Cockburn, Perugia & Schoetzau 2000)
-------------------------------------------------------------------------------
Introduce the auxiliary unknown ``q = grad(u)``.  On every element ``K`` and for
all test functions ``(v, w)`` the discrete problem reads::

    int_K q_h . v = -int_K u_h div(v) + int_{dK} u_hat (v . n)
    int_K q_h . grad(w) - int_{dK} q_hat . n w = int_K f w

with ``n`` the outward normal of ``K`` and the alternating traces, where ``-``
and ``+`` denote the element with the smaller / larger index in the
face-normal direction::

    u_hat = u_h^+     (trace from the "+" side)
    q_hat = q_h^-     (trace from the "-" side)

On a Dirichlet boundary ``u_hat = g_D`` and ``q_hat = q_h`` (interior trace).
Only the *normal* component of ``q_h`` enters the face terms, so each face
couples a single scalar trace; the global matrix is nonsymmetric but
well conditioned and has no saddle-point structure.  It is assembled in scipy
CSR form and solved with a sparse direct solver.

The scheme converges at order ``k+1`` in the L2 norm for smooth solutions; the
auxiliary variable ``q_h`` converges to ``grad(u)`` at the same order.

References
----------
* B. Cockburn, C.-W. Shu (1998). "The local discontinuous Galerkin method for
  time-dependent convection-diffusion systems." SIAM J. Numer. Anal. 35:2440.
* P. Castillo, B. Cockburn, I. Perugia, D. Schoetzau (2000). "An a priori error
  analysis of the local discontinuous Galerkin method for elliptic problems."
  SIAM J. Numer. Anal. 38:1676.  DOI:10.1137/S0036142900371003
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from tensordg.convergence import observed_order

__all__ = [
    "manufactured_solution",
    "superposition_solution",
    "solve_poisson_2d_ldg",
    "ldg_poisson_convergence",
]


# ---------------------------------------------------------------------------
# Reference-element operators
# ---------------------------------------------------------------------------

def _basis(degree: int):
    """Modal Legendre basis, mass matrix and derivative matrix on [-1, 1]."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    nq = max(2 * degree + 3, 6)
    nodes, weights = np.polynomial.legendre.leggauss(nq)
    modes = degree + 1
    phi = np.zeros((modes, nq))
    dphi = np.zeros((modes, nq))
    for k in range(modes):
        coeffs = np.zeros(k + 1)
        coeffs[-1] = 1.0
        phi[k] = np.polynomial.legendre.legval(nodes, coeffs)
        dphi[k] = np.polynomial.legendre.legval(
            nodes, np.polynomial.legendre.legder(coeffs),
        )
    mass = np.einsum("q,iq,kq->ik", weights, phi, phi)
    # deriv[i, k] = int_{-1}^{1} L'_i L_k dxi
    deriv = np.einsum("q,iq,kq->ik", weights, dphi, phi)
    return nodes, weights, phi, dphi, mass, deriv


def _kron2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tensor-product block with local index ``(i, j) -> i*m + j``."""
    modes = a.shape[0]
    return np.einsum("ik,jl->ijkl", a, b).reshape(modes * modes, modes * modes)


def _face_mass(
    endpoint: np.ndarray,
    mass: np.ndarray,
    scale: float,
    direction: str = "x",
) -> np.ndarray:
    """Face mass ``int_face phi_{ij} phi_{kl} ds`` for an endpoint trace.

    ``direction`` is the face-normal direction: the endpoint trace multiplies
    the tensor factor that is *constant* across the face -- the ``x`` factor for
    a face whose normal is ``x`` (and the ``y`` factor for a ``y`` face).
    """
    outer = np.outer(endpoint, endpoint)
    return scale * (
        _kron2(outer, mass) if direction == "x" else _kron2(mass, outer)
    )


def _face_operator(
    test_endpoint: np.ndarray,
    coef_endpoint: np.ndarray,
    mass: np.ndarray,
    scale: float,
    direction: str = "x",
) -> np.ndarray:
    """Face term for a trace taken from *another* element.

    The test function is evaluated at the endpoint of the element that owns the
    row, while the traced coefficient comes from the element that owns the
    trace; on a shared face those two endpoints are generally *different*
    (``+1`` for the left neighbour, ``-1`` for the right one).  Using the same
    endpoint vector for both factors silently replaces ``L_k(-1)=(-1)^k`` by
    ``L_k(+1)=1`` on half of the modes -- a defect that leaves the scheme
    non-singular but order-destroying.
    """
    outer = np.outer(test_endpoint, coef_endpoint)
    return scale * (
        _kron2(outer, mass) if direction == "x" else _kron2(mass, outer)
    )


# ---------------------------------------------------------------------------
# Manufactured solution (default Dirichlet benchmark)
# ---------------------------------------------------------------------------

def manufactured_solution() -> Dict[str, Callable]:
    """``u = sin(pi x) sin(pi y)`` with homogeneous Dirichlet data.

    The right-hand side is ``f = 2 pi^2 sin(pi x) sin(pi y)``.
    """
    pi = math.pi
    return {
        "u": lambda x, y: np.sin(pi * x) * np.sin(pi * y),
        "grad_x": lambda x, y: pi * np.cos(pi * x) * np.sin(pi * y),
        "grad_y": lambda x, y: pi * np.sin(pi * x) * np.cos(pi * y),
        "f": lambda x, y: 2.0 * pi * pi * np.sin(pi * x) * np.sin(pi * y),
        "dirichlet": lambda x, y: np.zeros(np.shape(x)),
    }


def superposition_solution() -> Dict[str, Callable]:
    """Non-homogeneous Dirichlet test: ``u = sin(pi x) sin(pi y) + x``.

    ``Delta(u)`` is unchanged, but the boundary data is ``g_D = x`` on
    ``dOmega`` -- used to prove that the boundary trace term is not silently
    dropped (a homogeneous-data-only implementation would fail this test).
    """
    pi = math.pi
    return {
        "u": lambda x, y: np.sin(pi * x) * np.sin(pi * y) + np.asarray(x, dtype=float),
        "grad_x": lambda x, y: pi * np.cos(pi * x) * np.sin(pi * y) + 1.0,
        "grad_y": lambda x, y: pi * np.sin(pi * x) * np.cos(pi * y),
        "f": lambda x, y: 2.0 * pi * pi * np.sin(pi * x) * np.sin(pi * y),
        "dirichlet": lambda x, y: np.asarray(x, dtype=float),
    }


# ---------------------------------------------------------------------------
# Assembly + solve
# ---------------------------------------------------------------------------

def solve_poisson_2d_ldg(
    ncells: int | Sequence[int] = 16,
    degree: int = 1,
    *,
    rhs: Callable | None = None,
    dirichlet: Callable | None = None,
    exact: Callable | None = None,
    exact_grad: tuple[Callable, Callable] | None = None,
    trace_policy: str = "ldg",
    boundary_closure: str = "flux",
    dirichlet_flux_beta: float = 4.0,
    return_matrix: bool = False,
) -> Dict[str, object]:
    """Solve the 2-D Dirichlet Poisson problem with the LDG method.

    Parameters
    ----------
    ncells : int or (nx, ny)
        Number of Cartesian cells in each direction.
    degree : int
        Polynomial degree ``k`` of the modal Legendre DG space.
    rhs, dirichlet, exact, exact_grad : callables, optional
        Manufactured data.  Defaults are the ``sin(pi x) sin(pi y)`` benchmark
        (``rhs = 2 pi^2 sin sin``, zero Dirichlet data).  ``exact`` /
        ``exact_grad`` are used only to evaluate the error.

    Returns
    -------
    dict
        ``u``, ``qx``, ``qy`` (modal coefficients shaped ``(nx, ny, k+1, k+1)``),
        ``l2_relative_error``, ``grad_relative_error``, ``h``, ``ndofs``,
        ``ncells``, ``degree`` and the element-centre coordinates ``xc``/``yc``.
        With ``return_matrix=True`` the assembled sparse matrix and right-hand
        side are returned (``matrix``/``rhs``) instead, which is what the rank
        and conditioning audits use.

    trace_policy : {"ldg", "reversed"}
        Interior trace orientation.  ``"ldg"`` (default) is the verified
        face-oriented alternating choice; ``"reversed"`` is the ablation
        control in which ``u_hat`` and ``q_hat`` are taken from the other side.
    boundary_closure : {"flux", "mirror", "bare"}
        Dirichlet-face closure for ``q_hat``.  ``"flux"`` (default) uses the
        discrete flux ``q_int + beta/h (u_int - g_D)``; ``"mirror"`` is the
        legacy prototype closure ``u_hat = 2 g_D - u_int, q_hat = q_int`` and
        ``"bare"`` takes ``u_hat = g_D, q_hat = q_int``.  The last two are
        singular (or catastrophically inaccurate) and exist only as controls.
    dirichlet_flux_beta : float
        Stabilisation strength ``beta > 0`` of the discrete Dirichlet flux; it
        does not affect the convergence order (default 4).
    """
    if isinstance(ncells, int):
        nx = ny = ncells
    else:
        nx, ny = (int(n) for n in ncells)
    if nx < 1 or ny < 1:
        raise ValueError("ncells must be >= 1 in each direction")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if trace_policy not in ("ldg", "reversed"):
        raise ValueError("trace_policy must be 'ldg' or 'reversed'")
    if boundary_closure not in ("flux", "mirror", "bare"):
        raise ValueError("boundary_closure must be 'flux', 'mirror' or 'bare'")
    if boundary_closure == "flux" and not dirichlet_flux_beta > 0.0:
        raise ValueError("dirichlet_flux_beta must be positive for the flux closure")

    default = manufactured_solution()
    rhs = rhs if rhs is not None else default["f"]
    dirichlet = dirichlet if dirichlet is not None else default["dirichlet"]
    exact = exact if exact is not None else default["u"]
    if exact_grad is None:
        exact_grad = (default["grad_x"], default["grad_y"])

    hx, hy = 1.0 / nx, 1.0 / ny
    nodes, weights, phi, dphi, mass, deriv = _basis(degree)
    modes = degree + 1
    block = modes * modes
    nq = nodes.size

    # Endpoint traces L_k(-1), L_k(+1) used by the face mass matrices.
    left = np.array([(-1.0) ** k for k in range(modes)])
    right = np.ones(modes)

    # ---- volume blocks -----------------------------------------------------
    mass2 = (0.5 * hx) * (0.5 * hy) * _kron2(mass, mass)
    # A[(i,j),(k,l)] = int_K d(phi_{ij})/dx phi_{kl}: derivative on the *test*
    # index (i,j), since both the gradient equation and the divergence equation
    # differentiate the test function.
    gx = (0.5 * hy) * _kron2(deriv, mass)
    gy = (0.5 * hx) * _kron2(mass, deriv)

    # ---- face mass matrices ------------------------------------------------
    mfx_left = _face_mass(left, mass, 0.5 * hy)
    mfx_right = _face_mass(right, mass, 0.5 * hy)
    mfy_left = _face_mass(left, mass, 0.5 * hx, "y")
    mfy_right = _face_mass(right, mass, 0.5 * hx, "y")

    # ---- sparse assembly ---------------------------------------------------
    ndofs = nx * ny * 3 * block
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    local = np.arange(block)

    def _block(er: int, br: int, ec: int, bc: int, data: np.ndarray):
        row = er * 3 * block + br * block + local
        col = ec * 3 * block + bc * block + local
        rows.append(np.repeat(row, block))
        cols.append(np.tile(col, block))
        vals.append(np.asarray(data, dtype=float).ravel())

    rhs_vec = np.zeros(ndofs)

    # Quadrature points (physical) for the source projection and error norms.
    xq = np.broadcast_to(
        (np.arange(nx)[:, None, None, None]
         + 0.5 * (nodes[None, None, :, None] + 1.0)) * hx,
        (nx, ny, nq, nq),
    )
    yq = np.broadcast_to(
        (np.arange(ny)[None, :, None, None]
         + 0.5 * (nodes[None, None, None, :] + 1.0)) * hy,
        (nx, ny, nq, nq),
    )
    weight2 = np.einsum("q,r->qr", weights, weights) * (0.5 * hx) * (0.5 * hy)

    fq = np.asarray(rhs(xq, yq), dtype=float)
    # F[(i,j)] = sum_{q,r} w2 f phi_i(xi_q) phi_j(eta_r)
    rhs_all = np.einsum("qr,efqr,iq,jr->efij", weight2, fq, phi, phi).reshape(
        nx * ny, block,
    )
    u_index = (np.arange(nx * ny)[:, None] * 3 * block
               + np.arange(block)[None, :])

    for ix in range(nx):
        for iy in range(ny):
            e = ix * ny + iy
            # qx row:  M qx + A u - [u_hat] = 0
            _block(e, 1, e, 0, gx)
            _block(e, 1, e, 1, mass2)
            # qy row:  M qy + Ay u - [u_hat] = 0
            _block(e, 2, e, 0, gy)
            _block(e, 2, e, 2, mass2)
            # u row:   A qx + Ay qy - [q_hat] = f
            _block(e, 0, e, 1, gx)
            _block(e, 0, e, 2, gy)
            rhs_vec[u_index[e]] += rhs_all[e]

            # ---- x-faces ---------------------------------------------------
            for side, sign, endpoint_self in (
                ("left", -1.0, left),
                ("right", +1.0, right),
            ):
                nbr = ix - 1 if side == "left" else ix + 1
                endpoint_nbr = right if side == "left" else left
                if 0 <= nbr < nx:
                    en = nbr * ny + iy
                    # Face-oriented alternating traces: on a face the element on
                    # the lower-index ("-") side contributes u_hat, the element
                    # on the higher-index ("+") side contributes q_hat.  This is
                    # the orientation verified in 1-D (tensordg.ldg_1d,
                    # docs/CASE-DG-LDG-1D-MIXED.md); "reversed" is an ablation.
                    if trace_policy == "ldg":
                        u_elem, q_elem = (en, e) if side == "left" else (e, en)
                    else:
                        u_elem, q_elem = (e, en) if side == "left" else (en, e)
                    u_ep = endpoint_nbr if u_elem == en else endpoint_self
                    q_ep = endpoint_nbr if q_elem == en else endpoint_self
                    _block(e, 1, u_elem, 0,
                           -sign * _face_operator(endpoint_self, u_ep, mass, 0.5 * hy))
                    _block(e, 0, q_elem, 1,
                           -sign * _face_operator(endpoint_self, q_ep, mass, 0.5 * hy))
                else:
                    x_face = 0.0 if side == "left" else 1.0
                    y_pts = ((iy + 0.5) + 0.5 * nodes) * hy
                    gvals = np.asarray(
                        dirichlet(np.full_like(y_pts, x_face), y_pts), dtype=float,
                    )
                    endpoint = endpoint_self
                    mfx = _face_operator(endpoint, endpoint, mass, 0.5 * hy)
                    gf = (0.5 * hy) * np.einsum("q,q,jq->j", weights, gvals, phi)
                    # load vector of the trace: int_face g phi_ij ds
                    trace = np.outer(endpoint, gf).reshape(block)
                    rhs_qx = e * 3 * block + block
                    rhs_u = e * 3 * block
                    if boundary_closure == "mirror":
                        # legacy prototype closure: u_hat = 2 g_D - u_int
                        _block(e, 1, e, 0, sign * mfx)
                        rhs_vec[rhs_qx: rhs_qx + block] += 2.0 * sign * trace
                    else:
                        # u_hat = g_D on the Dirichlet face (flux and bare)
                        rhs_vec[rhs_qx: rhs_qx + block] += sign * trace
                    # q_hat = q_int (+ discrete Dirichlet flux for "flux")
                    _block(e, 0, e, 1, -sign * mfx)
                    if boundary_closure == "flux":
                        beta_x = dirichlet_flux_beta / hx
                        _block(e, 0, e, 0, -sign * beta_x * mfx)
                        rhs_vec[rhs_u: rhs_u + block] += -sign * beta_x * trace

            # ---- y-faces ---------------------------------------------------
            for side, sign, endpoint_self in (
                ("left", -1.0, left),
                ("right", +1.0, right),
            ):
                nbr = iy - 1 if side == "left" else iy + 1
                endpoint_nbr = right if side == "left" else left
                if 0 <= nbr < ny:
                    en = ix * ny + nbr
                    if trace_policy == "ldg":
                        u_elem, q_elem = (en, e) if side == "left" else (e, en)
                    else:
                        u_elem, q_elem = (e, en) if side == "left" else (en, e)
                    u_ep = endpoint_nbr if u_elem == en else endpoint_self
                    q_ep = endpoint_nbr if q_elem == en else endpoint_self
                    _block(e, 2, u_elem, 0,
                           -sign * _face_operator(endpoint_self, u_ep, mass, 0.5 * hx, "y"))
                    _block(e, 0, q_elem, 2,
                           -sign * _face_operator(endpoint_self, q_ep, mass, 0.5 * hx, "y"))
                else:
                    y_face = 0.0 if side == "left" else 1.0
                    x_pts = ((ix + 0.5) + 0.5 * nodes) * hx
                    gvals = np.asarray(
                        dirichlet(x_pts, np.full_like(x_pts, y_face)), dtype=float,
                    )
                    endpoint = endpoint_self
                    mfy = _face_operator(endpoint, endpoint, mass, 0.5 * hx, "y")
                    gf = (0.5 * hx) * np.einsum("q,q,iq->i", weights, gvals, phi)
                    trace = np.outer(gf, endpoint).reshape(block)
                    rhs_qy = e * 3 * block + 2 * block
                    rhs_u = e * 3 * block
                    if boundary_closure == "mirror":
                        _block(e, 2, e, 0, sign * mfy)
                        rhs_vec[rhs_qy: rhs_qy + block] += 2.0 * sign * trace
                    else:
                        rhs_vec[rhs_qy: rhs_qy + block] += sign * trace
                    _block(e, 0, e, 2, -sign * mfy)
                    if boundary_closure == "flux":
                        beta_y = dirichlet_flux_beta / hy
                        _block(e, 0, e, 0, -sign * beta_y * mfy)
                        rhs_vec[rhs_u: rhs_u + block] += -sign * beta_y * trace

    matrix = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ndofs, ndofs),
    ).tocsr()
    if return_matrix:
        return {
            "matrix": matrix,
            "rhs": rhs_vec,
            "ndofs": int(ndofs),
            "nnz": int(matrix.nnz),
            "ncells": (nx, ny),
            "degree": degree,
            "h": max(hx, hy),
            "trace_policy": trace_policy,
            "boundary_closure": boundary_closure,
            "dirichlet_flux_beta": float(dirichlet_flux_beta),
        }
    solution = spla.spsolve(matrix, rhs_vec)
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("LDG linear solve produced a non-finite solution")

    shaped = solution.reshape(nx * ny, 3, block)
    u = shaped[:, 0, :].reshape(nx, ny, modes, modes)
    qx = shaped[:, 1, :].reshape(nx, ny, modes, modes)
    qy = shaped[:, 2, :].reshape(nx, ny, modes, modes)

    # ---- error norms -------------------------------------------------------
    uq = np.einsum("efmn,mq,nr->efqr", u, phi, phi)
    qxq = np.einsum("efmn,mq,nr->efqr", qx, phi, phi)
    qyq = np.einsum("efmn,mq,nr->efqr", qy, phi, phi)

    u_ex = np.asarray(exact(xq, yq), dtype=float)
    gx_ex = np.asarray(exact_grad[0](xq, yq), dtype=float)
    gy_ex = np.asarray(exact_grad[1](xq, yq), dtype=float)

    def _norm(a):
        return float(np.sqrt(np.einsum("qr,ijqr,ijqr->", weight2, a, a)))

    l2_abs = _norm(uq - u_ex)
    grad_abs = math.hypot(_norm(qxq - gx_ex), _norm(qyq - gy_ex))

    return {
        "u": u,
        "qx": qx,
        "qy": qy,
        "l2_absolute_error": l2_abs,
        "l2_relative_error": l2_abs / _norm(u_ex),
        "grad_relative_error": grad_abs / math.hypot(_norm(gx_ex), _norm(gy_ex)),
        "h": max(hx, hy),
        "hx": hx,
        "hy": hy,
        "ndofs": int(ndofs),
        "nnz": int(matrix.nnz),
        "ncells": (nx, ny),
        "degree": degree,
        "trace_policy": trace_policy,
        "boundary_closure": boundary_closure,
        "dirichlet_flux_beta": float(dirichlet_flux_beta),
        "xc": (np.arange(nx) + 0.5) * hx,
        "yc": (np.arange(ny) + 0.5) * hy,
    }


# ---------------------------------------------------------------------------
# Convergence study (the ledger acceptance metric)
# ---------------------------------------------------------------------------

def ldg_poisson_convergence(
    degrees: Iterable[int] = (1, 2),
    ncells: Iterable[int] = (8, 16, 32),
    **kwargs,
) -> Dict[str, object]:
    """h-convergence table for the manufactured Poisson benchmark.

    Returns ``{"rows": [...], "orders": {degree: [sigma_12, sigma_23]}}`` where
    the entries are the log-log slopes of the relative L2 error between
    consecutive meshes -- the ``CASE-DG-LDG-POISSON`` acceptance metric.
    """
    mesh = [int(n) for n in ncells]
    rows = []
    orders: Dict[int, list[float]] = {}
    for p in degrees:
        errors = []
        for n in mesh:
            result = solve_poisson_2d_ldg(n, p, **kwargs)
            errors.append(result["l2_relative_error"])
            rows.append({
                "degree": p,
                "ncells": n,
                "ndofs": result["ndofs"],
                "l2_relative_error": result["l2_relative_error"],
                "grad_relative_error": result["grad_relative_error"],
            })
        orders[p] = [
            float(math.log(errors[i] / errors[i + 1]) / math.log(mesh[i + 1] / mesh[i]))
            for i in range(len(mesh) - 1)
        ]
    return {
        "rows": rows,
        "orders": orders,
        "orders_least_squares": {
            p: float(observed_order(mesh, [r["l2_relative_error"]
                                          for r in rows if r["degree"] == p]))
            for p in orders
        },
        "ncells": mesh,
    }