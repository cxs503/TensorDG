"""Curved (isoparametric) geometry layer and verification for TensorDG.

Every previous TensorDG benchmark lives on a uniform Cartesian mesh: the
Jacobian is constant, the metric terms are trivial, and geometry never limits
the convergence rate.  On a *curved* mesh a DG scheme only keeps its formal
order ``p+1`` if three geometric conditions hold:

1. **Free-stream preservation (FSP)** -- a constant state must be preserved
   exactly by the semi-discrete operator, i.e. the discrete divergence of the
   metric fluxes has to vanish *with the same discrete derivative operator that
   is used for the flux divergence* (Kopriva & Batta, JCP 2014).
2. **Metric (Piola) identity** -- ``d/dxi_i (J dxi_i/dx_j) = 0`` and
   ``dxi_i/dx_j dx_j/dxi_k = delta_ik`` must hold in the discrete sense.
3. **Curved-wall boundary treatment** -- the wall normal must come from the
   isoparametric geometry, otherwise the wall leaks and the order collapses.

This module supplies the geometry layer (``IsoparametricMap``), the metric
checks, a strong-form DG discretization of scalar linear advection on
curvilinear meshes with exact (over-integrated) volume quadrature, and a
NURBS/CAD geometry contract (``tensordg.curved_geometry.v1``) so that a CAD
surface parameterization (control points + knot vectors + weights) can be
turned into curved DG element geometry.

References
----------
* D. A. Kopriva, "Metric identities and the discontinuous Galerkin spectral
  element method on curvilinear meshes", J. Sci. Comput. 26 (2006) 301-327.
* D. A. Kopriva, G. J. Gassner, "An energy stable DG spectral element method
  for the ...", and Kopriva & Batta, JCP 270 (2014) 550-565 (free-stream
  preservation with the metric identities).
* J. S. Hesthaven, T. Warburton, *Nodal Discontinuous Galerkin Methods*,
  Springer 2008, ch. 6 (curvilinear elements).
* T. J. R. Hughes, *The Finite Element Method* (isoparametric concept, Piola
  identity for the cofactor matrix).

All arithmetic is float64 NumPy.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "legendre_eval_matrix",
    "legendre_derivative_matrix",
    "gll_nodes_weights",
    "gll_differentiation_matrix",
    "gauss_nodes_weights",
    "gauss_differentiation_matrix",
    "IsoparametricMap",
    "metric_identity_report",
    "CurvedMesh",
    "CurvedAdvection",
    "warped_periodic_mesh",
    "cartesian_periodic_mesh",
    "annulus_mesh",
    "bspline_basis_matrix",
    "nurbs_curve_points",
    "circle_contract",
    "validate_geometry_contract",
    "mesh_from_contract",
    "CONTRACT_SCHEMA",
]

CONTRACT_SCHEMA = "tensordg.curved_geometry.v1"


# ---------------------------------------------------------------------------
# 1-D Legendre machinery
# ---------------------------------------------------------------------------
def legendre_eval_matrix(degree: int, x) -> np.ndarray:
    """``L[q, k] = P_k(x_q)`` for ``k = 0 .. degree`` (unnormalised Legendre)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    L = np.zeros((x.size, degree + 1))
    L[:, 0] = 1.0
    if degree >= 1:
        L[:, 1] = x
    for k in range(2, degree + 1):
        L[:, k] = ((2 * k - 1) * x * L[:, k - 1] - (k - 1) * L[:, k - 2]) / k
    return L


def legendre_derivative_matrix(degree: int) -> np.ndarray:
    """Coefficient-space derivative: ``p = sum c_k P_k`` -> ``p' = sum (D c)_k P_k``.

    Uses ``P_k' = sum_{j<k, k-j odd} (2j+1) P_j``.  Differentiating in Legendre
    coefficient space is *exact*: the derivative of a constant polynomial is the
    exact zero array (bitwise ``0.0``, no round-off), which is what makes the
    free-stream / metric-identity check bit-reproducible on affine meshes.
    """
    D = np.zeros((degree + 1, degree + 1))
    for k in range(1, degree + 1):
        for j in range(k):
            if (k - j) % 2 == 1:
                D[j, k] = 2 * j + 1
    return D


def gll_nodes_weights(degree: int):
    """Legendre-Gauss-Lobatto nodes/weights (exactness ``2*degree - 1``)."""
    if degree == 0:
        return np.array([0.0]), np.array([2.0])
    if degree == 1:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    # Interior GLL nodes are the roots of P'_p; Newton from the Chebyshev-Lobatto
    # guess.  A Newton iteration is used instead of an eigenvalue solve
    # (legroots) so that the nodes are bit-reproducible across BLAS/LAPACK
    # implementations (legroots failed with "Eigenvalues did not converge" on
    # the loong64 cluster).
    p = degree
    coef = np.zeros(p + 1)
    coef[-1] = 1.0
    d1 = np.polynomial.legendre.legder(coef)
    d2 = np.polynomial.legendre.legder(d1)
    x = -np.cos(np.pi * np.arange(1, p) / p)
    for _ in range(200):
        dx = np.polynomial.legendre.legval(x, d1) / \
            np.polynomial.legendre.legval(x, d2)
        x = x - dx
        if np.max(np.abs(dx)) < 4e-16:
            break
    else:  # pragma: no cover - defensive
        raise RuntimeError("GLL node Newton iteration did not converge")
    x = np.concatenate(([-1.0], np.sort(x), [1.0]))
    Pn = legendre_eval_matrix(p, x)[:, p]
    w = 2.0 / (p * (p + 1) * Pn ** 2)
    return x, w


def gll_differentiation_matrix(degree: int) -> np.ndarray:
    """``D[q, k] = l_k'(x_q)`` for the GLL nodal basis (SBP, ``D @ 1 = 0``)."""
    x, _ = gll_nodes_weights(degree)
    L = legendre_eval_matrix(degree, x)
    dL = L @ legendre_derivative_matrix(degree)
    return dL @ np.linalg.inv(L)


def gauss_nodes_weights(nq: int):
    x, w = np.polynomial.legendre.leggauss(nq)
    return np.asarray(x, float), np.asarray(w, float)


def gauss_differentiation_matrix(nq: int) -> np.ndarray:
    """Differentiation matrix exact for polynomials of degree ``<= nq-1``.

    Needed because the metric flux ``a(eta) u(eta)`` has degree up to ``2p`` and
    has to be differentiated *exactly*: no aliasing, hence no order loss.
    """
    x, _ = gauss_nodes_weights(nq)
    L = legendre_eval_matrix(nq - 1, x)
    dL = L @ legendre_derivative_matrix(nq - 1)
    return dL @ np.linalg.inv(L)


# ---------------------------------------------------------------------------
# Isoparametric map of one quadrilateral element
# ---------------------------------------------------------------------------
class IsoparametricMap:
    """Tensor-product isoparametric map ``[-1,1]^2 -> physical quadrilateral``.

    ``x(xi, eta) = sum_{k,l} cx[k,l] P_k(xi) P_l(eta)`` and likewise for ``y``.
    The geometry degree ``g`` is ``cx.shape[0] - 1``; ``g == 1`` reproduces a
    straight (bilinear) quad -- the straight-mesh negative control.
    """

    def __init__(self, cx, cy):
        self.cx = np.asarray(cx, dtype=float)
        self.cy = np.asarray(cy, dtype=float)
        if self.cx.shape != self.cy.shape or self.cx.ndim != 2:
            raise ValueError("cx/cy must be equally shaped 2-D arrays")
        if self.cx.shape[0] != self.cx.shape[1]:
            raise ValueError("geometry must be tensor-product (square in degree)")

    @property
    def degree(self) -> int:
        return self.cx.shape[0] - 1

    @classmethod
    def affine(cls, x0: float, x1: float, y0: float, y1: float,
               degree: int = 1) -> "IsoparametricMap":
        """Exactly affine rectangle ``[x0,x1] x [y0,y1]`` in modal form.

        The coefficients are set analytically (no nodal interpolation), so
        ``d/dxi`` and ``d/deta`` are *bitwise* exact -- this is what turns the
        straight-mesh free-stream control into an exact-zero test.
        """
        if int(degree) < 1:
            raise ValueError("an affine map needs geometry degree >= 1")
        K = int(degree) + 1
        cx = np.zeros((K, K))
        cy = np.zeros((K, K))
        cx[0, 0] = 0.5 * (x0 + x1)
        cx[1, 0] = 0.5 * (x1 - x0)
        cy[0, 0] = 0.5 * (y0 + y1)
        cy[0, 1] = 0.5 * (y1 - y0)
        return cls(cx, cy)

    @classmethod
    def from_nodal(cls, X, Y) -> "IsoparametricMap":
        """Build from physical coordinates sampled at the GLL nodes of degree g."""
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        g = X.shape[0] - 1
        nodes, _ = gll_nodes_weights(g)
        L = legendre_eval_matrix(g, nodes)
        Linv = np.linalg.inv(L)
        return cls(Linv @ X @ Linv.T, Linv @ Y @ Linv.T)

    # -- evaluation ---------------------------------------------------------
    def eval(self, xi, eta):
        g = self.degree
        Lx = legendre_eval_matrix(g, xi)
        Ly = legendre_eval_matrix(g, eta)
        x = np.einsum("qk,kl,rl->qr", Lx, self.cx, Ly)
        y = np.einsum("qk,kl,rl->qr", Lx, self.cy, Ly)
        return x, y

    def derivatives(self, xi, eta):
        """Exact polynomial derivatives ``(x_xi, x_eta, y_xi, y_eta)``."""
        g = self.degree
        Dc = legendre_derivative_matrix(g)
        Lx = legendre_eval_matrix(g, xi)
        Ly = legendre_eval_matrix(g, eta)
        Ldx = Lx @ Dc
        Ldy = Ly @ Dc
        x_xi = np.einsum("qk,kl,rl->qr", Ldx, self.cx, Ly)
        x_eta = np.einsum("qk,kl,rl->qr", Lx, self.cx, Ldy)
        y_xi = np.einsum("qk,kl,rl->qr", Ldx, self.cy, Ly)
        y_eta = np.einsum("qk,kl,rl->qr", Lx, self.cy, Ldy)
        return x_xi, x_eta, y_xi, y_eta

    def jacobian(self, xi, eta) -> np.ndarray:
        x_xi, x_eta, y_xi, y_eta = self.derivatives(xi, eta)
        return x_xi * y_eta - x_eta * y_xi

    def inverse_metrics(self, xi, eta):
        """``(J, xi_x, xi_y, eta_x, eta_y)`` with ``xi_x = J^{-1} y_eta`` etc."""
        x_xi, x_eta, y_xi, y_eta = self.derivatives(xi, eta)
        J = x_xi * y_eta - x_eta * y_xi
        return J, y_eta / J, -x_eta / J, -y_xi / J, x_xi / J

    # -- exact metric-flux coefficients (coefficient space) ------------------
    def metric_flux_coefficients(self, Ax: float, Ay: float):
        """Legendre coefficients of the contravariant metric fluxes.

        ``a1 = J dxi/dx_j A_j = y_eta A_x - x_eta A_y`` and
        ``a2 = J deta/dx_j A_j = -y_xi A_x + x_xi A_y`` (both polynomials, so
        their Legendre coefficients are exact linear combinations).
        """
        g = self.degree
        Dc = legendre_derivative_matrix(g)
        # for f = sum c_k P_k, the derivative coefficients are d = D @ c
        x_xi = Dc @ self.cx          # d/dxi  (first index)
        x_eta = self.cx @ Dc.T       # d/deta (second index)
        y_xi = Dc @ self.cy
        y_eta = self.cy @ Dc.T
        a1 = y_eta * Ax - x_eta * Ay
        a2 = -y_xi * Ax + x_xi * Ay
        return a1, a2

    def metric_divergence_coefficients(self, Ax: float, Ay: float) -> np.ndarray:
        """Legendre coefficients of ``d_xi a1 + d_eta a2`` (exactly 0 for
        divergence-free ``A`` on a straight mesh -- the discrete metric identity)."""
        a1, a2 = self.metric_flux_coefficients(Ax, Ay)
        Dc = legendre_derivative_matrix(self.degree)
        return Dc @ a1 + a2 @ Dc.T

    def divergence_coefficients(self, Ax: float, Ay: float) -> np.ndarray:
        """Legendre coefficients of ``d a1/dxi + d a2/deta`` (alias of above)."""
        return self.metric_divergence_coefficients(Ax, Ay)

    def nodal_metric_fluxes(self, Ax: float, Ay: float, degree: int | None = None):
        """Metric fluxes sampled at the GLL nodes of ``degree`` (default: own)."""
        p = self.degree if degree is None else degree
        nodes, _ = gll_nodes_weights(p)
        x_xi, x_eta, y_xi, y_eta = self.derivatives(nodes, nodes)
        a1 = y_eta * Ax - x_eta * Ay
        a2 = -y_xi * Ax + x_xi * Ay
        return nodes, a1, a2

    def area(self) -> float:
        nodes, w = gll_nodes_weights(self.degree)
        # exact: integrate J with a Gauss rule exact for degree 2g-1
        nq = max(2 * self.degree + 2, 3)
        xi, wq = gauss_nodes_weights(nq)
        return float(np.einsum("q,r,qr->", wq, wq, self.jacobian(xi, xi)))

    def __repr__(self):  # pragma: no cover - debug helper
        return f"IsoparametricMap(degree={self.degree})"


# ---------------------------------------------------------------------------
# Metric identity / free-stream diagnostics for one element
# ---------------------------------------------------------------------------
def metric_identity_report(mapping: IsoparametricMap, Ax: float = 0.6,
                           Ay: float = 0.8, degree: int | None = None) -> dict:
    """Quantify the three metric identities on one isoparametric element.

    ``inverse_metric``    : ``max |dxi_i/dx_j dx_j/dxi_k - delta_ik|`` at nodes
    ``piola_nodal``       : ``max |D_xi a1 + D_eta a2|`` with the GLL nodal
                            derivative operator (same norm as a nodal DG scheme)
    ``piola_coeff``       : ``max |coefficients of d_xi a1 + d_eta a2|``
                            (exact polynomial arithmetic; bitwise 0 on affine)
    ``jacobian_positive`` : minimum Jacobian (geometry validity)
    """
    p = mapping.degree if degree is None else degree
    nodes, w = gll_nodes_weights(p)
    J, xi_x, xi_y, eta_x, eta_y = mapping.inverse_metrics(nodes, nodes)
    x_xi, x_eta, y_xi, y_eta = mapping.derivatives(nodes, nodes)

    # dxi_i/dx_j * dx_j/dxi_k = delta_ik   (chain rule),  i,k in {xi, eta}
    e11 = xi_x * x_xi + xi_y * y_xi - 1.0
    e12 = xi_x * x_eta + xi_y * y_eta
    e21 = eta_x * x_xi + eta_y * y_xi
    e22 = eta_x * x_eta + eta_y * y_eta - 1.0
    inverse_metric = float(np.max(np.abs(np.stack([e11, e12, e21, e22]))))

    # discrete Piola identity with the GLL nodal derivative operator
    D = gll_differentiation_matrix(p)
    a1n = y_eta * Ax - x_eta * Ay
    a2n = -y_xi * Ax + x_xi * Ay
    piola_nodal = float(np.max(np.abs(D @ a1n + a2n @ D.T)))

    # exact coefficient-space Piola identity
    div_coeff = mapping.metric_divergence_coefficients(Ax, Ay)
    piola_coeff = float(np.max(np.abs(div_coeff)))

    return {
        "degree": p,
        "inverse_metric": inverse_metric,
        "piola_nodal": piola_nodal,
        "piola_coeff": piola_coeff,
        "jacobian_min": float(np.min(J)),
        "nodes": nodes,
        "weights": w,
    }

# ---------------------------------------------------------------------------
# Curvilinear structured mesh
# ---------------------------------------------------------------------------
class CurvedMesh:
    """Structured curvilinear mesh of isoparametric quadrilateral elements.

    ``maps[i][j]`` is the ``IsoparametricMap`` of cell ``(i, j)``; axis 0 is
    ``xi`` and axis 1 is ``eta``.  ``periodic`` gives the connectivity of the
    two axes, ``walls`` flags the four mesh boundaries
    ``(axis0_low, axis0_high, axis1_low, axis1_high)`` as no-penetration walls.
    """

    def __init__(self, maps, periodic=(False, False),
                 walls=(False, False, False, False)):
        self.maps = maps
        self.nx = len(maps)
        self.ny = len(maps[0])
        self.degree = maps[0][0].degree
        for row in maps:
            for m in row:
                if m.degree != self.degree:
                    raise ValueError("all elements must share the geometry degree")
        self.periodic = tuple(bool(v) for v in periodic)
        self.walls = tuple(bool(v) for v in walls)
        self.ncells = self.nx * self.ny
        self.neighbours = self._build_neighbours()

    # -- connectivity -------------------------------------------------------
    def index(self, i: int, j: int) -> int:
        return i * self.ny + j

    def _build_neighbours(self):
        """``nb[axis][side]``: flat neighbour index per cell, ``-1`` at a wall."""
        n = self.ncells
        nb = {}
        for axis in (0, 1):
            for side in (-1, 1):
                arr = np.full(n, -1, dtype=int)
                for i in range(self.nx):
                    for j in range(self.ny):
                        ii, jj = i, j
                        if axis == 0:
                            ii += side
                        else:
                            jj += side
                        lo_bad = (ii < 0) or (jj < 0)
                        hi_bad = (ii >= self.nx) or (jj >= self.ny)
                        if lo_bad or hi_bad:
                            if self.periodic[axis]:
                                if axis == 0:
                                    ii %= self.nx
                                else:
                                    jj %= self.ny
                            else:
                                continue
                        arr[self.index(i, j)] = self.index(ii, jj)
                nb[(axis, side)] = arr
        return nb

    def flat_maps(self):
        return [self.maps[i][j] for i in range(self.nx) for j in range(self.ny)]

    def coefficients(self):
        ms = self.flat_maps()
        CX = np.stack([m.cx for m in ms])
        CY = np.stack([m.cy for m in ms])
        return CX, CY

    def areas(self) -> np.ndarray:
        return np.array([m.area() for m in self.flat_maps()])


# ---------------------------------------------------------------------------
# DG discretization of scalar linear advection on a curvilinear mesh
# ---------------------------------------------------------------------------
class CurvedAdvection:
    """Strong-form DG for ``u_t + div(A u) = 0`` on ``CurvedMesh``.

    * polynomial degree ``p`` (Legendre modal basis ``P_k(xi) P_l(eta)``),
    * volume flux divergence integrated with an *exact* Gauss rule
      (``nq = 2p+2`` points: exact for the degree-``2p`` metric-flux product and
      for the degree-``3p-1`` integrand, so there is no aliasing and no order
      loss),
    * consistent upwind interface flux,
    * no-penetration wall flux on flagged boundaries,
    * exact diagonal-free mass matrix ``M = int phi phi J`` inverted per element.

    ``velocity`` is a callable ``velocity(x, y) -> (Ax, Ay)``; it has to be
    divergence free for free-stream preservation to be meaningful.
    """

    def __init__(self, mesh: CurvedMesh, degree: int, velocity, cfl: float = 0.2):
        self.mesh = mesh
        self.p = int(degree)
        self.K = self.p + 1
        self.velocity = velocity
        self.cfl = cfl
        # Minimal exact quadrature: the product a_i u has degree 2p (exact
        # differentiation needs nq-1 >= 2p) and the weak/strong volume integrand
        # has degree 3p-1 (Gauss exactness needs 2 nq - 1 >= 3p-1), so
        # nq = 2p+2 is both exact and better conditioned than a padded rule.
        self.nq = max(2 * self.p + 2, 4)
        self.xi, self.w = gauss_nodes_weights(self.nq)
        self.L = legendre_eval_matrix(self.p, self.xi)
        self.Ld = self.L @ legendre_derivative_matrix(self.p)
        self.Dg = gauss_differentiation_matrix(self.nq)
        # polynomial-exact differentiation of products of degree <= 2p:
        # interpolate at the nq Gauss nodes, differentiate in Legendre
        # coefficient space (derivative of a constant is bitwise zero), map back.
        self._Lg = legendre_eval_matrix(self.nq - 1, self.xi)
        self._Linv = np.linalg.inv(self._Lg)
        self._Dc_g = legendre_derivative_matrix(self.nq - 1)
        self._build_geometry()
        self._build_face_data()
        self._build_mass_matrix()

    # -- geometry at the quadrature nodes -----------------------------------
    def _build_geometry(self):
        CX, CY = self.mesh.coefficients()
        self.CX, self.CY = CX, CY
        g = self.mesh.degree
        Dc = legendre_derivative_matrix(g)
        Lx = legendre_eval_matrix(g, self.xi)
        Ldx = Lx @ Dc
        self._g = g
        Lx2, Ldx2 = Lx, Ldx
        def ev(C, LX, LDX):
            val = np.einsum("qk,nkl,rl->nqr", LX, C, Lx2)
            dxi = np.einsum("qk,nkl,rl->nqr", LDX, C, Lx2)
            deta = np.einsum("qk,nkl,rl->nqr", LX, C, Ldx2)
            return val, dxi, deta
        x, x_xi, x_eta = ev(CX, Lx2, Ldx2)
        y, y_xi, y_eta = ev(CY, Lx2, Ldx2)
        self.xg, self.yg = x, y
        self.Jg = x_xi * y_eta - x_eta * y_xi
        if np.any(self.Jg <= 0.0):
            raise ValueError("degenerate element: non-positive Jacobian")
        Ax, Ay = self.velocity(x, y)
        self.a1g = y_eta * Ax - x_eta * Ay
        self.a2g = -y_xi * Ax + x_xi * Ay
        self._Axg, self._Ayg = np.asarray(Ax), np.asarray(Ay)

    def _edge_fields(self, axis: int, side: int):
        """Metric normal flux ``c = A . n * J`` on one element edge."""
        g = self._g
        Dc = legendre_derivative_matrix(g)
        fixed = float(side)
        Lf = legendre_eval_matrix(g, [fixed])
        Ldf = Lf @ Dc
        Lg_ = legendre_eval_matrix(g, self.xi)
        Ldg_ = Lg_ @ Dc
        if axis == 0:
            LX, LDX, LY, LDY = Lf, Ldf, Lg_, Ldg_
        else:
            LX, LDX, LY, LDY = Lg_, Ldg_, Lf, Ldf
        drop = 1 if axis == 0 else 2   # collapse the constant (fixed) edge axis

        def ev(C):
            val = np.einsum("qk,nkl,rl->nqr", LX, C, LY)
            dxi = np.einsum("qk,nkl,rl->nqr", LDX, C, LY)
            deta = np.einsum("qk,nkl,rl->nqr", LX, C, LDY)
            return (np.squeeze(val, axis=drop), np.squeeze(dxi, axis=drop),
                    np.squeeze(deta, axis=drop))
        x, x_xi, x_eta = ev(self.CX)
        y, y_xi, y_eta = ev(self.CY)
        Ax, Ay = self.velocity(x, y)
        if axis == 0:
            c = y_eta * Ax - x_eta * Ay
        else:
            c = -y_xi * Ax + x_xi * Ay
        return float(side) * np.asarray(c)

    def _build_face_data(self):
        self.c_face = {}
        for axis in (0, 1):
            for side in (-1, 1):
                self.c_face[(axis, side)] = self._edge_fields(axis, side)

    # -- mass matrix --------------------------------------------------------
    def _build_mass_matrix(self):
        K, nq = self.K, self.nq
        T = np.einsum("q,qk,qi->qki", self.w, self.L, self.L)   # (nq, K, K)
        M = np.einsum("nqr,qka,rlb->nklab", self.Jg, T, T)
        M = M.reshape(self.mesh.ncells, K * K, K * K)
        # symmetrise to kill round-off asymmetry before inverting
        M = 0.5 * (M + np.transpose(M, (0, 2, 1)))
        self.M = M
        self.Minv = np.linalg.inv(M)
        ref = np.array([2.0 / (2 * k + 1) for k in range(K)])
        self.mass_ref_diag = np.outer(ref, ref)                 # (K, K)

    def _diff_axis0(self, F):
        """Exact ``d/dxi`` of ``F[n, q, m]`` sampled at the Gauss nodes."""
        C = np.einsum("sr,nrm->nsm", self._Linv, F)
        Cd = np.einsum("ts,nsm->ntm", self._Dc_g, C)
        return np.einsum("rt,ntm->nrm", self._Lg, Cd)

    def _diff(self, F, axis: int):
        if axis == 0:
            return self._diff_axis0(F)
        return self._diff_axis0(F.transpose(0, 2, 1)).transpose(0, 2, 1)

    def apply_mass_inverse(self, v):
        n = self.mesh.ncells
        return np.einsum("nab,nb->na", self.Minv, v.reshape(n, -1)).reshape(v.shape)

    # -- traces -------------------------------------------------------------
    def traces(self, U):
        """Interior traces ``(lo0, hi0, lo1, hi1)`` at the Gauss edge nodes."""
        K = self.K
        sgn = (-1.0) ** np.arange(K)
        lo0 = np.einsum("k,nkl,rl->nr", sgn, U, self.L)   # xi = -1, along eta
        hi0 = np.einsum("nkl,rl->nr", U, self.L)          # xi = +1
        lo1 = np.einsum("l,nkl,qk->nq", sgn, U, self.L)   # eta = -1, along xi
        hi1 = np.einsum("nkl,qk->nq", U, self.L)          # eta = +1
        return lo0, hi0, lo1, hi1

    def rhs(self, U):
        U = np.asarray(U, float)
        K = self.K
        # --- volume term: -int phi div(a u)  (strong form) ------------------
        Ug = np.einsum("qk,nkl,rl->nqr", self.L, U, self.L)
        f1 = self.a1g * Ug
        f2 = self.a2g * Ug
        df1 = self._diff(f1, 0)
        df2 = self._diff(f2, 1)
        div = df1 + df2
        wdiv = div * self.w[None, :, None] * self.w[None, None, :]
        vol = np.einsum("qk,rl,nqr->nkl", self.L, self.L, wdiv)
        # --- interface correction: +oint phi [flux* - flux] -----------------
        lo0, hi0, lo1, hi1 = self.traces(U)
        inner = {(0, -1): lo0, (0, 1): hi0, (1, -1): lo1, (1, 1): hi1}
        outer_src = {(0, -1): hi0, (0, 1): lo0, (1, -1): hi1, (1, 1): lo1}
        face = np.zeros_like(U)
        sgn = (-1.0) ** np.arange(K)
        wall_flags = {(0, -1): self.mesh.walls[0], (0, 1): self.mesh.walls[1],
                      (1, -1): self.mesh.walls[2], (1, 1): self.mesh.walls[3]}
        for axis in (0, 1):
            for side in (-1, 1):
                key = (axis, side)
                c = self.c_face[key]
                u_in = inner[key]
                nb = self.mesh.neighbours[key]
                u_ext = np.zeros_like(u_in)
                valid = nb >= 0
                if np.any(valid):
                    u_ext[valid] = outer_src[key][nb[valid]]
                num = np.where(c > 0.0, c * u_in, c * u_ext)
                if wall_flags[key]:
                    num = np.zeros_like(num)
                corr = num - c * u_in
                if axis == 0:
                    pk = np.ones(K) if side > 0 else sgn
                    face += np.einsum("k,rl,nr,r->nkl", pk, self.L, corr, self.w)
                else:
                    pl = np.ones(K) if side > 0 else sgn
                    face += np.einsum("l,qk,nq,q->nkl", pl, self.L, corr, self.w)
        # strong form: M u_t = -(vol) - oint phi [flux* - flux]
        return self.apply_mass_inverse(-vol - face)

    # -- projection / error / time stepping ---------------------------------
    def project(self, func):
        fg = np.asarray(func(self.xg, self.yg), float)
        fg = np.broadcast_to(fg, self.xg.shape)
        wf = fg * self.Jg * self.w[None, :, None] * self.w[None, None, :]
        R = np.einsum("qk,rl,nqr->nkl", self.L, self.L, wf)
        return self.apply_mass_inverse(R)

    def l2_error(self, U, func):
        Ug = np.einsum("qk,nkl,rl->nqr", self.L, U, self.L)
        ex = np.asarray(func(self.xg, self.yg), float)
        ex = np.broadcast_to(ex, self.xg.shape)
        wJ = self.Jg * self.w[None, :, None] * self.w[None, None, :]
        num = float(np.sqrt(np.sum(wJ * (Ug - ex) ** 2)))
        den = float(np.sqrt(np.sum(wJ * ex ** 2)))
        return num, den, (num / den if den > 0 else num)

    def max_speed(self) -> float:
        return float(np.max(np.hypot(self._Axg, self._Ayg)))

    def min_length(self) -> float:
        """Effective cell width ``min J / (|a1| + |a2|)`` over the quadrature set.

        ``a_i = J dxi_i/dx_j A_j`` so ``J / |a|`` is a local length; taking the
        minimum over the quadrature points of every element makes the estimate
        sensitive to distorted / high-aspect-ratio curved cells, which a
        ``sqrt(area)`` estimate misses.
        """
        denom = np.abs(self.a1g) + np.abs(self.a2g)
        ratio = np.where(denom > 0.0, self.Jg / np.maximum(denom, 1e-300), np.inf)
        return float(np.min(ratio))

    def stable_dt(self, cfl: float | None = None) -> float:
        """``dt = cfl * min(length) / (2p+1)`` (RK3 + upwind DG stability)."""
        cfl = self.cfl if cfl is None else cfl
        h = self.min_length()
        if not np.isfinite(h):
            return np.inf
        return cfl * h / (2 * self.p + 1)

    def ssp_rk3_step(self, U, dt):
        k1 = self.rhs(U)
        u1 = U + dt * k1
        k2 = self.rhs(u1)
        u2 = 0.75 * U + 0.25 * (u1 + dt * k2)
        k3 = self.rhs(u2)
        return U / 3.0 + (2.0 / 3.0) * (u2 + dt * k3)

    def evolve(self, U, T, dt, max_steps=2_000_000, guard=None):
        nsteps = int(np.ceil(T / dt))
        if nsteps > max_steps:
            raise RuntimeError(f"time step too small: {nsteps} steps")
        dt = T / nsteps
        scale = 1.0 + float(np.max(np.abs(U)))
        limit = 1e4 * scale if guard is None else guard
        for _ in range(nsteps):
            U = self.ssp_rk3_step(U, dt)
            if not np.all(np.isfinite(U)) or float(np.max(np.abs(U))) > limit:
                raise RuntimeError("advection solve went unstable "
                                   f"(max|u| > {limit:g}): reduce dt/cfl")
        return U, dt, nsteps

    # -- free-stream preservation ------------------------------------------
    def free_stream_defect(self, u0: float = 1.0, steps: int = 200) -> dict:
        """Free-stream preservation on the *current* mesh.

        ``u_t_operator`` : ``sup |d u0 / dt|`` of the realised DG operator
                           applied to the exact representation of a constant
                           (face fluxes cancel by consistency; only round-off
                           of the volume quadrature survives).
        ``u_t_exact``    : the same residual with the metric divergence taken in
                           Legendre coefficient space -- this is the metric
                           identity itself and is *bitwise* zero on an affine
                           mesh.
        ``drift``        : ``sup |u - u0|`` after ``steps`` SSP-RK3 steps.
        """
        K = self.K
        U0 = np.zeros((self.mesh.ncells, K, K))
        U0[:, 0, 0] = u0
        ut_op = self.rhs(U0)
        # exact metric-divergence path (requires constant velocity)
        div_coeffs = np.stack([
            m.metric_divergence_coefficients(*self._constant_velocity())
            for m in self.mesh.flat_maps()
        ])
        vol = u0 * div_coeffs * self.mass_ref_diag[None, :, :]
        ut_exact = self.apply_mass_inverse(-vol)
        U = U0.copy()
        dt = self.stable_dt()
        for _ in range(steps):
            U = self.ssp_rk3_step(U, dt)
            if not np.all(np.isfinite(U)) or float(np.max(np.abs(U))) > 1e3:
                raise RuntimeError("free-stream state went unstable")
        drift = float(np.max(np.abs(U - U0)))
        return {
            "u_t_operator": float(np.max(np.abs(ut_op))),
            "u_t_exact": float(np.max(np.abs(ut_exact))),
            "metric_div_coeff": float(np.max(np.abs(div_coeffs))),
            "drift": drift,
            "dt": dt,
        }

    def _constant_velocity(self):
        Ax, Ay = self.velocity(np.array([0.0]), np.array([0.0]))
        return float(np.ravel(Ax)[0]), float(np.ravel(Ay)[0])


# ---------------------------------------------------------------------------
# Mesh builders
# ---------------------------------------------------------------------------
def warped_periodic_mesh(nx: int, ny: int, degree: int, amplitude: float = 0.06):
    """Smooth doubly-periodic warp of the unit square (curvilinear control mesh).

    ``x = s + A sin(2 pi s) cos(2 pi t)``, ``y = t + A sin(2 pi t) cos(2 pi s)``
    with ``s = i/nx``, ``t = j/ny``.  Every element is a curved quad whenever
    ``degree >= 2``; ``degree == 1`` is only valid for ``amplitude == 0``.
    """
    degree = max(int(degree), 1)
    nodes, _ = gll_nodes_weights(degree)
    maps = []
    for i in range(nx):
        row = []
        for j in range(ny):
            s = (i + 0.5 * (nodes + 1.0)) / nx
            t = (j + 0.5 * (nodes + 1.0)) / ny
            S, T = np.meshgrid(s, t, indexing="ij")
            X = S + amplitude * np.sin(2 * np.pi * S) * np.cos(2 * np.pi * T)
            Y = T + amplitude * np.sin(2 * np.pi * T) * np.cos(2 * np.pi * S)
            row.append(IsoparametricMap.from_nodal(X, Y))
        maps.append(row)
    return CurvedMesh(maps, periodic=(True, True))


def cartesian_periodic_mesh(nx: int, ny: int, degree: int = 1):
    """Straight (exactly affine) periodic mesh -- the free-stream negative control.

    Elements are built analytically (not by nodal interpolation) so that the
    metric fluxes are bitwise constant and the discrete metric identity is
    bitwise satisfied.
    """
    gdeg = max(int(degree), 1)
    maps = []
    for i in range(nx):
        row = []
        for j in range(ny):
            row.append(IsoparametricMap.affine(i / nx, (i + 1) / nx,
                                               j / ny, (j + 1) / ny, gdeg))
        maps.append(row)
    return CurvedMesh(maps, periodic=(True, True))


def annulus_mesh(nr: int, ntheta: int, r_inner: float, r_outer: float,
                 degree: int, inner_curve=None, walls=(True, True, False, False),
                 periodic=(False, True)):
    """Polar annulus around a circular cylinder, curved walls at both radii.

    ``inner_curve(theta)`` optionally returns the inner wall points (e.g. from a
    CAD/NURBS contract); the default is the exact circle.  The radial map is the
    linear blend of the two wall curves, which is exactly the polar map when both
    walls are circles.  Axis 0 = radial (walls), axis 1 = theta (periodic).
    """
    nodes, _ = gll_nodes_weights(degree)
    outer = lambda th: np.stack([r_outer * np.cos(th), r_outer * np.sin(th)], -1)
    if inner_curve is None:
        inner = lambda th: np.stack([r_inner * np.cos(th), r_inner * np.sin(th)], -1)
    else:
        inner = inner_curve
    maps = []
    for i in range(nr):
        r = r_inner + (r_outer - r_inner) * (i + 0.5 * (nodes + 1.0)) / nr
        row = []
        for j in range(ntheta):
            th = 2.0 * np.pi * (j + 0.5 * (nodes + 1.0)) / ntheta
            R, TH = np.meshgrid(r, th, indexing="ij")
            lam = (R - r_inner) / (r_outer - r_inner)
            Pi = inner(TH)
            Po = outer(TH)
            X = (1.0 - lam) * Pi[..., 0] + lam * Po[..., 0]
            Y = (1.0 - lam) * Pi[..., 1] + lam * Po[..., 1]
            row.append(IsoparametricMap.from_nodal(X, Y))
        maps.append(row)
    return CurvedMesh(maps, periodic=periodic, walls=walls)


# ---------------------------------------------------------------------------
# CAD interface contract: NURBS boundary parameterization -> DG geometry
# ---------------------------------------------------------------------------
def bspline_basis_matrix(degree: int, knots, t) -> np.ndarray:
    """Cox--de Boor basis ``N[., i] = N_{i,degree}(t)`` on a clamped knot vector."""
    knots = np.asarray(knots, float)
    t = np.asarray(t, float).reshape(-1)
    p = int(degree)
    n = knots.size - p - 1
    if n <= 0:
        raise ValueError("knot vector too short for the requested degree")
    N = np.zeros((t.size, n))
    for i in range(n):
        N[:, i] = ((t >= knots[i]) & (t < knots[i + 1])).astype(float)
    # half-open spans: put the right end of the domain into the last basis
    N[t >= knots[n], :] = 0.0
    N[t >= knots[n], n - 1] = 1.0
    for k in range(1, p + 1):
        Nn = np.zeros_like(N)
        for i in range(n):
            d1 = knots[i + k] - knots[i]
            if d1 > 0.0:
                Nn[:, i] += (t - knots[i]) / d1 * N[:, i]
            d2 = knots[i + k + 1] - knots[i + 1]
            if d2 > 0.0 and i + 1 < n:
                Nn[:, i] += (knots[i + k + 1] - t) / d2 * N[:, i + 1]
        N = Nn
    return N


def nurbs_curve_points(degree: int, knots, control_points, weights, t) -> np.ndarray:
    """Rational B-spline curve ``C(t) = sum N_i w_i P_i / sum N_i w_i``."""
    N = bspline_basis_matrix(degree, knots, t)
    w = np.asarray(weights, float).reshape(-1)
    cp = np.asarray(control_points, float)
    if cp.ndim != 2 or cp.shape[1] != 2:
        raise ValueError("control_points must be an (n, 2) array")
    if w.size != cp.shape[0] or N.shape[1] != cp.shape[0]:
        raise ValueError("control point / weight / knot counts are inconsistent")
    denom = N @ w
    return (N * w) @ cp / denom[:, None]


def circle_contract(radius: float = 1.0, centre=(0.0, 0.0)) -> dict:
    """Exact rational-quadratic NURBS representation of a circle (CAD side).

    Nine control points, weights ``[1, 1/sqrt2, 1, ...]``, knot vector with
    doubled interior knots at ``1/4, 1/2, 3/4``.  The curve is *exactly* the
    circle: ``|C(t) - centre| = radius`` to round-off for every ``t``.
    """
    s = 1.0 / math_sqrt2()
    unit = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)]
    weights = [1.0, s, 1.0, s, 1.0, s, 1.0, s, 1.0]
    cx, cy = float(centre[0]), float(centre[1])
    cps = [[cx + radius * ux, cy + radius * uy] for ux, uy in unit]
    knots = [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0]
    return {
        "schema": CONTRACT_SCHEMA,
        "kind": "nurbs_curve",
        "name": "circle",
        "centre": [cx, cy],
        "degree": 2,
        "knots": knots,
        "control_points": cps,
        "weights": weights,
        "closed": True,
    }


def math_sqrt2() -> float:
    return np.sqrt(2.0)


def validate_geometry_contract(contract: dict) -> dict:
    """Validate a ``tensordg.curved_geometry.v1`` contract; return it normalised.

    Raises ``ValueError`` with an explicit reason on any inconsistency, so a
    malformed CAD payload can never silently produce a wrong geometry.
    """
    if not isinstance(contract, dict):
        raise ValueError("contract must be a mapping")
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"unsupported schema {contract.get('schema')!r}; "
                         f"expected {CONTRACT_SCHEMA!r}")
    if contract.get("kind") != "nurbs_curve":
        raise ValueError(f"unsupported geometry kind {contract.get('kind')!r}")
    try:
        degree = int(contract["degree"])
    except KeyError as exc:
        raise ValueError(f"missing field {exc.args[0]!r}") from None
    except (TypeError, ValueError) as exc:
        raise ValueError(f"degree is not an integer: {exc}") from None
    if degree < 1:
        raise ValueError("degree must be >= 1")
    for key in ("knots", "control_points", "weights"):
        if key not in contract:
            raise ValueError(f"missing field {key!r}")
    knots = np.asarray(contract["knots"], float).reshape(-1)
    cps = np.asarray(contract["control_points"], float)
    weights = np.asarray(contract["weights"], float).reshape(-1)
    if cps.ndim != 2 or cps.shape[1] != 2:
        raise ValueError("control_points must be an (n, 2) array")
    if not np.all(np.isfinite(knots)) or not np.all(np.isfinite(cps)):
        raise ValueError("non-finite knot or control point value")
    if knots.size != cps.shape[0] + degree + 1:
        raise ValueError(
            f"knot count {knots.size} != n_control_points + degree + 1 "
            f"= {cps.shape[0] + degree + 1}")
    if np.any(np.diff(knots) < 0.0):
        raise ValueError("knot vector must be non-decreasing")
    if knots[0] >= knots[-1]:
        raise ValueError("knot vector must span a non-empty parameter interval")
    if not np.allclose(knots[: degree + 1], knots[0]) or \
            not np.allclose(knots[-(degree + 1):], knots[-1]):
        raise ValueError("only clamped (open) knot vectors are supported")
    if weights.size != cps.shape[0]:
        raise ValueError("weights must have one entry per control point")
    if np.any(weights <= 0.0):
        raise ValueError("all weights must be strictly positive")
    if "centre" in contract:
        centre = np.asarray(contract["centre"], float).reshape(-1)
        if centre.size != 2 or not np.all(np.isfinite(centre)):
            raise ValueError("centre must be a finite 2-vector")
    if contract.get("closed", False):
        if not np.allclose(cps[0], cps[-1], rtol=0.0, atol=1e-12):
            raise ValueError("closed curve requires identical first/last control point")
    return {**contract, "degree": degree, "knots": knots.tolist(),
            "control_points": cps.tolist(), "weights": weights.tolist()}



def _derivative_of_curve(degree, knots, cps, weights, t) -> np.ndarray:
    """First derivative of the rational B-spline curve (for parameter inversion)."""
    knots = np.asarray(knots, float)
    t = np.asarray(t, float).reshape(-1)
    p = int(degree)
    w = np.asarray(weights, float).reshape(-1)
    cp = np.asarray(cps, float)
    N = bspline_basis_matrix(p, knots, t)
    Nn = _bspline_basis_derivative_matrix(p, knots, t)
    A = (N * w) @ cp
    An = (Nn * w) @ cp
    wv = N @ w
    wnv = Nn @ w
    C = A / wv[:, None]
    return (An - C * wnv[:, None]) / wv[:, None]


def _bspline_basis_derivative_matrix(degree: int, knots, t) -> np.ndarray:
    """Derivatives of the Cox--de Boor basis functions."""
    knots = np.asarray(knots, float)
    t = np.asarray(t, float).reshape(-1)
    p = int(degree)
    n = knots.size - p - 1
    out = np.zeros((t.size, n))
    if p == 0:
        return out
    Nlo = bspline_basis_matrix(p - 1, knots, t)   # n+1 basis functions of degree p-1
    for i in range(n):
        d1 = knots[i + p] - knots[i]
        if d1 > 0.0:
            out[:, i] += p / d1 * Nlo[:, i]
        d2 = knots[i + p + 1] - knots[i + 1]
        if d2 > 0.0:
            out[:, i] -= p / d2 * Nlo[:, i + 1]
    return out


def mesh_from_contract(contract: dict, nr: int, ntheta: int, r_outer: float,
                       degree: int, periodic_radial: bool = False) -> CurvedMesh:
    """Build an annulus DG mesh whose inner wall is the CAD/NURBS curve.

    Contract -> ``inner_curve(theta)`` -> isoparametric element geometry; this is
    the concrete TensorCAD(N03) -> TensorDG hand-off: CAD supplies control
    points, weights and knots, DG samples them at its GLL nodes.
    """
    contract = validate_geometry_contract(contract)
    deg = contract["degree"]
    knots = contract["knots"]
    cps = contract["control_points"]
    wts = contract["weights"]
    t0, t1 = knots[0], knots[-1]
    # parameter range is normalised to [0, 1] by construction
    if not (abs(t0) < 1e-12 and abs(t1 - 1.0) < 1e-12):
        raise ValueError("contract parameter range must be [0, 1]")
    tsample = np.linspace(0.0, 1.0, 721)
    pts = nurbs_curve_points(deg, knots, cps, wts, tsample)
    if not np.allclose(pts[0], pts[-1], rtol=0.0, atol=1e-10):
        raise ValueError("inner wall curve is not closed")
    if "centre" in contract:
        centre = np.asarray(contract["centre"], float).reshape(2)
    else:
        # circumcentre of three well separated curve points
        p0, p1, p2 = pts[0], pts[len(pts) // 3], pts[(2 * len(pts)) // 3]
        a = p1 - p0
        b = p2 - p0
        d = 2.0 * (a[0] * b[1] - a[1] * b[0])
        if abs(d) < 1e-14:
            raise ValueError("cannot infer the wall centre; supply 'centre'")
        centre = p0 + np.array([(b[1] * (a @ a) - a[1] * (b @ b)) / d,
                                (a[0] * (b @ b) - b[0] * (a @ a)) / d])
    rad = np.linalg.norm(pts - centre, axis=1)
    r_wall = float(np.mean(rad))
    if np.max(np.abs(rad - r_wall)) > 1e-8 * r_wall:
        raise ValueError("inner wall is not circular; the annulus builder needs "
                         "a circular wall (got radius spread "
                         f"{np.max(np.abs(rad - r_wall)):.3e})")
    if r_outer <= r_wall:
        raise ValueError("r_outer must exceed the inner wall radius")

    def inner_point(t):
        return nurbs_curve_points(deg, knots, cps, wts, np.asarray(t, float)
                                  .reshape(-1))

    def inner_curve(theta):
        """Wall point at polar angle ``theta`` (parameter inverted by Newton)."""
        th = np.mod(np.asarray(theta, float).reshape(-1), 2.0 * np.pi)
        ang = np.unwrap(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))
        ang = ang - ang[0]
        t = np.interp(th, ang, tsample)          # bracket / initial guess
        for _ in range(6):
            P = inner_point(t)
            dP = (_derivative_of_curve(deg, knots, cps, wts, t))
            dx = P[:, 0] - centre[0]
            dy = P[:, 1] - centre[1]
            r2 = dx * dx + dy * dy
            g = np.mod(np.arctan2(dy, dx) - th + np.pi, 2 * np.pi) - np.pi
            gp = (dx * dP[:, 1] - dy * dP[:, 0]) / r2
            step = np.where(np.abs(gp) > 1e-12, g / gp, 0.0)
            t = np.clip(t - step, 0.0, 1.0)
            if np.max(np.abs(step)) < 1e-15:
                break
        return inner_point(t).reshape(np.asarray(theta).shape + (2,))

    if periodic_radial:
        # geometry-verification variant: radially periodic (no walls), used to
        # check that CAD-derived geometry preserves a free stream exactly
        return annulus_mesh(nr, ntheta, r_wall, r_outer, degree,
                            inner_curve=inner_curve,
                            walls=(False, False, False, False),
                            periodic=(True, True))
    return annulus_mesh(nr, ntheta, r_wall, r_outer, degree, inner_curve=inner_curve)
