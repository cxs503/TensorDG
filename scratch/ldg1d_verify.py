"""Decisive 1-D LDG verification (scratch, m16388).

Findings so far
--------------
* Boundary data sign `+sgn` is the consistent one (polynomial-exactness residual
  is machine zero); the earlier "sign fix" attempt was wrong.
* Neighbour traces must be expanded in the neighbour's *facing* endpoint basis
  (outer(own, far)); the legacy `outer(own, own)` assembly is inconsistent for
  k >= 1 (residual O(1) for an exactly representable polynomial solution).
* `q_hat = q_int` on a Dirichlet face degenerates the last element's mass row
  to 0 = h f (rank deficiency 1): inherent to the "pure" closure, cured by a
  boundary flux correction q_hat.n = q_int.n + c/h (u_int - g).

This script prints, for every combination, the polynomial-exactness residual
and the L2 convergence order of u for -u'' = f with u = sin(pi x).
"""
import json
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def basis(k):
    nq = max(2 * k + 3, 6)
    xi, w = np.polynomial.legendre.leggauss(nq)
    phi = np.zeros((k + 1, nq)); dphi = np.zeros((k + 1, nq))
    for j in range(k + 1):
        c = np.zeros(j + 1); c[-1] = 1.0
        phi[j] = np.polynomial.legendre.legval(xi, c)
        dc = np.polynomial.legendre.legder(c)
        dphi[j] = np.polynomial.legendre.legval(xi, dc) if len(dc) else 0.0
    M = np.einsum("q,iq,jq->ij", w, phi, phi)
    D = np.einsum("q,iq,jq->ij", w, dphi, phi)
    left = np.array([(-1.0) ** j for j in range(k + 1)])
    return xi, w, phi, M, D, left, np.ones(k + 1)


def assemble(n, k, boundary="nitsche", c=1.0, oriented=True, f=None, g=None):
    """f, g: callables; default manufactured pair u = sin(pi x)."""
    f = f or (lambda x: math.pi ** 2 * np.sin(math.pi * x))
    g = g or (lambda x: np.sin(math.pi * x))
    xi, w, phi, M, D, left, right = basis(k)
    m = k + 1; h = 1.0 / n; nd = 2 * n * m
    A = np.zeros((nd, nd)); b = np.zeros(nd)
    iq = lambda e: e * 2 * m; iu = lambda e: e * 2 * m + m
    for e in range(n):
        xc = (e + 0.5) * h
        A[iq(e):iq(e) + m, iq(e):iq(e) + m] += 0.5 * h * M
        A[iq(e):iq(e) + m, iu(e):iu(e) + m] += D
        A[iu(e):iu(e) + m, iq(e):iq(e) + m] += D
        b[iu(e):iu(e) + m] += 0.5 * h * (w * f(xc + 0.5 * h * xi)) @ phi.T
        for side in ("l", "r"):
            own = left if side == "l" else right
            far = (right if side == "l" else left) if oriented else own
            sg = -1.0 if side == "l" else 1.0
            en = e - 1 if side == "l" else e + 1
            if 0 <= en < n:
                A[iq(e):iq(e) + m, iu(en):iu(en) + m] += -sg * np.outer(own, far)
                A[iu(e):iu(e) + m, iq(en):iq(en) + m] += -sg * np.outer(own, far)
            else:
                xf = e * h if side == "l" else (e + 1) * h
                gv = g(xf)
                b[iq(e):iq(e) + m] += sg * own * gv            # u_hat = g
                A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sg * np.outer(own, own)
                if boundary == "nitsche":
                    A[iu(e):iu(e) + m, iu(e):iu(e) + m] += -sg * c / h * np.outer(own, own)
                    b[iu(e):iu(e) + m] += -sg * c / h * own * gv
    return A, b


def poly_residual(n, k, **kw):
    """Residual of an exactly representable solution u = x^2 (needs k >= 2)."""
    A, b = assemble(n, k, f=lambda x: -2.0, g=lambda x: x ** 2, **kw)
    xi, w, phi, M, D, left, right = basis(k)
    m = k + 1; h = 1.0 / n; Minv = np.linalg.inv(M)
    x = np.zeros(2 * n * m)
    for e in range(n):
        xq = (e + 0.5) * h + 0.5 * h * xi
        x[e * 2 * m:e * 2 * m + m] = Minv @ (phi @ (w * 2 * xq))
        x[e * 2 * m + m:(e + 1) * 2 * m] = Minv @ (phi @ (w * xq ** 2))
    return float(np.linalg.norm(A @ x - b))


def l2_error(n, k, **kw):
    A, b = assemble(n, k, **kw)
    xi, w, phi, M, D, left, right = basis(k)
    m = k + 1; h = 1.0 / n
    rank = int(np.linalg.matrix_rank(A, tol=1e-10))
    sol = spla.spsolve(sp.csc_matrix(A), b)
    num = den = 0.0
    for e in range(n):
        xq = (e + 0.5) * h + 0.5 * h * xi
        uh = phi.T @ sol[e * 2 * m + m:(e + 1) * 2 * m]
        ex = np.sin(np.pi * xq)
        num += 0.5 * h * np.sum(w * (uh - ex) ** 2)
        den += 0.5 * h * np.sum(w * ex ** 2)
    return float(np.sqrt(num / den)), rank, A.shape[0]


if __name__ == "__main__":
    out = {"poly_exactness": {}, "convergence": {}}
    for oriented in (False, True):
        for k in (2, 3):
            key = f"{'oriented' if oriented else 'legacy'}_k{k}"
            try:
                res = poly_residual(4, k, oriented=oriented)
            except Exception as exc:                     # singular system
                res = f"error: {exc}"
            out["poly_exactness"][key] = res
            print(f"poly exactness {key}: {res}", flush=True)
    grid = (8, 16, 32)
    for boundary, c in [("pure", 0.0), ("nitsche", 1.0), ("nitsche", 10.0)]:
        for k in (1, 2, 3):
            errs = [l2_error(n, k, boundary=boundary, c=c) for n in grid]
            orders = [math.log(errs[i][0] / errs[i + 1][0]) / math.log(2)
                      for i in range(len(errs) - 1)]
            key = f"{boundary}_c{c:g}_k{k}"
            out["convergence"][key] = {"errs": [list(map(float, e)) for e in errs],
                                       "orders": orders}
            print(f"{key:18s} " + "  ".join(f"N={n}:{e[0]:.4e}({e[1]}/{e[2]})"
                                            for n, e in zip(grid, errs))
                  + "  ord=" + ",".join(f"{o:.2f}" for o in orders), flush=True)
    with open("validation/ldg_closure/verification_scan.json", "w") as fh:
        json.dump(out, fh, indent=1)