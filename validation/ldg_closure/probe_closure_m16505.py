"""LDG Dirichlet closure probe (m16505): closing the rank deficiency *and* the order barrier.

Context
-------
``docs/ldg_closure_reassessment_m16388.md`` established two facts and left one open
problem:

* the m16266 probe used ``D_coef`` where the mixed weak form needs ``D_test``
  (derivative on the test function) -> its negative verdict was WITHHELD;
* the ``pure`` alternating flux (``u_hat = u^-``, ``q_hat = q^+``) with Dirichlet
  data at *both* ends is structurally rank deficient, and the ``c/h`` penalty that
  restores full rank only reaches order 2 instead of ``k+1``.

This probe attacks the barrier where m16388 pointed: the Dirichlet faces.  The
exact 1-D algebra says what is missing.  For P0 the mass row of the u-equation on
the *last* element reads

    q_hat(right) = q_hat(left) = q_{N-1}   =>   0 = h f_e        (f != 0)

because the alternating rule borrows q from the right neighbour at every face, so
both boundary faces of the end element borrow from itself.  The trace pair
(``u_hat``, ``q_hat``) is therefore *not* a consistent numerical flux at a
Dirichlet face: consistency of ``u_hat = g`` alone only fixes the u-equation.
The remedy tested here is the standard symmetric one -- stabilise the Dirichlet
face with a penalty on the **u-flux** (``u_hat = g - sigma/h (u_int - g)``) and,
optionally, the matching penalty on the **q-flux** (``q_hat = q_int - sigma/h (u_int - g)``).

Falsifiable criteria fixed *before* running
------------------------------------------
C1 (consistency)  u = x^2, q = 2x is representable exactly for k >= 2; any
                  consistent assembly must give ||A x_exact - b|| = O(1e-12).
C2 (order)        L2 error of u must decay at rate k+1 for k = 1, 2, 3.
C3 (regularity)   the assembled matrix must be non-singular at every (k, N).

A closure is accepted only if C1, C2 and C3 all hold; anything less is recorded as
honest FAIL.  Every number in ``docs/ldg_closure_optimal_order_m16505.md`` is
injected from the JSON emitted here (no hand copying).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "closure_probe_m16505.json")

GRID = (4, 8, 16, 32, 64)
KS = (1, 2, 3)


def basis(k: int):
    """Modal Legendre basis on [-1, 1]; M and D_test from Gauss quadrature."""
    nq = max(2 * k + 4, 8)
    xi, w = np.polynomial.legendre.leggauss(nq)
    phi = np.zeros((k + 1, nq))
    dphi = np.zeros((k + 1, nq))
    for j in range(k + 1):
        c = np.zeros(j + 1)
        c[-1] = 1.0
        phi[j] = np.polynomial.legendre.legval(xi, c)
        dc = np.polynomial.legendre.legder(c)
        dphi[j] = np.polynomial.legendre.legval(xi, dc) if len(dc) else 0.0
    M = np.einsum("q,iq,jq->ij", w, phi, phi)
    D_test = np.einsum("q,iq,jq->ij", w, dphi, phi)
    face = {"l": np.array([(-1.0) ** j for j in range(k + 1)]),
            "r": np.ones(k + 1)}
    return xi, w, phi, M, D_test, face


def assemble(n: int, k: int, flux: str = "alt", sigma: float = 0.0,
             q_penalty: bool = False, f=None, g=None):
    """Mixed LDG assembly for -u'' = f, u(0) = u(1) = 0 (Dirichlet both ends).

    ``flux``     'alt'  : u_hat = u^-, q_hat = q^+ on interior faces
                 'mirror': u_hat = u^+, q_hat = q^-  (opposite pairing)
    ``sigma``    Dirichlet-face stabilisation, ``u_hat = g - sigma/h (u_int - g)``
                 (sigma = 0 reproduces the pure closure of m16388).
    ``q_penalty``add the matching ``- sigma/h (u_int - g)`` correction to ``q_hat``.

    Returns (A, b, meta).
    """
    f = f or (lambda x: math.pi ** 2 * np.sin(math.pi * x))
    g = g or (lambda x: math.sin(math.pi * x))
    xi, w, phi, M, D_test, face = basis(k)
    m = k + 1
    h = 1.0 / n
    nd = 2 * n * m
    A = np.zeros((nd, nd))
    b = np.zeros(nd)
    iq = lambda e: e * 2 * m
    iu = lambda e: e * 2 * m + m
    for e in range(n):
        xc = (e + 0.5) * h
        A[iq(e):iq(e) + m, iq(e):iq(e) + m] += 0.5 * h * M          # int q r
        A[iq(e):iq(e) + m, iu(e):iu(e) + m] += D_test               # int u r'
        A[iu(e):iu(e) + m, iq(e):iq(e) + m] += D_test               # int q v'
        b[iu(e):iu(e) + m] += 0.5 * h * (w * f(xc + 0.5 * h * xi)) @ phi.T
        for side in ("l", "r"):
            own = face[side]
            far = face["r" if side == "l" else "l"]
            sgn = -1.0 if side == "l" else 1.0      # -[. ]_d = -right + left
            en = e - 1 if side == "l" else e + 1
            if 0 <= en < n:
                # interior face: alternating (u_hat from one side, q_hat from the other)
                if flux == "alt":
                    u_e, q_e = (en, e) if side == "l" else (e, en)
                else:
                    u_e, q_e = (e, en) if side == "l" else (en, e)
                # the shared face is the neighbour's *facing* endpoint -> its `far` basis
                u_bas = own if u_e == e else far
                q_bas = own if q_e == e else far
                A[iq(e):iq(e) + m, iu(u_e):iu(u_e) + m] += -sgn * np.outer(own, u_bas)
                A[iu(e):iu(e) + m, iq(q_e):iq(q_e) + m] += -sgn * np.outer(own, q_bas)
            else:
                xf = e * h if side == "l" else (e + 1) * h
                gv = g(xf)
                # u_hat = g (weak Dirichlet) + optional stabilisation
                b[iq(e):iq(e) + m] += sgn * own * gv
                if sigma:
                    A[iq(e):iq(e) + m, iu(e):iu(e) + m] += -sgn * sigma / h * np.outer(own, own)
                    b[iq(e):iq(e) + m] += -sgn * sigma / h * own * gv
                # q_hat = q_int (+ matching penalty)
                A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sgn * np.outer(own, own)
                if q_penalty:
                    A[iu(e):iu(e) + m, iu(e):iu(e) + m] += sgn * sigma / h * np.outer(own, own)
                    b[iu(e):iu(e) + m] += sgn * sigma / h * own * gv
    return A, b, {"n": n, "k": k, "nd": nd, "h": h}


def _exact_coeffs(n: int, k: int, u=None, q=None):
    """Coordinates of a polynomial pair in the (q, u) block layout."""
    u = u or (lambda x: x * x)
    q = q or (lambda x: 2.0 * x)
    xi, w, phi, M, D_test, face = basis(k)
    m = k + 1
    h = 1.0 / n
    x = np.zeros(2 * n * m)
    for e in range(n):
        xc = (e + 0.5) * h
        pts = xc + 0.5 * h * xi
        # modal coefficients: c_j = (2j+1)/2 * int_{-1}^{1} f(xi) L_j dxi  (L2-normalised basis)
        for j in range(m):
            cj = np.zeros(j + 1)
            cj[-1] = 1.0
            Lj = np.polynomial.legendre.legval(xi, cj)
            norm = 2.0 / (2 * j + 1)
            x[e * 2 * m + j] = np.sum(w * q(pts) * Lj) / norm
            x[e * 2 * m + m + j] = np.sum(w * u(pts) * Lj) / norm
    return x


def poly_residual(n: int, k: int, sigma: float, q_penalty: bool) -> float:
    """C1: residual of the exactly representable pair (u = x^2, q = 2x)."""
    A, b, meta = assemble(n, k, "alt", sigma, q_penalty,
                          f=lambda x: -2.0, g=lambda x: x * x)
    xe = _exact_coeffs(n, k)
    return float(np.linalg.norm(A @ xe - b))


def l2_error(n: int, k: int, sigma: float, q_penalty: bool, field: str = "u"):
    """L2 error of the discrete solution (field 'u' or 'q') against the sinusoid."""
    A, b, meta = assemble(n, k, "alt", sigma, q_penalty)
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return float("nan"), 0
    xi, w, phi, M, D_test, face = basis(k)
    m = k + 1
    h = meta["h"]
    err2 = 0.0
    for e in range(n):
        xc = (e + 0.5) * h
        pts = xc + 0.5 * h * xi
        off = 0 if field == "q" else m
        num = x[e * 2 * m + off: e * 2 * m + off + m] @ phi
        ex = (np.pi * np.cos(np.pi * pts)) if field == "q" else np.sin(np.pi * pts)
        err2 += 0.5 * h * np.sum(w * (num - ex) ** 2)
    return float(math.sqrt(err2)), int(np.linalg.matrix_rank(A))


def rank_of(n: int, k: int, sigma: float, q_penalty: bool):
    A, b, meta = assemble(n, k, "alt", sigma, q_penalty)
    return int(np.linalg.matrix_rank(A)), meta["nd"]


def main():
    out = {"grid": list(GRID), "ks": list(KS), "poly": {}, "convergence": {},
           "rank": {}, "criteria": {}}
    # ---- C1 polynomial exactness -------------------------------------------------
    for k in KS:
        for sigma, qp in ((0.0, False), (10.0, False), (10.0, True)):
            key = f"k{k}_sigma{sigma:g}_qpen{int(qp)}"
            out["poly"][key] = poly_residual(4, k, sigma, qp)
    # ---- C3 rank ---------------------------------------------------------------
    for sigma, qp in ((0.0, False), (10.0, False), (10.0, True)):
        for k in KS:
            key = f"k{k}_sigma{sigma:g}_qpen{int(qp)}"
            out["rank"][key] = [list(rank_of(n, k, sigma, qp)) for n in GRID]
    # ---- C2 order --------------------------------------------------------------
    for sigma, qp in ((0.0, False), (10.0, False), (10.0, True)):
        for k in KS:
            for field in ("u", "q"):
                errs, ranks = [], []
                for n in GRID:
                    e, r = l2_error(n, k, sigma, qp, field)
                    errs.append(e)
                    ranks.append(r)
                orders = []
                for i in range(len(errs) - 1):
                    a, c = errs[i], errs[i + 1]
                    if np.isfinite(a) and np.isfinite(c) and a > 0 and c > 0:
                        orders.append(float(math.log(a / c) / math.log(2.0)))
                    else:
                        orders.append(float("nan"))
                out["convergence"][f"k{k}_sigma{sigma:g}_qpen{int(qp)}_{field}"] = {
                    "errs": [float(e) for e in errs], "orders": orders, "ranks": ranks}
    # ---- verdict per criterion -------------------------------------------------
    ok = []
    for key, val in out["convergence"].items():
        if not key.endswith("_u"):
            continue
        k = int(key[1])
        fin = [o for o in val["orders"] if np.isfinite(o)]
        tail = fin[-2:] if len(fin) >= 2 else fin
        out["criteria"][key] = {
            "tail_order": float(min(tail)) if tail else float("nan"),
            "target": k + 1, "pass": bool(tail) and min(tail) >= (k + 1) - 0.35}
        ok.append(out["criteria"][key]["pass"])
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    res = main()
    print("C1 polynomial residual (should be ~1e-12 for k>=2):")
    for k, v in res["poly"].items():
        print(f"  {k:28s} {v:.3e}")
    print("\nC2/C3 convergence (u):")
    for k, v in res["convergence"].items():
        if not k.endswith("_u"):
            continue
        rank_ok = all(r == res["rank"][k.replace("_u", "")][i][1]
                      for i, r in enumerate(v["ranks"]))
        print(f"  {k:32s} errs={['%.2e' % e if np.isfinite(e) else 'nan' for e in v['errs']]}"
              f" orders={['%.2f' % o if np.isfinite(o) else 'nan' for o in v['orders']]}"
              f" fullrank={rank_ok}")
    print("\nverdicts:")
    for k, v in res["criteria"].items():
        print(f"  {k:32s} tail_order={v['tail_order']:.2f} target={v['target']} pass={v['pass']}")