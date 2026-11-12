"""Scan 2: generalized fluxes  u_hat = own + a*(own - far),  q_hat = far + (b/h)*(own - far).

Baseline (a = b = 0) is the textbook one-sided LDG pair.  The a-term couples the
u-block of the q-row (it is what constrains the even modes that cancel in the
one-sided choice: for k = 0 the one-sided q-row degenerates to h*q_e = 0).
Consistency is automatic: [u] -> 0 for a smooth solution.
"""
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def basis(k):
    nq = max(2 * k + 4, 10)
    xi, w = np.polynomial.legendre.leggauss(nq)
    phi = np.zeros((k + 1, nq)); dphi = np.zeros((k + 1, nq))
    for j in range(k + 1):
        c = np.zeros(j + 1); c[-1] = 1.0
        phi[j] = np.polynomial.legendre.legval(xi, c)
        dc = np.polynomial.legendre.legder(c)
        dphi[j] = np.polynomial.legendre.legval(xi, dc) if len(dc) else 0.0
    M = np.einsum("q,iq,jq->ij", w, phi, phi)
    D = np.einsum("q,iq,jq->ij", w, dphi, phi)
    return xi, w, phi, M, D, np.array([(-1.0) ** j for j in range(k + 1)]), np.ones(k + 1)


def assemble(n, k, f, g, a=0.0, b=0.0):
    xi, w, phi, M, D, left, right = basis(k)
    m = k + 1; h = 1.0 / n; nd = 2 * n * m
    A = np.zeros((nd, nd)); bb = np.zeros(nd)
    iq = lambda e: e * 2 * m
    iu = lambda e: e * 2 * m + m
    O = np.outer
    for e in range(n):
        xc = (e + 0.5) * h
        A[iq(e):iq(e) + m, iq(e):iq(e) + m] += 0.5 * h * M
        A[iq(e):iq(e) + m, iu(e):iu(e) + m] += D
        A[iu(e):iu(e) + m, iq(e):iq(e) + m] += D
        bb[iu(e):iu(e) + m] += 0.5 * h * (w * f(xc + 0.5 * h * xi)) @ phi.T
        for side in ("l", "r"):
            own_end = left if side == "l" else right
            sgn = -1.0 if side == "l" else 1.0
            en = e - 1 if side == "l" else e + 1
            far_end = right if side == "l" else left
            if 0 <= en < n:
                # u_hat = own + a (own - far)
                A[iq(e):iq(e) + m, iu(e):iu(e) + m] += -sgn * (1 + a) * O(own_end, own_end)
                A[iq(e):iq(e) + m, iu(en):iu(en) + m] += -sgn * (-a) * O(own_end, far_end)
                # q_hat = far + (b/h) (own - far)
                A[iu(e):iu(e) + m, iq(en):iq(en) + m] += -sgn * 1.0 * O(own_end, far_end)
                if b:
                    A[iu(e):iu(e) + m, iu(e):iu(e) + m] += -sgn * b * O(own_end, own_end)
                    A[iu(e):iu(e) + m, iu(en):iu(en) + m] += -sgn * (-b) * O(own_end, far_end)
            else:
                gv = g[0] if side == "l" else g[1]
                bb[iq(e):iq(e) + m] += sgn * own_end * gv
                A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sgn * O(own_end, own_end)
                if b:
                    A[iu(e):iu(e) + m, iu(e):iu(e) + m] += -sgn * b * O(own_end, own_end)
                    bb[iu(e):iu(e) + m] += sgn * b / h * own_end * gv
    return A, bb


def l2_error(sol, n, k, ue):
    xi, w, phi, M, D, left, right = basis(k)
    xi2, w2 = np.polynomial.legendre.leggauss(max(2 * k + 8, 16))
    phi2 = np.zeros((k + 1, len(xi2)))
    for j in range(k + 1):
        c = np.zeros(j + 1); c[-1] = 1.0
        phi2[j] = np.polynomial.legendre.legval(xi2, c)
    h = 1.0 / n; m = k + 1; err = 0.0
    for e in range(n):
        xc = (e + 0.5) * h
        c = sol[e * 2 * m + m:(e + 1) * 2 * m]
        x = xc + 0.5 * h * xi2
        err += 0.5 * h * np.sum(w2 * (c @ phi2 - ue(x)) ** 2)
    return math.sqrt(err)


f = lambda x: np.pi ** 2 * np.sin(np.pi * x)
ue = lambda x: np.sin(np.pi * x)

print("%-14s %-3s %-34s %-14s %-9s" % ("(a,b)", "k", "L2 (N=8,16,32,64)", "orders", "cond@64"))
for a, b in [(0.0, 1.0), (0.0, 4.0), (1.0, 1.0), (1.0, 4.0), (0.0, 100.0), (1.0, 100.0)]:
    for k in (1, 2, 3):
        errs = []; cond = float("nan")
        for n in (8, 16, 32, 64):
            A, bb = assemble(n, k, f, (0.0, 0.0), a, b)
            if n == 64:
                cond = np.linalg.cond(A)
            try:
                sol = spla.spsolve(sp.csc_matrix(A), bb)
            except Exception:
                sol = np.full(A.shape[0], np.nan)
            ok = np.all(np.isfinite(sol))
            errs.append(l2_error(sol, n, k, ue) if ok else float("nan"))
        orders = [math.log2(errs[i] / errs[i + 1]) if (errs[i] > 0 and np.isfinite(errs[i])) else float("nan")
                  for i in range(3)]
        print("%-14s %-3d %-34s %-14s %.1e" % (
            "a=%.1f,b=%.1f" % (a, b), k, " ".join("%.3e" % e for e in errs),
            " ".join("%.2f" % o for o in orders), cond))