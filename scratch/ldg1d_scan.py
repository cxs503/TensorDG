"""Scan: which mixed-form DG trace family is (a) consistent, (b) well conditioned, (c) order k+1?

Family on each face of element e (own/far = trace seen from inside/outside):

    u_hat = alpha * u_own + (1 - alpha) * u_far
    q_hat = beta * q_far + (1 - beta) * q_own + eta / h * (u_own - u_far)

alpha = 1, beta = 1, eta = 0 is the textbook one-sided (LDG) pairing;
eta != 0 is a jump penalty (Nitsche/SIPG flavour).  At a Dirichlet face the far
trace is the boundary data g, so the penalty becomes eta/h (u_own - g).
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


def assemble(n, k, f, g, alpha=1.0, beta=1.0, eta=0.0):
    xi, w, phi, M, D, left, right = basis(k)
    m = k + 1; h = 1.0 / n; nd = 2 * n * m
    A = np.zeros((nd, nd)); b = np.zeros(nd)
    iq = lambda e: e * 2 * m
    iu = lambda e: e * 2 * m + m
    for e in range(n):
        xc = (e + 0.5) * h
        A[iq(e):iq(e) + m, iq(e):iq(e) + m] += 0.5 * h * M
        A[iq(e):iq(e) + m, iu(e):iu(e) + m] += D
        A[iu(e):iu(e) + m, iq(e):iq(e) + m] += D
        b[iu(e):iu(e) + m] += 0.5 * h * (w * f(xc + 0.5 * h * xi)) @ phi.T
        for side in ("l", "r"):
            own_end = left if side == "l" else right
            sgn = -1.0 if side == "l" else 1.0          # outward normal sign
            en = e - 1 if side == "l" else e + 1
            far_end = right if side == "l" else left
            if 0 <= en < n:
                # u_hat: alpha * own + (1-alpha) * far
                if alpha:
                    A[iq(e):iq(e) + m, iu(e):iu(e) + m] += -sgn * alpha * np.outer(own_end, own_end)
                if alpha != 1.0:
                    A[iq(e):iq(e) + m, iu(en):iu(en) + m] += -sgn * (1 - alpha) * np.outer(own_end, far_end)
                # q_hat: beta * far + (1-beta) * own
                if beta:
                    A[iu(e):iu(e) + m, iq(en):iq(en) + m] += -sgn * beta * np.outer(own_end, far_end)
                if beta != 1.0:
                    A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sgn * (1 - beta) * np.outer(own_end, own_end)
                # jump penalty  eta/h * (u_own - u_far)
                if eta:
                    A[iu(e):iu(e) + m, iu(e):iu(e) + m] += -sgn * eta / h * np.outer(own_end, own_end)
                    A[iu(e):iu(e) + m, iu(en):iu(en) + m] += +sgn * eta / h * np.outer(own_end, far_end)
            else:
                gv = g[0] if side == "l" else g[1]
                b[iq(e):iq(e) + m] += sgn * own_end * gv
                A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sgn * np.outer(own_end, own_end)
                if eta:
                    A[iu(e):iu(e) + m, iu(e):iu(e) + m] += -sgn * eta / h * np.outer(own_end, own_end)
                    b[iu(e):iu(e) + m] += sgn * eta / h * own_end * gv
    return A, b


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

print("%-28s %-4s %-30s %-12s" % ("(alpha,beta,eta)", "k", "L2 errors N=8,16,32,64", "orders"))
for pol in [(1.0, 1.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 10.0), (1.0, 1.0, 100.0),
            (0.5, 0.5, 10.0), (1.0, 0.5, 10.0), (0.5, 1.0, 10.0)]:
    for k in (1, 2, 3):
        errs = []; conds = []
        for n in (8, 16, 32, 64):
            A, b = assemble(n, k, f, (0.0, 0.0), *pol)
            conds.append(np.linalg.cond(A))
            try:
                sol = spla.spsolve(sp.csc_matrix(A), b)
            except Exception:
                sol = np.full(A.shape[0], np.nan)
            errs.append(l2_error(sol, n, k, ue) if np.all(np.isfinite(sol)) else float("nan"))
        orders = [math.log2(errs[i] / errs[i + 1]) if errs[i] > 0 else float("nan") for i in range(3)]
        print("%-28s %-4d %-30s %-12s cond=%.1e" % (
            str(pol), k, " ".join("%.2e" % e for e in errs), " ".join("%.2f" % o for o in orders), conds[-1]))