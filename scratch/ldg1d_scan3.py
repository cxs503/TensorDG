"""Scan 3: face-oriented (global) alternating traces + parametrised Dirichlet closure.

Interior (the classical staggered LDG orientation, u^- from the left element of the
face, q^+ from the right element):

    element e, left face :  u_hat = u_{e-1} (far) ,  q_hat = q_e (own)
    element e, right face:  u_hat = u_e     (own) ,  q_hat = q_{e+1} (far)

For k = 0 this collapses to  h q_e = u_e - u_{e-1},  q_e - q_{e+1} = h f_e,
i.e. the classical staggered scheme (nonsingular up to the constant mode).

Boundary: u_hat = g (Dirichlet).  For q_hat the pure "interior trace" choice gives
the degenerate last row  0 = h f  (k = 0, verified analytically); the classical
staggered scheme instead uses the discrete difference against the boundary data.
We scan   q_hat = q_int + s * beta / h * (u_int - g)   on both Dirichlet faces.
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


def assemble(n, k, f, g, beta=0.0, sgnb=-1.0):
    xi, w, phi, M, D, left, right = basis(k)
    m = k + 1; h = 1.0 / n; nd = 2 * n * m
    A = np.zeros((nd, nd)); b = np.zeros(nd)
    iq = lambda e: e * 2 * m
    iu = lambda e: e * 2 * m + m
    O = np.outer
    for e in range(n):
        xc = (e + 0.5) * h
        A[iq(e):iq(e) + m, iq(e):iq(e) + m] += 0.5 * h * M
        A[iq(e):iq(e) + m, iu(e):iu(e) + m] += D
        A[iu(e):iu(e) + m, iq(e):iq(e) + m] += D
        b[iu(e):iu(e) + m] += 0.5 * h * (w * f(xc + 0.5 * h * xi)) @ phi.T
        for side in ("l", "r"):
            own_end = left if side == "l" else right
            sgn = -1.0 if side == "l" else 1.0
            en = e - 1 if side == "l" else e + 1
            far_end = right if side == "l" else left
            if 0 <= en < n:
                # u_hat: from the left element of the face  -> right face: own, left: far
                ue = e if side == "r" else en
                A[iq(e):iq(e) + m, iu(ue):iu(ue) + m] += -sgn * O(own_end, own_end if ue == e else far_end)
                # q_hat: from the right element of the face -> right face: far, left: own
                qe = en if side == "r" else e
                A[iu(e):iu(e) + m, iq(qe):iq(qe) + m] += -sgn * O(own_end, own_end if qe == e else far_end)
            else:
                gv = g[0] if side == "l" else g[1]
                b[iq(e):iq(e) + m] += sgn * own_end * gv
                A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sgn * O(own_end, own_end)
                if beta:
                    A[iu(e):iu(e) + m, iu(e):iu(e) + m] += -sgn * sgnb * beta / h * O(own_end, own_end)
                    b[iu(e):iu(e) + m] += -sgn * sgnb * beta / h * own_end * gv
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

print("%-22s %-3s %-38s %-14s %-9s" % ("(beta,sgn)", "k", "L2 (N=8,16,32,64)", "orders", "cond@64"))
for beta, sgnb in [(0.0, -1.0), (2.0, -1.0), (2.0, 1.0), (10.0, -1.0), (10.0, 1.0)]:
    for k in (0, 1, 2, 3):
        errs = []; cond = float("nan")
        for n in (8, 16, 32, 64):
            A, b = assemble(n, k, f, (0.0, 0.0), beta, sgnb)
            if n == 64:
                sv = np.linalg.svd(A, compute_uv=False)
                cond = sv[0] / sv[-1]
            try:
                sol = spla.spsolve(sp.csc_matrix(A), b)
            except Exception:
                sol = np.full(A.shape[0], np.nan)
            errs.append(l2_error(sol, n, k, ue) if np.all(np.isfinite(sol)) else float("nan"))
        ords = [math.log2(errs[i] / errs[i + 1]) if (np.isfinite(errs[i]) and errs[i] > 0) else float("nan")
                for i in range(3)]
        print("%-22s %-3d %-38s %-14s %.1e" % (
            "b=%.0f s=%+.0f" % (beta, sgnb), k,
            " ".join("%.3e" % e for e in errs), " ".join("%.2f" % o for o in ords), cond))