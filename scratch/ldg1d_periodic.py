"""Is the one-sided mixed-form failure interior or boundary driven?

Periodic assembly (no Dirichlet closure at all): the same traces, wrap-around.
If the periodic problem is well posed and optimally convergent, the singular
behaviour seen with Dirichlet comes from the boundary closure, not the interior.
Reports the numerical rank deficiency via the singular-value spectrum.
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


def assemble_periodic(n, k, f):
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
            en = (e - 1) % n if side == "l" else (e + 1) % n
            far_end = right if side == "l" else left
            # u_hat = own (per-element u^-), q_hat = far (per-element q^+)
            A[iq(e):iq(e) + m, iu(e):iu(e) + m] += -sgn * O(own_end, own_end)
            A[iu(e):iu(e) + m, iq(en):iq(en) + m] += -sgn * O(own_end, far_end)
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


f = lambda x: 4 * np.pi ** 2 * np.sin(2 * np.pi * x)
ue = lambda x: np.sin(2 * np.pi * x)

for k in (0, 1, 2, 3):
    errs = []
    for n in (8, 16, 32):
        A, b = assemble_periodic(n, k, f)
        sv = np.linalg.svd(A, compute_uv=False)
        nsing = int(np.sum(sv < 1e-10 * sv[0]))
        sol = spla.spsolve(sp.csc_matrix(A), b)
        ok = np.all(np.isfinite(sol))
        errs.append(l2_error(sol, n, k, ue) if ok else float("nan"))
    ords = [math.log2(errs[i] / errs[i + 1]) if np.isfinite(errs[i]) and errs[i] > 0 else float("nan")
            for i in range(2)]
    print("periodic k=%d  err=%s  order=%s  cond=%.2e  nsing=%d" % (
        k, " ".join("%.3e" % e for e in errs), " ".join("%.2f" % o for o in ords), sv[0] / sv[-1], nsing))