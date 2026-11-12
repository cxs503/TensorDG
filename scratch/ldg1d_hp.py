"""High-precision probe: is the one-sided (LDG) mixed system merely ill-conditioned,
or does its exact solution actually converge?"""
import math
import mpmath as mp
import numpy as np

mp.mp.dps = 60


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


def assemble(n, k, u_case, q_case):
    """u_case/q_case in {'own','far'} — which element supplies the trace, per element."""
    xi, w, phi, M, D, left, right = basis(k)
    m = k + 1; h = 1.0 / n; nd = 2 * n * m
    A = mp.zeros(nd, nd); b = [mp.mpf(0)] * nd
    f = lambda x: mp.mpf(math.pi) ** 2 * mp.sin(mp.mpf(math.pi) * mp.mpf(x))
    iq = lambda e: e * 2 * m
    iu = lambda e: e * 2 * m + m
    for e in range(n):
        xc = (e + 0.5) * h
        for i in range(m):
            for j in range(m):
                A[iq(e) + i, iq(e) + j] += mp.mpf(0.5 * h * M[i, j])
                A[iq(e) + i, iu(e) + j] += mp.mpf(D[i, j])
                A[iu(e) + i, iq(e) + j] += mp.mpf(D[i, j])
        for i in range(m):
            for q in range(len(xi)):
                x = mp.mpf(xc + 0.5 * h * xi[q])
                b[iu(e) + i] = b[iu(e) + i] + mp.mpf(0.5 * h * w[q] * phi[i, q]) * f(x)
        # boundary u_hat = g = 0 at x=0 and x=1
        for side in ("l", "r"):
            own_end = left if side == "l" else right
            sgn = -1.0 if side == "l" else 1.0
            en = e - 1 if side == "l" else e + 1
            far_end = right if side == "l" else left
            if 0 <= en < n:
                ue = e if u_case == "own" else en
                end_u = own_end if ue == e else far_end
                qe = e if q_case == "own" else en
                end_q = own_end if qe == e else far_end
                for i in range(m):
                    for j in range(m):
                        A[iq(e) + i, iu(ue) + j] += mp.mpf(-sgn * own_end[i] * end_u[j])
                        A[iu(e) + i, iq(qe) + j] += mp.mpf(-sgn * own_end[i] * end_q[j])
            else:
                for i in range(m):
                    A[iu(e) + i, iq(e) + i] += mp.mpf(-sgn * own_end[i] * own_end[i])
    return A, b


def solve_err(n, k, u_case, q_case):
    A, b = assemble(n, k, u_case, q_case)
    lu = mp.lu(A)
    sol = mp.lu_solve(lu, b)
    xi2, w2 = np.polynomial.legendre.leggauss(max(2 * k + 8, 16))
    phi2 = np.zeros((k + 1, len(xi2)))
    for j in range(k + 1):
        c = np.zeros(j + 1); c[-1] = 1.0
        phi2[j] = np.polynomial.legendre.legval(xi2, c)
    h = 1.0 / n; m = k + 1; err = mp.mpf(0)
    for e in range(n):
        xc = (e + 0.5) * h
        for q in range(len(xi2)):
            uh = sum(sol[e * 2 * m + m + j] * mp.mpf(phi2[j, q]) for j in range(m))
            x = mp.mpf(xc + 0.5 * h * xi2[q])
            exact = mp.sin(mp.mpf(math.pi) * x)
            err += mp.mpf(0.5 * h * w2[q]) * (uh - exact) ** 2
    return mp.sqrt(err)


if __name__ == "__main__":
    for uc, qc in (("own", "far"), ("far", "own"), ("own", "own"), ("far", "far")):
        for k in (1, 2, 3):
            errs = [solve_err(n, k, uc, qc) for n in (4, 8, 16)]
            rates = [mp.log(errs[i] / errs[i + 1]) / mp.log(2) for i in range(2)]
            print("u=%-4s q=%-4s k=%d  err=%s  rate=%s" % (
                uc, qc, k, [mp.nstr(e, 4) for e in errs], [mp.nstr(r, 4) for r in rates]))