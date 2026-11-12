"""Clean 1-D LDG probe (scratch): is the alternating-flux LDG with a consistent
Dirichlet closure well posed and optimally convergent?

  flux row  (r):  (h/2)(q,r) + (u, r') - [u_hat r]_{ends} = 0
  mass row  (w):  (q, w') - [q_hat w]_{ends} = (h/2)(f, w)

Interior faces: u_hat = u^- (lower index side), q_hat = q^+ (higher index side).
Dirichlet closures: pure / mirror / nit(tau).
"""
import numpy as np


def basis(k):
    nq = max(2 * k + 3, 6)
    xi, w = np.polynomial.legendre.leggauss(nq)
    phi = np.zeros((k + 1, nq))
    dphi = np.zeros((k + 1, nq))
    for j in range(k + 1):
        c = np.zeros(j + 1); c[-1] = 1.0
        phi[j] = np.polynomial.legendre.legval(xi, c)
        dc = np.polynomial.legendre.legder(c)
        dphi[j] = np.polynomial.legendre.legval(xi, dc) if len(dc) else 0.0
    M = np.einsum("q,iq,jq->ij", w, phi, phi)
    D = np.einsum("q,iq,jq->ij", w, dphi, phi)
    left = np.array([(-1.0) ** j for j in range(k + 1)])
    right = np.ones(k + 1)
    return xi, w, phi, M, D, left, right


def assemble(n, k, closure="pure", tau=20.0):
    xi, w, phi, M, D, left, right = basis(k)
    m = k + 1
    h = 1.0 / n
    nd = 2 * n * m
    A = np.zeros((nd, nd)); b = np.zeros(nd)
    iq = lambda e: e * 2 * m
    iu = lambda e: e * 2 * m + m
    for e in range(n):
        xc = (e + 0.5) * h
        A[iq(e):iq(e) + m, iq(e):iq(e) + m] += 0.5 * h * M
        A[iq(e):iq(e) + m, iu(e):iu(e) + m] += D
        A[iu(e):iu(e) + m, iq(e):iq(e) + m] += D
        fq = np.pi ** 2 * np.sin(np.pi * (xc + 0.5 * h * xi))
        b[iu(e):iu(e) + m] += 0.5 * h * (w * fq) @ phi.T
        for side in ("l", "r"):
            phif = left if side == "l" else right
            sgn = -1.0 if side == "l" else 1.0
            outer = np.outer(phif, phif)
            en = e - 1 if side == "l" else e + 1
            if 0 <= en < n:
                ue = en if side == "l" else e
                qe = e if side == "l" else en
                A[iq(e):iq(e) + m, iu(ue):iu(ue) + m] += -sgn * outer
                A[iu(e):iu(e) + m, iq(qe):iq(qe) + m] += -sgn * outer
            else:
                xf = e * h if side == "l" else (e + 1) * h
                gv = np.sin(np.pi * xf)
                if closure == "pure":
                    b[iq(e):iq(e) + m] += sgn * phif * gv
                    A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sgn * outer
                elif closure == "mirror":
                    A[iq(e):iq(e) + m, iu(e):iu(e) + m] += sgn * outer
                    b[iq(e):iq(e) + m] += sgn * 2.0 * phif * gv
                    A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sgn * outer
                elif closure.startswith("nit"):
                    b[iq(e):iq(e) + m] += sgn * phif * gv
                    A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sgn * outer
                    A[iu(e):iu(e) + m, iu(e):iu(e) + m] += -sgn * tau / h * outer
                    b[iu(e):iu(e) + m] += sgn * tau / h * phif * gv
                else:
                    raise ValueError(closure)
    return A, b


def rel_l2(n, k, closure="pure", tau=20.0):
    A, b = assemble(n, k, closure, tau)
    xi, w, phi, *_ = basis(k)
    h = 1.0 / n
    m = k + 1
    rank = np.linalg.matrix_rank(A, tol=1e-10)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    num = den = 0.0
    for e in range(n):
        xq = (e + 0.5) * h + 0.5 * h * xi
        uh = phi.T @ sol[e * 2 * m + m:(e + 1) * 2 * m]
        ex = np.sin(np.pi * xq)
        num += 0.5 * h * np.sum(w * (uh - ex) ** 2)
        den += 0.5 * h * np.sum(w * ex ** 2)
    return np.sqrt(num / den), rank, A.shape[0]


if __name__ == "__main__":
    for closure, tau in [("pure", 0), ("mirror", 0), ("nit", 1.0), ("nit", 10.0), ("nit", 100.0)]:
        for k in (1, 2):
            errs = []
            for n in (8, 16, 32):
                e, rank, nd = rel_l2(n, k, closure, tau)
                errs.append((n, e, rank, nd))
            orders = [np.log(errs[i][1] / errs[i + 1][1]) / np.log(2) for i in range(2)]
            print(f"{closure:6s} tau={tau:6.1f} k={k}  " +
                  "  ".join(f"N={n}: e={e:.3e} rank={r}/{nd}" for n, e, r, nd in errs) +
                  f"  order={['%.2f' % o for o in orders]}")