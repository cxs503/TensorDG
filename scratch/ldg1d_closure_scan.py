"""1-D LDG: correct trace orientation + Dirichlet-closure scan (scratch, m16388).

Step 1 fixed the neighbour-trace orientation (outer(own, far)).
Step 2 (this file) scans Dirichlet closures to find one that is both
non-singular and order (k+1) in L2:

  u-trace on a Dirichlet face:  'g'      u_hat = g
                                'int'    u_hat = u_int   (natural, Nitsche-style)
  q-flux correction:            q_hat.n = q_int.n + c * (u_int - g) / h**p,  p in {0,1}
"""
import json
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


def assemble(n, k, ubnd="g", c=1.0, hp=1, dense=True):
    xi, w, phi, M, D, left, right = basis(k)
    m = k + 1; h = 1.0 / n; nd = 2 * n * m
    A = np.zeros((nd, nd)); b = np.zeros(nd)
    iq = lambda e: e * 2 * m
    iu = lambda e: e * 2 * m + m
    sc = c / h ** hp
    for e in range(n):
        xc = (e + 0.5) * h
        A[iq(e):iq(e) + m, iq(e):iq(e) + m] += 0.5 * h * M
        A[iq(e):iq(e) + m, iu(e):iu(e) + m] += D
        A[iu(e):iu(e) + m, iq(e):iq(e) + m] += D
        fq = np.pi ** 2 * np.sin(np.pi * (xc + 0.5 * h * xi))
        b[iu(e):iu(e) + m] += 0.5 * h * (w * fq) @ phi.T
        for side in ("l", "r"):
            own = left if side == "l" else right
            far = right if side == "l" else left
            sgn = -1.0 if side == "l" else 1.0
            en = e - 1 if side == "l" else e + 1
            if 0 <= en < n:
                A[iq(e):iq(e) + m, iu(en):iu(en) + m] += -sgn * np.outer(own, far)
                A[iu(e):iu(e) + m, iq(en):iq(en) + m] += -sgn * np.outer(own, far)
            else:
                xf = e * h if side == "l" else (e + 1) * h
                gv = np.sin(np.pi * xf)
                if ubnd == "g":
                    b[iq(e):iq(e) + m] += sgn * own * gv
                else:                                     # u_hat = u_int
                    A[iq(e):iq(e) + m, iu(e):iu(e) + m] += sgn * np.outer(own, own)
                A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sgn * np.outer(own, own)
                if sc:
                    A[iu(e):iu(e) + m, iu(e):iu(e) + m] += -sgn * sc * np.outer(own, own)
                    b[iu(e):iu(e) + m] += sgn * sc * own * gv
    return A, b


def solve(n, k, ubnd="g", c=1.0, hp=1):
    A, b = assemble(n, k, ubnd, c, hp)
    xi, w, phi, *_ = basis(k)
    h = 1.0 / n; m = k + 1
    rank = int(np.linalg.matrix_rank(A, tol=1e-10))
    sol = spla.spsolve(sp.csr_matrix(A), b) if A.shape[0] > 60 else np.linalg.solve(A, b)
    num = den = 0.0
    for e in range(n):
        xq = (e + 0.5) * h + 0.5 * h * xi
        uh = phi.T @ sol[e * 2 * m + m:(e + 1) * 2 * m]
        ex = np.sin(np.pi * xq)
        num += 0.5 * h * np.sum(w * (uh - ex) ** 2)
        den += 0.5 * h * np.sum(w * ex ** 2)
    return float(np.sqrt(num / den)), rank, A.shape[0]


if __name__ == "__main__":
    out = {}
    grid = (8, 16, 32)
    cases = [("g", 0.0, 1), ("g", 1.0, 0), ("g", 1.0, 1), ("g", 10.0, 1),
             ("g", 100.0, 1), ("int", 1.0, 1), ("int", 10.0, 1), ("int", 100.0, 1)]
    for ubnd, c, hp in cases:
        for k in (1, 2, 3):
            errs = [(*solve(n, k, ubnd, c, hp), n) for n in grid]
            orders = [np.log(errs[i][0] / errs[i + 1][0]) / np.log(2) for i in range(len(errs) - 1)]
            tag = f"u={ubnd} c={c:g} p={hp} k={k}"
            out[tag] = {"errs": [list(map(float, e[:2])) + [e[2], e[3]] for e in errs],
                        "orders": [float(o) for o in orders]}
            print(f"{tag:22s} " + "  ".join(f"N={e[3]}:{e[0]:.3e}({e[1]}/{e[2]})" for e in errs)
                  + "  ord=" + ",".join(f"{o:.2f}" for o in orders), flush=True)
    with open("validation/ldg_closure/closure_scan.json", "w") as fh:
        json.dump(out, fh, indent=1)