"""1-D LDG with CORRECT trace orientations (scratch probe, m16388).

Two defects isolated relative to the m16089/m16266 prototype:

D1 (trace orientation). On element e's right face the trace coming from the
   neighbour e+1 must be expanded in the neighbour's *left*-endpoint basis
   L_j(-1) = (-1)^j, not in the neighbour's right-endpoint values (all +1).
   The old code used outer(right, right); the correct block is outer(right, left).
   This is invisible for P0 (a single mode) and lethal for k >= 1.

D2 (Dirichlet diffusive flux). With u_hat = g and q_hat = q_int on a Dirichlet
   face the last element's mass row degenerates to 0 = h f (rank deficiency 1,
   the well-documented blocker).  The cure is a boundary flux correction
   q_hat.n = q_int.n + c*(u_int - g)/h  (Nitsche-type, c > 0).

This probe reports rank and L2 convergence for every combination.
"""
import json
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
    # int L_i' L_j  (derivative on the TEST index i) -- the only convention that
    # makes both the flux row and the mass row consistent (see docs).
    D = np.einsum("q,iq,jq->ij", w, dphi, phi)
    left = np.array([(-1.0) ** j for j in range(k + 1)])
    right = np.ones(k + 1)
    return xi, w, phi, M, D, left, right


def assemble(n, k, boundary="nitsche", c=1.0, oriented=True):
    xi, w, phi, M, D, left, right = basis(k)
    m = k + 1
    h = 1.0 / n
    A = np.zeros((2 * n * m, 2 * n * m)); b = np.zeros(2 * n * m)
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
            own = left if side == "l" else right        # test endpoint of e
            far = right if side == "l" else left        # neighbour's facing endpoint
            if not oriented:
                far = own
            sgn = -1.0 if side == "l" else 1.0
            en = e - 1 if side == "l" else e + 1
            if 0 <= en < n:
                A[iq(e):iq(e) + m, iu(en):iu(en) + m] += -sgn * np.outer(own, far)
                A[iu(e):iu(e) + m, iq(en):iq(en) + m] += -sgn * np.outer(own, far)
            else:
                xf = e * h if side == "l" else (e + 1) * h
                gv = np.sin(np.pi * xf)
                b[iq(e):iq(e) + m] += sgn * own * gv               # u_hat = g
                A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sgn * np.outer(own, own)
                if boundary == "nitsche":
                    A[iu(e):iu(e) + m, iu(e):iu(e) + m] += -sgn * c / h * np.outer(own, own)
                    b[iu(e):iu(e) + m] += sgn * c / h * own * gv
                elif boundary == "pure":
                    pass
                else:
                    raise ValueError(boundary)
    return A, b


def solve(n, k, boundary="nitsche", c=1.0, oriented=True):
    A, b = assemble(n, k, boundary, c, oriented)
    xi, w, phi, *_ = basis(k)
    h = 1.0 / n
    m = k + 1
    rank = int(np.linalg.matrix_rank(A, tol=1e-10))
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    num = den = 0.0
    for e in range(n):
        xq = (e + 0.5) * h + 0.5 * h * xi
        uh = phi.T @ sol[e * 2 * m + m:(e + 1) * 2 * m]
        qh = phi.T @ sol[e * 2 * m:e * 2 * m + m]
        ex = np.sin(np.pi * xq); eq = np.pi * np.cos(np.pi * xq)
        num += 0.5 * h * np.sum(w * (uh - ex) ** 2)
        den += 0.5 * h * np.sum(w * ex ** 2)
    return np.sqrt(num / den), rank, A.shape[0]


if __name__ == "__main__":
    out = {}
    grid = (8, 16, 32, 64)
    for oriented in (False, True):
        for boundary, c in [("pure", 0.0), ("nitsche", 1.0), ("nitsche", 6.0)]:
            for k in (1, 2, 3):
                errs = []
                for n in grid:
                    e, rank, nd = solve(n, k, boundary, c, oriented)
                    errs.append((n, e, rank, nd))
                orders = [np.log(errs[i][1] / errs[i + 1][1]) / np.log(2)
                          for i in range(len(errs) - 1)]
                tag = f"{'oriented' if oriented else 'legacy'}/{boundary}/c={c}/k={k}"
                out[tag] = {"errs": [[n, e, r, nd] for n, e, r, nd in errs],
                            "orders": orders}
                print(f"{tag:32s} " + "  ".join(f"N={n}:{e:.3e}({r}/{nd})" for n, e, r, nd in errs)
                      + "  ord=" + ",".join(f"{o:.2f}" for o in orders))
    with open("validation/ldg_closure/trace_orientation_probe.json", "w") as fh:
        json.dump(out, fh, indent=1)