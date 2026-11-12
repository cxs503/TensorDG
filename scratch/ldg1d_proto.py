"""Prototype v2: mixed-form LDG 1D with explicit trace-source / endpoint bookkeeping.

Weak form per element K = (x_l, x_r), test (r for q-row, v for the flux row):

    int q r + int u r_x - u_hat(x_r) r(x_r) + u_hat(x_l) r(x_l) = 0
    int q v_x - q_hat(x_r) v(x_r) + q_hat(x_l) v(x_l)          = int f v

LDG traces: u_hat = u^- (trace of the element on the left of the face),
            q_hat = q^+ (trace of the element on the right of the face).
"""
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def basis(k):
    nq = max(2 * k + 3, 8)
    xi, w = np.polynomial.legendre.leggauss(nq)
    phi = np.zeros((k + 1, nq)); dphi = np.zeros((k + 1, nq))
    for j in range(k + 1):
        c = np.zeros(j + 1); c[-1] = 1.0
        phi[j] = np.polynomial.legendre.legval(xi, c)
        dc = np.polynomial.legendre.legder(c)
        dphi[j] = np.polynomial.legendre.legval(xi, dc) if len(dc) else 0.0
    M = np.einsum("q,iq,jq->ij", w, phi, phi)
    D = np.einsum("q,iq,jq->ij", w, dphi, phi)   # derivative on the test function
    return xi, w, phi, M, D, np.array([(-1.0) ** j for j in range(k + 1)]), np.ones(k + 1)


def solve(n, k, f, g=(0.0, 0.0), u_trace="ldg", q_trace="ldg", tbasis="facing", regularize=0.0):
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
            sgn = -1.0 if side == "l" else 1.0      # matrix sign of the face term
            en = e - 1 if side == "l" else e + 1
            far_end = right if side == "l" else left

            def endpoint(src_own):
                if tbasis == "self":
                    return own_end
                return own_end if src_own else far_end

            if 0 <= en < n:
                # u_hat: LDG -> u^- (per-element: own trace on BOTH faces)
                if u_trace == "ldg":
                    own_u = (side == "r")
                else:
                    own_u = {"own": True, "far": False}.get(u_trace, False)
                ue = e if own_u else en
                A[iq(e):iq(e) + m, iu(ue):iu(ue) + m] += -sgn * np.outer(own_end, endpoint(own_u))
                # q_hat: LDG -> q^+ (per-element: neighbour trace on BOTH faces)
                if q_trace == "ldg":
                    own_q = (side == "l")
                else:
                    own_q = {"own": True, "far": False}.get(q_trace, False)
                qe = e if own_q else en
                A[iu(e):iu(e) + m, iq(qe):iq(qe) + m] += -sgn * np.outer(own_end, endpoint(own_q))
            else:
                gv = g[0] if side == "l" else g[1]
                b[iq(e):iq(e) + m] += sgn * own_end * gv
                # q_hat on the Dirichlet face = interior trace of the own element
                A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sgn * np.outer(own_end, endpoint(True))
    if regularize:
        A += regularize * np.eye(nd)
    sol = spla.spsolve(sp.csc_matrix(A), b)
    return sol if np.all(np.isfinite(sol)) else None


def l2_error(sol, n, k, ue):
    xi, w, phi, M, D, left, right = basis(k)
    xi2, w2 = np.polynomial.legendre.leggauss(max(2 * k + 6, 12))
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


f = lambda x: math.pi ** 2 * np.sin(math.pi * x)
ue = lambda x: np.sin(math.pi * x)
for pol in [("own", "far", "facing"), ("far", "own", "facing"), ("own", "own", "facing"),
            ("far", "far", "facing"), ("own", "far", "self")]:
    for k in (1, 2, 3):
        errs = []
        for n in (8, 16, 32):
            s = solve(n, k, f, u_trace=pol[0], q_trace=pol[1], tbasis=pol[2])
            errs.append(float("nan") if s is None else l2_error(s, n, k, ue))
        rate = [math.log2(errs[i] / errs[i + 1]) for i in range(2)]
        print("%-32s k=%d" % (str(pol), k), ["%.3e" % x for x in errs], ["%.2f" % x for x in rate])