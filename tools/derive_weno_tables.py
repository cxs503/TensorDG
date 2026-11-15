"""Exact derivation of the WENO reconstruction / smoothness tables (order 5, 7).

The tables used by :mod:`benchmarks.fv_weno7_1d` are *derived* here, not copied
from prose: for order ``q`` this script builds, with exact rationals,

1. the linear map from ``q`` cell averages to the right-edge face value
   ``u_{i+1/2}`` (the fully upwind reconstruction),
2. the same map for every ``r = (q+1)/2``-cell sub-stencil (right-biased),
3. the Jiang--Shu optimal weights ``d`` (solved exactly from consistency +
   normalisation),
4. the Jin--Shu smoothness indicators
   ``beta_k = sum_{l=1..r-1} int (d^l p_k / dx^l)^2 dx``   (unit cell width),

and prints python literals.  The emitted values are re-verified independently in
``tests/test_xval_dg_round2_band_honesty.py`` (ideal-weight blend must
reproduce degree ``q-2`` polynomials to machine precision, and the order-5
tables must coincide with the hard-coded Jiang--Shu WENO5 coefficients already
used by :mod:`benchmarks.fv_weno5_1d`).

Run: ``python tools/derive_weno_tables.py``
"""
from __future__ import annotations

import sympy as sp

X = sp.symbols("x")


def face_coeffs(ncells: int, face_cell: int | None = None) -> list[sp.Rational]:
    """``c`` with ``u_face = sum_k c_k * ubar_k`` on ``ncells`` unit cells [0,ncells].

    ``face_cell`` is the index of the cell whose **right edge** carries the face
    (default: the last cell, i.e. the fully upwind reconstruction).
    """
    ubar = sp.symbols(f"ub0:{ncells}")
    # ``a``: monomial coefficients of the degree-(ncells-1) polynomial whose
    # cell averages over cells j=0..ncells-1 equal ubar_j.
    moments = sp.Matrix([[sp.integrate(X ** k, (X, j, j + 1)) for j in range(ncells)]
                         for k in range(ncells)])
    if face_cell is None:
        face_cell = ncells - 1
    face = sp.Matrix([[sp.Integer(face_cell + 1) ** k for k in range(ncells)]])
    # ubar = moments^T @ a  =>  a = moments^{-T} @ ubar
    w = (face * moments.inv().T)[0, :]
    assert len(ubar) == ncells
    return [sp.nsimplify(w[j]) for j in range(ncells)]


def smoothness_beta(ncells: int) -> list[sp.Expr]:
    """Jin--Shu beta of an ``ncells``-cell sub-stencil (cells [0,1],...,[n-1,n])."""
    u = sp.symbols(f"v0:{ncells}")
    a = sp.symbols(f"a0:{ncells}")
    poly = sum(a[k] * X ** k for k in range(ncells))
    eqs = [sp.integrate(poly, (X, j, j + 1)) - u[j] for j in range(ncells)]
    sol = sp.solve(eqs, a, dict=True)[0]
    p = sp.expand(poly.subs(sol))
    beta = sp.Integer(0)
    for l in range(1, ncells):
        beta += sp.integrate(sp.diff(p, X, l) ** 2, (X, 0, ncells))
    return sp.expand(beta)


def derive(q: int) -> dict:
    r = (q + 1) // 2
    # the full stencil's target face i+1/2 is the right edge of its middle cell
    full = face_coeffs(q, face_cell=(q - 1) // 2)
    # sub-stencil t covers full-stencil cells t..t+r-1; the target face i+1/2 is
    # the right edge of full-stencil cell s = r-1, i.e. sub-cell index r-1-t.
    sub = [face_coeffs(r, face_cell=r - 1 - t) for t in range(r)]
    d = sp.symbols(f"d0:{r}")
    eqs = []
    for k in range(q):
        eqs.append(sum(d[t] * (sub[t][k - t] if 0 <= k - t < r else 0)
                       for t in range(r)) - full[k])
    eqs.append(sum(d) - 1)
    sol = sp.solve(eqs, d, dict=True)[0]
    return {"order": q, "sub_width": r,
            "full": [sp.nsimplify(c) for c in full],
            "sub": [[sp.nsimplify(c) for c in row] for row in sub],
            "weights": [sp.nsimplify(sol[d[t]]) for t in range(r)],
            "beta": smoothness_beta(r)}


def _rat(x) -> str:
    x = sp.Rational(x)
    return f"({x.p}/{x.q})"


def emit(q: int) -> None:
    tab = derive(q)
    r = tab["sub_width"]
    print(f"# order {q}: sub-stencil r={r}, optimal weights "
          f"{[str(w) for w in tab['weights']]}")
    print(f"FULL[{q}] = [{', '.join(_rat(c) for c in tab['full'])}]")
    for t, row in enumerate(tab["sub"]):
        print(f"SUB[{q}][{t}] = [{', '.join(_rat(c) for c in row)}]  "
              f"# cells i-{r}+{t} .. i+{t}")
    print(f"# beta({r} cells) in v0..v{r-1}:")
    print(sp.collect(sp.expand(tab["beta"]), sp.symbols(f"v0:{r}")))


if __name__ == "__main__":
    for order in (5, 7):
        emit(order)
        print()