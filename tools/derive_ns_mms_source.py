"""Generate the manufactured solution (and its source term) for the 2-D NS DG solver.

The Navier--Stokes module ``tensordg/ns_dg_2d.py`` integrates the *homogeneous*
compressible equations.  To verify its spatial + temporal order of accuracy we
need a smooth exact solution: a manufactured solution ``U(x, y, t)`` injected
through the volumetric ``source`` hook.  The source

    S = dU/dt + div F_inv - div F_visc

is derived symbolically here (no hand differentiation) and emitted as a static,
torch-compatible module so that neither the benchmark nor the test suite needs
sympy at run time.

Sign convention of ``ns_dg_2d``: the total flux is ``F_inv - F_visc`` with

    F_inv  = [rho u, rho u^2 + p, rho u v, u (E + p)]           (x direction)
    F_visc = [0, tau_xx, tau_xy, u tau_xx + v tau_xy + kappa T_x]
    tau_xx = 2 mu u_x - (2/3) mu (u_x + v_y)
    tau_xy = mu (u_y + v_x)
    kappa  = mu c_p / Pr,   c_p = gamma / (gamma - 1)   (unit gas constant)

so that ``S`` is exactly the source a manufactured field needs.

Usage::

    PYTHONPATH=. python tools/derive_ns_mms_source.py
"""
from __future__ import annotations

import io
import os

import sympy as sp
from sympy.printing.numpy import NumPyPrinter

GAMMA = sp.Rational(14, 10)
CV = 1 / (GAMMA - 1)
CP = GAMMA / (GAMMA - 1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "tensordg", "mms_ns_2d.py")

# Manufactured primitive fields: smooth, 1-periodic in x and y, genuinely
# time dependent, O(10 %) amplitude around a physical background state.
x, y, t = sp.symbols("x y t", real=True)
mu, kappa = sp.symbols("mu kappa", positive=True)
TWOPI = 2 * sp.pi

RHO = 1 + sp.Rational(2, 10) * sp.sin(TWOPI * x + sp.Rational(4, 10) * t) * sp.cos(TWOPI * y - sp.Rational(3, 10) * t)
U = sp.Rational(5, 10) + sp.Rational(1, 10) * sp.sin(TWOPI * y + sp.Rational(2, 10) * t) * sp.cos(TWOPI * x)
V = sp.Rational(-4, 10) + sp.Rational(1, 10) * sp.sin(TWOPI * x - sp.Rational(1, 10) * t) * sp.cos(TWOPI * y + sp.Rational(25, 100) * t)
P = 1 + sp.Rational(25, 100) * sp.cos(TWOPI * x + sp.Rational(15, 100) * t) * sp.sin(TWOPI * y + sp.Rational(2, 10) * t)

T = P / RHO
ENERGY = P / (GAMMA - 1) + sp.Rational(1, 2) * RHO * (U ** 2 + V ** 2)

CONSERVED = [RHO, RHO * U, RHO * V, ENERGY]
FLUX_X = [RHO * U, RHO * U ** 2 + P, RHO * U * V, U * (ENERGY + P)]
FLUX_Y = [RHO * V, RHO * U * V, RHO * V ** 2 + P, V * (ENERGY + P)]

TAU_XX = 2 * mu * sp.diff(U, x) - sp.Rational(2, 3) * mu * (sp.diff(U, x) + sp.diff(V, y))
TAU_YY = 2 * mu * sp.diff(V, y) - sp.Rational(2, 3) * mu * (sp.diff(U, x) + sp.diff(V, y))
TAU_XY = mu * (sp.diff(U, y) + sp.diff(V, x))
HEAT_X = kappa * sp.diff(T, x)
HEAT_Y = kappa * sp.diff(T, y)

VISCOUS_X = [sp.Integer(0), TAU_XX, TAU_XY, U * TAU_XX + V * TAU_XY + HEAT_X]
VISCOUS_Y = [sp.Integer(0), TAU_XY, TAU_YY, U * TAU_XY + V * TAU_YY + HEAT_Y]

SOURCE = [
    sp.diff(CONSERVED[i], t)
    + sp.diff(FLUX_X[i], x) + sp.diff(FLUX_Y[i], y)
    - sp.diff(VISCOUS_X[i], x) - sp.diff(VISCOUS_Y[i], y)
    for i in range(4)
]

PRINTER = NumPyPrinter()


def _emit(exprs, argnames):
    """Return (replacement-lines, expressions written in terms of the cse symbols)."""
    cse_pairs, reduced = sp.cse(exprs, optimizations="basic", order="none")
    lines = []
    subs = {sym: sp.Symbol(f"_c{i}") for i, (sym, _) in enumerate(cse_pairs)}
    for i, (_sym, expr) in enumerate(cse_pairs):
        code = PRINTER.doprint(expr.subs(subs)).replace("numpy.", "torch.")
        lines.append(f"    _c{i} = {code}")
    outs = [PRINTER.doprint(expr.subs(subs)).replace("numpy.", "torch.") for expr in reduced]
    return lines, outs


def build():
    header = '''"""Manufactured solution for the 2-D compressible Navier--Stokes DG solver.

GENERATED FILE -- do not edit by hand.
Regenerate with ``PYTHONPATH=. python tools/derive_ns_mms_source.py``.

The primitive fields are smooth, 1-periodic in ``x`` and ``y`` and genuinely
time dependent::

    rho = 1    + 0.20 sin(2 pi x + 0.40 t) cos(2 pi y - 0.30 t)
    u   = 0.5  + 0.10 sin(2 pi y + 0.20 t) cos(2 pi x)
    v   = -0.4 + 0.10 sin(2 pi x - 0.10 t) cos(2 pi y + 0.25 t)
    p   = 1    + 0.25 cos(2 pi x + 0.15 t) sin(2 pi y + 0.20 t)

They are *not* solutions of the homogeneous equations; :func:`mms_source`
returns exactly the conserved-variable source that makes them exact solutions,
derived symbolically (sympy, ``tools/derive_ns_mms_source.py``) from

    S = dU/dt + div F_inv - div F_visc

with the ``ns_dg_2d`` sign convention (total flux ``F_inv - F_visc``), unit gas
constant, ``gamma = 1.4`` and ``kappa`` given explicitly.
"""
from __future__ import annotations

import torch

__all__ = [
    "GAMMA",
    "CP",
    "mms_primitives",
    "mms_conserved",
    "mms_source",
]

GAMMA = 1.4
CV = 1.0 / (GAMMA - 1.0)
CP = GAMMA / (GAMMA - 1.0)


'''

    prim_lines, prim_out = _emit([RHO, U, V, P], ("x", "y", "t"))
    cons_lines, cons_out = _emit(CONSERVED, ("x", "y", "t"))
    src_lines, src_out = _emit(SOURCE, ("x", "y", "t", "mu", "kappa"))

    body = ["def mms_primitives(x, y, t):",
            '    """Return ``(rho, u, v, p)`` of the manufactured solution."""']
    body += prim_lines
    body += [f"    return {', '.join(prim_out)}", "", ""]

    body += ["def mms_conserved(x, y, t):",
             '    """Return the conserved state ``(rho, rho u, rho v, E)``."""']
    body += cons_lines
    body += [f"    return {', '.join(cons_out)}", "", ""]

    body += ["def mms_source(x, y, t, mu, kappa):",
             '    """Return the conserved-variable source ``(S_rho, S_rhou, S_rhov, S_E)``."""']
    body += src_lines
    body += [f"    return {', '.join(src_out)}", ""]

    return header + "\n".join(body)


if __name__ == "__main__":
    code = build()
    compile(code, TARGET, "exec")  # syntax gate before writing
    io.open(TARGET, "w", encoding="utf-8").write(code)
    print(f"wrote {TARGET} ({len(code)} bytes, {code.count(chr(10))} lines)")
    # Report the size of the symbolic expressions that were emitted.
    for name, exprs in (("conserved", CONSERVED), ("source", SOURCE)):
        ops = sum(sp.count_ops(e) for e in exprs)
        print(f"  {name}: {ops} operations")