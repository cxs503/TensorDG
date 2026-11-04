"""Manufactured solution for the 2-D compressible Navier--Stokes DG solver.

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


def mms_primitives(x, y, t):
    """Return ``(rho, u, v, p)`` of the manufactured solution."""
    _c0 = 2*torch.pi*y
    _c1 = (1/5)*t
    _c2 = torch.pi*x
    _c3 = 2*_c2
    _c4 = torch.sin(_c0 + _c1)
    return (1/5)*torch.sin(2*_c1 + 2*_c2)*torch.cos(_c0 - 3/10*t) + 1, (1/10)*_c4*torch.cos(_c3) + 1/2, (1/10)*torch.sin(_c3 - 1/10*t)*torch.cos(_c0 + (1/4)*t) - 2/5, (1/4)*_c4*torch.cos(_c3 + (3/20)*t) + 1


def mms_conserved(x, y, t):
    """Return the conserved state ``(rho, rho u, rho v, E)``."""
    _c0 = 2*torch.pi*y
    _c1 = (1/5)*t
    _c2 = torch.pi*x
    _c3 = torch.sin(2*_c1 + 2*_c2)*torch.cos(_c0 - 3/10*t)
    _c4 = (1/5)*_c3 + 1
    _c5 = 2*_c2
    _c6 = torch.sin(_c0 + _c1)
    _c7 = _c6*torch.cos(_c5) + 5
    _c8 = (1/10)*_c4
    _c9 = -torch.sin(_c5 - 1/10*t)*torch.cos(_c0 + (1/4)*t) + 4
    return _c4, _c7*_c8, -_c8*_c9, (5/8)*_c6*torch.cos(_c5 + (3/20)*t) + (1/1000)*(_c3 + 5)*(_c7**2 + _c9**2) + 5/2


def mms_source(x, y, t, mu, kappa):
    """Return the conserved-variable source ``(S_rho, S_rhou, S_rhov, S_E)``."""
    _c0 = torch.pi*x
    _c1 = 2*_c0
    _c2 = torch.cos(_c1)
    _c3 = (1/5)*t
    _c4 = 2*torch.pi
    _c5 = _c4*y
    _c6 = _c3 + _c5
    _c7 = torch.sin(_c6)
    _c8 = _c2*_c7
    _c9 = _c8 + 5
    _c10 = 2*_c0 + 2*_c3
    _c11 = torch.cos(_c10)
    _c12 = -_c5 + (3/10)*t
    _c13 = torch.cos(_c12)
    _c14 = torch.sin(_c10)
    _c15 = _c13*_c14
    _c16 = _c15 + 5
    _c17 = _c5 + (1/4)*t
    _c18 = torch.sin(_c17)
    _c19 = -_c1 + (1/10)*t
    _c20 = torch.sin(_c19)
    _c21 = torch.cos(_c17)
    _c22 = _c20*_c21
    _c23 = _c22 + 4
    _c24 = torch.sin(_c12)
    _c25 = _c14*_c24
    _c26 = torch.sin(_c1)
    _c27 = _c26*_c7
    _c28 = _c11*_c13
    _c29 = 3*_c25 - 4*_c28
    _c30 = _c1 + (3/20)*t
    _c31 = _c7*torch.sin(_c30)
    _c32 = torch.cos(_c6)
    _c33 = _c2*_c32
    _c34 = _c16*_c33
    _c35 = torch.cos(_c19)
    _c36 = _c18*_c35
    _c37 = _c36 + _c8
    _c38 = torch.pi**2
    _c39 = _c38*mu
    _c40 = _c9**2
    _c41 = 6*torch.pi
    _c42 = _c27*_c9
    _c43 = 12*torch.pi
    _c44 = _c16*_c43
    _c45 = 6*_c23
    _c46 = torch.pi*_c45
    _c47 = _c18*_c20
    _c48 = _c21*_c35
    _c49 = 5*_c47 - 2*_c48
    _c50 = _c26*_c32
    _c51 = _c39*(-_c22 + _c50)
    _c52 = torch.cos(_c30)
    _c53 = _c32*_c52
    _c54 = _c23**2
    _c55 = _c23*_c47
    _c56 = _c40 + _c54
    _c57 = 2*_c16
    _c58 = (1/5000)*torch.pi
    _c59 = _c52*_c7
    _c60 = _c16**(-1.0)
    _c61 = _c59 + 4
    _c62 = _c16**(-2.0)
    _c63 = 2*_c60
    _c64 = 5*_c38*_c60*kappa
    _c65 = _c16*_c56 + 875*_c59 + 3500
    _c66 = (2/75)*_c39
    _c67 = (1/25)*_c39*(_c33 + _c48)
    return (1/25)*torch.pi*_c11*_c13*_c9 + (1/25)*torch.pi*_c16*_c18*_c20 - 1/50*_c16*_c27*_c4 - 1/50*_c23*_c25*_c4 - 1/50*_c29, (1/1500)*_c16*_c41*_c47*_c9 - 1/1500*_c25*_c46*_c9 + (1/1500)*_c28*_c40*_c41 - 1/500*_c29*_c9 - 1/2*torch.pi*_c31 - 1/1500*_c34*_c46 + (1/250)*_c34 + (2/15)*_c37*_c39 + (4/5)*_c39*_c8 - 1/1500*_c42*_c44, (1/1000)*_c16*_c49 - 1/125*torch.pi*_c16*_c55 - 4/5*_c22*_c39 + (1/3000)*_c23*_c27*_c44 - 1/3000*_c23*_c28*_c43*_c9 + (1/3000)*_c25*_c43*_c54 + (1/3000)*_c29*_c45 + (1/3000)*_c44*_c48*_c9 + (2/15)*_c51 + (1/2)*torch.pi*_c53, (1/10000)*_c16*(4*_c2*_c32*_c9 - _c23*_c49) + (2/75)*_c18*_c20*_c38*mu*(-_c27 - 2*_c47) + (1/5000)*torch.pi*_c18*_c20*_c65 - 1/25*_c23*_c51 - _c23*_c58*(_c25*_c56 + 875*_c53 + _c57*(_c2*_c32*_c9 - _c55)) - _c23*_c66*(-2*_c22 - _c50) - _c27*_c58*_c65 - _c27*_c66*(2*_c27 + _c47) - 1/10000*_c29*_c56 - 3/32*_c31 + (1/8)*_c32*_c52 - _c33*_c67 + (1/25)*_c37*_c38*_c9*mu + (2/75)*_c38*_c9*mu*(-_c36 + 2*_c8) - _c48*_c67 - _c64*(2*_c11**2*_c13**2*_c61*_c62 + _c15*_c60*_c61 + _c28*_c31*_c63 - _c59) - _c64*(_c13*_c14*_c60*_c61 + 2*_c14**2*_c24**2*_c61*_c62 - _c25*_c53*_c63 - _c59) + (1/5000)*torch.pi*_c9*(_c11*_c13*_c56 - 875*_c31 - _c57*(_c23*_c48 + _c42))
