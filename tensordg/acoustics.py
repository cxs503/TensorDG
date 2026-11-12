"""Exact solution of the two-dimensional linear acoustic pulse (axisymmetric).

The case is the classical *acoustic pulse* verification problem for
computational aeroacoustics (Tam & Webb 1993, J. Comput. Phys. 107:262;
Tam & Dong 1996, J. Comput. Phys. 125:215).  A still, uniform fluid
``rho0, p0`` (sound speed ``c0 = sqrt(gamma p0 / rho0)``) carries an
axisymmetric Gaussian perturbation of the pressure

    p'(r, 0) = A exp(-alpha r^2),   alpha = ln 2 / b^2,   r = sqrt(x^2 + y^2)

with zero initial velocity and the *isentropic* density perturbation
``rho'(r,0) = p'(r,0) / c0^2`` (so the initial data contains no entropy mode).

Why this is an exact solution
-----------------------------
The linearised Euler equations with a velocity potential
``v = grad(phi)``, ``p' = -rho0 dphi/dt`` reduce to the wave equation
``d2phi/dt2 = c0^2 lap(phi)``.  Hankel transforming (order 0) gives, for each
radial wavenumber ``s``,

    Phi(s,t) = -(F(s) / (rho0 c0 s)) sin(c0 s t),   F(s) = Hankel0[p'(.,0)],

with ``F(s) = (1/(2 alpha)) exp(-s^2/(4 alpha))`` for the Gaussian, so

    p'(r,t) = (1/(2 alpha)) Int_0^inf exp(-s^2/(4 alpha)) cos(c0 s t) J0(s r) s ds,

and ``rho'(r,t) = p'(r,t)/c0^2``, ``u_r(r,t)`` radial.  At ``t = 0`` the
inverse Hankel transform returns the Gaussian, which is the identity this
module is unit-tested against.

The 2-D geometric spreading gives a closed-form on-axis asymptote used here as
an independent anchor (no quadrature involved).  Two integrations by parts of
the Hankel integral (the endpoint term at ``s=0`` of
``Int s exp(-a s^2) exp(i c0 s t) ds`` is the dominant contributor) give

    p'(0,t) ~ -A b^2 / (2 ln2 c0^2 t^2) = -A / (2 alpha c0^2 t^2),

i.e. the on-axis signal is a *rarefaction* wake decaying like ``t^-2`` for
``c0 t / b >> 1``.  The sign and the exponent are both checked against the
quadrature in ``tests/test_acoustic_pulse.py``.

Everything is numpy/scipy only, so the exact solution is available without the
solver and its cost is independent of the DG mesh.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import j0

__all__ = [
    "alpha_from_width",
    "acoustic_pulse_exact_pressure",
    "acoustic_pulse_on_axis_asymptote",
    "acoustic_pulse_initial_state",
    "acoustic_pulse_sound_speed",
]


def alpha_from_width(width: float) -> float:
    """Gaussian exponent ``alpha = ln 2 / b^2`` for the half-width ``b``."""
    if not width > 0.0:
        raise ValueError("width must be positive")
    return math.log(2.0) / (width * width)


def acoustic_pulse_sound_speed(gamma: float = 1.4, rho0: float = 1.0, p0: float = 1.0) -> float:
    """Sound speed of the uniform background state."""
    if not (gamma > 1.0 and rho0 > 0.0 and p0 > 0.0):
        raise ValueError("invalid background state")
    return math.sqrt(gamma * p0 / rho0)


@lru_cache(maxsize=64)
def _quadrature_rule(panels: int, order: int, decay: float) -> tuple[np.ndarray, np.ndarray]:
    """Cached composite Gauss-Legendre rule on ``[0, S]``, ``S^2 = 4 alpha decay``.

    A composite rule is used because the integrand oscillates (``cos(c0 s t)
    J0(s r)``) while the Gaussian weight cuts it off at ``S``: a single
    high-order rule needs an O(10^3) eigenvalue solve per call, a composite
    24-point rule per panel is both cheap and spectrally accurate for the
    O(10) oscillations of interest.  Caching keeps the exact solution bit-wise
    reproducible without re-solving for the nodes.
    """
    if panels < 8:
        raise ValueError("need at least 8 panels")
    if order < 8:
        raise ValueError("need at least an 8-point panel rule")
    if not decay > 4.0:
        raise ValueError("decay must exceed 4 so the Gaussian tail is negligible")
    s_max = 2.0 * math.sqrt(decay)
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(0.0, s_max, panels + 1)
    half = 0.5 * np.diff(edges)
    nodes = (0.5 * (edges[:-1] + edges[1:])[:, None] + half[:, None] * x[None, :]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    return nodes, weights


def acoustic_pulse_exact_pressure(
    radius,
    time: float,
    *,
    width: float = 0.4,
    amplitude: float = 1.0,
    sound_speed: float = 1.0,
    panels: int = 240,
    nodes: int = 24,
    decay: float = 40.0,
) -> np.ndarray:
    """Exact pressure perturbation ``p'(r,t)`` of the Gaussian acoustic pulse.

    Parameters
    ----------
    radius : array_like
        Radial distance from the pulse centre (any shape).
    time : float
        Physical time (``t = 0`` reproduces the initial Gaussian).
    width : float
        Gaussian half-width ``b`` in ``exp(-ln2 (r/b)^2)``.
    amplitude : float
        Peak initial pressure perturbation ``A``.
    sound_speed : float
        Background sound speed ``c0``.
    panels, nodes : int, int
        Composite Gauss-Legendre panels on ``[0, 2 sqrt(alpha decay)]`` and
        points per panel; defaults agree with a 2x refined rule to <1e-11.
    decay : float
        Gaussian-tail cut-off (``exp(-decay)``), default 40.
    """
    r = np.asarray(radius, dtype=float)
    if not np.all(np.isfinite(r)) or np.any(r < 0.0):
        raise ValueError("radius must be finite and non-negative")
    if not time >= 0.0:
        raise ValueError("time must be non-negative")
    alpha = alpha_from_width(width)
    s, w = _quadrature_rule(panels, nodes, decay)
    # Map the alpha = 1 rule to the actual Gaussian: s -> sqrt(alpha) s.
    scale = math.sqrt(alpha)
    s = s * scale
    w = w * scale
    weight = np.exp(-(s * s) / (4.0 * alpha)) * s * w
    # p'(r,t) = A/(2 alpha) Int exp(-s^2/(4 alpha)) cos(c0 s t) J0(s r) s ds
    phase = sound_speed * time
    out = np.empty_like(r, dtype=float)
    flat_r = r.reshape(-1)
    flat_out = out.reshape(-1)
    for i, rad in enumerate(flat_r):
        flat_out[i] = float(np.dot(weight * np.cos(phase * s), j0(rad * s)))
    return (amplitude / (2.0 * alpha)) * out


def acoustic_pulse_on_axis_asymptote(
    time,
    *,
    width: float = 0.4,
    amplitude: float = 1.0,
    sound_speed: float = 1.0,
) -> np.ndarray:
    """Closed-form large-time on-axis asymptote ``-A b^2 / (2 ln2 c0^2 t^2)``."""
    t = np.asarray(time, dtype=float)
    if np.any(t <= 0.0):
        raise ValueError("time must be positive")
    alpha = alpha_from_width(width)
    return -amplitude / (2.0 * alpha * sound_speed * sound_speed * t * t)


def acoustic_pulse_initial_state(
    x,
    y,
    *,
    width: float = 0.4,
    amplitude: float = 1.0,
    gamma: float = 1.4,
    rho0: float = 1.0,
    p0: float = 1.0,
):
    """Return ``(rho, u, v, p)`` for the initial pulse (velocity-free).

    The density perturbation is the isentropic companion ``p'/c0^2`` of the
    pressure pulse, so no entropy (contact) mode is excited and pressure and
    density obey the *same* normalised exact profile.
    """
    c0 = acoustic_pulse_sound_speed(gamma, rho0, p0)
    alpha = alpha_from_width(width)

    def _exp(r2):
        if hasattr(r2, "requires_grad") or type(r2).__module__.startswith("torch"):
            import torch

            return torch.exp(-alpha * r2)
        return np.exp(-alpha * r2)

    pulse = amplitude * _exp(x * x + y * y)
    rho = rho0 + pulse / (c0 * c0)
    pressure = p0 + pulse
    if hasattr(pulse, "requires_grad") or type(pulse).__module__.startswith("torch"):
        import torch

        zeros = torch.zeros_like(pulse)
    else:
        zeros = np.zeros_like(pulse)
    return rho, zeros, zeros, pressure