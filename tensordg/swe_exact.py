"""Exact solutions of the 1-D shallow water Riemann problem (dam break).

Both reference solutions below are for a **flat bed** with still water
``u = 0`` on either side of ``x = 0`` and gravity ``g``::

    h(x, 0) = h_L for x < 0,   h(x, 0) = h_R for x > 0,
    (hu)(x, 0) = 0.

The shallow water equations are invariant under the similarity scaling
``x -> lam x, t -> lam t``, so the solution depends only on ``xi = x / t``
and can be written in closed form.

Ritter (1892) -- dry downstream bed, ``h_R = 0``
    A single left rarefaction fan, ``u + 2c = 2 c_1``::

        -c_1 <= xi < 2 c_1 :  c = (2 c_1 - xi) / 3,  u = (2 xi + 2 c_1) / 3
        xi >= 2 c_1       :  dry (h = 0, u = 0)

Stoker (1957) -- wet downstream bed, ``0 < h_R < h_L``
    Left rarefaction fan + constant star state + right shock::

        xi <= -c_1        :  h = h_L, u = 0
        -c_1 <= xi < xi_* :  c = (2 c_1 - xi)/3, u = (2 xi + 2 c_1)/3
        xi_* <= xi <= S   :  h = h_*, u = u_*
        xi > S            :  h = h_R, u = 0

    with ``xi_* = u_* - c_*`` and the star state obtained from the jump
    conditions across the right shock, solved from

        u_* = 2 (c_1 - c_*)
        u_* = (h_* - h_R) sqrt(g (h_* + h_R) / (2 h_* h_R))

    (the two-curve intersection of the rarefaction and shock relations), and
    the shock speed ``S = h_* u_* / (h_* - h_R)`` from mass conservation
    across the shock.

References
----------
* Ritter, A. (1892). "Die Fortpflanzung der Wasserwellen."
  Z. Ver. Deutsch. Ing. 36, 947-954.
* Stoker, J. J. (1957). *Water Waves*, Interscience, New York, Sec. 10.4.
* Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid
  Dynamics*, 3rd ed., Springer, Sec. 13.4.2 (exact shallow water solver).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Star state (Stoker / wet-bed dam break)
# ---------------------------------------------------------------------------

def stoker_star_state(h_left: float, h_right: float,
                      gravity: float = GRAVITY) -> Dict[str, float]:
    """Exact star state of the wet-bed dam break.

    Solves the algebraic Riemann problem for still water on both sides by
    intersecting the left rarefaction curve ``u = 2 (c_1 - c)`` with the
    right shock curve ``u = (h - h_R) sqrt(g (h + h_R) / (2 h h_R))``.

    Parameters
    ----------
    h_left, h_right : water depths on the left / right of the discontinuity
    gravity : gravitational acceleration

    Returns
    -------
    dict with keys ``h_star``, ``u_star``, ``c_star``, ``shock_speed``,
    ``wave_head``, ``wave_tail``, ``depth_ratio``.
    """
    if h_left <= 0.0:
        raise ValueError("h_left must be positive")
    if h_right < 0.0:
        raise ValueError("h_right must be non-negative")
    if h_right > h_left:
        raise ValueError("this reference solution assumes h_left >= h_right")

    g = gravity
    c1 = math.sqrt(g * h_left)

    if h_right == 0.0:  # Ritter dry-bed limit
        h_star, u_star, c_star = 0.0, 2.0 * c1, 0.0
        shock = 2.0 * c1  # dry front travels with the tail of the fan
    else:
        c2 = math.sqrt(g * h_right)

        def residual(h: float) -> float:
            c = math.sqrt(g * h)
            return 2.0 * (c1 - c) - (h - h_right) * math.sqrt(
                g * (h + h_right) / (2.0 * h * h_right))

        lo, hi = h_right, h_left
        # residual(h_right) = 2 (c1 - c2) > 0, residual(h_left) < 0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if residual(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        h_star = 0.5 * (lo + hi)
        c_star = math.sqrt(g * h_star)
        u_star = 2.0 * (c1 - c_star)
        shock = h_star * u_star / (h_star - h_right)
        _ = c2

    return {
        "h_star": h_star,
        "u_star": u_star,
        "c_star": c_star,
        "shock_speed": shock,
        "wave_head": -c1,
        "wave_tail": u_star - c_star,
        "depth_ratio": h_right / h_left,
    }


def _vertical_state(xi, h_left: float, h_right: float, gravity: float,
                    star: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the self-similar solution at ``xi = x / t`` (arrays)."""
    g = gravity
    c1 = math.sqrt(g * h_left)
    xi = np.asarray(xi, dtype=float)

    h = np.empty_like(xi)
    u = np.empty_like(xi)

    head = star["wave_head"]
    tail = star["wave_tail"]
    shock = star["shock_speed"]
    h_star = star["h_star"]
    u_star = star["u_star"]

    left = xi <= head
    fan = (xi > head) & (xi < tail)
    star_region = (xi >= tail) & (xi <= shock)
    right = xi > shock
    if h_right == 0.0:
        # no shock: the fan runs straight into the dry bed
        fan = (xi > head) & (xi < shock)
        star_region = np.zeros_like(left)

    h[left] = h_left
    u[left] = 0.0

    xi_fan = xi[fan]
    c_fan = (2.0 * c1 - xi_fan) / 3.0
    h[fan] = c_fan ** 2 / g
    u[fan] = (2.0 * xi_fan + 2.0 * c1) / 3.0

    h[star_region] = h_star
    u[star_region] = u_star

    h[right] = h_right
    u[right] = 0.0
    return h, u


def stoker_exact(x, t: float, h_left: float, h_right: float,
                 gravity: float = GRAVITY) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point-wise Stoker solution. Returns ``(h, u, eta)`` (flat bed, eta = h)."""
    if t <= 0.0:
        raise ValueError("t must be positive")
    star = stoker_star_state(h_left, h_right, gravity)
    h, u = _vertical_state(np.asarray(x, dtype=float) / t, h_left, h_right,
                           gravity, star)
    return h, u, h


def ritter_exact(x, t: float, h_left: float,
                 gravity: float = GRAVITY) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point-wise Ritter (dry-bed) solution. Returns ``(h, u, eta)``."""
    return stoker_exact(x, t, h_left, 0.0, gravity)


# ---------------------------------------------------------------------------
# Cell averages (the natural quantity to compare against a DG solution)
# ---------------------------------------------------------------------------

def dam_break_cell_averages(faces, t: float, h_left: float, h_right: float,
                            gravity: float = GRAVITY,
                            n_gauss: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Exact cell averages of ``(h, hu)`` at time ``t``.

    The cell is split at the (at most two) similarity breakpoints
    ``x = t * xi_bp`` and each smooth piece is integrated with Gauss-Legendre
    quadrature. On a smooth piece ``h`` is at most quadratic in ``xi`` and
    ``hu`` at most quartic, so 6 points are exact up to round-off.

    Parameters
    ----------
    faces : array of cell faces, shape (ncells + 1,)
    """
    faces = np.asarray(faces, dtype=float)
    star = stoker_star_state(h_left, h_right, gravity)
    g = gravity

    bps = [star["wave_head"], star["wave_tail"], star["shock_speed"]]
    if h_right == 0.0:
        bps = [star["wave_head"], star["shock_speed"]]
    bp_x = sorted(t * b for b in bps)

    nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
    nodes = 0.5 * (nodes + 1.0)     # map to [0, 1]
    weights = 0.5 * weights

    h_bar = np.empty(len(faces) - 1)
    hu_bar = np.empty(len(faces) - 1)
    for i in range(len(faces) - 1):
        xl, xr = faces[i], faces[i + 1]
        cuts = [xl] + [b for b in bp_x if xl < b < xr] + [xr]
        m0 = 0.0
        m1 = 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            xs = a + (b - a) * nodes
            h, u = _vertical_state(xs / t, h_left, h_right, g, star)
            m0 += (b - a) * float(np.dot(weights, h))
            m1 += (b - a) * float(np.dot(weights, h * u))
        width = xr - xl
        h_bar[i] = m0 / width
        hu_bar[i] = m1 / width
    return h_bar, hu_bar