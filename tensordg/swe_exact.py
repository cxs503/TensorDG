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

# ---------------------------------------------------------------------------
# Thacker (1981) planar-surface sloshing in a parabolic basin (non-flat bed)
# ---------------------------------------------------------------------------
#
# A *nonlinear* exact solution of the shallow water equations over the parabola
#
#     z_b(x) = -h0 (1 - (x / L)^2),        x in [-L, L],
#
# whose free surface stays **planar**:  eta(x, t) = eta_c(t) + s(t) x.
# With a spatially uniform velocity u(t) the convective term vanishes and the
# momentum equation reduces to  u_t = -g s, while continuity splits into the
# two scalar relations
#
#     s_t   = 2 h0 u / L^2,          eta_c,t = -u s.
#
# Solving them with s(0) = A / L (i.e. eta(x, 0) = h0 + A x / L, still water)
# gives the closed form
#
#     omega = sqrt(2 g h0) / L,
#     s(t)  = (A / L) cos(omega t),
#     u(t)  = -(A omega L / (2 h0)) sin(omega t),
#     eta_c(t) = h0 + A^2 / (4 h0) sin^2(omega t),
#     h = eta - z_b.
#
# The three scalar identities above are what ``tests/test_thacker_benchmark.py``
# checks symbolically (they imply both PDE residuals vanish identically).
#
# Note: the free surface tilt A in this family is a *slope times length* and the
# velocity amplitude is A omega L / (2 h0) -- not A omega / g.  Because the
# basin rim z_b(+-L) = 0 carries water of depth h0 +- A, the domain behaves as a
# basin with open ends: the exact solution has a non-zero mass flux through
# x = +-L, so a verification run must impose the analytic boundary flux
# (``solve_swe_1d(..., bc_left_flux=..., bc_right_flux=...)``); the default
# transmissive closure reflects the outgoing wave back into the domain.
#
# References
# ----------
# * Thacker, W. C. (1981). "Some exact solutions of the nonlinear shallow water
#   equations." Comm. Pure Appl. Math. 34(4), 543-552.
# * Bermudez, A., Vazquez, M. E. (1994). "Upwind methods for hyperbolic
#   conservation laws with source terms." Comput. Fluids 23(8), 1049-1071.


def _thacker_params(h0: float, half_width: float, gravity: float):
    if h0 <= 0.0 or half_width <= 0.0:
        raise ValueError("h0 and half_width must be positive")
    return math.sqrt(2.0 * gravity * h0) / half_width


def thacker_parabolic_bed(x, h0: float = 1.0, half_width: float = 1.0):
    """Bed elevation ``z_b(x) = -h0 (1 - (x/L)^2)`` (zero at the rim ``x=+-L``)."""
    x = np.asarray(x, dtype=float)
    return -h0 * (1.0 - (x / half_width) ** 2)


def thacker_parabolic_bed_slope(x, h0: float = 1.0, half_width: float = 1.0):
    """Analytic bed slope ``dz_b/dx = 2 h0 x / L^2`` of :func:`thacker_parabolic_bed`.

    Supplying this to ``solve_swe_1d(..., zb_slope=...)`` makes the bed-slope
    source term consistent at quadrature points instead of using the
    cell-constant face difference ``(z_b,R - z_b,L)/dx``.  The latter is only
    the *cell average* of the slope, which for a curved bed leaves an
    O(dx) error in the discrete force (see
    ``docs/tensordg-swe-bed-slope-source-fix.md``).
    """
    x = np.asarray(x, dtype=float)
    return 2.0 * h0 * x / half_width ** 2


def thacker_planar_basin(x, t: float, h0: float = 1.0, amplitude: float = 0.5,
                         half_width: float = 1.0, gravity: float = GRAVITY,
                         period: float = 0.0):
    """Exact planar-surface sloshing solution. Returns ``(h, u, eta)``.

    ``amplitude`` is the initial surface tilt ``eta(x,0) = h0 + A x / L``.
    ``period`` may be passed to enforce exact periodicity of ``cos/sin`` for
    long runs (it only changes round-off level behaviour).
    """
    x = np.asarray(x, dtype=float)
    omega = _thacker_params(h0, half_width, gravity)
    if period:
        t = t % period
    s = (amplitude / half_width) * math.cos(omega * t)
    u = -(amplitude * omega * half_width / (2.0 * h0)) * math.sin(omega * t)
    eta_c = h0 + (amplitude ** 2 / (4.0 * h0)) * math.sin(omega * t) ** 2
    eta = eta_c + s * x
    h = eta - thacker_parabolic_bed(x, h0, half_width)
    return h, np.full_like(x, u), eta


def thacker_planar_basin_initial(x, h0: float = 1.0, amplitude: float = 0.5,
                                 half_width: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Initial condition ``(h, u)`` of the planar Thacker solution."""
    h, u, _ = thacker_planar_basin(x, 0.0, h0, amplitude, half_width)
    return h, u


def thacker_planar_basin_period(h0: float = 1.0, half_width: float = 1.0,
                                gravity: float = GRAVITY) -> float:
    """Exact oscillation period ``T = 2 pi L / sqrt(2 g h0)``."""
    return 2.0 * math.pi / _thacker_params(h0, half_width, gravity)


def thacker_planar_basin_boundary_flux(t: float, h0: float = 1.0,
                                       amplitude: float = 0.5, half_width: float = 1.0,
                                       gravity: float = GRAVITY) -> Tuple[np.ndarray, np.ndarray]:
    """Exact physical flux ``(hu, hu^2 + g h^2 / 2)`` at ``x = -L`` and ``x = +L``.

    The rim sits at ``z_b(+-L) = 0``, so ``h(+-L) = eta(+-L)`` and the flux is
    known in closed form. Use it as exact boundary data for a verification run.
    """
    omega = _thacker_params(h0, half_width, gravity)
    s = (amplitude / half_width) * math.cos(omega * t)
    u = -(amplitude * omega * half_width / (2.0 * h0)) * math.sin(omega * t)
    eta_c = h0 + (amplitude ** 2 / (4.0 * h0)) * math.sin(omega * t) ** 2

    def _flux(h: float) -> np.ndarray:
        return np.array([h * u, h * u * u + 0.5 * gravity * h * h])

    return _flux(eta_c - s * half_width), _flux(eta_c + s * half_width)


# ---------------------------------------------------------------------------
# 2-D axisymmetric (radial) Thacker basin
# ---------------------------------------------------------------------------
# ``docs/validation_status_report.md`` (m6315) listed "extend Thacker to 2-D
# (circular parabolic basin)" as an open action, and ``swe_2d`` could be run on
# a bowl but had no analytic reference to compare against.  The exact solution
# below is the circular-basin member of Thacker's (1981) family: the surface
# stays a paraboloid of revolution and the velocity stays linear in ``r``, so
# the 2-D system reduces to a closed ODE system that is solvable in closed
# form.  With ``omega = sqrt(8 g h0) / L`` and ``A in (0, 1)``:
#
#     H(t) = h0 sqrt(1 - A^2) / (1 - A cos(omega t))      (depth at r = 0)
#     B(t) = -H^2 / (h0 L^2)                              (h = H + B r^2)
#     eta(r,t) = (H - h0) + (B + h0/L^2) r^2              (free surface)
#     u_r(r,t) =  A omega r sin(omega t) / (2 (1 - A cos(omega t)))
#
# Its moving shoreline is ``r_s(t) = L sqrt(h0 / H(t))``, and the enclosed
# volume is exactly ``pi h0 L^2 / 2`` for every ``t`` (independent of ``H``),
# which is the mass invariant a numerical run must reproduce.
#
# ``docs/reports/tensordg-thacker-radial-2d-basin.md`` certifies the algebra two
# independent ways: the residual of the radial shallow-water equations under
# high-order finite differences, and agreement with a DOP853 integration of the
# reduced ODE system.

def thacker_radial_params(h0: float = 1.0, half_width: float = 1.0,
                          gravity: float = GRAVITY) -> float:
    """Angular frequency ``omega = sqrt(8 g h0) / L`` of the circular basin.

    The factor 8 (not 2, as in the 1-D planar basin) is what the linearised
    radial problem gives: ``L`` is the radius at which the paraboloid bed
    ``z_b = -h0 (1 - r^2/L^2)`` reaches the still-water level.
    """
    if h0 <= 0.0 or half_width <= 0.0:
        raise ValueError("h0 and half_width must be positive")
    return math.sqrt(8.0 * gravity * h0) / half_width


def thacker_radial_period(h0: float = 1.0, half_width: float = 1.0,
                          gravity: float = GRAVITY) -> float:
    """Exact oscillation period ``T = 2 pi L / sqrt(8 g h0)``."""
    return 2.0 * math.pi / thacker_radial_params(h0, half_width, gravity)


def thacker_radial_bed(r, h0: float = 1.0, half_width: float = 1.0):
    """Bed elevation ``z_b(r) = -h0 (1 - (r/L)^2)`` of the paraboloid bowl.

    The bowl is *not* truncated at ``r = L``: the exact solution needs the
    paraboloid continued outward, because its shoreline radius
    ``r_s = L sqrt(h0/H)`` exceeds ``L`` for part of every cycle.
    """
    r = np.asarray(r, dtype=float)
    return -h0 * (1.0 - (r / half_width) ** 2)


def thacker_radial_bed_slope(r, h0: float = 1.0, half_width: float = 1.0):
    """Radial bed slope ``dz_b/dr = 2 h0 r / L^2``."""
    r = np.asarray(r, dtype=float)
    return 2.0 * h0 * r / half_width ** 2


def _thacker_radial_H(t, h0, amplitude, half_width, gravity):
    omega = thacker_radial_params(h0, half_width, gravity)
    H = h0 * math.sqrt(1.0 - amplitude ** 2) / (1.0 - amplitude * math.cos(omega * t))
    return H, omega


def thacker_radial_shoreline(t, h0: float = 1.0, amplitude: float = 0.2,
                             half_width: float = 1.0, gravity: float = GRAVITY):
    """Radius ``r_s(t) = L sqrt(h0 / H(t))`` where the exact depth vanishes."""
    H, _ = _thacker_radial_H(t, h0, amplitude, half_width, gravity)
    return half_width * math.sqrt(h0 / H)


def thacker_radial_basin(r, t: float, h0: float = 1.0, amplitude: float = 0.2,
                         half_width: float = 1.0, gravity: float = GRAVITY,
                         clip: bool = True):
    """Exact circular-basin sloshing solution. Returns ``(h, u_r, eta)``.

    Parameters
    ----------
    r : array_like
        Radius from the bowl centre (``r >= 0``).
    t : float
        Time.
    amplitude : float
        ``A in (0, 1)``; the initial surface is ``eta(r,0) = (H0 - h0) +
        (B0 + h0/L^2) r^2`` with zero velocity (an impulsive start).
    clip : bool, default True
        Clip the depth at zero outside the instantaneous shoreline.  Set
        ``clip=False`` to obtain the analytic paraboloid everywhere, which is
        what the PDE-residual certificate differentiates.
    """
    if not 0.0 < amplitude < 1.0:
        raise ValueError("amplitude must lie in (0, 1)")
    H, omega = _thacker_radial_H(t, h0, amplitude, half_width, gravity)
    r = np.asarray(r, dtype=float)
    B = -H ** 2 / (h0 * half_width ** 2)
    h = H + B * r ** 2
    eta = (H - h0) + (B + h0 / half_width ** 2) * r ** 2
    s = 1.0 - amplitude * math.cos(omega * t)
    u_r = amplitude * omega * r * math.sin(omega * t) / (2.0 * s)
    if clip:
        h = np.maximum(h, 0.0)
    return h, u_r, eta


def thacker_radial_basin_initial(r, h0: float = 1.0, amplitude: float = 0.2,
                                 half_width: float = 1.0):
    """Initial condition ``(h, u_r)`` of the radial Thacker solution (``u_r = 0``)."""
    h, u_r, _ = thacker_radial_basin(r, 0.0, h0, amplitude, half_width)
    return h, u_r


def thacker_radial_velocity(x, y, t: float, h0: float = 1.0,
                            amplitude: float = 0.2, half_width: float = 1.0,
                            gravity: float = GRAVITY):
    """Cartesian velocity ``(u, v)`` of the radial solution at ``(x, y, t)``.

    ``u_r`` is linear in ``r``, so ``u = f(t) x`` and ``v = f(t) y`` with
    ``f = A omega sin(omega t) / (2 (1 - A cos(omega t)))``.  This is the
    field a 2-D Cartesian DG run has to initialise with.
    """
    H, omega = _thacker_radial_H(t, h0, amplitude, half_width, gravity)
    s = 1.0 - amplitude * math.cos(omega * t)
    f = amplitude * omega * math.sin(omega * t) / (2.0 * s)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return f * x, f * y
