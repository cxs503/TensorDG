"""Three-shock (von Neumann) theory reference for the double Mach reflection.

Why this module exists
----------------------
``docs/CASE-REPRO-DG-CKS1998-DOUBLE-MACH.md`` §5-1 states the anchor's remaining
honest gap in so many words:

    「三叉点轨迹、马赫杆高度、滑移线涡串这些文献里最常被目视比较的量，本仓**没有**
      可用的解析/半解析参考（三波理论需要解激波极曲线交点），因此**没有**给出数值判据」

This module supplies exactly that missing reference: the **pseudo-steady
three-shock theory** (von Neumann 1943; standard form in Ben-Dor, *Shock Wave
Reflection Phenomena*) for the Woodward--Colella / Cockburn--Shu double Mach
reflection (M_s = 10, incident front at 60 deg to the wall).  It turns the
"looks like the textbook picture" comparison into a *number*:

* ``chi`` -- the angle of the straight triple-point trajectory from the wall,
  so the predicted triple-point position at time ``t`` is
  ``(x0 + q cos(chi) t, q sin(chi) t)`` with ``x0 = 1/6`` (where the incident
  front meets the wall at ``t = 0``) and ``q`` fixed by the stationarity of the
  incident shock in the triple-point frame;
* the Mach-stem angle ``psi`` (the stem is *not* put in by hand: it is an
  unknown solved together with ``chi``) and the reflected-shock angle;
* the three downstream states, each of which satisfies the Rankine--Hugoniot
  conditions to machine precision (checked by :func:`rh_flux_residuals`, which
  is written in terms of conserved fluxes, i.e. independently of the closed-form
  jump relations used to build the states).

Geometry and frames (all angles in the x--y plane, wall at ``y = 0``)
---------------------------------------------------------------------
The incident front is the straight line ``y = sqrt(3) (x - 1/6)``, so it makes
``omega = 60 deg`` with the wall; its normal makes 60 deg with the wall normal.
It moves along ``n_i = (sin omega, -cos omega)`` at speed ``W = M_s a_1 = 10``
(``a_1 = 1`` for ``rho = 1.4, p = 1``).  Post-shock gas is up/left of the front,
undisturbed air down/right.  The foot of the front reaches the wall at
``(1/6, 0)`` at ``t = 0``; that point is where the reflection -- and therefore
the triple point -- starts.

In the frame moving with the triple point (unknown velocity
``q_vec = q (cos chi, sin chi)``) the whole pattern is stationary, so each of the
three waves is a *steady* oblique shock.  Stationarity of the incident shock
gives one relation, ``q sin(omega - chi) = W``, hence ``q = W / sin(omega - chi)``.

Unknowns and conditions (2 x 2, exactly determined)
--------------------------------------------------
Unknowns: ``chi`` (trajectory angle) and the reflected-shock normal angle
``phi_r``.  The Mach-stem normal is *not* an unknown in the primary
formulation: it is fixed to ``(-1, 0)`` (stem plane perpendicular to the wall,
the classical closure) -- note the normal is horizontal because the stem is
vertical.  Conditions:

1. equal pressure across the slip line: ``p_3 = p_3'``;
2. the two post-wave velocities are parallel (both along the slip line, same
   sense): ``u_3 x u_3' = 0`` and ``u_3 . u_3' > 0``.

:func:`solve_three_shock_galilean` re-solves the same problem with the stem
angle **freed** and the closure "post-stem flow is parallel to the wall in the
*lab* frame" imposed instead (three unknowns, three conditions).  The two
formulations are related by a Galilean boost, so they must return the same
``chi``; the test suite asserts that agreement, which is the defence against a
self-consistent-but-wrong implementation.

Reproduce
---------
    PYTHONPATH=. python -m benchmarks.dmr_three_shock            # theory table
    PYTHONPATH=. python -m benchmarks.dmr_three_shock --polar    # + polar figure
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

#: ratio of specific heats (air)
GAMMA = 1.4
#: undisturbed air ahead of the incident shock, ``(rho, u, v, p)``
PRE_SHOCK = (1.4, 0.0, 0.0, 1.0)
#: incident shock Mach number of the Woodward--Colella / CKS1998 case
SHOCK_MACH = 10.0
#: angle between the incident front and the wall (deg)
WALL_ANGLE_DEG = 60.0
#: wall abscissa where the incident front stands at ``t = 0``
X_HINGE = 1.0 / 6.0
#: density level used to trace the captured front (midpoint of the jump)
FRONT_LEVEL = 0.5 * (PRE_SHOCK[0] + 8.0)


class NoThreeShockSolution(RuntimeError):
    """Raised when the polar intersection does not exist for the given data."""


@dataclass(frozen=True)
class State:
    """Primitive state ``(rho, u, v, p)``."""

    rho: float
    u: float
    v: float
    p: float

    def sound(self, gamma: float = GAMMA) -> float:
        return math.sqrt(gamma * self.p / self.rho)

    def speed(self) -> float:
        return math.hypot(self.u, self.v)

    def as_dict(self) -> Dict[str, float]:
        return {"rho": self.rho, "u": self.u, "v": self.v, "p": self.p}


PRE = State(*PRE_SHOCK)


# ---------------------------------------------------------------- steady shocks
def _unit(normal: Sequence[float]) -> Tuple[float, float]:
    nx, ny = float(normal[0]), float(normal[1])
    n = math.hypot(nx, ny)
    if n == 0.0:
        raise ValueError("degenerate normal")
    return nx / n, ny / n


def stationary_shock(up: State, normal: Sequence[float],
                     gamma: float = GAMMA) -> State:
    """Downstream state of a *steady* shock whose normal points downstream.

    ``normal`` must satisfy ``up.u·n > a_up``; otherwise the wave does not exist
    (the upstream flow is not supersonic normal to the shock) and ``ValueError``
    is raised -- callers scanning parameter space are expected to catch it.
    """
    nx, ny = _unit(normal)
    vn1 = up.u * nx + up.v * ny
    a1 = up.sound(gamma)
    if vn1 <= 0.0:
        raise ValueError(f"upstream flow points upstream (vn1={vn1:.6g})")
    mn1 = vn1 / a1
    if mn1 <= 1.0:
        raise ValueError(f"normal Mach {mn1:.6g} <= 1: no steady shock")
    ratio = (gamma + 1.0) * mn1 * mn1 / ((gamma - 1.0) * mn1 * mn1 + 2.0)
    rho2 = up.rho * ratio
    vn2 = vn1 / ratio
    p2 = up.p * (1.0 + 2.0 * gamma / (gamma + 1.0) * (mn1 * mn1 - 1.0))
    # tangential unit vector t = (-ny, nx); tangential velocity is preserved
    vt = -up.u * ny + up.v * nx
    u2 = vn2 * nx + vt * (-ny)
    v2 = vn2 * ny + vt * nx
    return State(rho2, u2, v2, p2)


def rh_flux_residuals(up: State, down: State, normal: Sequence[float],
                      gamma: float = GAMMA) -> Dict[str, float]:
    """Jump residuals in *conserved-flux* form (independent of the jump formulas).

    Returns ``{"mass", "normal_momentum", "tangential_momentum", "energy"}``,
    each the downstream-minus-upstream flux difference normalised by the
    magnitude of the upstream mass flux.  Vanishes to machine precision for an
    exact shock.
    """
    nx, ny = _unit(normal)
    tx, ty = -ny, nx

    def fluxes(s: State) -> Tuple[float, float, float, float]:
        vn = s.u * nx + s.v * ny
        vt = s.u * tx + s.v * ty
        e = s.p / (gamma - 1.0) / s.rho + 0.5 * (vn * vn + vt * vt)
        return (s.rho * vn,
                s.p + s.rho * vn * vn,
                s.rho * vn * vt,
                s.rho * vn * (e + s.p / s.rho))

    f_up = fluxes(up)
    f_dn = fluxes(down)
    scale = abs(f_up[0]) if abs(f_up[0]) > 0 else 1.0
    names = ("mass", "normal_momentum", "tangential_momentum", "energy")
    return {k: (f_dn[i] - f_up[i]) / scale for i, k in enumerate(names)}


def moving_shock(up: State, direction: Sequence[float], speed: float,
                 gamma: float = GAMMA) -> State:
    """Downstream state of a shock *propagating* along ``direction`` at ``speed``.

    ``up`` is given in the lab frame (typically at rest) and ``direction`` is the
    direction the shock travels into the undisturbed gas; the post-shock region
    is therefore on the ``-direction`` side.  Implemented by boosting into the
    shock frame and back, so it shares no algebra with the closed-form jump
    relations -- the test suite uses it to reproduce
    ``benchmarks.double_mach_reflection.POST_SHOCK`` exactly.
    """
    dx, dy = _unit(direction)
    boosted = State(up.rho, up.u - speed * dx, up.v - speed * dy, up.p)
    down = stationary_shock(boosted, (-dx, -dy), gamma)   # normal: upstream -> downstream
    return State(down.rho, down.u + speed * dx, down.v + speed * dy, down.p)


# ------------------------------------------------------- oblique-shock polars
def deflection_from_beta(mach: float, beta: float, gamma: float = GAMMA) -> float:
    """Deflection angle (rad) behind an oblique shock of angle ``beta`` (rad)."""
    s = math.sin(beta)
    num = 2.0 / math.tan(beta) * (mach * mach * s * s - 1.0)
    den = mach * mach * (gamma + math.cos(2.0 * beta)) + 2.0
    return math.atan(num / den)


def beta_from_deflection(mach: float, theta: float, gamma: float = GAMMA,
                         branch: str = "weak") -> float:
    """Shock angle ``beta`` (rad) for deflection ``theta`` (rad).

    ``branch`` is ``"weak"`` (subsonic downstream normal component) or
    ``"strong"``.  ``ValueError`` if ``theta`` exceeds the detachment angle.
    """
    if mach <= 1.0:
        raise ValueError("oblique shock requires upstream Mach > 1")
    if theta < 0.0:
        raise ValueError("negative deflection")
    if theta == 0.0:
        return math.pi / 2.0 if branch == "strong" else math.asin(1.0 / mach)

    def f(beta: float) -> float:
        return deflection_from_beta(mach, beta, gamma) - theta

    b_min = math.asin(1.0 / mach)
    b_max = math.pi / 2.0
    # deflection rises from 0 at b_min to a maximum at b_detach, then falls to 0
    lo, hi = b_min, b_max
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    b_detach = 0.5 * (lo + hi)
    if theta > deflection_from_beta(mach, b_detach, gamma) + 1e-12:
        raise ValueError(f"deflection {math.degrees(theta):.3f} deg exceeds "
                         f"detachment angle "
                         f"{math.degrees(deflection_from_beta(mach, b_detach, gamma)):.3f} deg")
    lo, hi = (b_min, b_detach) if branch == "weak" else (b_detach, b_max)
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0.0:
        # theta sits on the tangent point: return the detachment angle
        return b_detach
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0.0) == (f_lo > 0.0):
            lo, f_lo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def detachment_angle(mach: float, gamma: float = GAMMA) -> float:
    """Maximum deflection (rad) for which an attached oblique shock exists."""
    b_min = math.asin(1.0 / mach)
    lo, hi = b_min, math.pi / 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        # d(theta)/dbeta > 0 below the detachment angle
        if deflection_from_beta(mach, mid + 1e-9, gamma) > deflection_from_beta(mach, mid, gamma):
            lo = mid
        else:
            hi = mid
    return deflection_from_beta(mach, 0.5 * (lo + hi), gamma)


def shock_polar(mach: float, gamma: float = GAMMA,
                n: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Polar curve ``(deflection_deg, p2/p1)`` from weak to strong branch."""
    b_min = math.asin(1.0 / mach)
    betas = np.linspace(b_min + 1e-9, math.pi / 2.0 - 1e-9, n)
    theta = np.array([deflection_from_beta(mach, b, gamma) for b in betas])
    mn = mach * np.sin(betas)
    ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (mn * mn - 1.0)
    return np.degrees(theta), ratio


# --------------------------------------------------------- primary formulation
def _stem_normal_preferred() -> Tuple[float, float]:
    """Mach-stem normal: stem plane perpendicular to the wall -> horizontal."""
    return (-1.0, 0.0)


def _incident_normal(omega: float) -> Tuple[float, float]:
    """Incident-shock normal pointing downstream (up/left)."""
    return (-math.sin(omega), math.cos(omega))


def _frame_states(chi: float, omega: float, mach: float, gamma: float
                  ) -> Tuple[State, State, State, float]:
    """Triple-point-frame states ``(u1_frame, post_incident, post_stem, q)``."""
    a1 = PRE.sound(gamma)
    w = mach * a1
    denom = math.sin(omega - chi)
    if denom <= 1e-9:
        raise ValueError("chi too close to omega: incident shock not stationary")
    q = w / denom
    up = State(PRE.rho, -q * math.cos(chi), -q * math.sin(chi), PRE.p)
    post_inc = stationary_shock(up, _incident_normal(omega), gamma)
    post_stem = stationary_shock(up, _stem_normal_preferred(), gamma)
    return up, post_inc, post_stem, q


def _cross(a: State, b: State) -> float:
    return a.u * b.v - a.v * b.u


def three_shock_residuals(params: Sequence[float], omega: float, mach: float,
                          gamma: float = GAMMA) -> Tuple[float, float]:
    """``(pressure mismatch, non-parallelism)`` for ``params = (chi, phi_r)``."""
    chi, phi = float(params[0]), float(params[1])
    try:
        _, post_inc, post_stem, _ = _frame_states(chi, omega, mach, gamma)
        post_refl = stationary_shock(post_inc, (math.cos(phi), math.sin(phi)), gamma)
    except ValueError:
        return (1e3, 1e3)
    r_p = (post_refl.p - post_stem.p) / post_stem.p
    scale = max(post_refl.speed() * post_stem.speed(), 1e-12)
    r_ang = _cross(post_refl, post_stem) / scale
    return (r_p, r_ang)


def _polish(params: Sequence[float], omega: float, mach: float,
            gamma: float) -> Optional[np.ndarray]:
    """Damped Newton on the 2x2 system with a numerical Jacobian."""
    p = np.array(params, dtype=float)
    for _ in range(120):
        f = np.array(three_shock_residuals(p, omega, mach, gamma))
        if not np.all(np.isfinite(f)):
            return None
        if np.max(np.abs(f)) < 1e-13:
            return p
        jac = np.zeros((2, 2))
        h = 1e-7
        for k in range(2):
            dp = np.zeros(2)
            dp[k] = h
            jac[:, k] = (np.array(three_shock_residuals(p + dp, omega, mach, gamma))
                         - np.array(three_shock_residuals(p - dp, omega, mach, gamma))) / (2 * h)
        try:
            step = np.linalg.solve(jac, -f)
        except np.linalg.LinAlgError:
            return None
        # keep the iterate inside the physical window
        alpha = 1.0
        for _ in range(40):
            trial = p + alpha * step
            if 0.0 < trial[0] < omega - 1e-6:
                ft = np.array(three_shock_residuals(trial, omega, mach, gamma))
                if np.all(np.isfinite(ft)) and np.max(np.abs(ft)) < np.max(np.abs(f)):
                    break
            alpha *= 0.5
        else:
            return None
        p = p + alpha * step
    return None


def solve_three_shock(mach: float = SHOCK_MACH,
                      wall_angle_deg: float = WALL_ANGLE_DEG,
                      gamma: float = GAMMA,
                      n_seeds: int = 24) -> Dict[str, object]:
    """Solve the pseudo-steady three-shock problem for the dMR configuration.

    Returns a dict of scalars/states; raises :class:`NoThreeShockSolution` when
    the polar intersection does not exist (honest FAIL, never a fabricated
    number).
    """
    omega = math.radians(wall_angle_deg)
    solutions = []
    chis = np.linspace(0.02 * omega, 0.95 * omega, n_seeds)
    phis = np.linspace(0.0, 2.0 * math.pi, 2 * n_seeds, endpoint=False)
    for chi0 in chis:
        for phi0 in phis:
            root = _polish((chi0, phi0), omega, mach, gamma)
            if root is None:
                continue
            if not (0.0 < root[0] < omega):
                continue
            res = three_shock_residuals(root, omega, mach, gamma)
            if max(abs(res[0]), abs(res[1])) > 1e-10:
                continue
            solutions.append(root)
    if not solutions:
        raise NoThreeShockSolution(
            f"no polar intersection for M={mach}, wall angle {wall_angle_deg} deg")
    # de-duplicate (each physical root shows up from many seeds)
    uniq = []
    for cand in solutions:
        if not any(abs(cand[0] - u[0]) < 1e-6 and abs(cand[1] - u[1]) < 1e-6 for u in uniq):
            uniq.append(cand)
    uniq.sort(key=lambda c: c[0])
    branches = [_branch(root[0], root[1], omega, mach, gamma, idx)
                for idx, root in enumerate(uniq)]
    primary = branches[0]
    return {
        "case": "double_mach_reflection · three-shock (von Neumann) theory",
        "mach": mach,
        "wall_angle_deg": wall_angle_deg,
        "gamma": gamma,
        "n_roots_found": len(branches),
        "roots_chi_deg": sorted(b["chi_deg"] for b in branches),
        "branch_labels": [b["label"] for b in branches],
        "branches": {b["label"]: b for b in branches},
        # convenience aliases for the lowest-chi branch (never the only answer)
        "chi_deg": primary["chi_deg"],
        "q": primary["q"],
        "states_triple_point_frame": primary["states_triple_point_frame"],
        "slip_line_properties": primary["slip_line_properties"],
        "rh_flux_residuals": primary["rh_flux_residuals"],
        "rh_worst": primary["rh_worst"],
        "shock_angles_deg": primary["shock_angles_deg"],
        "mach_numbers": primary["mach_numbers"],
    }


def _branch(chi: float, phi: float, omega: float, mach: float,
            gamma: float, index: int = 0) -> Dict[str, object]:
    """Full description of one polar-intersection branch."""
    up, post_inc, post_stem, q = _frame_states(chi, omega, mach, gamma)
    post_refl = stationary_shock(post_inc, (math.cos(phi), math.sin(phi)), gamma)
    n_inc = _incident_normal(omega)
    n_refl = (math.cos(phi), math.sin(phi))
    n_stem = _stem_normal_preferred()
    residuals = {
        "incident": rh_flux_residuals(up, post_inc, n_inc, gamma),
        "reflected": rh_flux_residuals(post_inc, post_refl, n_refl, gamma),
        "stem": rh_flux_residuals(up, post_stem, n_stem, gamma),
    }
    worst = max(abs(v) for r in residuals.values() for v in r.values())
    # normal-component Mach downstream of the reflected shock (<1 => strong shock)
    vn3 = post_refl.u * n_refl[0] + post_refl.v * n_refl[1]
    chi_deg = math.degrees(chi)
    branch = {
        "label": "b%d_chi%05.2fdeg" % (index, chi_deg),
        "chi_deg": chi_deg,
        "q": q,
        "triple_point_speed_lab": [q * math.cos(chi), q * math.sin(chi)],
        "slip_line_direction_deg": math.degrees(math.atan2(post_stem.v, post_stem.u)),
        "states_triple_point_frame": {
            "1_upstream": up.as_dict(),
            "2_post_incident": post_inc.as_dict(),
            "3_post_reflected": post_refl.as_dict(),
            "3p_post_stem": post_stem.as_dict(),
        },
        "slip_line_properties": {
            "p_post_reflected": post_refl.p,
            "p_post_stem": post_stem.p,
            "pressure_mismatch_rel": (post_refl.p - post_stem.p) / post_stem.p,
            "velocity_cross_norm": _cross(post_refl, post_stem)
            / max(post_refl.speed() * post_stem.speed(), 1e-12),
            "velocity_dot_sign": float(np.sign(post_refl.u * post_stem.u
                                               + post_refl.v * post_stem.v)),
        },
        "shock_angles_deg": {
            "incident_normal_from_y_axis": 180.0 - math.degrees(omega),
            "reflected_normal_from_x_axis": math.degrees(phi) % 360.0,
            "stem_normal_from_x_axis": 180.0,
            "stem_plane_from_wall": 90.0,
        },
        "mach_numbers": {
            "M2_post_incident": post_inc.speed() / post_inc.sound(gamma),
            "M1n_incident": mach,
            "M1n_stem": q * math.cos(chi) / PRE.sound(gamma),
            "Mn_downstream_reflected": vn3 / post_refl.sound(gamma),
        },
        "rh_flux_residuals": residuals,
        "rh_worst": worst,
    }
    branch["trajectory"] = triple_point_trajectory(branch)
    return branch


# ------------------------------------------------- Galilean (independent) form
def solve_three_shock_galilean(mach: float = SHOCK_MACH,
                               wall_angle_deg: float = WALL_ANGLE_DEG,
                               gamma: float = GAMMA,
                               n_seeds: int = 16) -> Dict[str, object]:
    """Same problem, different closure -- cross-check for :func:`solve_three_shock`.

    Unknowns ``(chi, phi_r, psi)``: ``psi`` is the *free* Mach-stem normal angle
    and the closure is "the post-stem flow is parallel to the wall in the **lab**
    frame", i.e. ``(u_3' + q_vec)·(0, 1) = 0``.  This is the Galilean-invariant
    statement of "stem plane perpendicular to the wall"; the two formulations
    must agree on ``chi``.
    """
    omega = math.radians(wall_angle_deg)

    def residual(p: np.ndarray) -> np.ndarray:
        chi, phi, psi = float(p[0]), float(p[1]), float(p[2])
        try:
            up, post_inc, _, q = _frame_states(chi, omega, mach, gamma)
            post_stem = stationary_shock(up, (math.cos(psi), math.sin(psi)), gamma)
            post_refl = stationary_shock(post_inc, (math.cos(phi), math.sin(phi)), gamma)
        except ValueError:
            return np.array([1e3, 1e3, 1e3])
        r_p = (post_refl.p - post_stem.p) / post_stem.p
        scale = max(post_refl.speed() * post_stem.speed(), 1e-12)
        r_ang = _cross(post_refl, post_stem) / scale
        r_wall = (post_stem.v + q * math.sin(chi)) / max(q, 1e-12)
        return np.array([r_p, r_ang, r_wall])

    best = None
    for chi0 in np.linspace(0.05 * omega, 0.9 * omega, n_seeds):
        for phi0 in np.linspace(0.5, 2.0 * math.pi, n_seeds, endpoint=False):
            for psi0 in np.linspace(1.2 * math.pi, 1.9 * math.pi, 5):
                p = np.array([chi0, phi0, psi0], dtype=float)
                ok = True
                for _ in range(80):
                    f = residual(p)
                    if np.max(np.abs(f)) < 1e-13:
                        break
                    jac = np.zeros((3, 3))
                    h = 1e-7
                    for k in range(3):
                        dp = np.zeros(3)
                        dp[k] = h
                        jac[:, k] = (residual(p + dp) - residual(p - dp)) / (2 * h)
                    try:
                        step = np.linalg.solve(jac, -f)
                    except np.linalg.LinAlgError:
                        ok = False
                        break
                    alpha = 1.0
                    for _ in range(40):
                        if np.max(np.abs(residual(p + alpha * step))) < np.max(np.abs(f)):
                            break
                        alpha *= 0.5
                    else:
                        ok = False
                        break
                    p = p + alpha * step
                if not ok:
                    continue
                if not (0.0 < p[0] < omega):
                    continue
                if np.max(np.abs(residual(p))) > 1e-10:
                    continue
                if best is None or p[0] < best[0]:
                    best = p
    if best is None:
        raise NoThreeShockSolution("galilean formulation found no root")
    return {
        "chi_deg": math.degrees(best[0]),
        "reflected_normal_deg": math.degrees(best[1]) % 360.0,
        "stem_normal_deg": math.degrees(best[2]) % 360.0,
        "stem_plane_from_wall_deg": _plane_deviation(best[2]),
        "residual": [float(r) for r in residual(best)],
    }


def _plane_deviation(normal_angle: float) -> float:
    """Deviation (deg) of a plane from the wall normal, given its normal angle."""
    m = math.degrees(normal_angle) % 180.0
    return float(min(m, 180.0 - m))


# --------------------------------------------------------------- trajectory
def triple_point_trajectory(solution: Dict[str, object],
                            times: Iterable[float] = (0.1, 0.15, 0.2),
                            x_hinge: float = X_HINGE) -> Dict[str, object]:
    """Predicted straight triple-point trajectory ``T(t)`` from the theory."""
    chi = math.radians(float(solution["chi_deg"]))
    q = float(solution["q"])
    pts = {float(t): {"x": x_hinge + q * math.cos(chi) * float(t),
                      "y": q * math.sin(chi) * float(t)} for t in times}
    return {"chi_deg": float(solution["chi_deg"]), "q": q,
            "x_hinge": x_hinge, "points": pts}


def trajectory_point(branch: Dict[str, object], t: float) -> Dict[str, float]:
    """Predicted triple point at time ``t`` (keys survive a JSON round-trip)."""
    pts = branch["trajectory"]["points"]                    # type: ignore[index]
    for key, val in pts.items():
        if abs(float(key) - float(t)) < 1e-12:
            return val                                        # type: ignore[return-value]
    raise KeyError(f"no predicted point at t={t}")


def trajectory_angle_from_point(x_t: float, y_t: float,
                                x_hinge: float = X_HINGE) -> float:
    """``chi`` (deg) implied by a measured triple point at ``(x_t, y_t)``."""
    return math.degrees(math.atan2(y_t, x_t - x_hinge))


# ------------------------------------------------------- measured triple point
def measure_triple_point(density: np.ndarray, xc: Sequence[float],
                         yc: Sequence[float],
                         level: float = FRONT_LEVEL,
                         top_fraction: float = 0.4,
                         kink_cells: float = 1.5,
                         y_max: float = 0.97) -> Dict[str, object]:
    """Locate the triple point in a captured density field.

    The front is traced row by row as the **rightmost** crossing of ``level``
    (undisturbed air is on the right, so scanning from +x inward finds the
    outermost wave).  Above the triple point that locus is the incident front, a
    straight line of slope ``1/sqrt(3)``; below it, the locus follows the
    near-vertical Mach stem and therefore leaves the line.  The triple point is
    the highest row where the departure exceeds ``kink_cells`` cells, refined by
    linear interpolation between the last agreeing and the first departing row.

    Returns ``{}`` when the field does not contain a resolved kink (no silent
    default): the caller decides what to do.
    """
    rho = np.asarray(density, dtype=float)
    xc = np.asarray(xc, dtype=float).ravel()
    yc = np.asarray(yc, dtype=float).ravel()
    ny = rho.shape[1] if rho.shape[0] == xc.size else rho.shape[0]
    # cell-centred array is (nx, ny) in this repo
    dx = float(np.mean(np.diff(xc)))
    locus = {}
    for j in range(ny):
        if yc[j] > y_max:
            continue
        prof = rho[:, j]
        above = np.where(prof >= level)[0]
        if above.size == 0:
            continue
        # rightmost crossing of ``level``, scanning inward from the right
        # boundary: undisturbed gas (low density) is on the +x side, so the
        # first transition  high -> low  met from the right is the outer wave.
        idx = None
        for i in range(prof.size - 1, 0, -1):
            if prof[i] < level <= prof[i - 1]:
                idx = i
                break
        if idx is None:
            continue
        w = (prof[idx - 1] - level) / (prof[idx - 1] - prof[idx])
        locus[float(yc[j])] = float(xc[idx - 1] + w * dx)
    if len(locus) < 6:
        return {}
    ys = np.array(sorted(locus))
    xs = np.array([locus[y] for y in ys])
    n_top = max(4, int(round(top_fraction * ys.size)))
    slope, intercept = np.polyfit(ys[-n_top:], xs[-n_top:], 1)
    resid = xs - (intercept + slope * ys)
    tol = kink_cells * dx
    depart = np.abs(resid) > tol
    if not depart.any():
        return {"locus": {float(y): float(x) for y, x in zip(ys, xs)},
                "front_slope": float(slope),
                "front_angle_deg": float(np.degrees(np.arctan2(1.0, slope))),
                "residual_cells": [float(r) for r in resid / dx],
                "note": "no kink resolved"}
    # Rows *below* the triple point depart from the incident line (the Mach stem
    # is nearly vertical), rows above it do not: the triple point is the first
    # row counted from the wall that is back on the line.
    if depart.all():
        return {"note": "the traced front never returns to the incident line"}
    j_first_ok = int(np.argmax(~depart))
    if j_first_ok == 0:
        return {"note": "no kink resolved",
                "front_slope": float(slope),
                "front_angle_deg": float(np.degrees(np.arctan2(1.0, slope))),
                "residual_cells": [float(r) for r in resid / dx]}
    i0 = j_first_ok - 1
    # The detected row is biased low by ~tol/|slope| rows: the residual only
    # exceeds the threshold once the (near-vertical) stem has drifted that far
    # from the extrapolated incident line.  Remove the bias geometrically: fit
    # the stem locus over the last few departing rows and intersect it with the
    # fitted incident line -- that intersection *is* the triple point.
    n_fit = min(4, int(np.count_nonzero(depart)))
    stem_rows = np.arange(i0 - n_fit + 1, i0 + 1)
    stem_rows = stem_rows[stem_rows >= 0]
    stem_slope, stem_intercept = np.polyfit(ys[stem_rows], xs[stem_rows], 1)
    if abs(stem_slope - slope) < 1e-9:
        y_t = float(ys[i0])
    else:
        y_t = float((intercept - stem_intercept) / (stem_slope - slope))
        if not (ys[stem_rows[0]] - 2.0 * abs(ys[1] - ys[0]) <= y_t
                <= ys[j_first_ok] + 2.0 * abs(ys[1] - ys[0])):
            y_t = float(ys[i0])                      # intersection out of range: keep raw
    x_t = float(intercept + slope * y_t)
    return {
        "x": x_t,
        "y": y_t,
        "front_slope_dx_dy": float(slope),
        "front_angle_deg": float(np.degrees(np.arctan2(1.0, slope))),
        "incident_angle_exact_deg": 60.0,
        "residual_cells": [float(r) for r in resid / dx],
        "locus": {float(y): float(x) for y, x in zip(ys, xs)},
        "n_rows": int(ys.size),
        "deviation_cells_at_kink": float(abs(resid[i0]) / dx),
        "stem_slope_dx_dy": float(stem_slope),
        "y_cells": float(y_t / (ys[1] - ys[0])) if ys.size > 1 else float("nan"),
    }


# ------------------------------------------------------------------- reporting
def summary(mach: float = SHOCK_MACH, wall_angle_deg: float = WALL_ANGLE_DEG,
            gamma: float = GAMMA, times: Sequence[float] = (0.1, 0.15, 0.2),
            galilean: bool = True) -> Dict[str, object]:
    """Theory bundle: every polar-intersection branch plus the Galilean cross-check."""
    primary = solve_three_shock(mach, wall_angle_deg, gamma)
    branches = {}
    for label, br in primary["branches"].items():
        br = dict(br)
        br["trajectory"] = triple_point_trajectory(br, times)
        branches[label] = br
    out: Dict[str, object] = {
        "theory": primary,
        "branches": branches,
        "branch_labels": primary["branch_labels"],
        "detachment_angle_deg": math.degrees(detachment_angle(mach, gamma)),
        "times": list(times),
    }
    if galilean:
        gal = solve_three_shock_galilean(mach, wall_angle_deg, gamma)
        out["galilean_cross_check"] = gal
        out["chi_agreement_deg"] = min(abs(br["chi_deg"] - gal["chi_deg"])
                                       for br in branches.values())
    return out


def main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mach", type=float, default=SHOCK_MACH)
    ap.add_argument("--wall-angle", type=float, default=WALL_ANGLE_DEG)
    ap.add_argument("--json", default=None, help="write the summary here")
    ap.add_argument("--polar", action="store_true", help="write the polar figure")
    ap.add_argument("--polar-out", default="validation/dmr/three_shock/polar.png")
    args = ap.parse_args()
    out = summary(args.mach, args.wall_angle)
    text = json.dumps(out, indent=1)
    print(text)
    if args.json:
        from pathlib import Path
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n")
    if args.polar:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        theta, pr = shock_polar(args.mach)
        ax.plot(theta, pr, lw=1.6, label=f"incident shock polar (M={args.mach:g})")
        ax.axvline(math.degrees(detachment_angle(args.mach)), ls=":", color="0.4",
                   label="detachment angle")
        ax.set_xlabel(r"deflection $\theta$ [deg]")
        ax.set_ylabel(r"$p_2/p_1$")
        ax.set_title("Oblique-shock polar used by the three-shock solver")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = Path(args.polar_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        print(f"figure -> {p}")


if __name__ == "__main__":  # pragma: no cover
    main()