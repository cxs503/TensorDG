"""Woodward--Colella double Mach reflection (Mach 10, 60 deg) on a 2-D DG mesh.

This is the signature two-dimensional *system* case of the anchor
``REPRO-DG-CKS1998`` (Cockburn & Shu, JCP 141 (1998) 199--224, Part V) and of
Woodward & Colella (1984, JCP 54, 115--173).

Physical setup
--------------
Domain ``[0, 4] x [0, 1]``.  A Mach-10 shock in air (``gamma = 1.4``) whose
front passes through ``(1/6, 0)`` and makes a 60 deg angle with the x-axis:

    front:  y = sqrt(3) * (x - 1/6)          (equivalently x = 1/6 + y/sqrt(3))

The shocked (post-shock) gas lies *behind* the front, i.e. in the upper-left
half plane ``y > sqrt(3) (x - 1/6)``; the undisturbed gas (``rho = 1.4``,
``p = 1``, at rest) lies ahead of it.  Therefore the front runs *down and to
the right* into the reflecting floor ``y = 0`` and the double Mach structure
(Mach stem, reflected shock, triple point, slip line) develops along the wall
and sweeps towards ``x = 4``.

Boundary conditions (the standard set, see e.g. Woodward & Colella 1984 sec. 4)

    bottom : impermeable slip wall
    left   : post-shock inflow (the whole left edge is behind the front)
    top    : exact motion of the incident front -- post-shock for
             ``x < x_front(1, t)``, pre-shock otherwise
    right  : outflow

Exact reference data (no hard-coded "answers")
----------------------------------------------
* the post-shock state is built from the one-dimensional Rankine--Hugoniot
  relations along the front normal ``n = (sin 60, -cos 60)`` -- ``rh_residuals``
  re-checks the jump from the two states alone, so a wrong sign cannot pass;
* the undisturbed part of the front is a *plane* convecting with the exact
  normal shock speed ``W = M a_1``, hence

      x_front(y, t) = 1/6 + y/sqrt(3) + W/sin(60 deg) * t

  which is what ``analytic_front_x`` returns and what ``measure_primary_front``
  compares the computed solution against.

History (why the orientation is written out so explicitly): the pre-m20397
version placed the post-shock state *ahead* of the front and gave it a velocity
pointing up-left, i.e. it shipped the time-reversed Riemann problem.  That jump
is a valid Rankine--Hugoniot discontinuity, so a state-only check passed, but
the front leaves the domain through the left edge at ``t = (1/6)/11.55 =
0.0144`` and no wall reflection ever happens.  ``tests/test_double_mach_reflection.py``
now pins both the jump *and* the direction of propagation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from tensordg.euler_dg_2d import solve_euler_dg_2d, cell_fields
from tensordg.euler_1d import GAMMA

# --- exact reference data -------------------------------------------------
PRE_SHOCK = (1.4, 0.0, 0.0, 1.0)          # rho, u, v, p  (undisturbed air)
SHOCK_MACH = 10.0
SHOCK_ANGLE_DEG = 60.0
FRONT_X0 = 1.0 / 6.0
FRONT_SLOPE = math.tan(math.radians(SHOCK_ANGLE_DEG))      # sqrt(3)

A1 = math.sqrt(GAMMA * PRE_SHOCK[3] / PRE_SHOCK[0])        # upstream sound speed
SHOCK_SPEED = SHOCK_MACH * A1                              # |W| along NORMAL
#: unit normal of the front pointing *into the undisturbed gas* (down-right)
NORMAL = (math.sin(math.radians(SHOCK_ANGLE_DEG)),
          -math.cos(math.radians(SHOCK_ANGLE_DEG)))
#: unit tangent of the front
TANGENTIAL = (-NORMAL[1], NORMAL[0])

#: horizontal speed of the plane front at fixed y (what the solver must match)
FRONT_X_SPEED = SHOCK_SPEED / NORMAL[0]


def normal_shock_downstream(pre=PRE_SHOCK, mach=SHOCK_MACH, normal=NORMAL,
                            gamma=GAMMA):
    """Exact Rankine--Hugoniot downstream state of a normal shock.

    ``pre`` is the upstream state in the frame where the *gas ahead* is at
    rest; the shock runs into it along ``normal`` with speed ``mach * a1``.
    The returned state is expressed in the same laboratory frame.
    """
    rho1, u1, v1, p1 = pre
    nx, ny = normal
    a1 = math.sqrt(gamma * p1 / rho1)
    m = mach
    rho2 = rho1 * (gamma + 1.0) * m * m / ((gamma - 1.0) * m * m + 2.0)
    p2 = p1 * (1.0 + 2.0 * gamma / (gamma + 1.0) * (m * m - 1.0))
    un1 = u1 * nx + v1 * ny
    un2 = un1 + 2.0 / (gamma + 1.0) * (m - 1.0 / m) * a1
    ut1 = -u1 * ny + v1 * nx
    u2 = un2 * nx + ut1 * TANGENTIAL[0]
    v2 = un2 * ny + ut1 * TANGENTIAL[1]
    return (rho2, u2, v2, p2)


#: shocked gas behind the front: rho = 8, p = 116.5, |V| = 8.25 pointing
#: down-right (towards the reflecting floor)
POST_SHOCK = normal_shock_downstream()


def rh_residuals(pre, post, mach=SHOCK_MACH, normal=NORMAL, gamma=GAMMA):
    """Rankine--Hugoniot residuals of the jump ``pre -> post``.

    Returns ``dict(mass, normal_momentum, tangential_momentum, energy,
    shock_speed)``; every residual must vanish for an exact discontinuity.
    ``shock_speed`` is recovered from the mass jump alone, so this routine does
    not trust ``mach``.
    """
    rho1, u1, v1, p1 = pre
    rho2, u2, v2, p2 = post
    nx, ny = normal
    un1 = u1 * nx + v1 * ny
    un2 = u2 * nx + v2 * ny
    ut1 = -u1 * ny + v1 * nx
    ut2 = -u2 * ny + v2 * nx
    speed = (rho2 * un2 - rho1 * un1) / (rho2 - rho1)      # lab-frame W
    mass = rho1 * (un1 - speed) - rho2 * (un2 - speed)
    momentum = (rho1 * (un1 - speed) ** 2 + p1
                - rho2 * (un2 - speed) ** 2 - p2)
    tangential = rho1 * (un1 - speed) * ut1 - rho2 * (un2 - speed) * ut2
    e1 = p1 / (gamma - 1.0) + 0.5 * rho1 * (u1 ** 2 + v1 ** 2)
    e2 = p2 / (gamma - 1.0) + 0.5 * rho2 * (u2 ** 2 + v2 ** 2)
    # [ (e + p) u_n ] = W [ e ]  (lab-frame total energy density e)
    energy = (e1 + p1) * un1 - (e2 + p2) * un2 - speed * (e1 - e2)
    return {"mass": mass, "normal_momentum": momentum,
            "tangential_momentum": tangential, "energy": energy,
            "shock_speed": speed}


def analytic_front_x(y, t):
    """Exact x-position of the undisturbed part of the incident front."""
    y = np.asarray(y, dtype=float)
    return FRONT_X0 + y / FRONT_SLOPE + FRONT_X_SPEED * np.asarray(t, dtype=float)


def _behind_front(x, y, slope=FRONT_SLOPE, x0=FRONT_X0):
    """``True`` where the point lies behind (post-shock side of) the front."""
    return y > slope * (x - x0)


def _initial_double_mach(x, y):
    """Exact initial data: shocked gas up-left of the front, air elsewhere."""
    mask = torch.as_tensor(np.asarray(_behind_front(x.cpu().numpy(),
                                                    y.cpu().numpy())))
    return tuple(torch.where(mask, torch.full_like(x, post), torch.full_like(x, pre))
                 for post, pre in zip(POST_SHOCK, PRE_SHOCK))


#: backwards-compatible alias (the initial data is the exact solution at t = 0)
initial_double_mach = _initial_double_mach


def _state_tensor(values, reference):
    return torch.as_tensor(values, device=reference.device, dtype=reference.dtype)


def top_boundary_state(X, Y, t):
    """Exact ghost state of the moving incident front on the top edge."""
    x_front = analytic_front_x(Y.cpu().numpy(), t)
    mask = torch.as_tensor(X.cpu().numpy() < x_front, device=X.device)
    out = []
    for post, pre in zip(POST_SHOCK, PRE_SHOCK):
        out.append(torch.where(mask, torch.full_like(X, post),
                               torch.full_like(X, pre)))
    return torch.stack(out, dim=-1)


def left_boundary_state(X, Y, t):
    """Post-shock inflow: the whole left edge stays behind the front."""
    return _state_tensor(POST_SHOCK, X)


def double_mach_bcs():
    """Standard boundary set of the double Mach reflection problem."""
    return {"bottom": "wall", "left": left_boundary_state,
            "right": "outflow", "top": top_boundary_state}


def solve_double_mach(
    ncells=(240, 60),
    degree=2,
    final_time=0.2,
    cfl=0.15,
    limiter="tvb",
    tvb_constant=100.0,
    limiter_stage="final",
    bcs=True,
    device="cpu",
    dtype=torch.float64,
):
    """Run the double Mach reflection benchmark.

    Parameters
    ----------
    ncells : (int, int)
        Cells in x and y.  Woodward--Colella used 800 x 200 for P0; for
        high-order DG, 240 x 60 with P2 gives comparable effective resolution.
    degree : int
        Polynomial degree (1, 2 or 3).
    final_time : float
        End time of the experiment (Woodward--Colella: t = 0.2).
    cfl : float
        CFL number.
    limiter : str or None
        Shock limiter ("tvb", "weno" or None).
    tvb_constant : float
        TVB constant M.
    limiter_stage : str
        When to apply the limiter ("every" or "final").
    bcs : bool or dict
        ``True`` (default) installs :func:`double_mach_bcs`; ``False`` keeps the
        legacy zero-gradient outer faces (only useful for boundary-condition
        studies -- the reflection cannot form without the wall).
    """
    domain = ((0.0, 4.0), (0.0, 1.0))
    face_bcs = double_mach_bcs() if bcs is True else (bcs or None)
    x, y, coeff, metadata = solve_euler_dg_2d(
        _initial_double_mach,
        ncells=ncells,
        degree=degree,
        domain=domain,
        final_time=final_time,
        cfl=cfl,
        limiter=limiter,
        tvb_constant=tvb_constant,
        limiter_stage=limiter_stage,
        periodic=(False, False),
        initial_projection="cell-average",
        bcs=face_bcs,
        device=device,
        dtype=dtype,
    )
    return x, y, coeff, metadata


# --- measurement ----------------------------------------------------------
DENSITY_LEVEL = 0.5 * (PRE_SHOCK[0] + 8.0)      # midpoint of the density jump


def _interp_crossing(xc, prof, level):
    """Rightmost crossing of ``level`` scanning from x = +inf, sub-cell."""
    diff = prof - level
    for i in range(len(diff) - 1, 0, -1):
        if diff[i] < 0.0 <= diff[i - 1]:
            w = diff[i - 1] - diff[i]
            frac = diff[i - 1] / w if w > 0 else 0.0
            return 0.5 * (xc[i] + xc[i - 1]) + (frac - 0.5) * (xc[i] - xc[i - 1])
    return None


def measure_primary_front(density, x, y, y_levels=(0.7, 0.8, 0.9),
                          level=DENSITY_LEVEL):
    """Locate the undisturbed part of the incident front at several heights.

    ``density`` is the cell-average density field of shape ``(nx, ny)``;
    ``x``/``y`` are the cell-centre coordinates.  Returns
    ``{y_level: x_front}``; the scan starts ahead of the front (right edge) and
    stops at the first downward crossing of ``level``, which is the incident
    front (the Mach stem / slip line lie behind it).
    """
    density = np.asarray(density)
    xc = np.asarray(x).ravel()
    yc = np.asarray(y).ravel()
    out = {}
    for target in y_levels:
        j = int(np.argmin(np.abs(yc - target)))
        crossing = _interp_crossing(xc, density[:, j], level)
        if crossing is not None:
            out[float(yc[j])] = float(crossing)
    return out


def measure_front_speed(density, x, y, t1, t2, y_level=0.85):
    """Front speed in x between two stored states (cell-average fields)."""
    p1 = measure_primary_front(density[0], x, y, y_levels=(y_level,))
    p2 = measure_primary_front(density[1], x, y, y_levels=(y_level,))
    (y1, x1), (y2, x2) = next(iter(p1.items())), next(iter(p2.items()))
    return (x2 - x1) / (t2 - t1)


def sample_behind_front(density_state, x, y, t, y_level=0.85, offset=0.05):
    """Cell-average state a fixed distance *behind* the analytic front."""
    rho, u, v, p = density_state
    xc = np.asarray(x).ravel()
    yc = np.asarray(y).ravel()
    j = int(np.argmin(np.abs(yc - y_level)))
    x_target = float(analytic_front_x(y_level, t)) - offset
    i = int(np.argmin(np.abs(xc - x_target)))
    return {"x": float(xc[i]), "y": float(yc[j]),
            "rho": float(np.asarray(rho)[i, j]),
            "u": float(np.asarray(u)[i, j]),
            "v": float(np.asarray(v)[i, j]),
            "p": float(np.asarray(p)[i, j])}


def _compute_shock_angle(x, y, density, level=DENSITY_LEVEL,
                         y_levels=np.linspace(0.5, 0.9, 9)):
    """Angle of the undisturbed incident front with the x-axis (degrees).

    The front is a straight line ``y = m (x - x0)``; the angle returned is
    ``atan(m)``, i.e. 60 deg for the exact initial data.  Points behind the
    triple point are excluded by only using ``y >= 0.5``.
    """
    positions = [(v, k) for k, v in
                 measure_primary_front(density, x, y, y_levels=y_levels,
                                       level=level).items()]
    if len(positions) < 3:
        return None, None
    arr = np.array(positions)
    slope = np.polyfit(arr[:, 0], arr[:, 1], 1)[0]
    return float(np.rad2deg(np.arctan(slope))), arr


def plot_double_mach(x, y, coeff, output_dir, degree=2, title_extra=""):
    """Density contours of the double Mach reflection (classic figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xc = x.cpu().numpy().ravel()
    yc = y.cpu().numpy().ravel()
    fields = cell_fields(coeff)
    density = fields["density"].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    im0 = axes[0].pcolormesh(xc, yc, density.T, cmap="viridis", shading="auto")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"Double Mach reflection | DG P{degree}, density{title_extra}")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0], label="density")

    levels = np.linspace(1.5, 22.0, 40)
    axes[1].contour(xc, yc, density.T, levels=levels, colors="black",
                    linewidths=0.4)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("density contours (Woodward--Colella levels)")
    axes[1].set_aspect("equal")

    fig.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "double_mach_density.png", dpi=150)
    plt.close(fig)
    return out / "double_mach_density.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=240)
    parser.add_argument("--ny", type=int, default=60)
    parser.add_argument("--degree", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--time", type=float, default=0.2)
    parser.add_argument("--cfl", type=float, default=0.15)
    parser.add_argument("--limiter", default="tvb", choices=["tvb", "weno", "none"])
    parser.add_argument("--tvb-constant", type=float, default=100.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="double_mach_results")
    args = parser.parse_args()

    limiter = args.limiter if args.limiter != "none" else None
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Double Mach Reflection | DG P{args.degree}")
    print(f"  Grid: {args.nx} x {args.ny} cells")
    print(f"  CFL: {args.cfl}, limiter: {args.limiter}, M={args.tvb_constant}")

    x, y, coeff, metadata = solve_double_mach(
        ncells=(args.nx, args.ny), degree=args.degree,
        final_time=args.time, cfl=args.cfl, limiter=limiter,
        tvb_constant=args.tvb_constant, limiter_stage="final",
        device=args.device,
    )
    fields = cell_fields(coeff)
    density = fields["density"].cpu().numpy()

    front = measure_primary_front(density, x, y)
    reference = {k: float(analytic_front_x(k, args.time)) for k in front}
    angle, _ = _compute_shock_angle(x, y, density)
    plot_double_mach(x, y, coeff, output, degree=args.degree,
                     title_extra=f", t={args.time}")

    record = {
        "benchmark": "double_mach_reflection",
        **metadata,
        "final_time": args.time,
        "front_x": front,
        "front_x_analytic": reference,
        "front_error": {k: front[k] - reference[k] for k in front},
        "shock_angle_deg": angle,
        "shock_angle_reference_deg": SHOCK_ANGLE_DEG,
        "density_min": float(density.min()),
        "density_max": float(density.max()),
        "reference": "Woodward & Colella (1984), JCP 54, 115-173; "
                     "Cockburn & Shu (1998), JCP 141, 199-224 (Part V)",
    }
    (output / "metrics.json").write_text(json.dumps(record, indent=2, default=str))

    print(f"  Steps: {metadata['steps']}")
    print(f"  Density range: {record['density_min']:.4f} .. {record['density_max']:.4f}")
    for k in sorted(front):
        print(f"  front x(y={k:.2f}) = {front[k]:.4f} (exact {reference[k]:.4f}, "
              f"err {front[k] - reference[k]:+.4f})")
    if angle is not None:
        print(f"  Primary front angle: {angle:.2f} deg (exact {SHOCK_ANGLE_DEG})")
    print(f"  Results: {output}")


if __name__ == "__main__":
    main()