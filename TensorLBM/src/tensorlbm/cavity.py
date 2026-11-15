"""Wall-bounded verification cases for D2Q9 BGK.

Two classic cases exercise :mod:`tensorlbm.boundaries` (half-way bounce-back
with a moving wall), which the periodic solvers in this package could not
test:

* **Plane Couette channel** -- walls in ``y`` only (periodic ``x``), top wall
  moving at ``U``, no driving pressure gradient.  The exact steady solution
  is the linear profile ``u(Y) = U * Y``; it validates the *moving-wall term
  itself* independently of any discretisation argument.
* **Lid-driven cavity** -- closed box, all four walls solid, top wall moving
  at ``U``.  Compared against Ghia, Ghia & Shin (1982), *J. Comput. Phys.*
  48, 387-411: Table I (``u`` along the vertical centreline) and Table II
  (``v`` along the horizontal centreline), Re = 100 and 400.

Conventions
-----------
With half-way bounce-back the physical wall plane sits half a lattice
spacing outside the first fluid node.  For a channel of ``N`` nodes with one
solid node on each side, the gap between the two wall planes is ``H = N - 2`` and fluid node ``y`` sits
at wall-normal coordinate ``Y = (y - 0.5) / H`` in ``(0, 1)``.  Velocities are
reported normalised by the lid speed ``U``.  ``Re = U H / nu`` with
``nu = (tau - 1/2)/3``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .boundaries import HalfwayBounceBack
from .collision import bgk_collision
from .lattice import D2Q9


@dataclass
class WallRunResult:
    """Container for a wall-bounded run."""

    Nx: int
    Ny: int
    U: float
    tau: float
    re: float
    steps: int
    converged: bool
    residual: float
    ux: torch.Tensor
    uy: torch.Tensor
    rho: torch.Tensor
    solid: torch.Tensor
    H: int
    history: list = field(default_factory=list)

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    def centreline_u(self):
        """``u/U`` along the vertical centreline and the corresponding ``Y``."""
        xc = self.Nx // 2
        ys = torch.arange(1, self.Ny - 1)
        Y = (ys - 0.5) / self.H
        return Y, self.ux[ys, xc] / self.U

    def centreline_v(self):
        """``v/U`` along the horizontal centreline and the corresponding ``X``."""
        xc = self.Ny // 2
        xs = torch.arange(1, self.Nx - 1)
        if not bool(self.solid[xc, 0]):
            X = (xs - 0.5) / self.H
        else:
            X = (xs - 0.5) / self.H
        return X, self.uy[xc, xs] / self.U


def _ring_mask(Ny: int, Nx: int) -> torch.Tensor:
    solid = torch.zeros((Ny, Nx), dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True
    solid[:, 0] = True
    solid[:, -1] = True
    return solid


def _channel_mask(Ny: int, Nx: int) -> torch.Tensor:
    solid = torch.zeros((Ny, Nx), dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True
    return solid


def _run(
    solid: torch.Tensor,
    U: float,
    tau: float,
    H: int,
    top_velocity: float = 0.0,
    drive_body_force: float = 0.0,
    n_steps: int = 20000,
    tol: float = 1.0e-7,
    min_steps: int = 0,
    check_every: int = 500,
    dtype: torch.dtype = torch.float64,
) -> WallRunResult:
    Ny, Nx = solid.shape
    omega = 1.0 / tau
    nu = (tau - 0.5) / 3.0
    re = U * H / nu if nu > 0 else float("inf")

    rho = torch.ones((Ny, Nx), dtype=dtype)
    ux = torch.zeros((Ny, Nx), dtype=dtype)
    uy = torch.zeros((Ny, Nx), dtype=dtype)
    f = D2Q9.equilibrium(rho, ux, uy)

    u_wall_x = torch.zeros((Ny, Nx), dtype=dtype)
    u_wall_x[-1, :] = top_velocity * U
    bc = HalfwayBounceBack(solid, u_wall_x=u_wall_x)
    fluid = bc.fluid

    force = [float(D2Q9.w[i]) * 3.0 * float(D2Q9.e[i, 0]) * drive_body_force
             for i in range(D2Q9.Q)] if drive_body_force else None

    prev = torch.zeros((Ny, Nx), dtype=dtype)
    converged = False
    residual = float("inf")
    history = []
    step = 0
    for step in range(1, n_steps + 1):
        rho, ux, uy = D2Q9.macroscopic(f)
        f_post = bgk_collision(f, rho, ux, uy, omega)
        if force is not None:
            for i in range(D2Q9.Q):
                f_post[i] += force[i]
        f = bc.stream(f_post)
        if step % check_every == 0:
            _, ux, _ = D2Q9.macroscopic(f)
            du = float((ux[fluid] - prev[fluid]).abs().max())
            prev = ux.clone()
            residual = du / U
            history.append((step, residual))
            if step >= max(min_steps, check_every) and residual < tol:
                converged = True
                break

    rho, ux, uy = D2Q9.macroscopic(f)
    zero = torch.zeros_like(ux)
    ux = torch.where(fluid, ux, zero)
    uy = torch.where(fluid, uy, zero)
    return WallRunResult(
        Nx=Nx, Ny=Ny, U=U, tau=tau, re=re, steps=step, converged=converged,
        residual=residual, ux=ux, uy=uy, rho=rho, solid=bc.solid, H=H,
        history=history,
    )


def run_couette(
    Ny: int = 17,
    Nx: int = 8,
    U: float = 0.05,
    tau: float = 1.0,
    n_steps: int = 20000,
    tol: float = 1.0e-9,
    min_steps: int = 0,
    check_every: int = 250,
) -> WallRunResult:
    """Plane Couette channel: solid walls at ``y = 0`` and ``y = Ny-1``,
    periodic in ``x``, top wall moving at ``U``."""
    solid = _channel_mask(Ny, Nx)
    return _run(solid, U=U, tau=tau, H=Ny - 2, top_velocity=1.0,
                n_steps=n_steps, tol=tol, min_steps=min_steps,
                check_every=check_every)


def run_poiseuille(
    Ny: int = 33,
    Nx: int = 8,
    body_force: float = 1.0e-5,
    tau: float = 0.8,
    n_steps: int = 20000,
    tol: float = 1.0e-9,
    min_steps: int = 0,
    check_every: int = 250,
) -> WallRunResult:
    """Force-driven plane channel: solid walls at ``y = 0, Ny-1``, periodic ``x``.

    Exact steady solution ``u(y) = F/(2 nu) * Y (H - Y)`` in physical
    coordinates ``Y = y - 0.5`` (wall planes at ``0`` and ``H``).
    """
    solid = _channel_mask(Ny, Nx)
    H = Ny - 2
    nu = (tau - 0.5) / 3.0
    # residual normalisation: the exact peak velocity, not an arbitrary scale
    U = body_force * H ** 2 / (8.0 * nu)
    return _run(solid, U=U, tau=tau, H=H,
                drive_body_force=body_force, n_steps=n_steps, tol=tol,
                min_steps=min_steps, check_every=check_every)


def run_cavity(
    N: int = 129,
    U: float = 0.1,
    re: float = 100.0,
    n_steps: int = 60000,
    tol: float = 1.0e-7,
    min_steps: int = 2000,
    check_every: int = 500,
) -> WallRunResult:
    """Lid-driven cavity, ``tau`` set from ``Re = U H / nu`` with ``H = N - 2``."""
    H = N - 2
    nu = U * H / re
    tau = 3.0 * nu + 0.5
    solid = _ring_mask(N, N)
    return _run(solid, U=U, tau=tau, H=H, top_velocity=1.0, n_steps=n_steps,
                tol=tol, min_steps=min_steps, check_every=check_every)


# ---------------------------------------------------------------------------
# Reference data and error measures
# ---------------------------------------------------------------------------

GHIA_U = {
    100: {
        "y/H": [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
                0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547,
                0.0000],
        "u/U": [1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332,
                -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434,
                -0.04775, -0.04192, -0.03717, 0.00000],
    },
    400: {
        "y/H": [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
                0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547,
                0.0000],
        "u/U": [1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256,
                0.02135, -0.11477, -0.17119, -0.32726, -0.24299, -0.14612,
                -0.10338, -0.09266, -0.08186, 0.00000],
    },
}

GHIA_V = {
    100: {
        "x/H": [1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
                0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625,
                0.0000],
        "v/U": [0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914,
                -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077,
                0.12317, 0.10890, 0.10091, 0.09233, 0.00000],
    },
    400: {
        "x/H": [1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
                0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625,
                0.0000],
        "v/U": [0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827,
                -0.44993, -0.38598, 0.05186, 0.30174, 0.30203, 0.28124,
                0.22965, 0.20920, 0.19713, 0.18360, 0.00000],
    },
}


def _lerp_at(coord: torch.Tensor, values: torch.Tensor, t: float) -> float:
    """Linear interpolation of ``values(coord)`` at ``t`` (clamped at ends)."""
    order = torch.argsort(coord)
    c = coord[order].to(torch.float64)
    v = values[order].to(torch.float64)
    if t <= float(c[0]):
        return float(v[0])
    if t >= float(c[-1]):
        return float(v[-1])
    k = int(torch.searchsorted(c, torch.tensor(t, dtype=c.dtype)).item())
    k = max(1, min(k, len(c) - 1))
    t0, t1 = float(c[k - 1]), float(c[k])
    v0, v1 = float(v[k - 1]), float(v[k])
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


def couette_error(result: WallRunResult) -> dict:
    """L-infinity error of the linear Couette profile ``u = U * Y``."""
    Y, u = result.centreline_u()
    err = float((u - Y).abs().max())
    return {
        "Ny": result.Ny,
        "steps": result.steps,
        "converged": result.converged,
        "linf": err,
        "u_wall_lb": float(u.max()),
        "Y_at_u_max": float(Y[int(torch.argmax(u))]),
    }


def poiseuille_error(result: WallRunResult, body_force: float) -> dict:
    """L-infinity error of the parabolic Poiseuille profile."""
    xc = result.Nx // 2
    ys = torch.arange(1, result.Ny - 1)
    Y = ys.to(torch.float64) - 0.5           # wall planes at 0 and H
    H = float(result.H)
    nu = result.nu
    u_exact = body_force / (2.0 * nu) * Y * (H - Y)
    u_lb = result.ux[ys, xc]
    umax = body_force * H ** 2 / (8.0 * nu)
    return {
        "Ny": result.Ny,
        "steps": result.steps,
        "converged": result.converged,
        "u_max_lb": float(u_lb.max()),
        "u_max_exact": umax,
        "linf_rel": float((u_lb - u_exact).abs().max() / umax),
        "l2_rel": float(torch.sqrt(torch.mean((u_lb - u_exact) ** 2)) / umax),
    }


def ghia_error(result: WallRunResult, re: int) -> dict:
    """L-infinity error of the centreline profiles against Ghia et al. (1982).

    Wall points (``Y = 0`` and ``Y = 1``) are excluded: their values are the
    boundary conditions themselves (``0`` and the lid speed), not predictions.
    """
    Y, u = result.centreline_u()
    X, v = result.centreline_v()

    refu = GHIA_U[re]
    idx_u = [i for i, y in enumerate(refu["y/H"]) if 0.0 < y < 1.0]
    u_err = {refu["y/H"][i]: abs(_lerp_at(Y, u, refu["y/H"][i]) - refu["u/U"][i])
             for i in idx_u}
    refv = GHIA_V[re]
    idx_v = [i for i, x in enumerate(refv["x/H"]) if 0.0 < x < 1.0]
    v_err = {refv["x/H"][i]: abs(_lerp_at(X, v, refv["x/H"][i]) - refv["v/U"][i])
             for i in idx_v}
    return {
        "re": re,
        "N": result.Nx,
        "steps": result.steps,
        "converged": result.converged,
        "u_linf": max(u_err.values()),
        "v_linf": max(v_err.values()),
        "u_err_per_point": u_err,
        "v_err_per_point": v_err,
        "u_min_lb": float(min(_lerp_at(Y, u, y) for y in refu["y/H"][1:-1])),
        "u_min_ghia": min(refu["u/U"]),
        "v_min_lb": float(min(_lerp_at(X, v, x) for x in refv["x/H"][1:-1])),
        "v_min_ghia": min(refv["v/U"]),
        "u_at_y0966_lb": float(_lerp_at(Y, u, 0.9766)),
        "u_at_y0966_ghia": refu["u/U"][1],
    }