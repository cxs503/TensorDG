"""Flow past a static immersed-marker cylinder in a channel (D2Q9 BGK).

Purpose
-------
This is the physical probe for the LIFE-style static-marker obstacle set in
:mod:`tensorlbm.immersed`: a circular cylinder is represented **only** by a
ring of Lagrangian markers held at zero velocity; there is no solid mask, no
staircase and no bounce-back link on the body.  The module reports the drag
coefficient ``Cd``, the lift coefficient ``Cl``, the Strouhal number ``St``
and -- because a validation report must not hide it -- the *achieved* no-slip
residual at the markers, the total-mass drift of the open inlet/outlet pair,
and an **independent momentum budget** that replays the same drag from the
inlet/outlet fluxes and the wall momentum exchange.

Numerics
--------
* D2Q9, BGK, double precision, ``u0 = 0.05`` (compressibility error
  ``u0^2/2 ~ 0.1 %``).
* ``Re = u0 D / nu`` with ``nu = (tau - 1/2)/3``.
* Open boundaries: non-equilibrium extrapolation (Guo, Zheng & Lin 2000) on
  both the velocity inlet and the outflow; no-slip half-way bounce-back on
  the two channel walls.
* Body force: Guo, Zheng & Shi (2002) source term fed by the marker traction
  from :func:`tensorlbm.immersed.multi_direct_forcing`.

Momentum budget (independent drag check)
----------------------------------------
For the time-averaged flow the streamwise momentum balance of the box reads

    F_body = (m_out - m_in) - (p_in - p_out) H - F_wall,

with ``m = int rho u_x^2 dy``, ``p = rho/3`` and ``F_wall`` the force the
walls exert on the fluid, measured by the bounce-back momentum exchange
``-sum_links 2 f_i e_ix``.  ``F_body`` from this budget must reproduce the
marker-traction drag ``-sum_m G_x ds``; the relative difference is reported
as :attr:`CylinderResult.momentum_budget_mismatch`.

The bibliography the numbers are compared against lives in
``t3_matrix.CASES["cylinder-2d-re100"]``; this module only *measures*.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .boundaries import HalfwayBounceBack
from .collision import bgk_collision
from .immersed import (
    StaticMarkerObstacles,
    guo_source,
    make_circle_markers,
    multi_direct_forcing,
)
from .lattice import D2Q9


@dataclass
class CylinderResult:
    """Everything a validation report needs, and nothing hand-typed."""

    re: float
    d_cells: int
    u0: float
    nu: float
    tau: float
    nx: int
    ny: int
    channel_height: float
    blockage: float
    steps: int
    sample_every: int
    average_from_step: int
    cd_mean: float
    cd_amp: float
    cl_mean: float
    cl_rms: float
    st: float
    st_zero_crossing: float
    n_periods: int
    marker_stats: Dict[str, float]
    marker_weights: Dict[str, float]
    slip_before: float
    slip_after: float
    slip_rel_to_u0: float
    mass_drift: float
    rho_min: float
    rho_max: float
    momentum_budget: Dict[str, float]
    wall_seconds: float
    history: Dict[str, List[float]] = field(default_factory=dict)
    snapshot: Optional[Dict[str, np.ndarray]] = None

    @property
    def momentum_budget_mismatch(self) -> float:
        return float(self.momentum_budget.get("rel_mismatch", float("nan")))

    @property
    def converged_wake(self) -> bool:
        """At least five lift periods inside the averaging window."""
        return self.n_periods >= 5 and bool(np.isfinite(self.cd_mean))


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


def _feq_1d(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q9 equilibrium for a 1-D slice of nodes (returns ``(9, n)``)."""
    out = torch.empty((D2Q9.Q,) + rho.shape, dtype=rho.dtype)
    usq = ux * ux + uy * uy
    for i in range(D2Q9.Q):
        eu = float(D2Q9.e[i, 0]) * ux + float(D2Q9.e[i, 1]) * uy
        out[i] = rho * float(D2Q9.w[i]) * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)
    return out


def apply_open_boundaries(f: torch.Tensor, u0: float, rho_outlet: float = 1.0) -> None:
    """Velocity inlet / pressure outlet by non-equilibrium extrapolation.

    ``f_b = f^eq(rho_b, u_b) + [f_n - f^eq(rho_n, u_n)]`` (Guo, Zheng & Lin
    2000, J. Comput. Phys. 165, 288), applied in place.

    * inlet: ``u_b = (u0, 0)``, boundary density taken from the first
      interior node (the mass flux is therefore ``rho_in u0``);
    * outlet: ``rho_b = rho_outlet`` (**the pressure anchor**: without it the
      mean density of the open domain floats and every pressure difference is
      meaningless), velocity extrapolated from the interior.

    A velocity inlet paired with a pressure outlet is the well-posed pair;
    extrapolating *both* rho and u on both ends leaves no pressure reference
    and was measured to make ``dp`` drift arbitrarily (see docs, "注意点").
    """
    rho, ux, uy = D2Q9.macroscopic(f)
    rows = slice(1, -1)
    # inlet
    rho_in = rho[rows, 1]
    u0_col = torch.full_like(rho_in, u0)
    zero_col = torch.zeros_like(rho_in)
    f_in_eq = _feq_1d(rho_in, u0_col, zero_col)
    f[:, rows, 0] = f_in_eq + (f[:, rows, 1] - _feq_1d(rho[rows, 1], ux[rows, 1], uy[rows, 1]))
    # outlet (zeroth-order extrapolation of the non-equilibrium part)
    rho_o = torch.full_like(rho[rows, -2], float(rho_outlet))
    ux_o, uy_o = ux[rows, -2], uy[rows, -2]
    f_o_eq = _feq_1d(rho_o, ux_o, uy_o)
    f[:, rows, -1] = f_o_eq + (f[:, rows, -2] - f_o_eq)


def wall_momentum_exchange_x(f_post: torch.Tensor, solid: torch.Tensor) -> float:
    """Streamwise force the walls exert on the fluid, from bounce-back links.

    Each wall link transfers ``2 f_i e_i`` of momentum per step from the fluid
    to the wall (Ladd 1994), so the force **on the fluid** is ``-2 f_i e_ix``.
    """
    from .boundaries import shift

    total = 0.0
    solid_f = solid.to(f_post.dtype)
    for i in range(1, D2Q9.Q):
        ex = int(D2Q9.e[i, 0])
        ey = int(D2Q9.e[i, 1])
        up_solid = shift(solid_f, ey, ex) > 0.5      # out[y, x] = solid[y - ey, x - ex]
        mask = up_solid & ~solid
        if bool(mask.any()):
            total += -2.0 * float(ex) * float(
                torch.where(mask, f_post[i], torch.zeros((), dtype=f_post.dtype)).sum()
            )
    return total


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_cylinder(
    d_cells: int = 32,
    re: float = 100.0,
    u0: float = 0.05,
    channel_diameters: float = 10.0,
    inlet_diameters: float = 6.5,
    length_diameters: float = 20.0,
    marker_spacing: float = 1.0,
    n_steps: int = 6000,
    sample_every: int = 2,
    average_from: float = 0.55,
    epsilon: float = 1.0,
    n_iter: int = 3,
    kernel: str = "4pt",
    symmetry_break: float = 0.1,
    threads: int = 4,
    field_snapshot: bool = True,
) -> CylinderResult:
    """Run the marker-cylinder case and return the measured coefficients."""
    if d_cells < 8:
        raise ValueError("d_cells must be at least 8")
    if not 0.0 < u0 < 0.2:
        raise ValueError("u0 must be a small Mach number (< 0.2)")

    torch.set_num_threads(max(1, int(threads)))
    dtype = torch.float64

    nu = u0 * d_cells / re
    tau = 3.0 * nu + 0.5
    omega = 1.0 / tau

    ny = int(round(channel_diameters * d_cells)) + 2
    nx = int(round(length_diameters * d_cells))
    height = float(ny - 2)
    xc = float(inlet_diameters * d_cells)
    yc = (ny - 1) / 2.0
    radius = d_cells / 2.0

    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True
    bbs = HalfwayBounceBack(solid)

    positions, mstats = make_circle_markers(
        xc, yc, radius, spacing=marker_spacing, min_sep=0.5
    )
    markers = StaticMarkerObstacles(
        shape=(ny, nx),
        positions=positions,
        spacing=mstats["spacing_realised"],
        kernel=kernel,
        mask=(~solid.numpy()),
    )

    rho = torch.ones((ny, nx), dtype=dtype)
    ux = torch.full((ny, nx), u0, dtype=dtype)
    uy = torch.zeros((ny, nx), dtype=dtype)
    if symmetry_break:
        yy, xx = torch.meshgrid(
            torch.arange(ny, dtype=dtype), torch.arange(nx, dtype=dtype), indexing="ij"
        )
        bump = torch.exp(-((xx - xc) ** 2 + (yy - yc) ** 2) / (2.0 * radius**2))
        uy = symmetry_break * u0 * bump
    f = D2Q9.equilibrium(rho, ux, uy)

    start_step = int(round(average_from * n_steps))

    times: List[float] = []
    cds: List[float] = []
    cls: List[float] = []
    budget = {"m_in": 0.0, "m_out": 0.0, "p_in": 0.0, "p_out": 0.0, "f_wall": 0.0,
              "f_body_traction": 0.0}
    n_samples = 0
    slip_before = 0.0
    slip_after = 0.0

    rows = slice(1, -1)
    t_wall0 = _time.time()
    for step in range(1, n_steps + 1):
        rho, ux, uy = D2Q9.macroscopic(f)
        gx, gy, info = multi_direct_forcing(
            markers, rho, ux, uy, u_body=(0.0, 0.0), epsilon=epsilon,
            n_iter=n_iter, gain=1.5 - 0.5 * omega,
        )
        slip_before = info["slip_before"]
        slip_after = info["slip_after"]
        fx, fy = markers.force_density(gx, gy)
        ux_e = ux + fx / (2.0 * rho)
        uy_e = uy + fy / (2.0 * rho)
        f_post = bgk_collision(f, rho, ux_e, uy_e, omega) + guo_source(
            rho, ux_e, uy_e, fx, fy, omega
        )
        f = bbs.stream(f_post)
        apply_open_boundaries(f, u0)

        if not bool(torch.isfinite(f).all()):
            raise RuntimeError(f"diverged at step {step} (non-finite populations)")

        if step >= start_step and (step - start_step) % sample_every == 0:
            bx, by = markers.body_force(gx, gy)
            q = 0.5 * u0 * u0 * float(d_cells)
            times.append(step / float(d_cells))       # in D/U units
            cds.append(bx / q)
            cls.append(by / q)
            rho_s, ux_s, _ = D2Q9.macroscopic(f)
            p_s = rho_s / 3.0
            budget["m_in"] += float((rho_s[rows, 0] * ux_s[rows, 0] ** 2).sum())
            budget["m_out"] += float((rho_s[rows, -1] * ux_s[rows, -1] ** 2).sum())
            budget["p_in"] += float(p_s[rows, 0].sum())
            budget["p_out"] += float(p_s[rows, -1].sum())
            budget["f_wall"] += wall_momentum_exchange_x(f_post, solid)
            budget["f_body_traction"] += bx
            n_samples += 1

    wall_seconds = _time.time() - t_wall0
    if n_samples < 8:
        raise ValueError("averaging window too short: increase n_steps or lower average_from")

    t_arr = np.asarray(times)
    cd_arr = np.asarray(cds)
    cl_arr = np.asarray(cls)
    cd_mean = float(cd_arr.mean())
    cd_amp = float(0.5 * (cd_arr.max() - cd_arr.min()))
    cl_mean = float(cl_arr.mean())
    cl_rms = float(np.sqrt(((cl_arr - cl_mean) ** 2).mean()))

    st_fft, n_periods = _strouhal(t_arr, cl_arr)
    st_zc = _strouhal_zero_crossing(t_arr, cl_arr)

    m_in = budget["m_in"] / n_samples
    m_out = budget["m_out"] / n_samples
    p_in = budget["p_in"] / n_samples
    p_out = budget["p_out"] / n_samples
    f_wall = budget["f_wall"] / n_samples
    f_traction = budget["f_body_traction"] / n_samples
    # F_body = (m_out - m_in) - (p_in - p_out) H - F_wall   (see module docstring)
    f_budget = (m_out - m_in) - (p_in - p_out) * height - f_wall
    denom = abs(f_budget) if abs(f_budget) > 1e-30 else 1.0
    momentum_budget = {
        "m_in": m_in, "m_out": m_out, "p_in": p_in, "p_out": p_out,
        "f_wall": f_wall,
        "f_body_traction": f_traction,
        "f_body_budget": float(f_budget),
        "rel_mismatch": float(abs(f_traction - f_budget) / denom),
        "n_samples": float(n_samples),
        "force_gain": float(1.5 - 0.5 * omega),
    }

    mstats_out = dict(mstats)
    mstats_out["d_cells"] = float(d_cells)
    mstats_out["radius"] = float(radius)
    mstats_out["marker_spacing_requested"] = float(marker_spacing)

    snapshot = None
    if field_snapshot:
        rho_e, ux_e, uy_e = D2Q9.macroscopic(f)
        wx = torch.zeros_like(ux_e)
        wy = torch.zeros_like(ux_e)
        wx[1:-1, 1:-1] = (uy_e[2:, 1:-1] - uy_e[:-2, 1:-1]) / 2.0
        wy[1:-1, 1:-1] = (ux_e[1:-1, 2:] - ux_e[1:-1, :-2]) / 2.0
        vort = (wy - wx)[1:-1, :]
        snapshot = {
            "vorticity": vort.detach().numpy(),
            "ux": ux_e[1:-1, :].detach().numpy(),
            "uy": uy_e[1:-1, :].detach().numpy(),
            "rho": rho_e[1:-1, :].detach().numpy(),
            "marker_x": positions[:, 0],
            "marker_y": positions[:, 1] - 1.0,   # first fluid row is row 1
            "solid": solid.numpy(),
        }

    rho_flat = rho_e[1:-1, 1:-1] if field_snapshot else D2Q9.macroscopic(f)[0][1:-1, 1:-1]
    return CylinderResult(
        re=float(re),
        d_cells=int(d_cells),
        u0=float(u0),
        nu=float(nu),
        tau=float(tau),
        nx=int(nx),
        ny=int(ny),
        channel_height=height,
        blockage=float(d_cells / height),
        steps=int(n_steps),
        sample_every=int(sample_every),
        average_from_step=int(start_step),
        cd_mean=cd_mean,
        cd_amp=cd_amp,
        cl_mean=cl_mean,
        cl_rms=cl_rms,
        st=float(st_fft),
        st_zero_crossing=float(st_zc),
        n_periods=int(n_periods),
        marker_stats=mstats_out,
        marker_weights=markers.weight_summary,
        slip_before=float(slip_before),
        slip_after=float(slip_after),
        slip_rel_to_u0=float(slip_after / u0),
        mass_drift=float(rho_flat.mean() - 1.0),
        rho_min=float(rho_flat.min()),
        rho_max=float(rho_flat.max()),
        momentum_budget=momentum_budget,
        wall_seconds=float(wall_seconds),
        history={"t_over_D": times, "cd": cds, "cl": cls},
        snapshot=snapshot,
    )


def _strouhal(t: np.ndarray, cl: np.ndarray) -> Tuple[float, int]:
    """Dominant Strouhal number from the lift spectrum.

    ``t`` is already in ``D/U`` units, so the dominant non-dimensional
    frequency is ``St`` directly.  Returns ``(St, n_periods_in_window)``.
    """
    x = cl - cl.mean()
    if x.size < 8 or not np.any(x):
        return float("nan"), 0
    dt = float(np.median(np.diff(t)))
    n = x.size
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=dt)
    spec[0] = 0.0
    k = int(np.argmax(spec))
    st = float(freqs[k])
    window = float(t[-1] - t[0])
    n_periods = int(np.floor(window * st)) if st > 0 else 0
    return st, n_periods


def _strouhal_zero_crossing(t: np.ndarray, cl: np.ndarray) -> float:
    """Strouhal from the mean rate of upward zero crossings of ``Cl``."""
    x = cl - cl.mean()
    if x.size < 4:
        return float("nan")
    s = np.sign(x)
    s[s == 0] = 1.0
    crossings = int(np.sum((s[:-1] < 0) & (s[1:] > 0)))
    window = float(t[-1] - t[0])
    if crossings == 0:
        return float("nan")
    return float(crossings / window)