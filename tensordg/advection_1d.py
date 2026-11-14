"""1-D periodic linear-advection DG benchmark using PyTorch tensors.

Modal Legendre basis, upwind numerical flux, and classical RK4 time stepping.
The spatial operator and solution state are evaluated with torch; NumPy is used
only to generate fixed Gauss-Legendre quadrature nodes during setup.

The operators are *degree-generic* (they are built from Legendre values and
derivatives at Gauss nodes), so the solver accepts any degree in
``0 <= degree <= MAX_DEGREE``.  Historically the entry point rejected
``degree > 3`` for no mathematical reason, which silently capped every
verification sweep at P3; that guard is now an explicit, documented bound.
Likewise the initial condition used to be hard-wired to ``sin(2*pi*x)``; it can
now be supplied as a callable, which is what makes discontinuous-data
(Cockburn-Shu style) verification possible at all.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import torch

from .limiters import apply_limiter
from .runge_kutta import TABLEAUS, rk_step

# Tableaus driven by the generic stage loop; ``"rk4"`` and ``"ssp_rk3"``
# keep their original hand-coded branches (bit-for-bit unchanged).
_GENERIC_TABLEAUS = ("euler", "rk2", "rk3", "ssp_rk54", "rk5")

# The quadrature order is ``max(2*degree + 3, 4)`` and float64 Legendre
# evaluation is stable far beyond this; the bound only keeps the Python-side
# basis assembly sane.
MAX_DEGREE = 8


def _sine(x: torch.Tensor) -> torch.Tensor:
    """Default initial condition / exact wave: ``sin(2*pi*x)``."""
    return torch.sin(2.0 * math.pi * x)


def _operators(degree: int, device: torch.device, dtype: torch.dtype):
    # Exact for all polynomial products needed by the selected DG degree.
    nq = max(2 * degree + 3, 4)
    x_np, w_np = np.polynomial.legendre.leggauss(nq)
    x = torch.as_tensor(x_np, device=device, dtype=dtype)
    w = torch.as_tensor(w_np, device=device, dtype=dtype)
    # Legendre basis P_0 ... P_p and derivatives with respect to reference xi.
    vals, derivs = [], []
    for k in range(degree + 1):
        coeff = np.zeros(k + 1); coeff[-1] = 1.0
        dcoeff = np.polynomial.legendre.legder(coeff)
        vals.append(torch.as_tensor(np.polynomial.legendre.legval(x_np, coeff), device=device, dtype=dtype))
        derivs.append(torch.as_tensor(np.polynomial.legendre.legval(x_np, dcoeff) if len(dcoeff) else np.zeros_like(x_np), device=device, dtype=dtype))
    phi = torch.stack(vals, dim=1)       # [quadrature, modes]
    dphi = torch.stack(derivs, dim=1)
    phi_l = torch.tensor([(-1.0) ** k for k in range(degree + 1)], device=device, dtype=dtype)
    phi_r = torch.ones(degree + 1, device=device, dtype=dtype)
    # V[i,j] = integral P_j(xi) * d(P_i)/dxi dxi
    volume = torch.einsum('q,qj,qi->ij', w, phi, dphi)
    return x, w, phi, volume, phi_l, phi_r


def _rhs(u: torch.Tensor, h: float, speed: float, volume: torch.Tensor,
         phi_l: torch.Tensor, phi_r: torch.Tensor) -> torch.Tensor:
    # u[cell, mode]. Positive and negative speeds use the corresponding upwind trace.
    right_trace = u @ phi_r
    left_trace = u @ phi_l
    if speed >= 0.0:
        flux_l = torch.roll(right_trace, shifts=1, dims=0)
        flux_r = right_trace
    else:
        flux_l = left_trace
        flux_r = torch.roll(left_trace, shifts=-1, dims=0)
    out = torch.empty_like(u)
    for k in range(u.shape[1]):
        out[:, k] = ((2 * k + 1) / h) * speed * (
            u @ volume[k, :] - flux_r * phi_r[k] + flux_l * phi_l[k]
        )
    return out


def _limit(coeff: torch.Tensor, dx: float, limiter: str | None,
           tvb_constant: float, troubled_policy: str = "tvb",
           weno_variant: str = "js") -> torch.Tensor:
    """Apply a 1-D limiter to the scalar modal state ``[ncells, nmodes]``.

    The limiter kernel works on ``[..., nvar]`` tensors, so the scalar field is
    given a trailing singleton axis.  ``periodic=True`` gives the end cells
    genuine wrap-around neighbours; without it a discontinuity crossing the
    periodic seam would leave cells 0 and N-1 unlimited (and un-limitable).
    """
    if limiter is None:
        return coeff
    limited = apply_limiter(
        coeff.unsqueeze(-1), dx, kind=limiter, tvb_constant=tvb_constant,
        periodic=True, limit_all=(troubled_policy == "all"),
        **({"weno_variant": weno_variant} if limiter == "weno" else {}),
    )
    return limited.squeeze(-1)


def solve_advection(ncells: int = 40, degree: int = 1, final_time: float = 1.0,
                    speed: float = 1.0, cfl: float = 0.08,
                    device: str = "cpu", dtype: torch.dtype = torch.float64,
                    initial: Callable[[torch.Tensor], torch.Tensor] | None = None,
                    limiter: str | None = None, tvb_constant: float = 1.0,
                    time_integrator: str = "rk4", troubled_policy: str = "tvb",
                    weno_variant: str = "js"):
    """Solve u_t + speed*u_x = 0 on [0,1] with periodic BC.

    ``initial`` defaults to ``sin(2*pi*x)`` (the historical benchmark); pass a
    callable to project any other profile (e.g. a square wave) as ``initial(x)``
    with ``x`` in physical coordinates in ``[0, 1]``.

    ``limiter`` enables the Cockburn-Shu shock-capturing path: ``None`` (the
    historical behaviour) leaves the scheme unlimited, ``"tvb"`` applies the
    TVB/minmod limiter (M = ``tvb_constant``, M=0 is the strict TVD variant),
    ``"weno"`` the WENO limiter and ``"ao"`` the high-order WENO limiter of
    :mod:`tensordg.limiter_ao` (design order preserved in the limited cells), all
    after every Runge-Kutta stage.
    ``time_integrator`` selects classical ``"rk4"`` (default, 4th-order but
    *not* strong-stability-preserving) or ``"ssp_rk3"`` (the TVD-RK scheme of
    Cockburn & Shu 1998, required for the TVD property when limiting).  The
    minimal-stage explicit tableaus ``"euler"`` (1), ``"rk2"`` (Heun), ``"rk3"``
    (Kutta), ``"ssp_rk54"`` (Spiteri & Ruuth SSP-RK(5,4), 5 stages / 4th order)
    and ``"rk5"`` (Butcher, 6 stages / 5th order) of
    :mod:`tensordg.runge_kutta` are also accepted.  ``"ssp_rk54"`` is what
    advances a degree-4 rung with the ``p + 1 = 5`` stages used by the RKDG
    accuracy tables (order 5 with 5 explicit stages is impossible -- Butcher
    barrier -- so the column ``RK5DG4`` is five-stage/fourth-order).

    ``weno_variant`` selects the non-linear weights of the WENO limiter
    (``"js"`` Jiang--Shu, the default and historical behaviour, or ``"z"``
    WENO-Z, Borges et al. 2008); see :func:`tensordg.limiters._weno_weights`.

    Returns (cell_centers, modal_coefficients, metadata). The first modal
    coefficient is the cell-average because P0=1 and higher Legendre modes
    have zero mean.
    """
    if ncells < 2 or not 0 <= degree <= MAX_DEGREE:
        raise ValueError(
            f"ncells must be >= 2 and degree must be in [0, {MAX_DEGREE}]"
        )
    if final_time < 0 or cfl <= 0:
        raise ValueError("final_time must be >= 0 and cfl must be > 0")
    if time_integrator not in ("rk4", "ssp_rk3", *_GENERIC_TABLEAUS):
        raise ValueError(
            "time_integrator must be one of "
            f"{('rk4', 'ssp_rk3') + _GENERIC_TABLEAUS}"
        )
    if troubled_policy not in ("tvb", "all"):
        raise ValueError("troubled_policy must be 'tvb' (paper indicator) or 'all' (strict)")
    profile = _sine if initial is None else initial
    dev = torch.device(device)
    h = 1.0 / ncells
    centers = (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * h
    xq, wq, phi, volume, phi_l, phi_r = _operators(degree, dev, dtype)
    # L2 projection of the initial condition into each element's modal space.
    x_phys = centers[:, None] + 0.5 * h * xq[None, :]
    u0 = profile(x_phys)
    coeff = torch.empty((ncells, degree + 1), device=dev, dtype=dtype)
    for k in range(degree + 1):
        coeff[:, k] = (2 * k + 1) * 0.5 * torch.sum(
            wq[None, :] * u0 * phi[:, k][None, :], dim=1
        )
    if final_time == 0 or speed == 0:
        dt_nominal = float("inf")
        steps = 0
    else:
        dt_nominal = cfl * h / (abs(speed) * (2 * degree + 1))
        steps = math.ceil(final_time / dt_nominal)
    dt = final_time / steps if steps else 0.0
    lim = lambda c: _limit(  # noqa: E731
        c, h, limiter, tvb_constant, troubled_policy, weno_variant
    )
    tableau = (TABLEAUS[time_integrator]
               if time_integrator in _GENERIC_TABLEAUS else None)
    for _ in range(steps):
        if tableau is not None:
            coeff = rk_step(
                lambda c: _rhs(c, h, speed, volume, phi_l, phi_r),
                coeff, dt, tableau, project=lim,
            )
            continue
        k1 = _rhs(coeff, h, speed, volume, phi_l, phi_r)
        if time_integrator == "ssp_rk3":
            u1 = lim(coeff + dt * k1)
            k2 = _rhs(u1, h, speed, volume, phi_l, phi_r)
            u2 = lim(0.75 * coeff + 0.25 * (u1 + dt * k2))
            k3 = _rhs(u2, h, speed, volume, phi_l, phi_r)
            coeff = lim((1.0 / 3.0) * coeff + (2.0 / 3.0) * (u2 + dt * k3))
            continue
        u2 = lim(coeff + 0.5 * dt * k1)
        k2 = _rhs(u2, h, speed, volume, phi_l, phi_r)
        u3 = lim(coeff + 0.5 * dt * k2)
        k3 = _rhs(u3, h, speed, volume, phi_l, phi_r)
        u4 = lim(coeff + dt * k3)
        k4 = _rhs(u4, h, speed, volume, phi_l, phi_r)
        coeff = lim(coeff + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4))
    meta = {"ncells": ncells, "degree": degree, "final_time": final_time,
            "speed": speed, "dt": dt, "steps": steps, "device": str(dev),
            "limiter": limiter or "none", "tvb_constant": tvb_constant,
            "time_integrator": time_integrator, "troubled_policy": troubled_policy,
            "rk_order": (tableau.order if tableau is not None else
                         3 if time_integrator == "ssp_rk3" else 4),
            "rk_stages": (tableau.stages if tableau is not None else
                          3 if time_integrator == "ssp_rk3" else 4)}
    return centers, coeff, meta


def _basis(degree: int, nodes, device, dtype):
    basis = []
    for k in range(degree + 1):
        c = np.zeros(k + 1); c[-1] = 1.0
        basis.append(torch.as_tensor(np.polynomial.legendre.legval(nodes, c),
                                     device=device, dtype=dtype))
    return torch.stack(basis, dim=1)


def l2_error(centers: torch.Tensor, coeff: torch.Tensor, final_time: float,
             speed: float = 1.0,
             exact: Callable[[torch.Tensor], torch.Tensor] | None = None) -> float:
    """Quadrature-based L2 error against the exact translated wave.

    ``exact`` defaults to the translated sine wave ``sin(2*pi*(x - speed*t))``.
    """
    degree = coeff.shape[1] - 1
    x_np, w_np = np.polynomial.legendre.leggauss(max(2 * degree + 4, 6))
    xq = torch.as_tensor(x_np, device=coeff.device, dtype=coeff.dtype)
    wq = torch.as_tensor(w_np, device=coeff.device, dtype=coeff.dtype)
    h = 1.0 / centers.numel()
    phi = _basis(degree, x_np, coeff.device, coeff.dtype)
    uh = coeff @ phi.T
    x = centers[:, None] + 0.5 * h * xq[None, :]
    truth = (_sine(x - speed * final_time) if exact is None
             else exact(x - speed * final_time))
    err2 = 0.5 * h * torch.sum(wq[None, :] * (uh - truth) ** 2)
    return float(torch.sqrt(err2).item())


def _split_segments(lo: float, hi: float, jumps: Sequence[float],
                    nsub: int) -> list[tuple[float, float]]:
    """Split [lo, hi] at the interior points of ``jumps`` and into sub-panels."""
    pts = sorted({lo, hi} | {j for j in jumps if lo < j < hi})
    segs: list[tuple[float, float]] = []
    for a, b in zip(pts[:-1], pts[1:]):
        for k in range(nsub):
            segs.append((a + (b - a) * k / nsub, a + (b - a) * (k + 1) / nsub))
    return segs


def _vector_legendre(nodes: np.ndarray, degree: int) -> np.ndarray:
    """Legendre values at arbitrary nodes, shape ``(len(nodes), degree + 1)``."""
    return np.polynomial.legendre.legval(nodes, np.eye(degree + 1)).T


def l1_error(centers: torch.Tensor, coeff: torch.Tensor, final_time: float,
             speed: float = 1.0,
             exact: Callable[[torch.Tensor], torch.Tensor] | None = None,
             jumps: Sequence[float] = (), nsub: int = 4) -> float:
    """Quadrature-based L1 error, robust to discontinuities.

    The quadrature is composite Gauss on each cell, split at every interior
    point of ``jumps`` (the exact jump abscissas of the reference solution,
    modulo the domain) and further subdivided into ``nsub`` panels.  A kinked
    integrand ``|u_h - u_exact|`` is therefore integrated panel-by-panel
    instead of across the kink, which is what makes an observed L1 *order*
    trustworthy: with a plain single Gauss rule per cell the quadrature error
    of the kink swamps the O(h^{p+1/2}) discretisation error being measured.
    """
    degree = coeff.shape[1] - 1
    nq = max(2 * degree + 4, 12)
    x_np, w_np = np.polynomial.legendre.leggauss(nq)
    ncells = centers.numel()
    h = 1.0 / ncells
    lo_all = centers.detach().cpu().numpy() - 0.5 * h
    jump_mod = sorted({float(j) % 1.0 for j in jumps})

    panels: list[tuple[float, float, int]] = []
    for c in range(ncells):
        lo, hi = float(lo_all[c]), float(lo_all[c]) + h
        cuts = sorted({lo, hi} | {j for j in jump_mod if lo < j < hi})
        for a, b in zip(cuts[:-1], cuts[1:]):
            for k in range(nsub):
                panels.append((a + (b - a) * k / nsub, a + (b - a) * (k + 1) / nsub, c))
    a = np.array([q[0] for q in panels])
    b = np.array([q[1] for q in panels])
    cell = np.array([q[2] for q in panels])
    width = b - a
    nodes = 0.5 * (a[:, None] + b[:, None]) + 0.5 * width[:, None] * x_np[None, :]
    centre = lo_all[cell] + 0.5 * h
    mu = (nodes - centre[:, None]) / (0.5 * h)
    vals = _vector_legendre(mu.ravel(), degree).reshape(len(panels), nq, degree + 1)
    cf = coeff.detach().cpu().numpy()
    uh = np.einsum("pqm,pm->pq", vals, cf[cell])
    xa = torch.as_tensor(nodes, dtype=coeff.dtype, device=coeff.device)
    truth = (_sine(xa - speed * final_time) if exact is None
             else exact(xa - speed * final_time))
    diff = np.abs(uh - truth.detach().cpu().numpy())
    return float(np.sum(0.5 * width[:, None] * diff * w_np[None, :]))


def linf_error(centers: torch.Tensor, coeff: torch.Tensor, final_time: float,
               speed: float = 1.0,
               exact: Callable[[torch.Tensor], torch.Tensor] | None = None,
               nsub: int = 8, nq: int = 16) -> float:
    """Discrete L-infinity error (max over a dense sub-cell quadrature net).

    Used for Gibbs-overshoot studies, where the extremum sits inside the cell
    that contains the discontinuity.
    """
    degree = coeff.shape[1] - 1
    x_np, _ = np.polynomial.legendre.leggauss(nq)
    ncells = centers.numel()
    h = 1.0 / ncells
    lo_all = centers.detach().cpu().numpy() - 0.5 * h
    a = (lo_all[:, None] + h * np.arange(nsub)[None, :] / nsub).ravel()
    width = h / nsub
    nodes = 0.5 * (2 * a[:, None] + width) + 0.5 * width * x_np[None, :]
    cell = np.repeat(np.arange(ncells), nsub)
    centre = lo_all[cell] + 0.5 * h
    mu = (nodes - centre[:, None]) / (0.5 * h)
    vals = _vector_legendre(mu.ravel(), degree).reshape(len(a), nq, degree + 1)
    cf = coeff.detach().cpu().numpy()
    uh = np.einsum("pqm,pm->pq", vals, cf[cell])
    xa = torch.as_tensor(nodes, dtype=coeff.dtype, device=coeff.device)
    truth = (_sine(xa - speed * final_time) if exact is None
             else exact(xa - speed * final_time))
    return float(np.max(np.abs(uh - truth.detach().cpu().numpy())))
