"""Inviscid Burgers shock capturing with modal DG and shock-capturing limiters.

This module continues the linear-transport (P5) verification line into the
nonlinear conservation law

    u_t + f(u)_x = 0,      f(u) = u^2 / 2,

for which characteristics cross and a genuine discontinuity (a shock) forms.
Unlike :mod:`tensordg.burgers_1d`, which is restricted to the *pre-shock*
smooth regime where the modal solution converges at design order, this module
is built for the post-shock regime and therefore provides

* two interface numerical fluxes for the scalar conservation law -- the
  Lax--Friedrichs/Rusanov flux (global) and the Roe flux (scalar Roe average
  ``a_Roe = (u_L + u_R)/2``) -- so that the shock speed error can be attributed
  to the flux rather than to the scheme as a whole;
* the TVB shock-capturing limiter already verified in :mod:`tensordg.limiters`
  (Cockburn--Shu modified minmod), wired to a modal DG discretisation so that
  the Gibbs oscillation *before* and *after* limiting can be measured on the
  same mesh;
* exact entropy solutions (shock branch and rarefaction branch) for the
  Riemann initial data used by the benchmarks, plus the diagnostics the
  verification needs: shock position by interface interpolation, Gibbs
  overshoot (total variation outside the exact solution's range), total
  variation, and a positivity filter for a passively advected density.

Nothing here is a production multidimensional shock-capturing capability.  The
limiters available through ``limiter=`` are the TVB/minmod one (a troubled cell
is formally reduced to first order) and the linear WENO one; the
WENO-type **high-order** limiter that used to be a roadmap item here now lives in
:mod:`tensordg.limiter_ao` (``limiter="ao"``), which keeps design order in the
limited cells -- verified in ``docs/limiter_ao_verification.md`` and
``tests/test_limiter_ao.py``.

Conventions
-----------
Modal coefficients use the unnormalised Legendre basis on ``xi in [-1, 1]``,
so the physical cell average is mode 0 and ``c_1 = (du/dx) * dx/2``.
Coefficient tensors passed in and out of the public API are
``(ncells, nmodes)``; internally the limiter/positivity helpers see the
``(ncells, nmodes, 1)`` layout they were verified on.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .limiters import apply_limiter

__all__ = [
    "RiemannData",
    "lax_friedrichs_flux",
    "roe_flux",
    "riemann_initial",
    "exact_solution",
    "positivity_filter",
    "gibbs_overshoot",
    "total_variation",
    "shock_position",
    "shock_position_moment",
    "shock_position_from_mass",
    "shock_expected_position",
    "solve_burgers_shock",
    "solve_burgers_tracer",
]


# ---------------------------------------------------------------------------
# Exact solutions
# ---------------------------------------------------------------------------

class RiemannData:
    """Riemann initial data ``u = u_left`` for ``x < x0`` and ``u_right`` otherwise.

    ``u_left > u_right`` produces a shock moving at ``s = (u_left+u_right)/2``;
    ``u_left < u_right`` produces a centred rarefaction.  Both branches are
    handled by :func:`exact_solution`, which is what makes the "wrong branch"
    negative check possible.
    """

    __slots__ = ("u_left", "u_right", "x0")

    def __init__(self, u_left: float = 1.0, u_right: float = 0.0, x0: float = 0.25):
        self.u_left = float(u_left)
        self.u_right = float(u_right)
        self.x0 = float(x0)

    @property
    def shock_speed(self) -> float:
        """Rankine--Hugoniot speed; only meaningful for ``u_left > u_right``."""
        return 0.5 * (self.u_left + self.u_right)

    def shock_ctime(self) -> float:
        """Earliest time at which this data is still a single shock (always 0)."""
        return 0.0

    def shock_location(self, time: float) -> float:
        """Exact shock position ``x0 + s t`` (Rankine--Hugoniot)."""
        return self.x0 + self.shock_speed * time

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"RiemannData(u_left={self.u_left:g}, u_right={self.u_right:g}, "
                f"x0={self.x0:g})")


def riemann_initial(x: torch.Tensor, data: RiemannData) -> torch.Tensor:
    """Piecewise-constant Riemann data sampled at ``x``."""
    return torch.where(
        x < data.x0,
        torch.full_like(x, data.u_left),
        torch.full_like(x, data.u_right),
    )


def exact_solution(x: torch.Tensor, time: float, data: RiemannData) -> torch.Tensor:
    """Exact entropy solution of the Riemann problem for ``u_t + (u^2/2)_x = 0``.

    Shock branch (``u_left > u_right``): ``u = u_left`` left of ``x0 + s t``
    and ``u_right`` right of it.  Rarefaction branch (``u_left < u_right``):
    the fan ``u = (x - x0)/t`` between the two characteristic speeds.
    """
    if time < 0.0:
        raise ValueError("time must be non-negative")
    if time == 0.0:
        return riemann_initial(x, data)
    if data.u_left >= data.u_right:  # shock (equality is a trivial contact)
        return torch.where(
            x < data.shock_location(time),
            torch.full_like(x, data.u_left),
            torch.full_like(x, data.u_right),
        )
    xi = (x - data.x0) / time  # rarefaction fan
    return torch.clamp(
        torch.maximum(torch.minimum(xi, torch.full_like(xi, data.u_right)),
                      torch.full_like(xi, data.u_left)),
        data.u_left, data.u_right,
    )


def shock_expected_position(data: RiemannData, time: float) -> float:
    """Rankine--Hugoniot shock position, rejecting rarefaction data.

    Raises
    ------
    ValueError
        When ``u_left < u_right``: those data have no shock, so a shock-based
        acceptance check would be comparing against the wrong branch.
    """
    _rarefaction_check(data, time)
    return data.shock_location(time)


def _rarefaction_check(data: RiemannData, time: float) -> None:
    if data.u_left < data.u_right:
        raise ValueError(
            "RiemannData with u_left < u_right has no shock; use exact_solution "
            "for the rarefaction branch"
        )
    if time < 0.0:
        raise ValueError("time must be non-negative")


# ---------------------------------------------------------------------------
# Numerical fluxes (scalar conservation law)
# ---------------------------------------------------------------------------

def lax_friedrichs_flux(u_left: torch.Tensor, u_right: torch.Tensor) -> torch.Tensor:
    """Local Lax--Friedrichs (Rusanov) flux with the wave speed of the pair."""
    f_left = 0.5 * u_left ** 2
    f_right = 0.5 * u_right ** 2
    alpha = torch.maximum(u_left.abs(), u_right.abs())
    return 0.5 * (f_left + f_right) - 0.5 * alpha * (u_right - u_left)


def roe_flux(u_left: torch.Tensor, u_right: torch.Tensor) -> torch.Tensor:
    """Roe flux for the scalar Burgers flux.

    For a scalar law the Roe average is the arithmetic mean ``a = (u_L+u_R)/2``
    and the Roe flux is ``(f_L+f_R)/2 - |a| (u_R-u_L)/2``.  Because
    ``f(u_R)-f(u_L) = a (u_R-u_L)`` holds exactly, the flux reduces to a pure
    upwind flux and is consistent by construction.
    """
    f_left = 0.5 * u_left ** 2
    f_right = 0.5 * u_right ** 2
    speed = 0.5 * (u_left + u_right)
    return 0.5 * (f_left + f_right) - 0.5 * speed.abs() * (u_right - u_left)


_FLUXES = {
    "lax_friedrichs": lax_friedrichs_flux,
    "rusanov": lax_friedrichs_flux,
    "roe": roe_flux,
}


# ---------------------------------------------------------------------------
# DG operators and right-hand side
# ---------------------------------------------------------------------------

def _operators(degree: int, device: torch.device, dtype: torch.dtype):
    nq = max(2 * degree + 3, 4)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(nq)
    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
    values, derivatives = [], []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        derivative = np.polynomial.legendre.legder(polynomial)
        values.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, polynomial),
            device=device, dtype=dtype,
        ))
        deriv_values = (
            np.polynomial.legendre.legval(nodes_np, derivative)
            if len(derivative) else np.zeros_like(nodes_np)
        )
        derivatives.append(torch.as_tensor(deriv_values, device=device, dtype=dtype))
    phi = torch.stack(values, dim=1)
    dphi = torch.stack(derivatives, dim=1)
    phi_left = torch.tensor([(-1.0) ** k for k in range(degree + 1)],
                            device=device, dtype=dtype)
    phi_right = torch.ones(degree + 1, device=device, dtype=dtype)
    return nodes, weights, phi, dphi, phi_left, phi_right


def _flux(name: str):
    if name not in _FLUXES:
        raise ValueError(f"unknown flux {name!r}; choose from {sorted(_FLUXES)}")
    return _FLUXES[name]


def _rhs(
    u: torch.Tensor,
    density: torch.Tensor | None,
    dx: float,
    weights: torch.Tensor,
    phi: torch.Tensor,
    dphi: torch.Tensor,
    phi_left: torch.Tensor,
    phi_right: torch.Tensor,
    flux_name: str,
    periodic: bool,
    bc_left: float | None = None,
    bc_right: float | None = None,
):
    """Weak DG operator for Burgers plus an optional passively advected density.

    ``u`` and ``density`` have shape ``(ncells, nmodes, 1)``.  For the
    non-periodic case ``bc_left``/``bc_right`` prescribe ghost-cell states at
    the two boundary interfaces, which turns the inflow boundary into an exact
    Dirichlet condition: the Roe flux then collapses to the physical inflow
    flux and the discrete mass becomes conserved to round-off.  With
    ``bc_left is None`` the boundary flux is the physical flux of the inner
    trace (first-order outflow/inflow), which is only first-order consistent
    when the numerical trace oscillates at the boundary.
    """
    flux = _flux(flux_name)
    at_quadrature = torch.einsum("cmv,qm->cqv", u, phi)   # [c, q, 1]
    volume_u = torch.einsum("cqv,q,qk->ckv", 0.5 * at_quadrature ** 2, weights, dphi)

    u_left_trace = torch.einsum("cmv,m->cv", u, phi_left)
    u_right_trace = torch.einsum("cmv,m->cv", u, phi_right)
    neighbour_left_trace = torch.roll(u_left_trace, shifts=-1, dims=0)
    flux_right = flux(u_right_trace, neighbour_left_trace)
    flux_left = torch.roll(flux_right, shifts=1, dims=0)
    if not periodic:
        flux_left[0] = (
            0.5 * u_left_trace[0] ** 2 if bc_left is None
            else flux(torch.full_like(u_left_trace[:1], bc_left), u_left_trace[:1])
        )
        flux_right[-1] = (
            0.5 * u_right_trace[-1] ** 2 if bc_right is None
            else flux(u_right_trace[-1:], torch.full_like(u_right_trace[:1], bc_right))
        )

    out_u = torch.empty_like(u)
    for k in range(u.shape[1]):
        out_u[:, k] = ((2 * k + 1) / dx) * (
            volume_u[:, k] - flux_right * phi_right[k] + flux_left * phi_left[k]
        )

    if density is None:
        return out_u, None

    density_q = torch.einsum("cmv,qm->cqv", density, phi)
    volume_d = torch.einsum("cqv,q,qk->ckv", at_quadrature * density_q, weights, dphi)
    d_left_trace = torch.einsum("cmv,m->cv", density, phi_left)
    d_right_trace = torch.einsum("cmv,m->cv", density, phi_right)
    d_neighbour = torch.roll(d_left_trace, shifts=-1, dims=0)
    alpha = torch.maximum(u_right_trace.abs(), neighbour_left_trace.abs())
    g_right = (
        0.5 * (u_right_trace * d_right_trace + neighbour_left_trace * d_neighbour)
        - 0.5 * alpha * (d_neighbour - d_right_trace)
    )
    g_left = torch.roll(g_right, shifts=1, dims=0)
    if not periodic:
        # Zero-gradient density ghost: the tracer enters and leaves with the
        # density the boundary cell already carries, so no spurious density
        # flux is created at the inflow/outflow faces.
        u_left_bc = bc_left if bc_left is not None else float(u_left_trace[0].item())
        u_right_bc = bc_right if bc_right is not None else float(u_right_trace[-1].item())
        g_left[0] = 0.5 * (u_left_trace[0] + u_left_bc) * d_left_trace[0]
        g_right[-1] = 0.5 * (u_right_trace[-1] + u_right_bc) * d_right_trace[-1]
    out_d = torch.empty_like(density)
    for k in range(density.shape[1]):
        out_d[:, k] = ((2 * k + 1) / dx) * (
            volume_d[:, k] - g_right * phi_right[k] + g_left * phi_left[k]
        )
    return out_u, out_d


# ---------------------------------------------------------------------------
# Positivity / admissibility
# ---------------------------------------------------------------------------

def positivity_filter(
    coeff: torch.Tensor,
    check_points: torch.Tensor,
    floor: float = 0.0,
) -> torch.Tensor:
    """Revert inadmissible cells to their cell average.

    A cell is admissible when every quadrature point (including both traces)
    satisfies ``value > floor`` and the coefficients are finite.  Inadmissible
    cells keep mode 0 (mass conserving) and lose the higher modes -- the same
    construction used by the Euler DG positivity fallback, applied here to a
    passively advected density that must stay strictly positive.

    ``coeff`` has shape ``(ncells, nmodes, nvar)`` or ``(ncells, nmodes)``;
    the trailing variable axis is added if missing, so the function can be
    unit-tested on a single scalar field.
    """
    squeezed = coeff.dim() == 2
    work = coeff.unsqueeze(-1) if squeezed else coeff
    values = torch.einsum("cmv,qm->cqv", work, check_points)   # [c, q, nvar]
    admissible = (values > floor).all(dim=-1).all(dim=-1)
    admissible &= torch.isfinite(work).all(dim=-1).all(dim=-1)
    average_only = torch.zeros_like(work)
    average_only[:, 0] = work[:, 0]
    filtered = torch.where(admissible[:, None, None], work, average_only)
    return filtered.squeeze(-1) if squeezed else filtered


def _require_positive_density(density: torch.Tensor, floor: float = 0.0) -> None:
    """Reject a negative (or non-finite) density state before time stepping."""
    if not torch.isfinite(density).all():
        raise ValueError("initial density state contains non-finite values")
    if (density <= floor).any():
        raise ValueError(
            "negative density state rejected: the advected density must be "
            f"strictly positive everywhere (floor={floor:g}, "
            f"minimum={float(density.min().item()):.6g})"
        )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def gibbs_overshoot(
    coeff: torch.Tensor,
    phi: torch.Tensor,
    u_min: float,
    u_max: float,
) -> float:
    """Largest excursion of the numerical solution outside ``[u_min, u_max]``.

    The exact entropy solution of a Riemann problem is bounded by its two
    states, so any value outside that range is a Gibbs oscillation.  The
    excursion is measured at Gauss quadrature nodes (the same nodes the weak
    form integrates on), which makes it a conservative lower bound on the
    polynomial maximum but an unambiguous, mesh-comparable metric.
    """
    work = coeff.unsqueeze(-1) if coeff.dim() == 2 else coeff
    values = torch.einsum("cmv,qm->cqv", work, phi)
    upper = (values - u_max).clamp_min(0.0).max().item()
    lower = (u_min - values).clamp_min(0.0).max().item()
    return float(max(upper, lower))


def total_variation(coeff: torch.Tensor, phi: torch.Tensor, dx: float) -> float:
    """Total variation of the piecewise-polynomial solution, node based."""
    work = coeff.unsqueeze(-1) if coeff.dim() == 2 else coeff
    values = torch.einsum("cmv,qm->cqv", work, phi)     # [c, q, 1]
    values = values.squeeze(-1)                          # [c, q]
    nodes_np = np.polynomial.legendre.leggauss(values.shape[1])[0]
    # map Gauss nodes to cell-local physical coordinates and flatten
    ncells, nq = values.shape
    local = torch.as_tensor(nodes_np, dtype=values.dtype, device=values.device)
    x = (torch.arange(ncells, device=values.device, dtype=values.dtype)[:, None]
         + 0.5 * (local[None, :] + 1.0)) * dx
    order = torch.argsort(x.flatten())
    flat = values.flatten()[order]
    return float((flat[1:] - flat[:-1]).abs().sum().item())


def shock_position(
    x: torch.Tensor,
    coeff: torch.Tensor,
    u_left: float,
    u_right: float,
) -> float:
    """Locate the captured shock by interpolating the steepest cell interface.

    The interface with the largest cell-average jump is taken to be the shock
    interface; the (u_left+u_right)/2 level crossing inside that interface is
    interpolated linearly in ``x``.  This returns the position where the
    entropy solution has its jump, so it converges to the Rankine--Hugoniot
    location under refinement without any assumed wave speed.
    """
    average = coeff if coeff.dim() == 1 else coeff[:, 0]
    jumps = (average[1:] - average[:-1]).abs()
    index = int(torch.argmax(jumps).item())
    middle = 0.5 * (u_left + u_right)
    a = float(average[index].item())
    b = float(average[index + 1].item())
    if b != a:
        fraction = min(max((middle - a) / (b - a), 0.0), 1.0)
    else:  # pragma: no cover - degenerate data
        fraction = 0.5
    return float(x[index].item() + fraction * (x[index + 1].item() - x[index].item()))


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------

def _run(
    coeff_u: torch.Tensor,
    coeff_d: torch.Tensor | None,
    dx: float,
    degree: int,
    final_time: float,
    cfl: float,
    flux_name: str,
    limiter: str | None,
    tvb_constant: float,
    periodic: bool,
    limiter_stage: str,
    density_floor: float,
    ops,
    bc_left: float | None = None,
    bc_right: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, int, float]:
    _, weights, phi, dphi, phi_left, phi_right = ops
    check_points = torch.cat((phi, phi_left[None, :], phi_right[None, :]), dim=0)
    limit_each_stage = limiter_stage == "every"

    def limit(c):
        return apply_limiter(c, dx, kind=limiter, tvb_constant=tvb_constant)

    time, steps, dt = 0.0, 0, 0.0
    while time < final_time:
        at_quadrature = torch.einsum("cmv,qm->cqv", coeff_u, phi)
        max_speed = float(at_quadrature.abs().max().item())
        if not math.isfinite(max_speed):
            raise FloatingPointError("non-finite Burgers state before a stage")
        dt = min(cfl * dx / ((2 * degree + 1) * max(max_speed, 1e-12)),
                 final_time - time)

        k1_u, k1_d = _rhs(coeff_u, coeff_d, dx, weights, phi, dphi, phi_left,
                          phi_right, flux_name, periodic, bc_left, bc_right)
        u1 = coeff_u + dt * k1_u
        d1 = None if coeff_d is None else coeff_d + dt * k1_d
        if limit_each_stage:
            u1 = limit(u1)
            d1 = None if d1 is None else limit(d1)
        if d1 is not None:
            d1 = positivity_filter(d1, check_points, floor=density_floor)

        k2_u, k2_d = _rhs(u1, d1, dx, weights, phi, dphi, phi_left, phi_right,
                          flux_name, periodic, bc_left, bc_right)
        u2 = 0.75 * coeff_u + 0.25 * (u1 + dt * k2_u)
        d2 = None if coeff_d is None else 0.75 * coeff_d + 0.25 * (d1 + dt * k2_d)
        if limit_each_stage:
            u2 = limit(u2)
            d2 = None if d2 is None else limit(d2)
        if d2 is not None:
            d2 = positivity_filter(d2, check_points, floor=density_floor)

        k3_u, k3_d = _rhs(u2, d2, dx, weights, phi, dphi, phi_left, phi_right,
                          flux_name, periodic, bc_left, bc_right)
        coeff_u = limit((1.0 / 3.0) * coeff_u + (2.0 / 3.0) * (u2 + dt * k3_u))
        if coeff_d is not None:
            coeff_d = (1.0 / 3.0) * coeff_d + (2.0 / 3.0) * (d2 + dt * k3_d)
            coeff_d = limit(coeff_d)
            coeff_d = positivity_filter(coeff_d, check_points, floor=density_floor)

        if not torch.isfinite(coeff_u).all():
            raise FloatingPointError("non-finite Burgers state after an RK stage")
        time += dt
        steps += 1
    return coeff_u, coeff_d, steps, dt


def _prepare(
    ncells: int,
    degree: int,
    data: RiemannData,
    lower: float,
    upper: float,
    initial,
    device: str,
    dtype: torch.dtype,
):
    """Build the initial modal coefficients as *cell averages*.

    Pointwise sampling at cell centres would bias the initial mass whenever a
    discontinuity is aligned with a centre (exactly half a cell of mass in the
    worst case), which would then be charged to the scheme as a conservation
    error.  The Riemann data is therefore averaged analytically, and a callable
    ``initial`` profile is integrated with Gauss quadrature (exact for
    polynomials up to degree ``2*nq-1``).
    """
    if ncells < 2:
        raise ValueError("ncells must be at least 2")
    if degree not in (0, 1, 2, 3):
        raise ValueError("degree must be one of 0, 1, 2, 3")
    dev = torch.device(device)
    dx = (upper - lower) / ncells
    x = lower + (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * dx
    if initial is None:
        left_edge = x - 0.5 * dx
        right_edge = x + 0.5 * dx
        fraction_left = ((data.x0 - left_edge) / dx).clamp(0.0, 1.0)
        average = data.u_left * fraction_left + data.u_right * (1.0 - fraction_left)
        average = torch.where(right_edge <= data.x0,
                              torch.full_like(x, data.u_left), average)
        average = torch.where(left_edge >= data.x0,
                              torch.full_like(x, data.u_right), average)
    else:
        profile = initial
        if not callable(profile):
            average = torch.as_tensor(profile, device=dev, dtype=dtype)
            if average.numel() != ncells:
                raise ValueError("initial profile must return ncells values")
            coeff = torch.zeros((ncells, degree + 1, 1), device=dev, dtype=dtype)
            coeff[:, 0, 0] = average
            return x, dx, coeff, dev
        nodes_np, weights_np = np.polynomial.legendre.leggauss(max(2 * degree + 3, 4))
        nodes = torch.as_tensor(nodes_np, device=dev, dtype=dtype)
        weights = torch.as_tensor(weights_np, device=dev, dtype=dtype)
        xq = x[:, None] + 0.5 * dx * nodes[None, :]
        values = torch.as_tensor(profile(xq), device=dev, dtype=dtype)
        if values.ndim == 1:  # profile given as cell centres only -> averages
            coeff = torch.zeros((ncells, degree + 1, 1), device=dev, dtype=dtype)
            coeff[:, 0, 0] = values
            return x, dx, coeff, dev
        # full modal L2 projection c_k = (2k+1)/2 * int u phi_k dxi
        coeff = torch.zeros((ncells, degree + 1, 1), device=dev, dtype=dtype)
        for k in range(degree + 1):
            polynomial = np.zeros(k + 1)
            polynomial[-1] = 1.0
            phi_k = torch.as_tensor(np.polynomial.legendre.legval(nodes_np, polynomial),
                                    device=dev, dtype=dtype)
            coeff[:, k, 0] = 0.5 * (2 * k + 1) * (values * weights[None, :]
                                                  * phi_k[None, :]).sum(dim=1)
        return x, dx, coeff, dev
    coeff = torch.zeros((ncells, degree + 1, 1), device=dev, dtype=dtype)
    coeff[:, 0, 0] = average
    return x, dx, coeff, dev


def solve_burgers_shock(
    ncells: int = 200,
    degree: int = 2,
    final_time: float = 0.5,
    cfl: float = 0.2,
    flux: str = "lax_friedrichs",
    limiter: str | None = None,
    tvb_constant: float = 1.0,
    data: RiemannData | None = None,
    periodic: bool = False,
    limiter_stage: str = "every",
    lower: float = 0.0,
    upper: float = 1.0,
    initial=None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    """Solve ``u_t + (u^2/2)_x = 0`` with modal DG on ``[lower, upper]``.

    Parameters
    ----------
    flux : {"lax_friedrichs", "rusanov", "roe"}
        Interface numerical flux.
    limiter : {None, "tvb", "weno", "ao"}
        Shock-capturing limiter applied after every SSP-RK stage; ``None``
        leaves the solution unlimited (the Gibbs oscillation reference).
        ``"ao"`` is the high-order WENO limiter of :mod:`tensordg.limiter_ao`,
        which preserves design order inside the limited cells.
    data : RiemannData
        Initialisation when ``initial`` is ``None``.

    Returns
    -------
    x : (ncells,) cell centres
    coeff : (ncells, degree+1) modal coefficients of ``u``
    metadata : dict with step count, dt, flux, limiter and the shock position
    """
    data = data if data is not None else RiemannData()
    x, dx, coeff, dev = _prepare(ncells, degree, data, lower, upper, initial,
                                    device, dtype)
    ops = _operators(degree, dev, dtype)
    bc_left, bc_right = _boundary_states(data, periodic)
    coeff, _, steps, dt = _run(coeff, None, dx, degree, final_time, cfl,
                               _canonical_flux(flux), limiter, tvb_constant,
                               periodic, limiter_stage, 0.0, ops, bc_left, bc_right)
    coeff2 = coeff[:, :, 0]
    metadata = {
        "ncells": ncells, "degree": degree, "final_time": final_time,
        "cfl": cfl, "steps": steps, "dt": dt, "flux": _canonical_flux(flux),
        "limiter": limiter, "tvb_constant": tvb_constant, "periodic": periodic,
        "limiter_stage": limiter_stage if limiter else None,
        "dx": dx, "domain": (lower, upper), "device": str(dev),
        "riemann": {"u_left": data.u_left, "u_right": data.u_right, "x0": data.x0},
    }
    if not periodic and data.u_left > data.u_right:
        exact_position = data.shock_location(final_time)
        metadata["shock_speed_exact"] = data.shock_speed
        metadata["shock_position_exact"] = exact_position
        # three independent locators: level crossing on cell averages,
        # wave-front centroid of |du/dx|, and the Rankine--Hugoniot position
        # implied by the discrete conserved mass.
        metadata["shock_position"] = shock_position(x, coeff2, data.u_left, data.u_right)
        metadata["shock_position_moment"] = shock_position_moment(
            x, coeff2, ops[3], dx)
        metadata["shock_position_from_mass"] = shock_position_from_mass(
            coeff2, dx, data.u_left, data.u_right, lower, upper)
        for key in ("shock_position", "shock_position_moment",
                    "shock_position_from_mass"):
            metadata[key + "_error"] = abs(metadata[key] - exact_position)
        metadata["conservation_defect"] = float(
            (coeff2[:, 0].sum() * dx).item()
            - (data.u_left * (exact_position - lower)
               + data.u_right * (upper - exact_position)))
    return x, coeff2, metadata


def solve_burgers_tracer(
    ncells: int = 200,
    degree: int = 2,
    final_time: float = 0.5,
    cfl: float = 0.2,
    flux: str = "lax_friedrichs",
    limiter: str | None = None,
    tvb_constant: float = 1.0,
    data: RiemannData | None = None,
    density=lambda x: torch.ones_like(x),
    periodic: bool = False,
    lower: float = 0.0,
    upper: float = 1.0,
    initial=None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    """Burgers velocity plus a passively advected density ``rho_t + (u rho)_x = 0``.

    The density carries the admissibility (positivity) requirement that the
    scalar velocity does not: a negative density state is rejected up front
    (:func:`positivity_filter` then keeps it non-negative during the run by
    reverting offending cells to their cell average).

    Returns ``(x, coeff_u, coeff_density, metadata)``.
    """
    data = data if data is not None else RiemannData()
    x, dx, coeff, dev = _prepare(ncells, degree, data, lower, upper, initial,
                                    device, dtype)
    density_avg = torch.as_tensor(
        density(x) if callable(density) else density, device=dev, dtype=dtype)
    if density_avg.numel() != ncells:
        raise ValueError("density must return ncells values")
    _require_positive_density(density_avg)
    coeff_d = torch.zeros_like(coeff)
    coeff_d[:, 0, 0] = density_avg
    ops = _operators(degree, dev, dtype)
    bc_left, bc_right = _boundary_states(data, periodic)
    coeff, coeff_d, steps, dt = _run(coeff, coeff_d, dx, degree, final_time, cfl,
                                     _canonical_flux(flux), limiter, tvb_constant,
                                     periodic, "every", 0.0, ops, bc_left, bc_right)
    metadata = {
        "ncells": ncells, "degree": degree, "final_time": final_time, "cfl": cfl,
        "steps": steps, "dt": dt, "flux": _canonical_flux(flux), "limiter": limiter,
        "tvb_constant": tvb_constant, "periodic": periodic, "dx": dx,
        "density_mass_initial": float(density_avg.sum().item() * dx),
        "density_mass_final": float(coeff_d[:, 0, 0].sum().item() * dx),
        "density_min": float(torch.einsum("cmv,qm->cqv", coeff_d, ops[2]).min().item()),
    }
    return x, coeff[:, :, 0], coeff_d[:, :, 0], metadata


def _boundary_states(data: RiemannData, periodic: bool):
    """Ghost states imposed at the two boundaries (None when periodic)."""
    if periodic:
        return None, None
    return data.u_left, data.u_right


def shock_position_moment(
    x: torch.Tensor,
    coeff: torch.Tensor,
    dphi: torch.Tensor,
    dx: float,
    ncells_window: int = 2,
) -> float:
    """Wave-front centroid: first moment of ``|du/dx|`` around the steepest interface.

    A captured shock is a smooth ramp of width ``O(dx)``; its centroid
    ``int x |u_x| dx / int |u_x| dx`` converges at first order and -- unlike a
    level-crossing estimate on cell averages -- is insensitive to the ramp
    being nearly flat at the mid level on a coarse mesh.  The window is the
    ``ncells_window`` cells on each side of the steepest cell-average jump,
    which excludes the (small) solution variation in the plateaus.
    """
    average = coeff if coeff.dim() == 1 else coeff[:, 0]
    jumps = (average[1:] - average[:-1]).abs()
    steepest = int(torch.argmax(jumps).item())
    lo = max(steepest - ncells_window, 0)
    hi = min(steepest + ncells_window + 2, coeff.shape[0])
    derivative = torch.einsum("cmv,qm->cqv", coeff.unsqueeze(-1), dphi)[lo:hi, :, 0]
    nq = derivative.shape[1]
    local = torch.as_tensor(np.polynomial.legendre.leggauss(nq)[0],
                            dtype=derivative.dtype, device=derivative.device)
    xq = ((torch.arange(lo, hi, device=derivative.device,
                        dtype=derivative.dtype)[:, None] + 0.5 * (local[None, :] + 1.0))
          * dx)
    weight = derivative.abs()
    total = weight.sum()
    if float(total.item()) == 0.0:  # pragma: no cover - degenerate data
        return float(x[steepest].item())
    return float((xq * weight).sum().item() / total.item())


def shock_position_from_mass(
    coeff: torch.Tensor,
    dx: float,
    u_left: float,
    u_right: float,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    """Shock position implied by the discrete conserved mass (Rankine--Hugoniot).

    For single-shock Riemann data the exact solution carries the mass
    ``u_left*(xs-lower) + u_right*(upper-xs)``, so inverting the discrete mass
    ``sum_i u_i dx`` gives the shock location implied by conservation.  This is
    the sharpest available check of the Rankine--Hugoniot speed: any error in
    the numerical shock speed shows up here as a mass defect.
    """
    mass = float(coeff[:, 0].sum().item()) * dx
    if u_left == u_right:  # pragma: no cover - degenerate data
        raise ValueError("u_left == u_right carries no shock")
    xs = (mass - u_right * (upper - lower)) / (u_left - u_right) + lower
    return float(xs)


def _canonical_flux(name: str) -> str:
    if name == "rusanov":
        return "lax_friedrichs"
    _flux(name)  # validate
    return name