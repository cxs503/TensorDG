"""EXPERIMENTAL / NOT VERIFIED -- do not use for production results.

Status (mission m2143): the tensor-product 2-D DG assembly in this module is
NOT verified.  A 2-D/1-D cross-code test (Sod tube extruded to two dimensions)
shows the projected face flux collapsing to zero while the Riemann solver
itself returns the correct interface flux, so the x-direction residual is
identically zero and the scheme does not transport the flow.  See
``docs/verify_2d_euler_dg_attempt.md`` for the full evidence chain and
``benchmarks/diagnose_euler_dg_2d.py`` for a one-second reproduction.
The module is therefore NOT exported from ``tensordg/__init__.py`` and its
regression test is pinned as ``xfail(strict=True)`` in
``tests/test_euler_dg_2d_experimental.py``.

Original intent of the module (kept for the follow-up mission):

Two-dimensional tensor-product modal DG for the compressible Euler equations.

``euler_2d.py`` in this package is the P0 (finite-volume limit) baseline on a
periodic Cartesian mesh.  This module adds the genuinely high-order 2-D
capability: a tensor-product Legendre modal basis of arbitrary degree, an HLLC
normal-direction interface flux (Toro's three-wave solver, the same flux used by
the 1-D high-order solver), SSP-RK3 time stepping, a positivity-preserving
projection fallback and an optional directional minmod (TVB, Cockburn & Shu
1989) shock limiter.

Everything is written with plain PyTorch tensor algebra, so the whole residual
pipeline stays autograd-compatible.

Verification (see ``benchmarks/euler_dg_2d_verify.py``):

* smooth isentropic vortex -> design-order convergence in 2-D,
* Sod tube extruded to 2-D -> agreement with the exact Riemann solution and
  cell-by-cell agreement with the independently written 1-D DG solver,
* genuinely 2-D four-quadrant Riemann problem -> positivity plus mesh
  self-convergence (no closed-form solution exists).
"""
from __future__ import annotations

import numpy as np
import torch

from .euler_1d import GAMMA
from .euler_2d import vortex_primitive
from .limiters import _minmod

__all__ = [
    "solve_euler_dg_2d",
    "solve_euler_dg_2d_sod",
    "solve_euler_dg_2d_riemann",
    "solve_euler_dg_2d_vortex",
    "cell_fields",
    "legendre_operators",
]


# ---------------------------------------------------------------------------
# Basis / quadrature
# ---------------------------------------------------------------------------

def legendre_operators(degree: int, quadrature: int, device="cpu", dtype=torch.float64):
    """Return (nodes, weights, basis, derivative, signs) for a modal DG basis.

    The basis is the Legendre family ``P_m`` on ``[-1, 1]`` normalised by
    ``int_{-1}^{1} P_m P_n dx = 2 / (2m + 1) delta_mn``.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if quadrature < degree + 1:
        raise ValueError("quadrature must be at least degree + 1")
    nodes_np, weights_np = np.polynomial.legendre.leggauss(quadrature)
    basis = np.polynomial.legendre.legvander(nodes_np, degree)  # [q, degree+1]
    derivative = np.zeros_like(basis)
    for mode in range(degree + 1):
        coefficients = np.zeros(mode + 1)
        coefficients[mode] = 1.0
        differentiated = np.polynomial.legendre.legder(coefficients)
        if differentiated.size:
            derivative[:, mode] = np.polynomial.legendre.legval(
                nodes_np, differentiated
            )
    # P_m(+1) = 1 and P_m(-1) = (-1)^m for every Legendre mode.
    signs = np.asarray([(-1.0) ** mode for mode in range(degree + 1)])
    return tuple(
        torch.as_tensor(array, device=device, dtype=dtype)
        for array in (nodes_np, weights_np, basis, derivative, signs)
    )


# ---------------------------------------------------------------------------
# Equation-of-state helpers (four-component 2-D state)
# ---------------------------------------------------------------------------

def _pressure(state: torch.Tensor) -> torch.Tensor:
    rho = state[..., 0].clamp_min(1e-12)
    return (GAMMA - 1.0) * (
        state[..., 3] - 0.5 * (state[..., 1] ** 2 + state[..., 2] ** 2) / rho
    )


def _primitive(state: torch.Tensor):
    rho = state[..., 0].clamp_min(1e-12)
    return rho, state[..., 1] / rho, state[..., 2] / rho, _pressure(state)


def _conservative(rho, velocity_x, velocity_y, pressure):
    energy = pressure / (GAMMA - 1.0) + 0.5 * rho * (
        velocity_x ** 2 + velocity_y ** 2
    )
    return torch.stack((rho, rho * velocity_x, rho * velocity_y, energy), dim=-1)


def _physical_flux(state: torch.Tensor, axis: int) -> torch.Tensor:
    rho, velocity_x, velocity_y, pressure = _primitive(state)
    if axis == 0:
        return torch.stack(
            (
                state[..., 1],
                state[..., 1] * velocity_x + pressure,
                state[..., 1] * velocity_y,
                velocity_x * (state[..., 3] + pressure),
            ),
            dim=-1,
        )
    return torch.stack(
        (
            state[..., 2],
            state[..., 2] * velocity_x,
            state[..., 2] * velocity_y + pressure,
            velocity_y * (state[..., 3] + pressure),
        ),
        dim=-1,
    )


def _hllc(ul: torch.Tensor, ur: torch.Tensor, axis: int) -> torch.Tensor:
    """Three-wave HLLC flux with a Rusanov fallback for degenerate data."""
    normal = 1 if axis == 0 else 2
    rho_l = ul[..., 0].clamp_min(1e-12)
    rho_r = ur[..., 0].clamp_min(1e-12)
    normal_l = ul[..., normal] / rho_l
    normal_r = ur[..., normal] / rho_r
    pressure_l = _pressure(ul).clamp_min(1e-12)
    pressure_r = _pressure(ur).clamp_min(1e-12)
    sound_l = torch.sqrt(GAMMA * pressure_l / rho_l)
    sound_r = torch.sqrt(GAMMA * pressure_r / rho_r)

    speed_l = torch.minimum(normal_l - sound_l, normal_r - sound_r)
    speed_r = torch.maximum(normal_l + sound_l, normal_r + sound_r)
    denominator = rho_l * (speed_l - normal_l) - rho_r * (speed_r - normal_r)
    safe_denominator = torch.where(
        denominator.abs() < 1e-30,
        torch.full_like(denominator, 1e-30),
        denominator,
    )
    speed_middle = (
        pressure_r
        - pressure_l
        + rho_l * normal_l * (speed_l - normal_l)
        - rho_r * normal_r * (speed_r - normal_r)
    ) / safe_denominator

    def star_state(state, rho, normal_velocity, pressure, speed):
        # ``speed - normal_velocity`` is legitimately negative for the left
        # acoustic wave (it equals the negative sound speed), so only the
        # magnitude may be guarded -- the sign must be preserved.
        gap = speed - normal_velocity
        gap = torch.where(
            gap.abs() < 1e-14,
            torch.where(gap < 0.0, -torch.full_like(gap, 1e-14), torch.full_like(gap, 1e-14)),
            gap,
        )
        rho_star = rho * (speed - normal_velocity) / (speed - speed_middle)
        energy_star = rho_star * (
            state[..., 3] / rho
            + (speed_middle - normal_velocity)
            * (speed_middle + pressure / (rho * gap))
        )
        momentum_normal = rho_star * speed_middle
        tangential_velocity = state[..., 2] / rho if axis == 0 else state[..., 1] / rho
        momentum_tangential = rho_star * tangential_velocity
        if axis == 0:
            momentum = torch.stack(
                (rho_star, momentum_normal, momentum_tangential, energy_star), dim=-1
            )
        else:
            momentum = torch.stack(
                (rho_star, momentum_tangential, momentum_normal, energy_star), dim=-1
            )
        return momentum

    flux_l = _physical_flux(ul, axis)
    flux_r = _physical_flux(ur, axis)
    star_l = star_state(ul, rho_l, normal_l, pressure_l, speed_l)
    star_r = star_state(ur, rho_r, normal_r, pressure_r, speed_r)
    star_flux_l = flux_l + speed_l[..., None] * (star_l - ul)
    star_flux_r = flux_r + speed_r[..., None] * (star_r - ur)

    flux = torch.where((speed_l >= 0.0)[..., None], flux_l, star_flux_l)
    flux = torch.where((speed_middle < 0.0)[..., None], star_flux_r, flux)
    flux = torch.where((speed_r <= 0.0)[..., None], flux_r, flux)

    alpha = torch.maximum(normal_l.abs() + sound_l, normal_r.abs() + sound_r)
    rusanov = 0.5 * (flux_l + flux_r) - 0.5 * alpha[..., None] * (ur - ul)
    degenerate = (speed_r - speed_l).abs() < 1e-13
    return torch.where(degenerate[..., None], rusanov, flux)


# ---------------------------------------------------------------------------
# Spatial operator
# ---------------------------------------------------------------------------

def _traces(coeff, basis, signs, axis):
    """Traces at ``xi = +1`` and ``xi = -1`` along ``axis``.

    The trace keeps the *surface* quadrature index of the remaining direction
    so the interface flux can be evaluated pointwise: index ``b`` (y) for
    x-faces and index ``a`` (x) for y-faces.  ``P_m(+1) = 1`` is used
    implicitly, ``P_m(-1) = (-1)**m`` is applied through ``signs``.
    """
    if axis == 0:
        right = torch.einsum("xyklv,bl->xybv", coeff, basis)
        left = torch.einsum("xyklv,k,bl->xybv", coeff, signs, basis)
    else:
        right = torch.einsum("xyklv,ak->xyav", coeff, basis)
        left = torch.einsum("xyklv,l,ak->xyav", coeff, signs, basis)
    return right, left


def _axis_fluxes(coeff, basis, signs, weights, periodic, axis):
    """Integrated interface fluxes ``(F_right, F_left)`` per cell.

    ``F_right[x, y, l, v]`` (x direction) is the numerical flux of the face
    between cell ``x`` and cell ``x+1``, projected onto the test functions of
    the remaining direction.  Non-periodic boundaries use the interior trace on
    both sides, which makes HLLC collapse to the physical flux (outflow).
    """
    trace_right, trace_left = _traces(coeff, basis, signs, axis)
    if axis == 0:
        if periodic:
            neighbour = torch.roll(trace_left, shifts=-1, dims=0)
        else:
            neighbour = trace_left.clone()
            neighbour[-1] = trace_right[-1]
        interface = _hllc(trace_right, neighbour, axis)
        flux_right = torch.einsum("xybv,b,bl->xylv", interface, weights, basis)
        flux_left = torch.roll(flux_right, shifts=1, dims=0)
        if not periodic:
            outer = _hllc(trace_left[:1], trace_left[:1], axis)
            flux_left[0] = torch.einsum("ybv,b,bl->ylv", outer[0], weights, basis)
    else:
        if periodic:
            neighbour = torch.roll(trace_left, shifts=-1, dims=1)
        else:
            neighbour = trace_left.clone()
            neighbour[:, -1] = trace_right[:, -1]
        interface = _hllc(trace_right, neighbour, axis)
        flux_right = torch.einsum("xyav,a,ak->xykv", interface, weights, basis)
        flux_left = torch.roll(flux_right, shifts=1, dims=1)
        if not periodic:
            outer = _hllc(trace_left[:, :1], trace_left[:, :1], axis)
            flux_left[:, 0] = torch.einsum(
                "xav,a,ak->xkv", outer[:, 0], weights, basis
            )
    return flux_right, flux_left


def _rhs(coeff, basis, derivative, weights, signs, dx, dy, periodic):
    """DG residual of the 2-D Euler equations on a Cartesian mesh."""
    degree = coeff.shape[2] - 1
    modes = torch.arange(degree + 1, device=coeff.device, dtype=coeff.dtype)
    outer = torch.outer(2.0 * modes + 1.0, 2.0 * modes + 1.0)

    at_quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    volume_x = torch.einsum(
        "xyabv,a,b,ak,bl->xyklv", _physical_flux(at_quad, 0),
        weights, weights, derivative, basis,
    )
    volume_y = torch.einsum(
        "xyabv,a,b,ak,bl->xyklv", _physical_flux(at_quad, 1),
        weights, weights, basis, derivative,
    )

    right_x, left_x = _axis_fluxes(coeff, basis, signs, weights, periodic[0], 0)
    right_y, left_y = _axis_fluxes(coeff, basis, signs, weights, periodic[1], 1)

    face_x = (
        right_x.unsqueeze(2)
        - left_x.unsqueeze(2) * signs.view(1, 1, -1, 1, 1)
        - volume_x
    )
    face_y = (
        right_y.unsqueeze(3)
        - left_y.unsqueeze(3) * signs.view(1, 1, 1, -1, 1)
        - volume_y
    )
    scale = (outer / (2.0 * dx)).view(1, 1, *outer.shape, 1), (
        outer / (2.0 * dy)
    ).view(1, 1, *outer.shape, 1)
    return -(scale[0] * face_x) - (scale[1] * face_y)


# ---------------------------------------------------------------------------
# Shock limiting and positivity
# ---------------------------------------------------------------------------

def _troubled_1d(averages: torch.Tensor, axis: int, spacing: float, tvb_constant: float):
    """Cockburn-Shu troubled-cell test along one axis on cell averages."""
    if axis == 0:
        centres = averages
        plus = torch.roll(averages, shifts=-1, dims=0)
        minus = torch.roll(averages, shifts=1, dims=0)
    else:
        centres = averages
        plus = torch.roll(averages, shifts=-1, dims=1)
        minus = torch.roll(averages, shifts=1, dims=1)
    delta_left = centres - minus
    delta_right = plus - centres
    variation = torch.maximum(delta_left.abs(), delta_right.abs()).max(dim=-1).values
    minmod = _minmod(delta_left, delta_right)
    not_monotone = ~((minmod - delta_left).abs().max(dim=-1).values < 1e-14)
    troubled = not_monotone & (variation > tvb_constant * spacing * spacing)
    return troubled, minmod


def apply_tvb_limiter_2d(coeff, dx, dy, tvb_constant=1.0):
    """Directional minmod/TVB limiting of a tensor-product modal expansion.

    Troubled cells (non-monotone cell averages with variation above
    ``tvb_constant * spacing**2``) are reduced to a limited linear
    reconstruction along the offending direction; the higher modes of that
    direction are removed.  The cell average is never modified, so
    conservation is preserved exactly.
    """
    if coeff.shape[2] <= 1 and coeff.shape[3] <= 1:
        return coeff
    averages = coeff[..., 0, 0, :]
    limited = coeff.clone()
    for axis, spacing in ((0, dx), (1, dy)):
        troubled, slope = _troubled_1d(averages, axis, spacing, tvb_constant)
        if not troubled.any():
            continue
        mask = troubled.unsqueeze(-1)  # broadcast over the 4 components
        if axis == 0:
            new_slope = 0.5 * slope
            limited[..., 1, 0, :] = torch.where(
                mask, new_slope, limited[..., 1, 0, :]
            )
            if coeff.shape[2] > 2:
                limited[..., 2:, :, :] = torch.where(
                    mask.unsqueeze(2), torch.zeros_like(limited[..., 2:, :, :]),
                    limited[..., 2:, :, :],
                )
        else:
            new_slope = 0.5 * slope
            limited[..., 0, 1, :] = torch.where(
                mask, new_slope, limited[..., 0, 1, :]
            )
            if coeff.shape[3] > 2:
                limited[..., :, 2:, :] = torch.where(
                    mask.unsqueeze(1), torch.zeros_like(limited[..., :, 2:, :]),
                    limited[..., :, 2:, :],
                )
    return limited


def _admissible(state: torch.Tensor) -> torch.Tensor:
    """Element-wise mask: density and pressure positive and state finite."""
    rho = state[..., 0]
    pressure = _pressure(state)
    return (rho > 0.0) & (pressure > 0.0) & torch.isfinite(state).all(-1)


def _all_over_sample_points(mask: torch.Tensor) -> torch.Tensor:
    """Reduce a point mask of shape ``(nx, ny, ...)`` to one flag per cell."""
    return mask.reshape(*mask.shape[:2], -1).all(dim=-1)


def _positivity_filter(coeff, basis, signs):
    """Collapse to the cell average wherever the solution is non-positive.

    Admissibility is tested on the cell average, on every volume quadrature
    node and on both traces in each direction, because the interface flux is
    evaluated from the trace polynomials and would otherwise receive
    non-positive states.
    """
    averages = coeff[..., 0, 0, :]
    bad = ~_all_over_sample_points(_admissible(averages))
    at_quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    bad = bad | ~_all_over_sample_points(_admissible(at_quad))
    for axis in (0, 1):
        trace_right, trace_left = _traces(coeff, basis, signs, axis)
        bad = bad | ~_all_over_sample_points(_admissible(trace_right))
        bad = bad | ~_all_over_sample_points(_admissible(trace_left))
    if bad.any():
        average_only = torch.zeros_like(coeff)
        average_only[..., 0, 0, :] = averages
        mask = bad.view(*bad.shape, *([1] * (coeff.dim() - bad.dim())))
        coeff = torch.where(mask, average_only, coeff)
    return coeff


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _initial_averages(initial, x, y, dx, dy, device, dtype, quadrature=4):
    nodes_np, weights_np = np.polynomial.legendre.leggauss(quadrature)
    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
    xq = x[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    state = _conservative(*initial(xq, yq))
    return 0.25 * torch.sum(
        state * weights[None, None, :, None, None] * weights[None, None, None, :, None],
        dim=(2, 3),
    )


def solve_euler_dg_2d(
    initial,
    ncells=32,
    degree=2,
    domain=((0.0, 1.0), (0.0, 1.0)),
    final_time=0.2,
    cfl=0.2,
    limiter=None,
    tvb_constant=1.0,
    periodic=(True, True),
    quadrature=None,
    device="cpu",
    dtype=torch.float64,
):
    """Advance a 2-D Euler initial condition with modal DG / HLLC / SSP-RK3.

    Parameters
    ----------
    initial : callable
        ``initial(x, y) -> (rho, u, v, p)`` evaluated on tensors of any shape.
    ncells : int or (int, int)
        Cell counts in ``x`` and ``y``.
    degree : int
        Polynomial degree of the tensor-product basis (0, 1, 2, 3 supported).
    limiter : str or None
        ``None`` (positivity fallback only) or ``"tvb"``.
    periodic : (bool, bool)
        Periodicity in ``x`` and ``y``; non-periodic faces are outflow.
    """
    nx, ny = (ncells, ncells) if isinstance(ncells, int) else ncells
    if degree not in (0, 1, 2, 3):
        raise ValueError("degree must be 0, 1, 2 or 3")
    if nx < 1 or ny < 1 or final_time < 0.0 or not 0.0 < cfl <= 0.5:
        raise ValueError("invalid cells, final_time or CFL")
    (xmin, xmax), (ymin, ymax) = domain
    if not xmax > xmin or not ymax > ymin:
        raise ValueError("each domain interval must be increasing")
    dev = torch.device(device)
    order = quadrature if quadrature is not None else max(degree + 2, 3)
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, order, dev, dtype
    )
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    x = xmin + (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * dx
    y = ymin + (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * dy

    averages = _initial_averages(initial, x, y, dx, dy, dev, dtype)
    coeff = torch.zeros((nx, ny, degree + 1, degree + 1, 4), device=dev, dtype=dtype)
    coeff[..., 0, 0, :] = averages

    def impose(state):
        if limiter == "tvb":
            state = apply_tvb_limiter_2d(state, dx, dy, tvb_constant)
        elif limiter is not None:
            raise ValueError("limiter must be None or 'tvb'")
        return _positivity_filter(state, basis, signs)

    time, steps, dt = 0.0, 0, 0.0
    while time < final_time:
        at_quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
        _, velocity_x, velocity_y, pressure = _primitive(at_quad)
        sound = torch.sqrt(GAMMA * pressure.clamp_min(1e-12) / at_quad[..., 0].clamp_min(1e-12))
        speed_x = float(torch.max(velocity_x.abs() + sound).item())
        speed_y = float(torch.max(velocity_y.abs() + sound).item())
        rate = (speed_x / dx + speed_y / dy) * (2 * degree + 1)
        dt = min(cfl / max(rate, 1e-12), final_time - time)
        k1 = _rhs(coeff, basis, derivative, weights, signs, dx, dy, periodic)
        u1 = impose(coeff + dt * k1)
        k2 = _rhs(u1, basis, derivative, weights, signs, dx, dy, periodic)
        u2 = impose(0.75 * coeff + 0.25 * (u1 + dt * k2))
        k3 = _rhs(u2, basis, derivative, weights, signs, dx, dy, periodic)
        coeff = impose(coeff / 3.0 + (2.0 / 3.0) * (u2 + dt * k3))
        if not torch.isfinite(coeff).all():
            raise FloatingPointError("non-finite 2-D DG state; reduce the CFL number")
        rho_now = coeff[..., 0, 0, 0]
        pressure_now = _pressure(coeff[..., 0, 0, :])
        if (rho_now <= 0.0).any() or (pressure_now <= 0.0).any():
            raise FloatingPointError(
                "nonphysical cell average in 2-D DG; reduce the CFL number"
            )
        time += dt
        steps += 1

    metadata = {
        "ncells": (nx, ny),
        "degree": degree,
        "final_time": final_time,
        "steps": steps,
        "dt": dt,
        "cfl": cfl,
        "domain": domain,
        "periodic": tuple(periodic),
        "flux": "hllc",
        "limiter": limiter if limiter else "local positivity fallback",
        "quadrature": order,
        "device": str(dev),
        "dtype": str(dtype).replace("torch.", ""),
        "dofs": nx * ny * (degree + 1) ** 2 * 4,
    }
    return x, y, coeff, metadata


def cell_fields(coeff: torch.Tensor):
    """Return cell-average primitive fields (density, velocities, pressure)."""
    averages = coeff[..., 0, 0, :]
    rho, velocity_x, velocity_y, pressure = _primitive(averages)
    return {
        "density": rho,
        "velocity_x": velocity_x,
        "velocity_y": velocity_y,
        "pressure": pressure,
        "conservative": averages,
    }


# ---------------------------------------------------------------------------
# Named benchmark set-ups
# ---------------------------------------------------------------------------

def solve_euler_dg_2d_vortex(
    ncells=32,
    degree=2,
    final_time=1.0,
    cfl=0.2,
    strength=5.0,
    background_velocity=(1.0, 1.0),
    limiter=None,
    domain=((-5.0, 5.0), (-5.0, 5.0)),
    device="cpu",
    dtype=torch.float64,
):
    """Periodic smooth isentropic vortex on a high-order 2-D DG mesh."""

    def initial(x, y):
        return vortex_primitive(x, y, 0.0, strength, background_velocity)

    return solve_euler_dg_2d(
        initial,
        ncells=ncells,
        degree=degree,
        domain=domain,
        final_time=final_time,
        cfl=cfl,
        limiter=limiter,
        periodic=(True, True),
        device=device,
        dtype=dtype,
    )


def solve_euler_dg_2d_sod(
    ncells=200,
    degree=2,
    final_time=0.2,
    cfl=0.2,
    limiter=None,
    tvb_constant=1.0,
    diaphragm=0.5,
    left_state=(1.0, 0.0, 1.0),
    right_state=(0.125, 0.0, 0.1),
    domain=((0.0, 1.0), (0.0, 0.25)),
    device="cpu",
    dtype=torch.float64,
):
    """Sod tube extruded to 2-D (one cell thick, y-invariant data).

    Because the initial data do not depend on ``y`` and the flux difference
    across the two ``y`` faces cancels exactly, the discrete solution must
    reproduce the 1-D DG solution cell by cell.  This makes the set-up a
    cross-code verification of the 2-D residual assembly.
    """

    def initial(x, y):
        shape = torch.broadcast_shapes(x.shape, y.shape)
        mask = torch.broadcast_to(x < diaphragm, shape)
        rho = torch.where(mask, torch.full(shape, left_state[0], dtype=x.dtype),
                          torch.full(shape, right_state[0], dtype=x.dtype))
        velocity_x = torch.where(mask, torch.full(shape, left_state[1], dtype=x.dtype),
                                 torch.full(shape, right_state[1], dtype=x.dtype))
        pressure = torch.where(mask, torch.full(shape, left_state[2], dtype=x.dtype),
                               torch.full(shape, right_state[2], dtype=x.dtype))
        return rho, velocity_x, torch.zeros_like(rho), pressure

    return solve_euler_dg_2d(
        initial,
        ncells=(ncells, 1),
        degree=degree,
        domain=domain,
        final_time=final_time,
        cfl=cfl,
        limiter=limiter,
        tvb_constant=tvb_constant,
        periodic=(False, True),
        device=device,
        dtype=dtype,
    )


# Four-quadrant Riemann problem, Lax & Liu (1998) configuration 3
# (interaction of two shocks).  Quadrant order: (lower-left, lower-right,
# upper-left, upper-right) with states (rho, u, v, p).
LAX_LIU_CONFIG_3 = {
    "lower-left": (0.138, 1.206, 1.206, 0.3),
    "lower-right": (0.5323, 0.0, 1.206, 0.3),
    "upper-left": (0.5323, 1.206, 0.0, 0.3),
    "upper-right": (1.5, 0.0, 0.0, 1.5),
}


def solve_euler_dg_2d_riemann(
    ncells=100,
    degree=2,
    final_time=0.25,
    cfl=0.2,
    limiter="tvb",
    tvb_constant=1.0,
    states=None,
    device="cpu",
    dtype=torch.float64,
):
    """Four-quadrant 2-D Riemann problem (Lax & Liu 1998, configuration 3)."""
    quadrants = LAX_LIU_CONFIG_3 if states is None else states

    def initial(x, y):
        shape = torch.broadcast_shapes(x.shape, y.shape)
        lower = torch.broadcast_to(y < 0.5, shape)
        left = torch.broadcast_to(x < 0.5, shape)
        chosen = (
            ("lower-left", lower & left),
            ("lower-right", lower & ~left),
            ("upper-left", ~lower & left),
            ("upper-right", ~lower & ~left),
        )
        rho = torch.zeros(shape, dtype=x.dtype)
        velocity_x = torch.zeros(shape, dtype=x.dtype)
        velocity_y = torch.zeros(shape, dtype=x.dtype)
        pressure = torch.zeros(shape, dtype=x.dtype)
        for name, mask in chosen:
            rho = torch.where(mask, torch.full(shape, quadrants[name][0], dtype=x.dtype), rho)
            velocity_x = torch.where(mask, torch.full(shape, quadrants[name][1], dtype=x.dtype), velocity_x)
            velocity_y = torch.where(mask, torch.full(shape, quadrants[name][2], dtype=x.dtype), velocity_y)
            pressure = torch.where(mask, torch.full(shape, quadrants[name][3], dtype=x.dtype), pressure)
        return rho, velocity_x, velocity_y, pressure

    return solve_euler_dg_2d(
        initial,
        ncells=ncells,
        degree=degree,
        domain=((0.0, 1.0), (0.0, 1.0)),
        final_time=final_time,
        cfl=cfl,
        limiter=limiter,
        tvb_constant=tvb_constant,
        periodic=(False, False),
        device=device,
        dtype=dtype,
    )