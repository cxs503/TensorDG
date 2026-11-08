"""Two-dimensional shallow water equations with tensor-product DG(P1/P2).

Solves the 2-D SWE system with bed slope source term.

Conservative variables U = [h, hu, hv].
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

GRAVITY = 9.81


def physical_flux_2d(state: torch.Tensor, axis: int) -> torch.Tensor:
    """Physical flux for 2-D SWE along given axis."""
    h = state[..., 0].clamp_min(0.0)
    hu = state[..., 1]
    hv = state[..., 2]
    p = 0.5 * GRAVITY * h ** 2

    if axis == 0:
        return torch.stack((hu, hu ** 2 / h.clamp_min(1e-12) + p, hu * hv / h.clamp_min(1e-12)), dim=-1)
    else:
        return torch.stack((hv, hu * hv / h.clamp_min(1e-12), hv ** 2 / h.clamp_min(1e-12) + p), dim=-1)


def sound_speed_2d(state: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(GRAVITY * state[..., 0].clamp_min(1e-12))


def hll_flux_2d(ul: torch.Tensor, ur: torch.Tensor, axis: int) -> torch.Tensor:
    """HLL flux for 2-D SWE."""
    h_l = ul[..., 0].clamp_min(1e-12)
    h_r = ur[..., 0].clamp_min(1e-12)

    normal_idx = 1 if axis == 0 else 2
    u_n_l = ul[..., normal_idx] / h_l
    u_n_r = ur[..., normal_idx] / h_r

    c_l = sound_speed_2d(ul)
    c_r = sound_speed_2d(ur)

    s_l = torch.minimum(u_n_l - c_l, u_n_r - c_r)
    s_r = torch.maximum(u_n_l + c_l, u_n_r + c_r)

    fl = physical_flux_2d(ul, axis)
    fr = physical_flux_2d(ur, axis)

    denom = s_r - s_l
    safe_denom = torch.where(denom.abs() < 1e-30, torch.full_like(denom, 1e-30), denom)
    flux = (s_r[..., None] * fl - s_l[..., None] * fr
            + s_l[..., None] * s_r[..., None] * (ur - ul)) / safe_denom[..., None]

    flux = torch.where((s_l >= 0.0)[..., None], fl, flux)
    flux = torch.where((s_r <= 0.0)[..., None], fr, flux)
    return flux


def legendre_operators_2d(degree: int, device="cpu", dtype=torch.float64):
    """Return 1-D Legendre operators for tensor-product DG."""
    nq = max(2 * degree + 3, 4)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(nq)
    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)

    values, derivatives = [], []
    for k in range(degree + 1):
        poly = np.zeros(k + 1)
        poly[-1] = 1.0
        deriv = np.polynomial.legendre.legder(poly)
        values.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, poly), device=device, dtype=dtype,
        ))
        deriv_vals = (np.polynomial.legendre.legval(nodes_np, deriv)
                      if len(deriv) else np.zeros_like(nodes_np))
        derivatives.append(torch.as_tensor(deriv_vals, device=device, dtype=dtype))

    phi = torch.stack(values, dim=1)
    dphi = torch.stack(derivatives, dim=1)
    phi_left = torch.tensor([(-1.0) ** k for k in range(degree + 1)], device=device, dtype=dtype)
    phi_right = torch.ones(degree + 1, device=device, dtype=dtype)
    return nodes, weights, phi, dphi, phi_left, phi_right


def _rhs_2d(coeff, dx, dy, weights, phi, dphi, phi_left, phi_right,
            zb_jump_x, zb_jump_y, periodic=(False, False)):
    """DG residual for 2-D SWE on Cartesian mesh.

    coeff: (nx, ny, nmodes, nmodes, 3)
    """
    nx, ny, nmodes, _, nvar = coeff.shape

    # Evaluate at tensor-product quadrature
    at_quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, phi, phi)

    # Volume integrals
    vol_x = torch.einsum("xyabv,a,b,ak,bl->xyklv",
                         physical_flux_2d(at_quad, 0), weights, weights, dphi, phi)
    vol_y = torch.einsum("xyabv,a,b,ak,bl->xyklv",
                         physical_flux_2d(at_quad, 1), weights, weights, phi, dphi)

    # X-direction fluxes: trace at x-faces (contract x-modes, evaluate at the
    # y-quadrature points, so that the interface flux can be projected onto the
    # y-modes afterwards).  trace[x, y, b, v] with b = y-quadrature index.
    trace_right_x = torch.einsum("xyklv,k,bl->xybv", coeff, phi_right, phi)
    trace_left_x = torch.einsum("xyklv,k,bl->xybv", coeff, phi_left, phi)

    neighbour_x = torch.roll(trace_left_x, shifts=-1, dims=0)
    if not periodic[0]:
        neighbour_x[-1] = trace_right_x[-1]

    iface_x = hll_flux_2d(trace_right_x, neighbour_x, 0)
    # project the x-interface flux onto the y-modes: [nx, ny, b, v] -> [nx, ny, l, v]
    flux_right_x = torch.einsum("xybv,b,bl->xylv", iface_x, weights, phi)
    flux_left_x = torch.roll(flux_right_x, shifts=1, dims=0)
    if not periodic[0]:
        flux_left_x[0] = torch.einsum("ybv,b,bl->ylv",
                                       hll_flux_2d(trace_left_x[:1], trace_left_x[:1], 0)[0],
                                       weights, phi)

    # Y-direction fluxes: trace at y-faces.  The trace is the pointwise state
    # at (x-quadrature node a, eta = +-1), i.e. *both* the y-mode AND the x-mode
    # are contracted: sigma_{k,l} c_kl phi_k(x_a) phi_l(+-1) -> [x, y, a, v].
    # Keeping the x-mode index in the output (an earlier "xykav" contraction,
    # mission m6386) multiplied every mode by phi_k(x_a) instead of summing the
    # x-modes, which injected a spurious y-momentum source in the cross mode
    # (1,1) of an otherwise y-invariant state and made the solver blow up in a
    # single time step for any unsteady flow.
    trace_right_y = torch.einsum("xyklv,l,ak->xyav", coeff, phi_right, phi)
    trace_left_y = torch.einsum("xyklv,l,ak->xyav", coeff, phi_left, phi)

    neighbour_y = torch.roll(trace_left_y, shifts=-1, dims=1)
    if not periodic[1]:
        neighbour_y[:, -1] = trace_right_y[:, -1]

    iface_y = hll_flux_2d(trace_right_y, neighbour_y, 1)
    # project the y-interface flux onto the x-modes: [nx, ny, a, v] -> [nx, ny, k, v]
    flux_right_y = torch.einsum("xyav,a,ak->xykv", iface_y, weights, phi)
    flux_left_y = torch.roll(flux_right_y, shifts=1, dims=1)
    if not periodic[1]:
        flux_left_y[:, 0] = torch.einsum("xav,a,ak->xkv",
                                          hll_flux_2d(trace_left_y[:, :1], trace_left_y[:, :1], 1)[:, 0],
                                          weights, phi)

    out = torch.empty_like(coeff)
    for k in range(nmodes):
        for l in range(nmodes):
            # d c_kl/dt = (2k+1)(2l+1)/(2 dx) * [x-terms]
            #          + (2k+1)(2l+1)/(2 dy) * [y-terms]
            # The x-terms are already integrated over eta (y-projection) and the
            # y-terms over xi, hence the separate 1/dx and 1/dy scalings: the
            # weak-form factor for direction d is (2k+1)(2l+1)/(2 d_d).
            fx = (2 * k + 1) * (2 * l + 1) / (2.0 * dx)
            fy = (2 * k + 1) * (2 * l + 1) / (2.0 * dy)
            out[:, :, k, l] = fx * (
                vol_x[:, :, k, l] - flux_right_x[:, :, l] * phi_right[k]
                + flux_left_x[:, :, l] * phi_left[k]
            ) + fy * (
                vol_y[:, :, k, l] - flux_right_y[:, :, k] * phi_right[l]
                + flux_left_y[:, :, k] * phi_left[l]
            )
            # Weak-form bed-slope source: the (2k+1)/dx prefactor and the
            # 2/(2k+1) modal moment cancel, leaving -g dbdx * c_kl^h.
            src_x = -GRAVITY * (zb_jump_x / dx) * coeff[:, :, k, l, 0]
            src_y = -GRAVITY * (zb_jump_y / dy) * coeff[:, :, k, l, 0]
            out[:, :, k, l, 1] += src_x
            out[:, :, k, l, 2] += src_y

    return out


def _positivity_filter_2d(coeff, check_points, h_min=1e-12):
    """Ensure h ≥ h_min at all quadrature points."""
    values = torch.einsum("xyklv,ak,bl->xyabv", coeff, check_points, check_points)
    h_vals = values[..., 0]
    valid = (h_vals > h_min) & torch.isfinite(values).all(dim=-1)
    valid_cell = valid.all(dim=(2, 3))

    average_only = torch.zeros_like(coeff)
    average_only[:, :, 0, 0] = coeff[:, :, 0, 0]

    return torch.where(valid_cell[:, :, None, None, None], coeff, average_only)


def solve_swe_2d(
    initial_h: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_u: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_v: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    zb: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    nx: int = 32,
    ny: int = 32,
    degree: int = 1,
    final_time: float = 1.0,
    domain: tuple = ((0.0, 1.0), (0.0, 1.0)),
    cfl: float = 0.15,
    limiter: Optional[str] = None,
    tvb_constant: float = 1.0,
    h_min: float = 1e-12,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    periodic: tuple = (False, False),
) -> tuple:
    """Solve the 2-D shallow water equations with tensor-product DG."""
    if nx < 4 or ny < 4 or degree not in (1, 2):
        raise ValueError("nx, ny >= 4 and degree in {1, 2}")
    if limiter not in (None, "tvb", "weno"):
        raise ValueError("limiter must be None, 'tvb' or 'weno'")

    if limiter is None:
        def _limit(state):
            return state
    else:
        # The tensor-product limiters are axis-generic in the number of
        # conserved variables (they only touch the modal axes), so the 2-D
        # Euler limiters apply to [h, hu, hv] unchanged.  Before mission
        # m6386 the ``limiter`` argument was accepted and silently dropped,
        # so a user asking for a limiter got an unlimited run.
        from .euler_dg_2d import apply_tvb_limiter_2d, apply_weno_limiter_2d

        if limiter == "tvb":
            def _limit(state):
                return apply_tvb_limiter_2d(
                    state, dx, dy, tvb_constant, periodic)
        else:
            def _limit(state):
                return apply_weno_limiter_2d(
                    state, dx, dy, tvb_constant, periodic)

    dev = torch.device(device)
    (xl, xr), (yl, yr) = domain
    dx = (xr - xl) / nx
    dy = (yr - yl) / ny

    x = xl + (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * dx
    y = yl + (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * dy

    # Bed elevation on the cell faces.  Both arguments must be shaped
    # (nx, 1) / (1, ny) so that the result is a genuine (nx, ny) array;
    # broadcasting (nx,) against (ny, 1) silently transposes the bed.
    x_faces = xl + torch.arange(nx + 1, device=dev, dtype=dtype) * dx
    y_faces = yl + torch.arange(ny + 1, device=dev, dtype=dtype) * dy
    zb_lx = zb(x_faces[:-1, None], y[None, :])
    zb_rx = zb(x_faces[1:, None], y[None, :])
    zb_ly = zb(x[:, None], y_faces[:-1][None, :])
    zb_ry = zb(x[:, None], y_faces[1:][None, :])

    zb_jump_x = zb_rx - zb_lx
    zb_jump_y = zb_ry - zb_ly

    nodes, weights, phi, dphi, phi_left, phi_right = legendre_operators_2d(degree, dev, dtype)
    check_points = torch.cat((phi, phi_left[None, :], phi_right[None, :]), dim=0)

    xq = x[:, None] + 0.5 * dx * nodes[None, :]
    yq = y[:, None] + 0.5 * dy * nodes[None, :]

    h0 = initial_h(xq[:, None, :, None], yq[None, :, None, :])
    u0 = initial_u(xq[:, None, :, None], yq[None, :, None, :])
    v0 = initial_v(xq[:, None, :, None], yq[None, :, None, :])
    hu0 = h0 * u0
    hv0 = h0 * v0

    nmodes = degree + 1
    coeff = torch.zeros((nx, ny, nmodes, nmodes, 3), device=dev, dtype=dtype)
    for k in range(nmodes):
        for l in range(nmodes):
            # L2 projection onto the tensor-product Legendre basis
            #   c_kl = ((2k+1)(2l+1)/4) * int f phi_k phi_l  (both basis
            #   functions must be present, otherwise every mode receives the
            #   plain cell average scaled by the (2k+1)(2l+1)/4 factor).
            prefac = ((2 * k + 1) / 2.0) * ((2 * l + 1) / 2.0)
            phi_k, phi_l = phi[:, k], phi[:, l]
            coeff[:, :, k, l, 0] = prefac * torch.einsum(
                "xyab,a,b,a,b->xy", h0, weights, weights, phi_k, phi_l)
            coeff[:, :, k, l, 1] = prefac * torch.einsum(
                "xyab,a,b,a,b->xy", hu0, weights, weights, phi_k, phi_l)
            coeff[:, :, k, l, 2] = prefac * torch.einsum(
                "xyab,a,b,a,b->xy", hv0, weights, weights, phi_k, phi_l)

    coeff = _positivity_filter_2d(_limit(coeff), check_points, h_min)

    time, steps, dt = 0.0, 0, 0.0
    while time < final_time:
        at_quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, phi, phi)
        h_vals = at_quad[..., 0].clamp_min(1e-12)
        u_vals = at_quad[..., 1] / h_vals
        v_vals = at_quad[..., 2] / h_vals
        c_vals = torch.sqrt(GRAVITY * h_vals)
        max_speed = float(torch.max(torch.sqrt(u_vals ** 2 + v_vals ** 2) + c_vals).item())
        dt = min(cfl * min(dx, dy) / ((2 * degree + 1) * max(max_speed, 1e-12)), final_time - time)
        if dt < 1e-15:
            break

        k1 = _rhs_2d(coeff, dx, dy, weights, phi, dphi, phi_left, phi_right,
                     zb_jump_x, zb_jump_y, periodic)
        u1 = coeff + dt * k1
        u1 = _positivity_filter_2d(_limit(u1), check_points, h_min)

        k2 = _rhs_2d(u1, dx, dy, weights, phi, dphi, phi_left, phi_right,
                     zb_jump_x, zb_jump_y, periodic)
        u2 = 0.75 * coeff + 0.25 * (u1 + dt * k2)
        u2 = _positivity_filter_2d(_limit(u2), check_points, h_min)

        k3 = _rhs_2d(u2, dx, dy, weights, phi, dphi, phi_left, phi_right,
                     zb_jump_x, zb_jump_y, periodic)
        coeff = (1.0 / 3.0) * coeff + (2.0 / 3.0) * (u2 + dt * k3)
        coeff = _positivity_filter_2d(_limit(coeff), check_points, h_min)

        if not torch.isfinite(coeff).all():
            raise FloatingPointError("Nonphysical 2D SWE state")

        time += dt
        steps += 1

    zb_faces = {"x": (zb_lx, zb_rx), "y": (zb_ly, zb_ry)}
    metadata = {
        "nx": nx, "ny": ny, "degree": degree, "final_time": final_time,
        "steps": steps, "dt": dt, "domain": domain, "cfl": cfl,
        "limiter": limiter, "device": str(dev),
    }
    return x, y, coeff, zb_faces, metadata
