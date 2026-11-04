"""Compressible Navier--Stokes on a 2-D tensor-product modal DG mesh (BR1).

Every solver in ``tensordg`` was inviscid: ``euler_dg_2d.py`` has no viscous
flux, no shear stress and no heat conduction.  This module closes that gap (the
last open entry of the expand-audit backlog, ``expand_audit_tensordg.md`` §6
item 5) on top of the *same* high-order infrastructure:

* the same Legendre modal basis and Gauss quadrature (``legendre_operators``);
* the same HLLC inviscid normal flux, SSP-RK3 staging and positivity fallback.

Viscous discretisation
----------------------
The diffusive terms use the **BR1** (Bassi & Rebay 1989) interface formulation,
which on a Cartesian mesh is the symmetric interior-penalty (SIPG) gradient

    {grad q . n}_f = 1/2 (grad q|-_f + grad q|+_f) . n
                      - penalty / h (q|-_f - q|+_f)

for the primitive variables ``q in {u, v, T}``.  ``n`` is always the ``+axis``
direction: the ``P_k(-1)`` factor of the weak form supplies the left-face
orientation, exactly as in ``euler_dg_2d._rhs``.  Tangential derivatives at a
face are the plain two-sided average.  Dirichlet walls reuse the same formula
with the mirror ghost ``q|+ = q_wall``, ``grad q|+ = grad q|-``, so interior
faces and walls share one code path.

The wall normal derivative is the *one-sided* DG polynomial derivative, i.e.
``grad q|+ = grad q|-`` is evaluated from the interior polynomial itself
(``wall_gradient="face"``, the default).  An O(h^2) two-point extrapolation of
centred derivatives (``wall_gradient="taylor"``) is available but must be
requested explicitly: measured on the exact Brinkman/Couette state it *caps*
the wall heat flux at second order at every degree, degrading P3 by up to 97x
(``validation/ns_dg_2d_wall/bisect_wall_gradient.json``,
``docs/reports/tensordg-ns-wall-gradient-bisect.md``).

Constitutive laws (non-dimensional, unit gas constant ``R = 1``, so ``T = p/rho``
and ``c_v = 1/(gamma-1)`` -- identical to ``euler_dg_2d._pressure``):

    tau_xx = 2 mu du/dx - (2/3) mu (du/dx + dv/dy)
    tau_yy = 2 mu dv/dy - (2/3) mu (du/dx + dv/dy)
    tau_xy = mu (du/dy + dv/dx)
    kappa  = mu c_p / Pr                       (constant Prandtl number)

with constant dynamic viscosity.  The viscous flux
``[0, tau_xn, tau_yn, u tau_xn + v tau_yn + kappa dT/dn]`` is added to the
inviscid HLLC flux, so ``mu = kappa = 0`` degenerates exactly to
``solve_euler_dg_2d`` (verified to 1e-12).

Verification
------------
``benchmarks/verify_ns_dg_2d.py`` (numbers + JSON) and ``tests/test_ns_dg_2d.py``
check the four cases named in the docstring of :func:`solve_navier_stokes_dg_2d`.
Known limitations are listed in ``docs/reports/tensordg-ns-dg-2d.md``.
"""
from __future__ import annotations

import numpy as np
import torch

from .euler_1d import GAMMA
from .euler_dg_2d import (
    _hllc,
    _initial_coefficients,
    _physical_flux,
    _positivity_filter,
    _primitive,
    _traces,
    cell_fields,
    legendre_operators,
)

__all__ = [
    "solve_navier_stokes_dg_2d",
    "solve_couette_dg_2d",
    "solve_heat_conduction_dg_2d",
    "solve_shear_wave_dg_2d",
    "couette_exact",
    "heat_conduction_exact",
    "shear_wave_exact",
    "gas_constants",
]

# Safety factor for the explicit diffusive step (see solve_navier_stokes_dg_2d).
VISCOUS_STABILITY = 0.1

# Unit gas constant (``euler_dg_2d._pressure`` implies p = rho R T with R = 1).
GAS_CONSTANT = 1.0
CV = 1.0 / (GAMMA - 1.0)
CP = GAMMA / (GAMMA - 1.0)


def gas_constants(prandtl: float = 0.71) -> dict:
    """Return the non-dimensional gas constants implied by ``GAMMA``."""
    return {
        "gamma": GAMMA,
        "gas_constant": GAS_CONSTANT,
        "cv": CV,
        "cp": CP,
        "prandtl": prandtl,
    }


# ---------------------------------------------------------------------------
# Basis values at the faces
# ---------------------------------------------------------------------------

def _face_weights(degree: int, signs: torch.Tensor, device, dtype):
    """Mode weights for value and normal derivative at ``xi = +/- 1``.

    ``P_k(+1) = 1``, ``P_k(-1) = (-1)**k``, ``P_k'(+1) = k(k+1)/2``,
    ``P_k'(-1) = (-1)**(k-1) k(k+1)/2``.
    """
    plus = torch.ones(degree + 1, device=device, dtype=dtype)
    derivative_plus = torch.tensor(
        [0.5 * k * (k + 1.0) for k in range(degree + 1)], device=device, dtype=dtype
    )
    derivative_minus = torch.tensor(
        [0.0] + [((-1.0) ** (k - 1)) * 0.5 * k * (k + 1.0) for k in range(1, degree + 1)],
        device=device,
        dtype=dtype,
    )
    return plus, signs, derivative_plus, derivative_minus


# ---------------------------------------------------------------------------
# Primitive variables and their derivatives
# ---------------------------------------------------------------------------

def _gas_gradients(state: torch.Tensor, d_normal, d_tangential):
    """Primitive fields plus their normal/tangential derivatives.

    ``u``, ``v`` and ``T = p/rho`` are rational in the conservative modes, so
    their derivatives follow exactly from the quotient rule applied to the
    conservative derivatives.  Differentiating *after* forming the primitives
    keeps the viscous flux consistent with the HLLC flux (same state).
    """
    rho = state[..., 0].clamp_min(1e-12)
    velocity_x = state[..., 1] / rho
    velocity_y = state[..., 2] / rho
    kinetic = 0.5 * (velocity_x * velocity_x + velocity_y * velocity_y)
    pressure = (GAMMA - 1.0) * (state[..., 3] - rho * kinetic)
    temperature = pressure / rho

    def _derivative(dstate):
        drho = dstate[..., 0]
        du = (dstate[..., 1] - velocity_x * drho) / rho
        dv = (dstate[..., 2] - velocity_y * drho) / rho
        dp = (GAMMA - 1.0) * (
            dstate[..., 3]
            - 0.5 * drho * (velocity_x * velocity_x + velocity_y * velocity_y)
            - rho * (velocity_x * du + velocity_y * dv)
        )
        dT = (dp - temperature * drho) / rho
        return du, dv, dT

    normal = _derivative(d_normal)
    tangential = _derivative(d_tangential)
    return {
        "state": state,
        "u": velocity_x,
        "v": velocity_y,
        "pressure": pressure,
        "temperature": temperature,
        "u_n": normal[0],
        "v_n": normal[1],
        "T_n": normal[2],
        "u_t": tangential[0],
        "v_t": tangential[1],
        "T_t": tangential[2],
    }


def _face_primitives(coeff, basis, derivative, signs, axis, side):
    """Primitive fields and derivatives on one face of every cell.

    ``side="right"`` evaluates at ``xi = +1``, ``side="left"`` at ``xi = -1``.
    Derivatives are per unit reference coordinate; the caller rescales them.
    """
    degree = int(coeff.shape[2] - 1)
    plus, minus, derivative_plus, derivative_minus = _face_weights(
        degree, signs, coeff.device, coeff.dtype
    )
    if side == "right":
        value_k, deriv_k = plus, derivative_plus
    else:
        value_k, deriv_k = minus, derivative_minus
    if axis == 0:
        state = torch.einsum("xyklv,k,bl->xybv", coeff, value_k, basis)
        d_normal = torch.einsum("xyklv,k,bl->xybv", coeff, deriv_k, basis)
        d_tangential = torch.einsum("xyklv,k,bl->xybv", coeff, value_k, derivative)
    else:
        state = torch.einsum("xyklv,ak,l->xyav", coeff, basis, value_k)
        d_normal = torch.einsum("xyklv,ak,l->xyav", coeff, basis, deriv_k)
        d_tangential = torch.einsum("xyklv,ak,l->xyav", coeff, derivative, value_k)
    return _gas_gradients(state, d_normal, d_tangential)


# ---------------------------------------------------------------------------
# Viscous flux
# ---------------------------------------------------------------------------

def _viscous_flux(face, axis, mu, kappa):
    """``[0, tau_xn, tau_yn, u tau_xn + v tau_yn + kappa dT/dn]`` at a face."""
    u_x, u_y = face["u_x"], face["u_y"]
    v_x, v_y = face["v_x"], face["v_y"]
    divergence = u_x + v_y
    tau_xx = 2.0 * mu * u_x - (2.0 / 3.0) * mu * divergence
    tau_yy = 2.0 * mu * v_y - (2.0 / 3.0) * mu * divergence
    tau_xy = mu * (u_y + v_x)
    u, v = face["u"], face["v"]
    if axis == 0:
        momentum_first, momentum_second = tau_xx, tau_xy
        energy = u * tau_xx + v * tau_xy + kappa * face["T_x"]
    else:
        momentum_first, momentum_second = tau_xy, tau_yy
        energy = u * tau_xy + v * tau_yy + kappa * face["T_y"]
    zeros = torch.zeros_like(momentum_first)
    return torch.stack([zeros, momentum_first, momentum_second, energy], dim=-1)


def _replace_last(tensor, dim, values):
    """Copy of ``tensor`` with the last slice along ``dim`` replaced by ``values``."""
    out = tensor.clone()
    key = [slice(None)] * tensor.dim()
    key[dim] = -1
    out[tuple(key)] = values
    return out


def _take_last(tensor, dim):
    """The last slice of ``tensor`` along ``dim`` (a face's own data)."""
    key = [slice(None)] * tensor.dim()
    key[dim] = -1
    return tensor[tuple(key)]


def _merge(interface, flux, axis, index):
    """Return ``interface`` with the face slice ``index`` replaced by ``flux``."""
    key = [slice(None), slice(None), slice(None), slice(None)]
    key[axis] = index
    merged = interface.clone()
    merged[tuple(key)] = flux[tuple(key)]
    return merged


def _take_at(tensor: torch.Tensor, dim: int, index: int):
    """Slice ``tensor`` at one index of ``dim`` (``index`` may be negative)."""
    key = [slice(None)] * tensor.dim()
    key[dim] = index
    return tensor[tuple(key)]


def _wall_normal_derivative(coeff, basis, signs, h, axis):
    """Second-order one-sided normal derivative of the primitives at both walls.

    A wall face only sees one side of the data, so the DG face derivative
    ``q_h'(face)`` is only ``O(h)`` accurate there: the interior faces average
    two one-sided derivatives whose leading errors cancel, a wall face cannot.
    Extrapolating from the *cell-centred* normal derivatives,

        f'(wall) = 1.5 f'(wall cell) - 0.5 f'(first inland cell) + O(h^2),

    removes that leading term (each centred derivative is ``O(h^2)``, their
    difference over one cell is ``O(h)`` and is multiplied by ``h/2``).  The
    returned values are physical (per unit length) derivatives along ``+axis``,
    which is the sign convention of ``_face_primitives``; ``None`` is returned
    when the mesh has a single cell along ``axis`` (no inland neighbour).

    Scope of validity (m3139, measured -- do not use this as the default):
    both terms of the stencil are ``O(h^2)`` approximations, so the estimate is
    ``O(h^2)`` *at every degree*.  At ``p >= 2`` the one-sided DG face
    derivative this replaces is already ``O(h^p)`` and wins: on the exact
    Brinkman/Couette state this stencil turns the P3 wall heat-flux order into
    2.03 (design order 3, error up to 97x larger) and the P3 residual order
    into 1.0.  At ``p = 1`` it lowers the derivative part of the wall heat-flux
    error but not the observed order (1.005 vs 1.001), because the P1 error is
    dominated by the Dirichlet penalty acting on the ``O(h^2)`` wall
    temperature projection error.  Kept for the m2920 spectral study, where it
    is requested explicitly.
    """
    if coeff.shape[axis] < 2:
        return None
    trace_right, trace_left = _traces(coeff, basis, signs, axis)
    centre = 0.5 * (trace_right + trace_left)
    d_state = (trace_right - trace_left) * (1.0 / h)
    gradients = _gas_gradients(centre, d_state, torch.zeros_like(d_state))
    out = {"left": {}, "right": {}}
    keys = ("u_n", "v_n", "T_n")
    for wall, index, inland in (("left", 0, 1), ("right", -1, -2)):
        for key in keys:
            value = gradients[key]
            estimate = 1.5 * _take_at(value, axis, index) - 0.5 * _take_at(value, axis, inland)
            # keep the full cell layout: ``_boundary_flux`` is evaluated for every
            # cell and only the outermost slice is kept by ``_merge``.
            filled = torch.zeros_like(value)
            slot = [slice(None)] * value.dim()
            slot[axis] = index
            filled[tuple(slot)] = estimate
            out[wall][key] = filled
    return out


def _boundary_flux(side, wall, axis, h, mu, kappa, penalty, normal_scale,
                   tangential_scale, normal_derivative=None):
    """Flux on an outermost face.

    ``wall is None`` gives an outflow face (physical flux, interior gradient);
    a wall dict gives the pressure-only inviscid flux plus the Dirichlet viscous
    flux (no penetration, no convective energy flux).

    ``normal_derivative`` optionally replaces the one-sided DG face derivative
    by a better estimate (see ``_wall_normal_derivative``); ``None`` keeps the
    historical one-sided face derivative.
    """
    inner_u, inner_v = side["u"], side["v"]
    if wall is None:
        normal_u = side["u_n"] * normal_scale
        normal_v = side["v_n"] * normal_scale
        normal_T = side["T_n"] * normal_scale
        face_u, face_v = inner_u, inner_v
        inviscid = _physical_flux(side["state"], axis)
    else:
        wall_velocity = wall.get("velocity", (0.0, 0.0))
        face_u = torch.as_tensor(
            wall_velocity[0], device=inner_u.device, dtype=inner_u.dtype
        ).expand_as(inner_u)
        face_v = torch.as_tensor(
            wall_velocity[1], device=inner_v.device, dtype=inner_v.dtype
        ).expand_as(inner_v)
        wall_temperature = wall.get("temperature", None)
        if wall_temperature is None:
            face_T = side["temperature"]
        else:
            face_T = torch.as_tensor(
                wall_temperature, device=inner_u.device, dtype=inner_u.dtype
            ).expand_as(inner_u)
        if normal_derivative is None:
            base_u = side["u_n"] * normal_scale
            base_v = side["v_n"] * normal_scale
            base_T = side["T_n"] * normal_scale
        else:
            base_u = normal_derivative["u_n"]
            base_v = normal_derivative["v_n"]
            base_T = normal_derivative["T_n"]
        normal_u = base_u - (penalty / h) * (inner_u - face_u)
        normal_v = base_v - (penalty / h) * (inner_v - face_v)
        normal_T = base_T - (penalty / h) * (side["temperature"] - face_T)
        zeros = torch.zeros_like(inner_u)
        pressure = side["pressure"]
        if axis == 0:
            inviscid = torch.stack([zeros, pressure, zeros, zeros], dim=-1)
        else:
            inviscid = torch.stack([zeros, zeros, pressure, zeros], dim=-1)
    tangential_u = side["u_t"] * tangential_scale
    tangential_v = side["v_t"] * tangential_scale
    tangential_T = side["T_t"] * tangential_scale
    face = {"u": face_u, "v": face_v}
    if axis == 0:
        face.update(u_x=normal_u, v_x=normal_v, T_x=normal_T,
                    u_y=tangential_u, v_y=tangential_v, T_y=tangential_T)
    else:
        face.update(u_y=normal_u, v_y=normal_v, T_y=normal_T,
                    u_x=tangential_u, v_x=tangential_v, T_x=tangential_T)
    # conservation form: dU/dt = -div(F_inv) + div(F_v), so the total flux
    # entering the strong-form divergence is F_inv - F_v
    return inviscid - _viscous_flux(face, axis, mu, kappa)


def _axis_fluxes_ns(coeff, basis, derivative, signs, weights, dx, dy, periodic,
                    wall_left, wall_right, mu, kappa, penalty, axis,
                    wall_gradient="face"):
    """Integrated total (inviscid + viscous) face fluxes for one axis.

    Structurally identical to ``euler_dg_2d._axis_fluxes``: the returned arrays
    are the already-projected fluxes of the face to the right and to the left of
    every cell (the normal mode enters through ``P_k(+/-1)`` in the residual).
    """
    h = dx if axis == 0 else dy
    other = dy if axis == 0 else dx
    normal_scale = 2.0 / h
    tangential_scale = 2.0 / other
    roll_dim = axis

    right = _face_primitives(coeff, basis, derivative, signs, axis, "right")
    left = _face_primitives(coeff, basis, derivative, signs, axis, "left")
    state_right, state_left = _traces(coeff, basis, signs, axis)

    outer = {key: torch.roll(value, shifts=-1, dims=roll_dim) for key, value in left.items()}
    outer_state = torch.roll(state_left, shifts=-1, dims=roll_dim)
    if not periodic:
        # outermost face: mirror the interior trace (outflow), as in euler_dg_2d
        for key in outer:
            outer[key] = _replace_last(outer[key], roll_dim, _take_last(right[key], roll_dim))
        outer_state = _replace_last(outer_state, roll_dim, _take_last(state_right, roll_dim))

    inviscid = _hllc(state_right, outer_state, axis)

    jump_u = right["u"] - outer["u"]
    jump_v = right["v"] - outer["v"]
    jump_T = right["temperature"] - outer["temperature"]
    penalty_term = penalty / h
    normal_u = 0.5 * (right["u_n"] + outer["u_n"]) * normal_scale - penalty_term * jump_u
    normal_v = 0.5 * (right["v_n"] + outer["v_n"]) * normal_scale - penalty_term * jump_v
    normal_T = 0.5 * (right["T_n"] + outer["T_n"]) * normal_scale - penalty_term * jump_T
    tangential_u = 0.5 * (right["u_t"] + outer["u_t"]) * tangential_scale
    tangential_v = 0.5 * (right["v_t"] + outer["v_t"]) * tangential_scale
    tangential_T = 0.5 * (right["T_t"] + outer["T_t"]) * tangential_scale
    face = {"u": 0.5 * (right["u"] + outer["u"]),
            "v": 0.5 * (right["v"] + outer["v"])}
    if axis == 0:
        face.update(u_x=normal_u, v_x=normal_v, T_x=normal_T,
                    u_y=tangential_u, v_y=tangential_v, T_y=tangential_T)
    else:
        face.update(u_y=normal_u, v_y=normal_v, T_y=normal_T,
                    u_x=tangential_u, v_x=tangential_v, T_x=tangential_T)
    interface = inviscid - _viscous_flux(face, axis, mu, kappa)

    recon = None
    if not periodic and wall_gradient == "taylor":
        recon = _wall_normal_derivative(coeff, basis, signs, h, axis)

    if not periodic:
        right_derivative = None if recon is None else recon["right"]
        boundary = _boundary_flux(right, wall_right, axis, h, mu, kappa, penalty,
                                  normal_scale, tangential_scale, right_derivative)
        interface = _merge(interface, boundary, roll_dim, -1)

    if axis == 0:
        flux_right = torch.einsum("xybv,b,bl->xylv", interface, weights, basis)
        flux_left = torch.roll(flux_right, shifts=1, dims=0)
    else:
        flux_right = torch.einsum("xyav,a,ak->xykv", interface, weights, basis)
        flux_left = torch.roll(flux_right, shifts=1, dims=1)

    if not periodic:
        left_derivative = None if recon is None else recon["left"]
        left_boundary = _boundary_flux(left, wall_left, axis, h, mu, kappa, penalty,
                                       normal_scale, tangential_scale,
                                       left_derivative)
        if axis == 0:
            flux_left[0] = torch.einsum("ybv,b,bl->ylv", left_boundary[0], weights, basis)
        else:
            flux_left[:, 0] = torch.einsum("xav,a,ak->xkv", left_boundary[:, 0],
                                           weights, basis)
    return flux_right, flux_left


# ---------------------------------------------------------------------------
# Residual
# ---------------------------------------------------------------------------

def _viscous_volume(coeff, basis, derivative, weights, dx, dy, mu, kappa):
    """Volume divergence of the viscous flux, evaluated with local gradients.

    The residual is written in strong form, so the divergence of the *total*
    flux has to be split into a face part and a volume part.  The volume part
    uses the element-local gradient of the DG polynomial (the standard BR1
    choice); omitting it leaves the diffusion operator with only half its
    terms and the scheme is no longer consistent.
    """
    state = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    d_xi = torch.einsum("xyklv,ak,bl->xyabv", coeff, derivative, basis)
    d_eta = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, derivative)
    local = _gas_gradients(state, d_xi, d_eta)
    scale_x = 2.0 / dx
    scale_y = 2.0 / dy
    face_x = {
        "u": local["u"], "v": local["v"],
        "u_x": local["u_n"] * scale_x, "v_x": local["v_n"] * scale_x,
        "T_x": local["T_n"] * scale_x,
        "u_y": local["u_t"] * scale_y, "v_y": local["v_t"] * scale_y,
        "T_y": local["T_t"] * scale_y,
    }
    face_y = {
        "u": local["u"], "v": local["v"],
        # local["u_n"] is the derivative with respect to the reference
        # coordinate xi and local["u_t"] the derivative with respect to eta,
        # so for the y direction the roles are the other way round.
        "u_y": local["u_t"] * scale_y, "v_y": local["v_t"] * scale_y,
        "T_y": local["T_t"] * scale_y,
        "u_x": local["u_n"] * scale_x, "v_x": local["v_n"] * scale_x,
        "T_x": local["T_n"] * scale_x,
    }
    flux_x = _viscous_flux(face_x, 0, mu, kappa)
    flux_y = _viscous_flux(face_y, 1, mu, kappa)
    volume_x = torch.einsum("xyabv,a,b,ak,bl->xyklv", flux_x, weights, weights,
                            derivative, basis)
    volume_y = torch.einsum("xyabv,a,b,ak,bl->xyklv", flux_y, weights, weights,
                            basis, derivative)
    return volume_x, volume_y


def _rhs_ns(coeff, basis, derivative, weights, signs, dx, dy, periodic, boundary,
            mu, kappa, penalty, wall_gradient="face"):
    """DG residual of the 2-D compressible Navier--Stokes equations."""
    degree = coeff.shape[2] - 1
    modes = torch.arange(degree + 1, device=coeff.device, dtype=coeff.dtype)
    scale = torch.outer(2.0 * modes + 1.0, 2.0 * modes + 1.0)

    at_quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    volume_x = torch.einsum(
        "xyabv,a,b,ak,bl->xyklv", _physical_flux(at_quad, 0),
        weights, weights, derivative, basis,
    )
    volume_y = torch.einsum(
        "xyabv,a,b,ak,bl->xyklv", _physical_flux(at_quad, 1),
        weights, weights, basis, derivative,
    )

    viscous_x, viscous_y = _viscous_volume(
        coeff, basis, derivative, weights, dx, dy, mu, kappa
    )
    right_x, left_x = _axis_fluxes_ns(
        coeff, basis, derivative, signs, weights, dx, dy, periodic[0],
        boundary["xmin"], boundary["xmax"], mu, kappa, penalty, 0,
        wall_gradient,
    )
    right_y, left_y = _axis_fluxes_ns(
        coeff, basis, derivative, signs, weights, dx, dy, periodic[1],
        boundary["ymin"], boundary["ymax"], mu, kappa, penalty, 1,
        wall_gradient,
    )

    face_x = (
        right_x.unsqueeze(2)
        - left_x.unsqueeze(2) * signs.view(1, 1, -1, 1, 1)
        - volume_x
        + viscous_x
    )
    face_y = (
        right_y.unsqueeze(3)
        - left_y.unsqueeze(3) * signs.view(1, 1, 1, -1, 1)
        - volume_y
        + viscous_y
    )
    scale_x = (scale / (2.0 * dx)).view(1, 1, *scale.shape, 1)
    scale_y = (scale / (2.0 * dy)).view(1, 1, *scale.shape, 1)
    return -(scale_x * face_x) - (scale_y * face_y)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_navier_stokes_dg_2d(
    initial,
    ncells=16,
    degree=1,
    domain=((0.0, 1.0), (0.0, 1.0)),
    final_time=0.1,
    cfl=0.2,
    cfl_viscous=1.0,
    mu=0.01,
    prandtl=0.71,
    periodic=(True, True),
    boundary=None,
    penalty=None,
    quadrature=None,
    initial_projection="l2",
    wall_gradient="face",
    device="cpu",
    dtype=torch.float64,
):
    """Advance a 2-D compressible Navier--Stokes initial condition with DG/BR1.

    The gas is calorically perfect with ``gamma = 1.4`` and unit gas constant, so
    ``T = p / rho``, ``c_v = 1/(gamma-1)`` and ``k = mu c_p / Pr``.  The viscous
    terms are the BR1/SIPG face gradients documented in the module docstring.

    Parameters
    ----------
    initial : callable
        ``initial(x, y) -> (rho, u, v, p)`` on tensors of any shape.
    ncells : int or (int, int)
        Cell counts in ``x`` and ``y``.
    degree : int
        Polynomial degree of the tensor-product basis (0, 1, 2 or 3).
    mu : float
        Constant dynamic viscosity; ``mu = 0`` (with ``k == 0``) degenerates to
        the inviscid Euler solver.
    prandtl : float
        Prandtl number used for the conductivity ``k = mu c_p / Pr``.
    periodic : (bool, bool)
        Periodicity in ``x`` and ``y``.  A non-periodic direction must be given
        explicit ``boundary`` entries (outflow by default).
    boundary : dict or None
        ``{"xmin", "xmax", "ymin", "ymax"}``; each entry is ``None`` (outflow)
        or ``{"velocity": (u, v), "temperature": T}`` (no-slip isothermal wall).
    penalty : float or None
        SIPG penalty coefficient; defaults to ``2 (degree + 1)**2``.
    wall_gradient : {"face", "taylor"}
        Normal-derivative estimator on Dirichlet walls.  ``"face"`` (default)
        uses the one-sided DG polynomial derivative, which converges at the
        element design order ``p``.  ``"taylor"`` replaces it by the
        ``1.5 q_0 - 0.5 q_1`` extrapolation of centred derivatives: it lowers
        the wall heat-flux error constant at P1 but caps the observed order at
        2, so it is an opt-in diagnostic, not the default (m3139 audit).
    cfl, cfl_viscous : float
        Convective and diffusive step-size fractions.

    Returns
    -------
    (x, y, coeff, metadata) exactly like ``solve_euler_dg_2d``; ``metadata``
    additionally reports ``mu``, ``kappa``, ``penalty``, ``viscous_steps`` and
    the number of steps that were limited by diffusion.

    Notes
    -----
    Verified cases (``benchmarks/verify_ns_dg_2d.py``):

    1. ``mu = kappa = 0`` reproduces ``solve_euler_dg_2d`` to machine precision;
    2. viscous Couette flow with isothermal no-slip walls, analytic
       ``u = U y/H``, ``T = T_wall + mu U^2/(2 kappa) (y/H)(1 - y/H)``;
    3. steady heat conduction across an isothermal channel (linear ``T``);
    4. periodic decaying shear wave, the design-order test of the diffusive
       operator.
    """
    nx, ny = (ncells, ncells) if isinstance(ncells, int) else ncells
    if degree not in (0, 1, 2, 3):
        raise ValueError("degree must be 0, 1, 2 or 3")
    if nx < 1 or ny < 1 or final_time < 0.0 or not 0.0 < cfl <= 0.5:
        raise ValueError("invalid cells, final_time or CFL")
    if not 0.0 < cfl_viscous <= 100.0:
        raise ValueError("cfl_viscous must be positive")
    if mu < 0.0:
        raise ValueError("mu must be non-negative")
    (xmin, xmax), (ymin, ymax) = domain
    if not xmax > xmin or not ymax > ymin:
        raise ValueError("each domain interval must be increasing")

    boundary = dict(boundary or {})
    for key in ("xmin", "xmax", "ymin", "ymax"):
        boundary.setdefault(key, None)
    if periodic[0] and (boundary["xmin"] is not None or boundary["xmax"] is not None):
        raise ValueError("wall boundary and periodicity in x are incompatible")
    if periodic[1] and (boundary["ymin"] is not None or boundary["ymax"] is not None):
        raise ValueError("wall boundary and periodicity in y are incompatible")

    dev = torch.device(device)
    order = quadrature if quadrature is not None else max(degree + 2, 3)
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, order, dev, dtype
    )
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    x = xmin + (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * dx
    y = ymin + (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * dy

    kappa = mu * CP / prandtl
    if penalty is None:
        penalty = 2.0 * (degree + 1.0) ** 2

    if initial_projection != "l2":
        raise ValueError("only the l2 initial projection is supported")
    if wall_gradient not in ("taylor", "face"):
        raise ValueError("wall_gradient must be 'taylor' or 'face'")
    coeff = _initial_coefficients(initial, x, y, dx, dy, nodes, weights, basis, dev, dtype)

    def impose(state):
        return _positivity_filter(state, basis, signs)

    h_min = min(dx, dy)
    time, steps, diffusive_steps, dt = 0.0, 0, 0, 0.0
    while time < final_time:
        at_quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
        _, velocity_x, velocity_y, pressure = _primitive(at_quad)
        rho = at_quad[..., 0].clamp_min(1e-12)
        sound = torch.sqrt(GAMMA * pressure.clamp_min(1e-12) / rho)
        speed_x = float(torch.max(velocity_x.abs() + sound).item())
        speed_y = float(torch.max(velocity_y.abs() + sound).item())
        rate_inviscid = (speed_x / dx + speed_y / dy) * (2 * degree + 1)
        dt = cfl / max(rate_inviscid, 1e-12)
        limited_by_diffusion = False
        if mu > 0.0:
            rho_min = float(torch.min(rho).item())
            diffusivity = max(
                (4.0 / 3.0) * mu / rho_min, kappa / (rho_min * CV)
            )
            # Explicit stability limit of the BR1/SIPG operator.  The
            # dimensionless group S = dt * diffusivity * (p+1)**4 / h**2 was
            # measured by bisection on the shear-wave and Couette cases
            # (S_max ~ 0.3-0.5); VISCOUS_STABILITY = 0.1 keeps a 3-5x margin.
            rate_viscous = diffusivity * (degree + 1.0) ** 4 / (VISCOUS_STABILITY * h_min**2)
            if rate_viscous > 0.0:
                candidate = cfl_viscous / rate_viscous
                if candidate < dt:
                    dt = candidate
                    limited_by_diffusion = True
        dt = min(dt, final_time - time)
        k1 = _rhs_ns(coeff, basis, derivative, weights, signs, dx, dy, periodic,
                     boundary, mu, kappa, penalty, wall_gradient)
        u1 = impose(coeff + dt * k1)
        k2 = _rhs_ns(u1, basis, derivative, weights, signs, dx, dy, periodic,
                     boundary, mu, kappa, penalty, wall_gradient)
        u2 = impose(0.75 * coeff + 0.25 * (u1 + dt * k2))
        k3 = _rhs_ns(u2, basis, derivative, weights, signs, dx, dy, periodic,
                     boundary, mu, kappa, penalty, wall_gradient)
        coeff = impose(coeff / 3.0 + (2.0 / 3.0) * (u2 + dt * k3))
        if not torch.isfinite(coeff).all():
            raise FloatingPointError("non-finite Navier-Stokes state; reduce the CFL")
        pressure_now = (GAMMA - 1.0) * (
            coeff[..., 0, 0, 3]
            - 0.5
            * (coeff[..., 0, 0, 1] ** 2 + coeff[..., 0, 0, 2] ** 2)
            / coeff[..., 0, 0, 0].clamp_min(1e-12)
        )
        if (coeff[..., 0, 0, 0] <= 0.0).any() or (pressure_now <= 0.0).any():
            raise FloatingPointError(
                "nonphysical cell average in Navier-Stokes DG; reduce the CFL"
            )
        time += dt
        steps += 1
        diffusive_steps += int(limited_by_diffusion)

    metadata = {
        "ncells": (nx, ny),
        "degree": degree,
        "final_time": final_time,
        "steps": steps,
        "diffusive_steps": diffusive_steps,
        "dt": dt,
        "cfl": cfl,
        "cfl_viscous": cfl_viscous,
        "domain": domain,
        "periodic": tuple(periodic),
        "mu": mu,
        "kappa": kappa,
        "prandtl": prandtl,
        "penalty": penalty,
        "boundary": {key: value for key, value in boundary.items()},
        "flux": "hllc + br1/sipg viscous",
        "quadrature": order,
        "initial_projection": initial_projection,
        "device": str(dev),
        "dtype": str(dtype).replace("torch.", ""),
        "dofs": nx * ny * (degree + 1) ** 2 * 4,
    }
    return x, y, coeff, metadata


# ---------------------------------------------------------------------------
# Analytic reference solutions
# ---------------------------------------------------------------------------

def couette_exact(y, ymin, ymax, wall_speed, wall_temperature, pressure, mu, kappa):
    """Analytic steady viscous Couette flow with isothermal no-slip walls.

    ``u = U (y - ymin)/H`` with ``H = ymax - ymin``; the y-momentum balance gives
    a uniform pressure, the energy balance ``kappa T'' + mu (u')^2 = 0`` with
    ``T = wall_temperature`` at both walls gives the Brinkman parabola

        T(y) = T_wall + mu U^2 / (2 kappa) * s (1 - s),    s = (y - ymin)/H,

    and the density follows from ``rho = p / (R T)``.  This is an exact solution
    of the compressible Navier--Stokes equations used for verification.
    """
    height = ymax - ymin
    s = (y - ymin) / height
    velocity = wall_speed * s
    amplitude = mu * wall_speed * wall_speed / (2.0 * kappa)
    temperature = wall_temperature + amplitude * s * (1.0 - s)
    density = pressure / temperature
    return {
        "density": density,
        "velocity_x": velocity,
        "velocity_y": torch.zeros_like(velocity),
        "pressure": torch.full_like(temperature, pressure),
        "temperature": temperature,
    }


def heat_conduction_exact(y, ymin, ymax, bottom_temperature, top_temperature,
                          pressure):
    """Steady pure-conduction state between two isothermal walls at rest."""
    s = (y - ymin) / (ymax - ymin)
    temperature = bottom_temperature + (top_temperature - bottom_temperature) * s
    density = pressure / temperature
    zeros = torch.zeros_like(temperature)
    return {
        "density": density,
        "velocity_x": zeros,
        "velocity_y": zeros,
        "pressure": torch.full_like(temperature, pressure),
        "temperature": temperature,
    }


def shear_wave_exact(x, y, time, amplitude, wavenumber, kinematic_viscosity,
                     pressure, ymin=0.0, ymax=1.0, background_temperature=1.0):
    """Decaying shear wave ``u = A sin(k (y - ymin)) exp(-nu k^2 t)``.

    An exact solution of the incompressible limit and of the compressible
    Navier--Stokes equations up to ``O(A^2)`` (the neglected viscous heating);
    with ``A ~ 0.01`` that term is ``< 1e-4`` of ``T``.
    """
    decay = float(np.exp(-kinematic_viscosity * wavenumber ** 2 * float(time)))
    velocity = amplitude * torch.sin(wavenumber * (y - ymin)) * decay
    temperature = torch.full_like(y, background_temperature)
    density = pressure / temperature
    return {
        "density": density,
        "velocity_x": velocity,
        "velocity_y": torch.zeros_like(velocity),
        "pressure": torch.full_like(velocity, pressure),
        "temperature": temperature,
    }


# ---------------------------------------------------------------------------
# Named set-ups
# ---------------------------------------------------------------------------

def solve_couette_dg_2d(
    ncells=(2, 16),
    degree=1,
    final_time=40.0,
    wall_speed=1.0,
    wall_temperature=1.0,
    pressure=1.0,
    mu=0.1,
    prandtl=0.71,
    cfl=0.3,
    cfl_viscous=1.0,
    ymin=0.0,
    ymax=1.0,
    xmin=0.0,
    xmax=1.0,
    device="cpu",
    dtype=torch.float64,
):
    """Impulsively started Couette flow (bottom wall at rest, top wall moving)."""

    def initial(x, y):
        return (
            torch.ones_like(x * y),
            torch.zeros_like(x * y),
            torch.zeros_like(x * y),
            pressure * torch.ones_like(x * y),
        )

    boundary = {
        "ymin": {"velocity": (0.0, 0.0), "temperature": wall_temperature},
        "ymax": {"velocity": (wall_speed, 0.0), "temperature": wall_temperature},
    }
    return solve_navier_stokes_dg_2d(
        initial,
        ncells=ncells,
        degree=degree,
        domain=((xmin, xmax), (ymin, ymax)),
        final_time=final_time,
        cfl=cfl,
        cfl_viscous=cfl_viscous,
        mu=mu,
        prandtl=prandtl,
        periodic=(True, False),
        boundary=boundary,
        device=device,
        dtype=dtype,
    )


def solve_heat_conduction_dg_2d(
    ncells=(2, 16),
    degree=1,
    final_time=40.0,
    bottom_temperature=1.0,
    top_temperature=0.5,
    pressure=1.0,
    mu=0.1,
    prandtl=0.71,
    cfl=0.3,
    cfl_viscous=1.0,
    device="cpu",
    dtype=torch.float64,
):
    """Pure heat conduction between two isothermal walls in a fluid at rest."""

    def initial(x, y):
        return (
            torch.ones_like(x * y),
            torch.zeros_like(x * y),
            torch.zeros_like(x * y),
            pressure * torch.ones_like(x * y),
        )

    boundary = {
        "ymin": {"velocity": (0.0, 0.0), "temperature": bottom_temperature},
        "ymax": {"velocity": (0.0, 0.0), "temperature": top_temperature},
    }
    return solve_navier_stokes_dg_2d(
        initial,
        ncells=ncells,
        degree=degree,
        domain=((0.0, 1.0), (0.0, 1.0)),
        final_time=final_time,
        cfl=cfl,
        cfl_viscous=cfl_viscous,
        mu=mu,
        prandtl=prandtl,
        periodic=(True, False),
        boundary=boundary,
        device=device,
        dtype=dtype,
    )


def solve_shear_wave_dg_2d(
    ncells=(4, 16),
    degree=1,
    final_time=0.5,
    amplitude=0.01,
    wavenumber=2.0 * np.pi,
    mu=0.05,
    pressure=1.0,
    temperature=1.0,
    cfl=0.3,
    cfl_viscous=1.0,
    device="cpu",
    dtype=torch.float64,
):
    """Periodic decaying shear wave: the design-order test of the viscous term."""
    density = pressure / temperature

    def initial(x, y):
        return (
            density * torch.ones_like(x * y),
            amplitude * torch.sin(wavenumber * y),
            torch.zeros_like(x * y),
            pressure * torch.ones_like(x * y),
        )

    return solve_navier_stokes_dg_2d(
        initial,
        ncells=ncells,
        degree=degree,
        domain=((0.0, 1.0), (0.0, 1.0)),
        final_time=final_time,
        cfl=cfl,
        cfl_viscous=cfl_viscous,
        mu=mu,
        periodic=(True, True),
        device=device,
        dtype=dtype,
    )