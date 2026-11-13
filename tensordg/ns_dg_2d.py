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

from .stepping import step_metadata
from .euler_1d import GAMMA
from .euler_dg_2d import (
    _conservative,
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
    "wall_heat_flux",
    "wall_heat_flux_summary",
    "shear_wave_exact",
    "gas_constants",
    "taylor_green_exact",
    "solve_taylor_green_dg_2d",
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

def _gas_gradients(state: torch.Tensor, d_normal, d_tangential,
                   conversion_state=None):
    """Primitive fields plus their normal/tangential derivatives.

    ``u``, ``v`` and ``T = p/rho`` are rational in the conservative modes, so
    their derivatives follow exactly from the quotient rule applied to the
    conservative derivatives.  Differentiating *after* forming the primitives
    keeps the viscous flux consistent with the HLLC flux (same state).

    ``conversion_state`` (optional) supplies the state used in the quotient
    rule instead of the face trace.  Evaluating the denominators at the cell
    centre makes ``grad T`` a *cell* quantity; evaluating them at each face
    makes the same cell gradient differ between its two faces, which destroys
    the two-sided cancellation that the interface average relies on (see
    ``docs/reports/tensordg-ns-wall-gradient-fix.md``).
    """
    base = state if conversion_state is None else conversion_state
    rho_c = base[..., 0].clamp_min(1e-12)
    u_c = base[..., 1] / rho_c
    v_c = base[..., 2] / rho_c
    kinetic_c = 0.5 * (u_c * u_c + v_c * v_c)
    pressure_c = (GAMMA - 1.0) * (base[..., 3] - rho_c * kinetic_c)
    temperature_c = pressure_c / rho_c

    rho = state[..., 0].clamp_min(1e-12)
    velocity_x = state[..., 1] / rho
    velocity_y = state[..., 2] / rho
    kinetic = 0.5 * (velocity_x * velocity_x + velocity_y * velocity_y)
    pressure = (GAMMA - 1.0) * (state[..., 3] - rho * kinetic)
    temperature = pressure / rho

    def _derivative(dstate, rho_conv, u_conv, v_conv, T_conv):
        drho = dstate[..., 0]
        du = (dstate[..., 1] - u_conv * drho) / rho_conv
        dv = (dstate[..., 2] - v_conv * drho) / rho_conv
        dp = (GAMMA - 1.0) * (
            dstate[..., 3]
            - 0.5 * drho * (u_conv * u_conv + v_conv * v_conv)
            - rho_conv * (u_conv * du + v_conv * dv)
        )
        dT = (dp - T_conv * drho) / rho_conv
        return du, dv, dT

    conversion = (rho_c, u_c, v_c, temperature_c)
    normal = _derivative(d_normal, *conversion)
    tangential = _derivative(d_tangential, *conversion)
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


def _centre_weights(degree: int, device, dtype):
    """Mode weights for the value and the normal derivative at ``xi = 0``.

    ``P_k(0) = 0`` for odd ``k``, ``(-1)**(k/2) C(k, k/2) / 2**k`` for even ``k``.
    Used by the *recovered* wall closure: the wall-normal gradient there is the
    quadratic extrapolation ``2 G(centre) - G(opposite face)``, which is one
    order more accurate than the one-sided face gradient (see
    :func:`_recover_wall_gradient`).
    """
    origin = np.zeros(1)
    values = np.polynomial.legendre.legvander(origin, degree)[0]
    derivatives = np.zeros(degree + 1)
    for mode in range(degree + 1):
        coefficients = np.zeros(mode + 1)
        coefficients[mode] = 1.0
        differentiated = np.polynomial.legendre.legder(coefficients)
        if differentiated.size:
            derivatives[mode] = float(
                np.polynomial.legendre.legval(origin, differentiated)[0]
            )
    return (
        torch.as_tensor(values, device=device, dtype=dtype),
        torch.as_tensor(derivatives, device=device, dtype=dtype),
    )

def _recover_wall_gradient(centre, interface, normal_scale, axis, side):
    """Quadratic extrapolation of the wall-normal gradient onto a Dirichlet face.

    The BR1 closure averages the two one-sided gradients on an *interior* face,
    which cancels their leading error and leaves ``O(h**(p+1))``.  A wall has no
    second side, so the historical mirror ghost ``grad|+ = grad|-`` reproduces
    the one-sided gradient, which is only ``O(h**p)``: divided by ``h`` in the
    strong-form residual it leaves an ``O(h**(p-1))`` (= ``O(1)`` for ``P1``)
    wall source, and the wall energy flux is first order
    (``docs/reports/tensordg-ns-wall-verification.md``).

    This closure rebuilds the missing side from data that *is* available inside
    the wall cell: the element-centre gradient (available to every order) and
    the already-averaged gradient on the opposite, interior face of the same
    cell.  For a polynomial-consistent state the extrapolation

        grad_n|_wall = 2 grad_n(centre) - grad_n(opposite face)

    is exact to one order beyond the one-sided value, because the leading error
    of the one-sided gradient is linear in the normal coordinate and the two
    samples bracket it symmetrically.  ``centre``/``interface`` are
    :func:`_face_primitives`-style dicts; ``interface`` holds the physical
    normal derivative on the interior face (already scaled).
    """
    cell = slice(-1, None) if side == "right" else slice(0, 1)
    # interface ``i`` sits between cell ``i`` and ``i + 1``: the wall cell's
    # opposite (interior) face is interface ``-2`` at the top, interface ``0``
    # at the bottom.
    face = slice(-2, -1) if side == "right" else slice(0, 1)
    return {
        key: 2.0 * centre[key][..., cell, :] * normal_scale
        - interface[key][..., face, :]
        for key in ("u_n", "v_n", "T_n")
    }

# ----------------------------------------------------------------------
# m2831 second-order one-sided wall stencil (``wall_gradient='fd2'``).
# The stencil fits a parabola through the imposed wall value and the two
# nearest cell centres, so the Dirichlet value is already built into the
# derivative and the redundant BR1 jump penalty is retired by default
# (``WALL_PENALTY``).  ``wall_gradient='br1'`` is accepted as an alias of
# the legacy mirror-ghost closure (``'one_sided'``).  The public default
# stays ``'taylor'`` (m2920); fd2 is opt-in.
# ----------------------------------------------------------------------
# m3139 settled the public default at the one-sided DG derivative;
# the m2831 fd2 stencil (and the m2838 recovered closure) stay opt-in.
WALL_GRADIENT = "face"

# Coefficient of the *symmetric* (derivative-projected, Nitsche adjoint) wall
# penalty relative to the value-projected SIPG penalty (m2972; see
# ``_wall_adjoint_flux`` and ``docs/reports/tensordg-ns-wall-symmetric.md``).
WALL_ADJOINT_FACTOR = 2.0

WALL_PENALTY = 0.0


def _legendre_at_zero(degree, device, dtype):
    """``P_k(0)`` for ``k = 0..degree`` (the modal basis evaluated at the centre)."""
    values = [1.0]
    if degree >= 1:
        values.append(0.0)
    for k in range(2, degree + 1):
        values.append(-((k - 1.0) / k) * values[k - 2])
    return torch.tensor(values, device=device, dtype=dtype)


def _cell_centre_primitives(coeff):
    """``u``, ``v``, ``T`` at the cell centres, from the cell averages.

    The DG polynomial is evaluated at the cell centre (``xi = eta = 0``) and
    converted to ``u = m/rho``, ``T = p/rho``.  Evaluating the polynomial -- as
    opposed to using the cell average -- matters: for ``p >= 2`` the
    mean/centre offset is ``(h**2/24) q''``, which does *not* vanish at the
    wall and would cost one order of accuracy in the one-sided stencil.
    """
    degree = coeff.shape[2] - 1
    at_zero = _legendre_at_zero(degree, coeff.device, coeff.dtype)
    state = torch.einsum("xyklv,k,l->xyv", coeff, at_zero, at_zero)
    rho = state[..., 0].clamp_min(1e-12)
    momentum_x, momentum_y, energy = state[..., 1], state[..., 2], state[..., 3]
    pressure = (GAMMA - 1.0) * (
        energy - 0.5 * (momentum_x**2 + momentum_y**2) / rho
    )
    return momentum_x / rho, momentum_y / rho, pressure / rho


def _wall_stencil(field, axis, side):
    """(wall cell, next inner cell) values of ``field`` along the normal axis."""
    count = field.shape[axis]
    if count < 2:
        raise ValueError("a Dirichlet wall needs at least two cells along the normal")
    if side == "left":
        first, second = 0, 1
    else:
        first, second = count - 1, count - 2
    values = []
    for index in (first, second):
        key = [slice(None)] * field.dim()
        key[axis] = slice(index, index + 1)
        # face data is (..., normal cell index, ..., tangential quadrature): the
        # trailing singleton broadcasts the cell value over those quadrature
        # points
        values.append(field[tuple(key)].unsqueeze(-1))
    return values[0], values[1]


def _wall_derivative(wall_value, centre, inner, h, side):
    """Second-order one-sided normal derivative at a Dirichlet wall.

    Nodes sit at distances ``0`` (wall), ``h/2`` and ``3h/2`` (the first two
    cell centres), so the derivative of the interpolating parabola in the
    inward direction ``s`` is ``(-8/3 q_w + 3 q_1 - q_2/3) / h``.  Code
    convention: every face flux is built with the normal along ``+axis``, so at
    the ``right`` wall the inward direction is ``-n``.
    """
    inward = (-8.0 / 3.0 * wall_value + 3.0 * centre - inner / 3.0) / h
    return inward if side == "left" else -inward


def _centre_mode_weights(degree: int, device, dtype):
    """``P_k(0)``: the Legendre modes evaluated at the element centre."""
    values = [1.0, 0.0]
    for k in range(2, degree + 1):
        values.append(-(k - 1.0) / k * values[k - 2])
    return torch.tensor(values[: degree + 1], device=device, dtype=dtype)


def _cell_centre_state(coeff):
    """Conserved state at the centre of every cell (quotient-rule operands)."""
    degree = int(coeff.shape[2] - 1)
    weights = _centre_mode_weights(degree, coeff.device, coeff.dtype)
    return torch.einsum("xyklv,k,l->xyv", coeff, weights, weights)


def _face_primitives(coeff, basis, derivative, signs, axis, side,
                     centre_denominators=False):
    """Primitive fields and derivatives on one face of every cell.

    ``side="right"`` evaluates at ``xi = +1``, ``side="left"`` at ``xi = -1``.
    Derivatives are per unit reference coordinate; the caller rescales them.
    The quotient rule that turns conservative derivatives into ``grad T`` uses
    the *cell-centre* state, so the trace derivative is a cell quantity.
    """
    degree = int(coeff.shape[2] - 1)
    plus, minus, derivative_plus, derivative_minus = _face_weights(
        degree, signs, coeff.device, coeff.dtype
    )
    if side == "right":
        value_k, deriv_k = plus, derivative_plus
    elif side == "centre":
        # m2838 recovered closure: value/derivative weights at xi = 0
        value_k, deriv_k = _centre_weights(degree, coeff.device, coeff.dtype)
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
    if centre_denominators:
        # m2879: evaluate the quotient-rule denominators at the cell centre so
        # the cell gradient is a *cell* quantity (opt-in; the solver defaults
        # keep the historical face-trace denominators of main).
        centre = _cell_centre_state(coeff)
        centre = centre.unsqueeze(2).expand(-1, -1, state.shape[2], -1)
        return _gas_gradients(state, d_normal, d_tangential,
                              conversion_state=centre)
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


def _extrapolate_trace_derivative(values: torch.Tensor, neighbour, factor: float = 1.5):
    """Second-order extrapolation of a one-sided face derivative to the face.

    Interior faces average the two one-sided traces, and the two O(h) offsets
    cancel, leaving an O(h^2) face derivative.  A boundary face has only one
    side available, so the same accuracy is recovered by linear extrapolation
    from the two innermost cells: ``g_face = 1.5 g_cell - 0.5 g_neighbour``
    (exact for a quadratic trace).  Without ``neighbour`` the old first-order
    mirror value is returned.
    """
    if neighbour is None:
        return values
    return factor * values - (factor - 1.0) * neighbour

def _shift_along(values, axis: int, shift: int):
    """Shift a per-cell face array (or dict of them) by ``shift`` cells."""
    if isinstance(values, dict):
        return {key: torch.roll(value, shifts=shift, dims=axis)
                for key, value in values.items()}
    return torch.roll(values, shifts=shift, dims=axis)


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
                   tangential_scale, normal_derivative=None, neighbour=None,
                   recovered=None,
                   centre=None, side_name="right", outward_sign=1.0):
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
        if normal_derivative is not None:
            base_u = normal_derivative["u_n"]
            base_v = normal_derivative["v_n"]
            base_T = normal_derivative["T_n"]
        elif neighbour is not None:
            # m2879: second-order extrapolation of the one-sided trace
            # derivative from the two innermost cells
            base_u = _extrapolate_trace_derivative(
                side["u_n"], neighbour["u_n"]) * normal_scale
            base_v = _extrapolate_trace_derivative(
                side["v_n"], neighbour["v_n"]) * normal_scale
            base_T = _extrapolate_trace_derivative(
                side["T_n"], neighbour["T_n"]) * normal_scale
        elif recovered is not None:
            # m2838: physical (already scaled) wall-normal derivatives from
            # the quadratic recovery ``_recover_wall_gradient``
            base_u = recovered["u_n"]
            base_v = recovered["v_n"]
            base_T = recovered["T_n"]
        elif centre is not None:
            # m2831 fd2: second-order one-sided stencil through the imposed
            # wall value and the two nearest cell centres
            wall_u, inner_u_c = _wall_stencil(centre["u"], axis, side_name)
            wall_v, inner_v_c = _wall_stencil(centre["v"], axis, side_name)
            wall_T, inner_T_c = _wall_stencil(centre["T"], axis, side_name)
            base_u = _wall_derivative(face_u, wall_u, inner_u_c, h, side_name)
            base_v = _wall_derivative(face_v, wall_v, inner_v_c, h, side_name)
            base_T = _wall_derivative(face_T, wall_T, inner_T_c, h, side_name)
        else:
            base_u = side["u_n"] * normal_scale
            base_v = side["v_n"] * normal_scale
            base_T = side["T_n"] * normal_scale
        # m2782: the face flux is written with the normal along +axis, so the
        # penalty jump must flip at the low-side wall (outward_sign = -1);
        # same-sign penalties are anti-dissipative there
        normal_u = base_u - outward_sign * (penalty / h) * (inner_u - face_u)
        normal_v = base_v - outward_sign * (penalty / h) * (inner_v - face_v)
        normal_T = (base_T - outward_sign * (penalty / h)
                    * (side["temperature"] - face_T))
        zeros = torch.zeros_like(inner_u)
        pressure = side["pressure"]
        if axis == 0:
            inviscid = torch.stack([zeros, pressure, zeros, zeros], dim=-1)
        else:
            inviscid = torch.stack([zeros, zeros, pressure, zeros], dim=-1)
        prescribed = wall.get("state")
        if prescribed is not None:
            # m16075: prescribed-state (inlet/outlet) face.  A wall dict without
            # ``"state"`` carries no convective flux (zero mass and energy
            # transport), which is right for an impermeable wall and wrong for a
            # through-flow boundary.  With ``"state"`` -- a dict of ``rho``,
            # ``velocity_x``, ``velocity_y``, ``pressure`` broadcastable to the
            # face arrays -- the inviscid flux becomes the HLLC flux of the
            # ghost (prescribed) state against the interior trace, so mass,
            # momentum and energy cross the face while the Dirichlet viscous
            # penalty above still imposes the prescribed velocity/temperature.
            # ``_hllc`` is written with the normal along ``+axis`` and expects
            # the state on the low side of the interface first: at the low-side
            # face the ghost is that state, at the high-side face it is not.
            ghost = _conservative(
                prescribed["rho"],
                prescribed["velocity_x"],
                prescribed["velocity_y"],
                prescribed["pressure"],
            ).expand_as(side["state"])
            if side_name == "left":
                inviscid = _hllc(ghost, side["state"], axis)
            else:
                inviscid = _hllc(side["state"], ghost, axis)
    if wall is None or neighbour is None:
        tangential_u = side["u_t"] * tangential_scale
        tangential_v = side["v_t"] * tangential_scale
        tangential_T = side["T_t"] * tangential_scale
    else:
        # m2879: extrapolate the tangential trace the same way
        tangential_u = _extrapolate_trace_derivative(
            side["u_t"], neighbour["u_t"]) * tangential_scale
        tangential_v = _extrapolate_trace_derivative(
            side["v_t"], neighbour["v_t"]) * tangential_scale
        tangential_T = _extrapolate_trace_derivative(
            side["T_t"], neighbour["T_t"]) * tangential_scale
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

def _put_face(target, dim, index, values):
    """Copy the ``index``-th face slice of ``values`` into ``target`` along ``dim``."""
    selector = [slice(None)] * target.dim()
    selector[dim] = index
    source = [slice(None)] * values.dim()
    source[dim] = index
    target[tuple(selector)] = values[tuple(source)]




def _wall_adjoint_flux(side, wall, axis, h, mu, kappa, penalty):
    """Symmetric (Nitsche adjoint) wall term, as a face quantity.

    The value-projected SIPG penalty ``-(penalty/h)(q^- - q_D)`` at a Dirichlet
    wall enters the strong-form residual through the *test function value* at
    the face.  A Nitsche boundary discretisation needs the mirrored term that
    enters through the *normal derivative* of the test function,

        -\int_{Gamma_D} (d_n phi) (q^- - q_D) ds ,

    otherwise the boundary contribution to the discrete operator has the
    rank-one, sign-indefinite structure that makes the wall cell's highest
    mode grow (m2589/m2972 diagnostics: growth rate ``mu sigma / h**2``, see
    ``validation/ns_dg_2d_wall/spectrum.json``).  This function returns the
    flux-like quantity ``(factor sigma / h) (q^- - q_D)`` per channel; the
    caller projects it with the *derivative* Legendre weights, which is exactly
    the discrete form of the integral above (``\int P_k'(+/-1)`` on the
    reference face).

    The term is proportional to the Dirichlet defect, so it vanishes for the
    exact solution: it is a consistent stabilisation, not a change of the
    boundary condition.
    """
    inner_u, inner_v = side["u"], side["v"]
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
    scale = WALL_ADJOINT_FACTOR * penalty / h
    zeros = torch.zeros_like(inner_u)
    return torch.stack(
        [zeros, scale * mu * (inner_u - face_u), scale * mu * (inner_v - face_v),
         scale * kappa * (side["temperature"] - face_T)],
        dim=-1,
    )



def _axis_fluxes_ns(coeff, basis, derivative, signs, weights, dx, dy, periodic,
                    wall_left, wall_right, mu, kappa, penalty, axis,
                    wall_gradient="face", wall_penalty=None, adjoint=False):
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

    if wall_gradient == "br1":
        # m2831 alias: the pre-m2831 mirror-ghost closure
        wall_gradient = "one_sided"

    recon = None
    if not periodic and wall_gradient == "taylor":
        recon = _wall_normal_derivative(coeff, basis, signs, h, axis)

    recovered_top = recovered_bottom = None
    if (not periodic and wall_gradient == "recovered"
            and (wall_left is not None or wall_right is not None)):
        centre = _face_primitives(coeff, basis, derivative, signs, axis, "centre")
        averaged = {"u_n": normal_u, "v_n": normal_v, "T_n": normal_T}
        if wall_right is not None:
            recovered_top = _recover_wall_gradient(centre, averaged, normal_scale,
                                                   axis, "right")
        if wall_left is not None:
            recovered_bottom = _recover_wall_gradient(centre, averaged,
                                                      normal_scale, axis, "left")

    centre_state = None
    if (not periodic and wall_gradient == "fd2"
            and (wall_left is not None or wall_right is not None)):
        centre_u, centre_v, centre_T = _cell_centre_primitives(coeff)
        centre_state = {"u": centre_u, "v": centre_v, "T": centre_T}

    if wall_gradient == "fd2":
        # m2831: the Dirichlet value is already inside the stencil, so the
        # jump penalty is retired unless explicitly requested
        wall_sigma = WALL_PENALTY if wall_penalty is None else wall_penalty
    else:
        wall_sigma = penalty if wall_penalty is None else wall_penalty

    if not periodic:
        right_derivative = None if recon is None else recon["right"]
        boundary = _boundary_flux(right, wall_right, axis, h, mu, kappa,
                                  wall_sigma, normal_scale, tangential_scale,
                                  normal_derivative=right_derivative,
                                  recovered=recovered_top,
                                  centre=centre_state, side_name="right",
                                  outward_sign=1.0)
        interface = _merge(interface, boundary, roll_dim, -1)

    if axis == 0:
        flux_right = torch.einsum("xybv,b,bl->xylv", interface, weights, basis)
        flux_left = torch.roll(flux_right, shifts=1, dims=0)
    else:
        flux_right = torch.einsum("xyav,a,ak->xykv", interface, weights, basis)
        flux_left = torch.roll(flux_right, shifts=1, dims=1)

    if not periodic:
        left_derivative = None if recon is None else recon["left"]
        left_boundary = _boundary_flux(left, wall_left, axis, h, mu, kappa,
                                       wall_sigma, normal_scale, tangential_scale,
                                       normal_derivative=left_derivative,
                                       recovered=recovered_bottom,
                                       centre=centre_state, side_name="left",
                                       outward_sign=-1.0)
        if axis == 0:
            flux_left[0] = torch.einsum("ybv,b,bl->ylv", left_boundary[0], weights, basis)
        else:
            flux_left[:, 0] = torch.einsum("xav,a,ak->xkv", left_boundary[:, 0],
                                           weights, basis)

    # Symmetric (Nitsche adjoint) wall term (m2972): the same face data as the
    # wall flux, projected exactly like flux_right/flux_left; _rhs_ns attaches
    # the *derivative* Legendre weights instead of the value weights, which is
    # the discrete form of -int_{Gamma_D} (d_n phi) (q^- - q_D) ds.
    adj_right = torch.zeros_like(flux_right)
    adj_left = torch.zeros_like(flux_left)
    if not periodic and adjoint:
        if wall_right is not None:
            adj_face = _wall_adjoint_flux(right, wall_right, axis, h, mu, kappa, penalty)
            if axis == 0:
                adj_right[-1] = torch.einsum("ybv,b,bl->ylv", adj_face[-1], weights, basis)
            else:
                adj_right[:, -1] = torch.einsum("xav,a,ak->xkv", adj_face[:, -1],
                                                weights, basis)
        if wall_left is not None:
            adj_face = _wall_adjoint_flux(left, wall_left, axis, h, mu, kappa, penalty)
            if axis == 0:
                adj_left[0] = torch.einsum("ybv,b,bl->ylv", adj_face[0], weights, basis)
            else:
                adj_left[:, 0] = torch.einsum("xav,a,ak->xkv", adj_face[:, 0],
                                              weights, basis)
    return flux_right, flux_left, adj_right, adj_left

# ---------------------------------------------------------------------------
# Wall heat-flux post-processing (m14554)
# ---------------------------------------------------------------------------

def wall_heat_flux(coeff, basis, derivative, signs, weights, dx, dy, boundary,
                   mu, kappa, penalty, wall_gradient="face", wall_penalty=None,
                   periodic=(False, False)):
    """Physical wall-normal conductive heat flux of a DG Navier--Stokes state.

    The Dirichlet (isothermal, no-slip) wall closure of :func:`_axis_fluxes_ns`
    puts a face flux :math:`F^*_E = u tau_{xn} + v tau_{yn} + kappa \partial_n T`
    into the strong-form divergence.  This function reads that flux back as a
    *physical* quantity: it takes the ``k = 0`` modal coefficient of the
    projected face flux (``int F^* phi_0 dxi``) and divides by
    ``int phi_0 dxi = 2``, which is the exact inverse of the L2 face projection
    for a face-constant flux.

    Parameters
    ----------
    coeff, basis, derivative, signs, weights, dx, dy :
        The DG state and the 1-D operators returned by
        :func:`tensordg.euler_dg_2d.legendre_operators`.
    boundary : dict
        ``{"xmin", "xmax", "ymin", "ymax"}``; ``None`` marks an outflow/periodic
        face (no wall flux is reported for it) and a
        ``{"velocity": (u, v), "temperature": T}`` dict marks an isothermal wall.
    mu, kappa, penalty :
        Viscosity, conductivity (``kappa = mu c_p / Pr``) and the SIPG penalty
        actually used by the operator (``wall_penalty`` overrides it on walls).
    wall_gradient : {"face", "taylor", "one_sided", "recovered", "fd2", "br1"}
        Normal-derivative estimator used on the wall, as in
        :func:`solve_navier_stokes_dg_2d`.
    wall_penalty : float or None
        Penalty used on the wall faces only (``None`` reproduces the solver).
    periodic : (bool, bool)
        Periodicity of the two axes; a periodic axis is skipped (its faces are
        interior faces, not walls).

    Returns
    -------
    dict
        ``{wall_name: torch.Tensor}`` with one entry per isothermal wall.  The
        tensor holds one value per wall cell: the *physical* heat-flux component
        along ``+axis``, ``q_axis = -kappa dT/dx_axis`` (Fourier's law with the
        same sign for both walls of an axis, because the operator stores the
        face flux with the normal along ``+axis``).  Steady conduction between
        ``T_hot`` (ymin) and ``T_cold`` (ymax) over a unit gap with
        ``kappa = mu c_p / Pr`` therefore reports ``q_y = kappa (T_hot -
        T_cold)`` at *both* walls: heat crosses the hot wall into the domain and
        leaves through the cold wall, so both values are positive.  An empty
        dict means the state has no isothermal wall.

    Notes
    -----
    The reported flux is exactly the flux the solver used, i.e. it inherits the
    accuracy of the chosen wall-gradient estimator (design order ``p`` for
    ``"face"``).  It is *not* a post-hoc higher-order reconstruction; see
    ``benchmarks/verify_ns_wall_heat_flux.py`` for the measured orders and
    ``docs/reports/tensordg-ns-heat-wall-flux-diagnosis.md`` for the audit.
    """
    if not boundary:
        return {}
    walls = {key: boundary.get(key) for key in ("xmin", "xmax", "ymin", "ymax")}
    if all(value is None for value in walls.values()):
        return {}
    norm0 = float(weights @ basis[:, 0])
    if norm0 == 0.0:
        raise ValueError("degenerate modal basis: int phi_0 dxi == 0")
    flux = {}
    for axis, left_name, right_name in ((0, "xmin", "xmax"), (1, "ymin", "ymax")):
        if periodic[axis]:
            continue
        wall_left, wall_right = walls[left_name], walls[right_name]
        if wall_left is None and wall_right is None:
            continue
        right, left, _, _ = _axis_fluxes_ns(
            coeff, basis, derivative, signs, weights, dx, dy, False,
            wall_left, wall_right, mu, kappa, penalty, axis,
            wall_gradient, wall_penalty)
        if wall_right is not None:
            face = right.narrow(axis, right.shape[axis] - 1, 1).squeeze(axis)
            flux[right_name] = face[..., 0, 3] / norm0
        if wall_left is not None:
            face = left.narrow(axis, 0, 1).squeeze(axis)
            flux[left_name] = face[..., 0, 3] / norm0
    return flux


def wall_heat_flux_summary(coeff, basis, derivative, signs, weights, dx, dy,
                           boundary, mu, kappa, penalty, wall_gradient="face",
                           wall_penalty=None, periodic=(False, False)):
    """Tangential-cell average of :func:`wall_heat_flux` as plain floats."""
    flux = wall_heat_flux(coeff, basis, derivative, signs, weights, dx, dy,
                          boundary, mu, kappa, penalty, wall_gradient,
                          wall_penalty, periodic)
    return {name: float(value.mean()) for name, value in flux.items()}


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
            mu, kappa, penalty, wall_gradient="face", wall_penalty=None,
            source_quad=None, wall_adjoint=False):
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
    right_x, left_x, adj_rx, adj_lx = _axis_fluxes_ns(
        coeff, basis, derivative, signs, weights, dx, dy, periodic[0],
        boundary["xmin"], boundary["xmax"], mu, kappa, penalty, 0,
        wall_gradient, wall_penalty, wall_adjoint,
    )
    right_y, left_y, adj_ry, adj_ly = _axis_fluxes_ns(
        coeff, basis, derivative, signs, weights, dx, dy, periodic[1],
        boundary["ymin"], boundary["ymax"], mu, kappa, penalty, 1,
        wall_gradient, wall_penalty, wall_adjoint,
    )
    _, _, derivative_plus, derivative_minus = _face_weights(
        degree, signs, coeff.device, coeff.dtype
    )

    face_x = (
        right_x.unsqueeze(2)
        - left_x.unsqueeze(2) * signs.view(1, 1, -1, 1, 1)
        - volume_x
        + viscous_x
        + adj_rx.unsqueeze(2) * derivative_plus.view(1, 1, -1, 1, 1)
        - adj_lx.unsqueeze(2) * derivative_minus.view(1, 1, -1, 1, 1)
    )
    face_y = (
        right_y.unsqueeze(3)
        - left_y.unsqueeze(3) * signs.view(1, 1, 1, -1, 1)
        - volume_y
        + viscous_y
        + adj_ry.unsqueeze(3) * derivative_plus.view(1, 1, 1, -1, 1)
        - adj_ly.unsqueeze(3) * derivative_minus.view(1, 1, 1, -1, 1)
    )
    scale_x = (scale / (2.0 * dx)).view(1, 1, *scale.shape, 1)
    scale_y = (scale / (2.0 * dy)).view(1, 1, *scale.shape, 1)
    residual = -(scale_x * face_x) - (scale_y * face_y)
    if source_quad is not None:
        # m2789: mass matrix of the tensor-product basis is
        # (dx dy / 4) * 4 / ((2i+1)(2j+1)), so the modal source contribution is
        # (2i+1)(2j+1)/4 * int int S phi_i phi_j dxi deta.
        volume = torch.einsum(
            "xyabv,a,b,ak,bl->xyklv", source_quad, weights, weights, basis, basis
        )
        factor = (scale / 4.0).view(1, 1, *scale.shape, 1)
        residual = residual + factor * volume
    return residual

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
    wall_penalty=None,
    source=None,
    quadrature=None,
    initial_projection="l2",
    wall_gradient="face",
    wall_adjoint=False,
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
    if wall_gradient == "br1":
        # m2831 alias for the legacy mirror-ghost closure
        wall_gradient = "one_sided"
    if wall_gradient not in ("taylor", "face", "one_sided", "recovered", "fd2"):
        raise ValueError(
            "wall_gradient must be 'taylor', 'face', 'one_sided', 'recovered' "
            "or 'fd2'"
        )
    coeff = _initial_coefficients(initial, x, y, dx, dy, nodes, weights, basis, dev, dtype)

    def impose(state):
        return _positivity_filter(state, basis, signs)

    h_min = min(dx, dy)
    time, steps, diffusive_steps, dt = 0.0, 0, 0, 0.0
    # m2789: physical coordinates of the quadrature nodes (used by ``source``).
    xi = nodes.to(dev, dtype)
    x_quad = (
        xmin
        + (torch.arange(nx, device=dev, dtype=dtype).view(nx, 1).unsqueeze(-1).unsqueeze(-1)
           + 0.5 * (xi.view(1, 1, -1, 1) + 1.0)) * dx
    ).expand(nx, ny, xi.numel(), xi.numel())
    y_quad = (
        ymin
        + (torch.arange(ny, device=dev, dtype=dtype).view(1, ny).unsqueeze(-1).unsqueeze(-1)
           + 0.5 * (xi.view(1, 1, 1, -1) + 1.0)) * dy
    ).expand(nx, ny, xi.numel(), xi.numel())

    def _source_quad(time_value):
        if source is None:
            return None
        values = source(x_quad, y_quad, torch.as_tensor(time_value, device=dev, dtype=dtype))
        if isinstance(values, (tuple, list)):
            values = torch.stack([torch.as_tensor(v, device=dev, dtype=dtype).expand_as(x_quad)
                                  for v in values], dim=-1)
        else:
            values = torch.as_tensor(values, device=dev, dtype=dtype)
        return values.reshape(nx, ny, xi.numel(), xi.numel(), 4)

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
            # dimensionless group S = dt * diffusivity * (p+1)**4 / h**2 is
            # measured at S_max = 0.16, 0.21, 0.24, 0.26 for p = 0..3 (SIPG
            # spectrum times the SSP-RK3 real-axis interval, see
            # docs/DG-SIPG-EXPLICIT-STABILITY.md), i.e. VISCOUS_STABILITY = 0.1
            # keeps a 1.6-2.6x margin -- not the 3-5x stated earlier, and the
            # lower end of the range is below the formerly claimed 0.3.
            rate_viscous = diffusivity * (degree + 1.0) ** 4 / (VISCOUS_STABILITY * h_min**2)
            if rate_viscous > 0.0:
                candidate = cfl_viscous / rate_viscous
                if candidate < dt:
                    dt = candidate
                    limited_by_diffusion = True
        dt = min(dt, final_time - time)
        k1 = _rhs_ns(coeff, basis, derivative, weights, signs, dx, dy, periodic,
                     boundary, mu, kappa, penalty, wall_gradient,
                     wall_penalty, wall_adjoint=wall_adjoint,
                     source_quad=_source_quad(time))
        u1 = impose(coeff + dt * k1)
        k2 = _rhs_ns(u1, basis, derivative, weights, signs, dx, dy, periodic,
                     boundary, mu, kappa, penalty, wall_gradient,
                     wall_penalty, wall_adjoint=wall_adjoint,
                     source_quad=_source_quad(time + dt))
        u2 = impose(0.75 * coeff + 0.25 * (u1 + dt * k2))
        k3 = _rhs_ns(u2, basis, derivative, weights, signs, dx, dy, periodic,
                     boundary, mu, kappa, penalty, wall_gradient,
                     wall_penalty, wall_adjoint=wall_adjoint,
                     source_quad=_source_quad(time + 0.5 * dt))
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

    wall_flux = wall_heat_flux_summary(
        coeff, basis, derivative, signs, weights, dx, dy, boundary, mu, kappa,
        penalty, wall_gradient, wall_penalty, periodic)

    metadata = {
        "ncells": (nx, ny),
        "degree": degree,
        "final_time": final_time,
        "final_time_reached": time,
        "has_source": source is not None,
        "steps": steps,
        "diffusive_steps": diffusive_steps,
        **step_metadata(final_time, steps, dt),
        "cfl": cfl,
        "cfl_viscous": cfl_viscous,
        "domain": domain,
        "periodic": tuple(periodic),
        "mu": mu,
        "kappa": kappa,
        "prandtl": prandtl,
        "penalty": penalty,
        "wall_gradient": wall_gradient,
        "wall_heat_flux": wall_flux,
        "wall_adjoint": wall_adjoint,
        "boundary": {key: value for key, value in boundary.items()},
        "flux": "hllc + br1/sipg viscous",
        "quadrature": order,
        "initial_projection": initial_projection,
        "device": str(dev),
        "dtype": str(dtype).replace("torch.", ""),
        "dofs": nx * ny * (degree + 1) ** 2 * 4,
    }
    return x, y, coeff, metadata

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
    penalty=None,
    wall_penalty=None,
    ymin=0.0,
    ymax=1.0,
    xmin=0.0,
    xmax=1.0,
    device="cpu",
    wall_gradient="face",
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
        penalty=penalty,
        wall_penalty=wall_penalty,
        wall_gradient=wall_gradient,
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
    penalty=None,
    wall_penalty=None,
    device="cpu",
    wall_gradient="face",
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
        penalty=penalty,
        wall_penalty=wall_penalty,
        wall_gradient=wall_gradient,
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
    penalty=None,
    wall_penalty=None,
    device="cpu",
    wall_gradient="face",
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
        penalty=penalty,
        wall_penalty=wall_penalty,
        wall_gradient=wall_gradient,
        device=device,
        dtype=dtype,
    )



def taylor_green_exact(x, y, time, amplitude, wavenumber, kinematic_viscosity,
                       pressure, background_temperature=1.0):
    """Taylor--Green vortex: exact solution of the incompressible Navier--Stokes.

    The 2D Taylor--Green vortex is a classical benchmark for viscous flow solvers.
    It has the exact solution (incompressible limit):

        u(x, y, t) = -A cos(k x) sin(k y) exp(-2 nu k^2 t)
        v(x, y, t) =  A sin(k x) cos(k y) exp(-2 nu k^2 t)
        p(x, y, t) = -A^2/4 (cos(2k x) + cos(2k y)) exp(-4 nu k^2 t) + p0

    For the compressible equations at low Mach number (A << c), this is an
    approximate solution with errors O(M^2) from the acoustic terms. With
    A ~ 0.1 and unit sound speed, the Mach number is ~0.1 and the compressibility
    error is ~1%.

    The velocity field is divergence-free by construction, so the continuity
    equation is satisfied exactly. The momentum equations reduce to the Stokes
    equations (the convective term u·∇u = -∇(A^2/8 (cos(2kx) + cos(2ky)) exp(-4νk²t))
    is exactly balanced by the pressure gradient), and the viscous term gives
    the exponential decay.

    Parameters
    ----------
    x, y : torch.Tensor
        Spatial coordinates (typically on [0, 2π/k] × [0, 2π/k] or [0, 1]² with k=2π).
    time : float
        Evaluation time.
    amplitude : float
        Initial velocity amplitude A.
    wavenumber : float
        Wavenumber k (typically 2π for unit domain).
    kinematic_viscosity : float
        Kinematic viscosity ν = μ/ρ.
    pressure : float
        Reference pressure p0 (background).
    background_temperature : float
        Background temperature T0 (for density = p/T).

    Returns
    -------
    dict with keys: density, velocity_x, velocity_y, pressure, temperature
    """
    k = wavenumber
    nu = kinematic_viscosity
    A = amplitude
    t = float(time)

    decay_vel = np.exp(-2.0 * nu * k**2 * t)
    decay_pres = np.exp(-4.0 * nu * k**2 * t)

    u = -A * torch.cos(k * x) * torch.sin(k * y) * decay_vel
    v = A * torch.sin(k * x) * torch.cos(k * y) * decay_vel
    p = pressure - 0.25 * A**2 * (torch.cos(2.0 * k * x) + torch.cos(2.0 * k * y)) * decay_pres

    # Low Mach: density ≈ p/T with T ≈ T0 (temperature variations are O(M^2))
    temperature = torch.full_like(x, background_temperature)
    density = p / temperature

    return {
        "density": density,
        "velocity_x": u,
        "velocity_y": v,
        "pressure": p,
        "temperature": temperature,
    }


def solve_taylor_green_dg_2d(
    ncells=(8, 8),
    degree=1,
    final_time=0.1,
    amplitude=0.1,
    wavenumber=2.0 * np.pi,
    mu=0.01,
    pressure=1.0,
    temperature=1.0,
    cfl=0.3,
    cfl_viscous=1.0,
    penalty=None,
    wall_penalty=None,
    domain=None,
    device="cpu",
    wall_gradient="face",
    dtype=torch.float64,
):
    """Taylor--Green vortex decay: 2D benchmark for convective + diffusive terms.

    The Taylor--Green vortex is the canonical test for viscous flow solvers:
    it exercises both the convective (nonlinear) and diffusive (viscous) terms
    simultaneously, with an exact solution for comparison. Unlike the shear wave
    (which is 1D and linear), the TGV has genuine 2D structure and nonlinear
    convective balance.

    Default domain is [0, 1]² with k = 2π (one full period in each direction).
    The flow is periodic in both directions.

    Parameters
    ----------
    ncells : tuple of int
        Number of cells (nx, ny).
    degree : int
        Polynomial degree (1, 2, or 3).
    final_time : float
        Integration time.
    amplitude : float
        Initial velocity amplitude (Mach number ~ A/c with c=1).
    wavenumber : float
        Wavenumber k (default 2π for unit domain).
    mu : float
        Dynamic viscosity.
    pressure : float
        Reference pressure.
    temperature : float
        Background temperature.
    cfl : float
        CFL number for convective step.
    cfl_viscous : float
        CFL number for viscous step.
    domain : tuple, optional
        ((xmin, xmax), (ymin, ymax)). Default ((0, 1), (0, 1)).
    device : str
        PyTorch device.
    wall_gradient : str
        Wall gradient treatment (not used for periodic, kept for API consistency).
    dtype : torch.dtype
        Data type.

    Returns
    -------
    x, y, coeff, metadata : as returned by solve_navier_stokes_dg_2d
    """
    if domain is None:
        domain = ((0.0, 1.0), (0.0, 1.0))

    density0 = pressure / temperature
    nu = mu / density0

    def initial(x, y):
        state = taylor_green_exact(x, y, 0.0, amplitude, wavenumber, nu, pressure, temperature)
        return (state["density"], state["velocity_x"], state["velocity_y"], state["pressure"])

    return solve_navier_stokes_dg_2d(
        initial,
        ncells=ncells,
        degree=degree,
        domain=domain,
        final_time=final_time,
        cfl=cfl,
        cfl_viscous=cfl_viscous,
        mu=mu,
        prandtl=0.71,
        periodic=(True, True),
        penalty=penalty,
        wall_penalty=wall_penalty,
        wall_gradient=wall_gradient,
        device=device,
        dtype=dtype,
    )
