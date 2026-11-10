"""Two-dimensional DG advection--diffusion with space--time dependent velocity.

Motivation
----------
TensorDG's existing 2-D advection solver (:mod:`tensordg.advection_2d`) assumes a
**constant** velocity and a *periodic* mesh, so the two classical DG verification
cases for variable-coefficient transport were out of reach:

* **solid-body rotation** in a square, ``u = -omega (y - y_c), v = omega (x - x_c)``
  -- a linear, divergence-free, *closed* flow whose exact solution after one
  revolution (``T = 2*pi/omega``) is the initial field itself;
* **deformational (cosine-hill) flow**, ``u = cos(pi t / T) sin^2(pi x) sin(2 pi y)``,
  ``v = -cos(pi t / T) sin^2(pi y) sin(2 pi x)`` -- the LeVeque deformation test.
  Because ``int_0^T cos(pi t / T) dt = 0`` the flow reverses exactly at ``T/2`` and
  the exact solution at ``T`` is again the initial field, even though the field is
  stretched into thin filaments at ``T/2``.  This makes the final-time L2 error a
  pure measure of the numerical dissipation incurred *while* the filaments are
  unresolved.

Both cases are impossible without a velocity field that varies in space (rotation)
and in time (deformation), and the deformation case needs a boundary treatment
that does not pollute a compactly supported field.

Discretisation
--------------
Modal Legendre basis on a uniform Cartesian mesh, strong-form DG with an upwind
(Rusanov-type) numerical flux, classical RK4.  The velocity is a *callable*
``velocity(t, x, y) -> (a_x, a_y)`` evaluated at the quadrature / face nodes, so
arbitrary divergence-free fields are allowed.  An optional constant diffusivity
``nu`` is added with symmetric interior penalty (SIPG, penalty ``sigma = P (p+1)^2``).

Conservation
------------
The upwind flux is single-valued across a face, so the DG update is conservative
cell-by-cell; with the ``"extrapolate"`` boundary (the ghost trace is the interior
trace, i.e. a transparent boundary) the only mass change is the physical flux
through the domain boundary.  For compactly supported data away from the boundary
this is zero to round-off, which is what the mass-drift criterion checks.

Public API
----------
``solve`` : the solver (returns coefficients, mass history, metadata).
``l2_error`` / ``mass`` / ``project`` : verification utilities.
``solid_body_rotation_velocity``, ``deformational_flow_velocity``,
``cosine_bell``, ``gaussian_hill``, ``sine_wave`` : case definitions.
``run_solid_body_rotation``, ``run_deformational_flow``,
``run_advection_diffusion_decay`` : the three benchmark drivers used by the
tests and by ``benchmarks/dg/``.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

Velocity = Callable[[float, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
ScalarField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

UNIT_SQUARE = (0.0, 0.0, 1.0, 1.0)  # x0, y0, Lx, Ly


# --------------------------------------------------------------------------- #
# one-dimensional reference operators
# --------------------------------------------------------------------------- #
class Operators:
    """Reference-element operators for the modal Legendre basis on ``[-1,1]``."""

    def __init__(self, degree: int, device: torch.device, dtype: torch.dtype):
        nq = max(2 * degree + 3, 6)
        nodes_np, weights_np = np.polynomial.legendre.leggauss(nq)
        self.degree = degree
        self.nq = nq
        self.nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
        self.weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
        values, derivatives = [], []
        for k in range(degree + 1):
            polynomial = np.zeros(k + 1)
            polynomial[-1] = 1.0
            derivative = np.polynomial.legendre.legder(polynomial)
            values.append(torch.as_tensor(
                np.polynomial.legendre.legval(nodes_np, polynomial),
                device=device, dtype=dtype,
            ))
            derivatives.append(torch.as_tensor(
                np.polynomial.legendre.legval(nodes_np, derivative)
                if len(derivative) else np.zeros_like(nodes_np),
                device=device, dtype=dtype,
            ))
        self.phi = torch.stack(values, dim=1)        # (nq, p+1)
        self.dphi = torch.stack(derivatives, dim=1)  # (nq, p+1)
        self.wphi = self.weights[:, None] * self.phi
        self.wdphi = self.weights[:, None] * self.dphi
        # V[k,i] = int phi_i phi_k' dxi      (strong-form volume matrix)
        self.volume = torch.einsum("q,qi,qk->ki", self.weights, self.phi, self.dphi)
        # S[k,i] = int phi_k' phi_i' dxi     (diffusion stiffness)
        self.stiffness = torch.einsum("q,qk,qi->ki", self.weights, self.dphi, self.dphi)
        # Mm[k,i] = int phi_k phi_i dxi      (diagonal: 2/(2k+1))
        self.mass = torch.einsum("q,qk,qi->ki", self.weights, self.phi, self.phi)
        k = torch.arange(degree + 1, device=device, dtype=dtype)
        self.phi_right = torch.ones(degree + 1, device=device, dtype=dtype)
        self.phi_left = torch.where(
            (k.to(torch.int64) % 2) == 0, torch.ones_like(k), -torch.ones_like(k),
        )
        # P_k'(1) = k(k+1)/2,  P_k'(-1) = (-1)^{k+1} k(k+1)/2
        slopes = 0.5 * k * (k + 1.0)
        self.dphi_right = slopes
        self.dphi_left = torch.where(
            (k.to(torch.int64) % 2) == 0, -slopes, slopes,
        )


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _mesh_arrays(nx: int, ny: int, domain, device, dtype):
    x0, y0, lx, ly = domain
    hx, hy = lx / nx, ly / ny
    xc = x0 + (torch.arange(nx, device=device, dtype=dtype) + 0.5) * hx
    yc = y0 + (torch.arange(ny, device=device, dtype=dtype) + 0.5) * hy
    xf = x0 + torch.arange(nx + 1, device=device, dtype=dtype) * hx
    yf = y0 + torch.arange(ny + 1, device=device, dtype=dtype) * hy
    return hx, hy, xc, yc, xf, yf


def _quad_points(xc, yc, hx, hy, ops):
    """Quadrature coordinates ``(nx,1,nq,1)`` / ``(1,ny,1,nq)`` -> ``(nx,ny,nq,nq)``.

    Axis 2 is the x-quadrature index and axis 3 the y-quadrature index, matching
    the ``"xyab"`` einsum convention used throughout this module.
    """
    xq = xc[:, None, None, None] + 0.5 * hx * ops.nodes[None, None, :, None]
    yq = yc[None, :, None, None] + 0.5 * hy * ops.nodes[None, None, None, :]
    return xq, yq


def _face_points(xf, yf, xc, yc, hx, hy, ops):
    """x-face nodes ``(nx+1, 1, nq)`` and y-face nodes ``(1, ny+1, nq)``."""
    yf_nodes = yc[None, :, None] + 0.5 * hy * ops.nodes[None, None, :]
    xf_nodes = xc[:, None, None] + 0.5 * hx * ops.nodes[None, None, :]
    return xf[:, None, None], yf_nodes, xf_nodes, yf[None, :, None]


# --------------------------------------------------------------------------- #
# projection and diagnostics
# --------------------------------------------------------------------------- #
def project(field: ScalarField, nx: int, ny: int, degree: int,
            domain=UNIT_SQUARE, device: str | torch.device = "cpu",
            dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, dict]:
    """L2-project ``field(x, y)`` onto the modal basis (degrees <= ``degree``)."""
    device = torch.device(device)
    ops = Operators(degree, device, dtype)
    hx, hy, xc, yc, xf, yf = _mesh_arrays(nx, ny, domain, device, dtype)
    xq, yq = _quad_points(xc, yc, hx, hy, ops)
    values = field(xq, yq)
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values, device=device, dtype=dtype)
    values = torch.broadcast_to(values, (nx, ny, ops.nq, ops.nq))
    gram = torch.einsum("xyir,ik,rm->xykm", values, ops.wphi, ops.wphi)
    k = torch.arange(degree + 1, device=device, dtype=dtype)
    scale_k = (2.0 * k + 1.0) / 2.0
    coeff = gram * scale_k[None, None, :, None] * scale_k[None, None, None, :]
    return coeff, {"nx": nx, "ny": ny, "degree": degree, "hx": hx, "hy": hy,
                   "domain": tuple(float(v) for v in domain)}


def cell_quadrature(coeff: torch.Tensor, ops: Operators) -> torch.Tensor:
    """Values of the DG field at the Gauss--Legendre nodes, shape ``(nx,ny,nq,nq)``."""
    return torch.einsum("xykl,ak,bl->xyab", coeff, ops.phi, ops.phi)


def mass_total(coeff: torch.Tensor, hx: float, hy: float) -> float:
    """Total integral of the DG field (exact for the modal representation)."""
    return float((coeff[:, :, 0, 0].sum() * 0.5 * hx * hy).item())


def l2_error(coeff: torch.Tensor, exact: ScalarField, ops: Operators,
             hx: float, hy: float, nx: int, ny: int,
             domain=UNIT_SQUARE, device="cpu", dtype=torch.float64) -> dict:
    """Quadrature-based absolute/relative L2 error against ``exact(x, y)``."""
    device = torch.device(device)
    x0, y0, _, _ = domain
    xc = x0 + (torch.arange(nx, device=device, dtype=dtype) + 0.5) * hx
    yc = y0 + (torch.arange(ny, device=device, dtype=dtype) + 0.5) * hy
    xq, yq = _quad_points(xc, yc, hx, hy, ops)
    numerical = cell_quadrature(coeff, ops)
    reference = exact(xq, yq)
    if not isinstance(reference, torch.Tensor):
        reference = torch.as_tensor(reference, device=device, dtype=dtype)
    reference = torch.broadcast_to(reference, numerical.shape)
    w2 = ops.weights[None, None, :, None] * ops.weights[None, None, None, :]
    scale = 0.25 * hx * hy
    err2 = scale * torch.sum(w2 * (numerical - reference) ** 2)
    ref2 = scale * torch.sum(w2 * reference ** 2)
    abs_err = float(torch.sqrt(err2).item())
    ref_norm = float(torch.sqrt(ref2).item())
    return {
        "absolute_l2": abs_err,
        "reference_l2": ref_norm,
        "relative_l2": abs_err / ref_norm if ref_norm > 0 else float("nan"),
        "max_abs_error": float(torch.max(torch.abs(numerical - reference)).item()),
    }


# --------------------------------------------------------------------------- #
# semi-discrete operator
# --------------------------------------------------------------------------- #
def _extend(states_left_trace, states_right_trace, boundary):
    """Ghost traces at the two x-faces (cell-local traces -> face states)."""
    if boundary == "periodic":
        face_left = torch.cat([states_right_trace[-1:], states_right_trace], dim=0)
        face_right = torch.cat([states_left_trace, states_left_trace[:1]], dim=0)
    elif boundary == "extrapolate":
        face_left = torch.cat([states_left_trace[:1], states_right_trace], dim=0)
        face_right = torch.cat([states_left_trace, states_right_trace[-1:]], dim=0)
    else:
        raise ValueError(f"unknown boundary {boundary!r}")
    return face_left, face_right


def rhs(coeff: torch.Tensor, t: float, velocity: Velocity, nu: float,
        hx: float, hy: float, ops: Operators, boundary: str = "extrapolate",
        penalty: float = 2.0, geo=None) -> torch.Tensor:
    """Semi-discrete DG residual for ``u_t + div(a u) = nu * lap(u)``.

    Factor bookkeeping (reference element ``[-1,1]^2``, cell measure ``hx*hy/4``):
    with ``L = (2k+1)(2m+1)/(hx*hy)`` the modal equation reads

        du_km/dt = L * [ (hy/2) X + (hx/2) Y ]                            (advection)
        du_km/dt = L * [ nu (hy/hx) Vx - nu (hy/2) Fx + (x <-> y) ]       (diffusion)

    where ``X = int int a_x u phi_k' phi_m - phi_k(1) int f*_R phi_m
    + phi_k(-1) int f*_L phi_m`` and ``Vx = int int d_xi u d_xi phi_k phi_m``.
    """
    xc, yc, xf, yf = geo
    xq, yq = _quad_points(xc, yc, hx, hy, ops)
    xf_x, yf_nodes, xf_nodes, yf_y = _face_points(xf, yf, xc, yc, hx, hy, ops)
    ax, ay = velocity(t, xq, yq)

    uq = torch.einsum("xykl,ak,bl->xyab", coeff, ops.phi, ops.phi)
    fx_vol = torch.einsum("xyab,ak,bm->xykm", ax * uq, ops.wdphi, ops.wphi)
    fy_vol = torch.einsum("xyab,ak,bm->xykm", ay * uq, ops.wphi, ops.wdphi)

    # ---- x-faces (upwind / Rusanov) --------------------------------------- #
    u_right = torch.einsum("xykl,k->xyl", coeff, ops.phi_right)
    u_left = torch.einsum("xykl,k->xyl", coeff, ops.phi_left)
    face_l, face_r = _extend(u_left, u_right, boundary)
    lq = torch.einsum("jyl,bl->jyb", face_l, ops.phi)
    rq = torch.einsum("jyl,bl->jyb", face_r, ops.phi)
    ax_f, _ = velocity(t, xf_x, yf_nodes)          # (nx+1, ny, nq)
    ax_f = torch.broadcast_to(ax_f, lq.shape)
    flux_x = 0.5 * ax_f * (lq + rq) - 0.5 * torch.abs(ax_f) * (rq - lq)
    flux_x_modes = torch.einsum("jyb,bm->jym", flux_x, ops.wphi)
    fx_face = (-torch.einsum("k,xym->xykm", ops.phi_right, flux_x_modes[1:])
               + torch.einsum("k,xym->xykm", ops.phi_left, flux_x_modes[:-1]))

    # ---- y-faces ---------------------------------------------------------- #
    u_top = torch.einsum("xykl,l->xyk", coeff, ops.phi_right)
    u_bot = torch.einsum("xykl,l->xyk", coeff, ops.phi_left)
    bot_l, bot_r = _extend(u_bot.transpose(0, 1), u_top.transpose(0, 1), boundary)
    bq = torch.einsum("jxk,ak->xja", bot_l, ops.phi)
    tq = torch.einsum("jxk,ak->xja", bot_r, ops.phi)
    _, ay_f = velocity(t, xf_nodes, yf_y)          # (nx, ny+1, nq)
    ay_f = torch.broadcast_to(ay_f, bq.shape)
    flux_y = 0.5 * ay_f * (bq + tq) - 0.5 * torch.abs(ay_f) * (tq - bq)
    flux_y_modes = torch.einsum("xja,ak->xjk", flux_y, ops.wphi)
    fy_face = (-torch.einsum("l,xyk->xykl", ops.phi_right, flux_y_modes[:, 1:])
               + torch.einsum("l,xyk->xykl", ops.phi_left, flux_y_modes[:, :-1]))

    k = torch.arange(ops.degree + 1, device=coeff.device, dtype=coeff.dtype)
    fac = (2.0 * k + 1.0)[None, None, :, None] * (2.0 * k + 1.0)[None, None, None, :]
    out = fac * (0.5 / hx) * (fx_vol + fx_face) + fac * (0.5 / hy) * (fy_vol + fy_face)

    if nu > 0.0:
        out = out + _diffusion(coeff, nu, hx, hy, ops, boundary, penalty, geo)
    return out


def _diffusion(coeff, nu, hx, hy, ops, boundary, penalty, geo):
    """SIPG diffusion ``nu * lap(u)`` with penalty ``sigma = penalty*(p+1)^2``."""
    xc, yc, xf, yf = geo
    xf_x, yf_nodes, xf_nodes, yf_y = _face_points(xf, yf, xc, yc, hx, hy, ops)
    degree = ops.degree
    sigma = penalty * (degree + 1) ** 2

    d_uq = torch.einsum("xykl,ak,bl->xyab", coeff, ops.dphi, ops.phi)
    e_uq = torch.einsum("xykl,ak,bl->xyab", coeff, ops.phi, ops.dphi)
    vol_x = torch.einsum("xyab,ak,bm->xykm", d_uq, ops.wdphi, ops.wphi)
    vol_y = torch.einsum("xyab,ak,bm->xykm", e_uq, ops.wphi, ops.wdphi)

    # x-faces: {du/dx} - sigma/h [u]
    d_right = torch.einsum("xykl,k->xyl", coeff, ops.dphi_right)
    d_left = torch.einsum("xykl,k->xyl", coeff, ops.dphi_left)
    u_right = torch.einsum("xykl,k->xyl", coeff, ops.phi_right)
    u_left = torch.einsum("xykl,k->xyl", coeff, ops.phi_left)
    face_l, face_r = _extend(u_left, u_right, boundary)
    dface_l, dface_r = _extend(d_left, d_right, boundary)
    lq = torch.einsum("jyl,bl->jyb", face_l, ops.phi)
    rq = torch.einsum("jyl,bl->jyb", face_r, ops.phi)
    dlq = torch.einsum("jyl,bl->jyb", dface_l, ops.phi)
    drq = torch.einsum("jyl,bl->jyb", dface_r, ops.phi)
    avg = 0.5 * (dlq + drq) * (2.0 / hx)
    flux_x = avg - sigma / hx * (lq - rq)
    flux_x_modes = torch.einsum("jyb,bm->jym", flux_x, ops.wphi)
    face_x = (-torch.einsum("k,xym->xykm", ops.phi_right, flux_x_modes[1:])
              + torch.einsum("k,xym->xykm", ops.phi_left, flux_x_modes[:-1]))

    ud_top = torch.einsum("xykl,l->xyk", coeff, ops.phi_right)
    ud_bot = torch.einsum("xykl,l->xyk", coeff, ops.phi_left)
    d_top = torch.einsum("xykl,l->xyk", coeff, ops.dphi_right)
    d_bot = torch.einsum("xykl,l->xyk", coeff, ops.dphi_left)
    bot_l, bot_r = _extend(ud_bot.transpose(0, 1), ud_top.transpose(0, 1), boundary)
    dbot_l, dbot_r = _extend(d_bot.transpose(0, 1), d_top.transpose(0, 1), boundary)
    bq = torch.einsum("jxk,ak->xja", bot_l, ops.phi)
    tq = torch.einsum("jxk,ak->xja", bot_r, ops.phi)
    dbq = torch.einsum("jxk,ak->xja", dbot_l, ops.phi)
    dtq = torch.einsum("jxk,ak->xja", dbot_r, ops.phi)
    avg_y = 0.5 * (dbq + dtq) * (2.0 / hy)
    flux_y = avg_y - sigma / hy * (bq - tq)
    flux_y_modes = torch.einsum("xja,ak->xjk", flux_y, ops.wphi)
    face_y = (-torch.einsum("l,xyk->xykl", ops.phi_right, flux_y_modes[:, 1:])
              + torch.einsum("l,xyk->xykl", ops.phi_left, flux_y_modes[:, :-1]))

    k = torch.arange(degree + 1, device=coeff.device, dtype=coeff.dtype)
    fac = (2.0 * k + 1.0)[None, None, :, None] * (2.0 * k + 1.0)[None, None, None, :]
    # weak form: du_km/dt = L * ( W2 - W1 ),  W1 = int nu grad(u).grad(phi),
    # W2 = oint nu (grad(u).n) phi  (see the docstring of :func:`rhs`)
    x_diff = -nu * ((hy / hx) * vol_x + 0.5 * hy * face_x)
    y_diff = -nu * ((hx / hy) * vol_y + 0.5 * hx * face_y)
    return fac / (hx * hy) * (x_diff + y_diff)


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #
def _max_speed(velocity, t, geo, domain, device, dtype, ops):
    x0, y0, lx, ly = domain
    n = 32
    xs = torch.linspace(x0, x0 + lx, n, device=device, dtype=dtype)[:, None]
    ys = torch.linspace(y0, y0 + ly, n, device=device, dtype=dtype)[None, :]
    ax, ay = velocity(t, xs, ys)
    ax = torch.broadcast_to(ax, (n, n))
    ay = torch.broadcast_to(ay, (n, n))
    return float(torch.max(torch.hypot(ax, ay)).item())


def solve(nx: int = 32, ny: int | None = None, degree: int = 1,
          final_time: float = 1.0, velocity: Velocity | None = None,
          initial: ScalarField | None = None, nu: float = 0.0,
          cfl: float = 0.4, domain=UNIT_SQUARE, boundary: str = "extrapolate",
          device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64,
          penalty: float = 2.0, record_every: int | None = None,
          seed: int | None = None) -> dict:
    """March ``u_t + div(a(x,y,t) u) = nu lap(u)`` to ``final_time``.

    ``dt`` is fixed from the initial-time maximum speed,
    ``dt = cfl / ((2p+1) (max|a_x|/hx + max|a_y|/hy))`` for advection, further
    limited by an explicit-RK4 diffusion bound when ``nu > 0``.
    """
    if seed is not None:  # deterministic reference for the metadata audit trail
        torch.manual_seed(seed)
    device = torch.device(device)
    ny = nx if ny is None else ny
    if velocity is None or initial is None:
        raise ValueError("velocity and initial callables are required")
    if degree not in (0, 1, 2, 3):
        raise ValueError("degree must be in [0, 3]")
    ops = Operators(degree, device, dtype)
    hx, hy, xc, yc, xf, yf = _mesh_arrays(nx, ny, domain, device, dtype)
    geo = (xc, yc, xf, yf)
    coeff, _ = project(initial, nx, ny, degree, domain, device, dtype)

    speed = _max_speed(velocity, 0.0, geo, domain, device, dtype, ops)
    dt_adv = (cfl / ((2 * degree + 1) * (speed / min(hx, hy)))
              if speed > 0.0 else float("inf"))
    dt = dt_adv
    if nu > 0.0:
        # explicit-RK4 diffusion bound.  The spectral radius of the SIPG operator
        # measures ~50*sigma*nu/h^2 for p in [0,3] (power iteration, see the
        # benchmark report), so dt <= 0.5*2.78*h^2/(50*sigma*nu).
        sigma = penalty * (degree + 1) ** 2
        dt_diff = 0.5 * 2.78 * min(hx, hy) ** 2 / (50.0 * sigma * nu)
        dt = min(dt, dt_diff)
    steps = max(1, int(math.ceil(final_time / dt - 1e-12)))
    dt = final_time / steps
    if record_every is None:
        record_every = max(1, steps // 20)

    def f(state, time):
        return rhs(state, time, velocity, nu, hx, hy, ops, boundary, penalty, geo)

    mass0 = mass_total(coeff, hx, hy)
    history_t = [0.0]
    history_mass = [mass0]
    for step in range(steps):
        time = step * dt
        k1 = f(coeff, time)
        k2 = f(coeff + 0.5 * dt * k1, time + 0.5 * dt)
        k3 = f(coeff + 0.5 * dt * k2, time + 0.5 * dt)
        k4 = f(coeff + dt * k3, time + dt)
        coeff = coeff + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if (step + 1) % record_every == 0 or step + 1 == steps:
            history_t.append((step + 1) * dt)
            history_mass.append(mass_total(coeff, hx, hy))

    metadata = {
        "nx": nx, "ny": ny, "degree": degree, "final_time": final_time,
        "nu": nu, "cfl": cfl, "dt": dt, "steps": steps, "boundary": boundary,
        "max_speed": speed, "penalty": penalty,
        "domain": tuple(float(v) for v in domain),
        "device": str(device), "dtype": str(dtype), "seed": seed,
        "mass_initial": mass0, "mass_final": history_mass[-1],
        "mass_history": history_mass, "time_history": history_t,
        "hx": hx, "hy": hy, "ops": ops, "geo": geo,
    }
    return {"coeff": coeff, "metadata": metadata}


# --------------------------------------------------------------------------- #
# case definitions
# --------------------------------------------------------------------------- #
def cosine_bell(center: tuple[float, float] = (0.5, 0.75), radius: float = 0.15,
                amplitude: float = 1.0) -> ScalarField:
    """LeVeque cosine bell, compactly supported in ``|x - center| < radius``."""
    cx, cy = center

    def field(x, y):
        r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return torch.where(
            r < radius, 0.5 * amplitude * (1.0 + torch.cos(math.pi * r / radius)),
            torch.zeros_like(r),
        )
    return field


def gaussian_hill(center: tuple[float, float] = (0.5, 0.5), sigma: float = 0.1,
                  amplitude: float = 1.0) -> ScalarField:
    """Smooth Gaussian bump (used for the pure-diffusion anchor)."""
    cx, cy = center

    def field(x, y):
        return amplitude * torch.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2)
        )
    return field


def sine_wave(kx: float = 2.0 * math.pi, ky: float = 2.0 * math.pi,
              amplitude: float = 1.0) -> ScalarField:
    """Periodic single-mode test field ``A sin(kx x) sin(ky y)``."""
    def field(x, y):
        return amplitude * torch.sin(kx * x) * torch.sin(ky * y)
    return field


def solid_body_rotation_velocity(omega: float = 2.0 * math.pi,
                                 center: tuple[float, float] = (0.5, 0.5),
                                 radius: float | None = None) -> Velocity:
    """Rigid rotation ``a = (-omega (y-y_c), omega (x-x_c))`` (divergence free)."""
    cx, cy = center

    def velocity(t, x, y):
        return -omega * (y - cy), omega * (x - cx)
    return velocity


def deformational_flow_velocity(period: float = 1.0,
                                amplitude: float = 1.0) -> Velocity:
    """LeVeque deformation field, time modulated by ``cos(pi t / period)``.

    ``u = amplitude cos(pi t/T) sin^2(pi x) sin(2 pi y)``,
    ``v = -amplitude cos(pi t/T) sin^2(pi y) sin(2 pi x)``.
    The modulation makes the flow reversible: ``int_0^T cos(pi t/T) dt = 0``, so
    the exact solution at ``t = T`` equals the initial condition.
    """
    def velocity(t, x, y):
        modulation = amplitude * math.cos(math.pi * t / period)
        ax = modulation * torch.sin(math.pi * x) ** 2 * torch.sin(2.0 * math.pi * y)
        ay = -modulation * torch.sin(math.pi * y) ** 2 * torch.sin(2.0 * math.pi * x)
        return ax, ay
    return velocity


def rotated_field(field: ScalarField, omega: float, t: float,
                  center: tuple[float, float] = (0.5, 0.5)) -> ScalarField:
    """Exact solution at time ``t`` of rigid rotation: ``u0(R(-omega t)(x-c)+c)``.

    For ``omega t = 2 pi k`` this returns the initial field, which is the anchor
    used by the solid-body-rotation criterion; for other times it is the exact
    analytic solution of the rotation problem (a non-trivial check that the
    solver transports along circular characteristics).
    """
    cx, cy = center
    theta = -omega * t
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def rotated(x, y):
        dx, dy = x - cx, y - cy
        return field(cx + cos_t * dx - sin_t * dy, cy + sin_t * dx + cos_t * dy)
    return rotated


def vanishing_velocity() -> Velocity:
    def velocity(t, x, y):
        return torch.zeros_like(x + y), torch.zeros_like(x + y)
    return velocity


def translation_velocity(speed: tuple[float, float]) -> Velocity:
    ax, ay = speed

    def velocity(t, x, y):
        return torch.full_like(x + y, ax), torch.full_like(x + y, ay)
    return velocity


# --------------------------------------------------------------------------- #
# benchmark drivers
# --------------------------------------------------------------------------- #
def _finish_metrics(out: dict, field: ScalarField, name: str, extra: dict | None = None):
    meta = out["metadata"]
    ops = meta["ops"]
    err = l2_error(out["coeff"], field, ops, meta["hx"], meta["hy"],
                   meta["nx"], meta["ny"], meta["domain"],
                   meta["device"], out["coeff"].dtype)
    mass0, mass1 = meta["mass_initial"], meta["mass_final"]
    drift = abs(mass1 - mass0) / max(abs(mass0), 1e-30)
    metrics = {
        "case": name,
        "relative_l2": err["relative_l2"],
        "absolute_l2": err["absolute_l2"],
        "max_abs_error": err["max_abs_error"],
        "mass_initial": mass0,
        "mass_final": mass1,
        "mass_drift": drift,
        "mesh": [meta["nx"], meta["ny"]],
        "degree": meta["degree"],
        "dt": meta["dt"],
        "steps": meta["steps"],
        "final_time": meta["final_time"],
        "cfl": meta["cfl"],
        "nu": meta["nu"],
        "boundary": meta["boundary"],
    }
    if extra:
        metrics.update(extra)
    return metrics


def run_solid_body_rotation(ncells: int = 32, degree: int = 1,
                            omega: float = 2.0 * math.pi,
                            revolutions: float = 1.0, cfl: float = 0.4,
                            boundary: str = "extrapolate", nu: float = 0.0,
                            device="cpu", dtype=torch.float64,
                            return_state: bool = False):
    """One full revolution of a cosine bell; exact final state = initial state."""
    field = cosine_bell(center=(0.5, 0.75), radius=0.15)
    velocity = solid_body_rotation_velocity(omega=omega)
    final_time = revolutions * 2.0 * math.pi / omega
    out = solve(ncells, ncells, degree, final_time, velocity, field, nu=nu,
                cfl=cfl, boundary=boundary, device=device, dtype=dtype)
    # exact solution at any time is the initial field rotated by -omega t
    reference = rotated_field(field, omega, final_time)
    metrics = _finish_metrics(out, reference, "solid_body_rotation",
                              {"omega": omega, "revolutions": revolutions})
    if return_state:
        return out, metrics
    return metrics


def run_deformational_flow(ncells: int = 48, degree: int = 2, period: float = 1.0,
                           radius: float = 0.15, cfl: float = 0.4,
                           boundary: str = "extrapolate", device="cpu",
                           dtype=torch.float64, return_state: bool = False,
                           snapshot_half: bool = False):
    """LeVeque deformation test: stretched at ``T/2``, recovered at ``T``."""
    field = cosine_bell(center=(0.5, 0.75), radius=radius)
    velocity = deformational_flow_velocity(period=period)
    out = solve(ncells, ncells, degree, period, velocity, field, cfl=cfl,
                boundary=boundary, device=device, dtype=dtype)
    metrics = _finish_metrics(out, field, "deformational_flow",
                              {"period": period, "bell_radius": radius})
    if snapshot_half:
        # The exact field at T/2 is a strongly stretched filament (no closed form);
        # record its peak amplitude as a stretching indicator for the figure only.
        mid = solve(ncells, ncells, degree, 0.5 * period, velocity, field,
                    cfl=cfl, boundary=boundary, device=device, dtype=dtype)
        ops = mid["metadata"]["ops"]
        mid_values = cell_quadrature(mid["coeff"], ops)
        metrics["half_time_max_amplitude"] = float(mid_values.max().item())
        metrics["half_state_coeff"] = mid["coeff"]
        metrics["half_state_metadata"] = {k: v for k, v in mid["metadata"].items()
                                         if k in ("nx", "ny", "degree", "hx", "hy",
                                                  "domain", "device")}
    if return_state:
        return out, metrics
    return metrics


def run_advection_diffusion_decay(ncells: int = 16, degree: int = 2,
                                  nu: float = 0.02, speed: tuple[float, float] = (1.0, 0.5),
                                  final_time: float = 0.25, cfl: float = 0.4,
                                  device="cpu", dtype=torch.float64,
                                  return_state: bool = False):
    """Periodic advection--diffusion of one Fourier mode with an exact solution.

    ``u_t + a.u_x = nu lap u`` with ``u0 = sin(2 pi x) sin(2 pi y)`` gives
    ``u = exp(-nu (kx^2+ky^2) t) sin(kx (x-a_x t)) sin(ky (y-a_y t))``; the
    exponential decay rate is a direct check of the SIPG diffusion operator.
    """
    kx = ky = 2.0 * math.pi
    decay = nu * (kx ** 2 + ky ** 2)
    ax, ay = speed

    def exact(x, y):
        return (math.exp(-decay * final_time)
                * torch.sin(kx * (x - ax * final_time))
                * torch.sin(ky * (y - ay * final_time)))

    out = solve(ncells, ncells, degree, final_time, translation_velocity(speed),
                sine_wave(kx, ky), nu=nu, cfl=cfl, boundary="periodic",
                device=device, dtype=dtype)
    metrics = _finish_metrics(out, exact, "advection_diffusion_decay",
                              {"kx": kx, "ky": ky, "decay_rate_exact": decay})
    # measured decay rate from the numerical amplitude
    ops = out["metadata"]["ops"]
    num_mass = mass_total(out["coeff"], out["metadata"]["hx"], out["metadata"]["hy"])
    metrics["amplitude_numerical"] = num_mass
    if return_state:
        return out, metrics
    return metrics

# Explicit public alias: the solver is the module's main entry point.
solve_advection_diffusion = solve
