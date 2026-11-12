"""Steady Kovasznay flow: the DG-NS verification case ``CASE-DG-KOVASZNAY-NS``.

Kovasznay (1948) found the one exact, steady, non-parallel solution of the
incompressible Navier--Stokes equations that is available in closed form: a
decaying array of vortices superposed on a uniform stream.  With the uniform
stream along ``x`` and the vortex wavenumber ``k``,

    lam = (Re - sqrt(Re**2 + 4 k**2)) / 2                    (decay rate)
    u   = 1 - exp(lam x) cos(k y)
    v   = (lam / k) exp(lam x) sin(k y)
    p   = p_ref - exp(2 lam x) / 2                            (density = 1)

is an exact solution of

    u u_x + v u_y = -p_x + nu (u_xx + u_yy)
    u v_x + v v_y = -p_y + nu (v_xx + v_yy)
    u_x + v_y = 0,                     nu = 1 / Re

on any rectangle.  The classical test domain is ``x in [-0.5, 1]``,
``y in [-0.5, 1.5]`` (two periods of ``k = 2 pi``), which is what
:func:`default_domain` returns: Dirichlet velocity on the streamwise ends and
periodicity in ``y``.  :func:`kovasznay_residual` evaluates the three PDE
residuals of the closed form with autograd, so the closed form is used as the
reference only after it has been shown to satisfy the equations to round-off --
the reference is a formula, not a transcribed table.

``tensordg.ns_dg_2d`` solves the *compressible* Navier--Stokes equations.  The
incompressible solution above is therefore recovered in the low-Mach limit: the
thermodynamic pressure is written as ``p = p0 + p_dyn`` with
``p0 = rho0 R T0 = c**2 / gamma`` large enough that the density stays at
``rho0`` to ``O(M**2)``, and the Mach number ``M = max|u| / c`` becomes a
*discretisation parameter* of the verification.  :func:`solve_kovasznay_dg_2d`
exposes it as ``sound_speed``; the benchmark sweeps it and shows the velocity
error collapsing towards the incompressible value like ``M**2``.

The through-flow boundary conditions need a face that carries mass, momentum
and energy, which ``ns_dg_2d``'s wall closure does not (it is a zero-mass-flux
impermeable wall).  ``ns_dg_2d._boundary_flux`` therefore accepts an optional
``"state"`` entry in a boundary dict; see the comment there.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .euler_1d import GAMMA
from .euler_dg_2d import legendre_operators
from .ns_dg_2d import _rhs_ns, solve_navier_stokes_dg_2d

__all__ = [
    "DEFAULT_REYNOLDS",
    "DEFAULT_WAVENUMBER",
    "CP",
    "default_domain",
    "decay_rate",
    "kovasznay_exact",
    "kovasznay_thermodynamic_state",
    "kovasznay_residual",
    "solve_kovasznay_dg_2d",
    "velocity_l2_error",
    "modal_evaluator",
]

CP = GAMMA / (GAMMA - 1.0)

#: Reynolds number of the benchmark case card (``Re = 1 / nu = 40``).
DEFAULT_REYNOLDS = 40.0
#: Kovasznay's vortex wavenumber, i.e. unit period in ``y``.
DEFAULT_WAVENUMBER = 2.0 * math.pi
#: Reference sound speed of the compressible embedding (``M = max|u| / c``).
DEFAULT_SOUND_SPEED = 60.0


def default_domain(wavenumber: float = DEFAULT_WAVENUMBER):
    """``((-0.5, 1.0), (-0.5, 1.5))`` for ``k = 2 pi``; two periods in ``y``."""
    period = 2.0 * math.pi / wavenumber
    return ((-0.5, 1.0), (-0.5, -0.5 + 2.0 * period))


def decay_rate(reynolds: float = DEFAULT_REYNOLDS,
               wavenumber: float = DEFAULT_WAVENUMBER) -> float:
    """Kovasznay decay rate ``lam = (Re - sqrt(Re**2 + 4 k**2)) / 2``.

    Negative for every ``Re``: the vortex array decays downstream.  The
    relation is ``lam**2 - Re lam - k**2 = 0``, i.e. exactly the balance
    between the convective term and the wall-normal diffusion that
    :func:`kovasznay_residual` checks numerically.
    """
    return 0.5 * (reynolds - math.sqrt(reynolds * reynolds + 4.0 * wavenumber ** 2))


def _as_tensor(value, like=None, dtype=torch.float64, device="cpu"):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def kovasznay_exact(x, y, reynolds: float = DEFAULT_REYNOLDS,
                    wavenumber: float = DEFAULT_WAVENUMBER,
                    pressure_offset: float = 0.0, dtype=torch.float64,
                    device="cpu"):
    """Closed-form steady Kovasznay solution (unit density, ``nu = 1/Re``).

    Returns ``{"u", "v", "p", "decay_rate", "density"}``; ``p`` is the
    *dynamic* pressure ``p_ref - exp(2 lam x)/2`` of the incompressible problem.
    """
    x = _as_tensor(x, dtype=dtype, device=device)
    y = _as_tensor(y, dtype=dtype, device=device)
    lam = decay_rate(reynolds, wavenumber)
    decay = torch.exp(lam * x)
    u = 1.0 - decay * torch.cos(wavenumber * y)
    v = (lam / wavenumber) * decay * torch.sin(wavenumber * y)
    p = pressure_offset - 0.5 * decay * decay
    return {
        "u": u,
        "v": v,
        "p": p,
        "density": torch.ones_like(u),
        "decay_rate": lam,
    }


def _grad(field, coordinate):
    """Autograd derivative; a field that does not depend on ``coordinate``
    (the pressure does not depend on ``y``) returns exact zeros."""
    value = torch.autograd.grad(
        field, coordinate, grad_outputs=torch.ones_like(field), create_graph=True,
        allow_unused=True,
    )[0]
    return torch.zeros_like(field) if value is None else value


def kovasznay_residual(reynolds: float = DEFAULT_REYNOLDS,
                       wavenumber: float = DEFAULT_WAVENUMBER,
                       points: int = 64, domain=None, seed: int = 20261112,
                       dtype=torch.float64):
    """Max residuals of the closed form in the three steady NS equations.

    Derivatives are taken with autograd on the closed form, so a zero residual
    is a machine-precision statement that the formula *is* the solution and the
    benchmark reference is not a transcribed number.  Residuals are reported
    both dimensional (``U**2 / L`` units) and normalised by the size of the
    individual terms (``|u u_x| + |v u_y| + |p_x| + nu |lap u|``), so the
    normalised value is the fraction of the momentum balance that is violated.
    """
    (xmin, xmax), (ymin, ymax) = domain or default_domain(wavenumber)
    generator = torch.Generator().manual_seed(seed)
    grid = torch.rand(points, points, generator=generator, dtype=dtype)
    x = (xmin + (xmax - xmin) * grid).requires_grad_(True)
    y = (ymin + (ymax - ymin) * grid.T).requires_grad_(True)
    exact = kovasznay_exact(x, y, reynolds, wavenumber, dtype=dtype)
    u, v, p = exact["u"], exact["v"], exact["p"]
    nu = 1.0 / reynolds
    u_x, u_y = _grad(u, x), _grad(u, y)
    v_x, v_y = _grad(v, x), _grad(v, y)
    u_xx, u_yy = _grad(u_x, x), _grad(u_y, y)
    v_xx, v_yy = _grad(v_x, x), _grad(v_y, y)
    p_x, p_y = _grad(p, x), _grad(p, y)
    continuity = u_x + v_y
    momentum_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    scale_x = (u * u_x).abs() + (v * u_y).abs() + p_x.abs() + nu * (u_xx + u_yy).abs()
    scale_y = (u * v_x).abs() + (v * v_y).abs() + p_y.abs() + nu * (v_xx + v_yy).abs()
    scale_c = u_x.abs() + v_y.abs()
    return {
        "continuity_max": float(continuity.abs().max()),
        "momentum_x_max": float(momentum_x.abs().max()),
        "momentum_y_max": float(momentum_y.abs().max()),
        "continuity_relative": float((continuity.abs() / scale_c).max()),
        "momentum_x_relative": float((momentum_x.abs() / scale_x).max()),
        "momentum_y_relative": float((momentum_y.abs() / scale_y).max()),
        "points": points,
        "decay_rate": decay_rate(reynolds, wavenumber),
        "reynolds": reynolds,
        "wavenumber": wavenumber,
    }


def kovasznay_thermodynamic_state(x, y, reynolds: float = DEFAULT_REYNOLDS,
                                  wavenumber: float = DEFAULT_WAVENUMBER,
                                  sound_speed: float = DEFAULT_SOUND_SPEED,
                                  density: float = 1.0, gas_constant: float = 1.0,
                                  dtype=torch.float64, device="cpu"):
    """Lift the incompressible solution into the compressible state variables.

    ``p0 = rho0 R T0 = c**2 / gamma`` and the incompressible dynamic pressure is
    added to it, so ``rho = rho0`` and ``T = p / (rho R)`` follow the pressure
    at ``O(M**2)``: this is the low-Mach embedding of Kovasznay's solution.
    """
    exact = kovasznay_exact(x, y, reynolds, wavenumber, dtype=dtype, device=device)
    rho = torch.full_like(exact["u"], density)
    t0 = sound_speed ** 2 / GAMMA
    p0 = density * gas_constant * t0
    return {
        "density": rho,
        "u": exact["u"],
        "v": exact["v"],
        "p": p0 + exact["p"],
        "pressure_offset": p0,
        "temperature_offset": t0,
        "dynamic_pressure": exact["p"],
        "sound_speed": sound_speed,
    }


# ---------------------------------------------------------------------------
# Modal evaluation (needed to restart the explicit march in chunks)
# ---------------------------------------------------------------------------

def _legendre_vander(points: np.ndarray, degree: int):
    return np.polynomial.legendre.legvander(points, degree)


def modal_evaluator(coeff, ncells=24, degree=3, domain=None,
                    wavenumber: float = DEFAULT_WAVENUMBER, dtype=torch.float64):
    """Return ``evaluate(x, y) -> (rho, u, v, p)`` for a DG solution.

    Used to restart the explicit march in chunks.  Re-projecting the returned
    values with ``ns_dg_2d``'s L2 projector reproduces ``coeff`` to round-off,
    because the solver's quadrature (``order + 1 >= degree + 3`` points) makes
    the modal mass matrix exact for the products of degree ``2 * degree``
    that appear in the projection.
    """
    (xmin, xmax), (ymin, ymax) = domain or default_domain(wavenumber)
    nx, ny = (ncells, ncells) if isinstance(ncells, int) else ncells
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    coefficients = coeff

    def evaluate(x, y):
        x = torch.as_tensor(x, dtype=dtype)
        y = torch.as_tensor(y, dtype=dtype)
        ix = torch.clamp(((x - xmin) / dx).long(), 0, nx - 1)
        iy = torch.clamp(((y - ymin) / dy).long(), 0, ny - 1)
        xi = 2.0 * ((x - xmin) - (ix.to(dtype) + 0.5) * dx) / dx
        eta = 2.0 * ((y - ymin) - (iy.to(dtype) + 0.5) * dy) / dy
        basis_x = _legendre_vander(xi.detach().cpu().numpy().ravel(), degree)
        basis_y = _legendre_vander(eta.detach().cpu().numpy().ravel(), degree)
        basis_x = torch.as_tensor(
            basis_x.reshape(*xi.shape, degree + 1), dtype=dtype).to(coefficients.device)
        basis_y = torch.as_tensor(
            basis_y.reshape(*eta.shape, degree + 1), dtype=dtype).to(coefficients.device)
        local = coefficients[ix, iy]
        state = torch.einsum("...klv,...k,...l->...v", local, basis_x, basis_y)
        rho = state[..., 0].clamp_min(1e-12)
        u = state[..., 1] / rho
        v = state[..., 2] / rho
        p = (GAMMA - 1.0) * (
            state[..., 3] - 0.5 * (state[..., 1] ** 2 + state[..., 2] ** 2) / rho
        )
        return rho, u, v, p

    return evaluate


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def _quadrature_grid(ncells, degree, domain, quadrature=None):
    (xmin, xmax), (ymin, ymax) = domain
    nx, ny = (ncells, ncells) if isinstance(ncells, int) else ncells
    order = quadrature if quadrature is not None else max(degree + 2, 3)
    nodes, weights, basis, derivative, signs = legendre_operators(degree, order)
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    xc = xmin + (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    yc = ymin + (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    xq = (xc[:, None] + 0.5 * dx * nodes[None, :])[:, None, :, None].expand(
        nx, ny, nodes.numel(), nodes.numel())
    yq = (yc[:, None] + 0.5 * dy * nodes[None, :])[None, :, None, :].expand(
        nx, ny, nodes.numel(), nodes.numel())
    return dict(x=xq.clone(), y=yq.clone(), nodes=nodes, weights=weights,
                basis=basis, derivative=derivative, signs=signs,
                dx=dx, dy=dy, nx=nx, ny=ny, order=order)


def velocity_l2_error(coeff, ncells=24, degree=3, domain=None,
                      reynolds: float = DEFAULT_REYNOLDS,
                      wavenumber: float = DEFAULT_WAVENUMBER,
                      quadrature=None, pressure_offset=None):
    """Quadrature-exact relative L2 errors of the DG field against Kovasznay.

    ``velocity`` is the vector norm ``||u_num - u_exact|| / ||u_exact||`` over
    both components; ``u`` and ``v`` are the per-component values; when
    ``pressure_offset`` is given the dynamic pressure is also compared.
    """
    grid = _quadrature_grid(ncells, degree, domain or default_domain(wavenumber),
                            quadrature)
    nodal = torch.einsum("xyklv,ak,bl->xyabv", coeff, grid["basis"], grid["basis"])
    rho = nodal[..., 0]
    u = nodal[..., 1] / rho
    v = nodal[..., 2] / rho
    p = (GAMMA - 1.0) * (
        nodal[..., 3] - 0.5 * (nodal[..., 1] ** 2 + nodal[..., 2] ** 2) / rho
    )
    exact = kovasznay_exact(grid["x"], grid["y"], reynolds, wavenumber)
    weights = grid["weights"]
    jacobian = (
        weights[:, None] * weights[None, :] * grid["dx"] * grid["dy"] / 4.0
    )

    def _relative(delta, reference):
        return float(
            torch.sqrt((jacobian * delta ** 2).sum()
                       / (jacobian * reference ** 2).sum())
        )

    out = {
        "u": _relative(u - exact["u"], exact["u"]),
        "v": _relative(v - exact["v"], exact["v"]),
        "velocity": _relative(
            torch.sqrt((u - exact["u"]) ** 2 + (v - exact["v"]) ** 2),
            torch.sqrt(exact["u"] ** 2 + exact["v"] ** 2),
        ),
    }
    if pressure_offset is not None:
        out["pressure_dynamic"] = _relative(
            (p - pressure_offset) - exact["p"], exact["p"]
        )
    out["max_velocity"] = float(torch.maximum(u.abs().max(), v.abs().max()))
    return out


# ---------------------------------------------------------------------------
# Solver driver
# ---------------------------------------------------------------------------

def solve_kovasznay_dg_2d(
    ncells=24,
    degree=3,
    reynolds: float = DEFAULT_REYNOLDS,
    wavenumber: float = DEFAULT_WAVENUMBER,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    final_time: float = 1.0,
    chunk_time=None,
    chunks: int = 1,
    cfl: float = 0.5,
    cfl_viscous: float = 1.0,
    prandtl: float = 0.71,
    outlet: str = "prescribed",
    initial_mode: str = "exact",
    domain=None,
    quadrature=None,
    penalty=None,
    wall_penalty=None,
    wall_gradient: str = "face",
    wall_adjoint: bool = False,
    density: float = 1.0,
    gas_constant: float = 1.0,
    monitor=None,
    device="cpu",
    dtype=torch.float64,
):
    """March the compressible NS solver to Kovasznay's steady state.

    Parameters
    ----------
    initial_mode : {"exact", "uniform"}
        ``"exact"`` starts the march on Kovasznay's solution (default: the march
        then only removes the projection residual, which is the cheap way to
        reach the discrete steady state).  ``"uniform"`` starts on the uniform
        stream ``(u, v) = (1, 0)`` instead, which is the initial-condition
        *independence* test: the boundary data alone must drive the solution to
        the same steady state.
    outlet : {"prescribed", "extrapolate"}
        ``"prescribed"`` closes the downstream end with the same HLLC
        prescribed-state face as the inlet (exact velocity, temperature and
        thermodynamic pressure; a subsonic characteristic-like closure).
        ``"extrapolate"`` uses ``ns_dg_2d``'s do-nothing outflow face
        (interior flux and interior normal derivative) -- a robustness check on
        how much the downstream closure contaminates the $L_2$ error.
    chunks, chunk_time, monitor
        The explicit march is restarted every ``chunk_time`` (default: the whole
        ``final_time`` in ``chunks`` equal pieces) so that the residual and the
        error can be sampled in time; ``monitor(time, coeff, metadata)`` is
        called after every chunk and its return value is appended to
        ``metadata["history"]``.  Restarting is lossless: the modal evaluator
        reproduces the polynomial to round-off before it is re-projected.

    Returns
    -------
    ``(x, y, coeff, metadata)`` like ``solve_navier_stokes_dg_2d``, plus
    ``metadata["kovasznay"]`` (case parameters) and ``metadata["history"]``.
    """
    if outlet not in ("prescribed", "extrapolate"):
        raise ValueError("outlet must be 'prescribed' or 'extrapolate'")
    if initial_mode not in ("exact", "uniform"):
        raise ValueError("initial_mode must be 'exact' or 'uniform'")
    if sound_speed <= 0.0:
        raise ValueError("sound speed must be positive")
    nx, ny = (ncells, ncells) if isinstance(ncells, int) else ncells
    net = (nx, ny)
    box = domain or default_domain(wavenumber)
    (xmin, xmax), (ymin, ymax) = box
    mu = 1.0 / reynolds
    order = quadrature if quadrature is not None else max(degree + 2, 3)
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, order, device, dtype
    )
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    y_centres = ymin + (torch.arange(ny, device=device, dtype=dtype) + 0.5) * dy
    # streamwise faces are quadrature points in y: the face array index ``b``
    # is exactly the y quadrature node, as in ``_face_primitives(axis=0)``
    y_face = (y_centres[:, None] + 0.5 * dy * nodes[None, :])[None, :, :]

    def _face_state(x_face):
        state = kovasznay_thermodynamic_state(
            torch.full_like(y_face, x_face), y_face, reynolds, wavenumber,
            sound_speed, density, gas_constant, dtype, device
        )
        wall = {
            "velocity": (state["u"], state["v"]),
            "temperature": state["p"] / (state["density"] * gas_constant),
            "state": {
                "rho": state["density"],
                "velocity_x": state["u"],
                "velocity_y": state["v"],
                "pressure": state["p"],
            },
        }
        return wall

    boundary = {"xmin": _face_state(xmin)}
    boundary["xmax"] = _face_state(xmax) if outlet == "prescribed" else None

    pressure_offset = density * gas_constant * sound_speed ** 2 / GAMMA

    def initial(x, y):
        state = kovasznay_thermodynamic_state(
            x, y, reynolds, wavenumber, sound_speed, density, gas_constant,
            dtype, device
        )
        if initial_mode == "uniform":
            zero = torch.zeros_like(state["u"])
            return (
                torch.full_like(state["u"], density),
                torch.ones_like(state["u"]),
                zero,
                torch.full_like(state["u"], pressure_offset),
            )
        return state["density"], state["u"], state["v"], state["p"]

    penalty_eff = penalty if penalty is not None else 2.0 * (degree + 1.0) ** 2
    # ``_rhs_ns`` indexes all four sides; the solver fills the missing ones in
    periodic_axes = (False, True)

    full_boundary = {key: boundary.get(key) for key in ("xmin", "xmax", "ymin", "ymax")}

    def _residual(coefficient):
        """Modal 2-norm of the DG steady residual, relative to the state."""
        rate = _rhs_ns(
            coefficient, basis, derivative, weights, signs, dx, dy, periodic_axes,
            full_boundary, mu, mu * CP / prandtl, penalty_eff, wall_gradient,
            wall_penalty, wall_adjoint=wall_adjoint,
        )
        return float(torch.sqrt((rate ** 2).sum() / (coefficient ** 2).sum()))

    total_chunks = max(1, int(chunks))
    span = final_time / total_chunks if chunk_time is None else float(chunk_time)
    history = []
    metadata = {}
    coeff = None
    x = y = None
    time = 0.0
    while time < final_time - 1e-15:
        step_time = min(span, final_time - time)
        x, y, coeff, metadata = solve_navier_stokes_dg_2d(
            initial,
            ncells=net,
            degree=degree,
            domain=box,
            final_time=step_time,
            cfl=cfl,
            cfl_viscous=cfl_viscous,
            mu=mu,
            prandtl=prandtl,
            periodic=(False, True),
            boundary=boundary,
            penalty=penalty,
            wall_penalty=wall_penalty,
            source=None,
            quadrature=order,
            wall_gradient=wall_gradient,
            wall_adjoint=wall_adjoint,
            device=device,
            dtype=dtype,
        )
        time += step_time
        entry = {"time": time, "steps": metadata.get("steps"),
                 "residual": _residual(coeff)}
        entry.update(velocity_l2_error(
            coeff, ncells=net, degree=degree, domain=box, reynolds=reynolds,
            wavenumber=wavenumber, quadrature=order,
            pressure_offset=pressure_offset,
        ))
        if monitor is not None:
            entry["monitor"] = monitor(time, coeff, metadata)
        history.append(entry)
        initial = modal_evaluator(coeff, net, degree, box, wavenumber, dtype)

    metadata = dict(metadata)
    metadata["history"] = history
    metadata["steady_residual"] = _residual(coeff)
    metadata["kovasznay"] = {
        "reynolds": reynolds,
        "wavenumber": wavenumber,
        "decay_rate": decay_rate(reynolds, wavenumber),
        "sound_speed": sound_speed,
        "mach": float(max(
            abs(1.0 + math.exp(decay_rate(reynolds, wavenumber) * xmin)),
            abs((decay_rate(reynolds, wavenumber) / wavenumber)
                * math.exp(decay_rate(reynolds, wavenumber) * xmin)),
        ) / sound_speed),
        "domain": tuple(box),
        "outlet": outlet,
        "pressure_offset": pressure_offset,
        "mu": mu,
        "final_time": final_time,
        "chunks": total_chunks,
        "initial_mode": initial_mode,
        "degree": degree,
        "ncells": net,
    }
    return x, y, coeff, metadata