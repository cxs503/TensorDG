"""High-order one-dimensional Euler DG with HLLC and local positivity fallback."""
from __future__ import annotations

import numpy as np
import torch

from .euler_1d import GAMMA, hllc_flux, physical_flux, primitive_to_conservative, sound_speed
from .limiters import apply_limiter


def _operators(degree, device, dtype):
    nodes_np, weights_np = np.polynomial.legendre.leggauss(max(degree + 2, 4))
    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
    values, derivatives = [], []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        derivative = np.polynomial.legendre.legder(polynomial)
        values.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, polynomial), device=device, dtype=dtype,
        ))
        deriv_values = (
            np.polynomial.legendre.legval(nodes_np, derivative)
            if len(derivative) else np.zeros_like(nodes_np)
        )
        derivatives.append(torch.as_tensor(deriv_values, device=device, dtype=dtype))
    phi = torch.stack(values, dim=1)
    dphi = torch.stack(derivatives, dim=1)
    phi_left = torch.tensor([(-1.0) ** k for k in range(degree + 1)], device=device, dtype=dtype)
    phi_right = torch.ones(degree + 1, device=device, dtype=dtype)
    return nodes, weights, phi, dphi, phi_left, phi_right


def _positivity_filter(coeff, check_points):
    """Revert only inadmissible cells to their positive cell-average state."""
    values = torch.einsum("cmv,qm->cqv", coeff, check_points)
    rho = values[..., 0]
    pressure = (GAMMA - 1.0) * (values[..., 2] - 0.5 * values[..., 1] ** 2 / rho.clamp_min(1e-12))
    valid = (rho > 0.0) & (pressure > 0.0) & torch.isfinite(values).all(dim=-1)
    valid_cell = valid.all(dim=1)
    average_only = torch.zeros_like(coeff)
    average_only[:, 0] = coeff[:, 0]
    return torch.where(valid_cell[:, None, None], coeff, average_only)


def _scaling_factors(coeff, check_points, eta=1e-10):
    """Zhang-Shu admissible scaling factors ``theta`` for one modal cell block.

    For every cell the state is written as ``u_theta = ubar + theta (u - ubar)``
    with ``theta in [0, 1]``.  Because the density is *linear* in the
    conservative variables and the pressure is *concave*, the point values of
    the scaled state satisfy

        rho(u_theta) = rho_bar + theta (rho_i - rho_bar)
        p(u_theta)   >= p_bar + theta (p_i - p_bar)          (concavity)

    so a per-point bound ``theta_i`` derived from the linear interpolation of
    the density and of the pressure guarantees admissibility of the whole cell
    at every check point (interior quadrature nodes *and* both traces), while
    keeping the cell average ``ubar`` unchanged.  Taking the minimum over the
    check points gives the largest admissible ``theta`` (Zhang & Shu, JCP 2010;
    Zhang, Xia & Shu, JCP 2012).  ``eta`` is the relative floor on the cell
    average (never a hard zero, so the bound stays finite).
    """
    values = torch.einsum("cmv,qm->cqv", coeff, check_points)
    average = coeff[:, 0]
    rho_avg = average[:, 0]
    pressure_avg = (GAMMA - 1.0) * (
        average[:, 2] - 0.5 * average[:, 1] ** 2 / rho_avg.clamp_min(1e-300)
    )
    # Non-admissible cell averages cannot be repaired by scaling: collapse.
    admissible_avg = (rho_avg > 0.0) & (pressure_avg > 0.0)
    rho = values[..., 0]
    pressure = (GAMMA - 1.0) * (
        values[..., 2] - 0.5 * values[..., 1] ** 2 / rho.clamp_min(1e-300)
    )
    finite = torch.isfinite(values).all(dim=-1) & torch.isfinite(pressure)
    rho_floor = (eta * rho_avg).clamp_min(0.0)
    pressure_floor = (eta * pressure_avg).clamp_min(0.0)

    def _bound(point_value, point_floor, cell_average):
        # theta such that average + theta (point - average) >= floor
        deficit = cell_average[:, None] - point_value
        safe = deficit.clamp_min(1e-300)
        bound = (cell_average[:, None] - point_floor[:, None]) / safe
        bound = torch.where(point_value >= point_floor[:, None], torch.ones_like(bound), bound)
        return torch.where(deficit <= 0.0, torch.ones_like(bound), bound)

    theta = torch.minimum(
        _bound(rho, rho_floor, rho_avg), _bound(pressure, pressure_floor, pressure_avg)
    )
    theta = torch.where(finite, theta, torch.zeros_like(theta))
    theta = theta.clamp(0.0, 1.0).min(dim=1).values
    return torch.where(admissible_avg, theta, torch.zeros_like(theta))


def _positivity_scale(coeff, check_points, eta=1e-10):
    """Zhang-Shu scaling limiter: admissible cell, maximum retained high order.

    Unlike :func:`_positivity_filter` (which throws the whole cell back to its
    first-order average), this only rescales the higher modes, so an admissible
    cell is left *bitwise* untouched and a nearly-admissible cell keeps the
    fraction of its high-order content that the positivity bound allows.
    """
    theta = _scaling_factors(coeff, check_points, eta=eta)
    out = coeff.clone()
    out[:, 1:] = coeff[:, 1:] * theta[:, None, None]
    return out


def _rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right, periodic=False):
    at_quadrature = torch.einsum("cmv,qm->cqv", coeff, phi)
    flux_quadrature = physical_flux(at_quadrature)
    volume = torch.einsum("cqv,q,qk->ckv", flux_quadrature, weights, dphi)
    right_trace = torch.einsum("cmv,m->cv", coeff, phi_right)
    left_trace = torch.einsum("cmv,m->cv", coeff, phi_left)
    flux_right = hllc_flux(right_trace, torch.roll(left_trace, shifts=-1, dims=0))
    flux_left = torch.roll(flux_right, shifts=1, dims=0)
    if not periodic:
        flux_right[-1] = physical_flux(right_trace[-1:])
        flux_left[0] = physical_flux(left_trace[:1])
    out = torch.empty_like(coeff)
    for k in range(coeff.shape[1]):
        out[:, k] = ((2 * k + 1) / dx) * (
            volume[:, k] - flux_right * phi_right[k] + flux_left * phi_left[k]
        )
    return out


def solve_euler_dg_1d(
    left_state,
    right_state,
    ncells=200,
    degree=2,
    final_time=0.13,
    diaphragm=0.5,
    domain=(0.0, 1.0),
    cfl=0.12,
    device="cpu",
    dtype=torch.float64,
    limiter=None,
    tvb_constant=1.0,
    limiter_stage="every",
    positivity="scale",
    limiter_kwargs=None,
):
    """Solve a Riemann problem with modal DG/HLLC/SSP-RK3.

    The local filter is positivity-based, not a TVD proof. It is intended for
    verification and moderate shock strengths, not as a general shock-capturing
    guarantee.

    Parameters
    ----------
    limiter : str or None
        Shock limiter to apply after each RK sub-step. Options:
        - None: no limiter (only positivity filter)
        - "tvb": TVB limiter (Cockburn & Shu, 1989)
        - "weno": WENO limiter (Qiu & Shu, 2005)
    tvb_constant : float
        TVB constant M; M=0 is strict TVD, larger M allows smooth extrema.
    limiter_stage : {"every", "final"}
        When the limiter is applied inside the SSP-RK3 sweep.  ``"every"``
        (default, historic behaviour) limits after every sub-step, which
        re-projects smooth cells that keep being flagged and therefore
        destroys high-order accuracy there (see
        ``docs/limiter_verification.md``).  ``"final"`` limits only the
        updated state at the end of the time step.
    limiter_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`tensordg.limiters.apply_limiter`,
        i.e. to the selected limiter.  Used for the sensor parameters of the
        ``"av"`` (modal artificial viscosity) kind, e.g.
        ``limiter_kwargs={"s0": -4.5, "kappa": 1.0, "alpha_max": 20.0}``.
    positivity : {"scale", "collapse", "none"}
        Admissibility treatment applied to every sub-step.  ``"scale"``
        (default) is the Zhang-Shu scaling limiter (:func:`_positivity_scale`):
        the higher modes of a cell are scaled by ``theta <= 1`` so that the
        density and the pressure stay positive at the quadrature nodes and at
        both traces, and the cell average is preserved exactly.  ``"collapse"``
        is the historic first-order fallback (:func:`_positivity_filter`) that
        resets a whole offending cell to its average.  ``"none"`` disables the
        treatment, which is only safe for well-resolved smooth data.
    """
    if ncells < 20 or degree not in (1, 2, 3) or final_time < 0.0 or not 0 < cfl <= 0.5:
        raise ValueError("invalid cells, degree, final_time, or CFL")
    if limiter_stage not in ("every", "final"):
        raise ValueError("limiter_stage must be 'every' or 'final'")
    if positivity not in ("scale", "collapse", "none"):
        raise ValueError("positivity must be 'scale', 'collapse' or 'none'")
    lower, upper = domain
    if not upper > lower or not lower < diaphragm < upper:
        raise ValueError("domain must be increasing and contain the diaphragm")
    dev = torch.device(device)
    dx = (upper - lower) / ncells
    x = lower + (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * dx
    mask = x < diaphragm
    rho = torch.where(mask, torch.full_like(x, left_state[0]), torch.full_like(x, right_state[0]))
    velocity = torch.where(mask, torch.full_like(x, left_state[1]), torch.full_like(x, right_state[1]))
    pressure = torch.where(mask, torch.full_like(x, left_state[2]), torch.full_like(x, right_state[2]))
    average = primitive_to_conservative(rho, velocity, pressure)
    coeff = torch.zeros((ncells, degree + 1, 3), device=dev, dtype=dtype)
    coeff[:, 0] = average
    nodes, weights, phi, dphi, phi_left, phi_right = _operators(degree, dev, dtype)
    check_points = torch.cat((phi, phi_left[None, :], phi_right[None, :]), dim=0)

    stats = {"limited_cells": 0, "cell_stages": 0, "min_theta": 1.0}

    def _admissible_fix(state):
        if positivity == "scale":
            theta = _scaling_factors(state, check_points)
            stats["limited_cells"] += int((theta < 1.0).sum().item())
            stats["cell_stages"] += int(theta.numel())
            stats["min_theta"] = min(stats["min_theta"], float(theta.min().item()))
            out = state.clone()
            out[:, 1:] = state[:, 1:] * theta[:, None, None]
            return out
        if positivity == "collapse":
            fixed = _positivity_filter(state, check_points)
            changed = (fixed != state).any(dim=-1).any(dim=-1)
            stats["limited_cells"] += int(changed.sum().item())
            stats["cell_stages"] += int(changed.numel())
            return fixed
        return state
    limiter_kwargs = dict(limiter_kwargs or {})
    limit_each_stage = limiter_stage == "every"
    time, steps, dt = 0.0, 0, 0.0
    while time < final_time:
        values = torch.einsum("cmv,qm->cqv", coeff, phi)
        speeds = torch.abs(values[..., 1] / values[..., 0].clamp_min(1e-12)) + sound_speed(values)
        max_speed = float(torch.max(speeds).item())
        dt = min(cfl * dx / ((2 * degree + 1) * max(max_speed, 1e-12)), final_time - time)
        k1 = _rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right)
        u1 = coeff + dt * k1
        if limit_each_stage:
            u1 = apply_limiter(u1, dx, kind=limiter, tvb_constant=tvb_constant, **limiter_kwargs)
        u1 = _admissible_fix(u1)
        k2 = _rhs(u1, dx, weights, phi, dphi, phi_left, phi_right)
        u2 = 0.75 * coeff + 0.25 * (u1 + dt * k2)
        if limit_each_stage:
            u2 = apply_limiter(u2, dx, kind=limiter, tvb_constant=tvb_constant, **limiter_kwargs)
        u2 = _admissible_fix(u2)
        k3 = _rhs(u2, dx, weights, phi, dphi, phi_left, phi_right)
        coeff = apply_limiter(
            (1.0 / 3.0) * coeff + (2.0 / 3.0) * (u2 + dt * k3),
            dx, kind=limiter, tvb_constant=tvb_constant, **limiter_kwargs,
        )
        coeff = _admissible_fix(coeff)
        average = coeff[:, 0]
        rho_now = average[:, 0]
        pressure_now = (GAMMA - 1.0) * (average[:, 2] - 0.5 * average[:, 1] ** 2 / rho_now.clamp_min(1e-12))
        if not torch.isfinite(coeff).all() or (rho_now <= 0).any() or (pressure_now <= 0).any():
            raise FloatingPointError("Nonphysical high-order Euler state; reduce CFL")
        time += dt
        steps += 1
    limiter_name = limiter if limiter else "local positivity fallback"
    metadata = {
        "ncells": ncells, "degree": degree, "final_time": final_time, "steps": steps,
        # ``dt`` is the *last* (time-truncated) step; it is not representative of
        # the run and must not be read as "the" step size.  ``dt_last`` repeats it
        # under an honest name and ``dt_mean`` is the actual mean step
        # (final_time / steps), which is the quantity a CFL study can trust.
        "dt": dt, "dt_last": dt,
        "dt_mean": (final_time / steps) if steps else 0.0,
        "diaphragm": diaphragm, "domain": (lower, upper), "cfl": cfl,
        "left_state": tuple(left_state), "right_state": tuple(right_state),
        "flux": "hllc", "limiter": limiter_name, "device": str(dev),
        "limiter_stage": limiter_stage,
        "positivity": positivity,
        "limiter_kwargs": dict(limiter_kwargs),
        "positivity_stats": dict(stats),
    }
    return x, coeff, metadata


def solve_periodic_euler_dg_1d(
    initial_state,
    ncells=64,
    degree=2,
    final_time=1.0,
    domain=(0.0, 1.0),
    cfl=0.12,
    device="cpu",
    dtype=torch.float64,
    limiter=None,
    tvb_constant=1.0,
    limiter_stage="every",
    boundary="periodic",
    limiter_kwargs=None,
):
    """Advance smooth Euler data with the same modal DG/HLLC kernels.

    Parameters
    ----------
    initial_state : callable
        ``initial_state(x)`` takes a ``[ncells, nquad]`` coordinate tensor and
        returns ``(rho, velocity, pressure)`` with the same shape.  The
        conservative state is projected onto **all** modal degrees of freedom
        with the Gauss--Legendre rule of :func:`_operators`, so smooth data
        start at the full design order (a cell-average-only projection would
        cap the scheme at first order).
    limiter, tvb_constant, limiter_stage
        As in :func:`solve_euler_dg_1d`.
    limiter_kwargs : dict, optional
        As in :func:`solve_euler_dg_1d`.
    boundary : {"periodic", "transmissive"}
        Face closure at the two ends of the domain.  ``"transmissive"``
        imposes the interior trace as the exterior state (exactly what
        :func:`solve_euler_dg_1d` does); it is required for open-domain
        benchmarks such as the Shu--Osher shock/entropy-wave interaction,
        where the free stream must leave the domain without reflection.

    Returns
    -------
    x : (ncells,) cell centres
    coeff : (ncells, degree + 1, 3) modal coefficients
    metadata : dict with step count, dt and relative conservation drift
    """
    if ncells < 8 or degree not in (1, 2, 3) or final_time < 0.0 or not 0 < cfl <= 0.5:
        raise ValueError("invalid cells, degree, final_time, or CFL")
    if limiter_stage not in ("every", "final"):
        raise ValueError("limiter_stage must be 'every' or 'final'")
    if boundary not in ("periodic", "transmissive"):
        raise ValueError("boundary must be 'periodic' or 'transmissive'")
    is_periodic = boundary == "periodic"
    lower, upper = domain
    if not upper > lower:
        raise ValueError("domain must be increasing")
    dev = torch.device(device)
    dx = (upper - lower) / ncells
    x = lower + (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * dx
    nodes, weights, phi, dphi, phi_left, phi_right = _operators(degree, dev, dtype)
    check_points = torch.cat((phi, phi_left[None, :], phi_right[None, :]), dim=0)

    node_x = x[:, None] + 0.5 * dx * nodes[None, :]
    rho, velocity, pressure = initial_state(node_x)
    conservative = primitive_to_conservative(rho, velocity, pressure)
    coeff = torch.zeros((ncells, degree + 1, 3), device=dev, dtype=dtype)
    for k in range(degree + 1):
        coeff[:, k] = ((2 * k + 1) / 2.0) * torch.einsum(
            "cqv,q,q->cv", conservative, weights, phi[:, k])

    def _totals(c):
        return (c[:, 0].sum(dim=0) * dx).clone()

    totals_initial = _totals(coeff)
    limiter_kwargs = dict(limiter_kwargs or {})
    limit_each_stage = limiter_stage == "every"
    time, steps, dt = 0.0, 0, 0.0
    while time < final_time:
        values = torch.einsum("cmv,qm->cqv", coeff, phi)
        speeds = torch.abs(values[..., 1] / values[..., 0].clamp_min(1e-12)) + sound_speed(values)
        max_speed = float(torch.max(speeds).item())
        dt = min(cfl * dx / ((2 * degree + 1) * max(max_speed, 1e-12)), final_time - time)
        k1 = _rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right, periodic=is_periodic)
        u1 = coeff + dt * k1
        if limit_each_stage:
            u1 = apply_limiter(u1, dx, kind=limiter, tvb_constant=tvb_constant, **limiter_kwargs)
        u1 = _positivity_filter(u1, check_points)
        k2 = _rhs(u1, dx, weights, phi, dphi, phi_left, phi_right, periodic=is_periodic)
        u2 = 0.75 * coeff + 0.25 * (u1 + dt * k2)
        if limit_each_stage:
            u2 = apply_limiter(u2, dx, kind=limiter, tvb_constant=tvb_constant, **limiter_kwargs)
        u2 = _positivity_filter(u2, check_points)
        k3 = _rhs(u2, dx, weights, phi, dphi, phi_left, phi_right, periodic=is_periodic)
        coeff = apply_limiter(
            (1.0 / 3.0) * coeff + (2.0 / 3.0) * (u2 + dt * k3),
            dx, kind=limiter, tvb_constant=tvb_constant, **limiter_kwargs,
        )
        coeff = _positivity_filter(coeff, check_points)
        if not torch.isfinite(coeff).all():
            raise FloatingPointError("Nonphysical periodic Euler state; reduce CFL")
        time += dt
        steps += 1

    totals_final = _totals(coeff)
    names = ("mass", "x_momentum", "energy")
    drift = {
        name: (
            float(abs(totals_final[i] - totals_initial[i]) / max(abs(totals_initial[i]), 1e-30))
            if is_periodic else float("nan")
        )
        for i, name in enumerate(names)
    }
    metadata = {
        "ncells": ncells, "degree": degree, "final_time": final_time, "steps": steps,
        "dt": dt, "dt_last": dt,
        "dt_mean": (final_time / steps) if steps else 0.0,
        "domain": (lower, upper), "cfl": cfl,
        "flux": "hllc", "limiter": limiter if limiter else "local positivity fallback",
        "limiter_stage": limiter_stage, "device": str(dev),
        "boundary": boundary, "initial_projection": "l2-all-modes",
        "conservation_drift": drift,
    }
    return x, coeff, metadata


def verify_shock_entropy(ncells: int = 200, degree: int = 1, limiter: str = "weno",
                         cfl: float = 0.4) -> dict:
    """Self-check of this solver on the Shu--Osher (1988) shock/entropy-wave case.

    The case data, the closed-form anchors and the acceptance report live in
    :mod:`tensordg.cases.shock_entropy`; this hook exists so that the Euler DG
    solver can validate itself without importing the benchmark scripts.  It is
    imported lazily to keep the module graph acyclic (the case module imports
    this solver at import time).

    Returns the acceptance report (density extrema, plateau/unshocked-sine/mass
    residuals, shock position vs the Rankine--Hugoniot value).  ``rho_positive``
    and ``mass_error_percent_of_initial`` are the two hard invariants: a negative
    density or a mass leak means the run is not admissible regardless of how
    small the profile error looks.
    """
    from .cases.shock_entropy import gate_report

    return gate_report(ncells, degree=degree, limiter=limiter, cfl=cfl)
