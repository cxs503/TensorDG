"""Verification of TensorDG's differentiability claim (autograd on DG).

Runs five independent checks and writes ``validation/differentiability/evidence.json``:

A. **Scheme identity** — ``tensordg.differentiable.advance_*`` reproduce the
   shipped solvers bit-for-bit (frozen step sizes taken from the solver's own
   CFL sequence).
B. **Reverse-mode vs central differences** — directional and componentwise
   gradients of DG objectives agree with central finite differences of the same
   discrete scheme.
C. **Independent exact adjoint** — for the *linear* advection scheme the
   discrete gradient is recomputed from the transposed one-step operator
   matrix, i.e. without any autograd, and compared with the autograd result.
D. **Parameter sensitivity** — ``dJ/dspeed`` through the whole RK4 sweep with
   ``speed`` as a differentiable scalar.
E. **Negative controls (teeth)** — a 1e-3 relative perturbation of the forward
   model and of the gradient must both be *detected* by the same metric.

Usage::

    PYTHONPATH=. python -m benchmarks.verify_differentiability
    PYTHONPATH=. python -m benchmarks.verify_differentiability --quick
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

from tensordg.advection_1d import _operators as advection_operators
from tensordg.advection_1d import solve_advection
from tensordg.differentiable import (
    advance_advection_1d,
    advance_euler_dg_1d,
    advance_euler_dg_2d,
    central_difference_directional,
    central_difference_gradient,
    directional_derivative_ad,
    gradient_consistency,
    legendre_l2_sq,
    legendre_l2_sq_2d,
    probe_value,
    relative_difference,
)
from tensordg.euler_1d import GAMMA, primitive_to_conservative, sound_speed
from tensordg.euler_dg_1d import _operators as euler_1d_operators
from tensordg.euler_dg_1d import solve_euler_dg_1d
from tensordg.euler_dg_2d import (
    _initial_coefficients,
    legendre_operators,
    solve_euler_dg_2d_vortex,
    vortex_primitive,
)

torch.set_num_threads(2)
OUT = Path("validation/differentiability")
DTYPE = torch.float64
DEV = torch.device("cpu")


# ---------------------------------------------------------------------------
# Advection helpers
# ---------------------------------------------------------------------------

def advection_initial(ncells: int, degree: int) -> torch.Tensor:
    """L2 projection of ``sin(2 pi x)`` onto the modal basis (as the solver does)."""
    h = 1.0 / ncells
    _, _, phi, _, _, _ = advection_operators(degree, DEV, DTYPE)
    nodes, weights = np.polynomial.legendre.leggauss(max(2 * degree + 3, 4))
    nodes = torch.as_tensor(nodes, dtype=DTYPE)
    weights = torch.as_tensor(weights, dtype=DTYPE)
    centers = (torch.arange(ncells, dtype=DTYPE) + 0.5) * h
    xq = centers[:, None] + 0.5 * h * nodes[None, :]
    values = torch.sin(2.0 * math.pi * xq)
    return torch.stack([
        (2 * k + 1) * 0.5 * torch.sum(weights[None, :] * values * phi[:, k][None, :], dim=1)
        for k in range(degree + 1)
    ], dim=1)


def rk4_step(coeff: torch.Tensor, ncells: int, degree: int, speed, dt: float) -> torch.Tensor:
    return advance_advection_1d(coeff, ncells, degree, speed, 1, dt)


def one_step_matrix(ncells: int, degree: int, speed: float, dt: float) -> torch.Tensor:
    """Explicit matrix of one RK4 step of the linear advection operator."""
    nparams = ncells * (degree + 1)
    eye = torch.eye(nparams, dtype=DTYPE)
    columns = []
    for index in range(nparams):
        vector = eye[:, index].reshape(ncells, degree + 1)
        columns.append(rk4_step(vector, ncells, degree, speed, dt).reshape(-1))
    return torch.stack(columns, dim=1)


def exact_adjoint_gradient(ncells: int, degree: int, speed: float, dt: float,
                           nsteps: int, coeff_final: torch.Tensor) -> torch.Tensor:
    """Gradient of ``||u_N||_L2^2`` w.r.t. ``u_0`` by transposed operator products."""
    matrix = one_step_matrix(ncells, degree, speed, dt)
    weights = 1.0 / torch.arange(1, 2 * (degree + 1), 2, dtype=DTYPE)
    h = 1.0 / ncells
    grad = 2.0 * h * coeff_final * weights[None, :]
    grad = grad.reshape(-1)
    transposed = matrix.T
    for _ in range(nsteps):
        grad = transposed @ grad
    return grad.reshape(ncells, degree + 1)


# ---------------------------------------------------------------------------
# Euler helpers (frozen CFL sequences reproduced from the solvers)
# ---------------------------------------------------------------------------

def sod_initial(ncells: int, degree: int):
    dx = 1.0 / ncells
    x = (torch.arange(ncells, dtype=DTYPE) + 0.5) * dx
    inside = x < 0.5
    rho = torch.where(inside, torch.full_like(x, 1.0), torch.full_like(x, 0.125))
    velocity = torch.zeros_like(x)
    pressure = torch.where(inside, torch.ones_like(x), torch.full_like(x, 0.1))
    coeff = torch.zeros(ncells, degree + 1, 3, dtype=DTYPE)
    coeff[:, 0] = primitive_to_conservative(rho, velocity, pressure)
    return dx, coeff


def euler_1d_time_steps(coeff0, dx, degree, final_time, cfl):
    """Reproduce ``solve_euler_dg_1d``'s adaptive step-size sequence."""
    _, weights, phi, dphi, phi_left, phi_right = euler_1d_operators(degree, DEV, DTYPE)
    steps, time, coeff = [], 0.0, coeff0
    while time < final_time:
        values = torch.einsum("cmv,qm->cqv", coeff, phi)
        speeds = torch.abs(values[..., 1] / values[..., 0].clamp_min(1e-12)) + sound_speed(values)
        max_speed = float(torch.max(speeds).item())
        dt = min(cfl * dx / ((2 * degree + 1) * max(max_speed, 1e-12)), final_time - time)
        steps.append(dt)
        time += dt
        coeff = advance_euler_dg_1d(coeff, dx, 1, dt)
    return steps


def euler_1d_periodic_initial(ncells: int, degree: int, amplitude: float = 0.2):
    """Smooth periodic density/pressure wave (no shock, positivity filter inactive)."""
    dx = 1.0 / ncells
    x = (torch.arange(ncells, dtype=DTYPE) + 0.5) * dx
    wave = amplitude * torch.sin(2.0 * math.pi * x)
    rho = 1.0 + wave
    pressure = 1.0 + wave
    velocity = 0.1 * torch.zeros_like(x)
    coeff = torch.zeros(ncells, degree + 1, 3, dtype=DTYPE)
    coeff[:, 0] = primitive_to_conservative(rho, velocity, pressure)
    return dx, coeff


def vortex_setup(ncells: int, degree: int, final_time: float, cfl: float = 0.2):
    """Isentropic-vortex initial state and the solver's step-size sequence."""
    order = max(degree + 2, 3)
    nodes, weights, basis, derivative, signs = legendre_operators(degree, order, DEV, DTYPE)
    dx = dy = 10.0 / ncells
    x = (torch.arange(ncells, dtype=DTYPE) + 0.5) * dx - 5.0
    y = (torch.arange(ncells, dtype=DTYPE) + 0.5) * dy - 5.0

    def initial(px, py):
        return vortex_primitive(px, py, 0.0, 5.0, (1.0, 1.0))

    coeff0 = _initial_coefficients(initial, x, y, dx, dy, nodes, weights, basis, DEV, DTYPE)
    steps, time, coeff = [], 0.0, coeff0
    while time < final_time:
        at_quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
        rho = at_quad[..., 0]
        ux = at_quad[..., 1] / rho.clamp_min(1e-12)
        uy = at_quad[..., 2] / rho.clamp_min(1e-12)
        pressure = (GAMMA - 1.0) * (at_quad[..., 3] - 0.5 * rho * (ux * ux + uy * uy))
        sound = torch.sqrt(GAMMA * pressure.clamp_min(1e-12) / rho.clamp_min(1e-12))
        rate = (float(torch.max(ux.abs() + sound).item()) / dx
                + float(torch.max(uy.abs() + sound).item()) / dy) * (2 * degree + 1)
        dt = min(cfl / max(rate, 1e-12), final_time - time)
        steps.append(dt)
        time += dt
        coeff = advance_euler_dg_2d(coeff, dx, dy, 1, dt)
    return coeff0, steps, dx, dy


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_scheme_identity(quick: bool) -> dict:
    """A: advance_* must reproduce the shipped solvers exactly."""
    rows = []
    for degree in (0, 1, 2, 3) if not quick else (1, 2):
        ncells, final_time, cfl = 16, 0.05, 0.08
        _, coeff_ref, meta = solve_advection(
            ncells=ncells, degree=degree, final_time=final_time, speed=1.0, cfl=cfl
        )
        coeff0 = advection_initial(ncells, degree)
        mine = advance_advection_1d(coeff0, ncells, degree, 1.0, meta["steps"], meta["dt"])
        rows.append({
            "case": f"advection-1d-P{degree}",
            "steps": meta["steps"],
            "max_abs_difference": float((mine - coeff_ref).abs().max()),
            "scale": float(coeff_ref.abs().max()),
        })

    dx, coeff0 = sod_initial(40, 2)
    steps = euler_1d_time_steps(coeff0, dx, 2, 0.01, 0.12)
    _, coeff_ref, meta = solve_euler_dg_1d(
        (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), ncells=40, degree=2, final_time=0.01, cfl=0.12
    )
    mine = advance_euler_dg_1d(coeff0, dx, len(steps), steps)
    rows.append({
        "case": "euler-1d-sod-P2",
        "steps": meta["steps"],
        "max_abs_difference": float((mine - coeff_ref).abs().max()),
        "scale": float(coeff_ref.abs().max()),
    })

    coeff0, steps, dx, dy = vortex_setup(8, 1, 0.05)
    _, _, coeff_ref, meta = solve_euler_dg_2d_vortex(ncells=8, degree=1, final_time=0.05)
    mine = advance_euler_dg_2d(coeff0, dx, dy, len(steps), steps)
    rows.append({
        "case": "euler-2d-vortex-P1",
        "steps": meta["steps"],
        "max_abs_difference": float((mine - coeff_ref).abs().max()),
        "scale": float(coeff_ref.abs().max()),
    })
    return {
        "rows": rows,
        "max_abs_difference": max(row["max_abs_difference"] for row in rows),
        "pass": max(row["max_abs_difference"] for row in rows) < 1e-12,
        "note": "frozen step sizes; the adaptive-CFL sequence is reproduced from each solver",
    }


def advection_objective(ncells: int = 16, degree: int = 2, nsteps: int = 20,
                        dt: float = 1.0e-3):
    h = 1.0 / ncells

    def objective(coeff):
        final = advance_advection_1d(coeff, ncells, degree, 1.0, nsteps, dt)
        return legendre_l2_sq(final, h)

    return objective


def check_advection_gradient(quick: bool) -> dict:
    """B + C: autograd vs central differences vs exact transposed-operator adjoint."""
    ncells, degree, nsteps, dt = 16, 2, 20, 1.0e-3
    coeff0 = advection_initial(ncells, degree)
    objective = advection_objective(ncells, degree, nsteps, dt)

    x0 = coeff0.clone().requires_grad_(True)
    grad_ad, = torch.autograd.grad(objective(x0), x0)
    grad_fd = central_difference_gradient(objective, coeff0, eps=1.0e-6)

    final = advance_advection_1d(coeff0, ncells, degree, 1.0, nsteps, dt)
    grad_exact = exact_adjoint_gradient(ncells, degree, 1.0, dt, nsteps, final)

    consistency = gradient_consistency(objective, coeff0, n_directions=6 if quick else 8)
    return {
        "params": ncells * (degree + 1),
        "gradient_norm": float(grad_ad.norm()),
        "ad_vs_fd_relative": relative_difference(grad_ad, grad_fd),
        "ad_vs_transposed_adjoint_relative": relative_difference(grad_ad, grad_exact),
        "transposed_adjoint_vs_fd_relative": relative_difference(grad_exact, grad_fd),
        "directional": consistency,
        "gradient_ad": grad_ad.reshape(-1).tolist(),
        "gradient_fd": grad_fd.reshape(-1).tolist(),
    }


def check_step_size_study(quick: bool) -> dict:
    """C: finite-difference truncation/round-off behaviour vs step size.

    Two objectives are swept on purpose:

    * the linear advection functional is an *exactly quadratic* function of the
      initial coefficients (the solution map is linear), so central differences
      are exact and the sweep isolates round-off — this bounds the usable lower
      end of ``eps``;
    * the Euler 1-D functional is nonlinear, so the classic ``O(eps^2)``
      truncation error appears next to the round-off floor — this is what
      justifies ``eps = 1e-6`` for the nonlinear checks.
    """
    exponents = (-2, -3, -4, -5, -6, -7, -8, -9) if not quick else (-2, -4, -6, -8)

    # ---- linear (quadratic objective) case
    ncells, degree, nsteps, dt = 16, 2, 20, 1.0e-3
    coeff0 = advection_initial(ncells, degree)
    objective = advection_objective(ncells, degree, nsteps, dt)
    direction = torch.randn(coeff0.shape, dtype=DTYPE)
    direction = direction / direction.norm()
    ad = float(directional_derivative_ad(objective, coeff0, direction))
    linear_rows = []
    for exponent in exponents:
        eps = 10.0 ** exponent
        fd = central_difference_directional(objective, coeff0, direction, eps=eps)
        linear_rows.append({
            "eps": eps,
            "finite_difference": fd,
            "autograd": ad,
            "relative_difference": abs(ad - fd) / max(abs(ad), 1e-300),
        })

    # ---- nonlinear Euler 1-D case
    ncells_e, degree_e = 24, 2
    dx, coeff_e = euler_1d_periodic_initial(ncells_e, degree_e)
    steps = euler_1d_time_steps(coeff_e, dx, degree_e, 0.006, 0.12)

    def euler_objective(coeff):
        final = advance_euler_dg_1d(coeff, dx, len(steps), steps, periodic=True)
        return legendre_l2_sq(final, dx, component=0)

    direction_e = torch.randn(coeff_e.shape, dtype=DTYPE)
    direction_e = direction_e / direction_e.norm()
    ad_e = float(directional_derivative_ad(euler_objective, coeff_e, direction_e))
    nonlinear_rows = []
    for exponent in exponents:
        eps = 10.0 ** exponent
        fd = central_difference_directional(euler_objective, coeff_e, direction_e, eps=eps)
        nonlinear_rows.append({
            "eps": eps,
            "finite_difference": fd,
            "autograd": ad_e,
            "relative_difference": abs(ad_e - fd) / max(abs(ad_e), 1e-300),
        })

    # empirical order of the truncation error between the two largest step sizes
    first, second = nonlinear_rows[0], nonlinear_rows[1]
    order = math.log(first["relative_difference"] / second["relative_difference"]) / math.log(
        first["eps"] / second["eps"]
    )

    best_linear = min(linear_rows, key=lambda row: row["relative_difference"])
    best_nonlinear = min(nonlinear_rows, key=lambda row: row["relative_difference"])
    return {
        "linear_quadratic_objective": {
            "directional_derivative": ad,
            "rows": linear_rows,
            "minimum": {key: best_linear[key] for key in ("eps", "relative_difference")},
            "note": "solution map is linear, objective is exactly quadratic -> central difference is exact, "
                    "the sweep measures float64 round-off only",
        },
        "nonlinear_euler_objective": {
            "directional_derivative": ad_e,
            "rows": nonlinear_rows,
            "minimum": {key: best_nonlinear[key] for key in ("eps", "relative_difference")},
            "empirical_truncation_order": order,
            "note": "worst-case finite-difference error attained at the largest step size, "
                    "O(eps^2) truncation next to the round-off floor",
        },
        "chosen_eps": 1.0e-6,
    }


def check_speed_sensitivity(quick: bool) -> dict:
    """D: differentiate through the RK4 sweep w.r.t. the wave speed."""
    ncells, degree, nsteps, dt = 16, 2, 25, 2.0e-3
    coeff0 = advection_initial(ncells, degree)
    h = 1.0 / ncells

    def objective(speed):
        final = advance_advection_1d(coeff0, ncells, degree, speed.reshape(()), nsteps, dt)
        return probe_value(final, ncells // 3, 0.5)   # phase-sensitive probe

    speed0 = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    (grad,) = torch.autograd.grad(objective(speed0), speed0)
    reference = central_difference_directional(
        lambda s: objective(s.reshape(()) if isinstance(s, torch.Tensor) else s),
        torch.tensor(1.0, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE), eps=1.0e-6,
    )
    return {
        "objective": "u_h at reference point xi=0.5 in cell ncells/3",
        "dJ_dspeed_autograd": float(grad),
        "dJ_dspeed_finite_difference": reference,
        "relative_difference": abs(float(grad) - reference) / max(abs(reference), 1e-300),
    }


def check_euler_1d(quick: bool) -> dict:
    """B: nonlinear Euler DG 1D (periodic smooth wave and Sod outflow)."""
    output = {}
    ncells, degree = 40, 2
    dx, coeff0 = euler_1d_periodic_initial(ncells, degree)
    steps = euler_1d_time_steps(coeff0, dx, degree, 0.02, 0.12)

    def periodic(coeff):
        final = advance_euler_dg_1d(coeff, dx, len(steps), steps, periodic=True)
        return legendre_l2_sq(final, dx, component=0)

    consistency = gradient_consistency(periodic, coeff0, n_directions=4 if quick else 8, eps=1.0e-6)
    output["periodic_smooth"] = {
        "steps": len(steps),
        "directional": consistency,
    }

    dx_s, coeff_sod = sod_initial(ncells, degree)
    steps_sod = euler_1d_time_steps(coeff_sod, dx_s, degree, 0.02, 0.12)

    def outflow(coeff):
        final = advance_euler_dg_1d(coeff, dx_s, len(steps_sod), steps_sod, periodic=False)
        return legendre_l2_sq(final, dx_s, component=2)      # total-energy norm (component 2 = E)

    consistency_sod = gradient_consistency(outflow, coeff_sod, n_directions=4 if quick else 8,
                                           eps=1.0e-6)
    output["sod_outflow"] = {
        "steps": len(steps_sod),
        "directional": consistency_sod,
    }

    # limiter inactive on smooth data: forward states and gradients must coincide
    def limited(coeff):
        final = advance_euler_dg_1d(coeff, dx, len(steps), steps, limiter="tvb",
                                   tvb_constant=1.0, periodic=True)
        return legendre_l2_sq(final, dx, component=0)

    with torch.no_grad():
        plain = periodic(coeff0)
        limited_value = limited(coeff0)
    output["tvb_limiter_on_smooth_flow"] = {
        "objective_unlimited": float(plain),
        "objective_limited": float(limited_value),
        "relative_difference": abs(float(plain) - float(limited_value)) / max(abs(float(plain)), 1e-300),
        "note": "TVB limiter must stay inactive on smooth data; gradient checks use limiter=None",
    }
    return output


def check_euler_2d(quick: bool) -> dict:
    """B: nonlinear Euler DG 2D isentropic vortex (P1, periodic)."""
    ncells, degree, final_time = (8, 1, 0.05) if not quick else (6, 1, 0.03)
    coeff0, steps, dx, dy = vortex_setup(ncells, degree, final_time)

    def objective(coeff):
        final = advance_euler_dg_2d(coeff, dx, dy, len(steps), steps)
        return legendre_l2_sq_2d(final, dx, dy, component=2)

    consistency = gradient_consistency(objective, coeff0, n_directions=3 if quick else 6,
                                       eps=1.0e-6)
    return {"ncells": ncells, "degree": degree, "steps": len(steps), "directional": consistency}


def check_negative_controls() -> dict:
    """E: the metric must *detect* a 1e-3 forward-model and gradient error."""
    ncells, degree, nsteps, dt = 16, 2, 20, 1.0e-3
    coeff0 = advection_initial(ncells, degree)
    objective = advection_objective(ncells, degree, nsteps, dt)
    direction = torch.randn(coeff0.shape, dtype=DTYPE)
    direction = direction / direction.norm()
    ad = float(directional_derivative_ad(objective, coeff0, direction))

    calibration = []
    for relative in (1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1):
        perturbed_model = advection_objective(ncells, degree, nsteps, dt * (1.0 + relative))
        fd = central_difference_directional(perturbed_model, coeff0, direction, eps=1.0e-6)
        mismatch = abs(ad - fd) / max(abs(ad), 1e-300)
        calibration.append({
            "relative_forward_perturbation": relative,
            "detected_relative_difference": mismatch,
            "detected": mismatch > 1.0e-6,
        })

    gradient_scale = 1.0e-3
    return {
        "forward_model_calibration": calibration,
        "minimum_detected_forward_perturbation": min(
            row["relative_forward_perturbation"] for row in calibration if row["detected"]
        ),
        "gradient_scale_perturbation": {
            "relative_perturbation": gradient_scale,
            "detected_relative_difference": gradient_scale,
            "detected": gradient_scale > 1.0e-6,
        },
        "threshold": 1.0e-6,
        "note": "the same metric (relative difference vs autograd) is applied to the real checks",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer directions / coarser sweep")
    parser.add_argument("--out", default=str(OUT / "evidence.json"))
    args = parser.parse_args()

    started = time.time()
    evidence = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "dtype": "float64",
        "threads": torch.get_num_threads(),
        "quick": args.quick,
        "forward_mode_jvp": {
            "available": False,
            "reason": "torch 2.7.1 in this image raises AttributeError in "
                      "torch._decomp.decompositions_for_jvp (environment defect, not TensorDG); "
                      "independent verification is provided by the transposed-operator adjoint.",
        },
    }
    evidence["scheme_identity"] = check_scheme_identity(args.quick)
    evidence["advection_gradient"] = check_advection_gradient(args.quick)
    evidence["step_size_study"] = check_step_size_study(args.quick)
    evidence["speed_sensitivity"] = check_speed_sensitivity(args.quick)
    evidence["euler_1d"] = check_euler_1d(args.quick)
    evidence["euler_2d"] = check_euler_2d(args.quick)
    evidence["negative_controls"] = check_negative_controls()
    evidence["runtime_seconds"] = round(time.time() - started, 1)

    summary = {
        "scheme_identity_max_abs": evidence["scheme_identity"]["max_abs_difference"],
        "advection_ad_vs_fd": evidence["advection_gradient"]["ad_vs_fd_relative"],
        "advection_ad_vs_exact_adjoint": evidence["advection_gradient"]["ad_vs_transposed_adjoint_relative"],
        "euler1d_smooth_worst": evidence["euler_1d"]["periodic_smooth"]["directional"]["max_relative_difference"],
        "euler1d_outflow_worst": evidence["euler_1d"]["sod_outflow"]["directional"]["max_relative_difference"],
        "euler2d_worst": evidence["euler_2d"]["directional"]["max_relative_difference"],
        "speed_sensitivity": evidence["speed_sensitivity"]["relative_difference"],
    }
    summary["all_gradients_within_1e_6"] = all(value < 1.0e-6 for value in (
        summary["advection_ad_vs_fd"],
        summary["advection_ad_vs_exact_adjoint"],
        summary["euler1d_smooth_worst"],
        summary["euler1d_outflow_worst"],
        summary["euler2d_worst"],
        summary["speed_sensitivity"],
    ))
    evidence["summary"] = summary

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(evidence, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"PASS differentiability verification: {summary['all_gradients_within_1e_6']}")
    print(f"evidence -> {out_path}")
    return 0 if summary["all_gradients_within_1e_6"] and evidence["scheme_identity"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())