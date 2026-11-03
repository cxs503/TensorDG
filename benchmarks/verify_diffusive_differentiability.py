"""Verify autograd sensitivities of the *diffusive* (SIPG + Crank--Nicolson) DG path.

Why this benchmark exists
-------------------------
``docs/differentiability_verification.md`` (m2552) turned TensorDG's
"differentiable DG" claim into reproducible evidence, but every case there is
hyperbolic: ``advance_advection_1d`` / ``advance_euler_dg_1d`` /
``advance_euler_dg_2d``.  The diffusive half of the operator zoo -- the SIPG
heat/elliptic path, i.e. the entry point for thermal and viscous inverse
problems -- had no differentiable entry point and no gradient evidence at all.
Central differences alone cannot close that gap either: they only prove that
autograd and finite differences differentiate *the same* discrete scheme, so a
wrong ``alpha`` factor or a mis-signed stiffness block would pass unnoticed.

This benchmark therefore compares three mutually independent sensitivity
sources against one another, and against the continuum physics:

1. autograd through :func:`tensordg.differentiable.advance_heat_1d`
   (frozen ``dt``, ``alpha`` a 0-dim tensor);
2. an *exact discrete adjoint* with no autograd and no finite differences:
   the Crank--Nicolson map is assembled explicitly as ``G = A^{-1} B`` with
   ``A = M + dt alpha K / 2``, ``B = M - dt alpha K / 2``, the adjoint state is
   propagated with ``G^T``, and ``dG/dalpha = A^{-1}(dB/dalpha - dA/dalpha G)``
   is evaluated analytically;
3. the analytical mode-decay sensitivity of ``u_t = alpha u_xx`` with
   ``u0 = sin(pi x)``: ``J(alpha) = |u(T)|^2 = exp(-2 pi^2 alpha T)/2`` and
   ``dJ/dalpha = -2 pi^2 T J``.

(1) vs (2) must agree to machine precision -- that is the "the adjoint is
implemented correctly" statement.  (1) vs (3) is the "the tangent points at the
physics, not just at the code" statement, and it is the only place where the
discretisation error enters.

Run with::

    PYTHONPATH=. python -m benchmarks.verify_diffusive_differentiability
    PYTHONPATH=. python -m benchmarks.verify_diffusive_differentiability --quick
"""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import torch

from tensordg.differentiable import (
    advance_heat_1d,
    heat_1d_operators,
    legendre_l2_sq,
    relative_difference,
)
from tensordg.heat_1d import heat_error, solve_heat_1d


def _exact_sensitivity(final_time: float, alpha: float) -> tuple[float, float]:
    """Continuum reference ``J`` and ``dJ/dalpha`` for ``u0 = sin(pi x)``."""
    decay = math.exp(-2.0 * math.pi ** 2 * alpha * final_time)
    return 0.5 * decay, -2.0 * math.pi ** 2 * final_time * 0.5 * decay


def _discrete_adjoint(coeff0: torch.Tensor, ncells: int, degree: int,
                      nsteps: int, dt: float, alpha: float,
                      penalty: float) -> dict:
    """Exact adjoint of the Crank--Nicolson map, assembled by hand.

    Returns ``dJ_dalpha`` and the full gradient ``dJ/dcoeff0`` together with the
    forward trajectory, so the caller can compare against autograd.
    """
    mass, diffusion, _centers = heat_1d_operators(
        ncells, degree, penalty, coeff0.device, coeff0.dtype
    )
    ndofs = mass.shape[0]
    identity = torch.eye(ndofs, dtype=mass.dtype, device=mass.device)
    system = mass + 0.5 * dt * alpha * diffusion
    explicit = mass - 0.5 * dt * alpha * diffusion
    step_map = torch.linalg.solve(system, explicit @ identity)
    d_system = 0.5 * dt * diffusion
    d_explicit = -0.5 * dt * diffusion
    d_step_map = torch.linalg.solve(system, d_explicit - d_system @ step_map)

    states = [coeff0.reshape(-1)]
    for _ in range(nsteps):
        states.append(step_map @ states[-1])
    final = states[-1]

    def objective(state):
        return state @ (mass @ state)

    value = objective(final)
    adjoint = 2.0 * (mass @ final)
    d_alpha = 0.0
    for index in range(nsteps, 0, -1):
        d_alpha += float(adjoint @ (d_step_map @ states[index - 1]))
        adjoint = step_map.T @ adjoint
    return {
        "value": float(value),
        "dJ_dalpha": d_alpha,
        "dJ_dcoeff0": adjoint.reshape(coeff0.shape),
        "step_map_spectral_radius": float(
            torch.linalg.eigvals(step_map).abs().max()
        ),
    }


def _autograd_sensitivities(coeff0: torch.Tensor, ncells: int, degree: int,
                            nsteps: int, dt: float, alpha: float,
                            penalty: float) -> dict:
    """``dJ/dalpha`` and ``dJ/dcoeff0`` from autograd through the CN loop."""
    alpha_t = torch.tensor(alpha, dtype=coeff0.dtype, requires_grad=True)
    state0 = coeff0.detach().clone().requires_grad_(True)
    final = advance_heat_1d(
        state0, ncells, degree, nsteps=nsteps, dt=dt, alpha=alpha_t, penalty=penalty
    )
    value = legendre_l2_sq(final, 1.0 / ncells)
    grad_alpha, grad_state = torch.autograd.grad(value, (alpha_t, state0))
    return {
        "value": float(value),
        "dJ_dalpha": float(grad_alpha),
        "dJ_dcoeff0": grad_state.detach(),
    }


def _central_difference_alpha(coeff0: torch.Tensor, ncells: int, degree: int,
                              nsteps: int, dt: float, alpha: float,
                              penalty: float, eps: float) -> float:
    """Central difference of the *same* discrete scheme in ``alpha``."""
    values = []
    for sign in (1.0, -1.0):
        with torch.no_grad():
            final = advance_heat_1d(
                coeff0, ncells, degree, nsteps=nsteps, dt=dt,
                alpha=alpha + sign * eps, penalty=penalty,
            )
            values.append(float(legendre_l2_sq(final, 1.0 / ncells)))
    return (values[0] - values[1]) / (2.0 * eps)


def _case(ncells: int, degree: int, alpha: float, final_time: float,
          penalty: float, eps_list: list[float]) -> dict:
    centers, coeff0, metadata = solve_heat_1d(
        ncells=ncells, degree=degree, final_time=0.0, penalty=penalty
    )
    steps = max(1, math.ceil(final_time / (0.02 * (1.0 / ncells))))
    dt = final_time / steps

    # (0) scheme identity: advance_heat_1d vs the shipped solver at alpha = 1
    _c, reference, ref_meta = solve_heat_1d(
        ncells=ncells, degree=degree, final_time=final_time,
        time_steps=steps, penalty=penalty,
    )
    with torch.no_grad():
        mirrored = advance_heat_1d(
            coeff0, ncells, degree, nsteps=ref_meta["steps"], dt=ref_meta["dt"],
            alpha=1.0, penalty=penalty,
        )
    identity = float((mirrored - reference).abs().max())

    adjoint = _discrete_adjoint(coeff0, ncells, degree, steps, dt, alpha, penalty)
    autograd = _autograd_sensitivities(coeff0, ncells, degree, steps, dt, alpha, penalty)

    fd_rows = []
    for eps in eps_list:
        fd = _central_difference_alpha(
            coeff0, ncells, degree, steps, dt, alpha, penalty, eps
        )
        fd_rows.append({
            "eps": eps,
            "finite_difference": fd,
            "relative_difference_vs_autograd":
                abs(fd - autograd["dJ_dalpha"]) / max(abs(fd), 1e-300),
        })

    analytic_value, analytic_sensitivity = _exact_sensitivity(final_time, alpha)
    h = 1.0 / ncells
    return {
        "ncells": ncells,
        "degree": degree,
        "alpha": alpha,
        "final_time": final_time,
        "steps": steps,
        "dt": dt,
        "penalty": penalty,
        "scheme_identity": {
            "max_abs_difference": identity,
            "pass": identity < 1e-12,
        },
        "objective": {
            "autograd": autograd["value"],
            "exact_discrete_adjoint": adjoint["value"],
            "analytic": analytic_value,
            "relative_error_vs_analytic":
                abs(autograd["value"] - analytic_value) / analytic_value,
            "relative_l2_of_solution": heat_error(centers, mirrored, final_time)[
                "relative_l2"
            ],
        },
        "sensitivity_alpha": {
            "autograd": autograd["dJ_dalpha"],
            "exact_discrete_adjoint": adjoint["dJ_dalpha"],
            "autograd_vs_adjoint_relative": abs(
                autograd["dJ_dalpha"] - adjoint["dJ_dalpha"]
            ) / max(abs(adjoint["dJ_dalpha"]), 1e-300),
            "analytic": analytic_sensitivity,
            "autograd_vs_analytic_relative": abs(
                autograd["dJ_dalpha"] - analytic_sensitivity
            ) / abs(analytic_sensitivity),
        },
        "gradient_initial_state": {
            "dofs": int(autograd["dJ_dcoeff0"].numel()),
            "autograd_vs_adjoint_relative": relative_difference(
                autograd["dJ_dcoeff0"], adjoint["dJ_dcoeff0"]
            ),
        },
        "step_map_spectral_radius": adjoint["step_map_spectral_radius"],
        "finite_difference_alpha": fd_rows,
        "h": h,
    }


def _alpha_sweep(ncells: int, degree: int, alphas: list[float],
                 final_time: float, penalty: float) -> list[dict]:
    centers, coeff0, _metadata = solve_heat_1d(
        ncells=ncells, degree=degree, final_time=0.0, penalty=penalty
    )
    steps = max(1, math.ceil(final_time / (0.02 * (1.0 / ncells))))
    dt = final_time / steps
    rows = []
    for alpha in alphas:
        adjoint = _discrete_adjoint(coeff0, ncells, degree, steps, dt, alpha, penalty)
        autograd = _autograd_sensitivities(
            coeff0, ncells, degree, steps, dt, alpha, penalty
        )
        analytic_value, analytic_sensitivity = _exact_sensitivity(final_time, alpha)
        rows.append({
            "alpha": alpha,
            "value_autograd": autograd["value"],
            "value_analytic": analytic_value,
            "value_relative_error":
                abs(autograd["value"] - analytic_value) / analytic_value,
            "sensitivity_autograd": autograd["dJ_dalpha"],
            "sensitivity_adjoint": adjoint["dJ_dalpha"],
            "sensitivity_analytic": analytic_sensitivity,
            "sensitivity_autograd_vs_adjoint_relative": abs(
                autograd["dJ_dalpha"] - adjoint["dJ_dalpha"]
            ) / max(abs(adjoint["dJ_dalpha"]), 1e-300),
            "sensitivity_vs_analytic_relative": abs(
                autograd["dJ_dalpha"] - analytic_sensitivity
            ) / abs(analytic_sensitivity),
            "step_map_spectral_radius": adjoint["step_map_spectral_radius"],
        })
    return rows


def build_evidence(quick: bool = False) -> dict:
    final_time = 0.1
    penalty = 20.0
    eps_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6] if not quick else [1e-4]
    grid = [(8, 1), (8, 2), (16, 2), (16, 3)] if not quick else [(8, 2)]
    cases = [
        _case(ncells, degree, 0.05, final_time, penalty, eps_list)
        for ncells, degree in grid
    ]
    sweep_grid = [0.01, 0.02, 0.05, 0.1, 0.2] if not quick else [0.05, 0.1]
    sweeps = [
        {
            "ncells": ncells,
            "degree": degree,
            "final_time": final_time,
            "rows": _alpha_sweep(ncells, degree, sweep_grid, final_time, penalty),
        }
        for ncells, degree in ([(16, 2), (16, 3)] if not quick else [(16, 2)])
    ]
    worst_identity = max(c["scheme_identity"]["max_abs_difference"] for c in cases)
    worst_adjoint_alpha = max(
        c["sensitivity_alpha"]["autograd_vs_adjoint_relative"] for c in cases
    )
    worst_adjoint_state = max(
        c["gradient_initial_state"]["autograd_vs_adjoint_relative"] for c in cases
    )
    worst_fd = max(
        min(row["relative_difference_vs_autograd"] for row in c["finite_difference_alpha"])
        for c in cases
    )
    worst_analytic = max(
        c["sensitivity_alpha"]["autograd_vs_analytic_relative"] for c in cases
    )
    worst_objective = max(c["objective"]["relative_error_vs_analytic"] for c in cases)
    return {
        "generated_utc": __import__("datetime").datetime.utcnow().strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "dtype": "float64",
        "threads": torch.get_num_threads(),
        "quick": quick,
        "cases": cases,
        "alpha_sweeps": sweeps,
        "summary": {
            "worst_scheme_identity_max_abs": worst_identity,
            "worst_autograd_vs_exact_adjoint_alpha": worst_adjoint_alpha,
            "worst_autograd_vs_exact_adjoint_initial_state": worst_adjoint_state,
            "worst_autograd_vs_central_difference_alpha": worst_fd,
            "worst_autograd_vs_analytic_sensitivity": worst_analytic,
            "worst_objective_relative_error_vs_analytic": worst_objective,
            "pass_adjoint_machine_precision": worst_adjoint_alpha < 1e-12
            and worst_adjoint_state < 1e-12,
            "pass_finite_difference": worst_fd < 1e-6,
            "pass_analytic_three_percent": worst_analytic < 0.03,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="single reduced case")
    parser.add_argument(
        "--out", default="validation/diffusive_differentiability/evidence.json"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    evidence = build_evidence(quick=args.quick)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=1))
    summary = evidence["summary"]
    for key, value in summary.items():
        print(f"{key:48s} {value}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()