"""Differentiable-inverse-problem study (tutorial driver).

The repository already proves that autograd differentiates the *discrete*
scheme correctly (``benchmarks/verify_differentiability.py``,
``benchmarks/verify_diffusive_differentiability.py``).  This study closes the
next loop and measures whether those gradients are *usable*:

A. **Gradient check at an off-optimum parameter** -- autograd ``dJ/dtheta``
   against a central difference of the same code, for the hyperbolic
   (advection speed) and the parabolic (SIPG diffusivity) model.
B. **Identification against discrete data** -- a secant/Newton loop driven only
   by autograd gradients must return the data-generating parameter to
   round-off: the sanity floor of the workflow.
C. **Identification against continuum data (consistency)** -- the measurement
   is the *analytic* solution projected onto the DG space, so the recovered
   parameter can only converge to the continuum truth at the scheme's rate.
   A refinement table plus the observed order ``p_obs`` of the *parameter*
   error is reported; that is the difference between "self-consistent" and
   "correct".
D. **Noise robustness** -- 1 % multiplicative measurement noise; how far does
   the estimate move?
E. **The frozen-``dt`` degeneracy (negative control with teeth)** -- when the
   measurements are produced with a CFL step ``dt ~ 1/c`` and the fit keeps a
   frozen nominal ``dt``, the misfit can be driven to **machine zero** while the
   identified speed is off by tens of per cent.  A workflow that only checks
   the residual would silently report the wrong parameter.

Outputs: ``validation/inverse_problem/evidence.json``,
``validation/inverse_problem/summary.md`` and figures under ``docs/figures/``.

Run from the repository root::

    python tests/inverse_study.py            # full study
    python tests/inverse_study.py --quick    # coarse meshes, no figures
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensordg.convergence import observed_order  # noqa: E402
from tensordg.differentiable import advance_advection_1d  # noqa: E402
from tests._inverse_tools import (  # noqa: E402
    add_relative_noise,
    advection_continuum_data,
    advection_data,
    advection_objective,
    advection_setup,
    central_difference,
    heat_continuum_data,
    heat_data,
    heat_objective,
    heat_setup,
    relative_error,
    scan,
    secant_fit,
    value_and_grad,
)

torch.set_num_threads(2)

OUT = Path("validation/inverse_problem")
FIGDIR = Path("docs/figures")

C_TRUE = 1.0
ALPHA_TRUE = 0.2
CFL = 0.2
MESHES = (8, 12, 16, 24, 32, 48)
#: ``--quick`` stays a smoke run but must keep the two order criteria
#: computable (>= 3 meshes) and the finest-mesh gates reachable.
QUICK_MESHES = (12, 24, 48)
NOISE_LEVEL = 0.01
BOUNDS_C = (0.1, 3.0)
BOUNDS_ALPHA = (1.0e-3, 5.0)


def _verdict(value: float, threshold: float) -> str:
    return "PASS" if value <= threshold else "FAIL"


# ---------------------------------------------------------------------------
# A. gradient check at an off-optimum parameter
# ---------------------------------------------------------------------------

def study_gradient_check() -> dict:
    """Autograd vs central difference, plus the ``1e-3`` teeth control."""
    adv = advection_setup()
    adv_obj = advection_objective(adv, advection_data(adv, C_TRUE))
    heat = heat_setup()
    heat_obj = heat_objective(heat, heat_data(heat, ALPHA_TRUE))

    results = {}
    # the finite-difference step is not arbitrary: see ``study_fd_step_sweep``,
    # where the heat model shows the classic truncation/round-off V-shape.
    for label, objective, theta, step in (
        ("advection_speed", adv_obj, 0.8, 1.0e-6),
        ("heat_diffusivity", heat_obj, 1.0, 1.0e-4),
    ):
        value, grad = value_and_grad(objective, theta)
        reference = central_difference(objective, theta, eps=step)
        shifted = value_and_grad(objective, theta * (1.0 + 1.0e-3))[1]
        results[label] = {
            "theta": theta,
            "objective": value,
            "autograd_gradient": grad,
            "central_difference": reference,
            "relative_gap": relative_error(grad, reference),
            "teeth_relative_gap_at_1e-3_shift": relative_error(shifted, grad),
        }
    for entry in results.values():
        entry["verdict"] = _verdict(entry["relative_gap"], 1.0e-7)
    return results


def study_fd_step_sweep(steps=(1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8)) -> dict:
    """Relative gap between autograd and central differences as a function of eps.

    A central difference of a *differentiable-in-the-solver* model is limited by
    two competing errors: truncation ``O(eps^2)`` and round-off ``O(machine_eps /
    eps)`` amplified by the Crank--Nicolson linear solve.  Sweeping ``eps``
    documents where the sweet spot is instead of silently picking one value.
    """
    adv = advection_setup()
    adv_obj = advection_objective(adv, advection_data(adv, C_TRUE))
    heat = heat_setup()
    heat_obj = heat_objective(heat, heat_data(heat, ALPHA_TRUE))
    sweep = {}
    for label, objective, theta in (
        ("advection_speed", adv_obj, 0.8),
        ("heat_diffusivity", heat_obj, 1.0),
    ):
        grad = value_and_grad(objective, theta)[1]
        sweep[label] = [
            {"eps": step, "central_difference": central_difference(objective, theta, eps=step),
             "relative_gap": relative_error(central_difference(objective, theta, eps=step), grad)}
            for step in steps
        ]
        sweep[label + "_best_gap"] = min(entry["relative_gap"] for entry in sweep[label])
    return sweep


# ---------------------------------------------------------------------------
# B. identification against discrete data
# ---------------------------------------------------------------------------

def study_discrete_identification() -> dict:
    adv = advection_setup()
    heat = heat_setup()
    adv_fit = secant_fit(advection_objective(adv, advection_data(adv, C_TRUE)), 0.8,
                         bounds=BOUNDS_C)
    heat_fit = secant_fit(heat_objective(heat, heat_data(heat, ALPHA_TRUE)), 1.0,
                          bounds=BOUNDS_ALPHA)
    return {
        "advection_speed": {
            "truth": C_TRUE,
            "initial_guess": 0.8,
            "estimate": adv_fit["theta"],
            "relative_error": relative_error(adv_fit["theta"], C_TRUE),
            "iterations": adv_fit["iterations"],
            "final_objective": adv_fit["objective"],
            "final_gradient": adv_fit["gradient"],
        },
        "heat_diffusivity": {
            "truth": ALPHA_TRUE,
            "initial_guess": 1.0,
            "estimate": heat_fit["theta"],
            "relative_error": relative_error(heat_fit["theta"], ALPHA_TRUE),
            "iterations": heat_fit["iterations"],
            "final_objective": heat_fit["objective"],
            "final_gradient": heat_fit["gradient"],
        },
    }


# ---------------------------------------------------------------------------
# C. consistency: identification against continuum data (refinement)
# ---------------------------------------------------------------------------

def study_continuum_consistency(meshes=MESHES) -> dict:
    adv_rows, heat_rows = [], []
    for ncells in meshes:
        adv = advection_setup(ncells=ncells)
        adv_fit = secant_fit(
            advection_objective(adv, advection_continuum_data(adv, C_TRUE)), 0.8,
            bounds=BOUNDS_C,
        )
        heat = heat_setup(ncells=ncells)
        heat_fit = secant_fit(
            heat_objective(heat, heat_continuum_data(heat, ALPHA_TRUE)), 1.0,
            bounds=BOUNDS_ALPHA,
        )
        adv_rows.append({
            "ncells": ncells,
            "h": adv["h"],
            "estimate": adv_fit["theta"],
            "relative_error": relative_error(adv_fit["theta"], C_TRUE),
            "iterations": adv_fit["iterations"],
            "final_objective": adv_fit["objective"],
        })
        heat_rows.append({
            "ncells": ncells,
            "h": heat["h"],
            "estimate": heat_fit["theta"],
            "relative_error": relative_error(heat_fit["theta"], ALPHA_TRUE),
            "iterations": heat_fit["iterations"],
            "final_objective": heat_fit["objective"],
        })

    adv_errors = [row["relative_error"] for row in adv_rows]
    heat_errors = [row["relative_error"] for row in heat_rows]
    adv_order = observed_order([row["ncells"] for row in adv_rows], adv_errors) if len(adv_rows) >= 3 else None
    heat_order = observed_order([row["ncells"] for row in heat_rows], heat_errors) if len(heat_rows) >= 3 else None
    return {
        "advection_speed": {
            "rows": adv_rows,
            "observed_order": adv_order,
            "finest_relative_error": adv_errors[-1],
        },
        "heat_diffusivity": {
            "rows": heat_rows,
            "observed_order": heat_order,
            "finest_relative_error": heat_errors[-1],
        },
    }


# ---------------------------------------------------------------------------
# D. noise robustness
# ---------------------------------------------------------------------------

def study_noise(level: float = NOISE_LEVEL) -> dict:
    adv = advection_setup()
    heat = heat_setup()
    adv_fit = secant_fit(
        advection_objective(adv, add_relative_noise(advection_data(adv, C_TRUE), level)),
        0.8, bounds=BOUNDS_C,
    )
    heat_fit = secant_fit(
        heat_objective(heat, add_relative_noise(heat_data(heat, ALPHA_TRUE), level)),
        1.0, bounds=BOUNDS_ALPHA,
    )
    return {
        "noise_level": level,
        "seed": 20261103,
        "advection_speed": {
            "truth": C_TRUE,
            "estimate": adv_fit["theta"],
            "relative_error": relative_error(adv_fit["theta"], C_TRUE),
            "iterations": adv_fit["iterations"],
        },
        "heat_diffusivity": {
            "truth": ALPHA_TRUE,
            "estimate": heat_fit["theta"],
            "relative_error": relative_error(heat_fit["theta"], ALPHA_TRUE),
            "iterations": heat_fit["iterations"],
        },
    }


# ---------------------------------------------------------------------------
# E. frozen-dt degeneracy (negative control)
# ---------------------------------------------------------------------------

def study_frozen_dt_degeneracy(c_true: float = 1.6, ncells: int = 32,
                               degree: int = 2, nsteps: int = 40) -> dict:
    """Measurements made with a CFL step ``dt * 1/c``, fitted with a frozen step.

    The nominal setup freezes ``dt = CFL*h/(2p+1)``, i.e. the step a production
    run would use at ``c = 1``.  The measurements are generated at ``c_true``
    with the *CFL-consistent* step ``dt_true = CFL*h/(2p+1)/c_true`` and the
    same number of steps, so the physical final time differs.  The frozen-dt
    model can compensate exactly by rescaling the speed, ``c_hat =
    c_true*dt_true/dt_frozen`` -- a perfect residual with the wrong parameter.
    Freezing ``dt`` on the measurement's own step removes the degeneracy.
    """
    setup = advection_setup(ncells=ncells, degree=degree, nsteps=nsteps, cfl=CFL)
    dt_frozen = setup["dt"]
    dt_true = dt_frozen / c_true
    data = advance_advection_1d(
        setup["coeff0"], ncells, degree, speed=c_true, nsteps=nsteps, dt=dt_true,
    ).detach()

    frozen_fit = secant_fit(advection_objective(setup, data), 1.4, bounds=BOUNDS_C)
    matched = dict(setup)
    matched["dt"] = dt_true
    matched["final_time"] = nsteps * dt_true
    matched_fit = secant_fit(advection_objective(matched, data), 1.4, bounds=BOUNDS_C)

    predicted = c_true * dt_true / dt_frozen
    return {
        "c_true": c_true,
        "dt_frozen": dt_frozen,
        "dt_used_for_measurements": dt_true,
        "frozen_dt_model": {
            "estimate": frozen_fit["theta"],
            "predicted_by_time_rescaling": predicted,
            "relative_error": relative_error(frozen_fit["theta"], c_true),
            "final_objective": frozen_fit["objective"],
            "iterations": frozen_fit["iterations"],
            "residual_looks_converged": frozen_fit["objective"] < 1.0e-20,
        },
        "matched_dt_model": {
            "estimate": matched_fit["theta"],
            "relative_error": relative_error(matched_fit["theta"], c_true),
            "final_objective": matched_fit["objective"],
            "iterations": matched_fit["iterations"],
        },
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_landscape(path: Path) -> dict:
    """Misfit landscapes with the true parameter and autograd gradients."""
    adv = advection_setup()
    heat = heat_setup()
    adv_obj = advection_objective(adv, advection_data(adv, C_TRUE))
    heat_obj = heat_objective(heat, heat_data(heat, ALPHA_TRUE))
    speeds = [0.6 + 0.02 * index for index in range(41)]
    alphas = [0.02 + 0.01 * index for index in range(59)]
    adv_scan = scan(adv_obj, speeds)
    heat_scan = scan(heat_obj, alphas)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    panels = (
        (axes[0], adv_scan, "theta", "advection speed c", adv_obj, C_TRUE, speeds, "tab:blue"),
        (axes[1], heat_scan, "theta", "diffusivity alpha", heat_obj, ALPHA_TRUE, alphas, "tab:red"),
    )
    for axis, rows, key, label, objective, truth, grid, color in panels:
        xs = [row[key] for row in rows]
        ys = [max(row["objective"], 1.0e-30) for row in rows]
        axis.semilogy(xs, ys, color=color, lw=2.0, label="misfit J(theta)")
        axis.axvline(truth, color="k", ls="--", lw=1.0, label="truth")
        picks = grid[:: max(1, len(grid) // 6)]
        width = (grid[-1] - grid[0]) * 0.06
        for pick in picks:
            _, grad = value_and_grad(objective, pick)
            level = objective_value(objective, pick)
            axis.annotate(
                "",
                xy=(pick - width * math.copysign(1.0, grad), level),
                xytext=(pick, level),
                arrowprops={"arrowstyle": "->", "color": "0.4", "lw": 1.0},
            )
        axis.set_xlabel(label)
        axis.set_ylabel("J (log scale)")
        axis.set_title(f"autograd descent direction, {label.split()[0]}")
        axis.grid(alpha=0.3)
        axis.legend(loc="upper center", fontsize=8)
    fig.suptitle("Least-squares misfit landscapes (arrows: -dJ/dtheta)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"figure": str(path), "advection_min_objective": min(r["objective"] for r in adv_scan),
            "heat_min_objective": min(r["objective"] for r in heat_scan)}


def objective_value(objective, theta: float) -> float:
    return float(objective(torch.tensor(float(theta), dtype=torch.float64)))


def plot_identification(evidence: dict, path: Path) -> str:
    """Identification convergence: parameter error vs h, and iteration history."""
    continuum = evidence["continuum_consistency"]
    adv_rows = continuum["advection_speed"]["rows"]
    heat_rows = continuum["heat_diffusivity"]["rows"]
    adv = advection_setup()
    heat = heat_setup()
    adv_fit = secant_fit(advection_objective(adv, advection_data(adv, C_TRUE)), 0.8,
                         bounds=BOUNDS_C)
    heat_fit = secant_fit(heat_objective(heat, heat_data(heat, ALPHA_TRUE)), 1.0,
                          bounds=BOUNDS_ALPHA)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axis = axes[0]
    axis.loglog([row["h"] for row in adv_rows], [row["relative_error"] for row in adv_rows],
                "o-", color="tab:blue", label="advection speed c")
    axis.loglog([row["h"] for row in heat_rows], [row["relative_error"] for row in heat_rows],
                "s-", color="tab:red", label="diffusivity alpha")
    axis.set_xlabel("h = 1/ncells")
    axis.set_ylabel("|theta_hat - theta_true| / theta_true")
    axis.set_title("identified-parameter error vs mesh (continuum data)")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)

    axis = axes[1]
    axis.semilogy([entry["iteration"] for entry in adv_fit["history"]],
                  [abs(entry["gradient"]) for entry in adv_fit["history"]],
                  "o-", color="tab:blue", label="advection (discrete data)")
    axis.semilogy([entry["iteration"] for entry in heat_fit["history"]],
                  [abs(entry["gradient"]) for entry in heat_fit["history"]],
                  "s-", color="tab:red", label="heat (discrete data)")
    axis.set_xlabel("secant iteration")
    axis.set_ylabel("|dJ/dtheta|")
    axis.set_title("adjoint-driven fit: gradient collapse")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(quick: bool = False, figures: bool = True) -> dict:
    started = time.time()
    meshes = QUICK_MESHES if quick else MESHES
    evidence = {
        "study": "differentiable inverse problems",
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dtype": "float64",
        "quick_mode": quick,
        "conventions": {
            "cfl": CFL,
            "advection": "P2 upwind DG + RK4, periodic, ncells=32, nsteps=40",
            "heat": "P2 SIPG + Crank-Nicolson, zero Dirichlet, ncells=24, final_time=0.02",
            "optimiser": "secant/Newton on dJ/dtheta from autograd (no finite differences)",
        },
    }
    evidence["gradient_check"] = study_gradient_check()
    evidence["fd_step_sweep"] = study_fd_step_sweep()
    evidence["discrete_identification"] = study_discrete_identification()
    evidence["continuum_consistency"] = study_continuum_consistency(meshes)
    evidence["noise"] = study_noise()
    evidence["frozen_dt_degeneracy"] = study_frozen_dt_degeneracy()

    criteria = {
        "gradient_gap_advection <= 1e-7": _verdict(
            evidence["gradient_check"]["advection_speed"]["relative_gap"], 1.0e-7),
        "gradient_gap_heat <= 1e-7": _verdict(
            evidence["gradient_check"]["heat_diffusivity"]["relative_gap"], 1.0e-7),
        "fd_step_sweep_best <= 1e-7": (
            "PASS" if max(evidence["fd_step_sweep"]["advection_speed_best_gap"],
                          evidence["fd_step_sweep"]["heat_diffusivity_best_gap"]) <= 1.0e-7
            else "FAIL"),
        "discrete_advection_error <= 1e-9": _verdict(
            evidence["discrete_identification"]["advection_speed"]["relative_error"], 1.0e-9),
        "discrete_heat_error <= 1e-6": _verdict(
            evidence["discrete_identification"]["heat_diffusivity"]["relative_error"], 1.0e-6),
        "finest_continuum_advection_error <= 1e-6": _verdict(
            evidence["continuum_consistency"]["advection_speed"]["finest_relative_error"], 1.0e-6),
        "finest_continuum_heat_error <= 1e-6": _verdict(
            evidence["continuum_consistency"]["heat_diffusivity"]["finest_relative_error"], 1.0e-6),
        "continuum_advection_order > 3.2": (
            "PASS" if (evidence["continuum_consistency"]["advection_speed"]["observed_order"] or 0.0) > 3.2
            else "FAIL"),
        "continuum_heat_order > 4.0": (
            "PASS" if (evidence["continuum_consistency"]["heat_diffusivity"]["observed_order"] or 0.0) > 4.0
            else "FAIL"),
        "noise_1pct_advection_error <= 3e-2": _verdict(
            evidence["noise"]["advection_speed"]["relative_error"], 3.0e-2),
        "noise_1pct_heat_error <= 3e-2": _verdict(
            evidence["noise"]["heat_diffusivity"]["relative_error"], 3.0e-2),
        "frozen_dt_trap_is_reproduced": (
            "PASS" if evidence["frozen_dt_degeneracy"]["frozen_dt_model"]["relative_error"] > 1.0e-2
            and evidence["frozen_dt_degeneracy"]["frozen_dt_model"]["residual_looks_converged"]
            else "FAIL"),
        "matched_dt_fit_recovers_truth": _verdict(
            evidence["frozen_dt_degeneracy"]["matched_dt_model"]["relative_error"], 1.0e-9),
    }
    evidence["criteria"] = criteria
    evidence["all_pass"] = all(value == "PASS" for value in criteria.values())

    if figures:
        FIGDIR.mkdir(parents=True, exist_ok=True)
        evidence["figures"] = {
            "landscape": plot_landscape(FIGDIR / "inverse_problem_landscape.png"),
            "identification": plot_identification(
                evidence, FIGDIR / "inverse_problem_identification.png"),
        }
    evidence["wall_time_s"] = time.time() - started
    return evidence


def summary_markdown(evidence: dict) -> str:
    check = evidence["gradient_check"]
    discrete = evidence["discrete_identification"]
    continuum = evidence["continuum_consistency"]
    noise = evidence["noise"]
    frozen = evidence["frozen_dt_degeneracy"]
    lines = [
        "# Differentiable inverse problems -- generated summary",
        "",
        "Generated by `python tests/inverse_study.py`; do not edit by hand.",
        "",
        f"- torch {evidence['torch_version']}, python {evidence['python']}, dtype float64",
        f"- wall time: {evidence['wall_time_s']:.2f} s (quick={evidence['quick_mode']})",
        "",
        "## A. Gradient vs central difference",
        "",
        "| model | theta | autograd dJ/dtheta | central difference | rel. gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, entry in check.items():
        lines.append(
            f"| {name} | {entry['theta']:.3f} | {entry['autograd_gradient']:.10e} | "
            f"{entry['central_difference']:.10e} | {entry['relative_gap']:.3e} |"
        )
    lines += [
        "",
        "## B. Identification against discrete data",
        "",
        "| model | truth | estimate | rel. error | iterations | final J |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, entry in discrete.items():
        lines.append(
            f"| {name} | {entry['truth']:.6g} | {entry['estimate']:.12g} | "
            f"{entry['relative_error']:.3e} | {entry['iterations']} | {entry['final_objective']:.3e} |"
        )
    lines += ["", "## C. Identification against continuum data (consistency)", ""]
    for name in ("advection_speed", "heat_diffusivity"):
        block = continuum[name]
        lines += [
            f"### {name} (observed order p_obs = {block['observed_order']})",
            "",
            "| ncells | h | estimate | rel. error | iterations | final J |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in block["rows"]:
            lines.append(
                f"| {row['ncells']} | {row['h']:.4e} | {row['estimate']:.12g} | "
                f"{row['relative_error']:.3e} | {row['iterations']} | {row['final_objective']:.3e} |"
            )
        lines.append("")
    lines += [
        "## D. Noise robustness ({:.0%} multiplicative noise)".format(noise["noise_level"]),
        "",
        "| model | truth | estimate | rel. error |",
        "|---|---:|---:|---:|",
    ]
    for name in ("advection_speed", "heat_diffusivity"):
        entry = noise[name]
        lines.append(
            f"| {name} | {entry['truth']:.6g} | {entry['estimate']:.8g} | "
            f"{entry['relative_error']:.3e} |"
        )
    lines += [
        "",
        "## E. Frozen-dt degeneracy (negative control)",
        "",
        f"- measurements at c_true = {frozen['c_true']} with dt = {frozen['dt_used_for_measurements']:.6e}",
        f"- frozen-dt fit: c_hat = {frozen['frozen_dt_model']['estimate']:.6f} "
        f"(predicted by time rescaling: {frozen['frozen_dt_model']['predicted_by_time_rescaling']:.6f}), "
        f"rel. error {frozen['frozen_dt_model']['relative_error']:.3e}, "
        f"final J {frozen['frozen_dt_model']['final_objective']:.3e}",
        f"- matched-dt fit: c_hat = {frozen['matched_dt_model']['estimate']:.9f}, "
        f"rel. error {frozen['matched_dt_model']['relative_error']:.3e}",
        "",
        "## Criteria",
        "",
        "| criterion | verdict |",
        "|---|---|",
    ]
    for name, verdict in evidence["criteria"].items():
        lines.append(f"| {name} | {verdict} |")
    lines.append("")
    return "\n".join(lines)


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="coarse meshes, no figures")
    parser.add_argument("--no-figures", action="store_true", help="skip matplotlib output")
    args = parser.parse_args()
    evidence = run(quick=args.quick, figures=not (args.quick or args.no_figures))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (OUT / "summary.md").write_text(summary_markdown(evidence))
    print(summary_markdown(evidence))
    if not evidence["all_pass"]:
        raise SystemExit("one or more criteria failed -- see evidence.json")
    return evidence


if __name__ == "__main__":
    main()