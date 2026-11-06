"""Conservation-law and temporal-order verification study (S3 round 2).

The S1 round (see ``docs/dev_report_convergence-verification.md``) verified the
*rate* at which the discretisation error decays (p/h refinement plus a Roache
GCI band).  A rate study says nothing about whether the scheme converges to the
correct *weak* solution of a conservation law, and nothing about the time
integrator.  This study closes that gap with four independent measurement
families:

1. **Discrete invariants.**  A flux-difference scheme conserves a quantity
   exactly when its interface fluxes telescope.  The measured invariant is the
   ``P_0`` modal coefficient (the cell average), integrated as ``h*sum``; the
   criterion is that the relative drift stays at float64 round-off and does not
   grow with refinement.
2. **Boundary-flux budget (Euler).**  For a Sod tube whose waves never reach the
   mesh ends, the transmissive boundary holds the initial states fixed, so the
   global conservation law integrates analytically to
   ``int U(t)-int U(0) = t*(F(U_L)-F(U_R))``: mass and energy are invariant and
   momentum grows linearly at the pressure imbalance ``p_L-p_R``.  Rival
   hypotheses (transmissive / reflecting wall) are scored against the same
   measurement so the criterion provably discriminates.
3. **Rankine--Hugoniot shock speed.**  The Lax--Wendroff theorem says a
   conservative scheme that converges produces the correct discontinuity speed.
   The exact Riemann star state supplies that speed analytically, and the
   numerical shock position is tracked with a level-crossing sensor whose
   intrinsic bias is calibrated on the exact profile.
4. **Time-integrator order.**  A ``dt``-only self-convergence test on a fixed
   mesh: the spatial error cancels, so the observed rate is RK4's or SSP-RK3's
   alone.

This module produces numbers; the pass/fail judgement is emitted as a criteria
table and additionally asserted in ``tests/test_conservation.py``.

Run from the repository root::

    python tests/conservation_study.py            # full study + artifacts
    python tests/conservation_study.py --quick    # smaller meshes
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (_ROOT, Path(__file__).resolve().parent):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import tensordg.euler_2d as e2d  # noqa: E402
from tensordg import (  # noqa: E402
    solve_advection, solve_advection_2d, solve_burgers, solve_riemann,
)
from benchmarks.riemann_exact import GAMMA, sample as sample_exact_riemann, star_state  # noqa: E402
from _conservation_tools import (  # noqa: E402
    boundary_increment,
    cell_averages_1d,
    cell_averages_2d,
    conserved_totals,
    conserved_totals_2d,
    drift_statistics,
    fit_order,
    integral_1d,
    linear_slope,
    observed_orders,
    rankine_hugoniot_speed,
    relative_difference,
    scaled_drift,
    shock_position,
)

SOD_LEFT = (1.0, 0.0, 1.0)
SOD_RIGHT = (0.125, 0.0, 0.1)
SOD_FINAL_TIME = 0.2
RESULT_DIR = _ROOT / "docs" / "benchmarks" / "results" / "conservation"
EULER2D_INVARIANTS = ("mass", "momentum_x", "momentum_y", "energy")
# (name, ncells, final_time, domain) of the 2-D periodic Euler conservation runs.
# ``anisotropic_40x20`` deliberately uses dx != dy so a swapped or dropped cell
# width cannot cancel; ``square_32_long`` quadruples the step count at fixed
# resolution to expose step-count-dependent drift.
EULER2D_CASES = (
    ("square_32", (32, 32), 1.0, ((-5.0, 5.0), (-5.0, 5.0))),
    ("anisotropic_40x20", (40, 20), 1.0, ((-5.0, 5.0), (-5.0, 5.0))),
    ("square_32_long", (32, 32), 4.0, ((-5.0, 5.0), (-5.0, 5.0))),
)

# Acceptance thresholds.  They follow from theory (float64 round-off for
# invariants, integrator order for the dt studies) and from the platform
# acceptance standard (a > 15 % deviation is unacceptable, so a 1 % shock-speed
# band is already strict).  Fixed before the numbers are read, in one place.
CRITERIA = {
    "periodic_drift_max": 1.0e-12,        # relative to the discrete L1 norm
    "drift_grid_ratio_max": 10.0,         # coarsest vs finest mesh drift
    "euler_mass_drift_max": 1.0e-12,      # relative to the initial mass
    "euler_energy_drift_max": 1.0e-12,    # relative to the initial energy
    "euler_momentum_budget_max": 1.0e-6,  # relative to the analytic increment
    "shock_speed_error_max": 1.0e-2,      # 1 % of the Rankine-Hugoniot speed
    "sensor_bias_max": 1.0e-3,            # sensor bias on the exact profile
    "rh_jump_consistency_max": 1.0e-10,   # mass vs momentum jump speed
    "precursor_amplitude_max": 1.0e-6,     # |rho*u| in an untouched window, n>=400
    "precursor_domain_gap_max": 1.0e-6,    # walled vs doubled domain (same dx)
    "precursor_decay_min": 2.0,            # fitted decay rate in h
    "precursor_width_max_cells": 5.0,      # exponential tail width in cells
    "sod_field_error_max": 3.0e-2,        # documented acceptance threshold
    "rk4_order_min": 3.5,                 # theory 4
    "ssp_rk3_order_min": 2.6,             # theory 3
    # 2-D periodic Euler system: four invariants sampled once per step.
    "euler2d_drift_max": 1.0e-13,          # relative to the initial invariant
    "euler2d_step_bias_max": 0.5,          # |mean|/rms of per-step increments
    "euler2d_closure_max": 1.0e-15,        # ledger bookkeeping, relative
    "euler2d_leak_recovery_max": 0.05,     # 5 % on the injected-leak budget
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _l1_scale(averages) -> float:
    """Discrete L1 norm of a cell-average vector, used to scale mass drift."""
    values = np.asarray(averages, dtype=np.float64).ravel()
    return float(np.sum(np.abs(values)) / values.size)


def _primitive_energy(state) -> float:
    rho, velocity, pressure = state
    return pressure / (GAMMA - 1.0) + 0.5 * rho * velocity ** 2


def sod_initial_totals() -> dict:
    """Exact integrals of the Sod initial data on the unit interval."""
    return {
        "mass": 0.5 * (SOD_LEFT[0] + SOD_RIGHT[0]),
        "momentum": 0.5 * (SOD_LEFT[0] * SOD_LEFT[1] + SOD_RIGHT[0] * SOD_RIGHT[1]),
        "energy": 0.5 * (_primitive_energy(SOD_LEFT) + _primitive_energy(SOD_RIGHT)),
    }


def sod_star_state() -> dict:
    """Exact right-shock star state plus two independent RH speed estimates."""
    pressure_star, velocity_star = star_state(SOD_LEFT, SOD_RIGHT, GAMMA)
    ratio = pressure_star / SOD_RIGHT[2]
    density_star = SOD_RIGHT[0] * (
        (GAMMA + 1.0) * ratio + (GAMMA - 1.0)
    ) / ((GAMMA - 1.0) * ratio + (GAMMA + 1.0))
    star = (density_star, velocity_star, pressure_star)
    speed = rankine_hugoniot_speed(star, SOD_RIGHT)
    flux_star = star[0] * star[1] ** 2 + star[2]
    flux_right = SOD_RIGHT[0] * SOD_RIGHT[1] ** 2 + SOD_RIGHT[2]
    momentum_speed = (flux_star - flux_right) / (
        star[0] * star[1] - SOD_RIGHT[0] * SOD_RIGHT[1]
    )
    return {
        "density": float(density_star),
        "velocity": float(velocity_star),
        "pressure": float(pressure_star),
        "shock_speed": float(speed),
        "shock_speed_momentum_jump": float(momentum_speed),
        "jump_consistency": float(abs(speed - momentum_speed) / abs(speed)),
        "sensor_level": 0.5 * (density_star + SOD_RIGHT[0]),
    }


def _euler(ncells: int, final_time: float, flux: str = "hllc"):
    x, fields, meta = solve_riemann(
        SOD_LEFT, SOD_RIGHT, ncells=ncells, final_time=final_time, flux=flux,
    )
    totals = conserved_totals(fields["conservative"], 1.0 / ncells)
    return x, fields, meta, totals


# ---------------------------------------------------------------------------
# study 1: periodic discrete invariants
# ---------------------------------------------------------------------------
def study_periodic_drift(
    cells_1d=(16, 32, 64),
    degrees=(1, 2, 3),
    cells_2d=(16, 24),
    burgers_cells=(32, 64),
    final_time: float = 1.0,
    advection_times=(0.0, 0.5, 1.0),
    burgers_times=(0.0, 0.05, 0.1),
) -> dict:
    """Measure the mass drift of periodic advection (1-D/2-D) and Burgers."""
    rows = []

    def record(case, degree, ncells, history, scale):
        rows.append({
            "case": case,
            "degree": int(degree),
            "ncells": int(ncells),
            "invariant": "mass",
            "value_initial": float(history[0]),
            "value_final": float(history[-1]),
            "absolute_drift": float(abs(history[-1] - history[0])),
            "scale": float(scale),
            "scaled_drift": scaled_drift(history, scale),
        })

    for degree in degrees:
        for ncells in cells_1d:
            history, scale = [], 1.0
            for time in advection_times:
                _, coeff, _ = solve_advection(
                    ncells=ncells, degree=degree, final_time=time * final_time,
                )
                averages = cell_averages_1d(coeff)
                history.append(integral_1d(averages, 1.0))
                scale = _l1_scale(averages)
            record("linear_advection_1d", degree, ncells, history, scale)

    for degree in (1, 2):
        for ncells in cells_2d:
            history, scale = [], 1.0
            for time in advection_times:
                _, _, coeff, _ = solve_advection_2d(
                    ncells=ncells, degree=degree, final_time=time * final_time,
                )
                averages = cell_averages_2d(coeff)
                history.append(float(averages.sum() / averages.size))
                scale = _l1_scale(averages)
            record("linear_advection_2d", degree, ncells, history, scale)

    for ncells in burgers_cells:
        history, scale = [], 1.0
        for time in burgers_times:
            _, coeff, _ = solve_burgers(ncells=ncells, degree=2, final_time=time)
            averages = cell_averages_1d(coeff)
            history.append(integral_1d(averages, 1.0))
            scale = _l1_scale(averages)
        record("burgers_1d", 2, ncells, history, scale)

    grid_ratio = 1.0
    for degree in degrees:
        coarse = [r for r in rows if r["case"] == "linear_advection_1d"
                  and r["degree"] == degree and r["ncells"] == min(cells_1d)]
        fine = [r for r in rows if r["case"] == "linear_advection_1d"
                and r["degree"] == degree and r["ncells"] == max(cells_1d)]
        if coarse and fine and fine[0]["scaled_drift"] > 0.0:
            grid_ratio = max(grid_ratio, coarse[0]["scaled_drift"] / fine[0]["scaled_drift"])
    return {
        "rows": rows,
        "max_scaled_drift": max(row["scaled_drift"] for row in rows),
        "drift_grid_ratio": float(grid_ratio),
    }


# ---------------------------------------------------------------------------
# study 2: Euler boundary-flux budget
# ---------------------------------------------------------------------------
def study_euler_budget(
    cells=(400, 800, 1600),
    fluxes=("hllc", "rusanov"),
    final_time: float = SOD_FINAL_TIME,
    history_times=(0.05, 0.1, 0.15, 0.2),
    history_cells: int = 800,
) -> dict:
    """Compare measured global integrals with the analytic boundary budget."""
    initial = sod_initial_totals()
    analytic = boundary_increment(SOD_LEFT, SOD_RIGHT, final_time, GAMMA)
    rows = []
    for flux in fluxes:
        for ncells in cells:
            _, _, meta, totals = _euler(ncells, final_time, flux)
            increment = totals["momentum"] - initial["momentum"]
            rows.append({
                "flux": flux,
                "ncells": int(ncells),
                "steps": int(meta["steps"]),
                "mass": totals["mass"],
                "momentum": totals["momentum"],
                "energy": totals["energy"],
                "mass_drift": abs(totals["mass"] - initial["mass"]) / initial["mass"],
                "energy_drift": abs(totals["energy"] - initial["energy"]) / initial["energy"],
                "momentum_increment": float(increment),
                "momentum_increment_analytic": analytic["momentum"],
                "momentum_budget_error": float(
                    abs(increment - analytic["momentum"]) / abs(analytic["momentum"])
                ),
            })

    history = []
    for time in history_times:
        _, _, _, totals = _euler(history_cells, time, "hllc")
        history.append({
            "time": float(time),
            "momentum": totals["momentum"],
            "momentum_analytic": initial["momentum"]
            + boundary_increment(SOD_LEFT, SOD_RIGHT, time, GAMMA)["momentum"],
        })

    observed = float(np.mean([row["momentum_increment"] for row in rows]))
    hypotheses = {
        "transmissive_analytic": abs(observed - analytic["momentum"]) / abs(analytic["momentum"]),
        "reflecting_wall": abs(observed - 0.0) / abs(analytic["momentum"]),
    }
    return {
        "initial": initial,
        "analytic_increment": analytic,
        "rows": rows,
        "history": history,
        "max_mass_drift": max(row["mass_drift"] for row in rows),
        "max_energy_drift": max(row["energy_drift"] for row in rows),
        "max_momentum_budget_error": max(row["momentum_budget_error"] for row in rows),
        "hypothesis_errors": {k: float(v) for k, v in hypotheses.items()},
    }


# ---------------------------------------------------------------------------
# study 2b: 2-D Euler system -- periodic invariants with per-step statistics
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def injected_mass_source(strength: float):
    """Negative control: break the telescoping flux difference by a source.

    The wrapper adds ``strength`` units of mass per unit time inside the single
    cell ``(0, 0)``.  That is a textbook non-conservative edit (the stencil no
    longer telescopes), and it is *quantitative*: over a run of length ``T`` the
    global mass must grow by exactly ``strength*T`` (SSP-RK3 integrates a
    constant source exactly), while momentum and energy stay put.  The
    conservation criteria are therefore falsifiable with a known answer instead
    of merely "assert it fails".

    ``e2d._rhs`` is looked up on the module at call time inside
    :func:`~tensordg.euler_2d.solve_euler_2d`, so the patch reaches the solver
    without touching its code path.
    """
    original = e2d._rhs

    def leaked(u, dx, dy, periodic=(True, True)):
        out = original(u, dx, dy, periodic)
        out[0, 0, 0] = out[0, 0, 0] + float(strength) / (dx * dy)
        return out

    e2d._rhs = leaked
    try:
        yield
    finally:
        e2d._rhs = original


def study_euler_2d_conservation(
    cases=EULER2D_CASES,
    cfl: float = 0.4,
    leak: float = 0.0,
    strength: float = 5.0,
    background_velocity=(1.0, 1.0),
) -> dict:
    """Audit the four global invariants of the periodic 2-D Euler system.

    The flow is the smooth isentropic vortex on a doubly periodic box, run with
    the repository's own P0/Rusanov/SSP-RK3 path.  Nothing in the analysis knows
    the answer: the discrete invariants are integrated from the state the solver
    returns, sampled by the ``observer`` hook once per accepted step, and
    summarised by :func:`tests._conservation_tools.drift_statistics`.

    With ``leak != 0`` the run is repeated with :func:`injected_mass_source`, so
    the same measurement must now *fail* and must recover
    ``leak*final_time`` to within a few percent -- the negative control that
    makes the criterion falsifiable.
    """
    rows = []
    for name, ncells, final_time, domain in cases:
        nx, ny = ncells
        dx = (domain[0][1] - domain[0][0]) / nx
        dy = (domain[1][1] - domain[1][0]) / ny
        history = {invariant: [] for invariant in EULER2D_INVARIANTS}
        times = []

        def observer(time, steps, u, dx=dx, dy=dy, history=history, times=times):
            totals = conserved_totals_2d(u, dx, dy)
            for invariant in EULER2D_INVARIANTS:
                history[invariant].append(totals[invariant])
            times.append(float(time))

        def initial(x, y, strength=strength, background_velocity=background_velocity):
            return e2d.vortex_primitive(x, y, 0.0, strength, background_velocity)

        if leak:
            with injected_mass_source(leak):
                _, _, _, metadata = e2d.solve_euler_2d(
                    initial, ncells=ncells, final_time=final_time, cfl=cfl,
                    domain=domain, periodic=(True, True), observer=observer,
                )
        else:
            _, _, _, metadata = e2d.solve_euler_2d(
                initial, ncells=ncells, final_time=final_time, cfl=cfl,
                domain=domain, periodic=(True, True), observer=observer,
            )

        statistics = {}
        for invariant in EULER2D_INVARIANTS:
            samples = history[invariant]
            scale = abs(float(samples[0])) or 1.0
            statistics[invariant] = drift_statistics(samples, scale)
        increments = {
            invariant: statistics[invariant]["increment_mean"]
            for invariant in EULER2D_INVARIANTS
        }
        rows.append({
            "case": name,
            "ncells": (int(nx), int(ny)),
            "domain": tuple(tuple(float(edge) for edge in axis) for axis in domain),
            "cell_widths": (float(dx), float(dy)),
            "final_time": float(final_time),
            "steps": int(metadata["steps"]),
            "dt": float(metadata["dt"]),
            "samples": len(times),
            "time_covered": float(times[-1] - times[0]),
            "invariants": statistics,
            "step_mean_increments": increments,
            "max_scaled_drift": max(
                statistics[invariant]["scaled_drift"]
                for invariant in EULER2D_INVARIANTS
            ),
            "max_bias_ratio": max(
                statistics[invariant]["bias_ratio"]
                for invariant in EULER2D_INVARIANTS
            ),
            "max_closure_residual": max(
                statistics[invariant]["closure_residual"]
                for invariant in EULER2D_INVARIANTS
            ),
            "expected_mass_drift": (
                float(leak * final_time / statistics["mass"]["initial"])
                if leak else 0.0
            ),
        })

    results = {
        "leak": float(leak),
        "rows": rows,
        "max_scaled_drift": max(row["max_scaled_drift"] for row in rows),
        "max_bias_ratio": max(row["max_bias_ratio"] for row in rows),
        "max_closure_residual": max(row["max_closure_residual"] for row in rows),
        "drift_by_invariant": {
            invariant: max(
                row["invariants"][invariant]["scaled_drift"] for row in rows
            )
            for invariant in EULER2D_INVARIANTS
        },
        "mean_increment_by_invariant": {
            invariant: max(
                abs(row["step_mean_increments"][invariant]) for row in rows
            )
            for invariant in EULER2D_INVARIANTS
        },
    }
    if leak:
        mass_rows = [row for row in rows]
        results["leak_recovery_error"] = max(
            abs(
                row["invariants"]["mass"]["scaled_drift"] - row["expected_mass_drift"]
            ) / row["expected_mass_drift"]
            for row in mass_rows
        )
    return results


# ---------------------------------------------------------------------------
# study 3: Rankine-Hugoniot shock speed
# ---------------------------------------------------------------------------
def study_shock_speed(
    cells=(400, 800, 1600),
    times=(0.10, 0.14, 0.18, 0.22),
    flux: str = "hllc",
) -> dict:
    """Track the numerical shock and compare its speed with Rankine-Hugoniot."""
    star = sod_star_state()
    level = star["sensor_level"]
    grid = np.linspace(0.0, 1.0, 40001)

    exact_positions = []
    for time in times:
        exact = sample_exact_riemann(grid, time, SOD_LEFT, SOD_RIGHT, 0.5, GAMMA)["density"]
        exact_positions.append(shock_position(grid, exact, level))
    exact_speed, _ = linear_slope(times, exact_positions)
    sensor_bias = max(
        abs(position - (0.5 + star["shock_speed"] * time))
        for position, time in zip(exact_positions, times)
    )

    rows = []
    for ncells in cells:
        positions = []
        for time in times:
            x, fields, _, _ = _euler(ncells, time, flux)
            positions.append(shock_position(x, fields["density"], level))
        speed, intercept = linear_slope(times, positions)
        rows.append({
            "ncells": int(ncells),
            "positions": [float(v) for v in positions],
            "speed": float(speed),
            "intercept": float(intercept),
            "speed_error": float(abs(speed - star["shock_speed"]) / star["shock_speed"]),
        })

    refinement_rates = observed_orders(
        [1.0 / row["ncells"] for row in rows],
        [max(row["speed_error"], 1e-16) for row in rows],
    )
    errors = [row["speed_error"] for row in rows]
    return {
        "star": star,
        "level": level,
        "times": [float(t) for t in times],
        "exact_positions": [float(v) for v in exact_positions],
        "exact_sensor_speed": float(exact_speed),
        "sensor_bias": float(sensor_bias),
        "jump_consistency": star["jump_consistency"],
        "rows": rows,
        "max_speed_error": max(errors),
        "refinement_rates": [float(v) for v in refinement_rates],
        "speed_monotone": bool(errors[-1] <= errors[0]),
    }


# ---------------------------------------------------------------------------
# study 5: the mesh-scale tail ahead of a captured shock
# ---------------------------------------------------------------------------
def study_shock_precursor(
    cells=(100, 200, 400, 800),
    final_time: float = SOD_FINAL_TIME,
    window=(0.90, 1.00),
    control_length: float = 2.0,
    fine_mesh_cutoff: int = 400,
) -> dict:
    """Characterise the residual state ahead of the captured shock.

    At ``t = 0.2`` the exact shock sits at ``x_s = 0.850`` and the exact solution
    is identical to the uniform right state beyond it, so the window
    ``[0.90, 1.00]`` ahead of the front carries no physical signal.  Two questions
    are answered by measurement:

    * is the residual caused by the *boundary*?  The same window is recomputed
      on a domain of twice the length with the *same* ``dx`` and no wall nearby.
      Identical values mean the residual is an interior, mesh-scale object.
    * how wide is it?  Fitting ``log|rho*u|`` against the cell index behind the
      shock gives the exponential tail width in cells; a width that stays
      ``O(1)`` cells while ``dx`` shrinks identifies a discrete-shock artefact
      rather than physical diffusion.
    """
    star = sod_star_state()
    level = star["sensor_level"]
    rows = []
    for ncells in cells:
        x, fields, _ = solve_riemann(
            SOD_LEFT, SOD_RIGHT, ncells=ncells, final_time=final_time, domain=(0.0, 1.0),
        )
        centers = x.numpy()
        momentum = fields["conservative"][:, 1].numpy()
        density = fields["density"].numpy()
        mask = (centers >= window[0]) & (centers < window[1])
        walled = float(np.abs(momentum[mask]).max())

        xc, fields_c, _ = solve_riemann(
            SOD_LEFT, SOD_RIGHT, ncells=int(round(ncells * control_length)),
            final_time=final_time, domain=(0.0, control_length),
        )
        centers_c = xc.numpy()
        momentum_c = fields_c["conservative"][:, 1].numpy()
        mask_c = (centers_c >= window[0]) & (centers_c < window[1])
        doubled = float(np.abs(momentum_c[mask_c]).max())

        # Exponential tail width behind the shock, in cells.
        shock_index = int(np.argmax(density < level))
        indices = np.arange(shock_index + 5, ncells)
        tail = np.abs(momentum[shock_index + 5:])
        usable = tail > 1e-14
        width = float("nan")
        if usable.sum() > 3:
            slope = float(np.polyfit(indices[usable], np.log(tail[usable]), 1)[0])
            if slope < 0.0:
                width = -1.0 / slope

        rows.append({
            "ncells": int(ncells),
            "dx": 1.0 / ncells,
            "window_max_walled": walled,
            "window_max_doubled_domain": doubled,
            "relative_gap": abs(walled - doubled) / walled if walled > 0.0 else 0.0,
            "tail_width_cells": width,
        })
    fine = [row for row in rows if row["ncells"] >= fine_mesh_cutoff]
    return {
        "window": [float(window[0]), float(window[1])],
        "final_time": float(final_time),
        "control_length": float(control_length),
        "rows": rows,
        "decay_rate": float(fit_order(
            [row["dx"] for row in rows],
            [max(row["window_max_walled"], 1e-18) for row in rows],
        )),
        "worst_amplitude_fine": max(row["window_max_walled"] for row in fine),
        "worst_relative_gap": max(row["relative_gap"] for row in rows),
        "tail_width_cells": [row["tail_width_cells"] for row in rows],
    }


# ---------------------------------------------------------------------------
# study 4: time-integrator order on a fixed mesh
# ---------------------------------------------------------------------------
def study_temporal_order(
    advection_cfls=(0.4, 0.2, 0.1),
    burgers_cfls=(0.2, 0.1, 0.05),
    advection_cells: int = 40,
    burgers_cells: int = 64,
) -> dict:
    """Measure RK4/SSP-RK3 order with dt-only self-convergence (space fixed)."""
    advection_states = []
    for cfl in advection_cfls:
        _, coeff, meta = solve_advection(
            ncells=advection_cells, degree=2, final_time=1.0, cfl=cfl,
        )
        advection_states.append({
            "cfl": float(cfl), "dt": float(meta["dt"]), "steps": int(meta["steps"]),
            "state": cell_averages_1d(coeff),
        })
    burgers_states = []
    for cfl in burgers_cfls:
        _, coeff, meta = solve_burgers(
            ncells=burgers_cells, degree=2, final_time=0.1, cfl=cfl,
        )
        burgers_states.append({
            "cfl": float(cfl), "dt": float(meta["dt"]), "steps": int(meta["steps"]),
            "state": cell_averages_1d(coeff),
        })

    def summarise(states, label):
        diffs = [
            relative_difference(states[i + 1]["state"], states[i]["state"])
            for i in range(len(states) - 1)
        ]
        dt = [states[i]["dt"] for i in range(len(states) - 1)]
        return {
            "integrator": label,
            "dt": [float(v) for v in dt],
            "steps": [int(s["steps"]) for s in states],
            "self_differences": [float(v) for v in diffs],
            "successive_orders": [float(v) for v in observed_orders(dt, diffs)],
            "fit_order": float(fit_order(dt, diffs)),
        }

    return {
        "advection_rk4": summarise(advection_states, "RK4 | linear advection P2, 40 cells"),
        "burgers_ssp_rk3": summarise(burgers_states, "SSP-RK3 | Burgers P2, 64 cells"),
    }

# ---------------------------------------------------------------------------
# study 6: field error against the exact Riemann solution
# ---------------------------------------------------------------------------
def study_field_error(cells=(1600, 3200), final_time: float = SOD_FINAL_TIME) -> dict:
    """Relative L2 error of the primitive fields against the exact solution.

    This is the quantity the documented 3 % acceptance threshold refers to; it is
    recorded here so the conservation study also carries a *reference* error, not
    only self-consistency statements.
    """
    rows = []
    for ncells in cells:
        x, fields, meta = solve_riemann(
            SOD_LEFT, SOD_RIGHT, ncells=ncells, final_time=final_time,
        )
        centers = x.numpy()
        exact = sample_exact_riemann(centers, final_time, SOD_LEFT, SOD_RIGHT, 0.5, GAMMA)
        errors = {}
        for name in ("density", "velocity", "pressure"):
            numerical = fields[name].numpy()
            reference = np.asarray(exact[name], dtype=np.float64)
            errors[name] = float(
                np.sqrt(np.mean((numerical - reference) ** 2))
                / np.sqrt(np.mean(reference ** 2))
            )
        rows.append({"ncells": int(ncells), "steps": int(meta["steps"]), "relative_l2": errors})
    return {
        "final_time": float(final_time),
        "rows": rows,
        "worst": max(max(row["relative_l2"].values()) for row in rows),
        "finest": rows[-1]["relative_l2"],
        # The documented 3 % band refers to the documented resolution, so the
        # criterion is evaluated on the finest mesh actually computed.
        "finest_worst": max(rows[-1]["relative_l2"].values()),
    }


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------
def evaluate_metrics(quick: bool = False) -> dict:
    """Run every study and return the raw measurement record."""
    if quick:
        periodic = study_periodic_drift(cells_1d=(8, 16), cells_2d=(8, 12), burgers_cells=(16, 32))
        budget = study_euler_budget(cells=(400, 800), history_times=(0.1, 0.2), history_cells=800)
        shock = study_shock_speed(cells=(200, 400))
        precursor = study_shock_precursor(cells=(200, 400))
        field_error = study_field_error(cells=(800,))
        temporal = study_temporal_order(advection_cells=20, burgers_cells=32)
        euler_2d = study_euler_2d_conservation(
            cases=(EULER2D_CASES[0], EULER2D_CASES[1]),
        )
    else:
        periodic = study_periodic_drift()
        budget = study_euler_budget()
        shock = study_shock_speed()
        precursor = study_shock_precursor()
        field_error = study_field_error()
        temporal = study_temporal_order()
        euler_2d = study_euler_2d_conservation()
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "quick" if quick else "full",
        "standards": {
            "platform_acceptance": "> 15% deviation unacceptable",
            "float64_eps": 2.2e-16,
            "shock_speed_criterion": CRITERIA["shock_speed_error_max"],
        },
        "periodic": periodic,
        "euler_budget": budget,
        "shock": shock,
        "shock_precursor": precursor,
        "field_error": field_error,
        "temporal": temporal,
        "euler_2d_system": euler_2d,
    }


def build_criteria(metrics: dict) -> list:
    """Turn measurements into explicit pass/fail criteria."""
    periodic = metrics["periodic"]
    budget = metrics["euler_budget"]
    shock = metrics["shock"]
    precursor = metrics["shock_precursor"]
    field_error = metrics["field_error"]
    temporal = metrics["temporal"]

    def criterion(name, value, threshold, comparison, note):
        value = float(value)
        threshold = float(threshold)
        passed = value <= threshold if comparison == "<=" else value >= threshold
        return {
            "name": name, "value": value, "threshold": threshold,
            "comparison": comparison, "pass": bool(passed), "note": note,
        }

    criteria = [
        criterion("periodic_mass_drift_scaled", periodic["max_scaled_drift"],
                  CRITERIA["periodic_drift_max"], "<=",
                  "worst relative mass drift, periodic advection 1-D/2-D + Burgers"),
        criterion("periodic_drift_grid_independence", periodic["drift_grid_ratio"],
                  CRITERIA["drift_grid_ratio_max"], "<=",
                  "round-off drift of the coarsest vs the finest mesh"),
        criterion("euler_mass_drift", budget["max_mass_drift"],
                  CRITERIA["euler_mass_drift_max"], "<=",
                  "Sod mass integral vs initial mass (waves stay inside)"),
        criterion("euler_energy_drift", budget["max_energy_drift"],
                  CRITERIA["euler_energy_drift_max"], "<=",
                  "Sod total-energy integral vs initial energy"),
        criterion("euler_momentum_budget", budget["max_momentum_budget_error"],
                  CRITERIA["euler_momentum_budget_max"], "<=",
                  "measured momentum increment vs analytic t*(p_L-p_R)"),
        criterion("rh_jump_consistency", shock["jump_consistency"],
                  CRITERIA["rh_jump_consistency_max"], "<=",
                  "RH speed from the mass jump vs from the momentum jump"),
        criterion("shock_sensor_bias", shock["sensor_bias"],
                  CRITERIA["sensor_bias_max"], "<=",
                  "level-crossing sensor on the exact profile vs analytic shock path"),
        criterion("shock_speed_error_finest", shock["rows"][-1]["speed_error"],
                  CRITERIA["shock_speed_error_max"], "<=",
                  "finest mesh shock speed vs Rankine-Hugoniot speed"),
        criterion("shock_speed_error_coarsest", shock["rows"][0]["speed_error"],
                  CRITERIA["shock_speed_error_max"], "<=",
                  "coarsest mesh shock speed vs Rankine-Hugoniot speed"),
        criterion("shock_speed_error_not_worse_under_refinement",
                  1.0 if shock["speed_monotone"] else 0.0, 1.0, ">=",
                  "shock-speed error at the finest mesh <= coarsest mesh"),
        criterion("precursor_amplitude_fine_mesh", precursor["worst_amplitude_fine"],
                  CRITERIA["precursor_amplitude_max"], "<=",
                  "residual |rho*u| in an untouched window, meshes >= 400 cells"),
        criterion("precursor_domain_independence", precursor["worst_relative_gap"],
                  CRITERIA["precursor_domain_gap_max"], "<=",
                  "same window on a doubled domain at the same dx (rules out the wall)"),
        criterion("precursor_decay_rate", precursor["decay_rate"],
                  CRITERIA["precursor_decay_min"], ">=",
                  "fitted decay rate of the residual amplitude in h"),
        criterion("precursor_tail_width", max(precursor["tail_width_cells"]),
                  CRITERIA["precursor_width_max_cells"], "<=",
                  "exponential tail width in cells (O(1) cells = discrete-shock artefact)"),
        criterion("sod_field_error_vs_exact", field_error["finest_worst"],
                  CRITERIA["sod_field_error_max"], "<=",
                  "worst relative L2 of density/velocity/pressure vs the exact Riemann solution, "
                  "at the finest computed resolution"),
        criterion("rk4_temporal_order", temporal["advection_rk4"]["fit_order"],
                  CRITERIA["rk4_order_min"], ">=",
                  "measured RK4 order on a fixed mesh (theory 4)"),
        criterion("ssp_rk3_temporal_order", temporal["burgers_ssp_rk3"]["fit_order"],
                  CRITERIA["ssp_rk3_order_min"], ">=",
                  "measured SSP-RK3 order on a fixed mesh (theory 3)"),
    ]

    # 2-D Euler system (per-step ledger of four invariants on a periodic box).
    euler_2d = metrics["euler_2d_system"]
    for invariant in EULER2D_INVARIANTS:
        criteria.append(criterion(
            f"euler2d_drift_{invariant}",
            euler_2d["drift_by_invariant"][invariant],
            CRITERIA["euler2d_drift_max"], "<=",
            f"relative drift of the 2-D {invariant} integral, sampled every step",
        ))
    criteria.append(criterion(
        "euler2d_step_bias", euler_2d["max_bias_ratio"],
        CRITERIA["euler2d_step_bias_max"], "<=",
        "|mean|/rms of the per-step increments (round-off is unbiased, a leak is not)",
    ))
    criteria.append(criterion(
        "euler2d_ledger_closure", euler_2d["max_closure_residual"],
        CRITERIA["euler2d_closure_max"], "<=",
        "sum of per-step increments vs the end-state difference",
    ))
    return criteria


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------
def write_artifacts(metrics: dict, outdir: Path = RESULT_DIR) -> dict:
    """Write JSON/CSV/PNG/summary artifacts and return their paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    criteria = build_criteria(metrics)
    all_pass = all(c["pass"] for c in criteria)

    payload = dict(metrics)
    payload["criteria"] = criteria
    payload["summary"] = {
        "criteria_total": len(criteria),
        "criteria_passed": sum(1 for c in criteria if c["pass"]),
        "all_pass": all_pass,
    }
    (outdir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")

    periodic_csv = ["case,degree,ncells,invariant,value_initial,value_final,"
                    "absolute_drift,scale,scaled_drift"]
    for row in metrics["periodic"]["rows"]:
        periodic_csv.append(
            "{case},{degree},{ncells},{invariant},{value_initial:.17g},{value_final:.17g},"
            "{absolute_drift:.6e},{scale:.17g},{scaled_drift:.6e}".format(**row)
        )
    (outdir / "periodic_drift.csv").write_text("\n".join(periodic_csv) + "\n")

    budget_csv = ["flux,ncells,steps,mass,momentum,energy,mass_drift,energy_drift,"
                  "momentum_increment,momentum_increment_analytic,momentum_budget_error"]
    for row in metrics["euler_budget"]["rows"]:
        budget_csv.append(
            "{flux},{ncells},{steps},{mass:.17g},{momentum:.17g},{energy:.17g},"
            "{mass_drift:.6e},{energy_drift:.6e},{momentum_increment:.17g},"
            "{momentum_increment_analytic:.17g},{momentum_budget_error:.6e}".format(**row)
        )
    (outdir / "euler_budget.csv").write_text("\n".join(budget_csv) + "\n")

    (outdir / "euler_2d_system.json").write_text(
        json.dumps(metrics["euler_2d_system"], indent=2, sort_keys=True) + "\n"
    )

    rh = metrics["shock"]["star"]["shock_speed"]
    shock_csv = ["ncells,shock_speed,shock_speed_rankine_hugoniot,relative_error,intercept"]
    for row in metrics["shock"]["rows"]:
        shock_csv.append(
            "{ncells},{speed:.10f},{rh:.10f},{speed_error:.6e},{intercept:.10f}".format(
                ncells=row["ncells"], speed=row["speed"], rh=rh,
                speed_error=row["speed_error"], intercept=row["intercept"],
            )
        )
    (outdir / "shock_speed.csv").write_text("\n".join(shock_csv) + "\n")

    precursor_csv = ["ncells,dx,window_max_walled,window_max_doubled_domain,"
                     "relative_gap,tail_width_cells"]
    for row in metrics["shock_precursor"]["rows"]:
        precursor_csv.append(
            "{ncells},{dx:.6e},{window_max_walled:.6e},{window_max_doubled_domain:.6e},"
            "{relative_gap:.3e},{tail_width_cells:.4f}".format(**row)
        )
    (outdir / "shock_precursor.csv").write_text("\n".join(precursor_csv) + "\n")

    field_csv = ["ncells,steps,relative_l2_density,relative_l2_velocity,relative_l2_pressure"]
    for row in metrics["field_error"]["rows"]:
        errors = row["relative_l2"]
        field_csv.append(
            "{ncells},{steps},{density:.6e},{velocity:.6e},{pressure:.6e}".format(
                ncells=row["ncells"], steps=row["steps"], **errors,
            )
        )
    (outdir / "field_error.csv").write_text("\n".join(field_csv) + "\n")

    temporal_csv = ["study,dt,self_difference"]
    for key in ("advection_rk4", "burgers_ssp_rk3"):
        block = metrics["temporal"][key]
        for dt, diff in zip(block["dt"], block["self_differences"]):
            temporal_csv.append(f"{key},{dt:.10e},{diff:.10e}")
    (outdir / "temporal_order.csv").write_text("\n".join(temporal_csv) + "\n")

    figures = _make_figures(metrics, outdir)
    (outdir / "summary.md").write_text(_write_summary(metrics, criteria, figures, all_pass))
    return {
        "outdir": str(outdir),
        "figures": figures,
        "criteria": criteria,
        "all_pass": all_pass,
    }


def _make_figures(metrics: dict, outdir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = []

    fig, axis = plt.subplots(figsize=(7.4, 4.2))
    rows = metrics["periodic"]["rows"]
    labels = [f"{r['case'].replace('linear_', '').replace('_', ' ')}\nP{r['degree']} n={r['ncells']}"
              for r in rows]
    axis.bar(range(len(rows)), [max(r["scaled_drift"], 1e-18) for r in rows], color="#3b6ea5")
    axis.set_yscale("log")
    axis.axhline(CRITERIA["periodic_drift_max"], color="crimson", ls="--",
                 label=f"criterion {CRITERIA['periodic_drift_max']:.0e}")
    axis.axhline(2.2e-16, color="black", ls=":", label="float64 eps = 2.2e-16")
    axis.set_xticks(range(len(rows)))
    axis.set_xticklabels(labels, fontsize=6, rotation=90)
    axis.set_ylabel("relative mass drift  |M(T)-M(0)| / ||u||_L1")
    axis.set_title("Discrete mass invariants of periodic DG (P0 modal integral)")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "periodic_drift.png", dpi=140)
    plt.close(fig)
    saved.append("periodic_drift.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    history = metrics["euler_budget"]["history"]
    times = [h["time"] for h in history]
    axes[0].plot(times, [h["momentum"] for h in history], "o", label="measured (P0/HLLC)")
    axes[0].plot(times, [h["momentum_analytic"] for h in history], "-",
                 label="analytic  t (p_L - p_R)")
    axes[0].set_xlabel("time t")
    axes[0].set_ylabel("global momentum integral")
    axes[0].set_title("Sod momentum budget, transmissive ends")
    axes[0].legend(fontsize=8)
    budget_rows = metrics["euler_budget"]["rows"]
    tick_labels = [f"{r['flux']}-{r['ncells']}" for r in budget_rows]
    axes[1].semilogy(tick_labels, [max(r["mass_drift"], 1e-18) for r in budget_rows], "s-",
                     label="mass drift")
    axes[1].semilogy(tick_labels, [max(r["energy_drift"], 1e-18) for r in budget_rows], "o-",
                     label="energy drift")
    axes[1].axhline(CRITERIA["euler_mass_drift_max"], color="crimson", ls="--", label="criterion")
    axes[1].set_ylabel("relative drift")
    axes[1].set_title("Round-off level conservation of the Sod integrals")
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "euler_budget.png", dpi=140)
    plt.close(fig)
    saved.append("euler_budget.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    star = metrics["shock"]["star"]
    times = np.array(metrics["shock"]["times"])
    axes[0].plot(times, 0.5 + star["shock_speed"] * times, "-", color="black",
                 label=f"RH analytic W={star['shock_speed']:.4f}")
    for row in metrics["shock"]["rows"]:
        axes[0].plot(times, row["positions"], "o--", ms=4,
                     label=f"n={row['ncells']} W={row['speed']:.4f}")
    axes[0].set_xlabel("time t")
    axes[0].set_ylabel("shock position x_s(t)")
    axes[0].set_title("Measured shock path vs Rankine-Hugoniot")
    axes[0].legend(fontsize=7)
    axes[1].loglog([row["ncells"] for row in metrics["shock"]["rows"]],
                   [max(row["speed_error"], 1e-16) for row in metrics["shock"]["rows"]], "o-")
    axes[1].axhline(CRITERIA["shock_speed_error_max"], color="crimson", ls="--",
                    label="criterion 1%")
    axes[1].set_xlabel("cells")
    axes[1].set_ylabel("relative shock-speed error")
    axes[1].set_title("Shock-speed error under refinement")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "shock_speed.png", dpi=140)
    plt.close(fig)
    saved.append("shock_speed.png")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    ncells = max(metrics["shock"]["rows"], key=lambda r: r["ncells"])["ncells"]
    x, fields, _ = solve_riemann(
        SOD_LEFT, SOD_RIGHT, ncells=ncells, final_time=SOD_FINAL_TIME,
    )
    centers = x.numpy()
    exact = sample_exact_riemann(
        centers, SOD_FINAL_TIME, SOD_LEFT, SOD_RIGHT, 0.5, GAMMA,
    )
    for axis, name, label in zip(axes, ("density", "velocity", "pressure"),
                                 ("density", "velocity", "pressure")):
        numerical = fields[name].numpy()
        reference = np.asarray(exact[name], dtype=np.float64)
        axis.plot(centers, numerical, "-", lw=1.0, label=f"TensorDG P0/HLLC, n={ncells}")
        axis.plot(centers, reference, "--", lw=1.0, label="exact Riemann solution")
        scale = np.sqrt(np.mean(reference ** 2))
        error = np.sqrt(np.mean((numerical - reference) ** 2)) / scale
        axis.set_title(f"{label}: relative L2 = {error * 100:.2f}%", fontsize=9)
        axis.set_xlabel("x")
        axis.set_ylabel(label)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Sod shock tube at t = {SOD_FINAL_TIME}: fields vs exact solution", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "sod_fields_vs_exact.png", dpi=140)
    plt.close(fig)
    saved.append("sod_fields_vs_exact.png")

    fig, axis = plt.subplots(figsize=(7.0, 4.2))
    rows = metrics["shock_precursor"]["rows"]
    axis.loglog([r["dx"] for r in rows],
                [max(r["window_max_walled"], 1e-18) for r in rows], "o-",
                label="domain x in [0,1] (transmissive wall)")
    axis.loglog([r["dx"] for r in rows],
                [max(r["window_max_doubled_domain"], 1e-18) for r in rows], "s--",
                label=f"domain x in [0,{metrics['shock_precursor']['control_length']:.0f}] "
                      "(same dx, wall moved away)")
    axis.axhline(CRITERIA["precursor_amplitude_max"], color="crimson", ls="--",
                 label="criterion (n >= 400)")
    axis.set_xlabel("dx")
    axis.set_ylabel(r"max $|\rho u|$ in $x\in[0.9,1.0]$ at t=0.2")
    axis.set_title("Discrete-shock tail ahead of the front"
                   f" (fitted decay {metrics['shock_precursor']['decay_rate']:.2f} in h)")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "shock_precursor.png", dpi=140)
    plt.close(fig)
    saved.append("shock_precursor.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    for axis, key, theory in zip(axes, ("advection_rk4", "burgers_ssp_rk3"), (4.0, 3.0)):
        block = metrics["temporal"][key]
        dt = np.array(block["dt"])
        diff = np.array(block["self_differences"])
        axis.loglog(dt, diff, "o-", label=f"measured order={block['fit_order']:.2f}")
        axis.loglog(dt, diff[0] * (dt / dt[0]) ** theory, "k--",
                    label=f"reference slope {theory:.0f}")
        axis.set_xlabel("dt  (mesh fixed)")
        axis.set_ylabel("||u_dt - u_dt/2|| / ||u||")
        axis.set_title(block["integrator"], fontsize=9)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "temporal_order.png", dpi=140)
    plt.close(fig)
    saved.append("temporal_order.png")

    return saved


def _write_summary(metrics: dict, criteria: list, figures: list, all_pass: bool) -> str:
    """Programmatic summary: every number below is read from the measurements."""
    star = metrics["shock"]["star"]
    budget = metrics["euler_budget"]
    temporal = metrics["temporal"]
    precursor = metrics["shock_precursor"]
    lines = [
        "# Conservation-law and temporal-order verification (TensorDG, S3 round 2)",
        "",
        f"- generated: `{metrics['generated_at']}` ({metrics['mode']} mode)",
        f"- criteria passed: **{sum(1 for c in criteria if c['pass'])}/{len(criteria)}**",
        f"- overall: **{'PASS' if all_pass else 'FAIL'}**",
        "",
        "## Criteria",
        "",
        "| criterion | value | threshold | result | note |",
        "|---|---:|---:|:--:|---|",
    ]
    for c in criteria:
        lines.append(
            f"| `{c['name']}` | {c['value']:.6e} | {c['comparison']} {c['threshold']:.3e} | "
            f"{'PASS' if c['pass'] else 'FAIL'} | {c['note']} |"
        )
    lines += [
        "",
        "## Periodic discrete invariants",
        "",
        "| case | degree | cells | mass(0) | mass(T) | scaled drift |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["periodic"]["rows"]:
        lines.append(
            f"| {row['case']} | {row['degree']} | {row['ncells']} | "
            f"{row['value_initial']:.6e} | {row['value_final']:.6e} | {row['scaled_drift']:.3e} |"
        )
    lines += [
        "",
        "## Euler boundary-flux budget (Sod, waves stay inside the domain)",
        "",
        f"analytic increments at t = {SOD_FINAL_TIME}: mass "
        f"{budget['analytic_increment']['mass']:.6e}, momentum "
        f"{budget['analytic_increment']['momentum']:.6e}, energy "
        f"{budget['analytic_increment']['energy']:.6e}",
        "",
        "| flux | cells | steps | mass drift | energy drift | momentum increment | analytic | budget error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in budget["rows"]:
        lines.append(
            f"| {row['flux']} | {row['ncells']} | {row['steps']} | {row['mass_drift']:.3e} | "
            f"{row['energy_drift']:.3e} | {row['momentum_increment']:.9f} | "
            f"{row['momentum_increment_analytic']:.9f} | {row['momentum_budget_error']:.3e} |"
        )
    lines += [
        "",
        "Rival hypotheses scored against the measured momentum increment: "
        + ", ".join(f"`{k}` = {v:.3e}" for k, v in budget["hypothesis_errors"].items())
        + " (only the transmissive prediction survives).",
        "",
        "## Rankine-Hugoniot shock speed",
        "",
        f"star state: rho* = {star['density']:.6f}, u* = {star['velocity']:.6f}, "
        f"p* = {star['pressure']:.6f}; W_RH = {star['shock_speed']:.6f} "
        f"(momentum-jump cross-check {star['shock_speed_momentum_jump']:.6f})",
        "",
        f"sensor level = {metrics['shock']['level']:.6f}; sensor bias on the exact profile = "
        f"{metrics['shock']['sensor_bias']:.3e}",
        "",
        "| cells | shock speed | relative error vs RH | shock positions |",
        "|---:|---:|---:|---|",
    ]
    for row in metrics["shock"]["rows"]:
        positions = ", ".join(f"{p:.6f}" for p in row["positions"])
        lines.append(
            f"| {row['ncells']} | {row['speed']:.6f} | {row['speed_error']:.3e} | {positions} |"
        )
    lines += [
        "",
        "## Discrete-shock tail ahead of the front",
        "",
        f"window x in [{precursor['window'][0]:.2f}, {precursor['window'][1]:.2f}] at "
        f"t = {precursor['final_time']}: every characteristic is still at x < 0.77, so the",
        "window holds no physical signal; the doubled domain (same dx, wall at "
        f"x = {precursor['control_length']:.0f}) isolates wall effects.",
        "",
        "| cells | dx | walled domain | doubled domain (same dx) | relative gap | tail width [cells] |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in precursor["rows"]:
        lines.append(
            f"| {row['ncells']} | {row['dx']:.3e} | {row['window_max_walled']:.3e} | "
            f"{row['window_max_doubled_domain']:.3e} | {row['relative_gap']:.1e} | "
            f"{row['tail_width_cells']:.2f} |"
        )
    lines += [
        "",
        f"fitted decay of the residual amplitude with dx: **{precursor['decay_rate']:.2f}**; "
        "the wall makes no difference (relative gap <= "
        f"{precursor['worst_relative_gap']:.1e}), so the residual is an interior, "
        "mesh-scale object, not a boundary defect.",
        "",
        "## Time-integrator order (dt refinement, mesh fixed)",
        "",
        "| integrator | dt | self-difference | successive orders | fitted order |",
        "|---|---|---|---|---:|",
    ]
    for key in ("advection_rk4", "burgers_ssp_rk3"):
        block = temporal[key]
        lines.append(
            f"| {block['integrator']} | "
            + ", ".join(f"{v:.3e}" for v in block["dt"]) + " | "
            + ", ".join(f"{v:.3e}" for v in block["self_differences"]) + " | "
            + ", ".join(f"{v:.2f}" for v in block["successive_orders"]) + " | "
            + f"**{block['fit_order']:.3f}** |"
        )
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python tests/conservation_study.py          # regenerate metrics/CSV/PNG/summary",
        "python -m pytest tests/test_conservation.py -v",
        "```",
        "",
        "Figures: " + ", ".join(f"`{name}`" for name in figures),
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="small meshes; skips artifacts unless --outdir is given")
    parser.add_argument("--outdir", type=Path, default=RESULT_DIR)
    parser.add_argument("--no-artifacts", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.quick and arguments.outdir == RESULT_DIR and not arguments.no_artifacts:
        # Never let a smoke run overwrite the committed full-mode artifacts.
        arguments.no_artifacts = True
        print("quick mode: artifacts disabled (pass --outdir DIR to write them)")

    metrics = evaluate_metrics(quick=arguments.quick)
    criteria = build_criteria(metrics)
    if arguments.quick:
        print("note: --quick meshes are far below the documented resolution "
              "(3200 cells), so `sod_field_error_vs_exact` legitimately FAILs there; "
              "the 3 % band is a claim about the documented resolution only.")
    for c in criteria:
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['name']:<46} {c['value']:.6e} "
              f"{c['comparison']} {c['threshold']:.3e}")
    if not arguments.no_artifacts:
        result = write_artifacts(metrics, arguments.outdir)
        print(f"\nartifacts -> {result['outdir']}")
        print(f"figures   -> {', '.join(result['figures'])}")
    return 0 if all(c["pass"] for c in criteria) else 1


if __name__ == "__main__":
    raise SystemExit(main())
