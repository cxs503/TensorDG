"""Discrete mathematical-entropy budget for the two-dimensional Euler DG solver.

Why this module exists
----------------------
``docs/entropy_verification.md`` established the discrete entropy budget for the
**one-dimensional** Euler DG solver (m2200-series work).  Its "limitations"
section states the open debt verbatim: *"仅一维、平直 Cartesian 网格"*.  The
two-dimensional solver :mod:`tensordg.euler_dg_2d` -- including its merged TVB
and WENO limiters -- therefore has no entropy-side evidence at all: nothing
tests whether the 2-D scheme produces entropy only where the second law allows
(shocks) and conserves it where the exact solution does (smooth flow).

This module closes that debt.  It supplies

* ``total_entropy_2d`` -- ``int int rho s dA`` evaluated on the **full** DG
  polynomial at Gauss points (not on cell averages, which would inject an
  ``O(h**2)`` averaging error into a nonlinear functional), with a quadrature
  self-check;
* ``smooth_advection_case`` -- a genuinely two-dimensional *exact* solution with
  non-constant entropy, ``rho = 1 + A sin(2 pi (x + y - 2t))``, ``u = v = 1``,
  ``p = 1``.  This is an exact solution of the compressible Euler equations
  (mass, momentum and energy all reduce to advection of ``rho``), and
  ``int int rho s dA = -gamma int int rho ln rho dA`` is an exact invariant.
  The measured drift must therefore converge to zero, and must be
  *non-negative* (numerical dissipation may produce entropy, never destroy it);
* ``extruded_sod_case`` -- the y-invariant Sod problem, whose entropy production
  has both an independent numerical reference (the 1-D DG solver) and an
  analytic Rankine-Hugoniot reference (``analytic_right_shock_budget``).

The distinction matters: the analytic RH budget is the *physically* correct
target but a smeared shock cannot reproduce it at engineering resolutions (the
1-D ledger reports 11.6 % error for Sod on its finest mesh and explicitly
records a FAIL against the 3 % gate).  The 1-D solver is therefore used here as
a *cross-implementation* reference for the 2-D entropy machinery: the same exact
physics integrated two independent ways.  Agreement there isolates 2-D-specific
defects from the known shock-smearing limitation.

Sign convention
---------------
``s = ln(p / rho**gamma)`` (the negated Harten entropy, so that ``rho s``
*increases* across a shock), matching :mod:`benchmarks._entropy_tools`.
``drift = S(t) - S(0)``; ``drift >= 0`` is the discrete entropy inequality.

Run
---
    python -m benchmarks.entropy_budget_2d                # full study
    python -m benchmarks.entropy_budget_2d --quick        # reduced mesh ladder

Outputs ``validation/entropy_2d/summary.json`` (machine readable, every number
in ``docs/reports/tensordg-entropy-2d.md`` is injected from it) and
``docs/figures/entropy_budget_2d.png``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from benchmarks._entropy_tools import (
    analytic_right_shock_budget,
    entropy_density,
    legendre_modes,
)
from tensordg.euler_dg_2d import solve_euler_dg_2d

GAMMA = 1.4
SOD_LEFT = (1.0, 0.0, 1.0)
SOD_RIGHT = (0.125, 0.0, 0.1)


# --------------------------------------------------------------------------- #
# discrete entropy of a 2-D modal DG state
# --------------------------------------------------------------------------- #
def total_entropy_2d(coeff, dx: float, dy: float, degree: int,
                     quadrature_order: int | None = None) -> float:
    """``int int rho s dx dy`` of a modal DG state.

    Parameters
    ----------
    coeff : array_like
        Modal coefficients with shape ``(nx, ny, degree + 1, degree + 1, 4)``
        and the coefficient convention of :mod:`tensordg.euler_dg_2d`.
    dx, dy : float
        Cell sizes (the quadrature is mapped onto each cell, so the result is a
        physical integral, not a sum of nodal values).
    quadrature_order : int, optional
        Gauss--Legendre order per direction; defaults to ``max(degree + 3, 6)``.
    """
    c = np.asarray(coeff.detach().cpu() if hasattr(coeff, "detach") else coeff,
                   dtype=float)
    if c.ndim != 5 or c.shape[-1] != 4:
        raise ValueError("coeff must have shape (nx, ny, degree+1, degree+1, 4)")
    order = quadrature_order if quadrature_order is not None else max(degree + 3, 6)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    basis = legendre_modes(nodes, degree)              # (q, degree + 1)
    # conservative variables on the (nx, ny, q, q) quadrature grid
    state = np.einsum("xyklv,ak,bl->xyabv", c, basis, basis)
    density = state[..., 0]
    momentum_x, momentum_y, energy = state[..., 1], state[..., 2], state[..., 3]
    pressure = (GAMMA - 1.0) * (
        energy - 0.5 * (momentum_x**2 + momentum_y**2) / density
    )
    weight = np.outer(weights, weights) * (0.5 * dx) * (0.5 * dy)
    return float(np.einsum("xyab,ab->", entropy_density(density, pressure), weight))


def entropy_density_field_2d(coeff, dx: float, dy: float, degree: int,
                             quadrature_order: int | None = None):
    """``rho s`` on the quadrature grid, for contours and pointwise diagnostics."""
    c = np.asarray(coeff.detach().cpu() if hasattr(coeff, "detach") else coeff,
                   dtype=float)
    order = quadrature_order if quadrature_order is not None else max(degree + 3, 6)
    nodes, _ = np.polynomial.legendre.leggauss(order)
    basis = legendre_modes(nodes, degree)
    state = np.einsum("xyklv,ak,bl->xyabv", c, basis, basis)
    density = state[..., 0]
    momentum_x, momentum_y, energy = state[..., 1], state[..., 2], state[..., 3]
    pressure = (GAMMA - 1.0) * (
        energy - 0.5 * (momentum_x**2 + momentum_y**2) / density
    )
    return entropy_density(density, pressure)


def total_entropy_1d_reference(coeff_2d, dx: float, degree: int, ny: int) -> float:
    """Entropy of a y-invariant 2-D state via the *1-D* implementation.

    Cross-implementation check of :func:`total_entropy_2d`: a y-invariant
    ``(nx, ny, ...)`` state carries no y-modes (``l > 0`` vanish), so every y
    slice is the same 1-D state and ``int int rho s dA`` equals the 1-D integral
    ``int rho s dx``.  Any disagreement between this helper and
    :func:`total_entropy_2d` is an integration bug, not a physics difference.
    """
    c = np.asarray(coeff_2d.detach().cpu() if hasattr(coeff_2d, "detach") else coeff_2d,
                   dtype=float)
    if np.abs(c[:, :, :, 1:, :]).max(initial=0.0) > 1e-12:
        raise ValueError("state is not y-invariant; use total_entropy_2d instead")
    order = max(degree + 3, 6)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    basis = legendre_modes(nodes, degree)
    state = np.einsum("xkv,ak->xav", c[:, 0, :, 0, :], basis)
    density = state[..., 0]
    momentum = state[..., 1]
    energy = state[..., 3]
    pressure = (GAMMA - 1.0) * (energy - 0.5 * momentum**2 / density)
    weight = weights * (0.5 * dx)
    # every y cell carries the same slice: dy = 1 / ny, summed over ny cells
    return float(np.einsum("xa,a->", entropy_density(density, pressure), weight) * ny * (1.0 / ny))


# --------------------------------------------------------------------------- #
# case 1: genuinely 2-D smooth flow with an exact entropy invariant
# --------------------------------------------------------------------------- #
def smooth_initial(amplitude=0.2, velocity=1.0):
    def initial(x, y):
        full = x + y                                # broadcasts to (nx, ny, q, q)
        density = 1.0 + amplitude * torch.sin(2.0 * np.pi * full)
        return (density,
                torch.full_like(full, velocity),
                torch.full_like(full, velocity),
                torch.ones_like(full))
    return initial


def smooth_exact_density(x, y, final_time, amplitude=0.2, velocity=1.0):
    """Exact density of the advected wave, ``rho0(x - u t, y - v t)``."""
    return 1.0 + amplitude * np.sin(
        2.0 * np.pi * (x + y - 2.0 * velocity * final_time)
    )


def smooth_advection_case(ncells, degree=1, final_time=0.05, cfl=0.4,
                          limiter=None, tvb_constant=1.0, amplitude=0.2,
                          velocity=1.0):
    """Entropy drift and density error of the exact 2-D smooth advection wave."""
    initial = smooth_initial(amplitude, velocity)
    common = dict(ncells=ncells, degree=degree, cfl=cfl, limiter=limiter,
                  tvb_constant=tvb_constant)
    _, _, coeff_0, _ = solve_euler_dg_2d(initial, final_time=0.0, **common)
    x, y, coeff_t, meta = solve_euler_dg_2d(initial, final_time=final_time, **common)
    nx, ny = meta["ncells"]
    dx, dy = 1.0 / nx, 1.0 / ny
    entropy_0 = total_entropy_2d(coeff_0, dx, dy, degree)
    entropy_t = total_entropy_2d(coeff_t, dx, dy, degree)
    exact = smooth_exact_density(x.numpy()[:, None], y.numpy()[None, :],
                                 final_time, amplitude, velocity)
    numerical = coeff_t[..., 0, 0, 0].numpy()
    return {
        "ncells": int(ncells),
        "degree": int(degree),
        "limiter": limiter if limiter else "none",
        "tvb_constant": float(tvb_constant),
        "final_time": float(final_time),
        "steps": int(meta["steps"]),
        "entropy_initial": entropy_0,
        "entropy_final": entropy_t,
        "entropy_drift": entropy_t - entropy_0,
        "density_linf_error": float(np.max(np.abs(numerical - exact))),
        "density_linf_exact_amplitude": float(np.max(np.abs(exact - 1.0))),
    }


def limiter_activity(ncells, degree=1, limiter="tvb", tvb_constant=1.0,
                     amplitude=0.2, velocity=1.0):
    """Relative change the limiter makes to the *smooth* projected initial state.

    A shock limiter must be (near) inactive on smooth data; the size of this
    number is the direct measure of how much spurious limiting the smooth flow
    suffers, independent of any time integration.
    """
    from tensordg.euler_dg_2d import apply_tvb_limiter_2d, apply_weno_limiter_2d

    initial = smooth_initial(amplitude, velocity)
    _, _, coeff, meta = solve_euler_dg_2d(initial, ncells=ncells, degree=degree,
                                          final_time=0.0, limiter=None)
    nx, ny = meta["ncells"]
    dx, dy = 1.0 / nx, 1.0 / ny
    if limiter == "tvb":
        limited = apply_tvb_limiter_2d(coeff, dx, dy, tvb_constant, (True, True))
    elif limiter == "weno":
        limited = apply_weno_limiter_2d(coeff, dx, dy, tvb_constant, (True, True))
    else:
        raise ValueError("limiter must be 'tvb' or 'weno'")
    delta = (limited - coeff).abs()
    scale = coeff.abs().max()
    return {
        "ncells": int(ncells),
        "limiter": limiter,
        "tvb_constant": float(tvb_constant),
        "max_relative_change": float(delta.max() / scale) if float(scale) else 0.0,
        "mean_absolute_change": float(delta.mean()),
    }


# --------------------------------------------------------------------------- #
# case 2: y-invariant Sod -- 2-D vs 1-D vs analytic entropy production
# --------------------------------------------------------------------------- #
def sod_initial(left=SOD_LEFT, right=SOD_RIGHT):
    def initial(x, y):
        full = x + y
        density = torch.where(x < 0.5, torch.full_like(full, left[0]),
                              torch.full_like(full, right[0]))
        pressure = torch.where(x < 0.5, torch.full_like(full, left[2]),
                               torch.full_like(full, right[2]))
        velocity = torch.where(x < 0.5, torch.full_like(full, left[1]),
                               torch.full_like(full, right[1]))
        return density, velocity, torch.zeros_like(full), pressure
    return initial


def extruded_sod_case(ncells, degree=1, final_time=0.1, cfl=0.3, limiter=None,
                      tvb_constant=1.0, y_cells=None):
    """2-D entropy production of the y-invariant Sod problem."""
    ny = ncells if y_cells is None else y_cells
    initial = sod_initial()
    common = dict(ncells=(ncells, ny), degree=degree, cfl=cfl, limiter=limiter,
                  tvb_constant=tvb_constant, periodic=(False, True))
    _, _, coeff_0, meta0 = solve_euler_dg_2d(initial, final_time=0.0, **common)
    _, _, coeff_t, meta = solve_euler_dg_2d(initial, final_time=final_time, **common)
    dx, dy = 1.0 / ncells, 1.0 / ny
    drift = (total_entropy_2d(coeff_t, dx, dy, degree)
             - total_entropy_2d(coeff_0, dx, dy, degree))
    return {
        "ncells": int(ncells),
        "y_cells": int(ny),
        "degree": int(degree),
        "limiter": limiter if limiter else "none",
        "final_time": float(final_time),
        "steps": int(meta["steps"]),
        "entropy_production": drift,
        "state_2d": coeff_t,
    }


def one_dimensional_sod_production(ncells, degree=1, final_time=0.1, cfl=0.3,
                                   limiter=None):
    """Entropy production of the *1-D* DG solver on the same Sod problem."""
    from benchmarks._entropy_tools import total_entropy_from_coefficients
    from tensordg.euler_dg_1d import solve_periodic_euler_dg_1d

    def initial(node_x):
        density = torch.where(node_x < 0.5, torch.ones_like(node_x),
                              torch.full_like(node_x, SOD_RIGHT[0]))
        pressure = torch.where(node_x < 0.5, torch.ones_like(node_x),
                               torch.full_like(node_x, SOD_RIGHT[2]))
        return density, torch.zeros_like(node_x), pressure

    dx = 1.0 / ncells
    common = dict(ncells=ncells, degree=degree, cfl=cfl, limiter=limiter,
                  boundary="transmissive")
    _, coeff_0, _ = solve_periodic_euler_dg_1d(initial, final_time=0.0, **common)
    _, coeff_t, meta = solve_periodic_euler_dg_1d(initial, final_time=final_time, **common)
    return {
        "ncells": int(ncells),
        "degree": int(degree),
        "limiter": limiter if limiter else "none",
        "steps": int(meta["steps"]),
        "entropy_production": (total_entropy_from_coefficients(coeff_t, dx, degree)
                               - total_entropy_from_coefficients(coeff_0, dx, degree)),
    }


# --------------------------------------------------------------------------- #
# study driver
# --------------------------------------------------------------------------- #
def _relative_error(reference, candidate):
    reference = float(reference)
    return abs(float(candidate) - reference) / max(abs(reference), 1e-300)


def observed_orders(meshes, values):
    """Pairwise convergence orders ``log(r)/log(h1/h2)`` of ``values``."""
    orders = []
    for index in range(1, len(meshes)):
        h1, h2 = 1.0 / meshes[index - 1], 1.0 / meshes[index]
        r = abs(values[index - 1]) / max(abs(values[index]), 1e-300)
        orders.append(float(np.log(r) / np.log(h1 / h2)))
    return orders


def quadrature_self_check(ncells=16, degree=1):
    """Diagnostics that must pass *before* the solver is judged.

    (a) a constant state has an exact entropy integral, so the quadrature and
        the Jacobian factors are checked against a closed form;
    (b) the entropy integral of the smooth initial state is independent of the
        quadrature order once the polynomial integrand is resolved
        (``rho s`` of a degree-``p`` polynomial is resolvable at ``2p``);
    (c) on y-invariant data :func:`total_entropy_2d` must reproduce the 1-D
        implementation, which is an independent code path.
    """
    def uniform(x, y):
        full = x + y
        return (torch.full_like(full, 2.0), torch.zeros_like(full),
                torch.zeros_like(full), torch.ones_like(full))

    _, _, coeff, meta = solve_euler_dg_2d(uniform, ncells=ncells, degree=degree,
                                          final_time=0.0)
    nx, ny = meta["ncells"]
    dx, dy = 1.0 / nx, 1.0 / ny
    exact = 2.0 * (np.log(1.0) - GAMMA * np.log(2.0))
    uniform_error = _relative_error(exact, total_entropy_2d(coeff, dx, dy, degree))

    initial = smooth_initial()
    _, _, smooth_coeff, meta = solve_euler_dg_2d(initial, ncells=ncells,
                                                 degree=degree, final_time=0.0)
    orders = [4, 6, 8, 10]
    values = [total_entropy_2d(smooth_coeff, dx, dy, degree, quadrature_order=order)
              for order in orders]
    spread = (max(values) - min(values)) / abs(values[-1])

    _, _, sod_coeff, meta = solve_euler_dg_2d(sod_initial(), ncells=ncells,
                                              degree=degree, final_time=0.0,
                                              periodic=(False, True))
    cross = total_entropy_2d(sod_coeff, dx, dy, degree)
    reference = total_entropy_1d_reference(sod_coeff, dx, degree, ny)
    cross_error = _relative_error(reference, cross)
    return {
        "uniform_state_relative_error": uniform_error,
        "quadrature_orders": orders,
        "quadrature_values": values,
        "quadrature_relative_spread": float(spread),
        "one_dimensional_cross_check_relative_error": cross_error,
        "pass": bool(uniform_error < 1e-12 and spread < 1e-10 and cross_error < 1e-10),
    }


def run_study(quick=False, out_path="validation/entropy_2d/summary.json",
              figure_path="docs/figures/entropy_budget_2d.png"):
    torch.set_num_threads(1)
    meshes = (16, 32) if quick else (16, 32, 64)
    sod_meshes = (16,) if quick else (16, 32)
    started = time.time()
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gamma": GAMMA,
        "cases": {},
    }

    # --- quadrature self-check (diagnostic before judging the solver) --------
    summary["self_checks"] = quadrature_self_check(meshes[0])

    # --- case 1: smooth 2-D advection, entropy invariant --------------------
    smooth = []
    for limiter in (None, "tvb", "weno"):
        for ncells in meshes:
            entry = smooth_advection_case(ncells, final_time=0.05, limiter=limiter)
            entry.pop("state", None)
            smooth.append(entry)
            print(f"[smooth] ncells={ncells:3d} limiter={str(limiter):5s} "
                  f"drift={entry['entropy_drift']:+.4e} "
                  f"rho_linf={entry['density_linf_error']:.3e}", flush=True)
    summary["cases"]["smooth_advection"] = smooth
    activity = [limiter_activity(meshes[-1], limiter=lim) for lim in ("tvb", "weno")]
    summary["cases"]["limiter_activity_on_smooth_data"] = activity

    # --- case 2: extruded Sod, 2-D vs 1-D vs analytic ----------------------
    budget = analytic_right_shock_budget(np.array(SOD_LEFT), np.array(SOD_RIGHT), 0.1)
    sod = []
    for limiter in (None, "tvb", "weno"):
        for ncells in sod_meshes:
            entry = extruded_sod_case(ncells, final_time=0.1, limiter=limiter)
            reference_1d = one_dimensional_sod_production(ncells, final_time=0.1,
                                                          limiter=limiter)
            proj = entry.pop("state_2d")
            del proj
            entry["entropy_1d_solver"] = reference_1d["entropy_production"]
            entry["relative_error_vs_1d"] = _relative_error(
                reference_1d["entropy_production"], entry["entropy_production"])
            entry["relative_error_vs_analytic"] = _relative_error(
                budget["production_total"], entry["entropy_production"])
            sod.append(entry)
            print(f"[sod]    ncells={ncells:3d} limiter={str(limiter):5s} "
                  f"dS={entry['entropy_production']:+.4e} "
                  f"vs1D={100 * entry['relative_error_vs_1d']:.3f}% "
                  f"vs_analytic={100 * entry['relative_error_vs_analytic']:.2f}%",
                  flush=True)
    summary["cases"]["extruded_sod"] = sod
    summary["analytic_budget"] = {
        "production_total": budget["production_total"],
        "shock_speed": budget["sigma"],
        "shock_position": budget["shock_position"],
        "final_time": budget["final_time"],
    }

    # --- verdicts ----------------------------------------------------------
    unlimited = [e for e in smooth if e["limiter"] == "none"]
    orders = observed_orders([e["ncells"] for e in unlimited],
                             [e["entropy_drift"] for e in unlimited])
    tvb = [e for e in smooth if e["limiter"] == "tvb"]
    weno = [e for e in smooth if e["limiter"] == "weno"]
    summary["verdicts"] = {
        "C1_smooth_entropy_conserved": {
            "finest_drift": unlimited[-1]["entropy_drift"],
            "finest_drift_relative_to_initial_entropy": abs(
                unlimited[-1]["entropy_drift"] / unlimited[-1]["entropy_initial"]),
            "observed_orders": orders,
            "pass": bool(unlimited[-1]["entropy_drift"] >= 0.0
                         and abs(unlimited[-1]["entropy_drift"]
                                 / unlimited[-1]["entropy_initial"]) < 1e-3),
        },
        "C2_tvb_dissipative_side": {
            "drifts": [e["entropy_drift"] for e in tvb],
            "pass": bool(all(e["entropy_drift"] >= 0.0 for e in tvb)),
            "amplification_vs_unlimited": float(
                tvb[-1]["entropy_drift"] / max(unlimited[-1]["entropy_drift"], 1e-300)),
        },
        "C3_weno_dissipative_side": {
            "drifts": [e["entropy_drift"] for e in weno],
            "pass": bool(all(e["entropy_drift"] >= 0.0 for e in weno)),
            "violating_meshes": [e["ncells"] for e in weno if e["entropy_drift"] < 0.0],
        },
        "C4_two_dimensional_path_matches_the_1d_solver": {
            "relative_errors": [e["relative_error_vs_1d"] for e in sod],
            "tolerance": 0.02,
            "pass": bool(all(e["relative_error_vs_1d"] < 0.02 for e in sod)),
        },
        "C5_three_percent_against_the_analytic_rh_budget": {
            "relative_errors": [e["relative_error_vs_analytic"] for e in sod],
            "pass": bool(all(e["relative_error_vs_analytic"] < 0.03 for e in sod)),
        },
    }
    summary["runtime_seconds"] = round(time.time() - started, 2)
    summary["quick_mode"] = bool(quick)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"wrote {out_path} in {summary['runtime_seconds']} s", flush=True)

    try:
        from benchmarks.plot_entropy_2d import make_figure
        make_figure(out_path, figure_path)
    except Exception as error:                     # plotting must not void the study
        print(f"figure skipped: {error}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="reduced mesh ladder")
    parser.add_argument("--out", default="validation/entropy_2d/summary.json")
    parser.add_argument("--figure", default="docs/figures/entropy_budget_2d.png")
    args = parser.parse_args(argv)
    summary = run_study(quick=args.quick, out_path=args.out, figure_path=args.figure)
    for name, verdict in summary["verdicts"].items():
        print(f"{name}: {'PASS' if verdict['pass'] else 'FAIL'}")
    return summary


if __name__ == "__main__":
    main()