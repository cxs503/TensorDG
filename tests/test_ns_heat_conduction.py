"""Regression tests for the pure heat-conduction path of ``tensordg.ns_dg_2d``.

The wrapped helpers ``solve_heat_conduction_dg_2d``, ``heat_conduction_exact``
and ``gas_constants`` shipped without any test.  These tests pin down

* what *is* correct: the analytic steady state satisfies the compressible
  equations, and on it the DG mass/momentum rows are at machine zero;
* what is *still broken*: the isothermal-wall energy closure is inconsistent, so
  the exact steady state is not a discrete steady state -- the wall-cell energy
  residual stagnates at an O(1) floor instead of decaying under refinement.

Ratchet status (re-adjudicated 2026-11-04 against main's m3139 wall repair,
which restored the one-sided DG wall derivative and added the positivity
guards): the shipped wrapper now integrates its documented defaults
(``final_time=40``) and preserves the exact steady state at ``final_time=0.05``.
Those two pins therefore XPASSed, which is a red ``strict`` xfail -- they are
flipped to positive assertions below.  The two residual-consistency pins still
fail and stay ``xfail(strict=True)``.
See ``docs/reports/tensordg-ns-heat-conduction-verification.md``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import tensordg.ns_dg_2d as ns

MU = 0.1
PRANDTL = 0.71
KAPPA = MU * ns.CP / PRANDTL
BOTTOM_TEMPERATURE, TOP_TEMPERATURE, PRESSURE = 1.0, 0.5, 1.0


def _steady_initial(x, y):
    ref = x * y + 0.0 * x + 0.0 * y
    s = (y + 0.0 * x) * torch.ones_like(ref)
    temperature = BOTTOM_TEMPERATURE + (TOP_TEMPERATURE - BOTTOM_TEMPERATURE) * s
    temperature = temperature + 0.0 * ref
    return (PRESSURE / temperature, torch.zeros_like(ref), torch.zeros_like(ref),
            PRESSURE * torch.ones_like(ref))


def _uniform_initial(x, y):
    ref = x * y + 0.0 * x + 0.0 * y
    return (torch.ones_like(ref), torch.zeros_like(ref), torch.zeros_like(ref),
            PRESSURE * torch.ones_like(ref))


def _wall_residual(ny, degree=1, wall_gradient="taylor"):
    """Energy residual of the exact steady conduction state, split by row."""
    nx = 2
    dx, dy = 1.0 / nx, 1.0 / ny
    nodes, weights, basis, derivative, signs = ns.legendre_operators(
        degree, degree + 2, torch.device("cpu"), torch.float64)
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    coeff = ns._initial_coefficients(_steady_initial, x, y, dx, dy, nodes, weights,
                                     basis, torch.device("cpu"), torch.float64)
    boundary = {
        "xmin": None, "xmax": None,
        "ymin": {"velocity": (0.0, 0.0), "temperature": BOTTOM_TEMPERATURE},
        "ymax": {"velocity": (0.0, 0.0), "temperature": TOP_TEMPERATURE},
    }
    rhs = ns._rhs_ns(coeff, basis, derivative, weights, signs, dx, dy,
                     (True, False), boundary, MU, KAPPA,
                     2.0 * (degree + 1.0) ** 2, wall_gradient)
    return rhs


def test_analytic_heat_conduction_state_solves_the_equations():
    y = torch.linspace(0.0, 1.0, 129, dtype=torch.float64)
    state = ns.heat_conduction_exact(y, 0.0, 1.0, BOTTOM_TEMPERATURE,
                                     TOP_TEMPERATURE, PRESSURE)
    temperature = state["temperature"]
    assert abs(float(temperature[0]) - BOTTOM_TEMPERATURE) < 1e-14
    assert abs(float(temperature[-1]) - TOP_TEMPERATURE) < 1e-14
    spacing = float(y[1] - y[0])
    laplace = (temperature[2:] - 2.0 * temperature[1:-1] + temperature[:-2]) / spacing ** 2
    assert float(laplace.abs().max()) < 1e-10, "conducting profile must be harmonic"
    # ideal gas at uniform pressure: rho = p / (R T); R = 1 in this package
    assert torch.allclose(state["density"], PRESSURE / temperature, rtol=1e-14)
    assert float(state["velocity_x"].abs().max()) == 0.0
    assert float(state["velocity_y"].abs().max()) == 0.0


def test_gas_constants_are_thermodynamically_consistent():
    constants = ns.gas_constants(PRANDTL)
    assert abs(constants["cp"] - constants["cv"] - ns.GAS_CONSTANT) < 1e-14
    assert abs(constants["gamma"] - constants["cp"] / constants["cv"]) < 1e-14
    assert constants["prandtl"] == PRANDTL
    assert ns.CP / ns.CV == pytest.approx(ns.GAMMA, rel=0, abs=1e-14)


def test_mass_and_momentum_rows_vanish_on_the_exact_steady_state():
    """Positive control: the parts of the operator that are right."""
    rhs = _wall_residual(32)
    assert float(rhs[..., 0].abs().max()) < 1e-12
    assert float(rhs[..., 1:3].abs().max()) < 1e-12


def test_interior_energy_residual_is_first_order_small():
    coarse = float(_wall_residual(16)[..., 3][:, 1:-1].abs().max())
    fine = float(_wall_residual(64)[..., 3][:, 1:-1].abs().max())
    assert fine < coarse / 2.0, "interior truncation error must shrink with h"


@pytest.mark.xfail(strict=True, reason=(
    "known defect: isothermal-wall energy closure is inconsistent, the wall "
    "residual of the exact steady state does not decay under refinement "
    "(see docs/reports/tensordg-ns-heat-conduction-verification.md)"))
def test_wall_energy_residual_decays_under_refinement():
    coarse = float(_wall_residual(16)[..., 3][:, -1].abs().max())
    fine = float(_wall_residual(128)[..., 3][:, -1].abs().max())
    assert fine < 0.25 * coarse, (
        f"cold-wall energy residual {coarse:.4f} -> {fine:.4f}: inconsistent closure")


@pytest.mark.xfail(strict=True, reason=(
    "known defect: the same wall residual appears for both wall_gradient "
    "reconstructions, so the defect is in the closure and not in the "
    "reconstruction choice"))
def test_wall_energy_residual_is_small_for_both_reconstructions():
    for gradient in ("taylor", "face"):
        value = float(_wall_residual(32, wall_gradient=gradient)[..., 3][:, -1].abs().max())
        assert value < 1e-6, f"{gradient}: wall energy residual {value:.4f}"


def test_shipped_wrapper_runs_at_its_documented_defaults():
    """Ratchet flip (2026-11-04): used to raise FloatingPointError at defaults.

    The m3139 wall repair (one-sided DG wall derivative as the default, plus the
    positivity filter and the pressure guard) lets the documented defaults
    (``ncells=(2,16)``, ``degree=1``, ``final_time=40``) complete.  The pin now
    asserts the run stays physical instead of asserting the crash.
    """
    x, y, coeff, metadata = ns.solve_heat_conduction_dg_2d(dtype=torch.float64)
    assert torch.isfinite(coeff).all()
    fields = ns.cell_fields(coeff)
    assert float(fields["pressure"].min()) > 0.0, "cell-average pressure left the physical cone"
    assert float(fields["density"].min()) > 0.0
    # No accuracy claim here: the wall closure is still inconsistent (the two
    # xfails above), so this pin only rules out the abort it used to assert.


def test_exact_steady_state_is_approximately_preserved():
    """Ratchet flip (2026-11-04): the collapse at ``final_time=0.05`` is gone.

    Kept as a positive pin because it separates the two original symptoms: the
    transient no longer collapses on the exact steady state, while the wall
    residual of that same state still does not decay (the two xfails below).
    """
    x, y, coeff, metadata = ns.solve_navier_stokes_dg_2d(
        _steady_initial, ncells=(2, 16), degree=1, final_time=0.05, cfl=0.3,
        mu=MU, prandtl=PRANDTL, periodic=(True, False),
        boundary={"ymin": {"velocity": (0.0, 0.0), "temperature": BOTTOM_TEMPERATURE},
                  "ymax": {"velocity": (0.0, 0.0), "temperature": TOP_TEMPERATURE}})
    fields = ns.cell_fields(coeff)
    pressure = fields["pressure"]
    assert float((pressure - PRESSURE).abs().max()) < 1e-3 * PRESSURE


def test_uniform_initial_state_loses_pressure_within_an_acoustic_time():
    """The collapse mechanism, measured while the run is still finite."""
    x, y, coeff, metadata = ns.solve_navier_stokes_dg_2d(
        _uniform_initial, ncells=(2, 16), degree=1, final_time=0.01, cfl=0.3,
        mu=MU, prandtl=PRANDTL, periodic=(True, False),
        boundary={"ymin": {"velocity": (0.0, 0.0), "temperature": BOTTOM_TEMPERATURE},
                  "ymax": {"velocity": (0.0, 0.0), "temperature": TOP_TEMPERATURE}})
    fields = ns.cell_fields(coeff)
    pressure_min = float(fields["pressure"].min())
    assert pressure_min < 0.9 * PRESSURE, (
        "the isothermal wall must already be draining the wall cell at t=0.01")