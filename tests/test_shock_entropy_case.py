"""The Shu--Osher (1988) shock/entropy-wave case as shipped by the library.

These tests pin the *contract* of :mod:`tensordg.cases.shock_entropy`: the case
definition is self-verifying through the Rankine--Hugoniot relations, the state
columns stay separated (density is column 0, not the whole conserved vector),
and a short DG run of the case is admissible.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tensordg.cases import shock_entropy as case


def test_case_constants_match_the_published_setup():
    assert case.DOMAIN == (-5.0, 5.0)
    assert case.DIAPHRAGM == -4.0
    assert case.FINAL_TIME == pytest.approx(1.8)
    assert case.GAMMA == pytest.approx(1.4)


def test_initial_state_is_the_documented_piecewise_data():
    x = torch.tensor([-4.5, -4.0 - 1e-9, -3.0, 0.5, 4.0], dtype=torch.float64)
    rho, u, p = case.initial_state(x)
    left_rho, left_u, left_p = case.LEFT_STATE
    # x < -4: uniform post-shock state
    assert float(rho[0]) == pytest.approx(left_rho, rel=1e-12)
    assert float(rho[1]) == pytest.approx(left_rho, rel=1e-12)
    assert float(u[0]) == pytest.approx(left_u, rel=1e-12)
    assert float(p[0]) == pytest.approx(left_p, rel=1e-12)
    # x >= -4: stationary sine in density, still gas otherwise
    xr = x[2:].numpy()
    assert np.allclose(rho[2:].numpy(), 1.0 + 0.2 * np.sin(5.0 * xr), atol=1e-14)
    assert np.allclose(u[2:].numpy(), 0.0, atol=1e-14)
    assert np.allclose(p[2:].numpy(), 1.0, atol=1e-14)


def test_conserved_state_round_trips_to_primitive_variables():
    x = torch.linspace(-5.0, 5.0, 21, dtype=torch.float64)
    rho, u, p = case.initial_state(x)
    conservative = case.conserved_state(x)
    assert conservative.shape == (21, 3)
    assert np.allclose(conservative[:, 0].numpy(), rho.numpy())
    assert np.allclose(conservative[:, 1].numpy(), (rho * u).numpy())
    energy = p / (case.GAMMA - 1.0) + 0.5 * rho * u ** 2
    assert np.allclose(conservative[:, 2].numpy(), energy.numpy(), rtol=1e-13)


def test_left_state_is_the_rankine_hugoniot_post_shock_state_of_a_mach3_shock():
    anchor = case.rankine_hugoniot_anchor()
    rel = anchor["hugoniot_relative_error"]
    assert rel["rho"] < 1e-6
    assert rel["u"] < 1e-6
    assert rel["p"] < 1e-5
    # shock speed from the mass jump == Mach number times pre-shock sound speed
    # (agreement is limited by the six significant figures of the published
    # left state, not by the algebra)
    assert anchor["shock_speed_analytic"] == pytest.approx(
        anchor["shock_speed_from_mach3"], rel=1e-6)
    # wave skeleton ordering: quiet zone head < contact < shock
    assert (anchor["quiet_zone_head_analytic"] < anchor["contact_position_analytic"]
            < anchor["shock_position_analytic"])


def test_exact_mass_budget_closure():
    budget = case.exact_mass_budget(samples=200_000)
    assert budget["mass_final_predicted"] == pytest.approx(
        budget["mass_initial"] + budget["mass_inflow"], rel=1e-12)
    assert budget["mass_inflow"] > 0.0


def test_solve_dg_case_keeps_the_state_columns_separate():
    """Regression: ``rho`` must be density only, never the (rho, rho u, rho E) state.

    The whole-state confusion is silent — the run still finishes and the numbers
    still look plausible — but ``rho E`` reaches ~39 on the post-shock plateau,
    so a "density" that peaks near 40 is the tell-tale sign.
    """
    out = case.solve_dg_case(ncells=40, degree=1, cfl=0.4, final_time=0.1)
    assert out["rho"].shape == (40,)
    assert out["rho_u"].shape == (40,)
    assert out["rho_e"].shape == (40,)
    assert np.isfinite(out["rho"]).all()
    assert out["rho"].min() > 0.5
    assert out["rho"].max() < 6.0  # a density, not a total energy density
    assert out["rho_e"].max() > 10.0  # the energy column really is separate