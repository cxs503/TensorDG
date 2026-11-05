"""Conservation-law and time-integrator verification criteria (S3 round 2).

Every test below is a *falsifiable* statement judged against something outside
the solver: an analytic invariant, an analytic boundary-flux budget, the exact
Riemann solution (Rankine-Hugoniot jump speed) or the theoretical order of a
Runge-Kutta method.  The measurement code lives in
``tests/_conservation_tools.py``; the full study that produces the committed
artifacts lives in ``tests/conservation_study.py``.

Estimator self-tests come first: a diagnostic that cannot recover a synthetic
answer cannot be trusted to judge a solver.
"""
import math

import numpy as np
import pytest

from tests._conservation_tools import (
    boundary_increment,
    cell_averages_1d,
    cell_averages_2d,
    conserved_totals,
    fit_order,
    integral_1d,
    linear_slope,
    observed_orders,
    physical_flux,
    rankine_hugoniot_speed,
    relative_difference,
    scaled_drift,
    shock_position,
)
from tests.conservation_study import (
    CRITERIA,
    SOD_FINAL_TIME,
    SOD_LEFT,
    SOD_RIGHT,
    sod_initial_totals,
    sod_star_state,
    study_shock_precursor,
    study_euler_budget,
    study_periodic_drift,
    study_shock_speed,
    study_temporal_order,
)


# ---------------------------------------------------------------------------
# estimator self-tests (no solver involved)
# ---------------------------------------------------------------------------
def test_cell_average_extraction_uses_the_p0_mode_only():
    coefficients = np.array([[2.0, 0.5, -0.25], [2.0, -0.5, 0.25]])
    averages = cell_averages_1d(coefficients)
    assert np.allclose(averages, [2.0, 2.0])
    # Higher Legendre modes have zero mean, so the integral is h*sum(u_bar).
    assert integral_1d(averages) == pytest.approx(2.0)
    tensor = np.zeros((2, 3, 2, 2))
    tensor[:, :, 0, 0] = 7.0
    tensor[:, :, 1, 0] = 1e3  # must be ignored
    assert np.allclose(cell_averages_2d(tensor), 7.0)


def test_scaled_drift_needs_an_external_scale_for_zero_mean_data():
    # A sine wave has (numerically) zero integral: normalising by max|M| would
    # turn round-off into "100 % drift", so an explicit scale is mandatory.
    assert scaled_drift([1e-17, 2e-17], scale=1.0) == pytest.approx(1e-17, abs=1e-20)
    assert scaled_drift([0.0, 3e-3], scale=1.0) == pytest.approx(3e-3)
    with pytest.raises(ValueError):
        scaled_drift([1.0])


def test_order_estimators_recover_synthetic_rates():
    h = np.array([0.1, 0.05, 0.025])
    cubic = 4.0 * h ** 3
    assert fit_order(h, cubic) == pytest.approx(3.0, abs=1e-8)
    assert observed_orders(h, cubic) == pytest.approx([3.0, 3.0], abs=1e-8)
    # A converging-but-saturated sequence must not fake a high order.
    saturated = np.array([1e-2, 1e-2, 1e-2])
    assert observed_orders(h, saturated)[0] == pytest.approx(0.0, abs=1e-12)


def test_relative_difference_is_scale_free_and_rejects_different_meshes():
    a = np.array([1.0, -1.0, 1.0, -1.0])
    b = a + 1e-3 * np.array([1.0, 1.0, -1.0, -1.0])
    assert relative_difference(a, b) == pytest.approx(1e-3)
    with pytest.raises(ValueError):
        relative_difference(a, np.array([1.0, 2.0]))


def test_boundary_increment_is_the_analytic_pressure_imbalance():
    increment = boundary_increment(SOD_LEFT, SOD_RIGHT, SOD_FINAL_TIME)
    # Both end states are at rest, so only the pressure imbalance drives the
    # momentum integral: p_L - p_R = 1.0 - 0.1.
    assert increment["momentum"] == pytest.approx((1.0 - 0.1) * SOD_FINAL_TIME)
    assert increment["mass"] == pytest.approx(0.0, abs=1e-15)
    assert increment["energy"] == pytest.approx(0.0, abs=1e-15)
    assert physical_flux((1.0, 0.0, 1.0)) == pytest.approx((0.0, 1.0, 0.0))


def test_rankine_hugoniot_speed_agrees_across_mass_and_momentum_jumps():
    star = sod_star_state()
    assert star["shock_speed"] > 0.0
    assert star["jump_consistency"] < CRITERIA["rh_jump_consistency_max"]
    # The star state must also bracket the Sod wave pattern: p_L > p* > p_R.
    assert SOD_LEFT[2] > star["pressure"] > SOD_RIGHT[2]
    assert SOD_RIGHT[0] < star["density"] < SOD_LEFT[0]
    with pytest.raises(ValueError):
        rankine_hugoniot_speed((1.0, 0.0, 1.0), (1.0, 0.0, 0.1))


def test_shock_sensor_recovers_the_analytic_shock_path():
    """A level-crossing sensor applied to the exact solution must be unbiased."""
    from benchmarks.riemann_exact import GAMMA, sample

    star = sod_star_state()
    grid = np.linspace(0.0, 1.0, 20001)
    times = (0.10, 0.14, 0.18, 0.22)
    positions = [
        shock_position(
            grid, sample(grid, t, SOD_LEFT, SOD_RIGHT, 0.5, GAMMA)["density"],
            star["sensor_level"],
        )
        for t in times
    ]
    speed, _ = linear_slope(times, positions)
    assert speed == pytest.approx(star["shock_speed"], rel=1e-3)
    for position, time in zip(positions, times):
        assert position == pytest.approx(0.5 + star["shock_speed"] * time, abs=1e-3)


# ---------------------------------------------------------------------------
# discrete invariants
# ---------------------------------------------------------------------------
def test_periodic_dg_conserves_mass_to_round_off_for_every_degree():
    result = study_periodic_drift(
        cells_1d=(16, 32), degrees=(1, 2, 3), cells_2d=(8,), burgers_cells=(16,),
    )
    one_d = [r for r in result["rows"] if r["case"] == "linear_advection_1d"]
    assert len(one_d) == 6
    for row in one_d:
        assert row["scaled_drift"] <= CRITERIA["periodic_drift_max"], row
        assert row["absolute_drift"] <= 1e-14, row
    assert result["max_scaled_drift"] <= CRITERIA["periodic_drift_max"]


def test_periodic_two_dimensional_advection_conserves_mass():
    result = study_periodic_drift(
        cells_1d=(8,), degrees=(1,), cells_2d=(8, 12), burgers_cells=(16,),
    )
    two_d = [r for r in result["rows"] if r["case"] == "linear_advection_2d"]
    # The 2-D sweep covers degrees (1, 2) on both requested meshes.
    assert len(two_d) == 4
    for row in two_d:
        assert row["scaled_drift"] <= CRITERIA["periodic_drift_max"], row


def test_burgers_nonlinear_flux_conserves_mass():
    result = study_periodic_drift(
        cells_1d=(8,), degrees=(1,), cells_2d=(8,), burgers_cells=(16, 32),
    )
    burgers = [r for r in result["rows"] if r["case"] == "burgers_1d"]
    assert len(burgers) == 2
    for row in burgers:
        assert row["scaled_drift"] <= CRITERIA["periodic_drift_max"], row
        # The sine+0.2 profile has a non-zero mean, so this is a real O(1) mass.
        assert row["value_initial"] == pytest.approx(0.2, abs=1e-13)


def test_periodic_mass_drift_does_not_grow_with_refinement():
    result = study_periodic_drift(
        cells_1d=(16, 64), degrees=(2,), cells_2d=(8,), burgers_cells=(16,),
    )
    assert result["drift_grid_ratio"] <= CRITERIA["drift_grid_ratio_max"]


# ---------------------------------------------------------------------------
# Euler global budgets
# ---------------------------------------------------------------------------
def test_sod_mass_and_energy_are_conserved_at_round_off():
    budget = study_euler_budget(
        cells=(400, 800), fluxes=("hllc",), history_times=(0.2,), history_cells=400,
    )
    for row in budget["rows"]:
        assert row["mass_drift"] <= CRITERIA["euler_mass_drift_max"], row
        assert row["energy_drift"] <= CRITERIA["euler_energy_drift_max"], row


def test_sod_momentum_growth_matches_the_analytic_pressure_budget():
    budget = study_euler_budget(
        cells=(400, 800), fluxes=("hllc", "rusanov"),
        history_times=(0.1, 0.2), history_cells=400,
    )
    initial = sod_initial_totals()
    assert initial["momentum"] == pytest.approx(0.0, abs=1e-15)
    for row in budget["rows"]:
        assert row["momentum_increment"] == pytest.approx(
            (SOD_LEFT[2] - SOD_RIGHT[2]) * SOD_FINAL_TIME, rel=1e-6,
        ), row
        assert row["momentum_budget_error"] <= CRITERIA["euler_momentum_budget_max"]
    # The measurement must reject the rival hypotheses (wall / no boundary flux).
    assert budget["hypothesis_errors"]["transmissive_analytic"] < 1e-6
    assert budget["hypothesis_errors"]["reflecting_wall"] > 0.5


def test_conserved_totals_integrates_the_conservative_variables():
    states = np.array([[1.0, 0.0, 2.5], [0.125, 0.0, 0.25]])
    totals = conserved_totals(states, cell_width=0.5)
    assert totals["mass"] == pytest.approx(0.5625)
    assert totals["momentum"] == pytest.approx(0.0)
    assert totals["energy"] == pytest.approx(1.375)


# ---------------------------------------------------------------------------
# Rankine-Hugoniot shock speed
# ---------------------------------------------------------------------------
def test_sod_shock_speed_matches_rankine_hugoniot():
    result = study_shock_speed(cells=(200, 400))
    assert result["sensor_bias"] <= CRITERIA["sensor_bias_max"]
    for row in result["rows"]:
        assert math.isfinite(row["speed"])
        assert row["speed_error"] <= CRITERIA["shock_speed_error_max"], row
    assert result["speed_monotone"], result["rows"]


# ---------------------------------------------------------------------------
# time-integrator order
# ---------------------------------------------------------------------------
def test_rk4_reaches_fourth_order_in_time_on_a_fixed_mesh():
    result = study_temporal_order(advection_cells=40, burgers_cells=32)
    block = result["advection_rk4"]
    assert block["fit_order"] >= CRITERIA["rk4_order_min"], block
    assert block["fit_order"] == pytest.approx(4.0, abs=0.15)


def test_ssp_rk3_reaches_third_order_in_time_on_a_fixed_mesh():
    result = study_temporal_order(advection_cells=20, burgers_cells=32)
    block = result["burgers_ssp_rk3"]
    assert block["fit_order"] >= CRITERIA["ssp_rk3_order_min"], block
    assert block["fit_order"] == pytest.approx(3.0, abs=0.25)


# ---------------------------------------------------------------------------
# why round-off level assertions need ncells >= 400 (measured, not assumed)
# ---------------------------------------------------------------------------
def test_discrete_shock_tail_is_mesh_scale_and_domain_independent():
    """The residual ahead of the captured shock is interior and mesh-scale.

    At t = 0.2 the exact shock sits at x_s = 0.850 and the exact solution is the
    uniform right state beyond it, so x in [0.9, 1.0] holds no physical signal.  Recomputing the same window on a
    domain of twice the length with the same dx leaves the value unchanged, which
    rules out the transmissive closure; the exponential tail width stays O(1)
    cells while dx shrinks, which identifies a discrete-shock artefact.
    """
    result = study_shock_precursor(cells=(200, 400))
    by_cells = {row["ncells"]: row for row in result["rows"]}
    assert by_cells[400]["window_max_walled"] <= CRITERIA["precursor_amplitude_max"]
    assert by_cells[200]["window_max_walled"] > by_cells[400]["window_max_walled"]
    for row in result["rows"]:
        assert row["relative_gap"] <= CRITERIA["precursor_domain_gap_max"], row
        assert row["tail_width_cells"] <= CRITERIA["precursor_width_max_cells"], row
    assert result["decay_rate"] >= CRITERIA["precursor_decay_min"], result
