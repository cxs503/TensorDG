"""Contract tests for the independent cross-check of the ``CASE-LAX-DG`` reference.

The case pool grades DG against algebra that lives in this same repository.  These
tests pin the three properties that make the new witness meaningful:

1. **structural independence** -- the finite-volume path must not evaluate the
   exact Riemann solver at all, and the Rusanov flux must equal its closed form
   (a flux that quietly called the exact solver would certify nothing);
2. **numerics** -- the independent family must converge to the analytic field,
   the second-order leg must beat the first-order one cell for cell, and the
   wave positions / star plateaus must approach the closed-form values;
3. **artifact** -- the committed ``metrics.json`` is re-adjudicated from its own
   numbers, so a hand-edited verdict cannot pass and the retired (honest-FAIL)
   threshold stays visible.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from benchmarks import lax_reference_crosscheck as crosscheck
from benchmarks.lax_reference_crosscheck import (
    DIAPHRAGM,
    FINAL_TIME,
    GAMMA,
    LEFT,
    OUT,
    RIGHT,
    bisection_star_state,
    conserved,
    exact_positions,
    exact_waves,
    gate_entropy,
    gate_positions,
    gate_star,
    measure_leg,
    physical_flux,
    primitive,
    relative_l1,
    rusanov_flux,
    solve_fv,
)
from benchmarks.riemann_exact import star_state

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / OUT / "metrics.json"


def _state(rho, u, p):
    return conserved(np.array([rho]), np.array([u]), np.array([p]), GAMMA)


# ------------------------------------------------------- structural independence
def test_rusanov_flux_is_the_local_lax_friedrichs_formula():
    """The flux must be the closed-form LLF formula, not something hierarchical."""
    left = _state(1.0, 0.5, 1.0)
    right = _state(0.8, -0.2, 0.6)
    gamma = GAMMA
    flux = rusanov_flux(left, right, gamma)
    # explicit reference: 0.5 (F_L + F_R) - 0.5 amax (U_R - U_L)
    rho_l, u_l, p_l = 1.0, 0.5, 1.0
    rho_r, u_r, p_r = 0.8, -0.2, 0.6
    energy_l = p_l / (gamma - 1.0) + 0.5 * rho_l * u_l ** 2
    energy_r = p_r / (gamma - 1.0) + 0.5 * rho_r * u_r ** 2
    reference = np.array([
        0.5 * (rho_l * u_l + rho_r * u_r)
        - 0.5 * max(abs(u_l) + math.sqrt(gamma * p_l / rho_l),
                    abs(u_r) + math.sqrt(gamma * p_r / rho_r)) * (rho_r - rho_l),
        0.5 * (rho_l * u_l ** 2 + p_l + rho_r * u_r ** 2 + p_r)
        - 0.5 * max(abs(u_l) + math.sqrt(gamma * p_l / rho_l),
                    abs(u_r) + math.sqrt(gamma * p_r / rho_r))
        * (rho_r * u_r - rho_l * u_l),
        0.5 * (u_l * (energy_l + p_l) + u_r * (energy_r + p_r))
        - 0.5 * max(abs(u_l) + math.sqrt(gamma * p_l / rho_l),
                    abs(u_r) + math.sqrt(gamma * p_r / rho_r))
        * (energy_r - energy_l),
    ])
    assert np.allclose(flux.ravel(), reference, rtol=1e-14, atol=0.0)


def test_flux_is_the_euler_flux_of_the_state():
    state = _state(1.2, 0.3, 2.0)
    flux = physical_flux(state, GAMMA)
    rho, u, p, _ = primitive(state, GAMMA)
    assert np.allclose(flux.ravel()[0], (rho * u)[0])
    assert np.allclose(flux.ravel()[1], (rho * u * u + p)[0])


@pytest.mark.parametrize("order", [1, 2])
def test_finite_volume_path_never_touches_the_exact_riemann_solver(monkeypatch, order):
    """A shared algebraic route to the answer would defeat the whole cross-check."""

    def forbidden(*args, **kwargs):                      # pragma: no cover - guard
        raise AssertionError("the FV path evaluated the exact Riemann solver")

    monkeypatch.setattr(crosscheck, "star_state", forbidden)
    monkeypatch.setattr(crosscheck, "sample", forbidden)
    fv = solve_fv(64, FINAL_TIME, order)
    assert np.all(np.isfinite(fv["density"]))
    assert fv["density"].min() > 0.0


def test_bisection_star_state_agrees_with_the_repository_newton_solve():
    pressure_newton, velocity_newton = star_state(LEFT, RIGHT)
    pressure_bisect, velocity_bisect = bisection_star_state(LEFT, RIGHT)
    assert abs(pressure_bisect - pressure_newton) / pressure_newton < 1e-13
    assert abs(velocity_bisect - velocity_newton) / abs(velocity_newton) < 1e-13


def test_reference_solution_satisfies_the_lax_entropy_condition():
    gate = gate_entropy(exact_waves(LEFT, RIGHT, FINAL_TIME))
    assert gate["status"] == "PASS"
    assert gate["lower_characteristic"] < gate["shock_speed"] < gate["upper_characteristic"]


# --------------------------------------------------------------------- numerics
def test_first_order_family_converges_to_the_analytic_field():
    legs = [measure_leg(cells, 1) for cells in (100, 200, 400)]
    errors = [leg["errors"]["density"] for leg in legs]
    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < 0.1
    order = math.log(errors[0] / errors[-1]) / math.log(4.0)
    # a first-order scheme attains O(h**0.5) in L1 at a discontinuity
    assert order > 0.4, order


def test_second_order_leg_beats_first_order_at_the_same_resolution():
    first = measure_leg(400, 1)
    second = measure_leg(400, 2)
    for field in ("density", "velocity", "pressure"):
        assert second["errors"][field] < first["errors"][field], field


def test_wave_positions_converge_to_the_closed_form_positions():
    waves = exact_waves(LEFT, RIGHT, FINAL_TIME)
    leg = measure_leg(800, 2, waves)
    gate = gate_positions(leg, waves)
    assert gate["status"] == "PASS", gate
    assert gate["worst_relative_error"] < 0.02
    # the independent family must move towards the closed form, not away from it
    coarse = gate_positions(measure_leg(200, 2, waves), waves)
    assert gate["worst_relative_error"] < coarse["worst_relative_error"]


def test_star_plateaus_converge_to_the_closed_form_star_state():
    waves = exact_waves(LEFT, RIGHT, FINAL_TIME)
    gate = gate_star(measure_leg(800, 2, waves), waves)
    assert gate["status"] == "PASS", gate
    assert gate["worst_relative_error"] < 0.03


def test_relative_l1_vanishes_for_the_exact_field_itself():
    """The measurement must be zero when the 'numerical' field is the exact one."""
    from benchmarks.riemann_exact import cell_averages

    cells = 200
    centres, exact = cell_averages(cells, FINAL_TIME, LEFT, RIGHT, DIAPHRAGM)
    errors = relative_l1(cells, exact)
    assert max(errors.values()) < 1e-13, errors


# --------------------------------------------------------------------- artifact
def _load_artifact():
    assert ARTIFACT.exists(), f"committed artifact missing: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


def test_committed_artifact_re_adjudicates_from_its_own_numbers():
    record = _load_artifact()
    gates = record["gates"]
    adjudicated = {
        name: gate for name, gate in gates.items() if not name.endswith("legacy_threshold")
    }
    recomputed = "PASS" if all(
        gate["status"] == "PASS" for gate in adjudicated.values()) else "FAIL"
    assert recomputed == record["verdict"]
    # convergence statuses must follow from the stored error ladders
    for name, order in (
        ("convergence_first_order", 1),
        ("convergence_second_order", 2),
    ):
        legs = sorted([leg for leg in record["legs"] if leg["order"] == order],
                      key=lambda leg: leg["cells"])
        errors = [leg["errors"]["density"] for leg in legs]
        monotone = all(b < a for a, b in zip(errors[:-1], errors[1:]))
        pairs = [
            math.log(a / b) / math.log(nf / nc)
            for (a, nc), (b, nf) in zip(zip(errors[:-1], [l["cells"] for l in legs[:-1]]),
                                        zip(errors[1:], [l["cells"] for l in legs[1:]]))
        ]
        expected = "PASS" if monotone and min(pairs) >= gates[name]["minimum_order"] else "FAIL"
        assert gates[name]["status"] == expected, name
        assert gates[name]["monotone"] == monotone
        assert min(pairs) == pytest.approx(gates[name]["observed_order"], rel=1e-12)


def test_committed_artifact_positions_and_star_gates_follow_from_stored_values():
    record = _load_artifact()
    waves = exact_waves(LEFT, RIGHT, FINAL_TIME)
    for name in ("positions_finest_first_order", "positions_finest_second_order"):
        gate = record["gates"][name]
        worst = max(row["relative_error"] for row in gate["waves"].values())
        assert worst == pytest.approx(gate["worst_relative_error"], rel=1e-12)
        assert gate["status"] == ("PASS" if worst <= gate["tolerance"] else "FAIL")
    for name in ("star_finest_first_order", "star_finest_second_order"):
        gate = record["gates"][name]
        worst = max(row["relative_error"] for row in gate["values"].values())
        assert worst == pytest.approx(gate["worst_relative_error"], rel=1e-12)
        assert gate["status"] == ("PASS" if worst <= gate["tolerance"] else "FAIL")
    # the stored 'exact' columns must be the closed-form values, not copies of 'got'
    stars = record["gates"]["star_finest_second_order"]["values"]
    assert stars["star_left.density"]["exact"] == pytest.approx(waves["density_star_left"], rel=1e-12)
    assert stars["star_right.density"]["exact"] == pytest.approx(
        waves["density_star_right"], rel=1e-12)


def test_retired_single_threshold_stays_visible_as_an_honest_fail():
    """The first-order legs measured 0.511 against the original 0.60 gate.

    The split into per-reconstruction thresholds is legitimate, but the original
    FAIL must remain in the record: deleting it would hide a retuned criterion.
    """
    record = _load_artifact()
    legacy = record["gates"]["convergence_first_order_legacy_threshold"]
    assert legacy["minimum_order"] == 0.60
    assert legacy["role"].startswith("diagnostic")
    assert legacy["status"] == "FAIL"
    assert legacy["observed_order"] < legacy["minimum_order"]
    assert record["config"]["tolerances"]["order_retired"] == 0.60


def test_artifact_records_the_independent_root_finds():
    record = _load_artifact()
    newton = record["star_state_independent"]["newton"]
    bisect = record["star_state_independent"]["bisection"]
    assert abs(bisect["pressure"] - newton["pressure"]) / newton["pressure"] < 1e-13
    assert abs(bisect["velocity"] - newton["velocity"]) / abs(newton["velocity"]) < 1e-13


def test_artifact_accuracy_table_is_consistent():
    record = _load_artifact()
    table = {row["cells"]: row for row in record["dg_versus_fv_density"]}
    for leg in record["legs"] + record["comparison_legs"]:
        row = table[leg["cells"]]
        assert row[f"fv_p{leg['order']}"] == pytest.approx(leg["errors"]["density"], rel=1e-12)
    bracket = record["accuracy_per_unknown"]["fv_second_order_bracket_matching_dg_error"]
    dg_error = record["accuracy_per_unknown"]["dg_finest"]["rel_l1_density"]
    low, high = sorted(bracket["errors"])
    assert low <= dg_error <= high