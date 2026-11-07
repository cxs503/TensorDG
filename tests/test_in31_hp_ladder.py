"""Pytest gates for the IN31 verification ladder (campaign node IN31-G01).

Two layers, following the repository convention:

* **the references and estimators** are checked against values that are known
  independently of the solver: the sharp L2-projection error, the analytic
  decay slope, the SIPG stiffness spectrum (``pi^2`` for the fundamental mode)
  and the sub-cell shock locator on synthetic fronts of known position.  A
  broken estimator therefore cannot fake a passing gate.
* **the gates themselves** are read back from the committed artifact
  ``docs/benchmarks/results/in31-hp-ladder/gates_verified.json`` and re-derived
  from the recorded measured value, target, threshold and comparator; the cheap
  blocks are additionally re-run from scratch.

The Sod block inside the artifact is expensive, so the fresh Sod re-run is
opt-in::

    pytest -q tests/test_in31_hp_ladder.py
    TENSORDG_IN31_SLOW=1 pytest -q tests/test_in31_hp_ladder.py
    python -m benchmarks.in31_hp_ladder        # regenerate the artifact
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks import in31_hp_ladder as ladder
from benchmarks.verification_ladder import (
    SOD_RIGHT,
    locate_shock,
    sine_projection_error,
)

ARTIFACT = (Path(__file__).resolve().parent.parent
            / "docs" / "benchmarks" / "results" / "in31-hp-ladder"
            / "gates_verified.json")
PREVIOUS_ARTIFACT = (Path(__file__).resolve().parent.parent
                     / "docs" / "benchmarks" / "results" / "verification_ladder"
                     / "gates_verified.json")
SLOW = pytest.mark.skipif(
    not os.environ.get("TENSORDG_IN31_SLOW"),
    reason="set TENSORDG_IN31_SLOW=1 to re-run the Sod block (several minutes)",
)


def _artifact() -> dict:
    if not ARTIFACT.exists():  # pragma: no cover - artifact is committed
        pytest.skip("gates_verified.json missing; run python -m benchmarks.in31_hp_ladder")
    return json.loads(ARTIFACT.read_text())


# ---------------------------------------------------------------------------
# Independent references and estimators
# ---------------------------------------------------------------------------

def test_projection_error_cross_validates_the_existing_reference():
    """The new reference must reproduce the m5280 reference at 2*pi."""
    for ncells in (4, 8, 16):
        for degree in (1, 2, 3, 4, 5):
            ours = ladder.projection_error(ncells, degree)
            theirs = sine_projection_error(ncells, degree)
            assert ours == pytest.approx(theirs, rel=1e-12, abs=1e-18), (ncells, degree)


def test_projection_error_wavenumber_pi_converges_at_p_plus_one():
    """The heat reference (sin(pi x)) must itself show the h**(p+1) rate."""
    for degree in (1, 2, 3, 4):
        errors = [ladder.projection_error(n, degree, math.pi) for n in (8, 16, 32)]
        order = ladder.observed_order((8, 16, 32), errors)
        assert abs(order - (degree + 1)) / (degree + 1) < 0.10, (degree, order)


def test_log_slope_recovers_a_known_exponential_decay():
    """-d log e/dp must return the prescribed decay constant exactly."""
    degrees = [1, 2, 3, 4, 5]
    errors = [3.0 * math.exp(-1.7 * p) for p in degrees]
    assert ladder.log_slope(degrees, errors) == pytest.approx(1.7, rel=1e-10)
    with pytest.raises(ValueError):
        ladder.log_slope(degrees, [1.0, 0.0, 0.1, 0.01, 0.001])


def test_estimated_shock_location_beats_the_mission_gate():
    """The locator is accurate on synthetic fronts, but not to 0.02 of a cell."""
    floor = ladder.estimator_floor(ncells=160, offsets=11)
    assert floor["estimator_floor_cells"] < ladder.SHOCK_POSITION_TOLERANCE_CELLS
    # The inherited m5280 gate (0.02 dx) sits *below* the estimator floor: the
    # claim is recorded here so the two artifacts cannot be confused.
    assert floor["estimator_floor_cells"] > 0.02


def test_temporal_convergence_of_the_advection_driver():
    """The RK4 temporal error is driven below 2 % at p=1 and p=4."""
    for degree in (1, 4):
        run = ladder.temporal_converged_advection(ncells=16, degree=degree)
        assert run["temporal_converged"]
        assert run["temporal_change"] <= ladder.TEMPORAL_AUDIT_TOLERANCE


def test_sipg_stiffness_spectrum_contains_pi_squared():
    """The fundamental SIPG eigenvalue must approach pi^2 from above."""
    from tensordg.heat_1d import assemble_sipg_1d

    for degree in (1, 2, 3, 4, 5):
        system = assemble_sipg_1d(ncells=16, degree=degree)
        mass = system["mass_matrix"].numpy()
        stiffness = system["diffusion"].numpy()
        eigenvalues = np.linalg.eigvals(np.linalg.solve(mass, stiffness)).real
        assert eigenvalues.min() > 0.0
        assert eigenvalues.min() / math.pi ** 2 - 1.0 <= 0.02, degree
        assert eigenvalues.max() > eigenvalues.min()


# ---------------------------------------------------------------------------
# Fresh runs of the cheap blocks (no artifact involved)
# ---------------------------------------------------------------------------

def test_fresh_advection_order_p1_and_p5():
    for degree, resolutions in ((1, (8, 16)), (5, (8, 16))):
        errors = [ladder.temporal_converged_advection(n, degree)["relative_l2"]
                  for n in resolutions]
        order = ladder.observed_order(resolutions, errors)
        assert abs(order - (degree + 1)) / (degree + 1) < 0.10, (degree, order)


def test_fresh_diffusion_order_p5_is_six():
    from tensordg.heat_1d import heat_error, solve_heat_1d_exact_time

    errors = []
    for ncells in (4, 8, 16):
        centers, coefficients, _ = solve_heat_1d_exact_time(
            ncells=ncells, degree=5, final_time=0.1,
        )
        errors.append(heat_error(centers, coefficients, 0.1)["relative_l2"])
    order = ladder.observed_order((4, 8, 16), errors)
    assert abs(order - 6.0) / 6.0 < 0.10, order
    assert errors[0] > errors[1] > errors[2] > 0.0


def test_time_exact_and_crank_nicolson_agree_when_cn_is_resolved():
    from tensordg.heat_1d import heat_error, solve_heat_1d, solve_heat_1d_exact_time

    centers, coefficients, _ = solve_heat_1d(
        ncells=8, degree=2, final_time=0.1, time_steps=20000,
    )
    cn = heat_error(centers, coefficients, 0.1)["relative_l2"]
    centers, coefficients, _ = solve_heat_1d_exact_time(
        ncells=8, degree=2, final_time=0.1,
    )
    exact = heat_error(centers, coefficients, 0.1)["relative_l2"]
    assert cn == pytest.approx(exact, rel=1e-6)


def test_high_degree_heat_runs_and_degree_zero_is_rejected():
    from tensordg.heat_1d import heat_error, solve_heat_1d, solve_heat_1d_exact_time

    previous = float("inf")
    for degree in (4, 5):
        centers, coefficients, metadata = solve_heat_1d(
            ncells=8, degree=degree, final_time=0.1, time_steps=2000,
        )
        error = heat_error(centers, coefficients, 0.1)["relative_l2"]
        assert np.isfinite(error) and error < previous
        assert metadata["dofs"] == 8 * (degree + 1)
        assert np.isfinite(solve_heat_1d_exact_time(
            ncells=8, degree=degree, final_time=0.1)[1]).all()
        previous = error
    with pytest.raises(ValueError):
        solve_heat_1d(ncells=8, degree=0, final_time=0.1)


# ---------------------------------------------------------------------------
# Artifact contract: the verdicts must follow from the records
# ---------------------------------------------------------------------------

def test_artifact_schema_and_verdict():
    artifact = _artifact()
    assert artifact["schema"] == ladder.SCHEMA
    assert artifact["verdict"] in ("PASS", "FAIL")
    assert sorted(artifact["gates"]) == ["C1", "C2", "C3", "C4", "C5"]
    assert artifact["config"]["quick"] is False
    assert artifact["provenance"]["dtype"] == "float64"


def test_thresholds_are_the_mission_criteria():
    thresholds = _artifact()["config"]["thresholds"]
    assert thresholds["order_relative_tolerance"] == 0.10
    assert thresholds["hp_rate_relative_tolerance"] == 0.15
    assert thresholds["shock_position_tolerance_cells"] == 2.0
    assert thresholds["shock_strength_tolerance"] == 0.05


def test_every_check_is_self_describing_and_rederivable():
    artifact = _artifact()
    for gate_id, body in artifact["gates"].items():
        assert body["id"] == gate_id
        assert body["checks"], gate_id
        for record in body["checks"]:
            for key in ("name", "metric", "measured", "comparator", "threshold",
                        "status"):
                assert key in record, (gate_id, record.get("name"), key)
            assert record["status"] == ladder._rederive(record), (gate_id, record["name"])
        failed = [r["name"] for r in body["checks"] if r["status"] != "PASS"]
        assert body["failed_checks"] == failed
        assert body["status"] == ("FAIL" if failed else "PASS")


def test_gates_pass():
    artifact = _artifact()
    for gate_id in ("C1", "C2", "C3", "C4", "C5"):
        body = artifact["gates"][gate_id]
        assert body["status"] == "PASS", (gate_id, body["failed_checks"])
    assert artifact["verdict"] == "PASS"


def test_c1_order_numbers_and_temporal_audit():
    measured = _artifact()["gates"]["C1"]["measured"]
    for degree in range(1, 6):
        entry = measured[str(degree)]
        assert abs(entry["observed_order"] - (degree + 1)) / (degree + 1) <= 0.10
        assert max(entry["temporal_changes"]) <= ladder.TEMPORAL_AUDIT_TOLERANCE


def test_c2_orders_and_fundamental_eigenvalue():
    measured = _artifact()["gates"]["C2"]["measured"]
    for degree in range(1, 6):
        entry = measured[str(degree)]
        assert abs(entry["observed_order"] - (degree + 1)) / (degree + 1) <= 0.10
        assert abs(entry["eigenvalue_min"] / math.pi ** 2 - 1.0) <= 0.02


def test_c3_shock_position_and_strength():
    measured = _artifact()["gates"]["C3"]["measured"]
    assert set(measured) == {"p1_tvb", "p1_unlimited", "p2_tvb", "p2_unlimited",
                             "p3_tvb", "p3_unlimited"}
    for label, entry in measured.items():
        assert entry["shock_position_error_cells"] <= 2.0, label
        assert entry["shock_position_interface_error_cells"] <= 2.0, label
        assert entry["density_jump_error"] <= 0.05, label
        assert entry["pressure_jump_error"] <= 0.05, label


def test_c4_hp_rates_match_the_sharp_projection_theory():
    artifact = _artifact()
    measured = artifact["gates"]["C4"]["measured"]
    assert measured["p_window"] == [2, 5]
    for name in ("advection_1d", "diffusion_1d"):
        for ncells, entry in measured[name].items():
            deviation = abs(entry["measured_slope_window"] - entry["theory_slope_window"])
            assert deviation / abs(entry["theory_slope_window"]) <= 0.15, (name, ncells)
            errors = entry["errors"]
            assert all(a > b for a, b in zip(errors, errors[1:])), (name, ncells)
    for record in artifact["gates"]["C4"]["checks"]:
        if record["name"].startswith("p_refinement_gain"):
            assert record["measured"] >= 1.0e3, record["name"]


def test_this_node_is_not_a_copy_of_the_previous_ladder():
    artifact = _artifact()
    assert ARTIFACT != PREVIOUS_ARTIFACT
    if PREVIOUS_ARTIFACT.exists():
        previous = json.loads(PREVIOUS_ARTIFACT.read_text())
        assert artifact["schema"] != previous["schema"]
        assert sorted(previous["gates"]) != sorted(artifact["gates"])
        assert "sod_density_jump_p1_tvb" in {
            r["name"] for r in artifact["gates"]["C3"]["checks"]}
    degrees = [int(k) for k in artifact["gates"]["C1"]["measured"]]
    assert max(degrees) == 5  # m5280 stopped at p = 3


@SLOW
def test_sod_block_reproduces_from_scratch():
    case = ladder.sod_run(ncells=160, degree=1, limiter="tvb")
    assert case["shock_position_error_cells"] <= ladder.SHOCK_POSITION_TOLERANCE_CELLS
    assert case["density_jump_error"] <= ladder.SHOCK_STRENGTH_TOLERANCE
    assert case["pressure_jump_error"] <= ladder.SHOCK_STRENGTH_TOLERANCE
    reference = ladder.sod_reference(case["final_time"])
    shock = locate_shock(np.asarray(case["density_profile"]),
                         reference["rho_star_right"], SOD_RIGHT[0])
    assert abs(shock["position"] - case["shock_position"]) < 1e-12
