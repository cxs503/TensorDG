"""Pytest gates for the zero-to-one verification ladder (C1--C6).

Two layers, mirroring ``tests/test_verification.py``:

* **the ladder's own references and estimators** are checked against values that
  are known independently of any solver -- the sharp L2-projection error of a
  sine wave, Toro's published Sod star state, a synthetic step whose position is
  known exactly.  A broken estimator therefore cannot fake a passing gate.
* **the gates themselves** are recomputed from the committed artifact
  ``docs/benchmarks/results/verification_ladder/gates_verified.json`` and, for
  the cheap blocks, re-derived from fresh solver runs.  Every gate record carries
  its measured value *and* its threshold, so the status can be re-derived from
  the record; a hand-edited verdict fails the contract test.

The Sod block is expensive (DG p = 1..3 with the TVB limiter) and is therefore
guarded by ``TENSORDG_LADDER_SLOW=1``::

    pytest -q tests/test_verification_ladder.py
    TENSORDG_LADDER_SLOW=1 pytest -q tests/test_verification_ladder.py
    python -m benchmarks.verification_ladder        # regenerate the artifact
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks import verification_ladder as ladder
from benchmarks.riemann_exact import GAMMA, sample as sample_riemann
from benchmarks.sod_exact import cell_averages

ARTIFACT = (Path(__file__).resolve().parent.parent
            / "docs" / "benchmarks" / "results" / "verification_ladder"
            / "gates_verified.json")
SLOW = pytest.mark.skipif(
    not os.environ.get("TENSORDG_LADDER_SLOW"),
    reason="set TENSORDG_LADDER_SLOW=1 to run the Sod ladder (several minutes)",
)


# ---------------------------------------------------------------------------
# Independent references and estimators
# ---------------------------------------------------------------------------

def test_sine_projection_floor_converges_at_p_plus_one():
    """The analytic reference used by C4 must itself show the h**(p+1) rate."""
    for degree in (1, 2, 3):
        resolutions = (8, 16, 32)
        errors = [ladder.sine_projection_error(n, degree) for n in resolutions]
        order = ladder.observed_order(resolutions, errors)
        assert abs(order - (degree + 1)) / (degree + 1) < 0.15, (degree, order)
        assert errors[0] > errors[1] > errors[2] > 0.0


def test_sod_reference_matches_published_star_state():
    """Sod star state and shock speed against Toro's book values."""
    reference = ladder.sod_reference(0.2)
    assert reference["p_star"] == pytest.approx(0.30313, abs=5e-5)
    assert reference["u_star"] == pytest.approx(0.92745, abs=5e-5)
    assert reference["rho_star_right"] == pytest.approx(0.26557, abs=5e-5)
    assert reference["shock_speed"] == pytest.approx(1.75216, abs=5e-4)
    assert reference["shock_position"] == pytest.approx(
        0.5 + reference["shock_speed"] * 0.2, abs=1e-12
    )


def test_locate_shock_resolves_a_sub_cell_ramp():
    """A one-cell-thick front is located to better than 10 % of a cell.

    This is the practical resolution floor of a mid-level crossing on sampled
    data: for a front thinner than one cell only two samples bracket the level,
    and linear interpolation of a curved front leaves a few percent of a cell.
    """
    ncells = 100
    dx = 1.0 / ncells
    centers = (np.arange(ncells) + 0.5) * dx
    for fraction in (0.1, 0.35, 0.8):
        true_position = 0.5 + fraction * dx
        # a front one cell thick: the steepest jump is the front itself
        profile = np.clip(1.0 - (centers - (true_position - 0.5 * dx)) / dx, 0.0, 1.0)
        found = ladder.locate_shock(profile, left_level=1.0, right_level=0.0,
                                    window=(0.45, 1.0))
        assert abs(found["position"] - true_position) / dx < 0.10, (
            fraction, found)


def test_locate_shock_step_is_limited_to_the_cell_grid():
    """Documented resolution limit: cell averages of a *step* only pin the centre.

    For a step that does not run through a cell centre the two adjacent cell
    averages are equal to the plateau values whatever the sub-cell position, so
    a level-crossing estimator returns the midpoint between the two centres.
    This test records that floor -- it is the reason the Sod shock-position
    criterion is reported per polynomial degree instead of as a single number.
    """
    ncells = 100
    dx = 1.0 / ncells
    centers = (np.arange(ncells) + 0.5) * dx
    true_position = 0.5 + 0.2 * dx
    profile = np.where(centers < true_position, 1.0, 0.0)
    found = ladder.locate_shock(profile, left_level=1.0, right_level=0.0,
                                window=(0.45, 1.0))
    assert abs(found["position"] - true_position) / dx == pytest.approx(0.2, abs=1e-12)


def test_locate_shock_ignores_a_larger_jump_outside_the_window():
    """The contact discontinuity carries the larger density jump in Sod."""
    ncells = 100
    dx = 1.0 / ncells
    centers = (np.arange(ncells) + 0.5) * dx
    profile = np.where(centers < 0.5, 1.0, np.where(centers < 0.7, 0.4, 0.2))
    found = ladder.locate_shock(profile, left_level=0.4, right_level=0.2,
                                window=(0.65, 1.0))
    assert abs(found["position"] - 0.7) / dx < 0.06, found
    # with the window forced onto the contact the (larger) first jump would be
    # reported instead -- that is exactly the failure mode the window prevents
    contact = ladder.locate_shock(profile, left_level=1.0, right_level=0.4,
                                  window=(0.45, 0.55))
    assert abs(contact["position"] - 0.5) / dx < 0.06, contact


def test_locate_shock_requires_a_window_inside_the_domain():
    with pytest.raises(ValueError):
        ladder.locate_shock(np.linspace(1.0, 0.0, 40), 1.0, 0.0, window=(2.0, 3.0))


def test_gibbs_amplitude_is_zero_inside_physical_bounds():
    exact = np.asarray(cell_averages(64, 0.2)[1]["density"], dtype=float)
    metrics = ladder.gibbs_amplitude(exact)
    assert metrics["amplitude"] == 0.0
    overshoot = exact.copy()
    overshoot[10] = 1.02
    assert ladder.gibbs_amplitude(overshoot)["overshoot"] == pytest.approx(
        0.02 / ladder.JUMP, rel=1e-12
    )


# ---------------------------------------------------------------------------
# Cheap gates, re-derived from fresh runs
# ---------------------------------------------------------------------------

def test_c1_advection_h_convergence_matches_p_plus_one():
    block = ladder.advection_ladder()
    gate = ladder.evaluate_c1(block)
    assert gate["status"] == "PASS", gate["failed_checks"]
    for degree, data in block["degrees"].items():
        assert abs(data["observed_order"] - (degree + 1)) / (degree + 1) < 0.15


def test_c1b_advection_2d_h_convergence_matches_p_plus_one():
    block = ladder.advection_2d_ladder(degrees=(1, 2))
    gate = ladder.evaluate_c1b(block)
    assert gate["status"] == "PASS", gate["failed_checks"]


def test_c2_heat_h_convergence_matches_p_plus_one():
    block = ladder.heat_ladder(time_steps=1200)
    gate = ladder.evaluate_c2(block)
    assert gate["status"] == "PASS", gate["failed_checks"]
    for degree, data in block["degrees"].items():
        assert abs(data["observed_order"] - (degree + 1)) / (degree + 1) < 0.15


def test_c4_p_refinement_tracks_the_projection_floor():
    block = ladder.advection_ladder()
    gate = ladder.evaluate_c4(block, ladder.heat_ladder(time_steps=1200), None)
    ratios = gate["measured"]["p_refinement"]
    for ncells, entry in ratios.items():
        if int(ncells) < 8:
            continue
        assert entry["ratios"], entry
        for item in entry["ratios"]:
            assert item["relative_error"] <= ladder.ORDER_RELATIVE_TOLERANCE, item
    gate_checks = [check for check in gate["checks"]
                   if check["name"].startswith("p_refinement_")]
    assert gate_checks and all(check["status"] == "PASS" for check in gate_checks)


# ---------------------------------------------------------------------------
# Artifact contract (C6) -- guards against hand-edited verdicts
# ---------------------------------------------------------------------------

def _load_artifact() -> dict:
    assert ARTIFACT.is_file(), (
        f"missing {ARTIFACT}; regenerate with `python -m benchmarks.verification_ladder`"
    )
    return json.loads(ARTIFACT.read_text())


def test_artifact_exists_with_all_gates_and_thresholds():
    result = _load_artifact()
    assert result["schema"] == "tensordg-verification-ladder/1"
    for gate_id in ("C1", "C1b", "C2", "C3", "C4", "C5", "C6"):
        gate = result["gates"][gate_id]
        assert gate["checks"], gate_id
        assert gate["criterion"]
        assert gate["status"] in ("PASS", "FAIL")
    thresholds = result["config"]["thresholds"]
    assert thresholds["order_relative_tolerance"] == ladder.ORDER_RELATIVE_TOLERANCE
    assert thresholds["shock_position_tolerance_cells"] == \
        ladder.SHOCK_POSITION_TOLERANCE_CELLS


def test_artifact_gate_status_is_derivable_from_its_checks():
    """The verdict must follow from the stored measurements, not from a flag."""
    result = _load_artifact()
    for gate_id in ("C1", "C1b", "C2", "C3", "C4", "C5", "C6"):
        gate = result["gates"][gate_id]
        recomputed = ("FAIL" if any(check["status"] != "PASS"
                                    for check in gate["checks"]) else "PASS")
        assert gate["status"] == recomputed, gate_id
        assert gate["failed_checks"] == [
            check["name"] for check in gate["checks"] if check["status"] != "PASS"
        ], gate_id
    derived = ("FAIL" if any(gate["status"] != "PASS"
                             for gate in result["gates"].values()) else "PASS")
    assert result["verdict"] == derived


def test_artifact_thresholds_are_respected_by_the_measurements():
    """Every numeric check must be consistent with its own stored threshold."""
    result = _load_artifact()
    for gate_id in ("C1", "C1b", "C2", "C3", "C4", "C5"):
        for check in result["gates"][gate_id]["checks"]:
            if "relative_error" in check:
                expected = ("PASS" if check["relative_error"] <= check["tolerance"]
                            else "FAIL")
                assert check["status"] == expected, (gate_id, check["name"])
            if "measured_cells" in check:
                expected = ("PASS"
                            if check["measured_cells"] <= check["tolerance_cells"]
                            else "FAIL")
                assert check["status"] == expected, (gate_id, check["name"])


def test_artifact_provenance_and_figures():
    result = _load_artifact()
    provenance = result["provenance"]
    assert provenance["git_commit"], "artifact must record the commit it came from"
    assert provenance["torch"] and provenance["numpy"] and provenance["python"]
    assert provenance["dtype"] == "float64" and provenance["device"] == "cpu"
    assert result["figures"], "the ladder must emit comparison figures"
    for figure in result["figures"]:
        assert Path(figure["path"]).is_file(), figure["path"]
        assert figure["caption"]
    assert (ARTIFACT.parent / "summary.md").is_file()
    for name in ("advection_1d_convergence.csv", "advection_2d_convergence.csv",
                 "heat_1d_convergence.csv", "sod_diagnostics.csv"):
        assert (ARTIFACT.parent / name).is_file(), name


def test_artifact_sod_reference_is_the_exact_riemann_value():
    result = _load_artifact()
    reference = result["studies"]["sod_1d"]["reference"]
    published = ladder.sod_reference(reference["final_time"])
    for key in ("p_star", "u_star", "shock_speed", "shock_position"):
        assert reference[key] == pytest.approx(published[key], rel=1e-12)
    assert reference["shock_speed"] == pytest.approx(1.75216, abs=5e-4)


# ---------------------------------------------------------------------------
# Slow block: Sod gates C3/C5 from a fresh, reduced-cost ladder
# ---------------------------------------------------------------------------

@SLOW
def test_sod_gibbs_monotone_and_limiter_needed():
    sod = ladder.sod_ladder(ncells=100, degrees=(1, 2, 3), cfl=0.3)
    gate_c5 = ladder.evaluate_c5(sod)
    assert gate_c5["status"] == "PASS", gate_c5["failed_checks"]
    amplitudes = [sod["cases"][f"p{degree}_unlimited"]["gibbs_amplitude"]
                  for degree in (1, 2, 3)]
    assert amplitudes[0] > amplitudes[1] > amplitudes[2] > 0.0, amplitudes
    for degree in (1, 2, 3):
        unlimited = sod["cases"][f"p{degree}_unlimited"]
        limited = sod["cases"][f"p{degree}_tvb"]
        assert unlimited["gibbs_amplitude"] > limited["gibbs_amplitude"]
        assert unlimited["total_variation_ratio"] > 1.0


@SLOW
def test_sod_shock_position_gate_against_exact_riemann_solution():
    """Shock position error in cell units, per polynomial degree.

    ``SHOCK_POSITION_TOLERANCE_CELLS = 0.02`` is the campaign criterion.  The
    measured values are asserted individually so that a regression in the
    shock-capturing path is visible even when a configuration does not meet the
    2 % bar: the cell averages of a captured shock locate the front to a few
    percent of a cell (see ``test_locate_shock_step_is_limited_to_the_cell_grid``),
    which is why the criterion is reported per configuration rather than as a
    single pass/fail number.
    """
    sod = ladder.sod_ladder(ncells=100, degrees=(1, 2, 3), cfl=0.3)
    gate = ladder.evaluate_c3(sod)
    worst = 0.0
    best = 1.0
    for degree in (1, 2, 3):
        for limiter in ("tvb", "unlimited"):
            case = sod["cases"][f"p{degree}_{limiter}"]
            error = case["shock_position_error_cells"]
            assert error <= 0.10, (degree, limiter, case)
            worst = max(worst, error)
            best = min(best, error)
    # at least one configuration must clear the campaign bar
    assert best <= ladder.SHOCK_POSITION_TOLERANCE_CELLS, (best, worst)
    # the stored statuses must follow from the stored measurements
    for check in gate["checks"]:
        if "measured_cells" in check:
            expected = ("PASS" if check["measured_cells"] <= check["tolerance_cells"]
                        else "FAIL")
            assert check["status"] == expected, check
    assert [check["status"] for check in gate["checks"]
            if check["name"].startswith("gibbs_monotone")] == ["PASS", "PASS"]


@SLOW
def test_sod_case_matches_exact_cell_averages():
    """The density relative L2 error must stay in the sub-percent range."""
    case = ladder.sod_case(ncells=100, degree=2, limiter=None, cfl=0.3)
    assert case["density_relative_l2"] < 0.01, case["density_relative_l2"]
    assert np.isfinite(sample_riemann(np.array([0.9]), 0.2, ladder.SOD_LEFT,
                                      ladder.SOD_RIGHT, 0.5, GAMMA)["density"]).all()