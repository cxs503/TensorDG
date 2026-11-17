"""Gate power of ``CASE-DG-EULER-VORTEX-ERR``: can each criterion actually fail?

The twelve acceptance criteria were certified green round by round; eleven of
them came with a hand-written counter-example test, C4 (entropy budget) did not.
This module grades the gate itself, on the **committed** evidence bundle, through
``tools/audit_vortex_gate_power.py``:

* every criterion must be killed by at least one plausible defect;
* two control mutations (timing / iteration bookkeeping) must not move anything;
* every named threshold must bracket: a defect just inside it passes, a defect
  just outside it fails (the number the criterion *states* is the number the code
  *applies*).

The last test is the regression guard for the C9 defect fixed in this round: the
mesh-refinement fit used to average three advection directions at the same cell
count, which made DG-P2 read a 4.09 order.
"""
import copy

import pytest

from benchmarks import case_euler_vortex_errconserve as ec
from tools import audit_vortex_gate_power as ap


@pytest.fixture(scope="module")
def bundle():
    return ap.load_bundle()


@pytest.fixture(scope="module")
def result(bundle):
    return ap.audit(ap.BUNDLE)


@pytest.fixture(scope="module")
def calibration(bundle):
    return ap.calibrate(ap.BUNDLE)


def test_baseline_reproduces_the_published_verdict(result):
    """The audit must start from the published state, not from a convenient one."""
    assert result["baseline"]["verdict"] == "PASS"
    assert result["record_count"] == 40
    assert result["certificate_count"] == 19
    assert all(status == "passed" for status in result["baseline"]["statuses"].values())


def test_every_acceptance_criterion_is_killed_by_a_plausible_defect(result):
    assert result["survivors"] == []
    assert result["power"] == "12/12"
    for name, killers in result["killed_by"].items():
        assert killers, f"{name} survived every mutation aimed at it"


def test_control_mutations_are_inert(result):
    """A gate that reacts to wall-clock or iteration metadata is fragile."""
    controls = [row for row in result["rows"] if row["kind"] == "control"]
    assert len(controls) == 2
    for row in controls:
        assert row["changed"] == []
        assert row["verdict"] == "PASS"
    assert result["control_inert"] is True


def test_entropy_budget_criterion_can_fail(result):
    """C4 had no counter-example before this round: pin the one that supplies it."""
    row = next(row for row in result["rows"] if row["target"] == "C4_entropy_budget")
    assert row["killed"] is True
    assert row["statuses"]["C4_entropy_budget"] == "failed"
    # and it is specific: the entropy overrun must not be read as a tolerance failure
    assert "C1_tolerance" not in row["changed"]


@pytest.mark.parametrize("criterion", sorted({entry[1] for entry in ap.CALIBRATIONS}))
def test_threshold_calibration_brackets_the_stated_constant(calibration, criterion):
    row = next(row for row in calibration["rows"] if row["criterion"] == criterion)
    assert row["readings"]["inside"]["status"] == "passed", row
    assert row["readings"]["outside"]["status"] == "failed", row
    assert row["brackets"] is True


def test_calibration_covers_every_constant_bearing_criterion(calibration):
    assert calibration["miscalibrated"] == []
    assert set(calibration["calibrated"]) == {entry[1] for entry in ap.CALIBRATIONS}


def test_c9_fit_uses_the_baseline_direction_only(bundle):
    """The C9 regression: a rotated leg must not enter the mesh-refinement fit.

    Three 32-cell legs (``diag45``/``axis0``/``obl30``) plus one 64-cell leg made
    the least-squares slope mostly a direction spread, and DG-P2 then read a 4.09
    "order" at t = 1 -- above the p + 1 = 3 asymptote.  The fit is now pinned to
    the baseline direction, exactly as it is pinned to one CFL.
    """
    records, exact, certificates = bundle
    gates = ap.evaluate(records, exact, certificates)
    for degree in ec.ORDER_FAMILIES:
        for checkpoint in ec.ORDER_CHECKPOINTS:
            row = gates["C9_design_order"]["per_case"][f"dg_d{degree}_t{checkpoint:g}"]
            assert sorted(row["cells"]) == list(ec.ORDER_CELLS), row
            assert row["order"] <= degree + 2, row      # no mixed-direction slope


def test_c9_is_immune_to_a_rotated_leg_with_a_wrong_error(bundle):
    records, exact, certificates = bundle
    base = ap.evaluate(records, exact, certificates)["C9_design_order"]["per_case"]
    mutated = copy.deepcopy(records)
    for record in mutated:
        if ec.record_direction(record) != ec.DIRECTION_BASELINE:
            ap.scale_errors(record, 100.0)
    after = ap.evaluate(mutated, exact, certificates)["C9_design_order"]["per_case"]
    assert after == base


def test_c9_detects_a_genuine_mesh_order_collapse(bundle):
    """With the contamination gone, C9 must react to the defect it exists for."""
    records, exact, certificates = bundle
    mutated = copy.deepcopy(records)
    ap.set_dg_order(mutated, 1.0, 1.0)      # first-order refinement at t = 1
    gates = ap.evaluate(mutated, exact, certificates)
    assert gates["C9_design_order"]["status"] == "failed"
    assert "dg_d2_t1" in gates["C9_design_order"]["failures"]
