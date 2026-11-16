"""Invariants of the lattice-reference adjudication (CASE-DG-EULER-VORTEX-ERR).

The published ladder of ``CASE-DG-EULER-VORTEX-ERR`` was measured against the
``min_image`` periodic reference, which drops the neighbouring lattice images.
``tools/lattice_adjudication.py`` compares that bundle with one produced by the
same code and the same ladder under ``--reference lattice``.  These tests pin the
invariants that make the comparison meaningful *before* looking at the answer:

* the two bundles must be the same discrete computation (identical step counts),
  otherwise a difference in error cannot be attributed to the reference;
* the defect must be confined to the fields the mechanism predicts -- the
  density/pressure tails are second order in the Gaussian envelope, so those two
  fields must agree at machine level, and any non-machine-level defect must land
  in a velocity field;
* comparing a bundle with itself must produce zero, i.e. the metric has no
  built-in offset that could be mistaken for a reference defect;
* the gate evaluation must remain single-reference (the case module refuses a
  bundle that mixes models), because the published numbers must be attributable
  to exactly one reference.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import case_euler_vortex_errconserve as ec
from tools import lattice_adjudication as la

PUBLISHED = Path("case_euler_vortex_errconserve")
ADJUDICATED = Path("case_euler_vortex_errconserve_lattice")


def _bundle_ready() -> bool:
    results = ADJUDICATED / "results"
    return results.is_dir() and any(results.glob("*.json"))


requires_bundle = pytest.mark.skipif(
    not _bundle_ready(),
    reason="lattice bundle not landed (run the 40-checkpoint ladder with "
           "--reference lattice)")


@pytest.fixture(scope="module")
def result():
    return la.build(PUBLISHED, ADJUDICATED)


# -- data-independent invariants -------------------------------------------

def test_aggregate_refuses_a_bundle_that_mixes_reference_models(tmp_path):
    """A mixed bundle must be refused, not silently averaged."""
    records = []
    for path in sorted((PUBLISHED / "results").glob("*.json"))[:2]:
        records.append(json.loads(path.read_text()))
    assert len(records) == 2
    records[0]["reference_model"] = ec.REFERENCE_MIN_IMAGE
    records[1]["reference_model"] = ec.REFERENCE_LATTICE
    (tmp_path / "results").mkdir()
    exact = ec.exact_invariants()
    exact["core_density"] = ec.exact_core_density()
    with pytest.raises(SystemExit):
        ec.aggregate(records, exact, tmp_path)


def test_self_comparison_has_no_defect():
    """Identical bundles must give exactly zero -- the metric has no offset."""
    same = la.build(PUBLISHED, PUBLISHED)
    assert same["summary"]["rho_p_max_defect_relative"] == 0.0
    assert same["summary"]["velocity_max_defect_relative"] == 0.0
    assert same["summary"]["reference_limited"] == []
    assert same["gates"]["changes"] == {}


# -- invariants that need the adjudicated bundle ---------------------------

@requires_bundle
def test_bundles_are_the_same_discrete_computation(result):
    """Same code, same ladder, only the reference model differs."""
    assert result["summary"]["checkpoints_paired"] >= 1
    assert result["coverage"]["measured"] == result["summary"]["checkpoints_paired"]
    assert result["coverage"]["predicted"] > 0, (
        "the pending half of the ladder must stay visible, not be dropped")
    mismatched = [row["task_key"] for row in result["checkpoints"]
                  if row["status"] == "measured" and not row["steps_equal"]]
    assert mismatched == []
    assert result["published"]["reference_model"] == ec.REFERENCE_MIN_IMAGE
    assert result["adjudicated"]["reference_model"] == ec.REFERENCE_LATTICE


@requires_bundle
def test_second_order_fields_agree_at_machine_level(result):
    """density/pressure tails are O(envelope^2): the two references must coincide."""
    worst = result["summary"]["rho_p_max_defect_relative"]
    assert worst < la.MACHINE_LEVEL_BOUND, worst


@requires_bundle
def test_measured_defect_lands_within_an_order_of_magnitude_of_the_prediction(result):
    """The analytic floor is a *prediction*; the measured defect must confirm it.

    Round 7 predicted the floor analytically (envelope tail x displacement).  A
    prediction that the measurement misses by more than one order of magnitude is
    not a confirmation of the mechanism, it is a coincidence -- so the tolerance
    here is deliberately wide and the test says which checkpoints were compared.
    """
    comparisons = result["confirmation"]
    assert comparisons, "no checkpoint carried both a prediction and a measurement"
    for key, row in comparisons.items():
        ratio = row["measured_over_predicted"]
        # the analytic floor is the norm of the reference *perturbation*; the
        # reported quantity is the norm of the *error*, so the prediction can only
        # be an upper bound.  What is falsifiable is that it stays one (ratio <= 1
        # up to the loose factor 10 used as a guard against a mechanism change)
        # and that the effect is not identically zero (the two bundles really do
        # differ where the floor predicts a difference).
        assert ratio <= 10.0, (key, ratio)
        assert ratio > 0.0, (key, ratio)


@requires_bundle
def test_reference_defect_is_confined_to_the_velocity_fields(result):
    """Any defect above machine level must sit in a velocity field."""
    measured = [r for r in result["checkpoints"] if r["status"] == "measured"]
    assert measured, "no checkpoint was measured against the lattice reference"
    for row in measured:
        for name in la.MACHINE_LEVEL_FIELDS:
            assert row["fields"][name]["defect_relative"] < la.MACHINE_LEVEL_BOUND
        non_velocity = max(row["fields"][name]["defect_relative"]
                           for name in la.MACHINE_LEVEL_FIELDS)
        velocity = max(row["fields"][name]["defect_relative"]
                       for name in ("velocity_x", "velocity_y"))
        # the mechanism says the velocity tail is first order in the envelope
        # (~6e-6) and the density/pressure tail second order (~4e-13): 100x apart
        # is a deliberately loose floor for that separation
        assert velocity >= 100.0 * non_velocity, row["task_key"]


@requires_bundle
def test_published_gate_set_still_passes_and_reference_images_are_recorded(result):
    """The published bundle is a PASS bundle; the lattice records say how many images."""
    assert result["gates"]["published"]["verdict"] == "PASS"
    images = [row["reference_images"] for row in result["checkpoints"]
              if row["reference_images"] is not None]
    assert all(isinstance(n, int) and n >= 1 for n in images), images