"""Measured reference-induced shift of CASE-EULER-VORTEX-ERR (round 20).

The published ladder is measured against the *min-image fold* of the isentropic
vortex.  ``tests/test_vortex_reference_floor.py`` pins the size of that fold's
defect and turns it into a worst-case bound.  These tests pin the two things the
bound alone cannot decide:

* **coverage**: every published record carries a bound.  The first version of the
  floor artifact was generated before the ``dg_d1_n64`` / ``dg_d3_n32`` /
  ``dg_d2_n32_t20`` records landed and silently reported ``None`` for them --
  so the headline "worst floor ratio 12.7 %" was a statement about the covered
  subset only.  The real worst checkpoint is ``dg_d3_n32`` at 25.3 %, which is
  why coverage is a gate and not a comment.
* **the bound is measured, not assumed**: the floor budget is worst-case
  (reference defect perfectly aligned with the solution error).  A lattice
  re-run of the same checkpoint measures the actual shift, and only velocity may
  move -- density and pressure are second order in the Gaussian envelope.

The cheap end-to-end case runs a deliberately coarse (8x8, p=2) checkpoint so the
CI cost stays at a few seconds; the production measurement lives in
``case_euler_vortex_errconserve_lattice/`` and is produced on the cluster.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import case_euler_vortex_errconserve as ec
from benchmarks import vortex_reference_floor as rf
from benchmarks import vortex_reference_upgrade as ru

REPO = Path(__file__).resolve().parents[1]
PUBLISHED_DIR = REPO / "case_euler_vortex_errconserve"
LATTICE_DIR = REPO / "case_euler_vortex_errconserve_lattice"
FLOOR_JSON = LATTICE_DIR / "reference_floor.json"


def _record(max_percent, errors=None, final_time=1.0, cells=32, degree=2,
            task="dg", task_key="dg_d2_n32_t1_cfl0.2", background=None,
            cfl=0.2):
    config = {"task": task, "cells": cells, "final_time": final_time,
              "cfl": cfl, "strength": ec.STRENGTH}
    if task == "dg":
        config["degree"] = degree
    if background is not None:
        config["background_velocity"] = list(background)
    errors = errors or {"density": max_percent / 100.0,
                        "velocity_x": max_percent / 100.0,
                        "velocity_y": max_percent / 100.0,
                        "pressure": max_percent / 100.0}
    return {"task": task, "task_key": task_key, "cells": cells, "degree": degree,
            "final_time": final_time, "cfl": cfl, "config": config,
            "errors": errors, "max_percent": max_percent,
            "max_relative_l2": max(errors.values())}


# ---------------------------------------------------------------------------
# arithmetic of the comparison
# ---------------------------------------------------------------------------

def test_relative_shift_is_field_wise_and_scaled_by_the_baseline():
    baseline = _record(1.0, errors={"density": 1.0e-5, "velocity_x": 2.0e-5,
                                    "velocity_y": 2.0e-5, "pressure": 1.0e-5})
    candidate = _record(1.2, errors={"density": 1.0e-5, "velocity_x": 2.4e-5,
                                     "velocity_y": 2.4e-5, "pressure": 1.0e-5})
    shift = ru.relative_shift(candidate, baseline)
    assert shift["per_field"]["density"] == pytest.approx(0.0, abs=1e-12)
    assert shift["per_field"]["velocity_x"] == pytest.approx(0.2, rel=1e-9)
    assert shift["worst_field"] in ("velocity_x", "velocity_y")
    assert shift["max_relative_shift"] == pytest.approx(0.2, rel=1e-9)
    assert shift["max_percent_shift"] == pytest.approx(0.2, rel=1e-9)


def test_only_a_shift_above_the_material_threshold_is_flagged():
    errors = {"density": 1.0e-5, "velocity_x": 2.0e-5, "velocity_y": 2.0e-5,
              "pressure": 1.0e-5}
    published = {"dg_d2_n32_t1_cfl0.2": _record(2.0e-3, errors=errors)}
    # 1e-4 relative move: a reference artifact three orders below the 6 % gate
    immaterial = {"dg_d2_n32_t1_cfl0.2": _record(
        2.0002e-3, errors={**errors, "velocity_x": 2.0e-5 * (1 + 1e-4)})}
    payload = ru.build_report(published, immaterial, {}, {})
    assert payload["verdict"] == "no_material_reference_effect"
    assert payload["material_checkpoints"] == []
    assert payload["measured_effect_max"] == pytest.approx(1e-4, rel=1e-6)

    # 5 % move: would rewrite a published number, so it must be flagged
    material = {"dg_d2_n32_t1_cfl0.2": _record(
        2.1e-3, errors={**errors, "velocity_x": 2.0e-5 * 1.05})}
    payload = ru.build_report(published, material, {}, {})
    assert payload["verdict"] == "material_reference_effect"
    assert payload["material_checkpoints"] == ["dg_d2_n32_t1_cfl0.2"]


def test_bound_soundness_is_decided_by_the_measured_effect():
    published = {"k": _record(1.0, errors={"density": 1.0e-5, "velocity_x": 2.0e-5,
                                           "velocity_y": 2.0e-5, "pressure": 1.0e-5},
                              task_key="k")}
    rerun = {"k": _record(1.0, errors={"density": 1.0e-5, "velocity_x": 2.0e-5,
                                       "velocity_y": 2.0e-5, "pressure": 1.0e-5},
                          task_key="k")}
    # a 1 % move under a 20 % bound: sound, and only 5 % of the bound realised
    lattice = {"k": _record(1.01, errors={"density": 1.0e-5,
                                          "velocity_x": 2.0e-5 * 1.01,
                                          "velocity_y": 2.0e-5,
                                          "pressure": 1.0e-5}, task_key="k")}
    row = ru.build_report(published, lattice, rerun, {"k": 0.2})["checkpoints"][0]
    assert row["bound_is_sound"] is True
    assert row["bound_materialised"] == pytest.approx(0.01 / 0.2, rel=1e-9)

    # a 30 % move under a 20 % bound: the bound was violated, and that must show
    lattice = {"k": _record(1.3, errors={"density": 1.0e-5,
                                         "velocity_x": 2.0e-5 * 1.3,
                                         "velocity_y": 2.0e-5,
                                         "pressure": 1.0e-5}, task_key="k")}
    row = ru.build_report(published, lattice, rerun, {"k": 0.2})["checkpoints"][0]
    assert row["bound_is_sound"] is False


def test_family_exponent_recovers_a_known_slope():
    records = {}
    for index, time in enumerate((1.0, 2.0, 5.0, 10.0)):
        record = _record(0.0, final_time=time, task_key=f"k{index}")
        record["max_relative_l2"] = 1.0e-4 * time ** 0.2
        record["errors"] = {name: record["max_relative_l2"] for name in ru.FIELDS}
        records[record["task_key"]] = record
    exponents = ru.family_exponents(records)
    (only,) = exponents.values()
    assert only["alpha"] == pytest.approx(0.2, rel=1e-9)
    assert only["checkpoints"] == 4


# ---------------------------------------------------------------------------
# the committed artifacts: coverage and direction dependence
# ---------------------------------------------------------------------------

needs_floor = pytest.mark.skipif(not FLOOR_JSON.exists(),
                                 reason="reference floor artifact not present")


@needs_floor
def test_floor_artifact_covers_every_published_record():
    """Anti-staleness gate: no published reading may go without a bound."""
    payload = json.loads(FLOOR_JSON.read_text())
    published = sorted(p.stem for p in (PUBLISHED_DIR / "results").glob("*.json"))
    rows = payload["checkpoints"]
    assert payload["coverage"]["uncovered"] == []
    assert payload["coverage"]["compared"] == len(published)
    assert {row["task_key"] for row in rows} == set(published)
    assert all(row["published_errors"] is not None for row in rows)


@needs_floor
def test_floor_artifact_names_the_true_worst_checkpoint():
    """The worst floor ratio belongs to ``dg_d3_n32``, not to the finest mesh.

    ``dg_d2_n64`` was the worst *covered* checkpoint when the artifact was first
    generated; ``dg_d3_n32`` (p=3, 32 cells) has a lower discretisation error and
    therefore a larger floor/error ratio.  Pinning the identity keeps a later
    "the worst point is X" claim from being an artifact of what was measured.
    """
    payload = json.loads(FLOOR_JSON.read_text())
    assert payload["worst"].startswith("dg_d3_n32_t1")
    assert payload["max_floor_over_error"] == pytest.approx(0.2531, abs=5e-3)
    # and it must still be a *bound*: the floor never exceeds the measurement
    assert payload["max_floor_over_error"] < 1.0


@needs_floor
def test_reference_floor_depends_on_the_advection_direction():
    """C10's direction axis is not direction-blind in the reference either."""
    payload = json.loads(FLOOR_JSON.read_text())
    rows = {row["task_key"]: row for row in payload["checkpoints"]}
    axis = rows["dg_d2_n32_t1_cfl0.2_diraxis0"]
    diag = rows["dg_d2_n32_t1_cfl0.2"]
    assert axis["direction"] == "axis0" and diag["direction"] == "diag45"
    # the axial shift folds the vortex to |shift| = 5 in one coordinate only, so
    # the dropped image sits much closer than on the diagonal -> larger floor
    assert axis["floor_relative_l2"]["velocity_y"] > 2.0 * diag["floor_relative_l2"]["velocity_y"]
    assert axis["worst_floor_over_error"] < 0.05   # still a small bound


# ---------------------------------------------------------------------------
# cheap end-to-end: the lattice reference moves velocity only
# ---------------------------------------------------------------------------

def test_lattice_reference_moves_velocity_and_not_density():
    config = {"task": "dg", "degree": 2, "cells": 8, "final_time": 1.0, "cfl": 0.2,
              "strength": ec.STRENGTH, "background_velocity": ec.BACKGROUND}
    fold = ec.run_task(dict(config, reference=ec.REFERENCE_MIN_IMAGE))
    lattice = ec.run_task(dict(config, reference=ec.REFERENCE_LATTICE))
    assert lattice["reference_model"] == ec.REFERENCE_LATTICE
    assert lattice["reference_images"] == ec.reference_images(1.0)
    shift = ru.relative_shift(lattice, fold)
    # density/pressure: second order in the envelope -> nothing above round-off
    assert abs(shift["per_field"]["density"]) < 1.0e-9
    assert abs(shift["per_field"]["pressure"]) < 1.0e-9
    # velocity: first order in the envelope + displacement weighted, so the very
    # same change of reference is visible there (measured 3.7e-7 at this point)
    assert abs(shift["per_field"]["velocity_x"]) > 1.0e-8
    assert abs(shift["per_field"]["velocity_y"]) > 1.0e-8
    # and the whole move stays inside the worst-case floor bound of this point
    floor = rf.floor_at(dict(config))
    assert shift["max_relative_shift"] < 1.5 * floor["max_floor_percent"] / fold["max_percent"]

# ---------------------------------------------------------------------------
# family labels and criterion invariance (round 21)
# ---------------------------------------------------------------------------

def test_family_label_matches_the_record():
    """The record's own ``family`` field is the label -- never a doubled one.

    The first version of ``ru.family`` re-appended the direction tag to a freshly
    built label, so the exponent table of the artifact was keyed by
    ``dg_d2_n64_cfl0.2dg_d2_n64_cfl0.2``: a key that matches no record and no
    other artifact in the repo.
    """
    record = _record(1.0, cells=64, degree=2, task_key="dg_d2_n64_t1_cfl0.2")
    record["family"] = "dg_d2_n64_cfl0.2"
    assert ru.family(record) == "dg_d2_n64_cfl0.2"
    # a record without the field still gets the constructed label
    del record["family"]
    assert ru.family(record) == "dg_d2_n64_cfl0.2"
    # and a direction family keeps its tag exactly once
    oblique = _record(1.0, task_key="dg_d2_n32_t1_cfl0.2_dirobl30",
                      background=ec.DIRECTION_AXIS[2][1])
    oblique["family"] = "dg_d2_n32_cfl0.2_dirobl30"
    assert ru.family(oblique) == "dg_d2_n32_cfl0.2_dirobl30"


def _exponent_records(alpha_by_reference, scale=1.0e-4):
    """Two references of one 4-point family with a prescribed exponent each."""
    published, lattice = {}, {}
    for reference, alpha in alpha_by_reference.items():
        bucket = published if reference == "published" else lattice
        for index, time in enumerate((1.0, 2.0, 5.0, 10.0)):
            record = _record(scale * 100.0 * time ** alpha, final_time=time,
                             cells=64, task_key=f"{reference}_{index}")
            record["max_relative_l2"] = scale * time ** alpha
            record["errors"] = {name: record["max_relative_l2"] for name in ru.FIELDS}
            record["family"] = "dg_d2_n64_cfl0.2"
            bucket[record["task_key"]] = record
    return published, lattice


def test_exponent_invariance_is_measured_not_assumed():
    """``alpha <= 1`` must survive the reference change; a big move is flagged."""
    published, lattice = _exponent_records({"published": 0.20, "lattice": 0.21})
    payload = ru.build_report(published, lattice)
    invariance = payload["criterion_invariance"]
    assert invariance["max_abs_delta_alpha"] == pytest.approx(0.01, abs=1e-9)
    assert invariance["exponent_invariant"] is True
    assert invariance["conservation_holds_under_both"] is True

    # a reference change that moves the exponent by more than the tolerance is
    # reported as breaking the invariance -- not silently absorbed
    published, lattice = _exponent_records({"published": 0.20, "lattice": 0.55})
    invariance = ru.build_report(published, lattice)["criterion_invariance"]
    assert invariance["max_abs_delta_alpha"] == pytest.approx(0.35, abs=1e-9)
    assert invariance["exponent_invariant"] is False


# ---------------------------------------------------------------------------
# the committed production measurement: the round-20 debt
# ---------------------------------------------------------------------------

UPGRADE_JSON = LATTICE_DIR / "reference_upgrade.json"
#: The checkpoints the lattice re-run had to cover: the two families that carry
#: the largest worst-case floor bound (``dg_d3_n32`` 25.3 %, ``dg_d2_n64`` 12.7 %).
#: Before round 21 only ``dg_d1_n32_t1`` had ever been re-referenced, i.e. the
#: *lowest* floor ratio in the ladder.
HIGH_ORDER = ("dg_d2_n64_t1_cfl0.2", "dg_d2_n64_t2_cfl0.2", "dg_d2_n64_t10_cfl0.2",
              "dg_d3_n32_t1_cfl0.2", "dg_d3_n32_t2_cfl0.2",
              "dg_d3_n32_t5_cfl0.2", "dg_d3_n32_t10_cfl0.2")

needs_upgrade = pytest.mark.skipif(not UPGRADE_JSON.exists(),
                                   reason="reference upgrade artifact not present")


@needs_upgrade
def test_upgrade_artifact_covers_the_high_order_families():
    payload = json.loads(UPGRADE_JSON.read_text())
    rows = {row["task_key"]: row for row in payload["checkpoints"]}
    missing = [key for key in HIGH_ORDER if key not in rows]
    assert missing == []
    # The effect is measured per checkpoint; where a same-condition min_image
    # control exists it is the baseline (``lattice_vs_rerun``), otherwise the
    # published record is, which can only *overstate* the effect by the solver
    # round-off -- never hide it.  The n64 controls pinned that round-off at
    # exactly 0.0 (``rerun_noise == 0``), i.e. the solver is bit-reproducible, so
    # a missing control cannot bias the p=3 readings in either direction.
    for key in HIGH_ORDER:
        assert rows[key]["effect_source"] in ("lattice_vs_rerun", "lattice_vs_published")
    controlled = [key for key in HIGH_ORDER if rows[key]["effect_source"] == "lattice_vs_rerun"]
    assert controlled, "no checkpoint has a same-condition control at all"
    for key in controlled:
        assert rows[key]["rerun_noise"]["max_relative_shift"] < 1.0e-9


@needs_upgrade
def test_measured_shift_is_immaterial_and_inside_its_bound():
    """The 25 % bound at ``dg_d3_n32`` is a bound, not the effect."""
    payload = json.loads(UPGRADE_JSON.read_text())
    rows = {row["task_key"]: row for row in payload["checkpoints"]}
    assert payload["material_checkpoints"] == []
    assert payload["verdict"] == "no_material_reference_effect"
    for key in HIGH_ORDER:
        row = rows[key]
        assert row["bound_is_sound"] is True
        # ``effect_max`` is the unified field: the controlled measurement where a
        # control exists, the published baseline otherwise
        assert row["effect_max"] < ru.MATERIAL_SHIFT
    # the worst-case point of the ladder: the bound is pessimistic by > 20x
    worst = rows["dg_d3_n32_t1_cfl0.2"]
    factor = worst["floor_bound"] / worst["effect_max"]
    assert factor > 20.0


@needs_upgrade
def test_reference_change_does_not_move_the_conservation_exponent():
    payload = json.loads(UPGRADE_JSON.read_text())
    invariance = payload["criterion_invariance"]
    assert invariance["conservation_holds_under_both"] is True
    assert invariance["exponent_invariant"] is True
    assert invariance["max_abs_delta_alpha"] < 5.0e-3
    shift = payload["exponent_shift"]
    assert {"dg_d2_n64_cfl0.2", "dg_d3_n32_cfl0.2"} <= set(shift)
    # no doubled labels in the artifact's exponent table
    assert all(name.count("dg_d") == 1 and name.count("_cfl") == 1 for name in shift)


@needs_upgrade
def test_reference_effect_is_velocity_only_in_the_committed_records():
    """Field selectivity: ``O(tail)`` in velocity, ``O(tail^2)`` in rho/p."""
    payload = json.loads(UPGRADE_JSON.read_text())
    rows = {row["task_key"]: row for row in payload["checkpoints"]}
    for key in HIGH_ORDER:
        fields = rows[key]["lattice_vs_published"]["per_field"]
        assert abs(fields["density"]) < 1.0e-6
        assert abs(fields["pressure"]) < 1.0e-6
        assert abs(fields["velocity_x"]) > 1.0e-4
        assert abs(fields["velocity_y"]) > 1.0e-4
