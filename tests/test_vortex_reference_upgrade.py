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