"""C10 tests: the error-conservation verdict must survive a change of direction.

The historical ladder moved the mesh (C7), the time step (C7) and the polynomial
degree (C8), but it always transported the vortex along the grid diagonal
``u_inf = (1, 1)``.  C10 closes that hole, and these tests pin the two things a
direction axis can get wrong:

* **identity** -- if the direction is not part of the family key, two runs of two
  different problems are fitted by one ``e ~ t^alpha`` and the *design* family
  silently absorbs the rotated readings.  The tests check the key derivation
  (including the case where a stored family label is stale) as well as the
  declared ladder;
* **discrimination** -- a criterion that passes whenever the baseline passes is
  vacuous.  The tests build a bundle whose rotated family accumulates like
  ``t^2`` and require C10 to read ``failed``, and a bundle with no rotated family
  at all (the pre-round-9 evidence) and require it to read ``not_evaluated`` --
  never ``passed``.

All bundles here are synthetic, so nothing in this file depends on a landed run.
"""
from __future__ import annotations

import math

import pytest

from benchmarks import case_euler_vortex_errconserve as ec


# ---------------------------------------------------------------------------
# synthetic evidence bundles (same vocabulary as the module's LADDER)
# ---------------------------------------------------------------------------

def _record(config, error, drift=1e-15, entropy=1e-6):
    """A synthetic record carrying the declared task key and config of ``config``."""
    record = {
        "task_key": ec.task_key(config),
        "task": config["task"],
        "degree": config.get("degree", 0),
        "cells": config["cells"],
        "cfl": config["cfl"],
        "final_time": config["final_time"],
        "steps": 1,
        "errors": {name: error for name in ec.FIELDS},
        "max_relative_l2": error,
        "max_percent": 100.0 * error,
        "invariant_drift": {"mass": drift, "momentum_x": drift,
                            "momentum_y": drift, "energy": drift,
                            "entropy": entropy},
        "entropy_drift": entropy,
        "entropy_production": -entropy,
        "density_min_relative_error": 0.0,
        "gate_pass": error < ec.TOLERANCE,
        "wall_seconds": 0.0,
        "final": {"density_min": 1.0, "pressure_min": 1.0},
        "initial": {"density_min": 1.0, "pressure_min": 1.0},
        "config": dict(config),
        "reference_model": ec.REFERENCE_MODEL,
    }
    record["family"] = ec.family_key(record)
    return record


def _ladder_records(dg_error=None, p0_error=None):
    """One synthetic record per declared checkpoint (C6-complete bundle)."""
    dg_error = dg_error or (lambda degree, cells, t, label:
                            0.02 / cells ** (degree + 1)
                            * (1.0 + 0.01 * math.log(t)))
    p0_error = p0_error or (lambda n, t: 0.4 / n * (1.0 + 0.01 * math.log(t)))
    records = []
    for config in ec.LADDER:
        if config["task"] == "dg":
            label = ec.direction_label(
                tuple(config.get("background_velocity", ec.BACKGROUND)))
            error = dg_error(config["degree"], config["cells"],
                             config["final_time"], label)
        else:
            error = p0_error(config["cells"], config["final_time"])
        records.append(_record(config, error))
    return records


def _gates(records):
    return ec.evaluate_gates(records, ec.exact_invariants(points=64))["gates"]


# ---------------------------------------------------------------------------
# the axis itself
# ---------------------------------------------------------------------------

def test_direction_axis_holds_the_speed_fixed():
    """C10 must isolate *direction*; a speed change would confound transport length."""
    speeds = [math.hypot(*direction) for _, direction in ec.DIRECTION_AXIS]
    for speed in speeds:
        assert speed == pytest.approx(ec.DIRECTION_SPEED, rel=1e-12)
    angles = sorted(math.degrees(math.atan2(d[1], d[0]))
                    for _, d in ec.DIRECTION_AXIS)
    # the axis brackets the historical diagonal instead of extending it one way
    assert angles[0] == pytest.approx(0.0, abs=1e-12)
    assert angles[1] == pytest.approx(30.0, rel=1e-9)
    assert angles[2] == pytest.approx(45.0, rel=1e-9)


def test_baseline_direction_keeps_the_historical_task_keys():
    """Every pre-round-9 result file must stay in its own family and checkpoint."""
    baseline = dict(ec.DIRECTION_AXIS)[ec.DIRECTION_BASELINE]
    config = {"task": "dg", "degree": 2, "cells": 32, "final_time": 10.0,
              "cfl": 0.2, "background_velocity": baseline}
    assert ec.task_key(config) == "dg_d2_n32_t10_cfl0.2"
    assert ec.direction_suffix(ec.DIRECTION_BASELINE) == ""
    # ...and the direction actually in use is the module background
    assert ec.direction_label(ec.BACKGROUND) == ec.DIRECTION_BASELINE


def test_rotated_direction_changes_task_and_family_keys():
    baseline = dict(ec.DIRECTION_AXIS)[ec.DIRECTION_BASELINE]
    config = {"task": "dg", "degree": 2, "cells": 32, "final_time": 10.0,
              "cfl": 0.2, "background_velocity": baseline}
    base_key, base_family = ec.task_key(config), ec.family_key(_record(config, 1e-4))
    for label, direction in ec.DIRECTION_AXIS:
        if label == ec.DIRECTION_BASELINE:
            continue
        rotated = dict(config, background_velocity=direction)
        rotated_record = _record(rotated, 1e-4)
        assert ec.task_key(rotated) != base_key
        assert rotated_record["family"] != base_family
        assert rotated_record["family"] == ec.direction_family(label)
        assert rotated_record["family"].endswith(f"_dir{label}")


def test_unknown_direction_is_rejected_instead_of_silently_aliased():
    """A mis-keyed run would otherwise join the baseline family and flatter C2."""
    with pytest.raises(ValueError):
        ec.direction_label((1.0, 0.0))       # same speed as axis0? no: |u| = 1
    with pytest.raises(ValueError):
        ec.direction_label((1.0, 1.5))


def test_stale_family_labels_are_re_derived_from_the_config():
    """A record must be graded by its own configuration, not by its stored label."""
    label, direction = "axis0", dict(ec.DIRECTION_AXIS)["axis0"]
    config = {"task": "dg", "degree": 2, "cells": 32, "final_time": 10.0,
              "cfl": 0.2, "background_velocity": direction}
    record = _record(config, 1e-4)
    record["family"] = ec.GATE_FAMILY          # the stale, direction-free label
    gates = _gates([record])
    direction_row = gates["C10_direction_invariance"]["per_direction"][label]
    # the rotated checkpoint is counted under its own direction, not the baseline,
    # so it cannot enter the baseline family's e ~ t^alpha fit
    assert direction_row["checkpoints"] == 1
    assert direction_row["family"] == ec.direction_family(label)
    assert ec.GATE_FAMILY not in gates.get("families", {})
    # a single checkpoint is not enough to decide alpha: honest not_evaluated
    assert gates["C10_direction_invariance"]["status"] == "not_evaluated"


def test_declared_ladder_carries_every_rotated_direction():
    declared = {ec.task_key(config) for config in ec.LADDER}
    for label, _ in ec.DIRECTION_AXIS:
        if label == ec.DIRECTION_BASELINE:
            continue
        for time in (1.0, 2.0, 5.0, 10.0):
            assert (f"dg_d2_n32_t{time:g}_cfl0.2_dir{label}") in declared


# ---------------------------------------------------------------------------
# C10 discrimination
# ---------------------------------------------------------------------------

def test_c10_passes_when_every_direction_conserves():
    gates = _gates(_ladder_records())
    c10 = gates["C10_direction_invariance"]
    assert c10["status"] == "passed"
    assert set(c10["directions_evaluated"]) == {label for label, _ in ec.DIRECTION_AXIS}
    # the decisive requirement: a rotated direction is among the decidable ones
    assert c10["rotated_evaluated"]


def test_c10_fails_when_a_rotated_direction_accumulates():
    """The defect C10 exists for: conservation on the diagonal and nowhere else."""
    def dg_error(degree, cells, t, label):
        base = 0.02 / cells ** (degree + 1) * (1.0 + 0.01 * math.log(t))
        return base * (1.0 + 0.5 * t * t) if label == "axis0" else base

    gates = _gates(_ladder_records(dg_error=dg_error))
    c10 = gates["C10_direction_invariance"]
    assert c10["status"] == "failed"
    assert c10["passed"] is False
    assert list(c10["offenders"]) == ["axis0"]
    assert c10["offenders"]["axis0"] > ec.MAX_GROWTH_EXPONENT
    # the baseline reading is untouched by the rotated failure -> C10 is not a copy
    # of C2: C2 still passes on the same bundle
    assert gates["C2_error_conservation"]["passed"] is True
    assert gates["verdict"] == "FAIL"


def test_c10_is_not_evaluated_without_a_rotated_direction():
    """The pre-round-9 evidence bundle (diagonal only) must not read as a pass."""
    baseline = [config for config in ec.LADDER
                if ec.direction_label(tuple(config.get("background_velocity",
                                                       ec.BACKGROUND)))
                == ec.DIRECTION_BASELINE]
    records = []
    for config in baseline:
        error = (0.02 / config["cells"] ** (config["degree"] + 1)
                 * (1.0 + 0.01 * math.log(config["final_time"]))
                 if config["task"] == "dg" else
                 0.4 / config["cells"] * (1.0 + 0.01 * math.log(config["final_time"])))
        records.append(_record(config, error))
    gates = _gates(records)
    c10 = gates["C10_direction_invariance"]
    assert c10["status"] == "not_evaluated"
    assert c10["passed"] is False
    assert c10["rotated_evaluated"] == []
    assert gates["C2_error_conservation"]["passed"] is True


def test_c10_needs_at_least_two_decidable_directions():
    """One decidable direction is the anecdote the criterion is written against."""
    def dg_error(degree, cells, t, label):
        base = 0.02 / cells ** (degree + 1) * (1.0 + 0.01 * math.log(t))
        return base if label in (ec.DIRECTION_BASELINE, "axis0") else base

    records = [r for r in _ladder_records(dg_error=dg_error)
               if not (r["final_time"] < 3.0 and "dirobl30" in r["family"])]
    gates = _gates(records)
    c10 = gates["C10_direction_invariance"]
    assert "obl30" not in c10["directions_evaluated"]      # only two checkpoints
    assert c10["status"] == "passed"
    assert len(c10["directions_evaluated"]) >= ec.MIN_DECIDABLE_DIRECTIONS


def test_c10_gate_is_listed_in_the_acceptance_set():
    gates = _gates(_ladder_records())
    assert "C10_direction_invariance" in gates
    assert "C10_direction_invariance" not in gates["informational"]
    assert "C10_direction_invariance" not in gates["not_evaluated"]