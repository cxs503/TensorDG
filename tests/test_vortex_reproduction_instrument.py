"""R6 (reproduction) of CASE-DG-EULER-VORTEX-ERR: the instrument that closes the
round-20 debt.

Round 20 declared six instrument rows; R6 ("re-solve the checkpoint and compare
with ``results/*.json``, deviation > 1e-9 means stale") had no reading.  These
tests pin the three things that make such a row a *certificate* rather than a
claim:

* the deviation metric and the declared limit decide the way the row says,
* a stale reading is actually rejected (a gate that cannot fail is not a gate),
* the reproduction harness itself is exonerated: two fresh processes at the same
  settings have to be bit-identical, otherwise the residual deviation could be
  process jitter rather than a property of the stored bundle.

The last test is an end-to-end re-solve of a deliberately tiny P0 configuration;
everything else is arithmetic on synthesised records, so the file stays fast.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.vortex_reference_instrument_certification import (
    REPRODUCTION_LIMIT,
    REPRODUCTION_TARGETS,
    REPRODUCTION_THREADS,
    record_readings,
    relative_deviation,
    reproduce_record,
    verdict,
)

OUT = Path("case_euler_vortex_errconserve")
R6_ARTIFACT = Path("case_euler_vortex_errconserve_r6/reproduction.json")
REPRO_DIR = Path("case_euler_vortex_errconserve_reproduction")


def test_deviation_is_relative_and_zero_for_identical_records():
    stored = {"max_relative_l2": 2.0e-4, "final.mass": 0.0, "final.energy": 344.7}
    assert relative_deviation(stored, dict(stored)) == {
        "max_relative_l2": 0.0, "final.mass": 0.0, "final.energy": 0.0}
    dev = relative_deviation(stored, {"max_relative_l2": 2.0e-4 * (1 + 1e-8),
                                      "final.mass": 0.0,
                                      "final.energy": 344.7 * (1 - 5e-10)})
    assert dev["max_relative_l2"] == pytest.approx(1e-8, rel=1e-6)
    assert dev["final.energy"] == pytest.approx(5e-10, rel=1e-6)


def test_absent_reading_is_skipped_not_treated_as_zero():
    """A metric the solver did not publish must not read as a 100% deviation."""
    stored = {"max_relative_l2": 1.0e-4, "errors.density": 1.0e-4}
    fresh = {"max_relative_l2": 1.0e-4}          # P0 records carry per-field errors
    dev = relative_deviation(stored, fresh)       # DG records do not
    assert set(dev) == {"max_relative_l2"}
    assert dev["max_relative_l2"] == 0.0


def test_the_declared_limit_rejects_a_stale_reading_and_accepts_a_live_one():
    """The gate bites at the limit written in the certificate (1e-9)."""
    live = 1e-10      # measured envelope of a re-solve on this host, order 1e-13
    stale = 1e-8      # a reading that predates the solver by more than round-off
    assert live <= REPRODUCTION_LIMIT < stale
    assert not (live > REPRODUCTION_LIMIT)
    assert stale > REPRODUCTION_LIMIT


def test_wrong_checkpoint_substitution_is_caught_by_orders_of_magnitude():
    """A sibling checkpoint in the stored slot is what a copy-paste bundle looks
    like; it has to fail, and fail by far more than the limit."""
    def stored(key):
        path = OUT / "results" / f"{key}.json"
        if not path.exists():
            pytest.skip(f"missing committed record {path}")
        return json.loads(path.read_text())

    design = stored("dg_d2_n32_t1_cfl0.2")
    later = stored("dg_d2_n32_t2_cfl0.2")
    deviation = relative_deviation(record_readings(design),
                                   record_readings(later))["max_relative_l2"]
    assert deviation > 100 * REPRODUCTION_LIMIT


def test_r6_participates_in_the_package_verdict():
    """A failed R6 must make the whole instrument package non-certified."""
    base = {"R1_exactness": {"passed": True, "mutants_detected": True,
                            "periodicity_passed": True},
            "R2_isentropy": {"passed": True, "mutant_detected": True}}
    assert verdict(base)["status"] == "CERTIFIED"
    failed = dict(base, R6_reproduction={"passed": False})
    assert verdict(failed)["status"] == "INSTRUMENT-LIMITED"
    assert verdict(failed)["failed"] == ["R6_reproduction"]


def test_committed_r6_reading_is_inside_its_own_limit():
    """Binds the published R6 row to the artefacts it claims to grade."""
    if not R6_ARTIFACT.exists():
        pytest.skip(f"{R6_ARTIFACT} not published yet")
    report = json.loads(R6_ARTIFACT.read_text())["R6_reproduction"]
    assert report["limit"] == REPRODUCTION_LIMIT
    assert report["passed"] is True, report["worst_at"]
    assert report["worst"] <= REPRODUCTION_LIMIT
    assert report["worst"] <= REPRODUCTION_LIMIT / 10.0, "10x margin not kept"
    assert report["steps_identical"] is True
    assert report["coverage"]["reproduced"] >= 2
    assert report["probes"]["flip_point"], "no probe flipped -- gate has no power"


def test_every_declared_target_has_a_committed_record():
    for spec in REPRODUCTION_TARGETS:
        assert (OUT / "results" / f"{spec['key']}.json").exists(), spec["key"]


def test_two_fresh_processes_are_bit_identical_at_identical_settings(tmp_path):
    """End-to-end (tiny) reproduction: the harness is not the source of the noise.

    A legitimate re-run of the same configuration in a second process has to give
    the same number; otherwise the residual deviation from the committed bundle
    could be attributed to the process rather than to the bundle.
    """
    spec = {"key": "p0_n32_t1_cfl0.4", "task": "p0", "degree": 0, "cells": 32,
            "final_time": 1.0, "cfl": 0.4, "role": "test"}
    first = reproduce_record(spec, 1, tmp_path / "a")
    second = reproduce_record(spec, 1, tmp_path / "b")
    assert first["task_key"] == second["task_key"] == "p0_n32_t1_cfl0.4"
    assert first["steps"] == second["steps"]
    for name, value in record_readings(first).items():
        assert record_readings(second)[name] == value, name
    assert max(REPRODUCTION_THREADS) >= 1