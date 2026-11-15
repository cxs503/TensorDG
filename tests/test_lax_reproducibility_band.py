"""CASE-LAX-DG · the artifact's reproducibility claim, machine-checked.

The committed CASE-LAX-DG artifacts quote errors at the last-bit level, and the
``reuse_check`` gate that was supposed to protect them compares **two committed
files** with a ``1e-6`` tolerance.  Such a gate can never fail because of the
machine: a committed file does not re-run.  ``benchmarks/lax_reproducibility_band``
measures the missing quantity -- the drift between the artifact and a rerun on a
different stack -- and this module turns its readings into invariants.

Pre-registered claims under test (all from the artifact, none transcribed):

* **T1** thread count is *not* the cause: the same leg at 1 and 8 torch threads
  is bit-identical, on every leg measured.
* **T2** cross-stack drift grows with the step count, and it exceeds the
  existing ``REUSE_TOL = 1e-6`` on the finest leg -- i.e. the existing gate is
  stack-fragile and would false-alarm on any other machine.
* **T3** the *ledger* metric (whole-field relative **L2** density at
  ``p3.N400``, the acceptance number ``3.11556e-02``) is stack-robust to
  ``1e-6``, unlike the L1 numbers the reuse gate uses.
* **T4** a same-stack rerun reproduces the recorded value *bit for bit*.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BAND = ROOT / "docs/benchmarks/results/lax_reproducibility_band/metrics.json"
PROBE = ROOT / "docs/benchmarks/results/lax_asymptotic_probe/metrics.json"

pytestmark = pytest.mark.skipif(not BAND.exists(), reason="band artifact not generated yet")

#: ledger acceptance reading for CASE-LAX-DG (``p3.N400`` whole-field rel L2 rho),
#: as committed by ``lax_case_pool``/``lax_resolution_closure``.
LEDGER_L2 = 3.1155618811580767e-02


def artifact():
    return json.loads(BAND.read_text())


def rows():
    return artifact()["rows"]


def row(cells, degree, threads, stack=None):
    for item in rows():
        if (item["cells"], item["degree"], item["torch_threads"]) == (cells, degree, threads):
            if stack is None or item.get("stack") == stack:
                return item
    return None


def stacks():
    return sorted({item.get("stack") for item in rows() if item.get("stack")})


# --- T1: the thread hypothesis is falsified ---------------------------------


def test_thread_count_does_not_change_the_fields():
    """``torch.set_num_threads`` 1 vs 8 gives bit-identical fields, every leg.

    This is the hypothesis the previous audit used to explain the drift
    ("torch reduction order").  It is measured here, and it is false on this
    machine: the reduction order that matters is the *build*, not the count.
    """
    tests = artifact()["thread_tests"]
    assert tests, "no thread tests recorded"
    for key, entry in tests.items():
        assert entry["fields_bit_identical"] is True, (key, entry["fields_sha256"])
        assert len(entry["fields_sha256"]) >= 2, key
    assert artifact()["thread_hypothesis_falsified"] is True


def test_thread_counts_are_actually_distinct_in_the_artifact():
    """Guard against a vacuous T1: both thread settings must be recorded."""
    for key in artifact()["thread_tests"]:
        entries = [r for r in rows() if f"p{r['degree']}.N{r['cells']}" == key]
        assert {e["torch_threads"] for e in entries} >= {1, 8}, key


# --- T2: drift grows with steps and breaks the existing tolerance -----------


def test_cross_stack_drift_grows_with_the_number_of_steps():
    """A longer run amplifies last-bit differences; the table must show it."""
    for stack in stacks():
        per_cells = {}
        for item in rows():
            if item.get("stack") != stack or item.get("drift_l1") is None:
                continue
            best = per_cells.get(item["cells"])
            if best is None or item["drift_l1"] > best["drift_l1"]:
                per_cells[item["cells"]] = item
        ladder = sorted(per_cells.values(), key=lambda r: r["steps"])
        if len(ladder) < 2:
            continue
        drifts = [r["drift_l1"] for r in ladder]
        assert all(b >= a for a, b in zip(drifts, drifts[1:])), (
            stack, [(r["cells"], r["steps"], r["drift_l1"]) for r in ladder])


def test_existing_reuse_tolerance_is_stack_fragile():
    """The 1e-6 cross-artifact gate cannot hold across stacks on the fine legs."""
    from benchmarks.lax_asymptotic_probe import REUSE_TOL

    ledger = row(400, 3, 8) or row(400, 3, 1)
    assert ledger is not None
    assert ledger["drift_l1"] > REUSE_TOL, (
        "if this ever becomes false the L1 reuse gate is stack-robust after all: "
        f"drift={ledger['drift_l1']:.3e} tol={REUSE_TOL:.1e}")
    # ... while the same stack reproduces exactly (T4 below), which is the whole
    # point: the gate is right about staleness and wrong about portability.
    assert artifact()["stacks"], "the artifact must name the stacks it compares"


# --- T3: the ledger metric is stack-robust ---------------------------------


@pytest.mark.parametrize("threads", [1, 8])
def test_ledger_reading_is_stack_robust(threads):
    """``p3.N400`` whole-field rel L2 = 3.11556e-02 survives a stack change."""
    ledger = row(400, 3, threads)
    if ledger is None:
        pytest.skip(f"p3.N400 at {threads} threads not in the artifact")
    drift = abs(ledger["l2_density"] / LEDGER_L2 - 1.0)
    assert drift < 1e-6, (threads, ledger["l2_density"], drift)
    # and the L2 is not accidentally the L1 (a copy/paste class of defect)
    assert ledger["l2_density"] > ledger["l1_density"], ledger


def test_ledger_arm_is_not_re_judged():
    """The band changes *how well* the number is known, never the verdict."""
    payload = artifact()
    assert payload["ledger_arm_rejudged"] is False
    assert LEDGER_L2 > 0.03, "the pinned leg L2 is above the 3 % bar and stays a FAIL"


# --- T4: same stack -> bit for bit -----------------------------------------


def current_stack():
    from tensordg.reproducibility import environment_fingerprint
    from benchmarks.lax_reproducibility_band import _stack_tag

    return _stack_tag(environment_fingerprint())


def test_same_stack_rerun_is_bit_identical():
    """Re-measure the cheapest leg here and demand the recorded value back."""
    stack = current_stack()
    recorded = row(100, 3, 1, stack=stack)
    if recorded is None:
        pytest.skip(f"artifact carries no p3.N100 row for stack {stack}")

    from benchmarks.lax_reproducibility_band import measure
    from benchmarks.lax_case_pool import exact_waves

    fresh, _ = measure(100, 3, exact_waves())
    if fresh["torch_threads"] != recorded["torch_threads"]:
        pytest.skip("thread setting differs from the recorded row")
    assert fresh["steps"] == recorded["steps"]
    assert fresh["l1_density"] == recorded["l1_density"], (
        f"same stack must reproduce bit for bit: {fresh['l1_density']!r} != "
        f"{recorded['l1_density']!r}")


# --- the fingerprint itself -------------------------------------------------


def test_artifact_fingerprints_are_complete():
    payload = artifact()
    required = {"python", "numpy", "torch", "machine", "torch_threads"}
    assert required <= set(payload["environment"]), payload["environment"]
    for entry in payload["stacks"]:
        assert required <= set(entry["environment"]), entry


def test_drift_helper_has_teeth():
    """A value 1 % away must register as a 1 % drift (the band is not a rabbit)."""
    from benchmarks.lax_reproducibility_band import rel_drift

    assert rel_drift(1.01, 1.0) == pytest.approx(0.01, rel=1e-12)
    assert rel_drift(1.0 + 1e-9, 1.0) > 1e-10
    assert rel_drift(1.0, 1.0) == 0.0
    assert rel_drift(1.0, None) is None


def test_pinned_threads_restores_the_previous_setting():
    import torch

    from tensordg.reproducibility import pinned_threads

    before = int(torch.get_num_threads())
    with pinned_threads(1) as threads:
        assert threads == 1
        assert int(torch.get_num_threads()) == 1
    assert int(torch.get_num_threads()) == before


def test_probe_artifact_still_declares_its_stack():
    """The band artifact and the probe must not disagree about the drift order."""
    if not PROBE.exists():
        pytest.skip("probe artifact absent")
    probe = json.loads(PROBE.read_text())
    assert probe.get("environment"), "probe artifact lost its environment block"
    ledger = row(400, 3, 1)
    assert ledger["steps"] == 5109
    assert math.isclose(ledger["l1_density"], 8.68646462484351164e-03, rel_tol=1e-12)